"""실행 엔진 - 상태 머신 · 버전 · durability (기획서 §5.3, §5.4, §17.4).

hub가 노드 × 축마다 상태를 들고 있고, 캔버스의 상태 링은 이 값을 그대로
그린다. 편집이 들어오면 영향을 받는 노드만 ``stale``이 되고, 실행 결과가
돌아오면 ``ok``/``error``가 되며 error의 하류는 ``blocked``로 전파된다.

``ponytail: 동기 flush. 우선순위 큐·비동기 워커·L1/L2 축은 M3·M7에서 붙는다.``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..ir import Composite, Graph, ModuleGraph, split_endpoint

AXES = ("L0", "L1fwd", "L1bwd")

IDLE, STALE, QUEUED, RUNNING = "idle", "stale", "queued", "running"
OK, OK_WARN, ERROR, BLOCKED = "ok", "ok.warn", "error", "blocked"

# 실행 의미가 없는 op. 좌표는 layout.json 소관이고, variant 저장은 스냅샷일 뿐이다.
VISUAL_OPS = {"move", "save_variant"}

# 프로브는 그래프 의미를 바꾸지 않는다(§6.3) - L0는 건드리지 않고 L1 축만 stale.
PROBE_OPS = {"add_probe", "remove_probe"}


@dataclass
class GraphIndex:
    """계층 그래프를 한 번 훑어 만든 인접·역참조 표.

    IR이 구조적으로 바뀔 때만 다시 만든다.
    """

    successors: dict[str, set[str]] = field(default_factory=dict)
    predecessors: dict[str, set[str]] = field(default_factory=dict)
    scope_of: dict[str, str] = field(default_factory=dict)
    durability: dict[str, str] = field(default_factory=dict)
    # 컴포지트 이름 -> 그 컴포지트를 호출하는 노드들. 안쪽 편집이 바깥으로 번지는 길.
    callers_of: dict[str, set[str]] = field(default_factory=dict)
    nodes_in_scope: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, ir: ModuleGraph) -> GraphIndex:
        index = cls()

        def walk(scope: Graph | Composite, scope_name: str) -> None:
            index.nodes_in_scope[scope_name] = [node.id for node in scope.nodes]
            for node in scope.nodes:
                index.scope_of[node.id] = scope_name
                index.successors.setdefault(node.id, set())
                index.predecessors.setdefault(node.id, set())
                instance = scope.instances.get(node.call) if node.call else None
                index.durability[node.id] = (
                    getattr(node, "durability", None)
                    or (getattr(instance, "durability", None) if instance else None)
                    or "mid"
                )
                for referenced in _composites_referenced(instance):
                    index.callers_of.setdefault(referenced, set()).add(node.id)
            for src, dst in scope.edges:
                source, _ = split_endpoint(src)
                target, _ = split_endpoint(dst)
                if source in index.successors and target in index.predecessors:
                    index.successors[source].add(target)
                    index.predecessors[target].add(source)

        walk(ir.graph, "$graph")
        for name, composite in ir.composites.items():
            walk(composite, name)
        return index

    def _closure(self, node_ids: Iterable[str], links: dict[str, set[str]]) -> set[str]:
        """``links``를 따라간 폐쇄. 컴포지트 안의 노드는 호출 노드로도 번진다."""
        seen: set[str] = set()
        stack = list(node_ids)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(links.get(current, ()))
            scope = self.scope_of.get(current)
            if scope and scope != "$graph":
                stack.extend(self.callers_of.get(scope, ()))
        return seen

    def descendants(self, node_ids: Iterable[str]) -> set[str]:
        """하류 폐쇄. 컴포지트 경계를 넘어 호출 노드까지 따라간다."""
        return self._closure(node_ids, self.successors)

    def ancestors(self, node_ids: Iterable[str]) -> set[str]:
        """상류 폐쇄. L1-bwd의 grad stale 범위가 이것이다(§5.1.2)."""
        return self._closure(node_ids, self.predecessors)

    def all_nodes(self) -> list[str]:
        return list(self.scope_of)


def _composites_referenced(instance) -> set[str]:
    """인스턴스가 참조하는 컴포지트 이름들 (Switch 변형과 Repeat body 포함)."""
    if instance is None:
        return set()
    found = set()

    def add(type_name: str | None) -> None:
        if type_name and type_name.startswith("composite:"):
            found.add(type_name.split(":", 1)[1])

    add(instance.type)
    add(instance.body)
    for variant in (instance.variants or {}).values():
        add(variant.get("type"))
    return found


class Engine:
    """편집 -> 상태 전이 -> 실행 -> 상태 전이."""

    def __init__(self, ir: ModuleGraph):
        self._callers: dict[tuple[str, str], set[str]] = {}
        self.index = GraphIndex.build(ir)
        self._bind_calls(ir)
        self.version: dict[str, int] = dict.fromkeys(self.index.all_nodes(), 0)
        self.state: dict[tuple[str, str], str] = {
            (node, axis): IDLE for node in self.index.all_nodes() for axis in AXES
        }
        self.results: dict[str, dict[str, Any]] = {}

    def rebuild(self, ir: ModuleGraph) -> None:
        """구조가 바뀐 뒤 인덱스를 다시 만든다. 기존 상태는 유지한다."""
        self.index = GraphIndex.build(ir)
        self._bind_calls(ir)
        for node in self.index.all_nodes():
            self.version.setdefault(node, 0)
            for axis in AXES:
                self.state.setdefault((node, axis), IDLE)

    # 편집
    def on_edit(self, op: dict[str, Any]) -> set[str]:
        """편집 하나를 반영하고 ``stale``이 된 노드들을 돌려준다(§5.4)."""
        kind = op.get("kind")
        if kind in VISUAL_OPS:
            return set()

        impact = "low" if kind in PROBE_OPS else "mid"
        axes = ("L1fwd",) if kind in PROBE_OPS else ("L0", "L1fwd")
        touched = self._touched_nodes(op)
        staled: set[str] = set()

        for node in self.index.descendants(touched):
            if self._durability_skip(node, impact):
                continue
            self.version[node] += 1
            for axis in axes:
                self.state[(node, axis)] = STALE
            staled.add(node)

        # grad는 하류에도 의존한다 - 조상 폐쇄까지 stale(grad)(§5.1.2).
        for node in staled | self.index.ancestors(touched):
            self.state[(node, "L1bwd")] = STALE
        return staled

    def _durability_skip(self, node: str, impact: str) -> bool:
        """``low`` 편집은 ``high`` 노드(데이터셋·사전학습 가중치)를 건드리지 않는다(§5.2)."""
        return impact == "low" and self.index.durability.get(node, "mid") == "high"

    def _touched_nodes(self, op: dict[str, Any]) -> set[str]:
        if op.get("kind") == "batch":
            return set().union(*(self._touched_nodes(inner) for inner in op.get("ops") or [{}]))
        payload = op.get("payload") or {}
        scope = payload.get("composite") or "$graph"
        node = payload.get("node")
        if isinstance(node, dict):   # add_node는 노드 명세를 통째로 싣는다
            node = node.get("id")
        if node:
            return {node}
        if "instance" in payload:
            # 인스턴스 편집은 그 인스턴스를 호출하는 노드 전부를 건드린다(공유 인스턴스 포함).
            return self._callers.get((scope, payload["instance"]), set())
        if "src" in payload:
            return {split_endpoint(payload["src"])[0], split_endpoint(payload["dst"])[0]}
        return set(self.index.nodes_in_scope.get(scope, []))

    def _bind_calls(self, ir: ModuleGraph) -> None:
        """``(scope, instance_id) -> 호출 노드`` 표."""
        self._callers = {}

        def walk(scope, name):
            for node in scope.nodes:
                if node.call:
                    self._callers.setdefault((name, node.call), set()).add(node.id)

        walk(ir.graph, "$graph")
        for composite_name, composite in ir.composites.items():
            walk(composite, composite_name)

    # 실행 결과 반영
    def absorb(self, reports, error: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        """L0 패스 결과를 상태로 옮긴다. 반환값은 바뀐 노드의 NodeState 필드들."""
        changed: dict[str, dict[str, Any]] = {}

        for report in reports:
            node = report.node_id
            path = getattr(report, "path", "") or ""
            self.state[(node, "L0")] = OK
            key = f"{path}/{node}" if path else node
            self.results[key] = {"spec": report.spec}
            changed[key] = {
                "node": node,
                "path": path,
                "state": OK,
                "spec": report.spec,
                "badges": {
                    "elapsed_ms": round(report.elapsed_ms, 2),
                    "cache_hit": report.cache_hit,
                },
            }

        if error is not None:
            node = error.get("node_id") or "?"
            self.state[(node, "L0")] = ERROR
            changed[node] = {
                "node": node,
                "path": "",
                "state": ERROR,
                "error": {"kind": error["kind"], "message": error["message"],
                          "mapping": error.get("mapping")},
                "badges": {},
            }
            # 하류는 실행되지 않았다 - blocked로 전파한다(§5.3).
            for descendant in self.index.descendants([node]) - {node}:
                self.state[(descendant, "L0")] = BLOCKED
                changed[descendant] = {"node": descendant, "path": "", "state": BLOCKED,
                                       "badges": {}}

        return changed

    def snapshot(self) -> dict[str, dict[str, str]]:
        return {
            node: {axis: self.state[(node, axis)] for axis in AXES}
            for node in self.index.all_nodes()
        }
