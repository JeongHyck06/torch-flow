"""권위 그래프 상태: op-log · seq · journal (기획서 §8.2.2).

hub가 그래프의 정본을 가진다. 클라이언트는 낙관적으로 먼저 그리고, 서버가
``seq``를 붙여 승인한다. journal은 append-only JSONL이라 hub가 죽어도
재시작 시 재생된다.

적용되는 op는 ``add_node``·``remove_node``·``set_param``·``set_switch_active``·
``connect``·``disconnect``다. 나머지 kind(``group``·``promote_hp``·``save_variant``
등)는 journal에 기록되지만 그래프에는 적용되지 않는다.

실행 취소는 서버 히스토리를 되감지 않는다(§8.2.3). 클라이언트가 **역 op를 새 op로**
보내므로 서버에는 "되돌리기"라는 개념이 없고, 여기 있는 op들이 서로의 역이 되도록
짝을 맞추는 것만이 계약이다: ``add_node`` <-> ``remove_node``,
``connect`` <-> ``disconnect``, ``set_param`` <-> 이전 값을 실은 ``set_param``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ir import ModuleGraph, canonical_json, save, split_endpoint

APPLIED_KINDS = {"add_node", "remove_node", "set_param", "set_switch_active",
                 "connect", "disconnect"}


class OpError(ValueError):
    """op가 그래프에 적용될 수 없다."""


class GraphStore:
    def __init__(self, ir: ModuleGraph, journal_path: Path | str | None = None):
        self.ir = ir
        self.seq = 0
        self.log: list[dict[str, Any]] = []
        self.journal_path = Path(journal_path) if journal_path else None
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    # op 적용
    def apply(self, op: dict[str, Any]) -> int:
        """op를 적용하고 부여된 ``seq``를 돌려준다."""
        kind = op.get("kind")
        if kind == "batch":
            for inner in op.get("ops") or []:
                self._mutate(inner)
        elif kind in APPLIED_KINDS:
            self._mutate(op)
        self.seq += 1
        record = {"seq": self.seq, "op": op}
        self.log.append(record)
        self._journal(record)
        return self.seq

    def _mutate(self, op: dict[str, Any]) -> None:
        kind, payload = op.get("kind"), op.get("payload") or {}
        scope = self._scope(payload.get("composite"))

        if kind == "add_node":
            self._add_node(scope, payload)
        elif kind == "remove_node":
            self._remove_node(scope, payload)
        elif kind == "set_param":
            # 인자는 인스턴스(모듈 생성 인자)나 노드(호출 인자) 어느 쪽에도 있다.
            target = self._param_target(scope, payload)
            if payload["value"] is None:
                target.args.pop(payload["path"], None)   # 값을 지우면 기본값으로 돌아간다
            else:
                target.args[payload["path"]] = payload["value"]
        elif kind == "set_switch_active":
            instance = scope.instances.get(payload["instance"])
            if instance is None:
                raise OpError(f"unknown instance {payload['instance']}")
            if payload["active"] not in (instance.variants or {}):
                raise OpError(f"unknown variant {payload['active']!r}")
            instance.active = payload["active"]
        elif kind == "connect":
            edge = (payload["src"], payload["dst"])
            if edge not in scope.edges:
                scope.edges.append(edge)
        elif kind == "disconnect":
            edge = (payload["src"], payload["dst"])
            if edge in scope.edges:
                scope.edges.remove(edge)

    def _param_target(self, scope, payload: dict[str, Any]):
        if payload.get("instance"):
            instance = scope.instances.get(payload["instance"])
            if instance is None:
                raise OpError(f"unknown instance {payload['instance']}")
            return instance
        node = next((n for n in scope.nodes if n.id == payload.get("node")), None)
        if node is None:
            raise OpError(f"unknown node {payload.get('node')}")
        return node

    def _add_node(self, scope, payload: dict[str, Any]) -> None:
        """노드 하나(+ 필요하면 그 모듈 인스턴스)를 스코프에 넣는다.

        id는 클라이언트가 만든다 - 낙관적 적용이 서버 왕복을 기다리지 않으려면
        화면에 그린 노드와 승인된 노드가 같은 id여야 한다(§8.2.2).
        """
        from ..ir import Instance, Node

        spec = payload.get("node") or {}
        if not spec.get("id"):
            raise OpError("add_node needs a node id")
        if any(node.id == spec["id"] for node in scope.nodes):
            raise OpError(f"duplicate node id {spec['id']}")

        instance = payload.get("instance")
        if instance is not None:
            instance = dict(instance)
            instance_id = instance.pop("id", None) or spec.get("call")
            if not instance_id:
                raise OpError("add_node instance needs an id")
            scope.instances[instance_id] = Instance.model_validate(instance)
            spec = {**spec, "call": instance_id}

        try:
            node = Node.model_validate(spec)
        except Exception as exc:
            raise OpError(f"invalid node: {exc}") from exc
        if (node.call is None) == (node.type is None):
            raise OpError("node needs exactly one of 'call' or 'type'")
        if node.call is not None and node.call not in scope.instances:
            raise OpError(f"unknown instance {node.call}")
        scope.nodes.append(node)

    def _remove_node(self, scope, payload: dict[str, Any]) -> None:
        """노드와 거기 닿은 엣지를 지운다. 참조가 사라진 인스턴스도 같이 지운다.

        지워진 엣지 목록은 돌려주지 않는다 - 실행 취소는 클라이언트가 자기 그래프를
        보고 ``batch{add_node, connect...}``를 만들어 보낸다(§8.2.3).
        """
        node_id = payload.get("node")
        target = next((node for node in scope.nodes if node.id == node_id), None)
        if target is None:
            raise OpError(f"unknown node {node_id}")

        scope.nodes.remove(target)
        scope.edges[:] = [
            (src, dst) for src, dst in scope.edges
            if split_endpoint(src)[0] != node_id and split_endpoint(dst)[0] != node_id
        ]
        if target.call and not any(node.call == target.call for node in scope.nodes):
            scope.instances.pop(target.call, None)

    def _scope(self, composite: str | None):
        if composite is None:
            return self.ir.graph
        scope = self.ir.composites.get(composite)
        if scope is None:
            raise OpError(f"unknown composite {composite!r}")
        return scope

    # journal
    def _journal(self, record: dict[str, Any]) -> None:
        if self.journal_path is None:
            return
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def replay(self) -> int:
        """journal을 재생한다. 재시작 복구 경로."""
        if self.journal_path is None or not self.journal_path.exists():
            return 0
        records = [
            json.loads(line)
            for line in self.journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.journal_path.write_text("", encoding="utf-8")
        self.seq, self.log = 0, []
        for record in records:
            self.apply(record["op"])
        return len(records)

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [record for record in self.log if record["seq"] > seq]

    def snapshot(self) -> dict[str, Any]:
        return {"seq": self.seq, "graph": json.loads(canonical_json(self.ir))}

    def save(self, path: Path | str) -> None:
        save(self.ir, path)
