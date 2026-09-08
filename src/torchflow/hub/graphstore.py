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

APPLIED_KINDS = {"add_node", "remove_node", "set_param", "set_ports", "rename",
                 "define_composite", "remove_composite", "set_instance",
                 "promote_hp", "demote_hp", "save_variant", "remove_variant", "apply_variant",
                 "set_switch_active", "connect", "disconnect"}


def _kind(node_type: str | None) -> str:
    """``torch.nn.Conv2d@2.11.0`` -> ``torch.nn.Conv2d``. 버전 꼬리를 뗀 블록 종류."""
    return (node_type or "").split("@")[0]


def _hparam_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _scopes(ir: ModuleGraph):
    yield "$graph", ir.graph
    for name, composite in ir.composites.items():
        yield name, composite


def _references_hp(ir: ModuleGraph, name: str) -> bool:
    from ..ir import iter_refs

    for _, scope in _scopes(ir):
        for source in [*scope.nodes, *scope.instances.values()]:
            for kind, inner in iter_refs(getattr(source, "args", {}) or {}):
                if kind == "$hp" and inner == name:
                    return True
    return False


def variant_snapshot(ir: ModuleGraph) -> dict[str, Any]:
    """지금 그래프의 "값" 전부 (§4.4.3 Variant Set).

    구조(노드 추가·삭제·재연결)가 아니라 **값**만 담는다: 인자, Switch 활성 변형,
    enabled 플래그, hparam 기본값. 논문 표의 행 대부분(depth 12->6, pre->post norm,
    aux loss 끄기)이 여기에 들어온다. 구조까지 담는 patch는 v1이다.
    """
    values: dict[str, Any] = {"hparams": {}, "instances": {}, "nodes": {}}
    for name, spec in ir.hparams.items():
        values["hparams"][name] = spec.default
    for scope_name, scope in _scopes(ir):
        for instance_id, instance in scope.instances.items():
            entry: dict[str, Any] = {"args": dict(instance.args)}
            if instance.active is not None:
                entry["active"] = instance.active
            if instance.count is not None:
                entry["count"] = instance.count
            values["instances"][f"{scope_name}/{instance_id}"] = entry
        for node in scope.nodes:
            if node.args or node.enabled is not None:
                values["nodes"][f"{scope_name}/{node.id}"] = {
                    "args": dict(node.args), "enabled": node.enabled}
    return values


def apply_snapshot(ir: ModuleGraph, values: dict[str, Any]) -> None:
    """스냅샷을 되돌려 놓는다. 그 사이에 사라진 블록은 조용히 건너뛴다."""
    for name, default in (values.get("hparams") or {}).items():
        if name in ir.hparams:
            ir.hparams[name].default = default
    for scope_name, scope in _scopes(ir):
        for instance_id, instance in scope.instances.items():
            entry = (values.get("instances") or {}).get(f"{scope_name}/{instance_id}")
            if entry is None:
                continue
            instance.args = dict(entry.get("args") or {})
            if "active" in entry:
                instance.active = entry["active"]
            if "count" in entry:
                instance.count = entry["count"]
        for node in scope.nodes:
            entry = (values.get("nodes") or {}).get(f"{scope_name}/{node.id}")
            if entry is None:
                continue
            node.args = dict(entry.get("args") or {})
            node.enabled = entry.get("enabled")


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
        elif kind == "rename":
            # 그래프 이름은 생성 코드에서 클래스 이름이 된다(§7.2). 빈 이름은 받지 않는다.
            name = str(payload.get("name") or "").strip()
            if not name:
                raise OpError("graph name cannot be empty")
            self.ir.graph.name = name
        elif kind == "set_ports":
            self._set_ports(scope, payload)
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
        elif kind == "set_instance":
            self._set_instance(scope, payload)
        elif kind == "promote_hp":
            self._promote_hp(scope, payload)
        elif kind == "demote_hp":
            self._demote_hp(scope, payload)
        elif kind == "save_variant":
            name = str(payload.get("name") or "").strip()
            if not name:
                raise OpError("변형 이름이 필요합니다")
            # 값을 실어 보내면 그것을 그대로 쓴다 - 지운 변형을 되살리는 역 op가 이 경로다.
            self.ir.variant_sets[name] = payload.get("values") or variant_snapshot(self.ir)
        elif kind == "remove_variant":
            if payload.get("name") not in self.ir.variant_sets:
                raise OpError(f"그런 변형이 없습니다: {payload.get('name')}")
            del self.ir.variant_sets[payload["name"]]
        elif kind == "apply_variant":
            values = self.ir.variant_sets.get(payload.get("name"))
            if values is None:
                raise OpError(f"그런 변형이 없습니다: {payload.get('name')}")
            apply_snapshot(self.ir, values)
            # 어떤 변형으로 돌렸는지가 run manifest에 실려야 표의 축이 된다(§6.4).
            self.ir.meta["variant"] = payload["name"]
        elif kind == "define_composite":
            self._define_composite(payload)
        elif kind == "remove_composite":
            self._remove_composite(payload)
        elif kind == "connect":
            edge = (payload["src"], payload["dst"])
            if edge not in scope.edges:
                scope.edges.append(edge)
        elif kind == "disconnect":
            edge = (payload["src"], payload["dst"])
            if edge in scope.edges:
                scope.edges.remove(edge)

    def _set_instance(self, scope, payload: dict[str, Any]) -> None:
        """인스턴스 하나를 통째로 갈아 끼운다.

        Switch를 만들거나(평범한 블록 -> 변형 둘), 변형을 더하거나 빼는 편집이 전부
        이 하나로 표현된다. 역 op는 **이전 인스턴스를 그대로 실은 같은 op**다.
        """
        from ..ir import Instance

        instance_id = payload.get("instance")
        if instance_id not in scope.instances:
            raise OpError(f"unknown instance {instance_id}")
        try:
            fresh = Instance.model_validate(payload.get("body") or {})
        except Exception as exc:
            raise OpError(f"invalid instance: {exc}") from exc
        if fresh.type == "torchflow.Switch":
            if not fresh.variants:
                raise OpError("Switch에는 변형이 하나 이상 있어야 합니다")
            active = fresh.active
            if isinstance(active, str) and active not in fresh.variants:
                raise OpError(f"활성 변형 {active!r}이 변형 목록에 없습니다")
        scope.instances[instance_id] = fresh

    def _promote_hp(self, scope, payload: dict[str, Any]) -> None:
        """지금 값을 하이퍼파라미터로 올리고 인자를 참조로 바꾼다(§4.3, §8.2.3).

        올린 뒤에는 상단에서 한 번 고치면 그 값을 쓰는 모든 블록이 같이 움직인다.
        """
        from ..ir import HParam

        target = self._param_target(scope, payload)
        path = payload["path"]
        name = str(payload.get("name") or path).strip()
        if not name.isidentifier():
            raise OpError(f"하이퍼파라미터 이름으로 쓸 수 없습니다: {name!r}")
        value = payload.get("value", target.args.get(path))
        existing = self.ir.hparams.get(name)
        if existing is None:
            self.ir.hparams[name] = HParam(type=_hparam_type(value), default=value)
        elif existing.default != value and payload.get("value") is None:
            # 이미 있는 이름에 다른 값을 얹으면 다른 블록의 값이 조용히 바뀐다.
            raise OpError(f"{name}은 이미 {existing.default}입니다 - 다른 이름을 쓰거나 값을 맞추세요")
        target.args[path] = {"$hp": name}

    def _demote_hp(self, scope, payload: dict[str, Any]) -> None:
        """참조를 다시 값으로 내린다. promote_hp의 역."""
        target = self._param_target(scope, payload)
        path = payload["path"]
        name = str(payload.get("name") or "")
        spec = self.ir.hparams.get(name)
        target.args[path] = payload.get("value", spec.default if spec else None)
        # 아무도 안 가리키면 하이퍼파라미터도 같이 사라진다 - 안 그러면 목록에 유령이 쌓인다.
        if name and not _references_hp(self.ir, name):
            self.ir.hparams.pop(name, None)

    def _define_composite(self, payload: dict[str, Any]) -> None:
        """묶음 블록 정의를 그래프에 넣는다 - 팔레트가 템플릿의 BasicBlock을 꺼내 쓸 때.

        같은 이름이 이미 있으면 몸체가 같을 때만 통과한다. 다른 몸체를 덮어쓰면 그 이름을
        부르는 기존 노드들이 소리 없이 바뀐다.
        """
        from ..ir import Composite

        name = str(payload.get("name") or "").strip()
        if not name:
            raise OpError("define_composite needs a name")
        try:
            composite = Composite.model_validate(payload.get("body") or {})
        except Exception as exc:
            raise OpError(f"invalid composite: {exc}") from exc
        existing = self.ir.composites.get(name)
        if existing is not None and existing != composite:
            raise OpError(f"컴포지트 {name}이 이미 있고 몸체가 다릅니다 - 다른 이름을 쓰세요")
        self.ir.composites[name] = composite

    def _remove_composite(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "")
        if name not in self.ir.composites:
            raise OpError(f"unknown composite {name}")
        scopes = [self.ir.graph, *self.ir.composites.values()]
        used = any(instance.type == f"composite:{name}" or instance.body == f"composite:{name}"
                   for scope in scopes for instance in scope.instances.values())
        if used:
            raise OpError(f"컴포지트 {name}을 부르는 블록이 있어 지울 수 없습니다")
        del self.ir.composites[name]

    def _set_ports(self, scope, payload: dict[str, Any]) -> None:
        """노드의 출력 포트 규격을 바꾼다.

        ``Input`` 노드에서는 이것이 곧 그래프의 입력 규격이다(§5.1.6) - shape와
        dtype이 여기 있고 ``args``에는 없다.
        """
        from ..ir import Port

        node = next((n for n in scope.nodes if n.id == payload.get("node")), None)
        if node is None:
            raise OpError(f"unknown node {payload.get('node')}")
        try:
            node.ports_out = [Port.model_validate(port) for port in payload.get("ports_out") or []]
        except Exception as exc:
            raise OpError(f"invalid ports: {exc}") from exc

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
        # 학습 블록은 그래프에 하나다. 단계 표시줄을 연타하거나 클라이언트가 폭주해도 여기서
        # 막힌다 - 실제로 Train 노드 1,600개가 한 번에 생긴 적이 있다.
        if _kind(node.type) == "torchflow.Train" and any(
                _kind(existing.type) == "torchflow.Train" for existing in scope.nodes):
            raise OpError("학습 블록은 하나면 됩니다 - 이미 있는 Train 블록의 값을 고치세요")
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
