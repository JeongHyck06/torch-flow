"""축소 3-way merge (기획서 §10.2의 4, §13.1 M9).

브랜치 둘이 같은 그래프를 고쳤을 때, **서로 다른 곳을 고쳤으면** 자동으로 합친다:
노드 추가·삭제, 인자 변경, 엣지 연결·해제, 하이퍼파라미터. 같은 자리를 서로 다르게
고쳤으면 합치지 않고 그 자리를 이름으로 보고한다 - 조용히 한쪽을 고르는 것보다 낫다.

좌표는 여기 없다. ``layout.json``은 ``merge=ours``라서 애초에 충돌하지 않는다(§10.1).

torch를 import하지 않는다.
"""

from __future__ import annotations

from typing import Any

from .ir import Composite, ModuleGraph


def merge(base: ModuleGraph, ours: ModuleGraph, theirs: ModuleGraph) -> tuple[ModuleGraph, list[str]]:
    """``(합친 그래프, 충돌 목록)``. 충돌 목록이 비어 있으면 자동 병합에 성공한 것이다."""
    merged = ours.model_copy(deep=True)
    conflicts: list[str] = []

    merged.graph.name = _pick("graph.name", base.graph.name, ours.graph.name, theirs.graph.name,
                              conflicts)
    _merge_scope("graph", base.graph, ours.graph, theirs.graph, merged.graph, conflicts)

    for name in sorted(set(ours.composites) | set(theirs.composites)):
        in_base = base.composites.get(name)
        mine, other = ours.composites.get(name), theirs.composites.get(name)
        if other is None:
            continue                                    # 저쪽이 안 건드렸거나 지웠다 - 우리 것을 둔다
        if mine is None:
            merged.composites[name] = other.model_copy(deep=True)
            continue
        _merge_scope(f"composites/{name}", in_base or Composite(), mine, other,
                     merged.composites[name], conflicts)

    for name in sorted(set(ours.hparams) | set(theirs.hparams)):
        mine, other = ours.hparams.get(name), theirs.hparams.get(name)
        if other is None:
            continue
        if mine is None:
            merged.hparams[name] = other.model_copy(deep=True)
            continue
        in_base = base.hparams.get(name)
        merged.hparams[name].default = _pick(
            f"hparams.{name}.default", in_base.default if in_base else None,
            mine.default, other.default, conflicts)

    for name, variant in theirs.variant_sets.items():
        merged.variant_sets.setdefault(name, variant)
    return merged, conflicts


def _merge_scope(where: str, base, ours, theirs, into, conflicts: list[str]) -> None:
    base_nodes = {node.id: node for node in base.nodes}
    our_nodes = {node.id: node for node in ours.nodes}
    their_nodes = {node.id: node for node in theirs.nodes}

    # 한쪽이 지운 것은 지운 채로. 양쪽이 다르게 만졌으면 충돌이다.
    for node_id in sorted(base_nodes.keys() - their_nodes.keys()):
        if node_id in our_nodes and _differs(base_nodes[node_id], our_nodes[node_id]):
            conflicts.append(f"{where}/{base_nodes[node_id].label}: 한쪽은 지우고 한쪽은 고쳤습니다")
    into.nodes = [node for node in into.nodes
                  if not (node.id in base_nodes and node.id not in their_nodes
                          and not _differs(base_nodes[node.id], node))]
    placed = {node.id for node in into.nodes}
    for node_id, node in their_nodes.items():
        if node_id not in placed and node_id not in base_nodes:
            into.nodes.append(node.model_copy(deep=True))   # 저쪽이 새로 놓은 블록
            placed.add(node_id)

    for node in into.nodes:
        mine, other = our_nodes.get(node.id), their_nodes.get(node.id)
        if mine is None or other is None:
            continue
        old = base_nodes.get(node.id)
        node.args = _merge_args(f"{where}/{node.label}", getattr(old, "args", {}) or {},
                                mine.args, other.args, conflicts)
        node.enabled = _pick(f"{where}/{node.label}.enabled",
                             getattr(old, "enabled", None), mine.enabled, other.enabled, conflicts)

    for instance_id in sorted(set(theirs.instances) - set(into.instances)):
        if instance_id not in base.instances:
            into.instances[instance_id] = theirs.instances[instance_id].model_copy(deep=True)
    for instance_id, instance in into.instances.items():
        mine, other = ours.instances.get(instance_id), theirs.instances.get(instance_id)
        if mine is None or other is None:
            continue
        old = base.instances.get(instance_id)
        label = instance.label
        instance.args = _merge_args(f"{where}/{label}", getattr(old, "args", {}) or {},
                                    mine.args, other.args, conflicts)
        instance.active = _pick(f"{where}/{label}.active", getattr(old, "active", None),
                                mine.active, other.active, conflicts)
        instance.count = _pick(f"{where}/{label}.count", getattr(old, "count", None),
                               mine.count, other.count, conflicts)

    # 엣지는 값이 없는 집합이다 - 한쪽이 이었으면 잇고, 한쪽이 끊었으면 끊는다.
    base_edges = {tuple(edge) for edge in base.edges}
    our_edges = {tuple(edge) for edge in ours.edges}
    their_edges = {tuple(edge) for edge in theirs.edges}
    alive = {node.id for node in into.nodes} | {"$in", "$out"}
    edges = (our_edges | (their_edges - base_edges)) - (base_edges - their_edges)
    into.edges = sorted(edge for edge in edges
                        if all(end.split(".")[0] in alive for end in edge))


def _merge_args(where: str, base: dict[str, Any], ours: dict[str, Any],
                theirs: dict[str, Any], conflicts: list[str]) -> dict[str, Any]:
    merged = dict(ours)
    for key in set(base) | set(ours) | set(theirs):
        chosen = _pick(f"{where}.{key}", base.get(key), ours.get(key), theirs.get(key), conflicts)
        if chosen is None and key not in ours and key not in theirs:
            merged.pop(key, None)
        elif key in theirs or key in ours:
            merged[key] = chosen
        else:
            merged.pop(key, None)
    return merged


def _pick(where: str, base: Any, ours: Any, theirs: Any, conflicts: list[str]) -> Any:
    """3-way 하나. 한쪽만 바꿨으면 그쪽, 둘 다 같게 바꿨으면 그것, 다르면 충돌."""
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    conflicts.append(f"{where}: {ours!r} vs {theirs!r}")
    return ours


def _differs(one, other) -> bool:
    return one.model_dump(exclude_none=True) != other.model_dump(exclude_none=True)


def summary(ir: ModuleGraph) -> str:
    """``git diff``가 읽을 수 있는 형태로 그래프를 편다(§10.2의 textconv).

    JSON 그대로 보면 한 줄 바뀐 것도 수십 줄 diff가 된다. 사람이 읽을 줄로 바꾼다.
    """
    lines = [f"graph {ir.graph.name}"]
    for name, spec in sorted(ir.hparams.items()):
        lines.append(f"hparam {name} = {spec.default!r}")
    for scope_name, scope in [("graph", ir.graph),
                              *sorted((f"composite {name}", one)
                                      for name, one in ir.composites.items())]:
        lines.append(f"[{scope_name}]")
        for node in sorted(scope.nodes, key=lambda one: (one.label, one.id)):
            instance = scope.instances.get(node.call) if node.call else None
            kind = (instance.type if instance else node.type) or "?"
            args = instance.args if instance else node.args
            rendered = ", ".join(f"{key}={args[key]!r}" for key in sorted(args))
            lines.append(f"  {node.label}: {kind}({rendered})")
        for src, dst in sorted(tuple(edge) for edge in scope.edges):
            lines.append(f"  {src} -> {dst}")
    return "\n".join(lines) + "\n"
