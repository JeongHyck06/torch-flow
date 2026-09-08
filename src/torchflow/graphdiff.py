"""두 그래프의 차이를 사람 말로 (기획서 §10.2의 ``torchflow diff --markdown``).

리뷰어가 보는 것은 생성 코드지만, **무엇이 바뀌었는지**는 그래프에서 훨씬 짧게
읽힌다: "노드 +3 -1, blocks.count 6->12, Switch attn 변형 linear 추가".
노드 id(ULID)가 커밋 사이에 보존되므로(§7.5) 짝짓기는 id로 한다.

torch를 import하지 않는다.
"""

from __future__ import annotations

from typing import Any

from .ir import ModuleGraph


def summarize(before: ModuleGraph, after: ModuleGraph) -> list[str]:
    """바뀐 것 한 줄씩. 빈 목록이면 그래프가 같다."""
    lines: list[str] = []
    if before.graph.name != after.graph.name:
        lines.append(f"이름 {before.graph.name} -> {after.graph.name}")

    for scope, old, new in _scopes(before, after):
        where = "" if scope == "$graph" else f"{scope}: "
        if old is None:
            lines.append(f"묶음 블록 {scope} 추가 ({len(new.nodes)}개 블록)")
            continue
        if new is None:
            lines.append(f"묶음 블록 {scope} 삭제")
            continue
        old_nodes = {node.id: node for node in old.nodes}
        new_nodes = {node.id: node for node in new.nodes}
        added = [new_nodes[key] for key in new_nodes.keys() - old_nodes.keys()]
        removed = [old_nodes[key] for key in old_nodes.keys() - new_nodes.keys()]
        if added:
            lines.append(f"{where}블록 +{len(added)} ({_names(added)})")
        if removed:
            lines.append(f"{where}블록 -{len(removed)} ({_names(removed)})")

        for key in sorted(old_nodes.keys() & new_nodes.keys()):
            lines.extend(f"{where}{line}" for line in
                         _args_diff(new_nodes[key].label, old_nodes[key].args,
                                    new_nodes[key].args))
        for key in sorted(set(old.instances) & set(new.instances)):
            one, other = old.instances[key], new.instances[key]
            lines.extend(f"{where}{line}" for line in
                         _args_diff(other.label, one.args, other.args))
            lines.extend(f"{where}{line}" for line in _variants(other.label, one, other))
            if one.count != other.count:
                lines.append(f"{where}{other.label}.count {one.count} -> {other.count}")
            if one.active != other.active:
                lines.append(f"{where}Switch {other.label} 활성 {one.active} -> {other.active}")

        old_edges, new_edges = set(map(tuple, old.edges)), set(map(tuple, new.edges))
        rewired = len(old_edges ^ new_edges)
        if rewired:
            lines.append(f"{where}연결 {rewired}군데 바뀜")

    for name in sorted(set(after.hparams) - set(before.hparams)):
        lines.append(f"하이퍼파라미터 {name} 추가")
    for name in sorted(set(before.hparams) - set(after.hparams)):
        lines.append(f"하이퍼파라미터 {name} 삭제")
    for name in sorted(set(before.hparams) & set(after.hparams)):
        one, other = before.hparams[name].default, after.hparams[name].default
        if one != other:
            lines.append(f"하이퍼파라미터 {name} 기본값 {one} -> {other}")

    for name in sorted(set(after.code_cells) - set(before.code_cells)):
        lines.append(f"Code Cell {name} 추가")
    for name in sorted(set(before.code_cells) & set(after.code_cells)):
        if before.code_cells[name].sha256 != after.code_cells[name].sha256:
            lines.append(f"Code Cell {name} 내용 바뀜")

    for slot in sorted(set(before.graph.user_code) | set(after.graph.user_code)):
        if before.graph.user_code.get(slot) != after.graph.user_code.get(slot):
            lines.append(f"user-slot {slot} 바뀜")
    return lines


def markdown(before: ModuleGraph, after: ModuleGraph, *, title: str = "TorchFlow 그래프 변경") -> str:
    """PR 본문에 그대로 붙일 요약(§10.2)."""
    lines = summarize(before, after)
    if not lines:
        return f"### {title}\n\n그래프는 그대로입니다.\n"
    body = "\n".join(f"- {line}" for line in lines)
    return f"### {title}\n\n{body}\n"


def _scopes(before: ModuleGraph, after: ModuleGraph):
    """(이름, 이전 스코프, 이후 스코프). 한쪽에만 있으면 그쪽이 ``None``."""
    yield "$graph", before.graph, after.graph
    for name in sorted(set(before.composites) | set(after.composites)):
        yield name, before.composites.get(name), after.composites.get(name)


def _names(nodes: list[Any], limit: int = 4) -> str:
    labels = [node.label for node in nodes[:limit]]
    return ", ".join(labels) + (" ..." if len(nodes) > limit else "")


def _args_diff(label: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            lines.append(f"{label}.{key} {_value(old.get(key))} -> {_value(new.get(key))}")
    return lines


def _variants(label: str, old, new) -> list[str]:
    one, other = set(old.variants or {}), set(new.variants or {})
    lines = [f"Switch {label} 변형 {name} 추가" for name in sorted(other - one)]
    lines += [f"Switch {label} 변형 {name} 삭제" for name in sorted(one - other)]
    return lines


def _value(value: Any) -> str:
    if value is None:
        return "(없음)"
    if isinstance(value, dict) and len(value) == 1:
        (key, inner), = value.items()
        if key.startswith("$"):
            return f"{key} {inner}"
    return str(value)
