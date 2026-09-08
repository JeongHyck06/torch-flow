"""TorchFlow IR v1.

스키마는 기획서 §17.1, 결정적 직렬화 규칙은 §10.1을 따른다.

설계 원칙 두 가지:

1. 값은 열어 둔다. 노드/인스턴스의 ``args`` 값은 스칼라이거나 참조
   (``{"$hp": ...}`` / ``{"$p": ...}`` / ``{"$expr": ...}``)이거나 중첩
   컨테이너다. 이를 타입으로 좁히지 않고 ``Any``로 두는 대신 :func:`ref_of`
   같은 헬퍼와 :func:`validate`로 검사한다. §10.1의 "미지의 노드 타입은
   opaque 보존" 요구가 이 선택을 강제한다.
2. 미지 필드는 살려서 되돌린다. 모든 모델이 ``extra="allow"``이므로
   상위 버전이 쓴 필드가 왕복에서 사라지지 않는다.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

# ``$expr``까지 셋만 참조로 인정한다(§4.3).
REF_KEYS = ("$hp", "$p", "$expr")

CELL_KINDS = ("CellModule", "CellFunction", "CellStep", "CellHook", "CellData")


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


# 값 참조 헬퍼


def ref_of(value: Any) -> tuple[str, Any] | None:
    """``value``가 참조면 ``("$hp", "dim")`` 꼴로, 아니면 ``None``."""
    if isinstance(value, dict) and len(value) == 1:
        (key, inner), = value.items()
        if key in REF_KEYS:
            return key, inner
    return None


def iter_refs(value: Any):
    """중첩 컨테이너를 훑어 모든 참조를 산출한다."""
    ref = ref_of(value)
    if ref is not None:
        yield ref
        return
    if isinstance(value, dict):
        for inner in value.values():
            yield from iter_refs(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from iter_refs(inner)


# 엣지 엔드포인트

IN_BOUNDARY = "$in"
OUT_BOUNDARY = "$out"


def split_endpoint(endpoint: str) -> tuple[str, str]:
    """``"01J9Q4A1.output"`` -> ``("01J9Q4A1", "output")``.

    ``$in.x`` / ``$out.y`` 경계도 같은 형태로 쪼갠다.
    """
    node, _, port = endpoint.partition(".")
    if not node or not port:
        raise ValueError(f"malformed endpoint: {endpoint!r}")
    return node, port


# 모델


class Port(_Base):
    name: str
    type: str = "Tensor"
    shape: list[str | int] | None = None
    dtype: str | None = None


class HParam(_Base):
    type: Literal["int", "float", "bool", "str", "enum"]
    default: Any = None
    choices: list[Any] | None = None
    space: dict[str, Any] | None = None


class Instance(_Base):
    """살아 있는 모듈 하나. 여러 호출 노드가 공유할 수 있다(tied embedding)."""

    label: str
    type: str
    args: dict[str, Any] = Field(default_factory=dict)

    # torchflow.Switch (§4.4.2)
    active: Any = None
    variants: dict[str, dict[str, Any]] | None = None

    # torchflow.Repeat (§4.4.1)
    body: str | None = None
    count: Any = None
    mode: Literal["sequential", "parallel", "shared_weights"] | None = None
    reduce: Any = None
    bind: dict[str, Any] | None = None


class Node(_Base):
    """호출 노드. ``call``(인스턴스 참조) 또는 ``type``(순수 연산) 중 하나."""

    label: str
    id: str
    type: str | None = None
    call: str | None = None
    method: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    ports_out: list[Port] | None = None
    enabled: Any = None


class Probe(_Base):
    label: str
    id: str
    type: str
    on_edge: tuple[str, str] | None = None
    on: str | None = None  # Attach Mode의 속성 경로
    auto_policy: str = "manual"


class CodeCell(_Base):
    kind: Literal["CellModule", "CellFunction", "CellStep", "CellHook", "CellData"]
    file: str
    # 본문 바이트. 그래프로 못 편 코드는 여기 그대로 살아 있고 codegen이 되돌려 쓴다
    # (§7.4.2 승격 규칙 - 손으로 쓴 코드가 다음 codegen에서 사라지면 안 된다).
    source: str | None = None
    ports: dict[str, list[Port]] = Field(default_factory=dict)
    shape_fn: str = "cpu_probe"
    export_compatible: bool = False
    children_ports: list[str] = Field(default_factory=list)
    sha256: str | None = None


class Composite(_Base):
    doc: str | None = None
    params: dict[str, HParam] = Field(default_factory=dict)
    ports: dict[str, list[Port]] = Field(default_factory=dict)
    instances: dict[str, Instance] = Field(default_factory=dict)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    user_code: dict[str, str] = Field(default_factory=dict)


class Graph(_Base):
    name: str
    instances: dict[str, Instance] = Field(default_factory=dict)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)
    init: dict[str, Any] = Field(default_factory=dict)
    probe_objective: str = "auto"
    user_code: dict[str, str] = Field(default_factory=dict)


class ModuleGraph(_Base):
    """``graph/model.tfg.json``의 루트."""

    schema_version: str = SCHEMA_VERSION
    meta: dict[str, Any] = Field(default_factory=dict)
    hparams: dict[str, HParam] = Field(default_factory=dict)
    composites: dict[str, Composite] = Field(default_factory=dict)
    graph: Graph
    code_cells: dict[str, CodeCell] = Field(default_factory=dict)
    experiment: dict[str, Any] | None = None


class ExperimentGraph(_Base):
    """``graph/experiment.tfg.json``의 루트."""

    schema_version: str = SCHEMA_VERSION
    nodes: list[Node] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    exec: list[tuple[str, str]] = Field(default_factory=list)


# 위상 정렬 (§10.1: 위상 순, label, ULID)


def topo_order(nodes: list[Node], edges: list[tuple[str, str]]) -> list[Node]:
    """노드를 위상 순으로 정렬한다. 동순위는 ``(label, id)``로 결정한다.

    사이클이 있으면 남은 노드를 ``(label, id)`` 순으로 뒤에 붙인다. 정렬은
    직렬화용이므로 잘못된 그래프에서도 결정적이어야 하고, 사이클 보고는
    :func:`validate`의 몫이다.
    """
    ids = {n.id for n in nodes}
    indeg = dict.fromkeys(ids, 0)
    succ: dict[str, set[str]] = {i: set() for i in ids}
    for src, dst in edges:
        s, _ = split_endpoint(src)
        d, _ = split_endpoint(dst)
        if s in ids and d in ids and d not in succ[s]:
            succ[s].add(d)
            indeg[d] += 1

    by_id = {n.id: n for n in nodes}
    rank = {n.id: (n.label, n.id) for n in nodes}
    ready = sorted((i for i in ids if indeg[i] == 0), key=lambda i: rank[i])
    out: list[Node] = []
    while ready:
        current = ready.pop(0)
        out.append(by_id[current])
        for nxt in sorted(succ[current], key=lambda i: rank[i]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda i: rank[i])

    placed = {n.id for n in out}
    out.extend(sorted((by_id[i] for i in ids - placed), key=lambda n: rank[n.id]))
    return out


# 결정적 직렬화 (§10.1)


def _normalize(value: Any) -> Any:
    """부동소수 정규화. ``-0.0``을 접고 NaN/Inf는 거부한다."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float in IR: {value!r}")
        if value == 0.0:
            return 0.0
        if value.is_integer() and abs(value) < 1e16:
            return value  # 1.0은 1.0으로 남긴다(int와 구분).
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        keys = sorted(value)
        # 노드/인스턴스 객체는 label을 첫 키로 고정한다(§10.1).
        if "label" in value:
            keys.remove("label")
            keys.insert(0, "label")
        return {k: _canonical(value[k]) for k in keys}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return _normalize(value)


def _ordered_payload(model: BaseModel) -> Any:
    # mode="json"은 NaN/Inf를 조용히 null로 바꾼다. 정본 직렬화에서는 손실이므로
    # python 덤프를 받아 _normalize가 직접 거부하게 한다.
    payload = model.model_dump(mode="python", exclude_defaults=True, exclude_none=True)

    def order_scope(scope: dict[str, Any], nodes: list[Node]) -> None:
        if "nodes" in scope:
            edges = [tuple(e) for e in scope.get("edges", [])]
            order = {n.id: i for i, n in enumerate(topo_order(nodes, edges))}
            scope["nodes"].sort(key=lambda n: order.get(n.get("id", ""), 1 << 30))
        if "edges" in scope:
            scope["edges"] = sorted(tuple(e) for e in scope["edges"])
        if "probes" in scope:
            scope["probes"].sort(key=lambda p: (p.get("label", ""), p.get("id", "")))

    if isinstance(model, ModuleGraph):
        order_scope(payload.get("graph", {}), model.graph.nodes)
        for name, composite in model.composites.items():
            order_scope(payload.get("composites", {}).get(name, {}), composite.nodes)
    elif isinstance(model, ExperimentGraph):
        order_scope(payload, model.nodes)
        if "exec" in payload:
            payload["exec"] = sorted(tuple(e) for e in payload["exec"])
    return payload


def canonical_json(model: BaseModel) -> str:
    """바이트 단위로 재현 가능한 직렬화. ``torchflow check``의 기준이다."""
    return json.dumps(
        _canonical(_ordered_payload(model)),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def dumps_pretty(model: BaseModel) -> str:
    """디스크에 쓰는 형태. 키 순서는 :func:`canonical_json`과 동일하다."""
    return json.dumps(
        _canonical(_ordered_payload(model)), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"


# 검증


def validate(ir: ModuleGraph) -> list[str]:
    """구조 오류를 문자열 목록으로 돌려준다. 빈 목록이면 통과."""
    errors: list[str] = []

    def check_scope(where: str, nodes, instances, edges, params: set[str]) -> None:
        ids = set()
        for node in nodes:
            if node.id in ids:
                errors.append(f"{where}: duplicate node id {node.id}")
            ids.add(node.id)
            if (node.call is None) == (node.type is None):
                errors.append(f"{where}/{node.id}: exactly one of 'call' or 'type' required")
            if node.call is not None and node.call not in instances:
                errors.append(f"{where}/{node.id}: unknown instance {node.call}")

        for source in list(nodes) + list(instances.values()):
            for kind, name in iter_refs(getattr(source, "args", {}) or {}):
                if kind == "$hp" and name not in ir.hparams:
                    errors.append(f"{where}: unknown $hp {name!r}")
                elif kind == "$p" and name not in params:
                    errors.append(f"{where}: unknown $p {name!r}")

        for src, dst in edges:
            for endpoint in (src, dst):
                node, _ = split_endpoint(endpoint)
                if node not in ids and node not in (IN_BOUNDARY, OUT_BOUNDARY):
                    errors.append(f"{where}: edge endpoint {endpoint!r} has no such node")

        order = topo_order(list(nodes), list(edges))
        if len(order) == len(nodes):
            index = {n.id: i for i, n in enumerate(order)}
            for src, dst in edges:
                s, _ = split_endpoint(src)
                d, _ = split_endpoint(dst)
                if s in index and d in index and index[s] > index[d]:
                    errors.append(f"{where}: cycle through {s} -> {d}")

    check_scope("graph", ir.graph.nodes, ir.graph.instances, ir.graph.edges, set())
    for name, composite in ir.composites.items():
        check_scope(
            f"composites/{name}",
            composite.nodes,
            composite.instances,
            composite.edges,
            set(composite.params),
        )

    for cid, cell in ir.code_cells.items():
        if cell.kind not in CELL_KINDS:
            errors.append(f"code_cells/{cid}: unknown kind {cell.kind}")
    return errors


def load(path) -> ModuleGraph:
    from pathlib import Path

    return ModuleGraph.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save(ir: ModuleGraph, path) -> None:
    from pathlib import Path

    Path(path).write_text(dumps_pretty(ir), encoding="utf-8")
