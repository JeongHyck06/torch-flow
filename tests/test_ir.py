"""IR 스키마와 결정적 직렬화 (기획서 §17.1, §10.1)."""

import json

import pytest

from torchflow.ir import (
    Composite, Graph, Instance, ModuleGraph, Node, canonical_json, load, topo_order, validate,
)
from conftest import MINIVIT


def test_spec_example_round_trips(minivit):
    """§17.1 형태의 그래프가 손실 없이 왕복한다."""
    once = canonical_json(minivit)
    twice = canonical_json(ModuleGraph.model_validate_json(once))
    assert once == twice


def test_serialization_is_key_order_independent(minivit):
    """디스크 키 순서가 달라도 정본 직렬화는 바이트 동일하다."""
    payload = json.loads(MINIVIT.read_text(encoding="utf-8"))
    shuffled = json.loads(json.dumps(payload, sort_keys=True, indent=4))
    assert canonical_json(ModuleGraph.model_validate(shuffled)) == canonical_json(minivit)


def test_label_is_the_first_key(minivit):
    """노드 객체는 label이 첫 키다 - diff가 코드 순서와 대응한다(§10.1)."""
    nodes = json.loads(canonical_json(minivit))["graph"]["nodes"]
    assert all(next(iter(node)) == "label" for node in nodes)


def test_nodes_serialize_in_topological_order(minivit):
    labels = [n["label"] for n in json.loads(canonical_json(minivit))["graph"]["nodes"]]
    assert labels.index("x") < labels.index("patch") < labels.index("blocks") < labels.index("head")


def test_unknown_fields_survive_a_round_trip():
    """미지의 필드는 opaque로 보존된다 - 상위 버전 파일을 깨뜨리지 않는다."""
    payload = {
        "graph": {
            "name": "X",
            "nodes": [{"label": "a", "id": "1", "type": "torch.add", "future_field": {"k": 1}}],
        }
    }
    restored = json.loads(canonical_json(ModuleGraph.model_validate(payload)))
    assert restored["graph"]["nodes"][0]["future_field"] == {"k": 1}


def test_topo_order_is_stable_under_input_shuffling():
    nodes = [Node(label=name, id=name, type="torch.add") for name in "abcd"]
    edges = [("a.output", "b.input"), ("b.output", "c.input"), ("a.output", "d.input")]
    forward = [n.id for n in topo_order(nodes, edges)]
    backward = [n.id for n in topo_order(list(reversed(nodes)), list(reversed(edges)))]
    assert forward == backward == ["a", "b", "c", "d"]


def test_topo_order_stays_deterministic_on_a_cycle():
    nodes = [Node(label=name, id=name, type="torch.add") for name in "ab"]
    edges = [("a.output", "b.input"), ("b.output", "a.input")]
    assert [n.id for n in topo_order(nodes, edges)] == ["a", "b"]


def test_float_normalization_rejects_non_finite():
    ir = ModuleGraph(graph=Graph(name="X", nodes=[
        Node(label="a", id="1", type="torch.add", args={"eps": float("nan")})]))
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(ir)


def test_validate_accepts_the_example(minivit):
    assert validate(minivit) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda ir: ir.graph.nodes.append(Node(label="dup", id="01J9Q4B1", type="torch.add")),
         "duplicate node id"),
        (lambda ir: setattr(ir.graph.nodes[0], "call", "nope"), "exactly one of"),
        (lambda ir: ir.graph.instances["01J9I103"].args.__setitem__("in_features", {"$hp": "ghost"}),
         "unknown $hp"),
        (lambda ir: ir.graph.edges.append(("ghost.out", "01J9Q4B4.input")), "no such node"),
    ],
)
def test_validate_reports_structural_faults(minivit, mutate, expected):
    mutate(minivit)
    assert any(expected in problem for problem in validate(minivit))


def test_validate_reports_cycles():
    ir = ModuleGraph(graph=Graph(
        name="X",
        nodes=[Node(label="a", id="a", type="torch.add"), Node(label="b", id="b", type="torch.add")],
        edges=[("a.output", "b.input"), ("b.output", "a.input")],
    ))
    assert any("cycle" in problem for problem in validate(ir))


def test_saved_file_reloads_identically(minivit, tmp_path):
    from torchflow.ir import save

    path = tmp_path / "model.tfg.json"
    save(minivit, path)
    assert canonical_json(load(path)) == canonical_json(minivit)
