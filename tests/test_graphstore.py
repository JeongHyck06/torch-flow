"""op-log · seq · journal (기획서 §8.2.2)."""

import pytest

from torchflow.ir import canonical_json, validate
from torchflow.hub.graphstore import GraphStore, OpError


def op(kind, **payload):
    return {"client_id": "c-1", "tmp_seq": 1, "kind": kind, "payload": payload}


def test_seq_is_monotonic(minivit):
    store = GraphStore(minivit)
    seqs = [store.apply(op("set_param", instance="01J9I103", path="bias", value=False))
            for _ in range(3)]
    assert seqs == [1, 2, 3]


def test_set_param_mutates_the_authoritative_graph(minivit):
    store = GraphStore(minivit)
    store.apply(op("set_param", instance="01J9I103", path="in_features", value=256))
    assert store.ir.graph.instances["01J9I103"].args["in_features"] == 256


def test_switch_rejects_an_unknown_variant(minivit):
    store = GraphStore(minivit)
    with pytest.raises(OpError, match="unknown variant"):
        store.apply(op("set_switch_active", composite="Block", instance="01J9I002", active="ghost"))
    assert store.seq == 0, "rejected op must not consume a seq"


def test_journal_replays_to_the_same_graph(minivit, tmp_path):
    from torchflow.ir import canonical_json, load

    from conftest import MINIVIT

    journal = tmp_path / "journal.jsonl"
    store = GraphStore(minivit, journal)
    store.apply(op("set_param", instance="01J9I103", path="bias", value=False))
    store.apply(op("set_switch_active", composite="Block", instance="01J9I002", active="none"))
    expected = canonical_json(store.ir)

    restarted = GraphStore(load(MINIVIT), journal)
    assert restarted.replay() == 2
    assert canonical_json(restarted.ir) == expected
    assert restarted.seq == 2


def test_since_returns_the_tail_for_resync(minivit):
    store = GraphStore(minivit)
    for _ in range(4):
        store.apply(op("set_param", instance="01J9I103", path="bias", value=False))
    assert [record["seq"] for record in store.since(2)] == [3, 4]


def test_unapplied_op_kinds_are_still_journaled(minivit, tmp_path):
    """M5에서 붙일 op도 지금 journal에는 남는다 - 재생 시 순서가 보존된다."""
    journal = tmp_path / "journal.jsonl"
    store = GraphStore(minivit, journal)
    store.apply(op("move", node="01J9Q4B1", x=10, y=20))
    assert store.seq == 1
    assert "move" in journal.read_text(encoding="utf-8")


# 편집 op (§8.2.2, M5)


def add_linear(store, node_id="n-new", instance_id="i-new"):
    return store.apply({
        "kind": "add_node",
        "payload": {
            "instance": {"id": instance_id, "label": "extra", "type": "torch.nn.Linear",
                         "args": {"in_features": 192, "out_features": 192}},
            "node": {"id": node_id, "label": "extra", "method": "forward"},
        },
    })


def test_add_node_creates_the_node_and_its_instance(minivit):
    store = GraphStore(minivit)
    add_linear(store)

    node = next(node for node in store.ir.graph.nodes if node.id == "n-new")
    assert node.call == "i-new" and node.method == "forward"
    assert store.ir.graph.instances["i-new"].type == "torch.nn.Linear"
    assert validate(store.ir) == []


def test_add_node_rejects_a_duplicate_id(minivit):
    store = GraphStore(minivit)
    with pytest.raises(OpError):
        store.apply({"kind": "add_node",
                     "payload": {"node": {"id": "01J9Q4B4", "label": "dup", "type": "torch.mean"}}})


def test_add_node_rejects_a_node_without_call_or_type(minivit):
    store = GraphStore(minivit)
    with pytest.raises(OpError):
        store.apply({"kind": "add_node", "payload": {"node": {"id": "n-bad", "label": "bad"}}})


def test_remove_node_takes_its_edges_and_orphan_instance(minivit):
    store = GraphStore(minivit)
    before = len(store.ir.graph.edges)
    store.apply({"kind": "remove_node", "payload": {"node": "01J9Q4B4"}})   # pool

    assert all(node.id != "01J9Q4B4" for node in store.ir.graph.nodes)
    assert len(store.ir.graph.edges) == before - 2
    assert not any("01J9Q4B4" in src or "01J9Q4B4" in dst for src, dst in store.ir.graph.edges)
    # head는 자기 인스턴스를 계속 쓴다 - 남의 인스턴스를 지우지 않는다.
    assert "01J9I103" in store.ir.graph.instances


def test_remove_node_keeps_a_shared_instance(minivit):
    """같은 인스턴스를 부르는 노드가 남아 있으면 인스턴스는 살아 있어야 한다(tied weights)."""
    store = GraphStore(minivit)
    add_linear(store, node_id="n-a", instance_id="i-shared")
    store.apply({"kind": "add_node",
                 "payload": {"node": {"id": "n-b", "label": "twin", "call": "i-shared",
                                      "method": "forward"}}})

    store.apply({"kind": "remove_node", "payload": {"node": "n-a"}})
    assert "i-shared" in store.ir.graph.instances

    store.apply({"kind": "remove_node", "payload": {"node": "n-b"}})
    assert "i-shared" not in store.ir.graph.instances


def test_add_and_remove_are_inverses(minivit):
    """실행 취소는 역 op를 새 op로 보내는 것이다(§8.2.3) - 왕복이 원본과 같아야 한다."""
    store = GraphStore(minivit)
    before = canonical_json(store.ir)

    add_linear(store)
    store.apply({"kind": "remove_node", "payload": {"node": "n-new"}, "inverse_of": 1})

    assert canonical_json(store.ir) == before
    assert store.seq == 2          # 되감기가 아니라 앞으로 나아간 두 op다


def test_edits_inside_a_composite_stay_there(minivit):
    store = GraphStore(minivit)
    store.apply({"kind": "add_node",
                 "payload": {"composite": "MLP",
                             "node": {"id": "n-drop", "label": "drop", "type": "torch.detach"}}})

    assert any(node.id == "n-drop" for node in store.ir.composites["MLP"].nodes)
    assert all(node.id != "n-drop" for node in store.ir.graph.nodes)


def test_set_param_reaches_node_args_too(minivit):
    """호출 인자(torch.mean의 dim)는 인스턴스가 아니라 노드에 붙는다."""
    store = GraphStore(minivit)
    store.apply({"kind": "set_param", "payload": {"node": "01J9Q4B4", "path": "dim", "value": 2}})

    node = next(node for node in store.ir.graph.nodes if node.id == "01J9Q4B4")
    assert node.args["dim"] == 2

    # null은 삭제다 - 기본값으로 되돌리는 역 op가 성립해야 한다(§8.2.3).
    store.apply({"kind": "set_param", "payload": {"node": "01J9Q4B4", "path": "dim", "value": None}})
    assert "dim" not in next(n for n in store.ir.graph.nodes if n.id == "01J9Q4B4").args


def test_set_ports_changes_the_graph_input_spec(minivit):
    """Input 노드의 shape는 args가 아니라 ports_out에 있다(§5.1.6)."""
    store = GraphStore(minivit)
    store.apply({"kind": "set_ports", "payload": {
        "node": "01J9Q4B1",
        "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", 3, 64, 64],
                       "dtype": "float32"}]}})

    node = next(node for node in store.ir.graph.nodes if node.id == "01J9Q4B1")
    assert node.ports_out[0].shape == ["B", 3, 64, 64]
    assert validate(store.ir) == []


def test_rename_changes_the_graph_name_and_is_undoable(minivit):
    store = GraphStore(minivit)
    store.apply({"kind": "rename", "payload": {"name": "TinyCNN"}})
    assert store.ir.graph.name == "TinyCNN"

    store.apply({"kind": "rename", "payload": {"name": "MiniViT"}, "inverse_of": 1})
    assert store.ir.graph.name == "MiniViT"

    with pytest.raises(OpError):
        store.apply({"kind": "rename", "payload": {"name": "  "}})


def test_a_graph_keeps_one_train_block(minivit):
    """단계 표시줄을 연타해도, 클라이언트가 폭주해도 학습 블록은 하나다."""
    store = GraphStore(minivit)
    train = {"id": "T1", "label": "train", "type": "torchflow.Train", "args": {"steps": 10}}
    store.apply(op("add_node", node=train))
    with pytest.raises(OpError, match="하나면"):
        store.apply(op("add_node", node={**train, "id": "T2", "type": "torchflow.Train@0.0.1"}))
    assert sum(1 for node in store.ir.graph.nodes if (node.type or "").startswith("torchflow.Train")) == 1


def test_define_and_remove_composite(minivit):
    """팔레트가 묶음 블록을 꺼내 쓸 때: 정의 넣기 -> 부르기 -> (부르는 노드가 있으면) 지우기 거부."""
    store = GraphStore(minivit)
    body = {"doc": "x2", "params": {"k": {"type": "int"}},
            "ports": {"in": [{"name": "x"}], "out": [{"name": "y"}]},
            "instances": {}, "nodes": [], "edges": []}
    store.apply(op("define_composite", name="Twice", body=body))
    assert "Twice" in store.ir.composites
    store.apply(op("define_composite", name="Twice", body=body))     # 같은 몸체는 통과
    with pytest.raises(OpError, match="몸체가 다릅니다"):
        store.apply(op("define_composite", name="Twice", body={**body, "doc": "other"}))

    store.apply(op("add_node", instance={"id": "I1", "label": "twice", "type": "composite:Twice", "args": {"k": 2}},
                   node={"id": "N1", "label": "twice", "method": "forward"}))
    with pytest.raises(OpError, match="부르는 블록"):
        store.apply(op("remove_composite", name="Twice"))
    store.apply(op("remove_node", node="N1"))
    store.apply(op("remove_composite", name="Twice"))
    assert "Twice" not in store.ir.composites
