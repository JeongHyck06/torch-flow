"""op-log · seq · journal (기획서 §8.2.2)."""

import pytest

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
