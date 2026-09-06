"""상태 머신과 stale 전파 (기획서 §5.3, §5.4, §17.4)."""

import pytest

from torchflow.hub.engine import BLOCKED, ERROR, OK, STALE, Engine


def edit(kind, **payload):
    return {"kind": kind, "payload": payload}


def stale_nodes(engine, axis="L0"):
    return {node for (node, ax), state in engine.state.items() if ax == axis and state == STALE}


def test_edit_stales_only_the_downstream_closure(minivit):
    engine = Engine(minivit)
    engine.on_edit(edit("set_param", instance="01J9I103", path="bias", value=False))
    # head(01J9Q4B5)와 그 하류 logits만. 상류(x·patch·blocks·pool)는 그대로다.
    assert stale_nodes(engine) == {"01J9Q4B5", "01J9Q4B6"}


def test_edit_inside_a_composite_propagates_out_to_its_callers(minivit):
    """컴포지트 안을 고치면 그것을 부르는 바깥 노드까지 번져야 한다."""
    engine = Engine(minivit)
    engine.on_edit(edit("set_param", composite="Block", instance="01J9I001",
                        path="normalized_shape", value=256))
    staled = stale_nodes(engine)
    assert "01J9Q4B3" in staled, "blocks 호출 노드가 stale이 되지 않았다"
    assert {"01J9Q4B4", "01J9Q4B5", "01J9Q4B6"} <= staled, "바깥 하류가 stale이 아니다"


def test_grad_staleness_reaches_ancestors_too(minivit):
    """L1-bwd는 하류 편집에도 stale이다 - autograd는 전역이다(§5.1.2)."""
    engine = Engine(minivit)
    engine.on_edit(edit("set_param", instance="01J9I103", path="bias", value=False))
    l0 = stale_nodes(engine, "L0")
    grad = stale_nodes(engine, "L1bwd")
    assert grad > l0
    assert "01J9Q4B2" in grad and "01J9Q4B2" not in l0  # patch: 상류인데 grad는 stale


def test_layout_moves_change_no_execution_state(minivit):
    """좌표는 layout.json 소관이다 - 실행 상태를 건드리면 안 된다."""
    engine = Engine(minivit)
    assert engine.on_edit(edit("move", node="01J9Q4B1", x=10, y=20)) == set()
    assert stale_nodes(engine, "L0") == set()
    assert stale_nodes(engine, "L1bwd") == set()


def test_probes_do_not_stale_l0(minivit):
    """프로브는 그래프 의미를 바꾸지 않는다(§6.3)."""
    engine = Engine(minivit)
    engine.on_edit(edit("add_probe", node="01J9Q4B3", type="probe.GradFlow"))
    assert stale_nodes(engine, "L0") == set()
    assert stale_nodes(engine, "L1fwd")


def test_high_durability_nodes_survive_low_impact_edits(minivit):
    """데이터셋 같은 high durability 노드는 low 편집에 재해시되지 않는다(§5.2)."""
    minivit.graph.nodes[0].durability = "high"  # Input 노드
    engine = Engine(minivit)
    engine.on_edit(edit("add_probe", node="01J9Q4B1"))
    assert "01J9Q4B1" not in stale_nodes(engine, "L1fwd")


def test_version_advances_only_for_staled_nodes(minivit):
    engine = Engine(minivit)
    before = dict(engine.version)
    engine.on_edit(edit("set_param", instance="01J9I103", path="bias", value=False))
    advanced = {node for node, version in engine.version.items() if version > before[node]}
    assert advanced == {"01J9Q4B5", "01J9Q4B6"}


def test_error_blocks_the_downstream(minivit):
    """error의 하류는 실행되지 않았으므로 blocked다(§5.3)."""
    engine = Engine(minivit)
    changed = engine.absorb([], {"kind": "shape", "node_id": "01J9Q4B4", "message": "boom"})
    assert changed["01J9Q4B4"]["state"] == ERROR
    assert engine.state[("01J9Q4B5", "L0")] == BLOCKED
    assert engine.state[("01J9Q4B6", "L0")] == BLOCKED


def test_absorb_keys_results_by_call_path(minivit):
    """경로가 다른 같은 노드 id는 서로 다른 결과로 남아야 한다."""
    from torchflow.kernel.l0 import NodeReport

    engine = Engine(minivit)
    changed = engine.absorb(
        [
            NodeReport("01J9Q4A1", "norm1", {"shape": ["B", 64, 192]}, path="01J9Q4B3#0"),
            NodeReport("01J9Q4A1", "norm1", {"shape": ["B", 64, 384]}, path="01J9Q4B3#5"),
        ],
        None,
    )
    assert set(changed) == {"01J9Q4B3#0/01J9Q4A1", "01J9Q4B3#5/01J9Q4A1"}
    assert changed["01J9Q4B3#0/01J9Q4A1"]["spec"]["shape"] == ["B", 64, 192]
    assert changed["01J9Q4B3#5/01J9Q4A1"]["node"] == "01J9Q4A1"


def test_absorb_marks_reported_nodes_ok(minivit):
    from torchflow.kernel.l0 import NodeReport

    engine = Engine(minivit)
    changed = engine.absorb(
        [NodeReport("01J9Q4B5", "head", {"shape": ["B", 10]}, elapsed_ms=1.5, cache_hit=True)],
        None,
    )
    assert changed["01J9Q4B5"]["state"] == OK
    assert changed["01J9Q4B5"]["badges"]["cache_hit"] is True


def test_snapshot_covers_every_node_and_axis(minivit):
    engine = Engine(minivit)
    snapshot = engine.snapshot()
    assert len(snapshot) == len(engine.index.all_nodes())
    assert set(next(iter(snapshot.values()))) == {"L0", "L1fwd", "L1bwd"}
