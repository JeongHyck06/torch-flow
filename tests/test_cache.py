"""L0 액션 캐시와 early cutoff (기획서 §5.2)."""

import pytest

pytest.importorskip("torch")

from torchflow.ir import load  # noqa: E402
from torchflow.kernel.l0 import L0Session, estimate_memory  # noqa: E402

from conftest import ROOT  # noqa: E402

RESNET = ROOT / "examples" / "resnet18.tfg.json"
RT = {"num_classes": 10}


def test_warm_run_reruns_almost_nothing(minivit):
    session = L0Session()
    session.run(minivit, rt=RT)
    warm = session.run(minivit, rt=RT)
    # KPI: 편집당 L0 재실행 비율 <15 %(§5.2). 편집이 없으면 Input 노드 하나뿐이다.
    assert warm.rerun_ratio < 0.05
    assert warm.elapsed_ms < session.run(minivit, rt=RT).elapsed_ms * 5


def test_warm_run_gives_identical_shapes(minivit):
    session = L0Session()
    cold = session.run(minivit, rt=RT)
    warm = session.run(minivit, rt=RT)
    assert [r.spec for r in cold.nodes] == [r.spec for r in warm.nodes]
    assert cold.total_params == warm.total_params


def test_local_edit_reruns_only_the_downstream(minivit):
    session = L0Session()
    session.run(minivit, rt=RT)
    minivit.graph.instances["01J9I103"].args["out_features"] = 100
    edited = session.run(minivit, rt=RT)
    assert edited.rerun_ratio < 0.15, "국소 편집이 그래프 전체를 재실행시켰다"
    assert edited.spec("head")["shape"] == ["B", 100]


def test_early_cutoff_stops_at_the_first_unchanged_output():
    """출력 spec이 그대로면 하류는 재실행되지 않는다 - early cutoff(§5.2)."""
    ir = load(RESNET)
    session = L0Session()
    session.run(ir, rt=RT)

    # 7x7 stride2 pad3 -> 5x5 stride2 pad2: 출력 shape가 정확히 같다.
    ir.graph.instances["01RG0001"].args.update(kernel_size=5, padding=2)
    cutoff = session.run(ir, rt=RT)
    assert cutoff.rerun_ratio < 0.05

    # stride를 바꾸면 하류 전부의 shape가 달라진다 - cutoff이 성립하지 않는다.
    ir.graph.instances["01RG0001"].args.update(kernel_size=7, padding=3, stride=1)
    propagated = session.run(ir, rt=RT)
    assert propagated.rerun_ratio > 0.8


def test_cache_survives_an_edit_and_its_undo(minivit):
    """undo는 이전 IR 상태로 돌아가므로 캐시 히트로 즉시 복원된다(§8.2.3)."""
    session = L0Session()
    session.run(minivit, rt=RT)
    original = minivit.graph.instances["01J9I103"].args["out_features"]

    minivit.graph.instances["01J9I103"].args["out_features"] = 100
    session.run(minivit, rt=RT)
    minivit.graph.instances["01J9I103"].args["out_features"] = original
    undone = session.run(minivit, rt=RT)

    assert undone.rerun_ratio < 0.05
    assert undone.spec("head")["shape"] == ["B", 10]


def test_invalidate_clears_everything(minivit):
    session = L0Session()
    session.run(minivit, rt=RT)
    session.invalidate()
    assert session.run(minivit, rt=RT).rerun_ratio > 0.9


def test_resnet18_matches_torchvision_param_count():
    """§13.3 템플릿 1번. 참조 구현과 파라미터 수가 정확히 같아야 한다."""
    result = L0Session().run(load(RESNET), rt={"num_classes": 1000})
    assert result.ok, result.error
    assert result.total_params == 11_689_512


def test_resnet18_shapes_follow_the_reference_downsampling():
    result = L0Session().run(load(RESNET), rt=RT)
    assert result.spec("conv1")["shape"] == ["B", 64, 112, 112]
    assert result.spec("maxpool")["shape"] == ["B", 64, 56, 56]
    assert result.spec("block8")["shape"] == ["B", 512, 7, 7]
    assert result.spec("fc")["shape"] == ["B", 10]


def test_memory_estimate_scales_with_batch(minivit):
    small = estimate_memory(minivit, batch=8, rt=RT)
    large = estimate_memory(minivit, batch=64, rt=RT)
    assert small["ok"] and large["ok"]
    activations = lambda m: m["breakdown_bytes"]["activations"]  # noqa: E731
    assert activations(large) == pytest.approx(activations(small) * 8, rel=0.05)
    # 파라미터·옵티마이저 상태는 배치와 무관하다.
    assert small["breakdown_bytes"]["params"] == large["breakdown_bytes"]["params"]


def test_memory_estimate_does_not_double_count_parameters(minivit):
    """conv는 입력 grad를 위해 weight를 저장한다 - 활성값으로 세면 이중 계상이다."""
    estimate = estimate_memory(minivit, batch=64, rt=RT)
    params = estimate["breakdown_bytes"]["params"]
    total = estimate["total_bytes"]
    assert total == sum(estimate["breakdown_bytes"].values())
    assert estimate["breakdown_bytes"]["grads"] == params


def test_optimizer_choice_changes_the_state_estimate(minivit):
    adam = estimate_memory(minivit, batch=8, optimizer="adam", rt=RT)
    sgd = estimate_memory(minivit, batch=8, optimizer="sgd", rt=RT)
    assert sgd["breakdown_bytes"]["optimizer_state"] == 0
    assert adam["breakdown_bytes"]["optimizer_state"] == 2 * adam["breakdown_bytes"]["params"]
    assert sgd["total_bytes"] < adam["total_bytes"]


def test_memory_estimate_reports_a_band_not_a_point(minivit):
    """cuDNN workspace·할당자 파편화는 밴드로 흡수한다(§6.2)."""
    estimate = estimate_memory(minivit, batch=64, rt=RT)
    low, high = estimate["band_gb"]
    point = estimate["total_bytes"] / 1024**3
    assert low <= point <= high
    assert high >= 0.5, "밴드 상한이 최소 slack(0.5 GB)보다 좁다"


def test_memory_estimate_reports_a_broken_graph_instead_of_raising(minivit):
    minivit.graph.instances["01J9I103"].args["in_features"] = 999
    estimate = estimate_memory(minivit, batch=8, rt=RT)
    assert estimate["ok"] is False and estimate["error"]["kind"] == "shape"


def test_repeated_composites_report_distinct_call_paths():
    """같은 컴포지트가 여러 번 인스턴스화되면 안쪽 노드 id는 같고 경로만 다르다.

    경로가 없으면 UI가 block1에 들어가도 block8의 shape를 보게 된다.
    """
    result = L0Session().run(load(RESNET), rt=RT)
    conv1 = {report.path: report.spec["shape"] for report in result.nodes
             if report.label == "conv1" and report.path}

    assert len(conv1) == 8, "BasicBlock ×8이 서로 다른 경로로 보고되지 않았다"
    assert sorted(conv1.values())[0] != sorted(conv1.values())[-1]
    # 첫 블록은 64채널 56×56, 마지막 블록은 512채널 7×7.
    assert ["B", 64, 56, 56] in conv1.values()
    assert ["B", 512, 7, 7] in conv1.values()


def test_report_key_disambiguates_shared_node_ids():
    result = L0Session().run(load(RESNET), rt=RT)
    keys = [report.key for report in result.nodes]
    assert len(keys) == len(set(keys)), "노드 리포트 키가 겹친다"
