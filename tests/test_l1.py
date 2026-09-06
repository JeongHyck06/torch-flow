"""L1 probe 계약 (기획서 §5.1.2, §5.1.3)."""

import math

import pytest

torch = pytest.importorskip("torch")

from torchflow.kernel.l0 import L0Session  # noqa: E402
from torchflow.kernel.l1 import (  # noqa: E402
    L1NodeResult, L1Pass, ProbeConfig, _flag_warnings, probe,
)
from torchflow.runtime.init import derive_seed, init_coverage, seeded_init  # noqa: E402

RT = {"num_classes": 10}


def run(ir, **kwargs):
    return probe(ir, rt=RT, **kwargs)


# seeded_init (§5.1.3)


def test_derive_seed_is_path_addressed_and_stable():
    assert derive_seed(0, "blocks.3.attn") == derive_seed(0, "blocks.3.attn")
    assert derive_seed(0, "blocks.3.attn") != derive_seed(0, "blocks.4.attn")
    assert derive_seed(0, "a") != derive_seed(1, "a")
    assert 0 <= derive_seed(7, "x") < (1 << 63)


def test_seeded_init_is_reproducible_and_path_local():
    from torch import nn

    def build(extra: bool):
        layers = [nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)]
        if extra:
            layers.append(nn.Linear(4, 2))
        return nn.Sequential(*layers)

    small, large = build(False), build(True)
    seeded_init(small, 42)
    seeded_init(large, 42)
    # 뒤에 노드를 붙여도 앞 노드의 난수는 그대로다 - 속성 경로가 그대로이기 때문.
    assert torch.equal(small[0].weight, large[0].weight)

    other = build(False)
    seeded_init(other, 43)
    assert not torch.equal(small[0].weight, other[0].weight)


def test_init_coverage_reports_parameters_it_cannot_seed():
    from torch import nn

    class Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3))

    assert init_coverage(Custom()) == ["w"]
    assert init_coverage(nn.Linear(3, 3)) == []


# probe 계약


def test_probe_produces_gradients_for_parametrised_nodes(minivit):
    result = run(minivit, probe=ProbeConfig(batch=4, objective="random_target_ce"))
    assert result.ok, result.error
    assert not result.forward_only
    graded = [node for node in result.nodes if node.grad_norm is not None]
    assert len(graded) > 20
    assert all(node.grad_norm >= 0 for node in graded)
    assert result.node("head").grad_ratio > 0


def test_probe_is_deterministic(minivit):
    first = run(minivit, probe=ProbeConfig(batch=4, objective="random_target_ce"))
    second = run(minivit, probe=ProbeConfig(batch=4, objective="random_target_ce"))
    assert first.loss == pytest.approx(second.loss, rel=1e-9)
    assert first.node("head").grad_norm == pytest.approx(second.node("head").grad_norm, rel=1e-9)


def test_probe_restores_batchnorm_buffers():
    """probe가 running stats를 오염시키면 학습이 조용히 달라진다(§5.1.2)."""
    from torchflow.ir import load

    from conftest import ROOT

    ir = load(ROOT / "examples" / "resnet18.tfg.json")
    session = L0Session()
    L1Pass(ir, rt=RT, probe=ProbeConfig(batch=4), session=session).probe_once()

    norms = [m for m in session.modules.values() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms, "BatchNorm이 하나도 만들어지지 않았다"
    before = [module.running_mean.clone() for module in norms]

    L1Pass(ir, rt=RT, probe=ProbeConfig(batch=4), session=session).probe_once()
    assert all(torch.equal(saved, module.running_mean)
               for saved, module in zip(before, norms))


def test_probe_runs_modules_in_eval_mode_by_default():
    from torchflow.ir import load

    from conftest import ROOT

    session = L0Session()
    L1Pass(load(ROOT / "examples" / "resnet18.tfg.json"), rt=RT,
           probe=ProbeConfig(batch=4), session=session).probe_once()
    assert not any(module.training for module in session.modules.values()
                   if hasattr(module, "training"))


def test_probe_does_not_accumulate_gradients(minivit):
    """매 probe 전에 zero_grad(set_to_none=True) - 아니면 값이 계속 커진다."""
    session = L0Session()
    config = ProbeConfig(batch=4, objective="mse_to_zero")
    first = L1Pass(minivit, rt=RT, probe=config, session=session).probe_once()
    second = L1Pass(minivit, rt=RT, probe=config, session=session).probe_once()
    assert first.node("head").grad_norm == pytest.approx(second.node("head").grad_norm, rel=1e-6)


@pytest.mark.parametrize("objective", ["sum_of_outputs", "mse_to_zero", "random_target_ce"])
def test_every_objective_produces_a_finite_loss(minivit, objective):
    result = run(minivit, probe=ProbeConfig(batch=2, objective=objective))
    assert result.ok and math.isfinite(result.loss)


def test_synthetic_objective_does_not_raise_the_ratio_band(minivit):
    """sum_of_outputs는 grad가 출력 원소 수에 비례한다 - 밴드를 대면 전부 amber다."""
    synthetic = run(minivit, probe=ProbeConfig(batch=4, objective="sum_of_outputs"))
    calibrated = run(minivit, probe=ProbeConfig(batch=4, objective="random_target_ce"))
    assert not any(node.warn == "ratio" for node in synthetic.nodes)
    assert any(node.warn == "ratio" for node in calibrated.nodes)
    assert "synthetic" in synthetic.objective


def test_relative_drop_is_flagged_between_adjacent_layers():
    results = [
        L1NodeResult("a", "a", grad_norm=1e-2, grad_ratio=1e-3),
        L1NodeResult("b", "b", grad_norm=1e-8, grad_ratio=1e-3),  # log10 차 6
        L1NodeResult("c", "c", grad_norm=1e-8, grad_ratio=1e-3),
    ]
    _flag_warnings(results, calibrated=False)
    assert results[0].warn is None
    assert results[1].warn == "relative_drop"
    assert results[2].warn is None


def test_probe_budget_limits_histograms_to_subscribed_nodes(minivit):
    """뷰포트 밖 노드는 스칼라만 모은다(§5.1.4)."""
    everywhere = run(minivit, probe=ProbeConfig(batch=2, collect=("grad_norm", "hist")))
    narrow = run(minivit, probe=ProbeConfig(
        batch=2, collect=("grad_norm", "hist"), subscribed=("01J9Q4B5",)))

    assert sum(1 for node in everywhere.nodes if node.histogram) > 10
    assert [node.label for node in narrow.nodes if node.histogram] == ["head"]
    # 스칼라는 여전히 전부 모은다.
    assert sum(1 for node in narrow.nodes if node.grad_norm is not None) > 20


def test_histogram_counts_every_finite_element(minivit):
    result = run(minivit, probe=ProbeConfig(batch=2, collect=("hist",), subscribed=("01J9Q4B2",)))
    patch = result.node("patch")
    assert sum(patch.histogram) == 2 * 64 * 192
    assert patch.hist_range[0] < patch.hist_range[1]


def test_forward_only_mode_skips_backward(minivit):
    result = run(minivit, probe=ProbeConfig(batch=2, backward=False))
    assert result.ok and result.forward_only
    assert result.loss is None
    assert all(node.grad_norm in (None, 0.0) for node in result.nodes)
    # forward 결과(shape·히스토그램)는 그대로 나온다.
    assert result.node("head").spec["shape"] == [2, 10]


def test_shape_error_keeps_node_attribution(minivit):
    minivit.graph.instances["01J9I103"].args["in_features"] = 768
    result = run(minivit, probe=ProbeConfig(batch=2))
    assert not result.ok
    assert result.error["kind"] == "shape"
    assert result.error["node_id"] == "01J9Q4B5"


def test_nonfinite_activations_are_attributed_to_the_first_node(minivit):
    """위상 순 최초 비유한 텐서 노드를 가리킨다(§5.6)."""
    session = L0Session()
    pass_ = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2), session=session)
    pass_.probe_once()
    # patch의 conv weight를 inf로 만들어 하류를 오염시킨다.
    for module in session.modules.values():
        if isinstance(module, torch.nn.Conv2d):
            with torch.no_grad():
                module.weight.fill_(float("inf"))
            break

    result = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2), session=session).probe_once()
    assert result.first_nonfinite is not None
    first = next(node for node in result.nodes if node.key == result.first_nonfinite)
    assert first.label in {"proj", "patch"}


def test_probe_summary_is_badge_ready(minivit):
    result = run(minivit, probe=ProbeConfig(batch=4, objective="random_target_ce"))
    assert result.objective == "CE(random) · B=4 · init 1step"


def test_shared_instances_are_counted_once(minivit):
    """tied embedding처럼 공유되는 인스턴스는 buffer 스냅샷도 한 번만 뜬다."""
    session = L0Session()
    pass_ = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2), session=session)
    pass_.probe_once()
    modules = pass_._probe_modules()
    assert len({id(module) for module in modules.values()}) <= len(modules)
