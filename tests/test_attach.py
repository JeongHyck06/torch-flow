"""Attach Mode (기획서 §3.5, §17.7)."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from torchflow.attach.session import Session  # noqa: E402
from torchflow.attach.trace import trace_module, trace_report  # noqa: E402
from torchflow.ir import validate  # noqa: E402


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 함수형 덧셈 - hook에 걸리지 않는다. 트레이서의 정직한 한계 지점.
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)


class Net(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(8) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Net()


@pytest.fixture
def example():
    return torch.randn(4, 3, 16, 16)


@pytest.fixture
def session(model, example, tmp_path):
    session = Session(model, trace_module(model, example), state_dir=tmp_path,
                      url="http://127.0.0.1:1/", token="t",
                      objective=lambda out, _batch: out.float().pow(2).mean())
    session._example = example
    return session


# 인스턴스 import (§7.4 경로 3)


def test_trace_produces_a_valid_ir(model, example):
    ir = trace_module(model, example)
    assert validate(ir) == []
    assert ir.graph.name == "Net"
    assert ir.meta["source"] == "attach"


def test_trace_records_measured_shapes(model, example):
    ir = trace_module(model, example)
    stem = next(node for node in ir.graph.nodes if node.label == "stem")
    assert stem.measured["shape"] == [4, 8, 16, 16]


def test_trace_shares_one_instance_per_module_object(model, example):
    """``self.relu``를 두 번 부르면 호출 노드는 둘, 인스턴스는 하나다(§7.1)."""
    ir = trace_module(model, example)
    calls = [node for node in ir.graph.nodes if node.call]
    assert len(calls) > len(ir.graph.instances)

    by_instance: dict[str, list[str]] = {}
    for node in calls:
        by_instance.setdefault(node.call, []).append(node.label)
    shared = {key: labels for key, labels in by_instance.items() if len(labels) > 1}
    assert shared, "재사용된 ReLU가 공유 인스턴스로 묶이지 않았다"
    assert all(len(set(labels)) == 1 for labels in shared.values())


def test_trace_report_admits_what_it_could_not_link(model, example):
    """모듈 사이의 함수형 연산은 hook에 안 걸린다 - 없는 엣지를 지어내지 않는다."""
    report = trace_report(trace_module(model, example))
    assert 0 < report["linked_ratio"] < 1
    assert report["unlinked"], "함수형 residual이 있는데 끊긴 노드가 없다고 보고했다"
    assert report["params"] == sum(p.numel() for p in Net().parameters())


def test_trace_leaves_no_hooks_behind(model, example):
    trace_module(model, example)
    assert all(not module._forward_hooks for module in model.modules())


def test_trace_accepts_keyword_and_positional_inputs(model):
    single = trace_module(model, torch.randn(2, 3, 16, 16))
    keyword = trace_module(model, {"x": torch.randn(2, 3, 16, 16)})
    positional = trace_module(model, (torch.randn(2, 3, 16, 16),))
    assert len(single.graph.nodes) == len(keyword.graph.nodes) == len(positional.graph.nodes)


# probe 계약


def test_probe_reports_gradients_against_the_user_objective(session):
    result = session.probe()
    graded = [node for node in result["nodes"] if node["grad_norm"] is not None]
    assert result["loss"] > 0
    assert result["objective"] == "user objective"
    assert len(graded) >= 8


def test_probe_leaves_the_model_untouched(session, model):
    """P2: 프로브는 연산 경로도, 상태도 바꾸지 않는다."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(session._example).float().sum()
    loss.backward()

    weights = [param.detach().clone() for param in model.parameters()]
    grads = [param.grad.clone() for param in model.parameters()]
    buffers = [buffer.clone() for _, buffer in model.named_buffers()
               if buffer.is_floating_point()]

    session.probe()

    assert model.training, "probe가 train 모드를 되돌리지 않았다"
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters()))
    assert all(torch.equal(a, b.grad) for a, b in zip(grads, model.parameters()))
    assert all(torch.equal(a, b) for a, b in
               zip(buffers, (b for _, b in model.named_buffers() if b.is_floating_point())))
    optimizer.zero_grad()


def test_probe_does_not_disturb_convergence(session, model):
    """probe를 사이에 끼워도 학습 궤적이 같아야 한다."""
    import copy

    def train(steps, probe_at):
        torch.manual_seed(1)
        local = copy.deepcopy(model)
        session.model = local
        optimizer = torch.optim.SGD(local.parameters(), lr=0.05)
        target = torch.randint(0, 10, (4,))
        losses = []
        for step in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(local(session._example), target)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 6))
            if step in probe_at:
                session.probe()
        return losses

    assert train(5, set()) == train(5, {1, 3})


def test_probe_forward_only_skips_backward(session):
    result = session.probe(backward=False)
    assert result["loss"] is None
    assert all(node["grad_norm"] in (None, 0.0) for node in result["nodes"])
    assert any(node["spec"] for node in result["nodes"])


def test_probe_collects_histograms_for_leaf_activations(session):
    result = session.probe()
    with_hist = [node for node in result["nodes"] if node["histogram"]]
    assert with_hist
    assert all(sum(node["histogram"]) > 0 for node in with_hist)


def test_probe_maps_results_onto_graph_nodes(session):
    """probe 결과가 캔버스 노드 id로 돌아와야 그래프 위에 얹힌다."""
    graph_ids = {node.id for node in session.ir.graph.nodes}
    reported = {node["node"] for node in session.probe()["nodes"]}
    assert reported & graph_ids, "어떤 결과도 그래프 노드에 매핑되지 않았다"


def test_probe_needs_an_example_input(model, tmp_path, example):
    session = Session(model, trace_module(model, example), state_dir=tmp_path,
                      url="http://127.0.0.1:1/", token="t")
    with pytest.raises(ValueError, match="no example input"):
        session.probe()


def test_log_buffers_scalars(session):
    session.log(0, loss=1.5, lr=1e-3)
    session.log(1, loss=1.2, lr=1e-3)
    assert [record["loss"] for record in session.scalars] == [1.5, 1.2]
    assert all("wall" in record for record in session.scalars)


def test_hub_push_failures_do_not_break_training(session):
    """hub가 죽어도 사용자 루프는 멈추지 않는다 - 관찰층이 학습을 인질로 잡지 않는다."""
    session.url = "http://127.0.0.1:9/"   # 아무도 듣지 않는 포트
    session.log(0, loss=1.0)
    session.probe()
