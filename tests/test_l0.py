"""L0 정적 패스 (기획서 §5.1). 기준은 실제 PyTorch 모델이다."""

import pytest

torch = pytest.importorskip("torch")

from torchflow.kernel.l0 import run_pass  # noqa: E402

RT = {"num_classes": 10}


def reference_minivit(dim=192, heads=6, depth=6, patch=4, num_classes=10):
    """``examples/minivit.tfg.json``과 같은 모델을 손으로 쓴 것. 정답지."""
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        def forward(self, x):
            a = self.norm1(x)
            x = x + self.attn(a, a, a, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class MiniViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.blocks = nn.Sequential(*[Block() for _ in range(depth)])
            self.head = nn.Linear(dim, num_classes)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            return self.head(self.blocks(x).mean(dim=1))

    return MiniViT()


def test_shapes_match_a_real_forward(minivit):
    """L0가 낸 shape가 실제 forward의 shape와 일치한다."""
    result = run_pass(minivit, rt=RT)
    assert result.ok, result.error

    reference = reference_minivit()
    with torch.no_grad():
        logits = reference(torch.zeros(2, 3, 32, 32))

    assert result.spec("head")["shape"] == ["B", logits.shape[1]]
    assert result.spec("patch")["shape"] == ["B", 64, 192]
    assert result.spec("pool")["shape"] == ["B", 192]


def test_param_count_matches_the_reference_model(minivit):
    """할당 없이 센 파라미터 수가 진짜 모델과 정확히 같다."""
    expected = sum(p.numel() for p in reference_minivit().parameters())
    assert run_pass(minivit, rt=RT).total_params == expected


def test_batch_dimension_stays_symbolic(minivit):
    """B는 상수로 접히지 않는다 - 배치 크기를 바꿔도 그래프는 그대로다."""
    for spec in (r.spec for r in run_pass(minivit, rt=RT).nodes if r.spec):
        if len(spec["shape"]) > 1:
            assert spec["shape"][0] == "B"


def test_l0_allocates_nothing(minivit):
    """FakeTensorMode 정적 패스는 메모리를 잡지 않는다."""
    import tracemalloc

    tracemalloc.start()
    run_pass(minivit, rt=RT)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    params = sum(p.numel() for p in reference_minivit().parameters())
    assert peak < params * 4  # 실제 텐서였다면 최소 이만큼은 잡혔다.


def test_switch_selects_the_active_variant(minivit):
    """Ablation Switch가 hparam 하나로 갈린다(§4.4.2)."""
    baseline = run_pass(minivit, rt=RT)
    ablated = run_pass(minivit, hp={"attn_type": "none"}, rt=RT)

    assert baseline.ok and ablated.ok
    assert ablated.total_params < baseline.total_params
    # 어텐션만 빠지고 출력 shape는 같다 - 포트 시그니처가 일치하기 때문이다.
    assert ablated.spec("head") == baseline.spec("head")

    attn_params = 4 * 192 * 192 + 4 * 192  # in_proj + out_proj
    assert baseline.total_params - ablated.total_params == 6 * attn_params


def test_hparam_override_propagates_through_repeat(minivit):
    shallow = run_pass(minivit, hp={"depth": 2}, rt=RT)
    deep = run_pass(minivit, hp={"depth": 4}, rt=RT)
    assert shallow.ok and deep.ok
    assert len(deep.nodes) > len(shallow.nodes)
    assert deep.spec("head") == shallow.spec("head")


def test_repeat_shared_weights_counts_params_once(minivit):
    minivit.graph.instances["01J9I102"].mode = "shared_weights"
    shared = run_pass(minivit, rt=RT)
    minivit.graph.instances["01J9I102"].mode = "sequential"
    separate = run_pass(minivit, rt=RT)
    assert shared.ok and shared.total_params < separate.total_params


def test_shape_error_is_attributed_to_a_node(minivit):
    """실패는 노드에 귀속된다(§5.6) - 트레이스백이 아니라 노드 하나를 가리킨다."""
    minivit.graph.instances["01J9I103"].args["in_features"] = 768
    result = run_pass(minivit, rt=RT)

    assert not result.ok
    assert result.error["kind"] == "shape"
    assert result.error["node_id"] == "01J9Q4B5"  # head
    assert "768" in result.error["message"]
    # 실패 전 노드들의 결과는 남아 있다.
    assert result.spec("pool") == {"shape": ["B", 192], "dtype": "float32", "device": "cpu"}


def test_unresolved_hparam_is_attributed_to_a_node(minivit):
    minivit.graph.instances["01J9I103"].args["out_features"] = {"$expr": "rt.ghost"}
    result = run_pass(minivit, rt=RT)
    assert not result.ok and result.error["node_id"] == "01J9Q4B5"


def test_missing_runtime_constant_does_not_crash(minivit):
    result = run_pass(minivit)  # rt 없음 → rt.num_classes 미해소
    assert not result.ok and "num_classes" in result.error["message"]
