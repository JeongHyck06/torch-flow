"""``$expr`` 평가기 (기획서 §4.3: 자체 파서, eval 금지)."""

import pytest

from torchflow.expr import ExprError, evaluate, resolve


@pytest.mark.parametrize(
    "source, env, expected",
    [
        ("hp.epochs * rt.steps_per_epoch", {"hp": {"epochs": 10}, "rt": {"steps_per_epoch": 391}}, 3910),
        ("p.dim // p.heads", {"p": {"dim": 384, "heads": 6}}, 64),
        ("p.dim * 2 if i > 0 else p.dim", {"p": {"dim": 8}, "i": 3}, 16),
        ("p.dim >= 8 and not p.frozen", {"p": {"dim": 8, "frozen": False}}, True),
    ],
)
def test_allowed_forms(source, env, expected):
    assert evaluate(source, **env) == expected


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "hp.dim.__class__.__mro__",
        "[x for x in (1, 2)]",
        "lambda: 1",
        "hp.dim; print(1)",
        "os.getcwd()",
    ],
)
def test_escapes_are_rejected(source):
    """그래프 파일은 신뢰 경계 밖에서 온다. 표현식은 실행되지 않는다."""
    with pytest.raises(ExprError):
        evaluate(source, hp={"dim": 8})


def test_unresolved_names_are_reported_not_silently_defaulted():
    with pytest.raises(ExprError, match="unresolved"):
        evaluate("hp.ghost + 1", hp={})
    with pytest.raises(ExprError, match="unresolved"):
        resolve({"$hp": "ghost"}, hp={})


def test_resolve_walks_containers():
    value = {"a": [{"$hp": "dim"}, 2], "b": {"$expr": "p.dim * 2"}}
    assert resolve(value, hp={"dim": 4}, p={"dim": 4}) == {"a": [4, 2], "b": 8}


def test_repeat_index_is_scoped():
    assert evaluate("i * 2", i=3) == 6
    with pytest.raises(ExprError, match="outside a Repeat"):
        evaluate("i * 2")
