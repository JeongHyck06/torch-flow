"""``$expr`` 평가기 (기획서 §4.3).

허용: 산술·비교·삼항·``i``·``hp.*``·``p.*``·``rt.*``.
``eval``은 쓰지 않는다. 그래프 파일은 신뢰 경계 밖(git clone, 동료가 보낸
``.tfg.json``)에서 오므로 표현식은 실행이 아니라 해석 대상이다.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

NAMESPACES = ("hp", "p", "rt")

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class ExprError(ValueError):
    """표현식이 문법을 벗어났거나 이름을 해소하지 못했다."""


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)) or node.value is None:
            return node.value
        raise ExprError(f"unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id == "i":
            if "i" not in env:
                raise ExprError("'i' used outside a Repeat body")
            return env["i"]
        raise ExprError(f"unknown name {node.id!r}")

    if isinstance(node, ast.Attribute):
        base = node.value
        if not isinstance(base, ast.Name) or base.id not in NAMESPACES:
            raise ExprError("attribute access is limited to hp.* / p.* / rt.*")
        scope = env.get(base.id)
        if scope is None or node.attr not in scope:
            raise ExprError(f"unresolved {base.id}.{node.attr}")
        return scope[node.attr]

    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ExprError(f"unsupported operator: {type(node.op).__name__}")
        return fn(_eval(node.left, env), _eval(node.right, env))

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +_eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not _eval(node.operand, env)
        raise ExprError(f"unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _CMPOPS.get(type(op))
            if fn is None:
                raise ExprError(f"unsupported comparison: {type(op).__name__}")
            right = _eval(comparator, env)
            if not fn(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)

    raise ExprError(f"unsupported syntax: {type(node).__name__}")


def evaluate(source: str, *, hp=None, p=None, rt=None, i=None) -> Any:
    """``$expr`` 문자열을 평가한다."""
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot parse expression {source!r}: {exc.msg}") from exc
    env: dict[str, Any] = {"hp": hp or {}, "p": p or {}, "rt": rt or {}}
    if i is not None:
        env["i"] = i
    return _eval(tree, env)


def resolve(value: Any, *, hp=None, p=None, rt=None, i=None) -> Any:
    """참조(``$hp``/``$p``/``$expr``)를 실제 값으로 바꾼다. 컨테이너는 재귀."""
    from .ir import ref_of

    ref = ref_of(value)
    if ref is not None:
        kind, inner = ref
        if kind == "$expr":
            return evaluate(inner, hp=hp, p=p, rt=rt, i=i)
        scope = (hp or {}) if kind == "$hp" else (p or {})
        if inner not in scope:
            raise ExprError(f"unresolved {kind} {inner!r}")
        return scope[inner]
    if isinstance(value, dict):
        return {k: resolve(v, hp=hp, p=p, rt=rt, i=i) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, hp=hp, p=p, rt=rt, i=i) for v in value]
    return value
