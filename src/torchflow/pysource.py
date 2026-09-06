"""사용자 .py를 실행하지 않고 읽는다.

hub는 torch도, 사용자 코드도 실행하지 않는다. 드롭된 파일에서 어떤 클래스를
인스턴스화할 수 있는지 고르려면 소스를 읽어야 하는데, 그 읽기는 `ast`로
충분하다. 실행은 커널의 몫이다.
"""

from __future__ import annotations

import ast
from typing import Any


def _annotation(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _default(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return ast.unparse(node)


def candidates(source: str) -> list[dict[str, Any]]:
    """``nn.Module`` 후보 클래스와 그 생성 인자를 뽑는다.

    상속을 이름으로만 판단한다. import를 따라가지 않으므로 별칭을 쓴 경우
    놓칠 수 있고, 그때는 후보에 안 뜬다. 잘못 실행하는 것보다 낫다.
    """
    tree = ast.parse(source)
    found: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(base) for base in node.bases}
        if not any(base.endswith("Module") or base.endswith("Sequential") for base in bases):
            continue

        init = next((item for item in node.body
                     if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
        params: list[dict[str, Any]] = []
        if init is not None:
            args = init.args
            padding = [None] * (len(args.args) - len(args.defaults))
            for argument, default in zip(args.args, padding + list(args.defaults)):
                if argument.arg == "self":
                    continue
                params.append({
                    "name": argument.arg,
                    "annotation": _annotation(argument.annotation),
                    "default": _default(default),
                })

        found.append({
            "name": node.name,
            "bases": sorted(bases),
            "params": params,
            "suggestion": _suggestion(node.name, params),
            "doc": (ast.get_docstring(node) or "").split("\n")[0][:120],
        })
    return found


def _suggestion(name: str, params: list[dict[str, Any]]) -> str:
    """``DiT(depth=28, hidden_size=1152)`` 꼴의 시작 문자열."""
    filled = [f"{param['name']}={param['default']!r}"
              for param in params if param["default"] is not None]
    required = [f"{param['name']}=" for param in params if param["default"] is None]
    return f"{name}({', '.join(required + filled)})"


DTYPE_ALIASES = {
    "f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64",
    "i64": "int64", "i32": "int32", "b": "bool", "l": "int64",
}


def parse_example_spec(text: str) -> dict[str, dict[str, Any]]:
    """``x=B,3,32,32:f32 y=B:i64`` 를 포트 명세로 바꾼다.

    ``B``는 배치 심볼이다. 정수가 아닌 항목은 심볼로 남는다.
    """
    spec: dict[str, dict[str, Any]] = {}
    for chunk in text.split():
        name, sep, rest = chunk.partition("=")
        if not sep:
            raise ValueError(f"expected name=shape, got {chunk!r}")
        shape_text, _, dtype_text = rest.partition(":")
        dims: list[str | int] = []
        for dim in shape_text.split(","):
            dim = dim.strip()
            if not dim:
                continue
            dims.append(int(dim) if dim.lstrip("-").isdigit() else dim)
        dtype = DTYPE_ALIASES.get(dtype_text.strip(), dtype_text.strip() or "float32")
        spec[name.strip()] = {"shape": dims, "dtype": dtype}
    if not spec:
        raise ValueError("example input spec is empty")
    return spec
