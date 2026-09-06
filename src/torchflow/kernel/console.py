"""Debug Console - 선택 노드의 마지막 probe 값을 표현식으로 조회한다 (기획서 §5.6.1).

L1 커널의 마지막 probe가 남긴 살아 있는 텐서에 이름 넷을 붙인다.

* ``x``     선택 노드의 입력. 포트가 하나면 텐서, 여럿이면 포트 이름 dict.
* ``y``     선택 노드의 출력 텐서.
* ``m``     그 노드의 모듈 인스턴스.
* ``batch`` 이번 probe가 그래프 입력에 넣은 텐서들.

Phase A는 **읽기 전용 표현식**이다(문장 실행은 v1). 표현식은 그래프 파일이
아니라 이 브라우저 앞에 앉은 사람이 직접 친 것이고, 같은 사람이 Code Cell로
임의 코드를 돌릴 수 있는 커널 프로세스 안에서 평가된다 - 그래서 ``$expr``
(:mod:`torchflow.expr`)처럼 AST를 다시 해석하지 않고 ``eval``에 맡긴다.
``ponytail: eval + 문장 금지. 샌드박스가 필요해지는 건 원격 공유 세션(v2)부터다.``
"""

from __future__ import annotations

import ast
from typing import Any

# 텐서를 통째로 repr하면 콘솔이 만 줄이 된다. 값 대신 형태와 요약을 낸다.
PREVIEW_ELEMENTS = 8


def bindings(pass_: Any, key: str) -> dict[str, Any]:
    """마지막 L1 패스에서 노드 하나의 ``x``/``y``/``m``/``batch``를 꺼낸다."""
    inputs = pass_.node_inputs.get(key) or {}
    module_key = pass_.node_modules.get(key)
    return {
        "x": next(iter(inputs.values())) if len(inputs) == 1 else inputs,
        "y": pass_.activations.get(key),
        "m": pass_.session.modules.get(module_key) if module_key else None,
        "batch": pass_.graph_inputs,
        "torch": pass_.torch,
    }


def evaluate(pass_: Any, key: str, source: str) -> dict[str, Any]:
    """표현식 하나를 평가하고 렌더링한 결과를 돌려준다."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        # 문장을 친 경우도 여기로 온다 - Phase A의 경계를 그대로 말해 준다.
        return {"ok": False, "error": f"표현식이 아닙니다 ({exc.msg})"}

    env = bindings(pass_, key)
    try:
        value = eval(compile(tree, "<console>", "eval"), {"__builtins__": __builtins__}, env)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **render(pass_.torch, value)}


def render(torch, value: Any) -> dict[str, Any]:
    """Inspector가 그대로 찍을 한 줄. 텐서는 데이터 렌즈 요약으로 축약한다."""
    if torch.is_tensor(value):
        detached = value.detach()
        spec = {"shape": list(detached.shape),
                "dtype": str(detached.dtype).removeprefix("torch."),
                "device": str(detached.device)}
        flat = detached.flatten().float()
        finite = flat[torch.isfinite(flat)]
        summary = (
            f"min {finite.min():.4g}  max {finite.max():.4g}  mean {finite.mean():.4g}"
            if finite.numel() else "all non-finite"
        )
        nonfinite = int(flat.numel() - finite.numel())
        if nonfinite:
            summary += f"  nonfinite {nonfinite}"
        head = ", ".join(f"{item:.4g}" for item in flat[:PREVIEW_ELEMENTS].tolist())
        if flat.numel() > PREVIEW_ELEMENTS:
            head += ", …"
        return {"text": f"{summary}\n[{head}]", "spec": spec}

    if value is None:
        return {"text": "None", "spec": None}
    text = repr(value)
    return {"text": text if len(text) <= 2000 else text[:2000] + " …", "spec": None}
