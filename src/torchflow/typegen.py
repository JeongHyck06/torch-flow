"""Pydantic -> TypeScript 타입 생성 (기획서 §9: "Pydantic -> TS 타입 생성").

프로토콜과 IR의 진실원은 Python 모델 하나뿐이다. 프론트가 손으로 옮겨 적은
타입을 들고 있으면 메시지를 하나 고칠 때마다 두 곳이 어긋난다.

    torchflow typegen --out web/src/types.gen.ts
"""

from __future__ import annotations

import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from . import ir, protocol

SCALARS = {str: "string", int: "number", float: "number", bool: "boolean", type(None): "null"}

# 내보낼 모델. 순서가 곧 파일 안의 선언 순서다.
IR_MODELS = [ir.Port, ir.HParam, ir.Instance, ir.Node, ir.Probe, ir.CodeCell, ir.Composite,
             ir.Graph, ir.ModuleGraph]
PROTOCOL_MODELS = [protocol.NodeRequest, protocol.TensorSpec, protocol.Op, protocol.OpAck,
                   protocol.OpBroadcast, protocol.NodeState, protocol.KernelStatus,
                   protocol.Resync, protocol.Error, protocol.Log, protocol.MemoryEstimate]

UNIONS = {
    "ToBrowser": [protocol.OpAck, protocol.OpBroadcast, protocol.NodeState, protocol.KernelStatus,
                  protocol.Resync, protocol.Error, protocol.Log],
}


def _atom(rendered: str) -> str:
    """유니온을 배열 원소로 쓰려면 괄호가 필요하다: ``(string | number)[]``."""
    return f"({rendered})" if " | " in rendered else rendered


def ts_type(annotation: Any) -> str:
    if annotation is Any or annotation is None:
        return "unknown"
    if annotation in SCALARS:
        return SCALARS[annotation]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return " | ".join(f'"{value}"' if isinstance(value, str) else str(value) for value in args)
    if origin in (Union, types.UnionType):
        return " | ".join(dict.fromkeys(ts_type(arg) for arg in args))
    if origin in (list, typing.List):
        return f"{_atom(ts_type(args[0]))}[]" if args else "unknown[]"
    if origin in (tuple, typing.Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return f"{_atom(ts_type(args[0]))}[]"
        return "[" + ", ".join(ts_type(arg) for arg in args) + "]"
    if origin in (dict, typing.Dict):
        return f"Record<{ts_type(args[0])}, {ts_type(args[1])}>" if args else "Record<string, unknown>"
    return "unknown"


def render(model: type[BaseModel]) -> str:
    lines = [f"export interface {model.__name__} {{"]
    for name, field in model.model_fields.items():
        key = field.alias or name
        rendered = ts_type(field.annotation)
        # None을 포함하는 필드는 서버가 exclude_none으로 지우므로 선택 필드로 낸다.
        optional = "null" in rendered.split(" | ") or not field.is_required()
        rendered = " | ".join(part for part in rendered.split(" | ") if part != "null") or "unknown"
        lines.append(f"  {key}{'?' if optional else ''}: {rendered};")
    if model.model_config.get("extra") == "allow":
        lines.append("  [key: string]: unknown;")
    lines.append("}")
    return "\n".join(lines)


def generate() -> str:
    parts = [
        "// 생성 파일 - 손으로 고치지 말 것.",
        "// torchflow typegen (src/torchflow/typegen.py) 이 Pydantic 모델에서 뽑아낸다.",
        "",
        "// IR (기획서 §17.1)",
        *(render(model) for model in IR_MODELS),
        "",
        "// 프로토콜 (기획서 §8.2)",
        *(render(model) for model in PROTOCOL_MODELS),
        "",
        *(f"export type {name} = {' | '.join(m.__name__ for m in models)};"
          for name, models in UNIONS.items()),
        "",
    ]
    return "\n\n".join(parts).replace("\n\n\n", "\n\n")
