"""SchemaExtractor: 블록 레지스트리 생성 (기획서 §8.1, 카탈로그 §17.3).

리플렉션은 L0 커널에서만 돈다. hub는 ``registry.json``을 읽기만 한다.
hub가 torch를 import하지 않는다는 불변식(§8.1)이 여기서 지켜진다.

Phase A 등재분만 담는다: ``torch.nn.*`` 리플렉션(R) + 내장 텐서 연산·구조
노드(I). 컴포지트(K)와 플러그인(P)은 각자의 마일스톤에서 붙는다.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

# 카테고리별 nn 클래스. §17.3에서 단계 A · 등재 R인 행 전부.
NN_BLOCKS: dict[str, list[str]] = {
    "기본 레이어": ["Linear", "Embedding", "Dropout", "Flatten", "Identity"],
    "정규화": ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "RMSNorm", "GroupNorm"],
    "활성화": ["ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax"],
    "컨볼루션": ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d"],
    "풀링/리샘플링": [
        "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d",
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "Upsample",
    ],
    "어텐션/트랜스포머": ["MultiheadAttention", "TransformerEncoderLayer"],
    "순환/SSM": ["LSTM", "GRU"],
    "손실": ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "L1Loss"],
}

# 내장 블록(I): 리플렉션 대상이 아니라 실행기가 직접 구현한다.
BUILTIN_BLOCKS: list[dict[str, Any]] = [
    {"type": "torchflow.Input", "category": "구조/제어", "params": {}, "ports": {"in": [], "out": ["x"]}},
    {"type": "torchflow.Output", "category": "구조/제어", "params": {}, "ports": {"in": ["input"], "out": ["output"]}},
    # 학습 블록. 모델 코드에는 들어가지 않고 Run이 이 값으로 워커 job을 만든다. 로짓을 받는 입력
    # 포트는 데이터 -> 모델 -> 학습이 캔버스에서 한 줄로 읽히게 하려는 것이다.
    {"type": "torchflow.Train", "category": "학습 제어",
     "params": {"optimizer": {"type": "str", "default": "adamw", "choices": ["adamw", "sgd"]},
                "lr": {"type": "float", "default": 0.001},
                "weight_decay": {"type": "float", "default": 0.0},
                "steps": {"type": "int", "default": 500},
                "batch": {"type": "int", "default": 32},
                "scheduler": {"type": "str", "default": "none", "choices": ["none", "cosine"]},
                "warmup_steps": {"type": "int", "default": 0}},
     "ports": {"in": ["input"], "out": []}},
    # Repeat와 Switch는 노드가 아니라 인스턴스 종류다(§4.4) - 본문 컴포지트 없이는 놓을 수 없어
    # 팔레트에 두면 "module 'torchflow' has no attribute 'Repeat'"로 끝난다. 컴포지트 승격과 함께 온다.
    {"type": "torch.add", "category": "텐서 연산", "params": {}, "ports": {"in": ["input", "other"], "out": ["output"]}},
    {"type": "torch.sub", "category": "텐서 연산", "params": {}, "ports": {"in": ["input", "other"], "out": ["output"]}},
    {"type": "torch.mul", "category": "텐서 연산", "params": {}, "ports": {"in": ["input", "other"], "out": ["output"]}},
    {"type": "torch.div", "category": "텐서 연산", "params": {}, "ports": {"in": ["input", "other"], "out": ["output"]}},
    {"type": "torch.cat", "category": "텐서 연산", "params": {"dim": {"type": "int", "default": 0}}, "ports": {"in": ["tensors"], "out": ["output"]}},
    {"type": "torch.mean", "category": "텐서 연산", "params": {"dim": {"type": "int", "default": None}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.sum", "category": "텐서 연산", "params": {"dim": {"type": "int", "default": None}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.reshape", "category": "텐서 연산", "params": {"shape": {"type": "str", "default": None}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.permute", "category": "텐서 연산", "params": {"dims": {"type": "str", "default": None}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.transpose", "category": "텐서 연산", "params": {"dim0": {"type": "int", "default": 0}, "dim1": {"type": "int", "default": 1}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.flatten", "category": "텐서 연산", "params": {"start_dim": {"type": "int", "default": 1}}, "ports": {"in": ["input"], "out": ["output"]}},
    {"type": "torch.detach", "category": "텐서 연산", "params": {}, "ports": {"in": ["input"], "out": ["output"]}},
]

_SCALAR_TYPES = {int: "int", float: "float", bool: "bool", str: "str"}


def _param_schema(param: inspect.Parameter) -> dict[str, Any]:
    """``__init__`` 인자 하나를 팔레트 폼이 쓸 스키마로 옮긴다."""
    default = param.default if param.default is not inspect.Parameter.empty else None
    kind = _SCALAR_TYPES.get(type(default))
    if kind is None:
        annotation = param.annotation
        kind = _SCALAR_TYPES.get(annotation, "any") if annotation is not inspect.Parameter.empty else "any"
    schema: dict[str, Any] = {"type": kind, "required": param.default is inspect.Parameter.empty}
    if default is not None and isinstance(default, (int, float, bool, str)):
        schema["default"] = default
    return schema


def extract(torch_module=None) -> dict[str, Any]:
    """``registry.json`` 본문을 만든다."""
    import torch

    torch_module = torch_module or torch
    version = torch_module.__version__.split("+")[0]
    blocks: list[dict[str, Any]] = []

    for category, names in NN_BLOCKS.items():
        for name in names:
            cls = getattr(torch_module.nn, name, None)
            if cls is None:
                continue  # torch 버전에 따라 없는 블록(RMSNorm 등)은 조용히 건너뛴다.
            try:
                signature = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                continue
            params = {
                pname: _param_schema(param)
                for pname, param in signature.parameters.items()
                if pname not in ("self", "device", "dtype") and param.kind is not param.VAR_KEYWORD
            }
            blocks.append(
                {
                    "type": f"torch.nn.{name}@{version}",
                    "label": name,
                    "category": category,
                    "params": params,
                    "ports": {"in": ["input"], "out": ["output"]},
                    "doc": (inspect.getdoc(cls) or "").split("\n\n")[0][:200],
                    "source": "reflection",
                }
            )

    for builtin in BUILTIN_BLOCKS:
        blocks.append({**builtin, "label": builtin["type"].rsplit(".", 1)[-1], "source": "builtin"})

    blocks.sort(key=lambda b: (b["category"], b["type"]))
    return {"torch_version": version, "blocks": blocks}


def write(path: str | Path, torch_module=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(extract(torch_module), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
