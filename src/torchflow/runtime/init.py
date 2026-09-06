"""결정적 초기화 (기획서 §5.1.3).

torchflow 무의존. codegen이 이 파일을 그대로 ``utils.py``에 인라인한다.

시드 파생 키는 ULID가 아니라 속성 경로(``blocks.3.attn.q_proj``)다. 코드와
UI가 같은 문자열에서 같은 값을 계산하므로, 노드를 추가해도 다른 노드의 난수가
바뀌지 않는다 - 속성 경로가 바뀌지 않는 한.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn


def derive_seed(seed: int, path: str) -> int:
    """``(seed, 속성 경로)`` -> 63비트 시드. 경로가 같으면 언제나 같은 값."""
    digest = hashlib.blake2b(f"{seed}:{path}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def seeded_init(module: nn.Module, seed: int, prefix: str = "") -> None:
    """모든 leaf 모듈을 경로별 시드로 다시 초기화한다.

    ``reset_parameters``를 가진 모듈만 재초기화되며, 그렇지 않은 모듈의
    파라미터는 :func:`init_coverage`가 보고한다.
    """
    for name, child in module.named_modules():
        reset = getattr(child, "reset_parameters", None)
        if not callable(reset):
            continue
        path = f"{prefix}.{name}" if prefix and name else (prefix or name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed, path))
            reset()


def init_coverage(module: nn.Module) -> list[str]:
    """``seeded_init``이 건드리지 못한 파라미터의 경로 목록.

    비어 있지 않으면 재현성이 깨진다 - CI가 이 목록을 실패 조건으로 쓴다(§5.1.3).
    """
    covered: set[str] = set()
    for name, child in module.named_modules():
        if callable(getattr(child, "reset_parameters", None)):
            for param_name, _ in child.named_parameters(recurse=False):
                covered.add(f"{name}.{param_name}" if name else param_name)
    return [name for name, _ in module.named_parameters() if name not in covered]
