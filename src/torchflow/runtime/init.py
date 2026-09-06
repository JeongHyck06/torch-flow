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


def _reset_of(module: nn.Module):
    """이 모듈을 다시 초기화하는 함수. 없으면 ``None``.

    ``nn.MultiheadAttention``은 ``reset_parameters``가 아니라 비공개
    ``_reset_parameters``를 갖는다. 이것까지 부르지 않으면 UI의 모듈 트리와
    생성 코드가 서로 다른 가중치에서 출발한다(§5.1.3).
    """
    for name in ("reset_parameters", "_reset_parameters"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def seeded_init(module: nn.Module, seed: int, prefix: str = "") -> None:
    """모든 leaf 모듈을 경로별 시드로 다시 초기화한다.

    다시 초기화할 방법이 없는 모듈의 파라미터는 :func:`init_coverage`가 보고한다.
    """
    for name, child in module.named_modules():
        reset = _reset_of(child)
        if reset is None:
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
        if _reset_of(child) is not None:
            for param_name, _ in child.named_parameters(recurse=False):
                covered.add(f"{name}.{param_name}" if name else param_name)
    return [name for name, _ in module.named_parameters() if name not in covered]
