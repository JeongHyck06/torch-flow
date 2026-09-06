"""프로브 예산 (기획서 §5.1.4, §3.4).

두 개의 상한이 있다.

* 한 번의 probe 시간 - 기본 2초. 넘으면 순서 고정으로 물러선다:
  B 축소 -> forward-only -> ``auto_policy: manual`` 제안.
* 분당 GPU-초 - 기본 30 s/min. 넘으면 L1을 자동으로 끄고, 1분 유휴 후 되살린다.

축소 순서가 고정인 이유: 사용자가 "왜 갑자기 grad 배지가 사라졌지"를 매번 다르게
겪지 않아야 하기 때문이다.

``ponytail: 실측 시간 기반. §5.1.4의 L0 FLOPs 예측은 첫 probe 전에도 판단하려는
것인데, 실측 한 번이 예측보다 정확하고 첫 probe는 어차피 돌려봐야 안다.``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MIN_BATCH = 1
DISABLED_COOLDOWN_S = 60.0


@dataclass
class BudgetDecision:
    batch: int
    backward: bool
    enabled: bool
    reason: str

    @property
    def degraded(self) -> bool:
        return not self.enabled or not self.backward


@dataclass
class ProbeBudget:
    budget_ms: float = 2000.0
    gpu_sec_per_min: float = 30.0
    batch: int = 4
    _samples: list[tuple[float, float]] = field(default_factory=list)   # (끝난 시각, 초)
    _disabled_until: float = 0.0
    _backward: bool = True

    def record(self, elapsed_ms: float, *, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._samples.append((now, elapsed_ms / 1000))
        self._trim(now)

        if elapsed_ms <= self.budget_ms:
            return
        # 순서 고정: B 축소 -> forward-only -> 비활성.
        if self.batch > MIN_BATCH:
            self.batch = max(MIN_BATCH, self.batch // 2)
        elif self._backward:
            self._backward = False
        else:
            self._disabled_until = now + DISABLED_COOLDOWN_S

    def _trim(self, now: float) -> None:
        self._samples = [sample for sample in self._samples if now - sample[0] <= 60.0]

    def spent_gpu_seconds(self, *, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        self._trim(now)
        return sum(seconds for _, seconds in self._samples)

    def decide(self, *, now: float | None = None) -> BudgetDecision:
        now = now if now is not None else time.monotonic()

        if now < self._disabled_until:
            return BudgetDecision(self.batch, False, False,
                                  f"probe over budget; retrying in "
                                  f"{self._disabled_until - now:.0f}s")

        if self.spent_gpu_seconds(now=now) > self.gpu_sec_per_min:
            self._disabled_until = now + DISABLED_COOLDOWN_S
            return BudgetDecision(self.batch, False, False,
                                  f"L1 used more than {self.gpu_sec_per_min:g}s/min")

        return BudgetDecision(self.batch, self._backward, True,
                              "within budget" if self._backward else "forward-only")

    def reset(self) -> None:
        self.__init__(budget_ms=self.budget_ms, gpu_sec_per_min=self.gpu_sec_per_min)
