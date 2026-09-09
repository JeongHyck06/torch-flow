"""과제 계약 - 무엇을 배우는가를 모델·데이터와 분리해 한곳에 둔다.

지금까지 학습 루프는 분류를 **가정**했다. `worker`가 `nn.CrossEntropyLoss()`를 직접
만들고, 평가는 `output.argmax(dim=1) == y`였고, 데이터 쪽은 정답 열을 무조건 클래스
인덱스로 바꿨다. 그래서 회귀는 표현할 길이 자체가 없었다.

여기서 과제 하나가 정하는 것은 넷이다.

* **목적 함수** - 어떤 loss를 쓸 수 있고 기본이 무엇인가.
* **정답의 자료형** - 클래스 인덱스(int64)인가 실수(float32)인가.
* **평가 지표** - 학습 중 곡선에 무엇을 얹고, 최종 테스트에 무엇을 적는가.
* **예측** - 모델 출력에서 사람이 읽는 답을 어떻게 꺼내는가.

모델 구조도, 데이터 종류도 여기 들어오지 않는다. 표 데이터 회귀와 시계열 예측은 같은
`regression` 계약을 쓰고, 이미지 분류와 텍스트 분류는 같은 `classification`을 쓴다.

torch를 모듈 수준에서 import하지 않는다 - hub가 이 파일을 읽어 과제 목록과 loss 선택지를
UI에 보내야 하는데, hub는 torch를 import하지 않는다(`test_hub_never_imports_torch`).
"""

from __future__ import annotations

from typing import Any

# 과제 종류. `label`은 화면에 나가는 이름이고, `losses`의 첫 항목이 기본값이다.
TASKS: dict[str, dict[str, Any]] = {
    "classification": {
        "label": "분류",
        "hint": "정답이 정해진 몇 갈래 중 하나일 때. 붓꽃 품종, 스팸 여부, 손글씨 숫자.",
        "losses": ["cross_entropy", "bce_with_logits"],
        "target_dtype": "int64",
        # 학습 곡선에 얹는 것. val_loss는 과제와 무관하게 언제나 있다.
        "metrics": ["acc"],
        "needs_classes": True,
    },
    "regression": {
        "label": "회귀",
        "hint": "정답이 연속된 수일 때. 집값, 온도, 다음 구간의 값.",
        "losses": ["mse", "l1", "huber"],
        "target_dtype": "float32",
        "metrics": ["mae", "rmse", "r2"],
        "needs_classes": False,
    },
}

LOSSES: dict[str, str] = {
    "cross_entropy": "교차 엔트로피",
    "bce_with_logits": "이진 교차 엔트로피",
    "mse": "평균 제곱 오차",
    "l1": "평균 절대 오차",
    "huber": "Huber",
}


def spec(task: str | None) -> dict[str, Any]:
    """과제 이름을 계약으로. 모르는 이름은 분류로 떨어뜨린다 - 예전 job.json에는
    ``task``가 아예 없고, 그것들은 전부 분류였다."""
    return TASKS.get(str(task or "classification"), TASKS["classification"])


def default_loss(task: str | None) -> str:
    return spec(task)["losses"][0]


def resolve_loss(task: str | None, loss: str | None) -> str:
    """고른 loss가 그 과제에 쓸 수 있는 것인지 본다. 아니면 기본값.

    과제를 분류에서 회귀로 바꾸면 예전에 고른 ``cross_entropy``가 job에 남아 있는데,
    그대로 쓰면 정답 dtype이 float32라 torch가 알 수 없는 말로 죽는다.
    """
    allowed = spec(task)["losses"]
    return loss if loss in allowed else allowed[0]


def metric_names(task: str | None) -> list[str]:
    """이 과제가 내는 검증 지표 이름. 곡선 축을 미리 세우는 데 쓴다."""
    return ["val_loss"] + [f"val_{name}" for name in spec(task)["metrics"]]


# ── torch가 필요한 부분. 워커와 커널에서만 부른다 ────────────────────────────


def build_loss(name: str):
    """이름을 loss 모듈로. 워커의 학습 루프와 테스트가 같은 것을 쓴다."""
    from torch import nn

    return {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "huber": nn.HuberLoss,
    }[name]()


def align(task: str | None, output, target):
    """loss에 넣기 전에 출력과 정답의 모양을 맞춘다.

    회귀에서 모델이 `[B, 1]`을 내고 정답이 `[B]`이면 MSELoss는 **오류 없이**
    `[B, B]`로 브로드캐스트해 완전히 다른 값을 배운다. 조용히 틀리는 종류라
    여기서 한 번에 막는다. 분류는 `[B, C]` × `[B]`가 정상이므로 건드리지 않는다.
    """
    if spec(task)["needs_classes"]:
        return output, target
    if output.dim() == 2 and output.shape[1] == 1:
        output = output.squeeze(1)
    if target.dim() == 2 and target.shape[1] == 1:
        target = target.squeeze(1)
    return output, target.to(output.dtype)


def predict(task: str | None, output):
    """모델 출력에서 사람이 읽는 답. 분류는 고른 클래스, 회귀는 값 그대로."""
    if spec(task)["needs_classes"]:
        return output.argmax(dim=1)
    return output.squeeze(1) if output.dim() == 2 and output.shape[1] == 1 else output


class Accumulator:
    """분할 하나를 배치로 훑으며 지표를 모은다.

    평가와 최종 테스트가 같은 것을 쓴다 - 두 곳에서 따로 세면 화면의 숫자와 test.json이
    어긋난다. 표본 수로 가중해 더하므로 마지막 배치가 작아도 평균이 맞는다.
    """

    def __init__(self, task: str | None, scale: dict[str, float] | None = None):
        self.task = task
        self.needs_classes = spec(task)["needs_classes"]
        # 회귀 정답을 표준화해서 배웠으면 지표는 원래 단위로 되돌려 보고한다.
        # loss는 실제로 최적화하는 값이므로 되돌리지 않는다 - 곡선이 바뀌면 안 된다.
        self.scale = scale
        self.n = 0
        self.loss_sum = 0.0
        self.correct = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.y_sum = 0.0
        self.y_sq_sum = 0.0

    def add(self, output, target, loss_value: float) -> None:
        count = len(target)
        self.n += count
        self.loss_sum += loss_value * count
        if self.needs_classes:
            self.correct += float((predict(self.task, output) == target).sum())
            return
        guess, truth = align(self.task, output, target)
        if self.scale is not None:
            guess = guess * self.scale["std"] + self.scale["mean"]
            truth = truth * self.scale["std"] + self.scale["mean"]
        error = (guess - truth).detach()
        self.abs_sum += float(error.abs().sum())
        self.sq_sum += float((error ** 2).sum())
        self.y_sum += float(truth.sum())
        self.y_sq_sum += float((truth ** 2).sum())

    def result(self, prefix: str = "val_") -> dict[str, float]:
        if not self.n:
            return {}
        out = {f"{prefix}loss": self.loss_sum / self.n}
        if self.needs_classes:
            out[f"{prefix}acc"] = self.correct / self.n
            return out
        mean = self.y_sum / self.n
        # 정답의 전체 분산. 0이면 R²가 정의되지 않는다(정답이 전부 같은 값).
        variance = self.y_sq_sum / self.n - mean * mean
        out[f"{prefix}mae"] = self.abs_sum / self.n
        out[f"{prefix}rmse"] = (self.sq_sum / self.n) ** 0.5
        if variance > 1e-12:
            out[f"{prefix}r2"] = 1.0 - (self.sq_sum / self.n) / variance
        return out


def headline(task: str | None, metrics: dict[str, float], prefix: str = "val_") -> str:
    """터미널 한 줄과 화면 요약에 쓰는 대표 숫자."""
    if spec(task)["needs_classes"]:
        value = metrics.get(f"{prefix}acc")
        return f"정확도 {value:.4f}" if value is not None else ""
    value = metrics.get(f"{prefix}rmse")
    return f"RMSE {value:.4f}" if value is not None else ""
