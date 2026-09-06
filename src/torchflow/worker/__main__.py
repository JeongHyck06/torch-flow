"""L2 학습 워커 (기획서 §5.5.3, §5.7, §13.1 M7).

hub와 **분리해서** 돈다. `start_new_session=True`로 기동되므로 hub가 죽어도,
브라우저를 닫아도 학습은 계속된다. 통신은 파일 두 개뿐이다.

* ``runs/<id>/events.jsonl`` - append-only. 스칼라·상태·오류가 여기 쌓이고 hub는
  읽기만 한다. hub가 없어도 기록은 남는다.
* ``runs/<id>/control.json`` - hub가 쓰고 워커가 매 스텝 읽는다(pause·stop·lr 변경).
  zmq 연결을 유지하지 않으므로 hub 재시작에도 명령 경로가 끊기지 않는다.
  ``ponytail: 파일 폴링. 스텝당 stat 한 번이고, 명령 지연은 스텝 하나다.``

학습 대상은 **생성 코드 그대로**다(§7.2). 그래프에서 뽑은 ``model.py``를 실행해
모델을 만들기 때문에, 화면에서 보던 모델과 학습되는 모델이 같은 파일이다.

    python -m torchflow.worker --job runs/<id>/job.json
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

HEARTBEAT_EVERY = 2.0      # 초. hub가 이 파일의 mtime으로 생존을 본다(§5.5.3).


class Events:
    """append-only 이벤트 로그. 한 줄이 하나의 사실이다."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> None:
        self.handle.write(json.dumps({"kind": kind, "wall": time.time(), **fields},
                                     ensure_ascii=False) + "\n")
        self.handle.flush()   # 크래시해도 마지막 스텝까지는 남아야 한다.


class Control:
    """hub가 남긴 명령. 없으면 그냥 계속 돈다."""

    def __init__(self, path: Path):
        self.path = path
        self.stamp = 0.0
        self.state: dict[str, Any] = {}

    def poll(self) -> dict[str, Any]:
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return self.state
        if stamp != self.stamp:
            self.stamp = stamp
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass          # 반쯤 쓰인 파일이면 다음 스텝에 다시 읽는다.
        return self.state


def build_model(job: dict[str, Any]):
    """생성 코드를 실행해 모델을 만든다. 학습 대상 = 사용자가 export할 그 코드."""
    code = Path(job["code"]).read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    exec(compile(code, job["code"], "exec"), namespace)   # noqa: S102 - 우리가 방금 생성한 코드다
    factory = namespace[job["class_name"]]
    return factory(**job.get("model_args", {}))


def build_data(job: dict[str, Any], device):
    """(입력, 타깃) 배치를 무한히 내놓는 것.

    기본은 **배울 것이 있는 합성 과제**다: 고정된 무작위 teacher 사영의 argmax를
    라벨로 쓴다. 순수 난수 라벨이면 손실이 ln(C)에서 평평해서 학습 루프가 도는지조차
    구분되지 않는다. 네트워크도 디스크도 건드리지 않는다(§3.1: 다운로드는 명시
    버튼으로만). 실제 데이터셋 노드는 Experiment Graph와 함께 온다.
    """
    import torch

    shape = job["input_shape"]
    classes = int(job.get("num_classes") or 10)
    batch = int(job.get("batch", 32))
    learnable = job.get("dataset", "teacher") != "noise"
    generator = torch.Generator().manual_seed(int(job.get("seed", 0)))

    features = 1
    for dim in shape[1:]:
        features *= int(dim)
    teacher = torch.randn(features, classes, generator=generator) if learnable else None

    while True:
        inputs = torch.randn([batch, *[int(dim) for dim in shape[1:]]], generator=generator)
        if teacher is None:
            targets = torch.randint(0, classes, (batch,), generator=generator)
        else:
            targets = (inputs.flatten(1) @ teacher).argmax(dim=1)
        yield inputs.to(device), targets.to(device)


def make_optimizer(torch, model, job: dict[str, Any]):
    name = str(job.get("optimizer", "adamw")).lower()
    lr = float(job.get("lr", 1e-3))
    weight_decay = float(job.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def pick_device(torch, requested: str | None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(job: dict[str, Any], run_dir: Path) -> None:
    import torch
    from torch import nn

    events = Events(run_dir / "events.jsonl")
    control = Control(run_dir / "control.json")
    heartbeat = run_dir / "heartbeat"

    try:
        model = build_model(job)
        device = pick_device(torch, job.get("device"))
        model.to(device).train()
        optimizer = make_optimizer(torch, model, job)
        loss_fn = nn.CrossEntropyLoss()
        data = build_data(job, device)
    except Exception as exc:
        events.write("error", stage="setup", message=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc())
        events.write("status", state="failed")
        return

    total = int(job.get("steps", 200))
    log_every = max(1, int(job.get("log_every", 1)))
    nan_policy = job.get("nan_policy", "pause")
    events.write("status", state="running", device=str(device), steps=total,
                 params=sum(p.numel() for p in model.parameters()))

    last_beat = 0.0
    step = 0
    while step < total:
        command = control.poll()
        if command.get("stop"):
            events.write("status", state="stopped", step=step)
            _checkpoint(torch, model, optimizer, run_dir, step)
            return
        if command.get("pause"):
            events.write("status", state="paused", step=step)
            while control.poll().get("pause") and not control.poll().get("stop"):
                _beat(heartbeat)
                time.sleep(0.2)
            if control.poll().get("stop"):
                events.write("status", state="stopped", step=step)
                _checkpoint(torch, model, optimizer, run_dir, step)
                return
            events.write("status", state="running", step=step)
        # lr은 HOT이다 - 재시작 없이 param_groups에 바로 반영한다(§5.7.1).
        if (lr := command.get("lr")) is not None:
            for group in optimizer.param_groups:
                if group["lr"] != float(lr):
                    group["lr"] = float(lr)
                    events.write("hparam", step=step, path="optim.lr", value=float(lr))

        inputs, targets = next(data)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_fn(output, targets)

        if not torch.isfinite(loss):
            events.write("numeric", step=step, nan=True)
            if nan_policy != "continue":
                _checkpoint(torch, model, optimizer, run_dir, step)
                events.write("status", state="paused" if nan_policy == "pause" else "stopped",
                             step=step, reason="nan")
                if nan_policy == "stop":
                    return
                while control.poll().get("pause", True) and not control.poll().get("stop"):
                    _beat(heartbeat)
                    time.sleep(0.2)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(job.get("grad_clip", 1e9)))
        optimizer.step()
        step += 1

        if step % log_every == 0 or step == total:
            events.write("scalar", step=step, loss=float(loss.detach()),
                         lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm))
        now = time.monotonic()
        if now - last_beat > HEARTBEAT_EVERY:
            last_beat = now
            _beat(heartbeat)

    _checkpoint(torch, model, optimizer, run_dir, step)
    events.write("status", state="done", step=step)


def _beat(path: Path) -> None:
    path.write_text(str(time.time()), encoding="utf-8")


def _checkpoint(torch, model, optimizer, run_dir: Path, step: int) -> None:
    """모델·옵티마이저·RNG. 재개는 M7 후반이고 지금은 잃지 않는 것이 목적이다."""
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state()}, run_dir / "ckpt.pt")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m torchflow.worker")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    train(job, job_path.parent)


if __name__ == "__main__":
    main()
