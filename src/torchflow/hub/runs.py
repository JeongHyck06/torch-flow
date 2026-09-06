"""L2 학습 run의 hub 쪽 (기획서 §5.5.3, §13.1 M7).

hub는 학습을 **돌리지 않는다**. 워커를 분리된 세션으로 띄우고, 워커가 남긴
``events.jsonl``을 읽어 트래커에 옮기고, 명령은 ``control.json``에 쓴다.
그래서 hub가 죽어도 학습은 계속되고, hub가 다시 떠도 커서만 이어 읽으면 된다.

torch를 import하지 않는다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

# 워커가 이 시간 넘게 heartbeat를 안 쓰면 응답 없음으로 본다(§5.5.2와 같은 10 s).
STALE_AFTER = 10.0


@dataclass
class RunHandle:
    run_id: str
    directory: Path
    process: subprocess.Popen | None = None
    cursor: int = 0                       # events.jsonl에서 어디까지 읽었나
    state: str = "starting"
    step: int = 0
    total: int = 0
    device: str = ""
    error: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    overrides_dirty: bool = False         # 새 hparam 이벤트를 읽었다 - 파일을 다시 써야 한다

    @property
    def alive(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            return True
        beat = self.directory / "heartbeat"
        try:
            return (time.time() - beat.stat().st_mtime) < STALE_AFTER
        except OSError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "state": self.state, "step": self.step,
                "total": self.total, "device": self.device, "alive": self.alive,
                "error": self.error, **self.extra}


def new_run_id() -> str:
    # 초 단위 타임스탬프만으로는 fork가 원본과 같은 id를 받을 수 있다.
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"


# §5.7.1의 분류표. 워커가 ``control.json``에서 매 스텝 읽는 것만 HOT이고,
# 여기 없는 경로는 - 아키텍처 변경을 포함해 - 재시작이다.
HOT = {"lr": "lr", "wd": "weight_decay", "weight_decay": "weight_decay",
       "grad_clip": "grad_clip", "log_every": "log_every"}
SCHEDULER = {"scheduler", "warmup", "schedule"}
COLD = {"seed", "dataset", "split", "input_shape", "num_classes"}


def classify(path: str) -> str:
    """hparam 경로 -> ``hot`` | ``scheduler`` | ``restart`` | ``cold`` (§5.7.1).

    ``restart``는 가중치를 유지한 재시작, ``cold``는 step 0부터다. Phase B에서
    재시작은 **표시만** 한다 - 자동 재시작은 신뢰를 깨므로 사람이 누른다(ADR-05).
    """
    segments = path.split(".")
    if segments[-1] in HOT:
        return "hot"
    if set(segments) & SCHEDULER:
        return "scheduler"
    if segments[-1] in COLD:
        return "cold"
    return "restart"


def hot_command(path: str, value: Any) -> dict[str, Any]:
    """HOT 경로를 ``control.json`` 키로 옮긴다."""
    return {HOT[path.split(".")[-1]]: float(value)}


def start(*, run_id: str, root: Path, job: dict[str, Any], code: str,
          python: str | None = None) -> RunHandle:
    """워커를 띄운다. 학습 대상은 함께 써 두는 생성 코드다(§7.2).

    ``start_new_session=True``로 프로세스 그룹을 분리한다 - hub를 Ctrl+C로 내려도
    학습은 살아 있어야 한다(§5.5.3).
    """
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(code, encoding="utf-8")

    job = {**job, "code": str(directory / "model.py")}
    (directory / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (directory / "control.json").write_text("{}", encoding="utf-8")

    return RunHandle(run_id=run_id, directory=directory, process=_spawn(directory, python),
                     total=int(job.get("steps", 0)))


def _spawn(directory: Path, python: str | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        # -u: 파이썬이 파이프로 나갈 때 stdout을 버퍼링한다. UI의 "학습 출력"이
        # 몇십 KB씩 뭉텅이로 늦게 도착하지 않으려면 꺼야 한다.
        [python or sys.executable, "-u", "-m", "torchflow.worker",
         "--job", str(directory / "job.json")],
        stdout=(directory / "stdout.log").open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(directory.parent.parent),
    )


def resume(handle: RunHandle, *, steps: int | None = None,
           python: str | None = None) -> RunHandle:
    """멈췄거나 죽은 run을 체크포인트에서 이어 돌린다(§5.7.2).

    run id도 ``events.jsonl``도 그대로 쓴다. 곡선은 끊긴 step에서 이어지고 hub의
    커서도 그대로라 이미 읽은 이벤트를 다시 읽지 않는다.
    """
    path = handle.directory / "job.json"
    job = json.loads(path.read_text(encoding="utf-8"))
    ckpt = handle.directory / "ckpt.pt"
    if ckpt.exists():
        job["resume"] = str(ckpt)
    if steps:
        job["steps"] = int(steps)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    # 이전 stop/pause 명령을 지운다 - 안 그러면 새 워커가 첫 스텝에 다시 멈춘다.
    (handle.directory / "control.json").write_text("{}", encoding="utf-8")

    handle.process = _spawn(handle.directory, python)
    handle.state = "starting"
    handle.total = int(job.get("steps", handle.total))
    return handle


def fork(handle: RunHandle, *, root: Path, changes: dict[str, Any],
         python: str | None = None, timeout: float = 5.0) -> RunHandle:
    """reported run을 갈라낸다(§5.7.2).

    reported run은 manifest의 hparam이 동결이라 in-place로 못 바꾼다. 원 run을
    멈추고 마지막 ckpt를 복사해 새 run id로 잇는다 - 원 run의 곡선은 갈라진
    지점까지 그대로 남고 보고에 쓸 수 있다.
    """
    ckpt = handle.directory / "ckpt.pt"
    stamp = ckpt.stat().st_mtime if ckpt.exists() else 0.0
    command(handle, stop=True)
    _await_checkpoint(handle, stamp, timeout)

    job = {**json.loads((handle.directory / "job.json").read_text(encoding="utf-8")), **changes}
    code = (handle.directory / "model.py").read_text(encoding="utf-8")

    child_id = new_run_id()
    directory = root / child_id
    directory.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        shutil.copy2(ckpt, directory / "ckpt.pt")
        job["resume"] = str(directory / "ckpt.pt")
    else:
        job.pop("resume", None)     # 아직 한 번도 저장 못 했으면 처음부터다.

    child = start(run_id=child_id, root=root, job=job, code=code, python=python)
    child.extra["parent_run"] = handle.run_id
    return child


def _await_checkpoint(handle: RunHandle, stamp: float, timeout: float) -> bool:
    """Stop 직후 워커가 마지막 ckpt를 쓸 시간을 준다. 없으면 없는 대로 간다."""
    path = handle.directory / "ckpt.pt"
    deadline = time.time() + timeout
    while time.time() < deadline and handle.alive:
        if path.exists() and path.stat().st_mtime > stamp:
            return True
        time.sleep(0.05)
    return path.exists()


def write_overrides(run_id: str, entries: list[dict[str, Any]], root: Path) -> Path:
    """``conf/overrides/<run_id>.yaml`` 물질화(§5.7.2).

    학습 중에 바꾼 값은 코드 어디에도 안 남는다. 이 파일이 남아야 ``reproduce.sh``가
    같은 step에서 같은 변경을 다시 걸 수 있다.
    """
    path = root / "conf" / "overrides" / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# TorchFlow run {run_id} - 학습 중 적용된 hparam 변경(기획서 §5.7.2)"]
    lines += [f"- {{step: {entry.get('step', 0)}, path: {entry['path']},"
              f" value: {entry['value']!r}}}" for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def command(handle: RunHandle, **fields: Any) -> dict[str, Any]:
    """``control.json``을 갈아 쓴다. 워커는 매 스텝 이 파일을 읽는다."""
    path = handle.directory / "control.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        current = {}
    current.update(fields)
    path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
    return current


def merge(handle: RunHandle, tracker) -> int:
    """새로 쌓인 이벤트를 트래커로 옮긴다. 읽은 위치는 커서로 기억한다.

    hub가 죽었다 살아나도 파일이 정본이므로 이어 읽으면 된다(§5.5.3).
    """
    path = handle.directory / "events.jsonl"
    if not path.exists():
        return 0

    written = 0
    # 텍스트 반복 중에는 tell()을 못 쓴다. 바이트로 읽고 소비한 만큼만 커서를 민다.
    with path.open("rb") as stream:
        stream.seek(handle.cursor)
        chunk = stream.read()
    for raw in chunk.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break                         # 반쯤 쓰인 줄은 다음 번에 읽는다.
        handle.cursor += len(raw)
        try:
            event = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        written += _absorb(handle, event, tracker)
    return written


def _absorb(handle: RunHandle, event: dict[str, Any], tracker) -> int:
    kind = event.get("kind")
    if kind == "scalar":
        step = int(event.get("step", 0))
        handle.step = step
        # 곡선과 별개로 "지금 lr"을 들고 있는다. 스케줄러가 있으면 base도 같이 -
        # 화면이 "현재 lr = base x schedule"을 쓸 수 있어야 한다(§5.7.1).
        for name in ("lr", "base_lr"):
            if (value := event.get(name)) is not None:
                handle.extra[name] = value
        values = {key: value for key, value in event.items()
                  if key not in ("kind", "wall", "step")}
        return tracker.log(handle.run_id, step, values, wall=event.get("wall"))
    if kind == "status":
        handle.state = str(event.get("state", handle.state))
        handle.step = int(event.get("step", handle.step))
        handle.total = int(event.get("steps", handle.total)) or handle.total
        handle.device = str(event.get("device", handle.device))
        if scheduler := event.get("scheduler"):
            handle.extra["scheduler"] = scheduler
        if handle.state in ("done", "stopped", "failed"):
            tracker.finish_run(handle.run_id, handle.state)
        if reason := event.get("reason"):
            handle.extra["reason"] = reason
    elif kind == "error":
        handle.error = {"stage": event.get("stage"), "message": event.get("message")}
    elif kind == "numeric" and event.get("nan"):
        # NaN은 학습을 멈추는 사건이다(§5.6 nan_policy) - 배지로 남긴다.
        handle.extra["nan_step"] = event.get("step")
    elif kind == "hparam":
        # 중복 방지: hub가 명령을 낼 때 이미 넣었을 수 있고, 복구 시 파일을 처음부터
        # 다시 읽기도 한다. 같은 사실을 두 번 적으면 overrides 파일이 거짓말을 한다.
        entry = {"step": event.get("step"), "path": event.get("path"),
                 "value": event.get("value")}
        overrides = handle.extra.setdefault("overrides", [])
        if entry not in overrides:
            overrides.append(entry)
            handle.overrides_dirty = True
    return 0
