"""L2 학습 run의 hub 쪽 (기획서 §5.5.3, §13.1 M7).

hub는 학습을 **돌리지 않는다**. 워커를 분리된 세션으로 띄우고, 워커가 남긴
``events.jsonl``을 읽어 트래커에 옮기고, 명령은 ``control.json``에 쓴다.
그래서 hub가 죽어도 학습은 계속되고, hub가 다시 떠도 커서만 이어 읽으면 된다.

torch를 import하지 않는다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid() % 1000:03d}"


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

    process = subprocess.Popen(
        [python or sys.executable, "-m", "torchflow.worker", "--job", str(directory / "job.json")],
        stdout=(directory / "stdout.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(root.parent),
    )
    return RunHandle(run_id=run_id, directory=directory, process=process,
                     total=int(job.get("steps", 0)))


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
        values = {key: value for key, value in event.items()
                  if key not in ("kind", "wall", "step")}
        return tracker.log(handle.run_id, step, values, wall=event.get("wall"))
    if kind == "status":
        handle.state = str(event.get("state", handle.state))
        handle.step = int(event.get("step", handle.step))
        handle.total = int(event.get("steps", handle.total)) or handle.total
        handle.device = str(event.get("device", handle.device))
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
        handle.extra.setdefault("overrides", []).append(
            {"step": event.get("step"), "path": event.get("path"), "value": event.get("value")})
    return 0
