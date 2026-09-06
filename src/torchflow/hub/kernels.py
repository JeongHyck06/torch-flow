"""KernelManager: 커널 프로세스 기동과 생존 감시 (기획서 §8.1, §5.5).

hub 쪽 규칙은 하나뿐이다: **이 파일도, hub의 어떤 파일도 torch를 import하지
않는다.** 커널은 별도 인터프리터 프로세스이며 여기서는 zmq 프레임만 오간다.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import zmq

from .. import protocol as proto


class KernelDead(RuntimeError):
    pass


class KernelManager:
    """커널 하나를 소유한다. ROUTER 소켓 한 개, 자식 프로세스 한 개."""

    def __init__(self, level: str = "L0", state_dir: Path | str = ".torchflow", python: str | None = None):
        self.level = level
        self.state_dir = Path(state_dir)
        self.python = python or sys.executable
        self.identity = f"{level.lower()}-kernel".encode()
        self.process: subprocess.Popen | None = None
        self.ready: proto.Ready | None = None

        # uvicorn은 동기 핸들러를 스레드풀에서 돌린다. pyzmq 소켓은 스레드 안전하지
        # 않아 두 요청이 겹치면 identity 프레임과 본문이 서로 다른 스레드로 갈린다.
        # 기동도 마찬가지다: 첫 화면과 첫 shape 요청이 겹치면 커널을 두 번 띄우고
        # Ready 핸드셰이크가 엇갈려 shape가 통째로 비어 돌아온다. ensure가 start를
        # 감싸므로 **재진입 가능한** 락이어야 한다(Lock이면 자기 자신에 걸린다).
        # ponytail: 커널 하나당 락 하나. 축(L0/L1/L2)이 늘어도 매니저가 늘 뿐이다.
        self.lock = threading.RLock()

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.ROUTER)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{self.port}"

    # 수명
    def start(self, timeout: float = 60.0) -> proto.Ready:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                self.python, "-m", "torchflow.kernel",
                "--endpoint", self.endpoint,
                "--level", self.level,
                "--identity", self.identity.decode(),
                "--state-dir", str(self.state_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with self.lock:
            message = self._recv(timeout)
        if message is None or message.type != "Ready":
            raise KernelDead(f"{self.level} kernel did not announce: {self._stderr()}")
        self.ready = message
        return message

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def responsive(self, timeout: float = 2.0) -> bool:
        if not self.alive():
            return False
        with self.lock:
            self._send(proto.Ping(req_id="ping"))
            message = self._recv(timeout)
        return message is not None and message.type == "Pong"

    def ensure(self) -> None:
        """죽었으면 다시 세운다. UI는 이 사이에도 멈추지 않는다(§5.5)."""
        with self.lock:
            if not self.alive():
                self.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self.alive():
            try:
                self.send(proto.Shutdown())
                self.process.wait(timeout)
            except (zmq.ZMQError, subprocess.TimeoutExpired):
                self.process.kill()
        self.socket.close(linger=0)

    # 왕복
    def send(self, message, frames=None) -> None:
        with self.lock:
            self._send(message, frames)

    def _send(self, message, frames=None) -> None:
        self.socket.send_multipart([self.identity, *proto.encode(message, frames)])

    def _recv(self, timeout: float):
        if not self.socket.poll(int(timeout * 1000)):
            return None
        parts = self.socket.recv_multipart()
        if len(parts) < 2:  # ROUTER는 identity 프레임을 앞에 붙인다.
            return None
        message, _ = proto.decode_from_kernel(parts[1:])
        return message

    def request(self, message, timeout: float = 120.0) -> list:
        """요청 하나를 보내고 종결 ``Progress``가 올 때까지 응답을 모은다."""
        deadline = time.monotonic() + timeout
        replies = []
        with self.lock:
            self._send(message)
            while time.monotonic() < deadline:
                reply = self._recv(max(0.0, deadline - time.monotonic()))
                if reply is None:
                    break
                replies.append(reply)
                if reply.type == "Progress":
                    return replies
        raise KernelDead(f"{self.level} kernel timed out: {self._stderr()}")

    def _stderr(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read(4000).decode(errors="replace")
        except Exception:
            return ""
