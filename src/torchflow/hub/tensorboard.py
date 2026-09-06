"""TensorBoard 어댑터 (기획서 §11 실험 추적).

트래커에 쌓인 곡선을 TensorBoard가 읽는 ``tfevents``로 내보낸다. 자체 클라우드는
비목표이고, 이미 TensorBoard를 켜 놓고 일하는 사람이 하던 대로 보게 하는 것이
목적이다.

``torch.utils.tensorboard.SummaryWriter``를 쓰지 않는 이유는 두 가지다. hub는 torch를
import하지 않고(§5.8), ``SummaryWriter``는 ``tensorboard`` 패키지를 요구한다. 우리가
쓰는 것은 스칼라 레코드뿐이라 형식을 직접 쓴다 - TFRecord 프레임 + protobuf 필드 3개.

``ponytail: 스칼라만. 히스토그램·이미지는 이 형식을 한 겹 더 쌓아야 하고, 필요해지면
그때 붙인다.``
"""

from __future__ import annotations

import socket
import struct
import time
from pathlib import Path

# CRC-32C(Castagnoli). TFRecord 프레임이 요구하는 체크섬이고 zlib에는 없다.
_POLY = 0x82F63B78
_TABLE = []
for _index in range(256):
    _crc = _index
    for _ in range(8):
        _crc = (_crc >> 1) ^ (_POLY if _crc & 1 else 0)
    _TABLE.append(_crc)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ _TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


def _masked(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _frame(payload: bytes) -> bytes:
    """TFRecord 한 칸: 길이 · 길이의 CRC · 내용 · 내용의 CRC."""
    header = struct.pack("<Q", len(payload))
    return (header + struct.pack("<I", _masked(crc32c(header)))
            + payload + struct.pack("<I", _masked(crc32c(payload))))


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _bytes_field(number: int, payload: bytes) -> bytes:
    return bytes([(number << 3) | 2]) + _varint(len(payload)) + payload


def _scalar_event(tag: str, value: float, step: int, wall: float) -> bytes:
    """Event{wall_time, step, summary: Summary{value: [{tag, simple_value}]}}."""
    entry = _bytes_field(1, tag.encode()) + b"\x15" + struct.pack("<f", float(value))
    summary = _bytes_field(1, entry)
    return _frame(b"\x09" + struct.pack("<d", wall)
                  + b"\x10" + _varint(int(step))
                  + _bytes_field(5, summary))


def _header_event(wall: float) -> bytes:
    return _frame(b"\x09" + struct.pack("<d", wall) + b"\x10\x00"
                  + _bytes_field(3, b"brain.Event:2"))


def export(tracker, logdir: Path | str, run_ids: list[str] | None = None) -> list[Path]:
    """트래커의 run들을 ``<logdir>/<run_id>/events.out.tfevents.*``로 쓴다.

    TensorBoard는 디렉터리 하나를 run 하나로 본다. 그래서 run별로 폴더를 판다 -
    그래야 곡선이 겹쳐 그려진다.
    """
    logdir = Path(logdir)
    ids = run_ids if run_ids is not None else [run.id for run in tracker.runs(limit=1000)]
    written = []
    for run_id in ids:
        keys = tracker.keys(run_id)
        if not keys:
            continue
        directory = logdir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        path = directory / (f"events.out.tfevents.{int(now)}."
                            f"{socket.gethostname()}.torchflow")
        with path.open("wb") as stream:
            stream.write(_header_event(now))
            for key in keys:
                for step, value in tracker.curve(run_id, key):
                    stream.write(_scalar_event(key, value, step, now))
        written.append(path)
    return written
