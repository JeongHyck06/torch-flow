"""TensorBoard 어댑터 (기획서 §11).

읽는 쪽을 테스트에 직접 두는 이유: tensorboard 패키지 없이도 우리가 쓴 바이트가
TFRecord 프레임과 protobuf 필드 번호를 지키는지 확인해야 하기 때문이다.
"""

import struct

from torchflow.hub.tensorboard import crc32c, export
from torchflow.hub.tracker import Tracker


def read_events(path):
    """tfevents를 되읽어 (tag, step, value)를 꺼낸다. 스칼라만 본다."""
    blob = path.read_bytes()
    found, offset = [], 0
    while offset < len(blob):
        (length,) = struct.unpack_from("<Q", blob, offset)
        (length_crc,) = struct.unpack_from("<I", blob, offset + 8)
        payload = blob[offset + 12:offset + 12 + length]
        (payload_crc,) = struct.unpack_from("<I", blob, offset + 12 + length)
        assert length_crc == _mask(crc32c(blob[offset:offset + 8])), "길이 CRC 불일치"
        assert payload_crc == _mask(crc32c(payload)), "내용 CRC 불일치"
        offset += 12 + length + 4
        if (entry := _scalar(payload)) is not None:
            found.append(entry)
    return found


def _mask(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _scalar(payload: bytes):
    """Event{wall_time(1,double), step(2,varint), summary(5)} 최소 파서."""
    offset, step = 0, 0
    while offset < len(payload):
        key = payload[offset]
        offset += 1
        if key == 0x09:                                  # wall_time
            offset += 8
        elif key == 0x10:                                # step
            step, offset = _varint(payload, offset)
        elif key in (0x1A, 0x2A):                        # file_version | summary
            length, offset = _varint(payload, offset)
            body = payload[offset:offset + length]
            offset += length
            if key == 0x2A:
                return (*_value(body), step)
        else:
            raise AssertionError(f"모르는 필드 {key:#x}")
    return None


def _value(summary: bytes):
    length, offset = _varint(summary, 1)                 # Summary.value
    entry = summary[offset:offset + length]
    length, offset = _varint(entry, 1)                   # Value.tag
    tag = entry[offset:offset + length].decode()
    assert entry[offset + length] == 0x15, "simple_value는 float 필드다"
    (value,) = struct.unpack_from("<f", entry, offset + length + 1)
    return tag, value


def _varint(data: bytes, offset: int):
    result = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def test_curves_come_back_out_as_tfevents(tmp_path):
    tracker = Tracker(tmp_path / "runs.db")
    tracker.ensure_run("run-1")
    for step, loss in enumerate([2.5, 1.5, 0.5], start=1):
        tracker.log("run-1", step, {"loss": loss})

    written = export(tracker, tmp_path / "tb")

    assert [path.parent.name for path in written] == ["run-1"]
    assert "tfevents" in written[0].name, "TensorBoard는 파일 이름으로 run을 찾는다"
    assert read_events(written[0]) == [("loss", 2.5, 1), ("loss", 1.5, 2), ("loss", 0.5, 3)]
    tracker.close()


def test_runs_without_scalars_are_skipped(tmp_path):
    """곡선이 없는 run까지 빈 폴더를 만들면 TensorBoard 목록이 쓰레기가 된다."""
    tracker = Tracker(tmp_path / "runs.db")
    tracker.ensure_run("empty")

    assert export(tracker, tmp_path / "tb") == []
    assert not (tmp_path / "tb" / "empty").exists()
    tracker.close()
