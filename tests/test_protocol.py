"""와이어 프로토콜 - 기획서 §17.5의 정본 예시를 그대로 먹인다."""

import json

import msgpack
import pytest

from torchflow import protocol as proto

RUN_NODES = {
    "type": "RunNodes", "req_id": "r-8801", "level": "L0", "pass": "static", "probe_batch": 2,
    "batch": [{"node_id": "01J9Q4B2", "version": 12, "inputs": ["spec:3f9c"]},
              {"node_id": "01J9Q4B3", "version": 12, "inputs": ["spec:@01J9Q4B2"]}],
}

DONE = {
    "type": "Done", "req_id": "r-8812", "node_id": "01J9Q4A2", "version": 12,
    "output_key": "sha256:a71d", "digest": "sha256:0be2",
    "spec": {"shape": ["B", 64, 384], "dtype": "bfloat16", "device": "cuda:1"},
    "elapsed_ms": 412, "cache_hit": False,
    "grad": {"norm": 3.2e-7, "ratio": 4.1e-6, "objective": "CE(real batch)", "init_step": True},
    "numeric": {"nan": 0, "inf": 0, "first_node": None},
    "frames": [{"kind": "hist", "dtype": "float32", "shape": [64]},
               {"kind": "thumb", "format": "webp", "bytes_len": 2140}],
}

ERROR = {
    "type": "Error", "req_id": "r-8813", "kind": "shape", "node_id": "01J9Q4B5",
    "message": "mat1 and mat2 shapes cannot be multiplied (256x384 and 768x10)",
    "mapping": {"expected": {"in_features": 768}, "received": {"shape": ["B", 384]},
                "source_port": "01J9Q4B4.output",
                "suggestions": [{"op": "set_param", "instance": "01J9I103",
                                 "path": "in_features", "value": 384}]},
    "traceback": None,
}


def test_run_nodes_example_parses():
    message, _ = proto.decode_to_kernel([msgpack.packb(RUN_NODES)])
    assert message.pass_ == "static"  # 'pass'는 예약어라 alias로 받는다.
    assert [item.node_id for item in message.batch] == ["01J9Q4B2", "01J9Q4B3"]


def test_pass_alias_survives_encoding():
    message, _ = proto.decode_to_kernel([msgpack.packb(RUN_NODES)])
    assert msgpack.unpackb(proto.encode(message)[0], raw=False)["pass"] == "static"


def test_done_example_parses_with_fields_we_have_not_modelled_yet():
    """L1 프로브 필드(grad)는 아직 모델에 없다. 조용히 보존되어야 한다."""
    message, _ = proto.decode_from_kernel([msgpack.packb(DONE)])
    assert message.spec["shape"] == ["B", 64, 384]
    assert message.grad["ratio"] == pytest.approx(4.1e-6)
    assert msgpack.unpackb(proto.encode(message)[0], raw=False)["grad"]["norm"] == pytest.approx(3.2e-7)


def test_error_example_carries_node_attribution_and_suggestions():
    message, _ = proto.decode_from_kernel([msgpack.packb(ERROR)])
    assert message.kind == "shape" and message.node_id == "01J9Q4B5"
    assert message.mapping["suggestions"][0]["op"] == "set_param"


def test_multipart_frames_ride_alongside_the_header():
    frames = [b"\x00" * 256, b"webp-bytes"]
    parts = proto.encode(proto.Done(req_id="r-1", node_id="n"), frames)
    message, received = proto.decode_from_kernel(parts)
    assert message.node_id == "n" and received == frames


def test_inverse_op_round_trips():
    """Undo는 서버 히스토리 되감기가 아니라 역 op 전송이다(§8.2.3)."""
    message = proto.decode_from_browser(json.dumps(
        {"type": "Op", "client_id": "c-01", "tmp_seq": 93, "kind": "set_switch_active",
         "inverse_of": 5121, "payload": {"instance": "01J9I002", "active": "baseline"}}))
    assert message.inverse_of == 5121 and message.payload["active"] == "baseline"


def test_unknown_message_type_is_rejected():
    with pytest.raises(Exception):
        proto.decode_to_kernel([msgpack.packb({"type": "Nope", "req_id": "r"})])
