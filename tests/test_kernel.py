"""커널 프로세스: 기동 · 왕복 · 격리 (기획서 §8.1, §5.5, §5.8)."""

import json
import os
import signal
import time

import pytest

from conftest import MINIVIT
from torchflow import protocol as proto
from torchflow.hub.kernels import KernelManager
from torchflow.ir import canonical_json, load

pytest.importorskip("torch")


@pytest.fixture
def kernel(tmp_path):
    manager = KernelManager("L0", state_dir=tmp_path / "state")
    manager.start()
    yield manager
    manager.stop()


def snapshot():
    payload = json.loads(canonical_json(load(MINIVIT)))
    payload["rt"] = {"num_classes": 10}
    return payload


def test_kernel_announces_itself_and_writes_the_registry(kernel, tmp_path):
    assert kernel.ready.level == "L0" and kernel.ready.pid > 0
    registry = json.loads((tmp_path / "state" / "registry.json").read_text(encoding="utf-8"))
    labels = {block["label"] for block in registry["blocks"]}
    # §17.3 Phase A 등재분의 대표 블록들.
    assert {"Linear", "LayerNorm", "Conv2d", "MultiheadAttention", "GELU"} <= labels


def test_ping_round_trip_is_fast(kernel):
    started = time.perf_counter()
    assert kernel.responsive()
    assert (time.perf_counter() - started) * 1000 < 50


def test_run_nodes_returns_specs_for_the_requested_batch(kernel):
    replies = kernel.request(
        proto.RunNodes(
            req_id="r-1",
            graph=snapshot(),
            batch=[proto.NodeRequest(node_id=node, version=1) for node in ("01J9Q4B2", "01J9Q4B5")],
        )
    )
    done = {reply.node_id: reply for reply in replies if reply.type == "Done"}
    assert set(done) == {"01J9Q4B2", "01J9Q4B5"}
    assert done["01J9Q4B5"].spec["shape"] == ["B", 10]
    assert replies[-1].type == "Progress"


def test_kernel_errors_are_reported_not_fatal(kernel):
    broken = snapshot()
    broken["rt"] = {}  # rt.num_classes 미해소
    replies = kernel.request(proto.RunNodes(req_id="r-2", graph=broken))
    assert any(reply.type == "Error" for reply in replies)
    assert kernel.responsive(), "kernel died on a user-graph error"


def test_kernel_survives_a_malformed_frame(kernel):
    kernel.socket.send_multipart([kernel.identity, b"\xff\xff not msgpack"])
    kernel._recv(2.0)
    assert kernel.responsive()


def test_ui_survives_a_kernel_kill(kernel):
    """커널을 SIGKILL해도 hub는 살아 있고 10초 안에 다시 선다(M1 완료 기준)."""
    os.kill(kernel.ready.pid, signal.SIGKILL)
    time.sleep(0.2)
    assert not kernel.alive()

    started = time.perf_counter()
    kernel.ensure()
    elapsed = time.perf_counter() - started
    assert kernel.alive() and kernel.responsive()
    assert elapsed < 10.0, f"kernel restart took {elapsed:.1f}s"


def test_concurrent_requests_do_not_corrupt_the_socket(kernel):
    """uvicorn은 동기 핸들러를 스레드풀에서 돌린다 - 소켓 접근이 겹쳐도 살아남아야 한다."""
    import concurrent.futures

    def one(index: int):
        return kernel.request(proto.RunNodes(req_id=f"r-{index}", graph=snapshot()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = [future.result() for future in
                   concurrent.futures.as_completed([pool.submit(one, i) for i in range(6)])]

    assert all(replies[-1].type == "Progress" for replies in results)
    assert kernel.responsive()


def make_l1_kernel(tmp_path):
    manager = KernelManager("L1", state_dir=tmp_path / "l1")
    manager.start()
    return manager


def test_l1_kernel_returns_gradients_per_node(tmp_path):
    """RunClosure 하나로 조상 폐쇄 전체의 grad가 돌아온다(§5.1.2)."""
    kernel = make_l1_kernel(tmp_path)
    try:
        replies = kernel.request(proto.RunClosure(
            req_id="p-1", graph=snapshot(),
            probe_cfg={"batch": 2, "objective": "random_target_ce"}))
        done = [reply for reply in replies if reply.type == "Done"]
        assert done, [reply.type for reply in replies]
        graded = [reply for reply in done if (reply.grad or {}).get("norm") is not None]
        assert len(graded) > 10
        assert all(reply.grad["objective"] for reply in graded)
        assert replies[-1].type == "Progress"
    finally:
        kernel.stop()


def test_l1_kernel_reports_the_device_it_chose(tmp_path):
    kernel = make_l1_kernel(tmp_path)
    try:
        replies = kernel.request(proto.RunClosure(
            req_id="p-2", graph=snapshot(), probe_cfg={"batch": 1}))
        devices = {(reply.grad or {}).get("device") for reply in replies if reply.type == "Done"}
        assert len(devices) == 1 and next(iter(devices))
    finally:
        kernel.stop()


def test_concurrent_requests_start_one_kernel(tmp_path):
    """첫 화면과 첫 shape 요청은 겹친다 - 커널이 둘 뜨면 shape가 비어 돌아온다."""
    import threading

    from torchflow.hub.kernels import KernelManager

    manager = KernelManager("L0", tmp_path / "state")
    pids, errors = [], []

    def touch() -> None:
        try:
            manager.ensure()
            pids.append(manager.ready.pid)
        except Exception as exc:   # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=touch) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(set(pids)) == 1
    finally:
        manager.stop()


def test_kernel_exits_when_its_parent_dies(tmp_path):
    """hub가 어떻게 죽든(SIGKILL 포함) 커널이 고아로 남지 않는다."""
    import subprocess
    import sys
    import textwrap

    parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import os, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-m", "torchflow.kernel",
                                  "--endpoint", "tcp://127.0.0.1:1",
                                  "--state-dir", {str(tmp_path / "state")!r},
                                  "--parent", str(os.getpid())])
        print(child.pid, flush=True)
        time.sleep(60)
    """)], stdout=subprocess.PIPE, text=True)
    kernel_pid = int(parent.stdout.readline())
    parent.kill()
    parent.wait()

    def alive() -> bool:
        try:
            os.kill(kernel_pid, 0)
        except ProcessLookupError:
            return False
        return True

    deadline = time.time() + 20
    while time.time() < deadline and alive():
        time.sleep(0.2)
    assert not alive(), "커널이 부모 없이 살아남았다"
