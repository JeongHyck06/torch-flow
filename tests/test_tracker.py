"""실험 트래커 (기획서 §10.3, §11)."""

import math

import pytest

from torchflow.hub.tracker import Tracker, downsample


@pytest.fixture
def tracker(tmp_path):
    store = Tracker(tmp_path / "runs.db")
    yield store
    store.close()


def test_run_and_scalars_survive_a_restart(tmp_path):
    """곡선을 잃으면 실험이 사라진다. 재시작해도 남아야 한다."""
    first = Tracker(tmp_path / "runs.db")
    first.ensure_run("run-1", name="ResNet", manifest={"seed": 0})
    first.log("run-1", 0, {"loss": 2.5})
    first.log("run-1", 1, {"loss": 2.1})
    first.close()

    second = Tracker(tmp_path / "runs.db")
    try:
        assert [run.id for run in second.runs()] == ["run-1"]
        assert second.run("run-1").manifest == {"seed": 0}
        assert second.curve("run-1", "loss") == [(0, 2.5), (1, 2.1)]
    finally:
        second.close()


def test_non_finite_values_are_dropped(tracker):
    """NaN 하나가 곡선 전체를 못 쓰게 만든다."""
    tracker.ensure_run("run-1")
    written = tracker.log("run-1", 0, {"loss": float("nan"), "lr": 1e-3,
                                       "inf": float("inf"), "note": "text"})
    assert written == 1
    assert tracker.keys("run-1") == ["lr"]


def test_curve_is_downsampled_but_keeps_the_ends(tracker):
    tracker.ensure_run("run-1")
    for step in range(9000):
        tracker.log("run-1", step, {"loss": 1.0 / (step + 1)})
    points = tracker.curve("run-1", "loss", max_points=500)
    assert len(points) == 500
    assert points[0][0] == 0 and points[-1][0] == 8999


def test_downsampling_keeps_spikes():
    """균일 추출은 스파이크를 놓친다. 스파이크가 대개 보고 싶은 그것이다."""
    points = [(index, 0.0) for index in range(2000)]
    points[1234] = (1234, 42.0)
    assert (1234, 42.0) in downsample(points, 100)


def test_seed_aggregate_gives_mean_and_std(tracker):
    for seed in range(3):
        tracker.ensure_run(f"run-{seed}")
        tracker.log(f"run-{seed}", 0, {"loss": 1.0 + seed})

    aggregate = tracker.aggregate([f"run-{seed}" for seed in range(3)], "loss")
    assert len(aggregate) == 1
    assert aggregate[0]["mean"] == pytest.approx(2.0)
    assert aggregate[0]["std"] == pytest.approx(math.sqrt(2 / 3), rel=1e-6)
    assert aggregate[0]["n"] == 3


def test_ensure_run_is_idempotent(tracker):
    tracker.ensure_run("run-1", name="first", manifest={"a": 1})
    tracker.ensure_run("run-1")
    run = tracker.run("run-1")
    assert run.name == "first" and run.manifest == {"a": 1}
    assert len(tracker.runs()) == 1


def test_logs_are_tailed_in_order(tracker):
    for index in range(5):
        tracker.log_text(f"line {index}", node_id="n1")
    tail = tracker.tail(limit=3)
    assert [item["text"] for item in tail] == ["line 2", "line 3", "line 4"]
    assert tail[0]["node_id"] == "n1"


def test_writes_from_many_threads_do_not_collide(tmp_path):
    """학습 곡선 폴링과 스칼라 적재는 hub 스레드풀에서 겹친다 - 같은 연결이라도 버텨야 한다."""
    import threading

    tracker = Tracker(tmp_path / "runs.db")
    tracker.ensure_run("r1")
    errors: list[Exception] = []

    def hammer(offset: int) -> None:
        try:
            for step in range(40):
                tracker.log("r1", offset * 100 + step, {"loss": 1.0})
                tracker.curve("r1", "loss")
                tracker.runs()
        except Exception as exc:   # noqa: BLE001 - 무엇이 터지든 여기서 잡아야 보인다
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(tracker.curve("r1", "loss")) == 160
