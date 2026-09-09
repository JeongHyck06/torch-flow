"""L2 학습 워커: Smoke 재현성과 체크포인트 재개 (기획서 §5.7.2, §10.4)."""

import json
import math
from pathlib import Path

import pytest

from conftest import MINIVIT
from torchflow import codegen
from torchflow.ir import load
from torchflow.worker.__main__ import train

pytest.importorskip("torch")


def make_job(tmp_path: Path, **overrides) -> tuple[dict, Path]:
    """그래프에서 뽑은 생성 코드를 그대로 학습 대상으로 쓴다 - hub가 하는 것과 같다."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    code = codegen.generate(load(MINIVIT), version="0.0.1")
    (run_dir / "model.py").write_text(code, encoding="utf-8")

    accepted = codegen.model_params(load(MINIVIT))
    model_args = {name: spec["default"] for name, spec in accepted.items()
                  if spec.get("default") is not None}
    model_args.setdefault("num_classes", 10)
    model_args["seed"] = 0

    job = {
        "code": str(run_dir / "model.py"),
        "class_name": "MiniViT",
        "model_args": model_args,
        "input_shape": ["B", 3, 32, 32],
        "num_classes": 10,
        "batch": 4, "steps": 4, "lr": 1e-3, "log_every": 1,
        "seed": 0, "device": "cpu", "smoke": True,
        **overrides,
    }
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return job, run_dir


def events(run_dir: Path, kind: str) -> list[dict]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [event for event in map(json.loads, lines) if event["kind"] == kind]


def losses(run_dir: Path) -> list[float]:
    return [event["loss"] for event in events(run_dir, "scalar")]


def test_smoke_is_bitwise_reproducible(tmp_path):
    """같은 시드의 Smoke 두 번은 loss가 bitwise로 같다(§10.4 CI 3단계).

    이게 깨지면 재현성 약속 전체가 깨진다 - 화면에서 본 곡선과 스크립트의 곡선이
    다른 실험의 것이 된다.
    """
    first, first_dir = make_job(tmp_path / "a")
    second, second_dir = make_job(tmp_path / "b")

    train(first, first_dir)
    train(second, second_dir)

    assert losses(first_dir) and losses(first_dir) == losses(second_dir)
    # float 비교가 아니라 바이트 비교여야 한다 - 근사 동일은 이 게이트를 통과시키면 안 된다.
    assert [f.hex() for f in losses(first_dir)] == [f.hex() for f in losses(second_dir)]


def test_resume_continues_the_same_trajectory(tmp_path):
    """4 step을 한 번에 돌린 것과, 2 step 뒤 ckpt에서 재개한 것이 같아야 한다.

    가중치·옵티마이저·RNG·데이터 스트림 중 하나라도 빠지면 여기서 갈라진다.
    """
    whole, whole_dir = make_job(tmp_path / "whole", steps=4)
    train(whole, whole_dir)

    half, half_dir = make_job(tmp_path / "half", steps=2)
    train(half, half_dir)
    assert (half_dir / "ckpt.pt").exists()

    rest = {**half, "steps": 4, "resume": str(half_dir / "ckpt.pt")}
    train(rest, half_dir)

    assert losses(half_dir) == losses(whole_dir)
    assert [event["step"] for event in events(half_dir, "scalar")] == [1, 2, 3, 4]


@pytest.mark.parametrize("scheduler", ["none", "cosine"])
def test_a_fork_resumes_with_the_new_hparam(tmp_path, scheduler):
    """갈라진 run의 lr은 ckpt에 저장된 lr을 이겨야 한다.

    optimizer.load_state_dict가 param_groups를, scheduler.load_state_dict가 base_lrs를
    통째로 되돌리기 때문에, 그냥 두면 "lr을 바꾸려고 갈라낸 run"이 옛 lr로 돈다.
    스케줄러가 있으면 job의 lr은 실제 lr이 아니라 base다.
    """
    job, run_dir = make_job(tmp_path, steps=2, lr=1e-3, scheduler=scheduler)
    train(job, run_dir)

    forked = {**job, "steps": 4, "lr": 2e-4, "resume": str(run_dir / "ckpt.pt")}
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    train(forked, child_dir)

    if scheduler == "none":
        assert {event["lr"] for event in events(child_dir, "scalar")} == {2e-4}
    else:
        assert {event["base_lr"] for event in events(child_dir, "scalar")} == {2e-4}
        # 실제 lr은 base를 넘지 않는다 - 스케줄이 곱해질 뿐이다.
        assert max(lrs(child_dir)) <= 2e-4


def lrs(run_dir: Path) -> list[float]:
    return [event["lr"] for event in events(run_dir, "scalar")]


def test_cosine_decays_and_warms_up(tmp_path):
    job, run_dir = make_job(tmp_path, steps=8, lr=1e-3, scheduler="cosine", warmup_steps=2)
    train(job, run_dir)

    schedule = lrs(run_dir)
    assert schedule[0] < schedule[1], "warmup 구간은 올라가야 한다"
    assert schedule[2] > schedule[-1], "warmup 뒤에는 내려가야 한다"
    assert schedule[0] == pytest.approx(1e-3 * 2 / 3), "warmup 첫 스텝은 base x 2/3"


def test_changing_lr_survives_the_scheduler(tmp_path):
    """스케줄러가 돌 때 lr 변경은 base로 들어가야 산다.

    param_groups에 직접 쓰면 다음 scheduler.step()이 base_lrs에서 다시 계산해
    덮어쓴다. 그러면 사람이 바꾼 lr이 한 스텝만 반영되고 조용히 사라진다.
    """
    plain, plain_dir = make_job(tmp_path / "plain", steps=6, lr=1e-3, scheduler="cosine")
    train(plain, plain_dir)

    scaled, scaled_dir = make_job(tmp_path / "scaled", steps=6, lr=1e-3, scheduler="cosine")
    # 워커는 매 스텝 control.json을 읽는다 - 시작 전에 써 두면 첫 스텝부터 반영된다.
    (scaled_dir / "control.json").write_text(json.dumps({"lr": 5e-4}), encoding="utf-8")
    train(scaled, scaled_dir)

    # base가 절반이면 모든 스텝의 lr이 정확히 절반이다 - 스케줄 모양은 그대로다.
    assert lrs(scaled_dir) == pytest.approx([value / 2 for value in lrs(plain_dir)])
    changes = events(scaled_dir, "hparam")
    assert len(changes) == 1 and changes[0]["value"] == 5e-4


def cosine(step: int, total: int = 6, warmup: int = 2) -> float:
    if step < warmup:
        return (step + 1) / (warmup + 1)
    return 0.5 * (1 + math.cos(math.pi * min((step - warmup) / (total - warmup), 1.0)))


def test_resume_keeps_its_place_in_the_schedule(tmp_path):
    """재개한 run은 스케줄의 **지금 자리**에서 이어야 한다.

    ckpt의 last_epoch를 복원하는 것만으로는 부족하다. 3 step짜리 코사인을 6 step으로
    늘려 재개하면 스케줄의 모양 자체가 달라지므로, ckpt에 박힌 lr이 아니라 새 스케줄이
    지금 자리에서 말하는 값으로 시작해야 한다 - 안 그러면 첫 스텝이 lr 0으로 돈다.
    """
    job, run_dir = make_job(tmp_path, steps=3, lr=1e-3, scheduler="cosine", warmup_steps=2)
    train(job, run_dir)
    train({**job, "steps": 6, "resume": str(run_dir / "ckpt.pt")}, run_dir)

    assert [event["step"] for event in events(run_dir, "scalar")] == [1, 2, 3, 4, 5, 6]
    assert lrs(run_dir)[3:] == pytest.approx([1e-3 * cosine(step) for step in (4, 5, 6)])


def test_a_stopped_run_leaves_a_checkpoint(tmp_path):
    job, run_dir = make_job(tmp_path, steps=10)
    (run_dir / "control.json").write_text(json.dumps({"stop": True}), encoding="utf-8")

    train(job, run_dir)

    assert [event["state"] for event in events(run_dir, "status")][-1] == "stopped"
    # 존재만으로는 부족하다 - 쓰다 만 파일도 존재한다. 실제로 열려야 fork가 산다.
    import torch

    assert set(torch.load(run_dir / "ckpt.pt")) == {"step", "model", "optimizer",
                                                    "scheduler", "rng", "data_rng"}


def make_mnist_job(tmp_path: Path) -> tuple[dict, Path]:
    """가짜 MNIST 위의 한 층짜리 분류기. 데이터셋 경로를 끝까지 타는 가장 작은 job."""
    from test_datasets import write_fake_mnist
    from torchflow.ir import ModuleGraph

    write_fake_mnist(tmp_path / "data")
    ir = ModuleGraph.model_validate({"graph": {
        "name": "Tiny",
        "instances": {"i1": {"label": "fc", "type": "torch.nn.Linear",
                             "args": {"in_features": 784, "out_features": 10}}},
        "nodes": [
            {"id": "n0", "label": "x", "type": "torchflow.Input",
             "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", 1, 28, 28],
                            "dtype": "float32"}]},
            {"id": "n1", "label": "flat", "type": "torch.flatten", "args": {"start_dim": 1}},
            {"id": "n2", "label": "fc", "call": "i1", "method": "forward"},
            {"id": "n3", "label": "out", "type": "torchflow.Output"}],
        "edges": [["n0.x", "n1.input"], ["n1.output", "n2.input"], ["n2.output", "n3.input"]],
    }})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model.py").write_text(codegen.generate(ir, version="0.0.1"), encoding="utf-8")
    job = {"code": str(run_dir / "model.py"), "class_name": "Tiny", "model_args": {"seed": 0},
           "input_shape": ["B", 1, 28, 28], "num_classes": 10,
           "dataset": "mnist", "data_dir": str(tmp_path / "data"), "eval_every": 2,
           "batch": 4, "steps": 4, "lr": 1e-2, "log_every": 1, "seed": 0, "device": "cpu"}
    return job, run_dir


def test_a_dataset_job_logs_accuracy_and_validation(tmp_path):
    """내장 데이터셋이면 train acc를 매 로그에, test 분할의 val_acc를 eval_every마다 적는다."""
    job, run_dir = make_mnist_job(tmp_path)
    train(job, run_dir)

    scalars = events(run_dir, "scalar")
    assert all("acc" in event for event in scalars if "loss" in event)
    assert [event["step"] for event in scalars if "val_acc" in event] == [2, 4]


def test_test_mode_scores_the_checkpoint_on_the_held_out_split(tmp_path):
    """--test는 ckpt를 학습이 보지 않은 분할에 돌려 정확도·혼동 행렬·샘플 그림을 남긴다."""
    from torchflow.worker.__main__ import test as run_test

    job, run_dir = make_mnist_job(tmp_path)
    train(job, run_dir)
    result = run_test(job, run_dir)

    assert result["ok"] and result["split"] == "test" and result["count"] == 8
    assert sum(map(sum, result["confusion"])) == 8 and len(result["per_class"]) == 10
    assert 0.0 <= result["acc"] <= 1.0 and result["samples"] and result["samples"][0]["png"]
    assert json.loads((run_dir / "test.json").read_text(encoding="utf-8"))["ok"]


def test_test_mode_makes_a_holdout_for_the_synthetic_task(tmp_path):
    """합성 과제는 test 분할이 없다 - 같은 teacher에 학습이 보지 않은 입력을 뽑아 잰다."""
    from torchflow.worker.__main__ import test as run_test

    job, run_dir = make_job(tmp_path, steps=2)
    train(job, run_dir)
    result = run_test(job, run_dir)
    assert result["ok"] and result["split"] == "합성 홀드아웃" and result["count"] == 1000


def test_a_crash_in_the_loop_leaves_a_failed_status(tmp_path):
    """shape 불일치처럼 첫 forward에서 죽어도 events.jsonl은 failed로 끝나야 hub가 안다."""
    job, run_dir = make_job(tmp_path, input_shape=["B", 2, 32, 32])    # 3채널 모델에 2채널 입력

    with pytest.raises(RuntimeError):
        train(job, run_dir)

    assert [event["state"] for event in events(run_dir, "status")][-1] == "failed"
    error = events(run_dir, "error")[-1]
    assert error["stage"] == "train" and "RuntimeError" in error["message"]


def write_linear_csv(base: Path, rows: int = 240) -> Path:
    """정답이 특징의 선형 결합인 표 데이터. 회귀가 실제로 배우는지 보려면 배울 것이 있어야 한다."""
    import random

    rng = random.Random(0)
    target = base / "homes"
    target.mkdir(parents=True, exist_ok=True)
    lines = ["area,rooms,price"]
    for _ in range(rows):
        area = rng.uniform(20, 200)
        rooms = rng.randint(1, 6)
        lines.append(f"{area:.2f},{rooms},{3.2 * area + 12.0 * rooms + rng.gauss(0, 2):.2f}")
    (target / "samples.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base


def make_regression_job(tmp_path: Path, **overrides) -> tuple[dict, Path]:
    from torchflow.ir import ModuleGraph

    write_linear_csv(tmp_path / "data")
    ir = ModuleGraph.model_validate({"graph": {
        "name": "Reg",
        "instances": {"i1": {"label": "fc", "type": "torch.nn.Linear",
                             "args": {"in_features": 2, "out_features": 1}}},
        "nodes": [
            {"id": "n0", "label": "x", "type": "torchflow.Input",
             "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", 2], "dtype": "float32"}]},
            {"id": "n1", "label": "fc", "call": "i1", "method": "forward"},
            {"id": "n2", "label": "out", "type": "torchflow.Output"}],
        "edges": [["n0.x", "n1.input"], ["n1.output", "n2.input"]],
    }})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model.py").write_text(codegen.generate(ir, version="0.0.1"), encoding="utf-8")
    job = {"code": str(run_dir / "model.py"), "class_name": "Reg", "model_args": {"seed": 0},
           "input_shape": ["B", 2], "task": "regression", "loss": "mse",
           "dataset": "homes", "data_dir": str(tmp_path / "data"), "eval_every": 50,
           "recipe": {"task": "regression", "label_column": "price",
                      "features": ["area", "rooms"], "normalize": "standard",
                      "target_normalize": "standard", "missing": "drop", "onehot": True,
                      "val_fraction": 0.2, "seed": 0},
           "batch": 32, "steps": 300, "lr": 1e-2, "log_every": 50, "seed": 0, "device": "cpu",
           **overrides}
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return job, run_dir


def test_a_regression_job_learns_and_reports_in_original_units(tmp_path):
    """회귀가 끝까지 돈다. 지표는 표준화 이전 단위여야 사람이 읽을 수 있다."""
    job, run_dir = make_regression_job(tmp_path)
    train(job, run_dir)

    scalars = events(run_dir, "scalar")
    validations = [event for event in scalars if "val_rmse" in event]
    assert validations, "회귀는 val_rmse를 적어야 한다"
    # 분류 지표는 나오면 안 된다 - 정확도는 회귀에서 뜻이 없다.
    assert not any("val_acc" in event for event in scalars)
    assert all("acc" not in event for event in scalars)

    # 정답 범위가 100단위인데 지표가 1 미만이면 표준화한 값을 그대로 보고한 것이다.
    assert validations[-1]["val_rmse"] > 1.0
    # 노이즈 표준편차가 2이므로 배웠다면 그 근처로 내려온다. 안 배웠으면 정답의 표준편차(~170)다.
    assert validations[-1]["val_rmse"] < 20.0, validations[-1]


def test_the_regression_test_split_reports_error_not_a_confusion_matrix(tmp_path):
    """`5 테스트`가 회귀에서는 혼동 행렬 대신 오차와 가장 크게 틀린 행을 준다."""
    from torchflow.worker.__main__ import test as run_test

    job, run_dir = make_regression_job(tmp_path)
    train(job, run_dir)
    result = run_test(job, run_dir)

    assert result["task"] == "regression"
    assert "confusion" not in result and "per_class" not in result
    assert result["rmse"] > 1.0 and result["r2"] > 0.9
    assert len(result["scatter"]) == result["count"]
    worst = result["worst"][0]
    assert abs(worst["truth"] - worst["pred"]) == pytest.approx(abs(worst["error"]), abs=1e-4)
    # 가장 크게 틀린 것부터 내려와야 한다.
    assert abs(result["worst"][0]["error"]) >= abs(result["worst"][-1]["error"])
