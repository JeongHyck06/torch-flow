"""L2 학습 워커: Smoke 재현성과 체크포인트 재개 (기획서 §5.7.2, §10.4)."""

import json
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


def test_a_fork_resumes_with_the_new_hparam(tmp_path):
    """갈라진 run의 lr은 ckpt에 저장된 lr을 이겨야 한다.

    optimizer.load_state_dict가 param_groups를 통째로 되돌리기 때문에, 그냥 두면
    "lr을 바꾸려고 갈라낸 run"이 옛 lr로 돈다.
    """
    job, run_dir = make_job(tmp_path, steps=2, lr=1e-3)
    train(job, run_dir)

    forked = {**job, "steps": 4, "lr": 2e-4, "resume": str(run_dir / "ckpt.pt")}
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    train(forked, child_dir)

    assert {event["lr"] for event in events(child_dir, "scalar")} == {2e-4}


def test_a_stopped_run_leaves_a_checkpoint(tmp_path):
    job, run_dir = make_job(tmp_path, steps=10)
    (run_dir / "control.json").write_text(json.dumps({"stop": True}), encoding="utf-8")

    train(job, run_dir)

    assert [event["state"] for event in events(run_dir, "status")][-1] == "stopped"
    # 존재만으로는 부족하다 - 쓰다 만 파일도 존재한다. 실제로 열려야 fork가 산다.
    import torch

    assert set(torch.load(run_dir / "ckpt.pt")) == {"step", "model", "optimizer",
                                                    "rng", "data_rng"}
