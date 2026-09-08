"""`package` 단위 codegen과 리뷰 계약 (기획서 §7.3, §10.2, §13.1 M6).

여기서 지키는 약속은 하나다: **나온 코드는 torchflow 없이 돈다.**
"""

import json
import subprocess
import sys

import pytest

from conftest import MINIVIT
from torchflow import codegen, graphdiff, package
from torchflow.ir import load


@pytest.fixture
def built(tmp_path):
    ir = load(MINIVIT)
    out = tmp_path / "project"
    files = package.build(ir, out, version="0.1.0", source=str(MINIVIT), graph_path=MINIVIT)
    return ir, out, files


def test_the_package_has_everything_needed_to_run(built):
    _, out, files = built
    names = {path.relative_to(out).as_posix() for path in files}
    assert {"src/minivit/model.py", "src/minivit/train.py", "src/minivit/utils.py",
            "src/minivit/data.py", "conf/config.json", "reproduce.sh",
            "pyproject.toml", "env.lock.json"} <= names
    assert (out / "reproduce.sh").stat().st_mode & 0o111, "reproduce.sh는 실행 가능해야 한다"


def test_no_generated_file_imports_torchflow(built):
    """§13.1 M6 게이트. 이 줄이 깨지면 생성 코드는 남의 컴퓨터에서 안 돈다."""
    _, out, files = built
    for path in files:
        if path.suffix == ".py":
            assert not package.IMPORTS_TORCHFLOW.search(path.read_text(encoding="utf-8")), path


def test_the_generated_package_trains(built, tmp_path):
    pytest.importorskip("torch")
    _, out, _ = built
    done = subprocess.run(
        [sys.executable, "-m", "minivit.train", "--steps", "4", "--batch", "2",
         "--log_every", "1", "--device", "cpu", "--dataset", "teacher",
         "--run_dir", str(tmp_path / "run")],
        cwd=out, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src", "HOME": str(tmp_path)},
        capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    steps = [json.loads(line)["step"]
             for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4]
    assert (tmp_path / "run" / "last.pt").exists()


def test_step_zero_weights_match_the_single_file_output(built):
    """같은 그래프에서 나온 두 단위는 **같은 가중치**에서 출발한다(§10.4 CI 2단계)."""
    torch = pytest.importorskip("torch")
    ir, out, _ = built

    def weights(code: str, module_name: str) -> dict:
        namespace: dict = {}
        exec(compile(code, module_name, "exec"), namespace)  # noqa: S102 - 방금 생성한 코드다
        model = namespace[codegen.class_name(ir)](num_classes=10, seed=0)
        return model.state_dict()

    single = weights(codegen.generate(ir, version="0.1.0", source=str(MINIVIT)), "single")
    # package의 model.py는 seeded_init을 utils에서 가져온다 - 소스는 같은 파일이다.
    package_code = (out / "src" / "minivit" / "model.py").read_text(encoding="utf-8")
    utils = (out / "src" / "minivit" / "utils.py").read_text(encoding="utf-8")
    utils = utils.replace("from __future__ import annotations", "")   # 파일 중간에 올 수 없다
    package_code = package_code.replace("from .utils import seeded_init", utils)
    both = weights(package_code, "package")
    assert single.keys() == both.keys()
    for name, tensor in single.items():
        assert torch.equal(tensor, both[name]), name


def test_check_passes_and_then_catches_a_hand_edit(built):
    _, out, _ = built
    assert package.check(out) == []
    model = out / "src" / "minivit" / "model.py"
    model.write_text(model.read_text(encoding="utf-8") + "# 손댐\n", encoding="utf-8")
    assert any("손으로" in problem for problem in package.check(out))


def test_check_notices_a_graph_that_moved_on(built):
    ir, out, _ = built
    ir.graph.nodes[0].label = "renamed"
    from torchflow.ir import save

    save(ir, MINIVIT.parent / "tmp.tfg.json")
    recorded = json.loads((out / package.SOURCEMAP).read_text(encoding="utf-8"))
    recorded["graph"] = str((MINIVIT.parent / "tmp.tfg.json").resolve())
    (out / package.SOURCEMAP).write_text(json.dumps(recorded), encoding="utf-8")
    try:
        assert any("바뀌었습니다" in problem for problem in package.check(out))
    finally:
        (MINIVIT.parent / "tmp.tfg.json").unlink()


def test_overrides_from_a_run_replay_into_the_optimizer(built):
    """hub가 남긴 conf/overrides/<run>.yaml을 생성 코드가 그대로 읽는다(§5.7.2)."""
    torch = pytest.importorskip("torch")
    _, out, _ = built
    namespace: dict = {}
    exec(compile((out / "src" / "minivit" / "utils.py").read_text(encoding="utf-8"),
                 "utils", "exec"), namespace)  # noqa: S102

    from torchflow.hub import runs as l2

    path = l2.write_overrides("run-1", [{"step": 2, "path": "optim.lr", "value": 0.0005}], out)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = namespace["warmup_cosine"](optimizer, 0, 10)
    overrides = namespace["StepOverrides"].load(path, optimizer, scheduler)

    overrides.step(1)
    assert scheduler.base_lrs == [0.001]
    overrides.step(2)
    assert scheduler.base_lrs == pytest.approx([0.0005])


def test_diff_reads_a_changed_parameter():
    before = load(MINIVIT)
    after = load(MINIVIT)
    instance = next(iter(after.graph.instances))
    after.graph.instances[instance].args["dim"] = 999
    label = after.graph.instances[instance].label
    assert any(f"{label}.dim" in line for line in graphdiff.summarize(before, after))
    assert graphdiff.summarize(before, before) == []
