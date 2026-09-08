"""ablation 표 (기획서 §6.4). 시드 접기와 최고값 굵게가 전부다."""

from dataclasses import dataclass

from torchflow import table as tables


@dataclass
class FakeRun:
    id: str
    kind: str
    name: str
    manifest: dict


def runs():
    return [
        FakeRun("r1", "reported", "net", {"variant": "baseline", "job": {"seed": 0}}),
        FakeRun("r2", "reported", "net", {"variant": "baseline", "job": {"seed": 1}}),
        FakeRun("r3", "reported", "net", {"variant": "no-attn", "job": {"seed": 0}}),
        FakeRun("r4", "exploratory", "net", {"variant": "half-depth", "job": {"seed": 0}}),
    ]


FINAL = {"r1": {"val_acc": 0.90}, "r2": {"val_acc": 0.92},
         "r3": {"val_acc": 0.81}, "r4": {"val_acc": 0.99}}


def test_seeds_fold_into_mean_and_std():
    built = tables.build(runs(), axis="variant", metrics=["val_acc"], final=FINAL)
    assert built.rows == ["baseline", "no-attn"], "exploratory는 기본으로 빠진다"
    cell = built.cells[("baseline", "val_acc")]
    assert cell.mean == 0.91
    assert round(cell.std, 4) == round(0.02 / (2 ** 0.5), 4)
    assert built.best("val_acc") == "baseline"


def test_loss_columns_pick_the_smallest():
    built = tables.build(runs(), axis="variant", metrics=["val_loss"],
                         final={"r1": {"val_loss": 1.0}, "r3": {"val_loss": 0.5}})
    assert built.best("val_loss") == "no-attn"


def test_latex_is_booktabs_with_the_best_in_bold():
    built = tables.build(runs(), axis="variant", metrics=["val_acc"], final=FINAL)
    latex = tables.to_latex(built, caption="Ablation")
    assert "\\toprule" in latex and "\\bottomrule" in latex
    assert "\\textbf{0.91 ± 0.01}" in latex
    assert "no-attn" in latex


def test_exploratory_runs_are_marked_when_included():
    built = tables.build(runs(), axis="variant", metrics=["val_acc"], final=FINAL,
                         include_exploratory=True)
    assert "half-depth" in built.rows and "half-depth" in built.exploratory
    assert "half-depth†" in tables.to_markdown(built)
    assert "$^\\dagger$" in tables.to_latex(built)


def test_an_axis_that_is_missing_falls_back_to_the_run_name():
    built = tables.build(runs(), axis="job.optimizer", metrics=["val_acc"], final=FINAL)
    assert built.rows == ["net"]
    assert "," in tables.to_csv(built)
