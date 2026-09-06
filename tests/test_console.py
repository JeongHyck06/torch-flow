"""노드별 출력 캡처와 Debug Console (기획서 §5.6.1)."""

import pytest

torch = pytest.importorskip("torch")

from torchflow.ir import ModuleGraph  # noqa: E402
from torchflow.kernel.console import evaluate  # noqa: E402
from torchflow.kernel.l0 import L0Pass  # noqa: E402
from torchflow.kernel.l1 import L1Pass, ProbeConfig  # noqa: E402

RT = {"num_classes": 10}

PRINTER = """
def emit(x, tag):
    print(f"forward {tag}")
    return x
"""


@pytest.fixture
def printer(tmp_path, monkeypatch):
    (tmp_path / "tf_printer.py").write_text(PRINTER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "tf_printer.emit"


def two_printers(kind: str) -> ModuleGraph:
    return ModuleGraph.model_validate({
        "graph": {
            "name": "printers",
            "nodes": [
                {"label": "x", "id": "in", "type": "torchflow.Input",
                 "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", 4],
                                "dtype": "float32"}]},
                {"label": "first", "id": "one", "type": kind, "args": {"tag": "one"}},
                {"label": "second", "id": "two", "type": kind, "args": {"tag": "two"}},
                {"label": "out", "id": "out", "type": "torchflow.Output"},
            ],
            "edges": [["in.x", "one.input"], ["one.output", "two.input"],
                      ["two.output", "out.input"]],
        },
    })


def test_stdout_is_attributed_to_the_node_that_printed(printer):
    result = L0Pass(two_printers(printer)).run()

    assert result.ok
    captured = [entry for entry in result.captured if entry["stream"] == "stdout"]
    assert [(entry["node_id"], entry["text"]) for entry in captured] == [
        ("one", "forward one"), ("two", "forward two")]


def test_capture_survives_a_failing_node(printer):
    ir = two_printers(printer)
    # 두 번째 노드가 받을 수 없는 인자를 준다 - 예외 직전까지 찍힌 것은 남아야 한다.
    ir.graph.nodes[2].args = {"tag": "two", "nope": 1}
    result = L0Pass(ir).run()

    assert not result.ok
    assert [entry["node_id"] for entry in result.captured] == ["one"]


def test_console_reads_the_last_probe_values(minivit):
    pass_ = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2))
    result = pass_.probe_once()
    assert result.ok

    key = next(key for key, node, _ in pass_.node_order if node.label == "pool")
    assert evaluate(pass_, key, "y.shape[0]") == {"ok": True, "text": "2", "spec": None}
    assert evaluate(pass_, key, "x.dim()")["text"] == "3"       # pool 입력은 [B, L, dim]
    assert evaluate(pass_, key, "batch['x'].shape[1]")["text"] == "3"

    tensor = evaluate(pass_, key, "y")
    assert tensor["spec"]["shape"] == [2, 192]
    assert "mean" in tensor["text"]


def test_console_binds_the_module_instance(minivit):
    pass_ = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2))
    pass_.probe_once()

    key = next(key for key, node, _ in pass_.node_order if node.label == "head")
    assert evaluate(pass_, key, "m.out_features")["text"] == "10"


def test_console_refuses_statements_and_reports_errors(minivit):
    pass_ = L1Pass(minivit, rt=RT, probe=ProbeConfig(batch=2))
    pass_.probe_once()
    key = pass_.node_order[0][0]

    # Phase A는 읽기 전용 표현식이다(문장 실행은 v1).
    assert evaluate(pass_, key, "y = 1")["ok"] is False
    assert "표현식" in evaluate(pass_, key, "y = 1")["error"]

    failed = evaluate(pass_, key, "y.no_such_method()")
    assert failed["ok"] is False and "AttributeError" in failed["error"]
