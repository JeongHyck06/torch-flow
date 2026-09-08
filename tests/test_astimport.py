"""코드 -> 그래프, AST 경로 (기획서 §7.4 경로 1, §7.5).

가장 중요한 계약은 왕복이다: 생성한 코드를 다시 읽으면 같은 코드가 나와야 한다.
그것이 성립해야 "코드가 아니라 IR이 정본"이라는 약속이 사용자에게 손해가 아니다.
"""

import re
import textwrap

import pytest

from conftest import MINIVIT
from torchflow import astimport, codegen, graphdiff
from torchflow.ir import ModuleGraph, load, validate

ROOT = MINIVIT.parent
IR_HASH = re.compile(r"ir_sha256=[0-9a-f]+")


def roundtrip(ir: ModuleGraph) -> tuple[str, str, dict]:
    code = codegen.generate(ir, version="1.0", source="graph.tfg.json")
    back, report = astimport.import_source(code)
    again = codegen.generate(back, version="1.0", source="graph.tfg.json")
    return IR_HASH.sub("", code), IR_HASH.sub("", again), report


@pytest.mark.parametrize("template", ["minivit", "resnet18", "mnist_cnn"])
def test_generated_code_reads_back_to_the_same_code(template):
    """§13.1 M6의 왕복 CI이자 M8의 structural >= 90% 게이트."""
    code, again, report = roundtrip(load(ROOT / f"{template}.tfg.json"))
    assert code == again
    assert report["structural_ratio"] == 1.0
    assert report["problems"] == []


def test_hand_written_code_becomes_a_working_graph():
    """사람이 쓴 코드: 위치 인자, 같은 모듈 두 번 호출, 중첩 호출, 계산이 섞인 인자."""
    source = textwrap.dedent('''
        import torch
        from torch import nn

        class SmallNet(nn.Module):
            def __init__(self, num_classes=10, width=32):
                super().__init__()
                self.conv1 = nn.Conv2d(3, width, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(width * 8 * 8, num_classes)

            def forward(self, x):
                h = self.pool(self.pool(self.conv1(x)))
                return self.fc(h.flatten(1))
    ''')
    ir, report = astimport.import_source(source)
    assert validate(ir) == []
    assert report["structural_ratio"] == 1.0

    instances = {instance.label: instance for instance in ir.graph.instances.values()}
    assert instances["conv1"].args == {"in_channels": 3, "out_channels": {"$hp": "width"},
                                       "kernel_size": 3, "padding": 1}
    assert instances["fc"].args["in_features"] == {"$expr": "hp.width * 8 * 8"}
    # 같은 self.pool을 두 번 불렀다 - 인스턴스는 하나, 호출 노드는 둘(§7.2 가중치 공유).
    pool = next(key for key, one in ir.graph.instances.items() if one.label == "pool")
    assert sum(1 for node in ir.graph.nodes if node.call == pool) == 2
    assert len(ir.graph.instances) == 3

    # h는 pool 출력의 다른 이름일 뿐이다 - 변수 이름은 블록 라벨에서 다시 나온다.
    code = codegen.generate(ir, version="1.0")
    assert "flatten = torch.flatten(pool2, start_dim=1)" in code


def test_a_forward_it_cannot_read_becomes_a_cell():
    """문장 하나를 버리는 대신 forward 전체를 CellModule로 올린다(§7.4.2)."""
    source = textwrap.dedent('''
        from torch import nn

        class Loopy(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(8)

            def forward(self, x):
                for _ in range(3):
                    x = self.norm(x)
                return x
    ''')
    ir, report = astimport.import_source(source)
    assert report["cell"] == 1 and report["problems"]
    assert list(ir.code_cells) == ["loopy_forward"]
    # 자식 모듈은 그래프에 남는다 - Cell 안에서 부르는 것이 무엇인지 보여야 한다.
    assert ir.code_cells["loopy_forward"].children_ports == ["norm"]
    assert any((node.type or "").startswith("cell:") for node in ir.graph.nodes)


def test_repeat_and_switch_survive_the_round_trip():
    ir = load(MINIVIT)
    back, _ = astimport.import_source(codegen.generate(ir, version="1.0"))
    kinds = {instance.type for scope in [back.graph, *back.composites.values()]
             for instance in scope.instances.values()}
    assert "torchflow.Repeat" in kinds
    assert "torchflow.Switch" in kinds
    switch = next(instance for scope in [back.graph, *back.composites.values()]
                  for instance in scope.instances.values()
                  if instance.type == "torchflow.Switch")
    assert set(switch.variants) == {"baseline", "none"}
    assert switch.active == {"$p": "attn_type"}


def test_node_ids_are_stable_so_a_second_import_keeps_identity():
    """§7.5: 같은 속성은 같은 id. 좌표·프로브·run 해시가 그 위에 얹혀 있다."""
    source = textwrap.dedent('''
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.ReLU()

            def forward(self, x):
                return self.b(self.a(x))
    ''')
    first, _ = astimport.import_source(source)
    grown, _ = astimport.import_source(source.replace(
        "self.b = nn.ReLU()", "self.b = nn.ReLU()\n        self.c = nn.Linear(4, 2)"))
    kept = {node.id for node in first.graph.nodes} & {node.id for node in grown.graph.nodes}
    assert len(kept) == len(first.graph.nodes), "블록을 더해도 원래 id는 그대로여야 한다"


def test_reimport_keeps_labels_probes_and_the_input_spec():
    source = textwrap.dedent('''
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)

            def forward(self, x):
                return self.a(x)
    ''')
    previous, _ = astimport.import_source(source)
    entry = next(node for node in previous.graph.nodes if node.type == "torchflow.Input")
    entry.ports_out[0].shape = ["B", 4]
    entry.ports_out[0].dtype = "float32"
    linear = next(node for node in previous.graph.nodes if node.call)
    linear.label = "사람이 붙인 이름"
    previous.graph.probe_objective = "loss"

    fresh, _ = astimport.import_source(
        source.replace("nn.Linear(4, 4)", "nn.Linear(4, 8)"), previous=previous)

    kept = next(node for node in fresh.graph.nodes if node.id == linear.id)
    assert kept.label == "사람이 붙인 이름"
    assert fresh.graph.probe_objective == "loss"
    entry_again = next(node for node in fresh.graph.nodes if node.type == "torchflow.Input")
    assert entry_again.ports_out[0].shape == ["B", 4]
    # 바뀐 것은 바뀐 대로 보여야 한다.
    assert any("out_features" in line for line in graphdiff.summarize(previous, fresh))


def test_user_slots_come_back_as_bytes():
    code = codegen.generate(load(MINIVIT), version="1.0")
    edited = code.replace("# tf: user-begin imports_helpers\n",
                          "# tf: user-begin imports_helpers\nimport math  # 내 코드\n")
    ir, _ = astimport.import_source(edited)
    assert ir.graph.user_code["imports_helpers"] == "import math  # 내 코드"
    assert "import math  # 내 코드" in codegen.generate(ir, version="1.0")


def test_a_file_without_modules_is_refused():
    with pytest.raises(astimport.AstImportError, match="nn.Module"):
        astimport.import_source("x = 1\n")
    with pytest.raises(astimport.AstImportError, match="문법"):
        astimport.import_source("class (:\n")
