"""ResNet-18 템플릿을 IR로 만든다 (기획서 §13.3 검증 템플릿 1번).

여덟 개의 BasicBlock은 stride와 shortcut이 제각각이라 하나의 Repeat으로 접히지
않는다. 대신 shortcut을 Ablation Switch로 두어 identity/conv 분기를 IR 안에서
표현한다 - codegen이 ``__init__``의 if 하나로 떨어지는 형태(§4.4.2).

    python examples/build_resnet18.py
"""

from __future__ import annotations

import json
from pathlib import Path

CONV = "torch.nn.Conv2d"
BN = "torch.nn.BatchNorm2d"


def conv_instance(iid, label, in_ch, out_ch, kernel, stride, padding):
    return iid, {
        "label": label, "type": CONV,
        "args": {"in_channels": in_ch, "out_channels": out_ch, "kernel_size": kernel,
                 "stride": stride, "padding": padding, "bias": False},
    }


def basic_block() -> dict:
    """BasicBlock: conv-bn-relu-conv-bn + shortcut + add + relu."""
    return {
        "doc": "ResNet BasicBlock with a switchable shortcut (identity / 1x1 conv).",
        "params": {"in_ch": {"type": "int"}, "out_ch": {"type": "int"},
                   "stride": {"type": "int"},
                   "shortcut": {"type": "enum", "choices": ["identity", "conv"]}},
        "ports": {"in": [{"name": "x", "type": "Tensor", "shape": ["B", "in_ch", "H", "W"]}],
                  "out": [{"name": "y", "type": "Tensor", "shape": ["B", "out_ch", "H", "W"]}]},
        "instances": {
            "01RB0001": {"label": "conv1", "type": CONV,
                         "args": {"in_channels": {"$p": "in_ch"}, "out_channels": {"$p": "out_ch"},
                                  "kernel_size": 3, "stride": {"$p": "stride"}, "padding": 1,
                                  "bias": False}},
            "01RB0002": {"label": "bn1", "type": BN, "args": {"num_features": {"$p": "out_ch"}}},
            "01RB0003": {"label": "relu1", "type": "torch.nn.ReLU", "args": {"inplace": False}},
            "01RB0004": {"label": "conv2", "type": CONV,
                         "args": {"in_channels": {"$p": "out_ch"}, "out_channels": {"$p": "out_ch"},
                                  "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}},
            "01RB0005": {"label": "bn2", "type": BN, "args": {"num_features": {"$p": "out_ch"}}},
            "01RB0006": {
                "label": "shortcut", "type": "torchflow.Switch", "active": {"$p": "shortcut"},
                "variants": {
                    "identity": {"type": "composite:IdentityShortcut", "label": "identity",
                                 "args": {"in_ch": {"$p": "in_ch"}, "out_ch": {"$p": "out_ch"},
                                          "stride": {"$p": "stride"}}},
                    "conv": {"type": "composite:ConvShortcut", "label": "1x1 conv",
                             "args": {"in_ch": {"$p": "in_ch"}, "out_ch": {"$p": "out_ch"},
                                      "stride": {"$p": "stride"}}},
                }},
            "01RB0007": {"label": "relu2", "type": "torch.nn.ReLU", "args": {"inplace": False}},
        },
        "nodes": [
            {"label": "conv1", "id": "01RN0001", "call": "01RB0001", "method": "forward"},
            {"label": "bn1", "id": "01RN0002", "call": "01RB0002", "method": "forward"},
            {"label": "relu1", "id": "01RN0003", "call": "01RB0003", "method": "forward"},
            {"label": "conv2", "id": "01RN0004", "call": "01RB0004", "method": "forward"},
            {"label": "bn2", "id": "01RN0005", "call": "01RB0005", "method": "forward"},
            {"label": "shortcut", "id": "01RN0006", "call": "01RB0006", "method": "forward",
             "ports_out": [{"name": "y", "type": "Tensor"}]},
            {"label": "add", "id": "01RN0007", "type": "torch.add"},
            {"label": "relu2", "id": "01RN0008", "call": "01RB0007", "method": "forward"},
        ],
        "edges": [
            ["$in.x", "01RN0001.input"], ["01RN0001.output", "01RN0002.input"],
            ["01RN0002.output", "01RN0003.input"], ["01RN0003.output", "01RN0004.input"],
            ["01RN0004.output", "01RN0005.input"],
            ["$in.x", "01RN0006.x"],
            ["01RN0005.output", "01RN0007.input"], ["01RN0006.y", "01RN0007.other"],
            ["01RN0007.output", "01RN0008.input"], ["01RN0008.output", "$out.y"],
        ],
    }


SHORTCUT_PARAMS = {"in_ch": {"type": "int"}, "out_ch": {"type": "int"}, "stride": {"type": "int"}}
SHORTCUT_PORTS = {"in": [{"name": "x", "type": "Tensor"}], "out": [{"name": "y", "type": "Tensor"}]}


def build() -> dict:
    stages = [  # (in_ch, out_ch, stride, shortcut)
        (64, 64, 1, "identity"), (64, 64, 1, "identity"),
        (64, 128, 2, "conv"), (128, 128, 1, "identity"),
        (128, 256, 2, "conv"), (256, 256, 1, "identity"),
        (256, 512, 2, "conv"), (512, 512, 1, "identity"),
    ]

    instances = dict([
        conv_instance("01RG0001", "conv1", 3, 64, 7, 2, 3),
    ])
    instances["01RG0002"] = {"label": "bn1", "type": BN, "args": {"num_features": 64}}
    instances["01RG0003"] = {"label": "relu", "type": "torch.nn.ReLU", "args": {"inplace": False}}
    instances["01RG0004"] = {"label": "maxpool", "type": "torch.nn.MaxPool2d",
                             "args": {"kernel_size": 3, "stride": 2, "padding": 1}}
    instances["01RG0005"] = {"label": "avgpool", "type": "torch.nn.AdaptiveAvgPool2d",
                             "args": {"output_size": 1}}
    instances["01RG0006"] = {"label": "fc", "type": "torch.nn.Linear",
                             "args": {"in_features": 512, "out_features": {"$expr": "rt.num_classes"}}}

    nodes = [
        {"label": "x", "id": "01RG1000", "type": "torchflow.Input",
         "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", 3, 224, 224],
                        "dtype": "float32"}]},
        {"label": "conv1", "id": "01RG1001", "call": "01RG0001", "method": "forward"},
        {"label": "bn1", "id": "01RG1002", "call": "01RG0002", "method": "forward"},
        {"label": "relu", "id": "01RG1003", "call": "01RG0003", "method": "forward"},
        {"label": "maxpool", "id": "01RG1004", "call": "01RG0004", "method": "forward"},
    ]
    edges = [
        ["01RG1000.x", "01RG1001.input"], ["01RG1001.output", "01RG1002.input"],
        ["01RG1002.output", "01RG1003.input"], ["01RG1003.output", "01RG1004.input"],
    ]

    previous = "01RG1004.output"
    for index, (in_ch, out_ch, stride, shortcut) in enumerate(stages):
        instance_id = f"01RG01{index:02d}"
        node_id = f"01RG11{index:02d}"
        instances[instance_id] = {
            "label": f"layer{index // 2 + 1}.{index % 2}", "type": "composite:BasicBlock",
            "args": {"in_ch": in_ch, "out_ch": out_ch, "stride": stride, "shortcut": shortcut},
        }
        nodes.append({"label": f"block{index + 1}", "id": node_id, "call": instance_id,
                      "method": "forward", "ports_out": [{"name": "y", "type": "Tensor"}]})
        edges.append([previous, f"{node_id}.x"])
        previous = f"{node_id}.y"

    nodes += [
        {"label": "avgpool", "id": "01RG1200", "call": "01RG0005", "method": "forward"},
        {"label": "flatten", "id": "01RG1201", "type": "torch.flatten", "args": {"start_dim": 1}},
        {"label": "fc", "id": "01RG1202", "call": "01RG0006", "method": "forward"},
        {"label": "logits", "id": "01RG1203", "type": "torchflow.Output"},
    ]
    edges += [
        [previous, "01RG1200.input"], ["01RG1200.output", "01RG1201.input"],
        ["01RG1201.output", "01RG1202.input"], ["01RG1202.output", "01RG1203.input"],
    ]

    return {
        "schema_version": "1.0.0",
        "meta": {"app_version": "0.0.1", "template": "resnet18",
                 "reference": "kuangliu/pytorch-cifar · torchvision.models.resnet18"},
        "hparams": {"init_std": {"type": "float", "default": 0.02}},
        "composites": {
            "BasicBlock": basic_block(),
            "IdentityShortcut": {
                "doc": "Shortcut variant that passes the input through unchanged.",
                "params": SHORTCUT_PARAMS, "ports": SHORTCUT_PORTS,
                "instances": {"01RS0001": {"label": "id", "type": "torch.nn.Identity"}},
                "nodes": [{"label": "id", "id": "01RS1001", "call": "01RS0001",
                           "method": "forward"}],
                "edges": [["$in.x", "01RS1001.input"], ["01RS1001.output", "$out.y"]],
            },
            "ConvShortcut": {
                "doc": "Shortcut variant that projects with a strided 1x1 conv + BN.",
                "params": SHORTCUT_PARAMS, "ports": SHORTCUT_PORTS,
                "instances": {
                    "01RS0002": {"label": "proj", "type": CONV,
                                 "args": {"in_channels": {"$p": "in_ch"},
                                          "out_channels": {"$p": "out_ch"}, "kernel_size": 1,
                                          "stride": {"$p": "stride"}, "bias": False}},
                    "01RS0003": {"label": "bn", "type": BN,
                                 "args": {"num_features": {"$p": "out_ch"}}},
                },
                "nodes": [
                    {"label": "proj", "id": "01RS1002", "call": "01RS0002", "method": "forward"},
                    {"label": "bn", "id": "01RS1003", "call": "01RS0003", "method": "forward"},
                ],
                "edges": [["$in.x", "01RS1002.input"], ["01RS1002.output", "01RS1003.input"],
                          ["01RS1003.output", "$out.y"]],
            },
        },
        "graph": {
            "name": "ResNet18",
            "instances": instances,
            "nodes": nodes,
            "edges": edges,
            "init": {"policy": "kaiming_normal"},
            "probe_objective": "auto",
        },
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("resnet18.tfg.json")
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
