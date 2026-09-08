"""블록 상세 (기획서 §6.3): 마지막 probe가 남긴 값으로 선택한 블록이 무엇을 했는지 그린다.

L1 패스가 들고 있는 입력(``node_inputs``)·출력(``activations``)·모듈을 읽기만 한다.
모델을 다시 돌리지 않는다. 그림은 PIL이 있을 때만 - 없으면 히스토그램과 숫자로.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from ..ir import ModuleGraph

BINS = 48
MAX_TILES = 16
TILE = 32


def describe_node(pass_, key: str, node_id: str) -> dict[str, Any]:
    torch = pass_.torch
    node, instance = _find(pass_.ir, node_id)
    if node is not None and node.type == "torchflow.Train":
        # 학습 블록은 probe에 실리지 않는다. 설정을 말로 풀어 준다.
        return {"ok": True, "node": node_id, "label": node.label, "kind": "torchflow.Train",
                "params": 0, "input_shape": None, "output_shape": None,
                "explain": _explain_train(node.args or {}), "panels": []}
    output = pass_.activations.get(key)
    if output is None:
        return {"ok": False, "error": "이 블록은 마지막 probe에서 실행되지 않았습니다. Probe를 다시 돌려 주세요"}
    inputs = pass_.node_inputs.get(key) or {}
    x = next((value for value in inputs.values() if torch.is_tensor(value)), None)
    module_key = pass_.node_modules.get(key)
    module = pass_.session.modules.get(module_key) if module_key else None
    kind = (instance.type if instance else (node.type if node else "?")).split("@")[0]
    short = kind.rsplit(".", 1)[-1]
    params = sum(p.numel() for p in module.parameters()) if module is not None and hasattr(module, "parameters") else 0

    panels: list[dict[str, Any]] = []
    if x is not None and x.dim() == 4 and short not in ("Input",):
        panels.append(_grid("입력 채널", x[0], "첫 샘플, 채널마다 정규화"))
    if output.dim() == 4:
        panels.append(_grid("출력 채널" if short != "Input" else "입력 샘플", output[0],
                            f"{output.shape[1]}채널 중 {min(output.shape[1], MAX_TILES)}"))
    if module is not None and hasattr(module, "weight") and torch.is_tensor(module.weight):
        weight = module.weight.detach()
        if weight.dim() == 4:
            panels.append(_grid("필터 가중치", weight[:, 0], f"{weight.shape[0]}개 중 앞 {min(weight.shape[0], MAX_TILES)}, 첫 입력 채널"))
        elif weight.dim() == 2:
            panels.append(_heatmap("가중치 [out, in]", weight))
    if x is not None and short in ("ReLU", "GELU", "SiLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
                                   "BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm", "RMSNorm", "Dropout",
                                   "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "MaxPool1d", "AvgPool1d"):
        panels.append(_hist_pair("전후 값 분포", x, output))
    if short in ("BatchNorm2d", "BatchNorm1d", "LayerNorm", "GroupNorm") and x is not None:
        panels.append(_channel_stats("채널별 평균 (전 → 후)", x, output, "mean"))
        panels.append(_channel_stats("채널별 표준편차 (전 → 후)", x, output, "std"))
    if output.dim() == 2 and output.shape[1] <= 32 and short in ("Linear", "Output"):
        values = output[0].detach().float().tolist()
        panels.append({"type": "bars", "title": "첫 샘플 출력" + (" (로짓)" if short == "Output" or output.shape[1] <= 10 else ""),
                       "labels": [str(index) for index in range(len(values))], "values": values})
    if not panels or output.dim() <= 2 and short not in ("Linear", "Output"):
        panels.append(_hist_pair("출력 값 분포", None, output))

    return {"ok": True, "node": node_id, "label": node.label if node else node_id, "kind": kind,
            "params": params,
            "input_shape": list(x.shape) if x is not None else None, "output_shape": list(output.shape),
            "explain": _explain(short, module, x, output), "panels": panels}


def _find(ir: ModuleGraph, node_id: str):
    scopes = [ir.graph, *ir.composites.values()]
    for scope in scopes:
        for node in scope.nodes:
            if node.id == node_id:
                return node, (scope.instances.get(node.call) if node.call else None)
    return None, None


def _explain_train(args: dict[str, Any]) -> str:
    optimizer = str(args.get("optimizer", "adamw"))
    scheduler = str(args.get("scheduler", "none"))
    return (f"{optimizer} 옵티마이저로 lr {args.get('lr', 0.001)}, batch {args.get('batch', 32)}씩 "
            f"{args.get('steps', 500)} step 학습한다. 손실은 cross entropy, 검증은 eval 주기마다 val 분할로 잰다"
            + (f". 스케줄 {scheduler}" if scheduler != "none" else "")
            + ". 단계 표시줄의 4 학습이 이 값으로 워커를 띄운다")


def _explain(short: str, module, x, y) -> str:
    def hw(t):
        return f"{t.shape[-2]}×{t.shape[-1]}" if t is not None and t.dim() == 4 else "?"
    if short == "Conv2d":
        k = module.kernel_size[0]
        return (f"{k}×{k} 커널 {module.out_channels}개를 밀며 지역 패턴을 뽑는다. "
                f"{module.in_channels}채널 {hw(x)} → {module.out_channels}채널 {hw(y)}"
                + (f" (stride {module.stride[0]})" if module.stride[0] != 1 else "")
                + (f", padding {module.padding[0]}으로 가장자리를 지킨다" if module.padding[0] else ""))
    if short == "Linear":
        return (f"{module.in_features}차원을 {module.out_features}차원으로 선형 사상한다 "
                f"(가중치 {module.out_features}×{module.in_features}). 마지막 층이면 클래스 점수(로짓)다")
    if short == "ReLU":
        return "음수를 0으로 자른다. 분포의 왼쪽이 0에 쌓인 것이 보인다"
    if short in ("GELU", "SiLU"):
        return "음수를 부드럽게 억제한다. 0 근처가 살짝 음수로 남는다"
    if short in ("MaxPool2d", "MaxPool1d"):
        return f"{module.kernel_size}칸 창에서 최댓값만 남겨 해상도를 {hw(x)} → {hw(y)}로 줄인다. 위치는 대충, 존재는 확실히"
    if short in ("AvgPool2d", "AdaptiveAvgPool2d", "AvgPool1d", "AdaptiveAvgPool1d"):
        return f"창 안 평균으로 {hw(x)} → {hw(y)}. 전체를 요약한다"
    if short in ("Flatten", "flatten"):
        return f"격자 {list(x.shape[1:]) if x is not None else '?'}를 길이 {y.shape[-1]} 벡터로 편다. 값은 그대로, 모양만 바뀐다"
    if short in ("BatchNorm2d", "BatchNorm1d"):
        return "채널마다 배치 평균을 빼고 표준편차로 나눠 스케일을 맞춘다. 깊은 망이 흔들리지 않게"
    if short == "LayerNorm":
        return "샘플(토큰)마다 특징 평균 0, 표준편차 1로 맞춘다"
    if short == "Dropout":
        return f"학습 때 {module.p:.0%}를 무작위로 끈다. probe는 eval 모드라 지금은 그대로 지나간다"
    if short == "MultiheadAttention":
        return "토큰들이 서로를 얼마나 볼지(어텐션) 정해 섞는다"
    if short == "Embedding":
        return "정수 id를 학습되는 벡터로 바꾼다"
    if short in ("LSTM", "GRU"):
        return "순서대로 읽으며 상태를 갱신한다"
    if short == "Input":
        return "데이터가 들어오는 자리. 규격이 곧 데이터셋의 모양이다"
    if short == "Output":
        return "모델의 출력. 손실은 여기에 걸린다"
    if short in ("Softmax", "LogSoftmax"):
        return "점수를 확률(합 1)로 바꾼다"
    if short == "mean":
        return "지정한 축을 평균으로 접는다"
    return f"{short}"


def _to_uint8(tile):
    lo, hi = float(tile.min()), float(tile.max())
    scaled = (tile - lo) / (hi - lo) if hi > lo else tile * 0
    return (scaled * 255).to("cpu").to(dtype=__import__("torch").uint8)


def _grid(title: str, channels, note: str) -> dict[str, Any]:
    """[C, H, W] -> 채널 타일 격자 PNG. 타일마다 정규화한다."""
    import torch

    channels = channels.detach().float()
    count = min(channels.shape[0], MAX_TILES)
    cols = min(count, 4)
    rows = (count + cols - 1) // cols
    try:
        from PIL import Image
    except ImportError:
        return _hist_pair(title, None, channels)
    sheet = Image.new("L", (cols * (TILE + 1), rows * (TILE + 1)), 255)
    for index in range(count):
        tile = Image.fromarray(_to_uint8(channels[index]).numpy(), mode="L").resize((TILE, TILE), Image.NEAREST)
        sheet.paste(tile, ((index % cols) * (TILE + 1), (index // cols) * (TILE + 1)))
    return {"type": "grid", "title": title, "png": _png(sheet), "cols": cols, "rows": rows, "note": note}


def _heatmap(title: str, weight) -> dict[str, Any]:
    import torch

    try:
        from PIL import Image
    except ImportError:
        return _hist_pair(title, None, weight)
    small = torch.nn.functional.adaptive_avg_pool2d(weight.detach().float()[None, None], (min(weight.shape[0], 64), min(weight.shape[1], 64)))[0, 0]
    image = Image.fromarray(_to_uint8(small).numpy(), mode="L")
    return {"type": "heatmap", "title": title, "png": _png(image), "shape": list(weight.shape),
            "note": f"{weight.shape[0]}×{weight.shape[1]}" + (" (축약)" if small.shape != weight.shape else "")}


def _png(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _hist_pair(title: str, before, after) -> dict[str, Any]:
    import torch

    tensors = [t.detach().float().flatten().cpu() for t in (before, after) if t is not None]
    finite = torch.cat([t[torch.isfinite(t)] for t in tensors])
    lo, hi = (float(finite.min()), float(finite.max())) if finite.numel() else (0.0, 1.0)
    if lo == hi:
        hi = lo + 1e-6
    hist = lambda t: [int(v) for v in torch.histc(t[torch.isfinite(t)], bins=BINS, min=lo, max=hi)]
    return {"type": "hist", "title": title, "range": [lo, hi],
            "before": hist(tensors[0]) if before is not None else None, "after": hist(tensors[-1])}


def _channel_stats(title: str, before, after, what: str) -> dict[str, Any]:
    def per_channel(t):
        t = t.detach().float()
        flat = t.transpose(0, 1).flatten(1) if t.dim() == 4 else t.reshape(-1, t.shape[-1]).t()
        stat = flat.mean(dim=1) if what == "mean" else flat.std(dim=1)
        return stat[:32].cpu().tolist()
    b, a = per_channel(before), per_channel(after)
    return {"type": "bars", "title": title, "labels": [str(i) for i in range(len(a))], "values": a, "before": b}
