"""인스턴스 import (기획서 §7.4 경로 3) - 살아 있는 모듈에서 그래프를 뽑는다.

AST를 읽지 않는다. 이미 만들어진 ``nn.Module``에 forward hook을 걸고 한 번
돌려서, 실제로 실행된 것만 기록한다. 그래서 HF·timm·사용자 코드가 무엇으로
만들어졌든 상관없이 동작하고, 제어흐름도 실행된 경로 그대로 잡힌다.

한계는 정직하게 드러난다: 모듈 사이의 순수 함수 연산(``x + y``, ``x.view(...)``)은
hook에 걸리지 않으므로 그 자리의 엣지가 비어 있다. 노드는 다 보이고 연결만
끊긴다 - 없는 연결을 지어내지 않는다.

``ponytail: 평면 그래프 + 속성 경로 라벨. 계층을 컴포지트로 접는 것은 다음 단계.``
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

from ..ir import Graph, Instance, ModuleGraph, Node, Port


def _spec_of(torch, tensor) -> dict[str, Any] | None:
    if not torch.is_tensor(tensor):
        return None
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def _tensors(value):
    """PyTree에서 텐서만 평평하게 꺼낸다."""
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _module_args(module) -> dict[str, Any]:
    """``extra_repr``에서 표시용 인자를 건져 온다.

    생성 인자를 되살릴 방법은 없다(파이썬이 기억하지 않는다). Attach 모드의
    Inspector는 읽기 전용이므로 표시가 되면 충분하고, 편집이 필요하면
    ``Session.promote()``로 IR 모드에 넘긴다.
    """
    extra = module.extra_repr()
    if not extra:
        return {}
    # 이름 없는 앞쪽 값은 __init__ 인자 순서를 따른다 - torch.nn의 extra_repr 관례다.
    try:
        names = [name for name in inspect.signature(type(module).__init__).parameters
                 if name != "self"]
    except (TypeError, ValueError):
        names = []
    args: dict[str, Any] = {}
    for index, chunk in enumerate(_split_top_level(extra)):
        key, sep, value = chunk.partition("=")
        if not sep or not key.strip().isidentifier():
            key, value = (names[index] if index < len(names) else f"arg{index}"), chunk
        args[key.strip()] = _literal(value.strip())
    return args


def _split_top_level(text: str) -> list[str]:
    """최상위 쉼표에서만 자른다. ``kernel_size=(8, 8)`` 안의 쉼표는 지나간다."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def trace_module(model, example_input, *, name: str | None = None) -> ModuleGraph:
    """모듈 하나를 한 번 실행해 Module Graph IR을 만든다.

    ``example_input``은 ``dict``(키워드), ``tuple``(위치), 또는 텐서 하나.
    """
    import torch

    leaves = [(path, module) for path, module in model.named_modules()
              if not list(module.children())]
    handles = []
    order: list[tuple[str, Any, Any, Any]] = []   # (경로, 모듈, 입력, 출력)

    def make_hook(path, module):
        def hook(_module, inputs, output):
            order.append((path, module, inputs, output))
        return hook

    for path, module in leaves:
        handles.append(module.register_forward_hook(make_hook(path, module)))

    try:
        with torch.no_grad():
            if isinstance(example_input, dict):
                model(**example_input)
            elif isinstance(example_input, (list, tuple)):
                model(*example_input)
            else:
                model(example_input)
    finally:
        for handle in handles:
            handle.remove()

    return _build_ir(torch, model, example_input, order, name or type(model).__name__)


def _build_ir(torch, model, example_input, order, name: str) -> ModuleGraph:
    instances: dict[str, Instance] = {}
    nodes: list[Node] = []
    edges: list[tuple[str, str]] = []
    produced: dict[int, str] = {}   # 텐서 id -> "노드.포트"

    # 그래프 입력.
    input_ports = []
    if isinstance(example_input, dict):
        items = list(example_input.items())
    elif isinstance(example_input, (list, tuple)):
        items = [(f"arg{index}", value) for index, value in enumerate(example_input)]
    else:
        items = [("x", example_input)]
    for port_name, value in items:
        for tensor in _tensors(value):
            input_ports.append(Port(name=port_name, type="Tensor",
                                    shape=list(tensor.shape),
                                    dtype=str(tensor.dtype).removeprefix("torch.")))
            produced[id(tensor)] = f"tf-input.{port_name}"
            break
    nodes.append(Node(label="input", id="tf-input", type="torchflow.Input",
                      ports_out=input_ports))

    seen_modules: dict[int, str] = {}
    for index, (path, module, inputs, output) in enumerate(order):
        node_id = f"tf-{index:04d}"
        # 같은 모듈 객체를 두 번 부르면(`self.relu(...)` 두 줄) 호출 노드는 둘,
        # 인스턴스는 하나다 - IR의 인스턴스/호출 분리 그대로다(§7.1). 파라미터도
        # 한 번만 세어지고 캔버스는 ⟲ shared로 묶는다.
        instance_id = seen_modules.get(id(module))
        if instance_id is None:
            instance_id = f"tfi-{len(seen_modules):04d}"
            seen_modules[id(module)] = instance_id
            instances[instance_id] = Instance(
                label=path or type(module).__name__,
                type=f"torch.nn.{type(module).__name__}",
                args=_module_args(module),
            )
        node = Node(
            label=path or type(module).__name__,
            id=node_id,
            call=instance_id,
            method="forward",
        )
        # 실측 spec을 노드에 박아 둔다 - hub가 재실행 없이도 shape를 안다.
        node.measured = _spec_of(torch, next(iter(_tensors(output)), None))
        node.parent = path.rsplit(".", 1)[0] if "." in path else ""
        nodes.append(node)

        for tensor in _tensors(inputs):
            source = produced.get(id(tensor))
            if source is not None:
                edges.append((source, f"{node_id}.input"))
                break   # 첫 입력 포트만 잇는다. 다입력 모듈은 아래 opaque로 남는다.
        for tensor in _tensors(output):
            produced[id(tensor)] = f"{node_id}.output"

    nodes.append(Node(label="output", id="tf-output", type="torchflow.Output"))
    if len(nodes) > 2:
        edges.append((f"{nodes[-2].id}.output", "tf-output.input"))

    graph = Graph(name=name, instances=instances, nodes=nodes, edges=edges)
    return ModuleGraph(
        meta={"source": "attach", "model": type(model).__name__,
              "params": sum(p.numel() for p in model.parameters())},
        graph=graph,
    )


def trace_report(ir: ModuleGraph) -> dict[str, Any]:
    """import 리포트 - 무엇이 이어졌고 무엇이 끊겼는지(§2.3 3분류의 Attach 판).

    Attach에는 ``structural``/``cell``/``opaque`` 대신 "연결됨/끊김"만 있다.
    코드를 읽지 않았으므로 분류할 근거가 없다.
    """
    call_nodes = [node for node in ir.graph.nodes if node.call]
    connected = {dst.split(".")[0] for _, dst in ir.graph.edges}
    unlinked = [node.label for node in call_nodes if node.id not in connected]
    return {
        "nodes": len(call_nodes),
        "edges": len(ir.graph.edges),
        "unlinked": unlinked,
        "linked_ratio": round(1 - len(unlinked) / max(len(call_nodes), 1), 3),
        "params": ir.meta.get("params", 0),
    }
