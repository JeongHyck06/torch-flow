"""코드 -> 그래프, AST로만 (기획서 §7.4 경로 1).

경로 3(인스턴스 트레이스)과 **다른 도구**다. 그쪽은 사용자 코드를 실행해 실측
트리를 얻고 읽기 전용이며, 이쪽은 실행하지 않고 구조만 읽어 **편집 가능한 IR**을
만든다. 둘은 대체가 아니라 병렬이다.

읽는 문법은 codegen이 쓰는 문법의 상위집합이다. 그래서 왕복이 성립한다:

    codegen(import_source(codegen(ir))) == codegen(ir)

읽지 못하는 문장이 forward에 있으면 그 문장만 버리는 대신 **그 클래스의 forward
전체를 CellModule로 승격**한다(§7.4.2). 반쪽짜리 그래프보다 "여기는 코드 그대로"가
정직하다.

노드 id는 무작위가 아니라 ``(클래스, 속성 경로)``에서 파생한다. 같은 파일을 다시
읽으면 안 바뀐 속성은 같은 id를 얻고, 그래야 좌표·프로브·run 해시가 살아남는다(§7.5).

torch를 import하지 않는다 - hub 프로세스에서 그대로 돈다.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .ir import Composite, Graph, HParam, ModuleGraph, Node, Port

# 텐서 메서드 호출은 같은 이름의 torch 함수 노드로 옮긴다(§7.4.1의 straight-line 범위).
TENSOR_METHODS = {
    "mean", "sum", "flatten", "reshape", "view", "permute", "transpose", "squeeze",
    "unsqueeze", "detach", "chunk", "softmax", "sigmoid", "relu", "tanh", "exp", "log",
    "cat", "matmul", "clamp", "contiguous",
}

OPERATOR_NODES = {ast.Add: "torch.add", ast.Sub: "torch.sub",
                  ast.Mult: "torch.mul", ast.Div: "torch.div"}

# 위치 인자를 이름으로 옮기는 표. 사용자가 손으로 쓴 코드는 대개 위치 인자다
# (``nn.Conv2d(3, 64, 3, padding=1)``). 생성 코드는 항상 키워드라 이 표를 안 탄다.
POSITIONAL: dict[str, list[str]] = {
    "torch.nn.Conv1d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "torch.nn.Conv2d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "torch.nn.Conv3d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "torch.nn.ConvTranspose2d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "torch.nn.Linear": ["in_features", "out_features", "bias"],
    "torch.nn.LayerNorm": ["normalized_shape", "eps"],
    "torch.nn.GroupNorm": ["num_groups", "num_channels"],
    "torch.nn.BatchNorm1d": ["num_features", "eps", "momentum"],
    "torch.nn.BatchNorm2d": ["num_features", "eps", "momentum"],
    "torch.nn.BatchNorm3d": ["num_features", "eps", "momentum"],
    "torch.nn.Embedding": ["num_embeddings", "embedding_dim"],
    "torch.nn.Dropout": ["p"],
    "torch.nn.MaxPool2d": ["kernel_size", "stride", "padding"],
    "torch.nn.AvgPool2d": ["kernel_size", "stride", "padding"],
    "torch.nn.AdaptiveAvgPool2d": ["output_size"],
    "torch.nn.MultiheadAttention": ["embed_dim", "num_heads", "dropout"],
    "torch.flatten": ["input", "start_dim"],
    "torch.cat": ["tensors", "dim"],
    "torch.mean": ["input", "dim"],
    "torch.sum": ["input", "dim"],
    "torch.reshape": ["input", "shape"],
    "torch.permute": ["input", "dims"],
    "torch.transpose": ["input", "dim0", "dim1"],
}

# codegen이 남기는 보호 구역 마커(§7.6.1). 안쪽 바이트는 IR이 그대로 들고 있다.
SLOT_BEGIN = re.compile(r"^(\s*)#\s*tf: user-begin\s+(\S+)\s*$")
SLOT_END = re.compile(r"^\s*#\s*tf: user-end\s*$")

# 코드에는 없어서 재import 때 옛 IR에서 가져오는 것들(§7.5 보존 필드).
PRESERVED = ("probes", "probe_objective", "init")


class AstImportError(ValueError):
    """이 파일에서는 그래프를 만들 수 없다."""


class _Unsupported(Exception):
    """이 문장은 AST로 못 옮긴다 - forward 전체가 Cell이 된다(§7.4.2)."""


@dataclass
class Report:
    classes: list[str] = field(default_factory=list)
    structural: int = 0
    cell: int = 0
    opaque: int = 0
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        total = max(self.structural + self.cell + self.opaque, 1)
        return {"path": "ast", "classes": self.classes, "structural": self.structural,
                "cell": self.cell, "opaque": self.opaque,
                "structural_ratio": round(self.structural / total, 4),
                "problems": self.problems}


def node_id(*parts: str) -> str:
    """``(클래스, 속성)`` -> 안정적인 16자 id. 무작위가 아니어야 재import가 정체성을 지킨다."""
    digest = hashlib.blake2b(":".join(parts).encode(), digest_size=10).digest()
    return base64.b32encode(digest).decode().rstrip("=")[:16]


def import_source(source: str, *, filename: str = "model.py", name: str | None = None,
                  previous: ModuleGraph | None = None) -> tuple[ModuleGraph, dict[str, Any]]:
    """파이썬 소스 하나를 ModuleGraph로. 돌려주는 둘째 값은 성공 지표(§7.4.4)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AstImportError(f"파이썬 문법 오류 {exc.lineno}행: {exc.msg}") from exc

    classes = {node.name: node for node in tree.body
               if isinstance(node, ast.ClassDef) and _is_module(node)}
    if not classes:
        raise AstImportError("nn.Module을 상속한 클래스가 없습니다")
    top = name or _top_class(classes)
    if top not in classes:
        raise AstImportError(f"{top} 클래스가 파일에 없습니다")

    report = Report(classes=sorted(classes))
    ir = ModuleGraph(graph=Graph(name=top), meta={"imported_from": filename, "import_path": "ast"})
    slots = user_slots(source)
    if slots:
        ir.graph.user_code = slots

    for class_name in _order(classes, top):
        reader = _Class(classes[class_name], classes, top=class_name == top, ir=ir,
                        source=source, report=report)
        if class_name == top:
            reader.fill(ir.graph)
        else:
            composite = Composite()
            reader.fill(composite)
            ir.composites[class_name] = composite

    if previous is not None:
        merge_preserved(ir, previous)
    return ir, report.as_dict()


def user_slots(source: str) -> dict[str, str]:
    """``# tf: user-begin <slot>`` 안쪽 바이트를 슬롯별로 걷는다(§7.6.1)."""
    found: dict[str, str] = {}
    slot: str | None = None
    indent = ""
    body: list[str] = []
    for line in source.splitlines():
        if slot is None:
            begin = SLOT_BEGIN.match(line)
            if begin:
                indent, slot = begin.group(1), begin.group(2)
                body = []
            continue
        if SLOT_END.match(line):
            text = "\n".join(one[len(indent):] if one.startswith(indent) else one for one in body)
            if text.strip():
                found[slot] = text
            slot = None
            continue
        body.append(line)
    return found


def merge_preserved(ir: ModuleGraph, previous: ModuleGraph) -> ModuleGraph:
    """옛 IR의 정체성과 보존 필드를 새로 읽은 그래프에 옮긴다(§7.5).

    먼저 **id를 물려받는다**: 같은 속성에서 온 블록은 같은 ULID를 그대로 쓴다. 좌표와
    프로브와 run 해시가 전부 id에 걸려 있으므로, 이것이 어긋나면 재import는 "전부 새 블록"이
    된다. 짝짓기 키는 소스맵(``source_attr``)이 1순위, 라벨이 2순위다.
    """
    _adopt_identity(ir.graph, previous.graph)
    for name, composite in ir.composites.items():
        old = previous.composites.get(name)
        if old is not None:
            _adopt_identity(composite, old)

    for field_name in PRESERVED:
        value = getattr(previous.graph, field_name, None)
        if value:
            setattr(ir.graph, field_name, value)
    if previous.experiment is not None:
        ir.experiment = previous.experiment

    old_nodes = {node.id: node for node in previous.graph.nodes}
    for node in ir.graph.nodes:
        old = old_nodes.get(node.id)
        if old is None:
            continue
        if old.label and old.label != node.label:
            node.label = old.label
        # Input의 shape와 dtype은 코드에 없다. 옛 규격이 있으면 그것이 맞다.
        if (node.type or "").split("@")[0] == "torchflow.Input" and old.ports_out:
            node.ports_out = old.ports_out

    for name, spec in previous.hparams.items():
        if name in ir.hparams and spec.space:
            ir.hparams[name].space = spec.space
    return ir


def _key_of(model: Any) -> str:
    """짝짓기 키. 코드의 속성 이름이 먼저고, 없으면 라벨이다."""
    return getattr(model, "source_attr", None) or model.label


def _adopt_identity(fresh, old) -> None:
    """새로 읽은 스코프의 노드·인스턴스 id를 옛 스코프의 것으로 갈아 끼운다."""
    instances: dict[str, str] = {}
    taken: set[str] = set()
    old_instances: dict[str, list[str]] = {}
    for identifier, instance in old.instances.items():
        old_instances.setdefault(_key_of(instance), []).append(identifier)
    for identifier, instance in fresh.instances.items():
        pool = [one for one in old_instances.get(_key_of(instance), []) if one not in taken]
        if pool:
            instances[identifier] = pool[0]
            taken.add(pool[0])
    fresh.instances = {instances.get(key, key): value for key, value in fresh.instances.items()}

    nodes: dict[str, str] = {}
    taken = set()
    old_nodes: dict[str, list[str]] = {}
    for node in old.nodes:
        old_nodes.setdefault(_key_of(node), []).append(node.id)
    # 그래프 경계는 스코프에 하나씩뿐이라 이름이 아니라 종류로 짝짓는다 - 사람이 Input을
    # "x"라고 불렀든 코드가 "input"이라고 부르든 같은 블록이다.
    boundary = {(node.type or "").split("@")[0]: node.id for node in old.nodes
                if (node.type or "").split("@")[0] in ("torchflow.Input", "torchflow.Output")}
    for node in fresh.nodes:
        kind = (node.type or "").split("@")[0]
        if kind in boundary:
            nodes[node.id] = boundary[kind]
            taken.add(boundary[kind])
            continue
        pool = [one for one in old_nodes.get(_key_of(node), []) if one not in taken]
        if pool:
            nodes[node.id] = pool[0]
            taken.add(pool[0])

    def endpoint(text: str) -> str:
        head, _, port = text.partition(".")
        return f"{nodes.get(head, head)}.{port}"

    for node in fresh.nodes:
        node.id = nodes.get(node.id, node.id)
        if node.call:
            node.call = instances.get(node.call, node.call)
    fresh.edges = [(endpoint(src), endpoint(dst)) for src, dst in fresh.edges]


def _is_module(node: ast.ClassDef) -> bool:
    for base in node.bases:
        text = ast.unparse(base)
        if text.endswith("nn.Module") or text == "Module":
            return True
    return False


def _top_class(classes: dict[str, ast.ClassDef]) -> str:
    """아무도 부르지 않는 클래스가 최상위다. 여럿이면 마지막에 정의된 것."""
    referenced: set[str] = set()
    for definition in classes.values():
        for node in ast.walk(definition):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in classes and node.func.id != definition.name:
                    referenced.add(node.func.id)
    free = [name for name in classes if name not in referenced]
    return free[-1] if free else list(classes)[-1]


def _order(classes: dict[str, ast.ClassDef], top: str) -> list[str]:
    """안쪽 클래스부터. 최상위가 마지막이다."""
    return [name for name in classes if name != top] + [top]


def _method(definition: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in definition.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


class _Class:
    """클래스 하나를 스코프(그래프 또는 컴포지트) 하나로 옮긴다."""

    def __init__(self, definition: ast.ClassDef, classes: dict[str, ast.ClassDef], *,
                 top: bool, ir: ModuleGraph, source: str, report: Report):
        self.definition = definition
        self.classes = classes
        self.top = top
        self.ir = ir
        self.source = source
        self.report = report
        self.params: dict[str, Any] = {}          # 이름 -> 기본값 (없으면 None)
        self.runtime: set[str] = set()            # 기본값 없는 키워드 전용 = rt 상수
        self.env: dict[str, str] = {}             # 변수 -> 엔드포인트
        self.counts: dict[str, int] = {}

    # 진입점
    def fill(self, scope: Graph | Composite) -> None:
        self.scope = scope
        doc = ast.get_docstring(self.definition)
        if doc and isinstance(scope, Composite):
            scope.doc = doc
        self._read_params()
        self._read_init()
        try:
            self._read_forward()
        except _Unsupported as exc:
            self._promote_to_cell(str(exc))

    # __init__
    def _read_params(self) -> None:
        init = _method(self.definition, "__init__")
        if init is None:
            return
        args = init.args
        defaults = dict(zip([arg.arg for arg in args.args[-len(args.defaults):]] if args.defaults
                            else [], args.defaults))
        for arg in args.args[1:]:                       # self 제외
            self.params[arg.arg] = _literal(defaults.get(arg.arg))
            self._declare(arg, defaults.get(arg.arg))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if arg.arg == "seed":
                continue                                # 생성 코드가 붙이는 것, 그래프의 것이 아니다
            # 기본값 없는 키워드 인자는 최상위에서만 런타임 상수다(rt.num_classes).
            # 컴포지트에서는 그냥 "부를 때 반드시 주는 값"이다.
            if default is None and self.top:
                self.runtime.add(arg.arg)
            else:
                self.params[arg.arg] = _literal(default)
            self._declare(arg, default)

    def _declare(self, arg: ast.arg, default: ast.expr | None) -> None:
        if arg.arg == "seed" or (self.top and default is None):
            return
        kind = _kind_of(arg.annotation, _literal(default))
        if self.top:
            self.ir.hparams.setdefault(arg.arg, HParam(type=kind, default=_literal(default)))
        else:
            self.scope.params[arg.arg] = HParam(type=kind, default=_literal(default))

    def _read_init(self) -> None:
        init = _method(self.definition, "__init__")
        if init is None:
            return
        for statement in init.body:
            try:
                self._init_statement(statement)
            except _Unsupported as exc:
                self.report.problems.append(
                    f"{self.definition.name}.__init__ {ast.unparse(statement)[:60]}: {exc}")

    def _init_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Expr):
            return                                       # docstring, super().__init__(), seeded_init
        if isinstance(statement, ast.If):
            return self._switch(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            attr = _self_attr(statement.targets[0] if isinstance(statement, ast.Assign)
                              else statement.target)
            if attr is None or statement.value is None:
                return
            return self._instance(attr, statement.value)
        if isinstance(statement, ast.Pass):
            return
        raise _Unsupported("읽을 수 없는 문장")

    def _instance(self, attr: str, value: ast.expr) -> None:
        from .ir import Instance

        if not isinstance(value, ast.Call):
            # self.dim = 384 처럼 상수를 들고 있는 속성은 인자 참조로만 쓰인다.
            self.params.setdefault(attr, _literal(value))
            return
        repeat = self._repeat(attr, value)
        if repeat is not None:
            self.scope.instances[self._instance_id(attr)] = repeat
            return
        kind = self._type_of(value.func)
        if kind is None:
            raise _Unsupported(f"{ast.unparse(value.func)} 는 블록으로 옮길 수 없습니다")
        instance = Instance(label=attr, type=kind, args=self._call_args(kind, value))
        instance.source_attr = attr
        self.scope.instances[self._instance_id(attr)] = instance

    def _repeat(self, attr: str, value: ast.Call):
        """``nn.Sequential(*[Block(dim=dim) for _ in range(depth)])`` -> Repeat(§4.4.1)."""
        from .ir import Instance

        if self._type_of(value.func) != "torch.nn.Sequential" or len(value.args) != 1:
            return None
        starred = value.args[0]
        if not isinstance(starred, ast.Starred) or not isinstance(starred.value, ast.ListComp):
            return None
        comprehension = starred.value
        body = comprehension.elt
        if not isinstance(body, ast.Call) or len(comprehension.generators) != 1:
            return None
        generator = comprehension.generators[0]
        if not (isinstance(generator.iter, ast.Call)
                and isinstance(generator.iter.func, ast.Name)
                and generator.iter.func.id == "range"):
            return None
        kind = self._type_of(body.func)
        if kind is None or not kind.startswith("composite:"):
            return None
        return Instance(label=attr, type="torchflow.Repeat", body=kind, mode="sequential",
                        count=self._value(generator.iter.args[-1]),
                        bind=self._call_args(kind, body))

    def _switch(self, statement: ast.If) -> None:
        """``if attn_type == 'x': self.attn = A(...) elif ...`` -> Switch(§4.4.2)."""
        from .ir import Instance

        variants: dict[str, dict[str, Any]] = {}
        control: ast.expr | None = None
        attr: str | None = None
        current: ast.If | None = statement
        while current is not None:
            test = current.test
            if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)):
                raise _Unsupported("Switch로 읽을 수 없는 조건")
            control = control or test.left
            key = _literal(test.comparators[0])
            body = [one for one in current.body if not isinstance(one, ast.Expr)]
            if len(body) != 1 or not isinstance(body[0], (ast.Assign, ast.AnnAssign)):
                raise _Unsupported("변형 하나에 대입 하나여야 합니다")
            assignment = body[0]
            target = (assignment.targets[0] if isinstance(assignment, ast.Assign)
                      else assignment.target)
            attr = attr or _self_attr(target)
            call = assignment.value
            if not isinstance(call, ast.Call):
                raise _Unsupported("변형은 생성 호출이어야 합니다")
            kind = self._type_of(call.func)
            if kind is None:
                raise _Unsupported(f"{ast.unparse(call.func)} 변형")
            variants[str(key)] = {"type": kind, "args": self._call_args(kind, call)}
            following = [one for one in current.orelse if isinstance(one, ast.If)]
            current = following[0] if following else None
        if attr is None or control is None:
            raise _Unsupported("Switch 대상을 찾지 못했습니다")
        self.scope.instances[self._instance_id(attr)] = Instance(
            label=attr, type="torchflow.Switch", active=self._value(control), variants=variants)

    # forward
    def _read_forward(self) -> None:
        forward = _method(self.definition, "forward")
        if forward is None:
            raise _Unsupported("forward가 없습니다")

        inputs = [arg.arg for arg in forward.args.args[1:]]
        if self.top:
            entry = Node(id=self._node_id("$input"), label="input", type="torchflow.Input",
                         ports_out=[Port(name=name) for name in inputs])
            self.scope.nodes.append(entry)
            for name in inputs:
                self.env[name] = f"{entry.id}.{name}"
        else:
            self.scope.ports["in"] = [Port(name=name) for name in inputs]
            for name in inputs:
                self.env[name] = f"$in.{name}"

        returned: list[str] = []
        for statement in forward.body:
            if isinstance(statement, ast.Expr):
                continue                                  # docstring
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                    and isinstance(statement.targets[0], ast.Name):
                self.env[statement.targets[0].id] = self._value_of(
                    statement.value, statement.targets[0].id)
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                    and isinstance(statement.targets[0], ast.Tuple):
                self._assign_many(statement.targets[0], statement.value)
                continue
            if isinstance(statement, ast.Return) and statement.value is not None:
                returned = [self._value_of(statement.value, None)]
                break
            raise _Unsupported(f"{ast.unparse(statement)[:60]}")

        if self.top:
            sink = Node(id=self._node_id("$output"), label="output", type="torchflow.Output")
            self.scope.nodes.append(sink)
            for endpoint in returned:
                self.scope.edges.append((endpoint, f"{sink.id}.input"))
        else:
            self.scope.ports["out"] = [Port(name="output")]
            for endpoint in returned:
                self.scope.edges.append((endpoint, "$out.output"))
        self.report.structural += len(self.scope.nodes)

    def _assign_many(self, target: ast.Tuple, value: ast.expr) -> None:
        """``mha_y, mha_weights = self.mha(...)`` - 출력 포트가 여럿인 노드 하나다.

        포트 이름은 변수 이름에서 노드 이름 접두어를 떼어 되찾는다. codegen이
        ``{label}_{port}``로 쓰기 때문에 그 역이 성립한다.
        """
        from .codegen import _snake

        names = [element.id for element in target.elts if isinstance(element, ast.Name)]
        if len(names) != len(target.elts):
            raise _Unsupported("풀어서 받는 대상이 변수가 아닙니다")
        label = _self_attr(value.func) if isinstance(value, ast.Call) else None
        label = label or names[0]
        prefix = f"{_snake(label)}_"
        ports = [name[len(prefix):] if name.startswith(prefix) else name for name in names]
        endpoint = self._value_of(value, label, ports=ports)
        base = endpoint.split(".")[0]
        for name, port in zip(names, ports):
            self.env[name] = f"{base}.{port}"

    def _value_of(self, expression: ast.expr, hint: str | None,
                  ports: list[str] | None = None) -> str:
        """식 하나를 노드로 옮기고 그 출력 엔드포인트를 돌려준다.

        ``hint``는 이 값을 받는 변수 이름이다. 중첩된 식에는 그런 이름이 없으므로
        ``None``이고, 그때는 부르는 함수 이름이 블록 이름이 된다.
        """
        if isinstance(expression, ast.Name):
            if expression.id not in self.env:
                raise _Unsupported(f"{expression.id} 가 어디서 왔는지 모르겠습니다")
            return self.env[expression.id]

        if isinstance(expression, ast.BinOp) and type(expression.op) in OPERATOR_NODES:
            kind = OPERATOR_NODES[type(expression.op)]
            left = self._value_of(expression.left, None)
            right = self._value_of(expression.right, None)
            return self._add_node(hint or kind.split(".")[-1], kind=kind,
                                  inputs={"input": left, "other": right})

        if isinstance(expression, ast.Call):
            return self._call(expression, hint, ports)

        raise _Unsupported(f"{ast.unparse(expression)[:60]}")

    def _call(self, call: ast.Call, hint: str | None, ports: list[str] | None = None) -> str:
        target = call.func
        # self.attr(...) 또는 self.attr.method(...)
        attr = _self_attr(target)
        method = None
        if attr is None and isinstance(target, ast.Attribute):
            attr = _self_attr(target.value)
            method = target.attr if attr is not None else None
        if attr is not None:
            instance_id = self._instance_id(attr)
            if instance_id not in self.scope.instances:
                raise _Unsupported(f"self.{attr} 는 __init__에서 만들어지지 않았습니다")
            inputs = self._inputs(call, hint)
            args = {keyword.arg: self._value(keyword.value) for keyword in call.keywords
                    if keyword.arg and not self._is_tensor(keyword.value)}
            return self._add_node(attr, call=instance_id, method=method or "forward",
                                  inputs=inputs, args=args, ports=ports)

        # torch.f(...) 또는 텐서 메서드 x.mean(dim=1)
        kind = self._type_of(target)
        # x.flatten(1)에서 첫 자리는 이미 x가 차지했다 - 위치 인자 이름도 한 칸 밀린다.
        offset = 0
        if kind is None and isinstance(target, ast.Attribute) and target.attr in TENSOR_METHODS:
            kind = f"torch.{target.attr}"
            offset = 1
            inputs = {"input": self._value_of(target.value, None)}
            inputs.update(self._inputs(call, hint, start=1))
        elif kind is not None:
            inputs = self._inputs(call, hint)
        else:
            raise _Unsupported(f"{ast.unparse(target)} 호출")
        args = self._call_args(kind, call, skip_tensors=True, start=offset)
        label = hint or (target.attr if isinstance(target, ast.Attribute) else kind.split(".")[-1])
        return self._add_node(label, kind=kind, inputs=inputs, args=args, ports=ports)

    def _inputs(self, call: ast.Call, hint: str | None, start: int = 0) -> dict[str, str]:
        """위치 인자 = 입력 포트. 이름은 input, other, input3 ... 순서다."""
        names = ["input", "other"] + [f"input{index}" for index in range(3, 12)]
        found: dict[str, str] = {}
        for index, argument in enumerate(call.args):
            if not self._is_tensor(argument):
                continue
            found[names[index + start]] = self._value_of(argument, None)
        for keyword in call.keywords:
            if keyword.arg and self._is_tensor(keyword.value):
                found[keyword.arg] = self._value_of(keyword.value, None)
        return found

    def _is_tensor(self, expression: ast.expr) -> bool:
        """이 식이 값(상수·파라미터)이 아니라 흐르는 텐서인가."""
        if isinstance(expression, ast.Name):
            return expression.id in self.env
        return isinstance(expression, (ast.Call, ast.BinOp))

    def _add_node(self, label: str, *, kind: str | None = None, call: str | None = None,
                  method: str | None = None, inputs: dict[str, str],
                  args: dict[str, Any] | None = None, ports: list[str] | None = None) -> str:
        seen = self.counts.get(label, 0)
        self.counts[label] = seen + 1
        identifier = self._node_id(label if not seen else f"{label}#{seen}")
        node = Node(id=identifier, label=label, args=args or {})
        # 이 노드가 코드의 어느 자리에서 왔는지. 사람이 라벨을 바꿔도 다음 재import가
        # 같은 블록을 알아본다(§7.5의 매칭 1순위).
        node.source_attr = label if not seen else f"{label}#{seen}"
        if ports:
            node.ports_out = [Port(name=port) for port in ports]
        if call is not None:
            node.call = call
            node.method = method
        else:
            node.type = kind
        self.scope.nodes.append(node)
        for port, endpoint in inputs.items():
            self.scope.edges.append((endpoint, f"{identifier}.{port}"))
        return f"{identifier}.{ports[0] if ports else 'output'}"

    # 값과 이름
    def _type_of(self, expression: ast.expr) -> str | None:
        text = ast.unparse(expression)
        if isinstance(expression, ast.Name) and expression.id in self.classes:
            return f"composite:{expression.id}"
        if text.startswith("nn."):
            return f"torch.{text}"
        if text.startswith("torch.nn.") or text.startswith("torch."):
            return text
        return None

    def _call_args(self, kind: str | None, call: ast.Call,
                   skip_tensors: bool = False, start: int = 0) -> dict[str, Any]:
        args: dict[str, Any] = {}
        names = POSITIONAL.get(kind or "", [])[start:]
        for index, argument in enumerate(call.args):
            if skip_tensors and self._is_tensor(argument):
                continue
            if isinstance(argument, ast.Starred):
                continue
            if index < len(names):
                args[names[index]] = self._value(argument)
            else:
                raise _Unsupported(f"{kind} 의 {index + 1}번째 위치 인자를 이름으로 옮길 수 없습니다")
        for keyword in call.keywords:
            if keyword.arg is None or (skip_tensors and self._is_tensor(keyword.value)):
                continue
            args[keyword.arg] = self._value(keyword.value)
        return args

    def _value(self, expression: ast.expr) -> Any:
        """식 -> IR 값. 파라미터 이름은 참조로 바뀐다(§4.3)."""
        if isinstance(expression, ast.Name):
            return self._reference(expression.id)
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, (ast.Tuple, ast.List)):
            return [self._value(item) for item in expression.elts]
        # dim * 4 처럼 계산이 섞이면 표현식 참조로 남긴다. 네임스페이스를 붙여야
        # codegen이 다시 인자 이름으로 풀 수 있다.
        text = ast.unparse(expression)
        for name in sorted(self.params | {one: None for one in self.runtime}, key=len,
                           reverse=True):
            prefix = "rt" if name in self.runtime else ("hp" if self.top else "p")
            text = re.sub(rf"\b{re.escape(name)}\b", f"{prefix}.{name}", text)
        return {"$expr": text}

    def _reference(self, name: str) -> Any:
        if name in self.runtime:
            return {"$expr": f"rt.{name}"}
        if name in self.params:
            return {"$hp": name} if self.top else {"$p": name}
        return name

    def _instance_id(self, attr: str) -> str:
        return self._node_id(f"i:{attr}")

    def _node_id(self, key: str) -> str:
        return node_id(self.definition.name, key)

    # Cell 승격
    def _promote_to_cell(self, why: str) -> None:
        """forward 하나를 통째로 CellModule로(§7.4.2).

        자식 모듈은 인스턴스로 남는다 - 그래야 Cell 안에서 부르는 ``self.norm1``이
        그래프에도 계속 보인다(``children_ports``).
        """
        from .ir import CodeCell

        forward = _method(self.definition, "forward")
        self.scope.nodes.clear()
        self.scope.edges.clear()
        cell_id = f"{self.definition.name.lower()}_forward"
        # 본문을 원문 그대로 담는다. 주석과 들여쓰기까지 살아야 codegen이 되돌려 쓸 때
        # 사람이 쓴 코드가 그대로 나온다.
        body = _source_of(self.source, forward) if forward else ""
        inputs = [arg.arg for arg in forward.args.args[1:]] if forward else ["x"]
        self.ir.code_cells[cell_id] = CodeCell(
            kind="CellModule", file=f"cells/{cell_id}.py", source=body,
            # 커널이 이 이름으로 셀 함수를 만든다 - forward의 인자 이름 그대로여야 한다.
            ports={"in": [Port(name=name) for name in inputs], "out": [Port(name="output")]},
            children_ports=[instance.label for instance in self.scope.instances.values()],
            sha256=hashlib.sha256(body.encode()).hexdigest() if body else None)
        node = Node(id=self._node_id("$cell"), label=self.definition.name.lower(),
                    type=f"cell:{cell_id}")
        if self.top:
            entry = Node(id=self._node_id("$input"), label="input", type="torchflow.Input",
                         ports_out=[Port(name=name) for name in inputs])
            sink = Node(id=self._node_id("$output"), label="output", type="torchflow.Output")
            self.scope.nodes.extend([entry, node, sink])
            self.scope.edges.extend([(f"{entry.id}.{inputs[0]}", f"{node.id}.input"),
                                     (f"{node.id}.output", f"{sink.id}.input")])
        else:
            self.scope.ports["in"] = [Port(name=name) for name in inputs]
            self.scope.ports["out"] = [Port(name="output")]
            self.scope.nodes.append(node)
            self.scope.edges.extend([(f"$in.{inputs[0]}", f"{node.id}.input"),
                                     (f"{node.id}.output", "$out.output")])
        self.report.cell += 1
        self.report.problems.append(f"{self.definition.name}.forward: {why} - 코드 그대로 둡니다")


def _source_of(source: str, function: ast.FunctionDef) -> str:
    """``forward`` 본문의 원문. 들여쓰기 한 겹을 벗겨 돌려준다."""
    lines = source.splitlines()
    body = function.body
    start = body[0].lineno - 1
    # 첫 문장 앞의 주석도 사람이 쓴 것이다. def 줄을 만날 때까지 거슬러 올라간다.
    while start > 0 and not lines[start - 1].strip().startswith("def ") \
            and (not lines[start - 1].strip() or lines[start - 1].lstrip().startswith("#")):
        start -= 1
    end = max(getattr(one, "end_lineno", one.lineno) for one in body)
    block = lines[start:end]
    indent = min((len(line) - len(line.lstrip()) for line in block if line.strip()), default=0)
    return "\n".join(line[indent:] if len(line) > indent else line for line in block)


def _self_attr(target: ast.expr) -> str | None:
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
            and target.value.id == "self":
        return target.attr
    return None


def _literal(expression: ast.expr | None) -> Any:
    if expression is None:
        return None
    try:
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        return None


def _kind_of(annotation: ast.expr | None, default: Any) -> str:
    text = ast.unparse(annotation) if annotation is not None else ""
    if text in ("int", "float", "bool", "str"):
        return text
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "int"
