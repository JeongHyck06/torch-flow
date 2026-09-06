"""L0 정적 패스 - FakeTensorMode 기반 shape/dtype 추론 (기획서 §5.1.1, §5.2, §6.2).

L0는 메모리를 할당하지 않는다. 모듈은 FakeTensorMode 안에서 만들어지고
forward도 그 안에서 돌기 때문에 675M 파라미터 모델도 ms 단위로 shape가 나온다.
GPU가 없어도 돈다 - ``torchflow view``가 노트북에서 뜨는 이유다.

:class:`L0Session`이 편집 사이에 캐시를 들고 있다. L0에서는 노드의 출력이
``TensorSpec`` 하나로 완전히 기술되므로 캐시 히트는 **spec에서 fake 텐서를
되만들어** 하류를 잇는다 - 모듈 생성도 forward도 건너뛴다. 이것이 §5.2의
early cutoff이 L0에서 취하는 형태다.

심볼 차원(``"B"``)은 각 심볼에 서로 다른 큰 소수를 바인딩하고 결과에서 되돌린다.
``ponytail: 소수 바인딩, 파생 차원(B*L)은 정수로 남는다. ShapeEnv 심볼 추론은 v1.``
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from ..expr import ExprError, resolve
from ..ir import Composite, Graph, ModuleGraph, Node, canonical_json, split_endpoint, topo_order

# 실제 모델 차원과 겹치지 않도록 고른 큰 소수들.
_SYMBOL_PRIMES = [5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081]

# shape 실패를 알아보는 휴리스틱. torch는 이들을 모두 RuntimeError로 던지므로
# 문구로 가른다. 놓치면 kind가 "exception"이 될 뿐 노드 귀속은 유지된다.
_SHAPE_HINTS = (
    "shapes cannot be multiplied",
    "same reduction dim",
    "size mismatch",
    "must match the size",
    "must match except in dimension",
    "expected input",
    "Expected size",
    "Expected more than",
    "invalid for input of size",
    "Given groups=",
    "normalized_shape",
    "Dimension out of range",
)

# 옵티마이저 상태가 파라미터 1개당 차지하는 배수(§6.2).
OPTIMIZER_STATE_MULTIPLIER = {"sgd": 0.0, "sgd_momentum": 1.0, "adam": 2.0, "adamw": 2.0, "adam8bit": 0.5}


def _digest(payload: Any) -> str:
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, default=str).encode(), digest_size=16
    ).hexdigest()


class L0Error(Exception):
    """노드에 귀속되는 실패. ``node_id``가 항상 붙는다(§5.6)."""

    def __init__(self, node_id: str, message: str, kind: str = "exception", mapping=None):
        super().__init__(message)
        self.node_id = node_id
        self.message = message
        self.kind = kind
        self.mapping = mapping


@dataclass
class NodeReport:
    node_id: str
    label: str
    spec: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    cache_hit: bool = False
    # 호출 경로 = 이 노드에 닿기까지 거친 호출 노드 id 체인. 컴포지트 하나가 여러 번
    # 인스턴스화되면 안쪽 노드 id는 같고 경로만 다르다 - UI가 둘을 구분하는 열쇠다.
    path: str = ""

    @property
    def key(self) -> str:
        return f"{self.path}/{self.node_id}" if self.path else self.node_id


@dataclass
class CacheEntry:
    specs: dict[str, dict[str, Any] | None]
    output_key: str


@dataclass
class PassResult:
    nodes: list[NodeReport] = field(default_factory=list)
    # 노드별 stdout/stderr(§5.6.1). 실행 중 찍힌 것만, 캐시 히트에는 없다.
    captured: list[dict[str, Any]] = field(default_factory=list)
    total_params: int = 0
    error: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    cache_hits: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def rerun_ratio(self) -> float:
        """편집당 재실행된 노드 비율. §5.2 KPI는 <15 %."""
        return 0.0 if not self.nodes else 1 - self.cache_hits / len(self.nodes)

    def spec(self, label: str) -> dict[str, Any] | None:
        for report in self.nodes:
            if report.label == label:
                return report.spec
        return None


def _import_target(path: str):
    """``"torch.nn.Linear@2.14"`` -> 클래스. 버전 꼬리표는 떼고 찾는다."""
    import importlib

    name = path.split("@", 1)[0]
    parts = name.split(".")
    module, attr_start = None, 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
            attr_start = stop
            break
        except ImportError:
            continue
    if module is None:
        raise ValueError(f"cannot resolve {path!r}")
    target = module
    for attr in parts[attr_start:]:
        target = getattr(target, attr)
    return target


class L0Session:
    """편집 사이에 살아남는 L0 상태.

    커널 프로세스가 하나 들고 있으면 편집 -> 재실행에서 바뀐 노드만 실제로 돈다.
    """

    def __init__(self, max_entries: int = 50_000):
        self.cache: dict[str, CacheEntry] = {}
        self.modules: dict[str, Any] = {}
        self.max_entries = max_entries

    def run(self, ir: ModuleGraph, **kwargs) -> PassResult:
        return L0Pass(ir, session=self, **kwargs).run()

    def invalidate(self) -> None:
        self.cache.clear()
        self.modules.clear()

    def _trim(self) -> None:
        # ponytail: 삽입 순 FIFO. LRU·CAS 디스크 계층은 M3(L1 활성값이 커질 때).
        while len(self.cache) > self.max_entries:
            self.cache.pop(next(iter(self.cache)))


class L0Pass:
    """IR 하나에 대한 정적 패스 1회."""

    def __init__(
        self,
        ir: ModuleGraph,
        *,
        hp: dict[str, Any] | None = None,
        rt: dict[str, Any] | None = None,
        bind: dict[str, int] | None = None,
        session: L0Session | None = None,
    ):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode

        # shape 실패는 L0의 정상 경로다(사용자가 편집 중인 그래프는 자주 안 맞는다).
        # torch가 실패마다 내부 트레이스백을 stderr로 쏟으면 커널 로그가 묻힌다.
        logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)

        self.torch = torch
        self.ir = ir
        self.rt = dict(rt or {})
        self.hp = {name: spec.default for name, spec in ir.hparams.items()}
        self.hp.update(hp or {})
        self.fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        self.session = session or L0Session()
        self.forced = dict(bind or {})  # 심볼 -> 구체값(메모리 추정용)

        self._symbols: dict[str, int] = {}
        self._reports: list[NodeReport] = []
        self.captured: list[dict[str, Any]] = []
        self._used_modules: set[str] = set()
        self._misses = 0
        self._composite_hashes: dict[str, str] = {}
        # 호출 키 -> 모듈 캐시 키. 공유 인스턴스(tied embedding)는 같은 값을 갖고,
        # L1이 "이 노드의 파라미터"를 찾을 때도 이 표를 쓴다.
        self.node_modules: dict[str, str] = {}
        self._env_fp = _digest([torch.__version__, "L0", "static"])

    # 서브클래스 훅
    # L1은 같은 그래프 순회를 실제 텐서로 다시 돈다. 순회 로직을 복제하지 않고
    # 이 네 지점만 갈아끼운다.

    def _context(self):
        return self.fake_mode

    def _input_tensor(self, node: Node, port, shape: list[int], dtype):
        return self.torch.empty(shape, dtype=dtype)

    def _instantiate(self, node: Node, kind: str, args: dict, module_key: str, call_path: str):
        return _import_target(kind)(**args)

    def _on_node(self, node: Node, call_path: str, outputs: dict[str, Any]) -> None:
        """노드 하나가 끝날 때마다 불린다. L0에서는 할 일이 없다."""

    # 심볼 차원
    def _bind(self, dim: Any) -> int:
        if isinstance(dim, int):
            return dim
        if dim in self.forced:
            return self.forced[dim]
        if dim not in self._symbols:
            if len(self._symbols) >= len(_SYMBOL_PRIMES):
                raise ValueError(f"too many symbolic dims: {dim!r}")
            self._symbols[dim] = _SYMBOL_PRIMES[len(self._symbols)]
        return self._symbols[dim]

    def _unbind(self, size: int) -> str | int:
        for name, prime in self._symbols.items():
            if size == prime:
                return name
        return size

    def _demangle(self, text: str) -> str:
        """진단 문구에 새어 나온 바인딩 소수를 심볼 이름으로 되돌린다.

        사용자가 보는 것은 ``[5003, 192] X [768, 10]``이 아니라
        ``[B, 192] X [768, 10]``이어야 한다.
        """
        for name, prime in self._symbols.items():
            text = text.replace(str(prime), name)
        return text

    def _spec(self, value: Any) -> dict[str, Any] | None:
        if not self.torch.is_tensor(value):
            return None
        return {
            "shape": [self._unbind(int(d)) for d in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
        }

    def _materialize(self, spec: dict[str, Any]):
        shape = [self._bind(d) for d in spec["shape"]]
        return self.torch.empty(shape, dtype=getattr(self.torch, spec["dtype"]))

    # 캐시 (§5.2)
    def _composite_hash(self, name: str) -> str:
        if name not in self._composite_hashes:
            composite = self.ir.composites.get(name)
            self._composite_hashes[name] = (
                _digest(json.loads(canonical_json(composite))) if composite else "missing"
            )
        return self._composite_hashes[name]

    def _node_key(self, identity: str, args: dict, inputs: dict) -> str:
        """``node_key(n, L0)`` - §5.2. 상류 output_key가 들어가므로 early cutoff이 성립한다."""
        return _digest(
            {
                "id": identity,
                "args": args,
                "inputs": {port: self._spec(value) for port, value in sorted(inputs.items())},
                "env": self._env_fp,
            }
        )

    def _cache_specs(self, outputs: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
        """캐시에 넣을 형태로 출력을 기술한다. 되만들 수 없는 값이 있으면 ``None``.

        ``None`` 출력은 유효한 값이다 - ``need_weights=False``인 MHA가 그렇다.
        되살릴 수 없는 것은 텐서도 ``None``도 아닌 객체뿐이다.
        """
        specs = {}
        for name, value in outputs.items():
            if value is None:
                specs[name] = {"kind": "none"}
            elif self.torch.is_tensor(value):
                specs[name] = {"kind": "tensor", **self._spec(value)}
            else:
                return None
        return specs

    def _lookup(self, key: str) -> dict[str, Any] | None:
        entry = self.session.cache.get(key)
        if entry is None:
            return None
        return {
            name: None if spec["kind"] == "none" else self._materialize(spec)
            for name, spec in entry.specs.items()
        }

    def _store(self, key: str, outputs: dict[str, Any]) -> None:
        specs = self._cache_specs(outputs)
        if specs is None:
            return
        self.session.cache[key] = CacheEntry(specs=specs, output_key=_digest(specs))
        self.session._trim()

    # 실행
    def run(self) -> PassResult:
        started = time.perf_counter()
        error = None
        try:
            with self._context():
                self._run_scope(self.ir.graph, {"hp": self.hp, "p": {}, "rt": self.rt}, {}, "")
        except L0Error as exc:
            error = {
                "kind": exc.kind,
                "node_id": exc.node_id,
                "message": exc.message,
                "mapping": exc.mapping,
            }
        return PassResult(
            nodes=self._reports,
            captured=self.captured,
            total_params=self._count_params(),
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            cache_hits=sum(1 for report in self._reports if report.cache_hit),
        )

    def _count_params(self) -> int:
        """이번 패스가 실제로 쓴 모듈만 센다. 공유 인스턴스는 1회(§6.1 ⟲ shared)."""
        seen, total = set(), 0
        for key in self._used_modules:
            module = self.session.modules.get(key)
            if module is None or not hasattr(module, "parameters"):
                continue
            for param in module.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    total += param.numel()
        return total

    def _run_scope(
        self,
        scope: Graph | Composite,
        env: dict[str, Any],
        inputs: dict[str, Any],
        path: str,
        call_path: str = "",
    ) -> dict[str, Any]:
        values: dict[str, Any] = {f"$in.{k}": v for k, v in inputs.items()}
        incoming: dict[str, dict[str, str]] = {}
        for src, dst in scope.edges:
            node_id, port = split_endpoint(dst)
            incoming.setdefault(node_id, {})[port] = src

        for node in topo_order(list(scope.nodes), list(scope.edges)):
            kwargs = {}
            for port, src in incoming.get(node.id, {}).items():
                if src not in values:
                    raise L0Error(node.id, f"input port {port!r} is not connected")
                kwargs[port] = values[src]
            started = time.perf_counter()
            before = self._misses
            outputs = self._exec_capture(node, scope, env, kwargs, path, call_path)
            elapsed = (time.perf_counter() - started) * 1000
            self._on_node(node, call_path, outputs)
            for name, value in outputs.items():
                values[f"{node.id}.{name}"] = value
            first = next(iter(outputs.values()), None)
            self._reports.append(
                # 컨테이너 노드(composite·Repeat·Switch)의 히트는 "내 서브트리에서
                # 아무것도 실행되지 않았다"는 뜻이다. 리프에서는 같은 말이 된다.
                NodeReport(
                    node.id, node.label, self._spec(first),
                    elapsed_ms=elapsed, cache_hit=self._misses == before, path=call_path,
                )
            )

        results: dict[str, Any] = {}
        for src, dst in scope.edges:
            node_id, port = split_endpoint(dst)
            if node_id == "$out":
                results[port] = values[src]
        return results

    def _exec_capture(self, node: Node, scope, env, kwargs, path, call_path="") -> dict[str, Any]:
        """노드가 찍은 stdout/stderr를 그 노드에 귀속시킨다(§5.6.1 stdout 라우팅).

        리다이렉트는 중첩되므로 컴포지트 안쪽에서 찍힌 줄은 바깥 호출 노드가 아니라
        실제로 찍은 리프 노드에 붙는다. 예외로 끝나도 그때까지 나온 출력은 남긴다 -
        디버깅에서 보고 싶은 것이 대개 그것이다.
        """
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                return self._exec_node(node, scope, env, kwargs, path, call_path)
        finally:
            for stream, buffer in (("stdout", out), ("stderr", err)):
                text = buffer.getvalue().rstrip()
                if text:
                    self.captured.append({"node_id": node.id, "path": call_path,
                                          "stream": stream, "text": text})

    def _exec_node(self, node: Node, scope, env, kwargs, path, call_path="") -> dict[str, Any]:
        if node.enabled is not None and not self._resolve(node.id, node.enabled, env):
            # 구성 시점 비활성 노드는 bypass - 단일 입력을 그대로 흘린다(§4.4).
            return {"output": next(iter(kwargs.values()))} if kwargs else {}

        if node.type == "torchflow.Input":
            self._misses += 1
            return self._make_inputs(node)
        if node.type == "torchflow.Output":
            # 종단 노드는 입력을 그대로 통과시킨다 - 그래프 출력 shape가 노드에 뜨고,
            # L1이 목적함수를 걸 지점이 생긴다.
            return {"output": next(iter(kwargs.values()))} if kwargs else {}

        args = {k: self._resolve(node.id, v, env) for k, v in node.args.items()}

        if node.call is not None:
            instance = scope.instances.get(node.call)
            if instance is None:
                raise L0Error(node.id, f"unknown instance {node.call}")
            inner = f"{call_path}/{node.id}" if call_path else node.id
            return self._call_instance(
                node, instance, env, kwargs, f"{path}/{node.call}", args, inner)

        try:
            target = _import_target(node.type or "")
        except (ValueError, AttributeError) as exc:
            raise L0Error(node.id, str(exc)) from exc

        key = self._node_key(f"{path}:{node.id}:{node.type}", args, kwargs)
        cached = self._lookup(key)
        if cached is not None:
            return cached
        self._misses += 1
        outputs = self._pack(node, self._invoke(node.id, target, kwargs, args))
        self._store(key, outputs)
        return outputs

    def _call_instance(
        self, node, instance, env, kwargs, path, node_args=None, call_path=""
    ) -> dict[str, Any]:
        kind = instance.type
        args = {k: self._resolve(node.id, v, env) for k, v in (instance.args or {}).items()}

        if kind == "torchflow.Switch":
            active = self._resolve(node.id, instance.active, env)
            variant = (instance.variants or {}).get(active)
            if variant is None:
                raise L0Error(node.id, f"switch has no variant {active!r}")
            from ..ir import Instance

            chosen = Instance(
                label=variant.get("label", active), type=variant["type"],
                args=variant.get("args", {}),
            )
            return self._call_instance(
                node, chosen, env, kwargs, f"{path}#{active}", node_args, call_path)

        if kind == "torchflow.Repeat":
            return self._call_repeat(node, instance, env, kwargs, path, call_path)

        if kind.startswith("composite:"):
            return self._call_composite(
                node, kind.split(":", 1)[1], args, kwargs, path, call_path)

        if kind.startswith("cell:"):
            # ponytail: Code Cell의 CPU 실측 shape는 M8. 지금은 노드에 귀속해 실패시킨다.
            raise L0Error(node.id, f"code cells are not executable yet ({kind})", kind="kernel")

        # 모듈 캐시 키에 인자를 포함시켜, 인자가 바뀌면 자동으로 새 모듈이 된다.
        # 같은 인스턴스를 참조하는 호출 노드는 같은 키를 얻어 가중치를 공유한다.
        module_key = _digest({"path": path, "type": kind, "args": args})
        module = self.session.modules.get(module_key)
        if module is None:
            try:
                module = self._instantiate(node, kind, args, module_key, call_path or node.id)
            except Exception as exc:
                raise self._fail(node.id, exc) from exc
            self.session.modules[module_key] = module
        self._used_modules.add(module_key)
        # _exec_node가 이미 "스코프 경로 + 노드 id"를 call_path로 넘겨준다.
        self.node_modules[call_path or node.id] = module_key

        key = self._node_key(f"{module_key}:{node.method}:{node.id}", node_args or {}, kwargs)
        cached = self._lookup(key)
        if cached is not None:
            return cached

        self._misses += 1
        method = getattr(module, node.method) if node.method else module
        outputs = self._pack(node, self._invoke(node.id, method, kwargs, node_args or {}))
        self._store(key, outputs)
        return outputs

    def _pack(self, node: Node, value: Any) -> dict[str, Any]:
        """결과를 선언된 출력 포트에 얹는다. 튜플 반환은 위치 순으로 펼친다(§4.3)."""
        names = [port.name for port in (node.ports_out or [])] or ["output"]
        if len(names) == 1:
            return {names[0]: value}
        if isinstance(value, (tuple, list)):
            return dict(zip(names, value))
        raise L0Error(node.id, f"node declares {len(names)} output ports but returned one value")

    def _call_composite(self, node, name, params, kwargs, path, call_path="") -> dict[str, Any]:
        composite = self.ir.composites.get(name)
        if composite is None:
            raise L0Error(node.id, f"unknown composite {name!r}")
        missing = set(composite.params) - set(params)
        if missing:
            raise L0Error(node.id, f"composite {name!r} missing params: {sorted(missing)}")
        env = {"hp": self.hp, "p": params, "rt": self.rt}
        return self._run_scope(
            composite, env, kwargs, f"{path}!{self._composite_hash(name)[:8]}", call_path)

    def _call_repeat(self, node, instance, env, kwargs, path, call_path="") -> dict[str, Any]:
        count = self._resolve(node.id, instance.count, env)
        mode = instance.mode or "sequential"
        body = instance.body or ""
        if not body.startswith("composite:"):
            raise L0Error(node.id, f"Repeat body must be a composite, got {body!r}")
        name = body.split(":", 1)[1]
        body_composite = self.ir.composites.get(name)
        if body_composite is None:
            raise L0Error(node.id, f"unknown composite {name!r}")

        if mode == "parallel":
            raise L0Error(node.id, "Repeat mode 'parallel' is not implemented yet", kind="kernel")

        current, outputs = kwargs, kwargs
        for index in range(int(count)):
            bind = {
                key: self._resolve(node.id, value, {**env, "i": index})
                for key, value in (instance.bind or {}).items()
            }
            # shared_weights는 단일 인스턴스를 반복 호출한다(§4.4.1).
            step_path = path if mode == "shared_weights" else f"{path}#{index}"
            outputs = self._call_composite(
                node, name, bind, current, step_path, f"{call_path}#{index}")
            current = self._chain(node, body_composite, outputs)
        return outputs

    @staticmethod
    def _chain(node: Node, composite: Composite, outputs: dict[str, Any]) -> dict[str, Any]:
        """sequential Repeat의 다음 회차 입력을 만든다.

        포트는 이름 그대로 상속되지만(§4.4.1) 연쇄는 이름이 아니라 선언 순서로
        이어진다 - 생성 코드의 ``for layer in self.layers: x = layer(x)``와 같은 규칙.
        """
        in_ports = [port.name for port in composite.ports.get("in", [])]
        out_ports = [port.name for port in composite.ports.get("out", [])]
        if len(in_ports) != len(out_ports):
            raise L0Error(
                node.id,
                f"sequential Repeat needs matching port arity, "
                f"got {len(in_ports)} in / {len(out_ports)} out",
            )
        return {name: outputs[out_ports[i]] for i, name in enumerate(in_ports)}

    def _make_inputs(self, node: Node) -> dict[str, Any]:
        outputs = {}
        for port in node.ports_out or []:
            shape = [self._bind(d) for d in (port.shape or [])]
            dtype = getattr(self.torch, port.dtype or "float32")
            outputs[port.name] = self._input_tensor(node, port, shape, dtype)
        if not outputs:
            raise L0Error(node.id, "Input node declares no output ports")
        return outputs

    def _fail(self, node_id: str, exc: Exception) -> L0Error:
        return L0Error(node_id, self._demangle(f"{type(exc).__name__}: {exc}"), self._kind_of(exc))

    def _invoke(self, node_id: str, target, kwargs: dict, args: dict):
        try:
            return target(**kwargs, **args)
        except TypeError:
            # torch 함수 상당수가 첫 인자를 위치로만 받는다.
            positional = list(kwargs.values())
            try:
                return target(*positional, **args)
            except Exception as exc:
                raise self._fail(node_id, exc) from exc
        except Exception as exc:
            raise self._fail(node_id, exc) from exc

    def _resolve(self, node_id: str, value, env):
        try:
            return resolve(value, hp=env.get("hp"), p=env.get("p"), rt=env.get("rt"), i=env.get("i"))
        except ExprError as exc:
            raise L0Error(node_id, str(exc)) from exc

    @staticmethod
    def _kind_of(exc: Exception) -> str:
        text = str(exc)
        if any(hint in text for hint in _SHAPE_HINTS):
            return "shape"
        if "out of memory" in text.lower():
            return "oom"
        return "exception"


def run_pass(ir: ModuleGraph, **kwargs) -> PassResult:
    return L0Pass(ir, **kwargs).run()


# 메모리 밴드 추정 (§6.2)


def estimate_memory(
    ir: ModuleGraph,
    *,
    batch: int = 64,
    optimizer: str = "adam",
    amp: bool = False,
    param_bytes: int = 4,
    hp=None,
    rt=None,
) -> dict[str, Any]:
    """``est. 5.4–6.8 GB @B=64`` 를 만드는 추정.

    활성값은 FakeTensorMode 아래에서 ``saved_tensors_hooks``로 backward 그래프가
    저장하려는 텐서 바이트를 세어 얻는다(실행은 fake이므로 할당은 없다).
    cuDNN workspace와 할당자 파편화는 밴드 ±(0.5 GB + 10 %)로 흡수한다.
    """
    import torch

    saved_bytes = 0
    seen: set[int] = set()

    def derives_from_parameter(tensor) -> bool:
        """파라미터이거나 파라미터의 뷰인가.

        ``nn.Linear``는 ``weight.t()``를 저장하므로 ``isinstance`` 하나로는 놓친다 -         뷰 사슬(``_base``)을 끝까지 따라가야 한다. 놓치면 파라미터가 활성값으로도
        세어져 이중 계상이 된다.
        """
        node = tensor
        while node is not None:
            if isinstance(node, torch.nn.Parameter):
                return True
            node = getattr(node, "_base", None)
        return False

    def pack(tensor):
        nonlocal saved_bytes
        if derives_from_parameter(tensor) or id(tensor) in seen:
            return tensor
        seen.add(id(tensor))
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    # 캐시된 세션을 쓰면 forward를 건너뛰어 hook이 걸리지 않는다 - 항상 새 세션.
    pass_ = L0Pass(ir, hp=hp, rt=rt, bind={"B": batch})
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = pass_.run()
    if not result.ok:
        return {"ok": False, "error": result.error}

    params = result.total_params * param_bytes
    grads = params
    state = int(params * OPTIMIZER_STATE_MULTIPLIER.get(optimizer, 2.0))
    amp_copy = params // 2 if amp else 0  # bf16/fp16 가중치 복사본
    activations = saved_bytes
    total = params + grads + state + amp_copy + activations

    slack = 0.5 * 1024**3 + 0.10 * total
    return {
        "ok": True,
        "batch": batch,
        "optimizer": optimizer,
        "breakdown_bytes": {
            "params": params, "grads": grads, "optimizer_state": state,
            "amp_copy": amp_copy, "activations": activations,
        },
        "total_bytes": total,
        "band_gb": [round(max(0.0, total - slack) / 1024**3, 2), round((total + slack) / 1024**3, 2)],
    }
