"""L1 probe - 살아 있는 모듈 트리 위의 forward/backward (기획서 §5.1.2, §5.1.3).

L0가 shape만 본다면 L1은 실제 값을 본다. 목적함수의 조상 폐쇄 전체를
autograd 포함 1회 forward+backward로 돌려 노드별 ``‖g‖``, ``‖g‖/‖w‖``,
활성값 히스토그램을 얻는다.

probe 계약(§5.1.2)에서 물러설 수 없는 것 셋:

* 모델 상태를 오염시키지 않는다. buffer(BN running stats)는 스냅샷 후 복원하고,
  grad는 ``zero_grad(set_to_none=True)``로 초기화하며, Dropout은 기본 ``eval``이다.
* 초기화는 재현된다. ``seeded_init``이 속성 경로로 시드를 파생하므로 UI와
  생성 코드가 같은 가중치에서 출발한다.
* 비용은 프로브에 귀속된다. 예산을 넘으면 B를 줄이고, 그래도 넘으면
  forward-only로 내려간다(§5.1.4).
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from ..ir import Composite, Graph, ModuleGraph, Node
from ..runtime.init import derive_seed, seeded_init
from .l0 import L0Pass, L0Session

# 상대 기준(§5.1.2). 절대값은 옵션이다.
ADJACENT_LOG10_GAP = 2.0
RATIO_LOW, RATIO_HIGH = 1e-7, 1e-1

OBJECTIVES = ("sum_of_outputs", "random_target_ce", "mse_to_zero", "loss_node")

# ``‖g‖/‖w‖`` 밴드는 손실의 스케일에 의존한다. ``sum_of_outputs``는 grad가 출력
# 원소 수에 비례해 커지므로(4×64×192면 dL/dy=1이 5만 개) 이 밴드를 그대로 대면
# 모든 노드가 amber가 된다 - 경보가 아니라 배경이 된다. 평균 축약된 목적함수와
# 실제 Loss 노드에서만 절대 밴드를 적용하고, 나머지는 상대 기준만 쓴다.
SCALE_CALIBRATED = {"random_target_ce", "mse_to_zero", "loss_node"}


@dataclass
class ProbeConfig:
    """probe 계약의 정본. 캐시 키(``closure_key``)에 그대로 들어간다."""

    batch: int = 4
    objective: str = "sum_of_outputs"
    dropout: str = "eval"          # eval | train
    bn: str = "frozen"             # frozen | live
    collect: tuple[str, ...] = ("grad_norm", "grad_ratio")
    subscribed: tuple[str, ...] = ()   # 히스토그램·썸네일을 받을 노드(§5.1.4)
    seed: int = 0
    hist_bins: int = 64
    budget_ms: float = 2000.0
    backward: bool = True

    def summary(self) -> str:
        """배지에 그대로 찍히는 한 줄."""
        names = {"sum_of_outputs": "Σ out", "random_target_ce": "CE(random)",
                 "mse_to_zero": "MSE(0)", "loss_node": "Loss"}
        scale = "" if self.objective in SCALE_CALIBRATED else " · synthetic"
        return f"{names.get(self.objective, self.objective)} · B={self.batch} · init 1step{scale}"


@dataclass
class L1NodeResult:
    node_id: str
    label: str
    path: str = ""
    spec: dict[str, Any] | None = None
    params: int = 0
    grad_norm: float | None = None
    grad_ratio: float | None = None
    weight_norm: float | None = None
    warn: str | None = None
    histogram: list[int] | None = None
    hist_range: tuple[float, float] | None = None
    numeric: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.path}/{self.node_id}" if self.path else self.node_id


@dataclass
class L1Result:
    nodes: list[L1NodeResult] = field(default_factory=list)
    objective: str = ""
    loss: float | None = None
    device: str = "cpu"
    elapsed_ms: float = 0.0
    forward_only: bool = False
    error: dict[str, Any] | None = None
    first_nonfinite: str | None = None
    captured: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def node(self, label: str) -> L1NodeResult | None:
        return next((result for result in self.nodes if result.label == label), None)


class L1Pass(L0Pass):
    """L0의 그래프 순회를 실제 텐서로 다시 돈다."""

    def __init__(self, ir: ModuleGraph, *, device: str = "cpu", seed: int = 0,
                 probe: ProbeConfig | None = None, session: L0Session | None = None, **kwargs):
        super().__init__(ir, session=session, **kwargs)
        self.device = self.torch.device(device)
        self.seed = seed
        self.probe = probe or ProbeConfig()
        # B를 구체값으로 묶는다 - 실제 텐서를 만들어야 하므로 심볼로 둘 수 없다.
        self.forced.setdefault("B", self.probe.batch)
        self.activations: dict[str, Any] = {}
        # Debug Console(§5.6.1)이 ``x``와 ``batch``로 내보이는 것들.
        self.node_inputs: dict[str, dict[str, Any]] = {}
        self.graph_inputs: dict[str, Any] = {}
        self.node_order: list[tuple[str, Node, str]] = []
        self.outputs: dict[str, Any] = {}
        self._backward_error: dict[str, Any] | None = None
        self._labels = _label_index(ir)

    # L0 훅 재정의
    def _context(self):
        return nullcontext()

    def _lookup(self, key: str):
        """L1은 L0의 spec 캐시를 쓰지 않는다.

        L0의 캐시 히트는 spec에서 ``torch.empty``를 되만들어 하류를 잇는다. 그건
        shape만 볼 때는 옳지만 L1에서는 autograd 그래프가 끊긴 분리된 텐서가
        되어 backward가 조용히 0 grad를 낸다. 살아 있는 모듈 트리(모듈 캐시)는
        그대로 쓰고 값 캐시만 끈다.
        """
        return None

    def _store(self, key: str, outputs: dict[str, Any]) -> None:
        return None

    def _label_path(self, call_path: str) -> str:
        """호출 경로(ULID 체인) -> 속성 경로(``blocks.3.attn``).

        시드 파생 키는 ULID가 아니라 속성 경로여야 생성 코드와 UI가 같은 값을
        계산한다(§5.1.3).
        """
        parts = []
        for segment in call_path.split("/"):
            if not segment:
                continue
            node_id, _, index = segment.partition("#")
            parts.append(self._labels.get(node_id, node_id))
            if index:
                parts.append(index)
        return ".".join(parts)

    def _input_tensor(self, node, port, shape, dtype):
        generator = self.torch.Generator(device="cpu").manual_seed(self.probe.seed)
        if dtype.is_floating_point:
            values = self.torch.randn(shape, generator=generator, dtype=dtype)
        else:
            # 정수 입력은 라벨·토큰이다. 상한을 모를 때는 보수적으로 좁게 잡는다.
            values = self.torch.randint(0, 2, shape, generator=generator, dtype=dtype)
        return values.to(self.device)

    def _instantiate(self, node, kind, args, module_key, call_path):
        path = self._label_path(call_path) or node.label
        # 생성 자체를 경로 시드로 감싼다. nn.MultiheadAttention처럼
        # `reset_parameters`가 없는 모듈(`_reset_parameters`)은 seeded_init이 닿지
        # 못하므로, 생성 시점 RNG까지 고정해야 재현된다.
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(derive_seed(self.seed, path))
            module = super()._instantiate(node, kind, args, module_key, call_path)
            if hasattr(module, "to"):
                module = module.to(self.device)
                seeded_init(module, self.seed, prefix=path)
                if self.probe.dropout == "eval":
                    module.eval()
        return module

    def _exec_node(self, node, scope, env, kwargs, path, call_path=""):
        key = f"{call_path}/{node.id}" if call_path else node.id
        self.node_inputs[key] = {port: value for port, value in kwargs.items()
                                 if self.torch.is_tensor(value)}
        outputs = super()._exec_node(node, scope, env, kwargs, path, call_path)
        if node.type == "torchflow.Input" and not call_path:
            self.graph_inputs.update({port: value for port, value in outputs.items()
                                      if self.torch.is_tensor(value)})
        return outputs

    def _on_node(self, node, call_path, outputs):
        key = f"{call_path}/{node.id}" if call_path else node.id
        first = next(iter(outputs.values()), None)
        if self.torch.is_tensor(first):
            self.activations[key] = first
            if node.type == "torchflow.Output":
                self.outputs[key] = first
        self.node_order.append((key, node, call_path))

    # 목적함수
    def _objective(self, outputs: dict[str, Any]):
        """Experiment Graph가 아직 없을 때의 대리 목적함수(§5.1.2 (2))."""
        torch = self.torch
        tensors = [value for value in outputs.values() if torch.is_tensor(value)]
        if not tensors:
            raise ValueError("graph produced no tensor output to probe")
        head = tensors[-1]

        if self.probe.objective == "mse_to_zero":
            return (head.float() ** 2).mean()
        if self.probe.objective == "random_target_ce":
            if head.dim() < 2:
                raise ValueError("random_target_ce needs a [B, C] output")
            generator = torch.Generator(device="cpu").manual_seed(self.probe.seed + 1)
            target = torch.randint(0, head.shape[-1], (head.shape[0],), generator=generator)
            return torch.nn.functional.cross_entropy(head.float(), target.to(head.device))
        return torch.stack([tensor.float().sum() for tensor in tensors]).sum()

    # 실행
    def probe_once(self) -> L1Result:
        torch = self.torch
        started = time.perf_counter()
        buffers: dict[int, Any] = {}

        try:
            with self._context():
                graph_outputs = self._run_scope(
                    self.ir.graph, {"hp": self.hp, "p": {}, "rt": self.rt}, {}, "")
        except Exception as exc:  # L0Error 포함 - 노드 귀속은 그대로 살아 있다.
            return L1Result(
                device=str(self.device),
                error=_error_of(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                captured=self.captured,
            )

        modules = self._probe_modules()
        buffers = self._snapshot_buffers(modules)
        for module in modules.values():
            module.zero_grad(set_to_none=True)

        # 최상위 그래프는 Output 노드로 끝나고, 컴포지트는 $out 엣지로 끝난다.
        targets = self.outputs or graph_outputs
        loss_value, forward_only = None, not self.probe.backward
        try:
            if self.probe.backward:
                loss = self._objective(targets)
                loss.backward()
                loss_value = float(loss.detach())
        except Exception as exc:
            # backward 실패는 forward 결과까지 버릴 이유가 아니다 - forward-only로 내려간다.
            forward_only = True
            self._backward_error = _error_of(exc)

        results = self._collect(modules)
        self._restore_buffers(buffers)
        self._synchronize()

        first_nonfinite = next(
            (result.key for result in results if result.numeric.get("nan") or result.numeric.get("inf")),
            None,
        )
        return L1Result(
            nodes=results,
            objective=self.probe.summary(),
            loss=loss_value,
            device=str(self.device),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            forward_only=forward_only,
            first_nonfinite=first_nonfinite,
            error=self._backward_error,
            captured=self.captured,
        )

    def _synchronize(self) -> None:
        """가속기 큐를 비운다.

        MPS와 CUDA는 커널을 비동기로 던진다. 동기화 없이 잰 시간은 "제출에 걸린
        시간"이지 실행 시간이 아니다 - ``⏱`` 배지가 거짓말을 하게 된다.
        """
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _probe_modules(self) -> dict[str, Any]:
        """이번 패스가 쓴 노드별 모듈. 공유 인스턴스는 같은 객체를 가리킨다."""
        return {
            key: module
            for key, module_key in self.node_modules.items()
            if (module := self.session.modules.get(module_key)) is not None
            and hasattr(module, "parameters")
        }

    def _snapshot_buffers(self, modules) -> dict[int, list[tuple[Any, Any]]]:
        """BN running stats 오염 방지(§5.1.2). 객체 id로 묶어 공유 모듈을 한 번만 뜬다."""
        snapshot: dict[int, list[tuple[Any, Any]]] = {}
        for module in modules.values():
            if id(module) in snapshot:
                continue
            snapshot[id(module)] = [
                (buffer, buffer.detach().clone())
                for buffer in module.buffers()
                if self.torch.is_tensor(buffer) and buffer.is_floating_point()
            ]
        return snapshot

    def _restore_buffers(self, snapshot) -> None:
        for entries in snapshot.values():
            for buffer, saved in entries:
                buffer.copy_(saved)

    def _collect(self, modules) -> list[L1NodeResult]:
        results = [
            node_stats(
                self.torch, key, node.label, self.activations.get(key), modules.get(key),
                self.probe, path=path,
            )
            for key, node, path in self.node_order
        ]
        _flag_warnings(results, calibrated=self.probe.objective in SCALE_CALIBRATED)
        return results


def node_stats(torch, key: str, label: str, activation, module, probe: ProbeConfig,
               *, path: str = "", node_id: str | None = None) -> L1NodeResult:
    """노드 하나의 probe 결과. IR 트리와 Attach의 실사용 모델이 함께 쓴다.

    grad는 파라미터에 붙으므로 그 파라미터를 소유한 모듈의 노드로 귀속된다.
    """
    subscribed = set(probe.subscribed)
    result = L1NodeResult(node_id=node_id or key, label=label, path=path,
                          spec=_spec_of(torch, activation))

    if torch.is_tensor(activation):
        nan = bool(torch.isnan(activation).any())
        inf = bool(torch.isinf(activation).any())
        result.numeric = {"nan": int(nan), "inf": int(inf)}
        # 히스토그램은 구독 노드에서만 - 프로브 예산 규칙(§5.1.4).
        wanted = not subscribed or key in subscribed or (node_id or key) in subscribed
        if wanted and "hist" in probe.collect:
            result.histogram, result.hist_range = _histogram(torch, activation, probe.hist_bins)

    if module is not None:
        grads = weights = 0.0
        count = 0
        for param in module.parameters():
            count += param.numel()
            weights += float(param.detach().float().pow(2).sum())
            if param.grad is not None:
                grads += float(param.grad.detach().float().pow(2).sum())
        result.params = count
        if count:
            result.weight_norm = math.sqrt(weights)
            result.grad_norm = math.sqrt(grads)
            if result.weight_norm > 0:
                result.grad_ratio = result.grad_norm / result.weight_norm
    return result


def _spec_of(torch, tensor) -> dict[str, Any] | None:
    if not torch.is_tensor(tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
    }


def _label_index(ir: ModuleGraph) -> dict[str, str]:
    """노드 id -> 라벨. 라벨은 생성 코드에서 그대로 속성 이름이 된다."""
    labels: dict[str, str] = {}

    def walk(scope: Graph | Composite) -> None:
        for node in scope.nodes:
            labels[node.id] = node.label

    walk(ir.graph)
    for composite in ir.composites.values():
        walk(composite)
    return labels


def _histogram(torch, tensor, bins: int) -> tuple[list[int], tuple[float, float]]:
    """활성값 히스토그램. 노드마다 커널을 부르지 않도록 한 번에 센다(§5.1.4)."""
    flat = tensor.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return [0] * bins, (0.0, 0.0)
    low, high = float(flat.min()), float(flat.max())
    if low == high:
        high = low + 1e-6
    counts = torch.histc(flat.cpu(), bins=bins, min=low, max=high)
    return [int(value) for value in counts], (low, high)


def _flag_warnings(results: list[L1NodeResult], *, calibrated: bool) -> None:
    """§5.1.2의 임계 판정. 상대 기준이 기본, 비율 밴드는 보정된 목적함수에서만."""
    with_grad = [result for result in results if result.grad_norm is not None]

    if calibrated:
        for result in with_grad:
            if result.grad_ratio is not None and not (RATIO_LOW <= result.grad_ratio <= RATIO_HIGH):
                result.warn = "ratio"

    # 인접 레이어 사이 log10 ‖g‖ 차가 2를 넘으면 소실/폭주로 본다.
    for previous, current in zip(with_grad, with_grad[1:]):
        if previous.grad_norm <= 0 or current.grad_norm <= 0:
            continue
        if abs(math.log10(previous.grad_norm) - math.log10(current.grad_norm)) > ADJACENT_LOG10_GAP:
            current.warn = current.warn or "relative_drop"


def _error_of(exc: Exception) -> dict[str, Any]:
    node_id = getattr(exc, "node_id", None)
    return {
        "kind": getattr(exc, "kind", "exception"),
        "node_id": node_id,
        "message": getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}",
        "mapping": getattr(exc, "mapping", None),
    }


def probe(ir: ModuleGraph, **kwargs) -> L1Result:
    return L1Pass(ir, **kwargs).probe_once()
