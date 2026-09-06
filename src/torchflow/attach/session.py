"""Attach Mode (기획서 §3.5, API 요약 §17.7).

IR도, codegen도, 학습 루프 교체도 없다. 사용자의 ``train.py``에 두 줄을 넣으면
브라우저에 Module Graph와 grad-flow가 뜬다 - Phase A가 독립 제품으로 성립하는
지점이 여기다.

프로세스 구조가 IR 모드와 뒤집힌다(§8.1): hub가 커널을 띄우는 대신, **사용자
프로세스가 L1 커널 역할을 겸하고 hub를 자식으로 띄운다.** 모델이 이미 사용자
프로세스에 살아 있으니 그것을 다른 프로세스로 옮길 이유가 없다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable

from ..ir import ModuleGraph, save
from ..kernel.l1 import ProbeConfig, SCALE_CALIBRATED, _flag_warnings, node_stats
from .trace import trace_module, trace_report

_ACTIVE: "Session | None" = None


class Session:
    """살아 있는 모델 하나에 붙은 관찰 세션."""

    def __init__(self, model, ir: ModuleGraph, *, state_dir: Path, url: str, token: str,
                 objective: Callable | None = None, seed: int = 0,
                 hub: subprocess.Popen | None = None):
        self.model = model
        self.ir = ir
        self.state_dir = state_dir
        self.url = url
        self.token = token
        self.objective = objective
        self.seed = seed
        self.hub = hub
        self.report = trace_report(ir)
        self.scalars: list[dict[str, Any]] = []
        self.run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        self.kind = "exploratory"
        self._run_open = False
        self._example: Any = None
        self._step_handle = None
        # 속성 경로 -> 노드 id. probe 결과를 캔버스 노드에 얹는 열쇠다.
        self._node_of_path: dict[str, str] = {
            node.label: node.id for node in ir.graph.nodes if node.call
        }

    # probe
    def probe(self, batch=None, *, backward: bool = True,
              config: ProbeConfig | None = None) -> dict[str, Any]:
        """정의된 probe 계약대로 한 번 관측한다(§5.1.2).

        모델 상태는 건드리지 않는다: buffer는 복원하고, grad는 원래 값으로
        되돌리며, train/eval 모드도 원래대로 남긴다.
        """
        import torch

        probe = config or ProbeConfig(collect=("grad_norm", "grad_ratio", "hist"))
        inputs = batch if batch is not None else self._example
        if inputs is None:
            raise ValueError("no example input: pass batch= or call tf.watch(example_input=...)")

        modules = {path: module for path, module in self.model.named_modules()
                   if not list(module.children())}
        activations: dict[str, Any] = {}
        handles = [
            module.register_forward_hook(
                lambda _m, _i, out, path=path: activations.__setitem__(path, _first(out)))
            for path, module in modules.items()
        ]

        was_training = self.model.training
        buffers = [(buffer, buffer.detach().clone())
                   for _, buffer in self.model.named_buffers()
                   if torch.is_tensor(buffer) and buffer.is_floating_point()]
        saved_grads = {name: (None if param.grad is None else param.grad.detach().clone())
                       for name, param in self.model.named_parameters()}

        started = time.perf_counter()
        loss_value = None
        try:
            if probe.dropout == "eval":
                self.model.eval()
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad() if backward else torch.no_grad():
                output = _forward(self.model, inputs)
                if backward:
                    loss = self._loss(output, inputs)
                    loss.backward()
                    loss_value = float(loss.detach())
        finally:
            for handle in handles:
                handle.remove()

        results = [
            node_stats(torch, path, path, activations.get(path), module, probe,
                       node_id=self._node_of_path.get(path, path))
            for path, module in modules.items()
        ]
        _flag_warnings(results, calibrated=self.objective is not None
                       or probe.objective in SCALE_CALIBRATED)

        # 모델을 원래대로 되돌린다 - probe는 관찰이지 개입이 아니다(P2).
        for buffer, saved in buffers:
            buffer.copy_(saved)
        for name, param in self.model.named_parameters():
            param.grad = saved_grads[name]
        self.model.train(was_training)

        payload = {
            "loss": loss_value,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "objective": "user objective" if self.objective else probe.summary(),
            "nodes": [
                {"node": result.node_id, "spec": result.spec, "grad_norm": result.grad_norm,
                 "grad_ratio": result.grad_ratio, "warn": result.warn,
                 "histogram": result.histogram, "numeric": result.numeric}
                for result in results
            ],
        }
        self._post("/api/l1", payload)
        return payload

    def _loss(self, output, batch):
        import torch

        if self.objective is not None:
            return self.objective(output, batch)
        tensors = [tensor for tensor in _flatten(output) if torch.is_tensor(tensor)]
        if not tensors:
            raise ValueError("model produced no tensor output to probe")
        return torch.stack([tensor.float().sum() for tensor in tensors]).sum()

    def add_probe(self, kind: str, on: str) -> None:
        """속성 경로에 프로브를 붙인다 - Attach에서 편집 가능한 유일한 것(§3.5)."""
        self._post("/api/probes", {"kind": kind, "on": on,
                                   "node": self._node_of_path.get(on, on)})

    # 트래커
    def log(self, step: int, **scalars: float) -> None:
        """스칼라 기록. 첫 호출에서 manifest가 만들어진다(§10.3)."""
        if not self._run_open:
            self._open_run()
        record = {"run_id": self.run_id, "step": step, **scalars}
        self.scalars.append(record)
        self._post("/api/scalars", record)

    def _open_run(self) -> None:
        self._run_open = True
        self._post("/api/runs", {
            "run_id": self.run_id, "kind": self.kind,
            "name": type(self.model).__name__,
            "manifest": self.manifest(),
        })

    def manifest(self, kind: str | None = None) -> dict[str, Any]:
        """재현에 필요한 것 전부. Attach도 같은 스키마를 쓴다(§10.3)."""
        from ..runtime.manifest import build

        return build(
            run_id=self.run_id, kind=kind or self.kind, model=self.model, seed=self.seed,
            probe={"objective": "user objective" if self.objective else "sum_of_outputs",
                   "linked_ratio": self.report.get("linked_ratio")},
            root=self.state_dir.parent,
        )

    def save_manifest(self, path=None, *, anonymous: bool = False) -> Path:
        from ..runtime.manifest import save

        target = Path(path) if path else self.state_dir / "runs" / self.run_id / "manifest.json"
        return save(self.manifest(), target, anonymous=anonymous)

    def display(self, height: int = 640):
        """노트북 인라인 iframe (§3.5)."""
        try:
            from IPython.display import IFrame
        except ImportError:
            print(self.url)
            return None
        return IFrame(self.url, width="100%", height=height)

    def close(self) -> None:
        if self._step_handle is not None:
            self._step_handle.remove()
            self._step_handle = None
        if self.hub is not None and self.hub.poll() is None:
            self.hub.terminate()
            try:
                self.hub.wait(5)
            except subprocess.TimeoutExpired:
                self.hub.kill()

    # hub 통신
    def _post(self, path: str, payload: dict[str, Any]) -> None:
        """hub가 아직 안 떴거나 이미 죽었어도 사용자 학습을 멈추지 않는다."""
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{self.url.split('?')[0].rstrip('/')}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"token {self.token}"},
        )
        try:
            urllib.request.urlopen(request, timeout=2).close()
        except (urllib.error.URLError, OSError):
            pass


def _first(value):
    for item in _flatten(value):
        return item
    return None


def _flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    else:
        yield value


def _forward(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


# 공개 API (§17.7)


def watch(model, example_input, *, objective: Callable | None = None, port: int | None = None,
          host: str = "127.0.0.1", open_browser: bool = True, state_dir=None,
          seed: int = 0, name: str | None = None) -> Session:
    """모델에 붙어 hub를 띄운다. 반환된 URL을 브라우저에서 열면 그래프가 보인다."""
    global _ACTIVE

    from ..hub.auth import new_token

    state = Path(state_dir or ".torchflow")
    state.mkdir(parents=True, exist_ok=True)
    ir = trace_module(model, example_input, name=name)
    graph_path = state / "attached.tfg.json"
    save(ir, graph_path)

    token = new_token()
    port = port or 8765
    hub = subprocess.Popen(
        [sys.executable, "-m", "torchflow.cli", "view", str(graph_path),
         "--port", str(port), "--host", host, "--state-dir", str(state),
         "--no-browser", "--token", token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "TORCHFLOW_ATTACHED": "1"},
    )
    url = f"http://{host}:{port}/?token={token}"

    session = Session(model, ir, state_dir=state, url=url, token=token,
                      objective=objective, seed=seed, hub=hub)
    session._example = example_input
    _ACTIVE = session

    if open_browser:
        webbrowser.open(url)
    print(f"  TorchFlow → {url}")
    print(f"  {session.report['nodes']} nodes · "
          f"{session.report['params'] / 1e6:.2f}M params · "
          f"연결 {session.report['linked_ratio']:.0%}")
    return session


def hook_step(model, optimizer, *, every: int = 10,
              collect: tuple[str, ...] = ("grad_norm", "grad_ratio", "hist"),
              subscribed_only: bool = True):
    """step 경계에서 grad를 읽는다 (§17.7).

    ``optimizer.step()`` 직전에 걸린다 - 그 순간에만 grad가 살아 있다.
    forward를 다시 돌리지 않으므로 사용자 학습에 붙는 비용은 통계 계산뿐이다.
    """
    import torch

    session = _ACTIVE
    if session is None:
        raise RuntimeError("call tf.watch(...) before tf.hook_step(...)")

    modules = {path: module for path, module in model.named_modules()
               if not list(module.children())}
    probe = ProbeConfig(collect=collect,
                        subscribed=() if not subscribed_only else ())
    state = {"step": 0}

    def on_step(*_args, **_kwargs):
        state["step"] += 1
        if (state["step"] - 1) % every:
            return
        results = [
            node_stats(torch, path, path, None, module, probe,
                       node_id=session._node_of_path.get(path, path))
            for path, module in modules.items()
        ]
        _flag_warnings(results, calibrated=True)  # 사용자 손실은 실제 손실이다.
        session._post("/api/l1", {
            "step": state["step"],
            "objective": f"step {state['step']:,}",
            "nodes": [
                {"node": result.node_id, "grad_norm": result.grad_norm,
                 "grad_ratio": result.grad_ratio, "warn": result.warn}
                for result in results
            ],
        })

    session._step_handle = optimizer.register_step_pre_hook(on_step)
    return session._step_handle


def log(step: int, **scalars: float) -> None:
    if _ACTIVE is None:
        raise RuntimeError("call tf.watch(...) before tf.log(...)")
    _ACTIVE.log(step, **scalars)


def active() -> Session | None:
    return _ACTIVE
