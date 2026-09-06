"""L0/L1 커널 프로세스 진입점 (기획서 §8.1).

hub와는 zmq DEALER로만 말한다. 이 프로세스만 torch를 import하므로 커널이
죽어도 hub와 UI는 살아 있다. 커널 격리(§5.8)의 실체가 여기다.

    python -m torchflow.kernel --endpoint tcp://127.0.0.1:5555 --level L0
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import zmq

from .. import protocol as proto
from ..ir import ModuleGraph, canonical_json


class Kernel:
    def __init__(self, endpoint: str, level: str, identity: str, state_dir: Path,
                 parent: int | None = None):
        self.level = level
        self.state_dir = state_dir
        self.graph: ModuleGraph | None = None
        self.rt: dict = {}
        self.session = None      # L0Session - 첫 RunNodes에서 만든다(torch import 지연).
        self.l1_session = None   # L1의 살아 있는 모듈 트리
        self.l1_pass = None      # 마지막 probe 패스 - Debug Console이 여기서 값을 읽는다.
        self.l1_device = None    # 모듈 트리가 지금 올라가 있는 디바이스
        self.budget = None       # ProbeBudget
        # hub의 pid. ppid를 스스로 읽으면 hub가 이미 죽은 뒤에 뜬 커널은 바뀐 값을 원래
        # 부모로 알고 영영 살아남는다 - 그래서 띄우는 쪽이 자기 pid를 넘긴다.
        self.parent = parent if parent is not None else os.getppid()
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, identity.encode())
        self.socket.connect(endpoint)

    def send(self, message, frames=None) -> None:
        self.socket.send_multipart(proto.encode(message, frames))

    def send_captured(self, req_id: str, captured: list[dict]) -> None:
        """노드가 찍은 stdout/stderr를 노드 태그와 함께 올린다(§5.6.1)."""
        for entry in captured:
            key = f"{entry['path']}/{entry['node_id']}" if entry["path"] else entry["node_id"]
            self.send(proto.Log(req_id=req_id, level_name=entry["stream"],
                                node_id=key, text=entry["text"]))

    def announce(self) -> None:
        import torch

        registry_path = None
        if self.level == "L0":
            from .registry import write

            registry_path = str(write(self.state_dir / "registry.json"))
        self.send(
            proto.Ready(
                level=self.level,
                pid=os.getpid(),
                torch_version=torch.__version__,
                device="cpu" if self.level == "L0" else "cuda" if torch.cuda.is_available() else "cpu",
                registry_path=registry_path,
            )
        )

    def serve(self) -> None:
        self.announce()
        while True:
            # hub가 어떻게 죽든(SIGKILL 포함) 커널이 고아로 남지 않는다 - 부모가 바뀌면 나간다.
            if not self.socket.poll(1000):
                if os.getppid() != self.parent:
                    return
                continue
            try:
                message, _ = proto.decode_to_kernel(self.socket.recv_multipart())
            except Exception as exc:  # 손상된 프레임에 커널이 죽지 않는다.
                self.send(proto.Log(level_name="error", text=f"bad frame: {exc}"))
                continue
            if message.type == "Shutdown":
                return
            try:
                self.dispatch(message)
            except Exception as exc:
                req_id = getattr(message, "req_id", "?")
                self.send(proto.Error(req_id=req_id, kind="kernel",
                                      message=f"{type(exc).__name__}: {exc}",
                                      traceback=traceback.format_exc()))
                # 요청은 Progress로 끝나야 한다. 안 보내면 hub가 타임아웃(120 s)까지 기다리고,
                # 그 사이 이벤트 루프가 서서 화면 전체가 멈춘다.
                self.send(proto.Progress(req_id=req_id, done=0, total=0))

    def dispatch(self, message) -> None:
        if message.type == "Ping":
            self.send(proto.Pong(req_id=message.req_id, level=self.level))
        elif message.type == "RunNodes":
            self.run_nodes(message)
        elif message.type == "RunClosure":
            self.run_closure(message)
        elif message.type == "ImportTrace":
            self.import_trace(message)
        elif message.type == "EstimateMemory":
            self.estimate_memory(message)
        elif message.type == "Eval":
            self.eval_expr(message)
        elif message.type == "ReloadBlocks":
            from .registry import write

            write(self.state_dir / "registry.json")
            if self.session is not None:
                self.session.invalidate()  # 블록 스키마가 바뀌면 캐시는 못 믿는다.
            self.send(proto.Progress(req_id=message.req_id, done=1, total=1))

    def run_closure(self, message) -> None:
        """L1-bwd 한 번. 디바이스 배정과 프로브 예산이 여기서 적용된다."""
        import torch

        from .budget import ProbeBudget
        from .devices import pick_l1_device
        from .l1 import L1Pass, ProbeConfig

        if message.graph is not None:
            self.graph = ModuleGraph.model_validate(message.graph)
            self.rt = dict(message.graph.get("rt") or {})
        if self.graph is None:
            self.send(proto.Error(req_id=message.req_id, kind="kernel", message="no graph loaded"))
            return

        if self.budget is None:
            self.budget = ProbeBudget()
        decision = self.budget.decide()
        if not decision.enabled:
            self.send(proto.Busy(req_id=message.req_id, level="L1", reason=decision.reason))
            self.send(proto.Progress(req_id=message.req_id, done=0, total=0))
            return

        # L2가 GPU를 가져갔으면 L1은 비켜선다. 그래프를 죽이지 않고 CPU로 내려가되,
        # 살아 있는 모듈 트리는 버리지 않고 옮긴다 - 다시 만들면 콜드 비용을 다시 낸다.
        choice = pick_l1_device(torch, occupied=tuple(message.occupied), batch=decision.batch)
        if self.l1_session is not None and self.l1_device != choice.device:
            for module in self.l1_session.modules.values():
                if hasattr(module, "to"):
                    module.to(choice.device)
        self.l1_device = choice.device
        config = ProbeConfig(
            **{**message.probe_cfg,
               "batch": decision.batch,
               "backward": decision.backward and not choice.forward_only,
               "subscribed": tuple(message.probe_cfg.get("subscribed") or ())},
        )
        if self.l1_session is None:
            from .l0 import L0Session

            self.l1_session = L0Session()

        # 패스를 들고 있어야 Debug Console이 마지막 활성값·모듈을 볼 수 있다(§5.6.1).
        self.l1_pass = L1Pass(self.graph, rt=self.rt, device=choice.device, probe=config,
                              session=self.l1_session)
        result = self.l1_pass.probe_once()
        self.budget.record(result.elapsed_ms)
        self.send_captured(message.req_id, result.captured)

        if result.error:
            self.send(proto.Error(req_id=message.req_id, kind=result.error["kind"],
                                  node_id=result.error["node_id"],
                                  message=result.error["message"]))

        wanted = set(message.node_ids)
        emitted = 0
        for node in result.nodes:
            if wanted and node.node_id not in wanted:
                continue
            self.send(proto.Done(
                req_id=message.req_id, node_id=node.node_id, path=node.path,
                spec=node.spec, elapsed_ms=result.elapsed_ms,
                numeric=node.numeric, histogram=node.histogram, feature=node.feature,
                grad={"norm": node.grad_norm, "ratio": node.grad_ratio,
                      "weight_norm": node.weight_norm, "warn": node.warn,
                      "objective": result.objective, "device": result.device,
                      "forward_only": result.forward_only},
            ))
            emitted += 1
        self.send(proto.Progress(req_id=message.req_id, done=emitted, total=emitted))

    def eval_expr(self, message) -> None:
        """Debug Console 표현식 1개(§5.6.1). 마지막 probe가 없으면 그렇게 답한다."""
        from .console import evaluate

        if self.l1_pass is None:
            self.send(proto.EvalResult(req_id=message.req_id, ok=False,
                                       error="probe를 한 번 돌린 뒤에 값을 볼 수 있습니다"))
        else:
            key = f"{message.path}/{message.node_id}" if message.path else message.node_id
            self.send(proto.EvalResult(req_id=message.req_id,
                                       **evaluate(self.l1_pass, key, message.expr)))
        self.send(proto.Progress(req_id=message.req_id, done=1, total=1))

    def import_trace(self, message) -> None:
        """사용자 .py를 인스턴스화해서 그래프로 만든다(§7.4 경로 3)."""
        from .importer import ImportError_, import_instance

        try:
            ir, report = import_instance(message.file, message.factory, message.example_inputs)
        except ImportError_ as exc:
            self.send(proto.Error(req_id=message.req_id, kind="exception",
                                  message=exc.message, mapping={"stage": exc.stage}))
        else:
            self.send(proto.Imported(
                req_id=message.req_id,
                graph=json.loads(canonical_json(ir)),
                report=report,
            ))
        self.send(proto.Progress(req_id=message.req_id, done=1, total=1))

    def estimate_memory(self, message) -> None:
        from .l0 import estimate_memory

        if message.graph is not None:
            self.graph = ModuleGraph.model_validate(message.graph)
            self.rt = dict(message.graph.get("rt") or {})
        if self.graph is None:
            self.send(proto.Error(req_id=message.req_id, kind="kernel", message="no graph loaded"))
            return
        estimate = estimate_memory(
            self.graph, batch=message.batch, optimizer=message.optimizer, amp=message.amp,
            rt=self.rt,
        )
        self.send(proto.MemoryEstimate(req_id=message.req_id, **estimate))
        self.send(proto.Progress(req_id=message.req_id, done=1, total=1))

    def run_nodes(self, message) -> None:
        from .l0 import L0Session

        if self.session is None:
            self.session = L0Session()
        if message.graph is not None:
            self.graph = ModuleGraph.model_validate(message.graph)
            self.rt = dict((message.graph.get("rt") or {}))
        if self.graph is None:
            self.send(proto.Error(req_id=message.req_id, kind="kernel", message="no graph loaded"))
            return

        # 세션 캐시가 바뀐 노드만 실제로 돌린다(§5.2 early cutoff). 캐시 히트 노드는
        # spec에서 fake 텐서를 되만들어 하류를 잇는다.
        result = self.session.run(self.graph, rt=self.rt)
        wanted = {item.node_id for item in message.batch} or None
        versions = {item.node_id: item.version for item in message.batch}

        self.send_captured(message.req_id, result.captured)
        emitted = 0
        for report in result.nodes:
            if wanted is not None and report.node_id not in wanted:
                continue
            self.send(
                proto.Done(
                    req_id=message.req_id,
                    node_id=report.node_id,
                    path=report.path,
                    version=versions.get(report.node_id, 0),
                    spec=report.spec,
                    elapsed_ms=report.elapsed_ms,
                    cache_hit=report.cache_hit,
                )
            )
            emitted += 1

        if result.error:
            self.send(
                proto.Error(
                    req_id=message.req_id,
                    kind=result.error["kind"],
                    node_id=result.error["node_id"],
                    message=result.error["message"],
                    mapping=result.error["mapping"],
                )
            )
        self.send(proto.Progress(
            req_id=message.req_id, done=emitted, total=emitted,
            total_params=result.total_params, rerun_ratio=round(result.rerun_ratio, 4)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m torchflow.kernel")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--level", default="L0", choices=["L0", "L1"])
    parser.add_argument("--identity", default="kernel")
    parser.add_argument("--state-dir", default=".torchflow")
    parser.add_argument("--parent", type=int, default=None, help="띄운 hub의 pid. 죽으면 따라 나간다")
    args = parser.parse_args()
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    Kernel(args.endpoint, args.level, args.identity, state_dir, parent=args.parent).serve()


if __name__ == "__main__":
    main()
