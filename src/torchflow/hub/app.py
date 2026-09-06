"""hub: FastAPI 애플리케이션 (기획서 §8.1, §8.2.2).

이 파일과 hub 패키지 전체는 torch를 import하지 않는다. shape 추론은 전부
L0 커널 프로세스에서 돈다. ``tests/test_hub.py``가 이 불변식을 강제한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import protocol as proto
from ..ir import ModuleGraph, canonical_json, load
from ..pysource import candidates, parse_example_spec
from .auth import DEFAULT_HOSTS, AuthMiddleware, COOKIE_NAME, extract_token, new_token, token_matches
from .engine import Engine
from .graphstore import GraphStore, OpError
from .tracker import Tracker
from .kernels import KernelManager


class Hub:
    """그래프 상태 · 커널 · 접속 클라이언트를 한 곳에서 소유한다."""

    def __init__(self, ir: ModuleGraph | None, state_dir: Path, rt: dict[str, Any] | None = None):
        self.state_dir = state_dir
        # 그래프 없이도 뜬다 - 첫 화면에서 무엇으로 시작할지 고르게 하기 위해(§2.2).
        self.store: GraphStore | None = None
        self.engine: Engine | None = None
        self.graph_path: Path | None = None
        self.rt = dict(rt or {})
        self.kernel = KernelManager("L0", state_dir)
        self.l1 = KernelManager("L1", state_dir)
        self.clients: list[WebSocket] = []
        self.node_states: dict[str, dict[str, Any]] = {}
        self.tracker = Tracker(state_dir / "runs.db")
        self.probes: list[dict[str, Any]] = []
        # Attach 모드에서는 사용자 프로세스가 L1 커널이다(§8.1). hub는 커널을 띄우는
        # 대신 그쪽이 밀어 넣는 결과를 받아 브로드캐스트한다.
        self.attached = False
        self.total_params = 0
        self.rerun_ratio = 0.0
        self.layout_path = state_dir / "layout.json"
        self.layout = self._load_layout()
        if ir is not None:
            self.open(ir)

    def open(self, ir: ModuleGraph, path: Path | None = None) -> None:
        """그래프를 연다. 첫 화면에서 템플릿을 고르면 이 경로로 들어온다."""
        self.store = GraphStore(ir, self.state_dir / "journal.jsonl")
        self.engine = Engine(ir)
        self.graph_path = path
        self.node_states.clear()
        self.total_params = 0

    def _load_layout(self) -> dict[str, Any]:
        if self.layout_path.exists():
            try:
                return json.loads(self.layout_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass   # 좌표가 깨져도 그래프는 열려야 한다 - 자동 배치로 돌아간다.
        return {"positions": {}}

    def update_layout(self, positions: dict[str, Any]) -> None:
        self.layout.setdefault("positions", {}).update(positions)
        self.layout_path.write_text(
            json.dumps(self.layout, sort_keys=True, indent=2), encoding="utf-8")

    @property
    def seq(self) -> int:
        return self.store.seq if self.store is not None else 0

    def devices(self) -> list[dict[str, Any]]:
        """디바이스 스트립(§2.2). torch는 커널만 안다 - hub는 Ready 메시지를 읽을 뿐."""
        ready = self.kernel.ready
        if ready is None:
            return []
        return [{"name": ready.device, "label": ready.device}]

    def templates(self) -> list[dict[str, Any]]:
        """검증된 템플릿 목록(§13.3). 파일이 실제로 있는 것만 열 수 있다."""
        catalogue = [
            {"id": "resnet18", "name": "ResNet-18 / CIFAR-10",
             "recipe": "200 ep · SGD 0.1 + cosine", "metric": "≈95.0 % top-1",
             "file": "resnet18.tfg.json"},
            {"id": "minivit", "name": "MiniViT / CIFAR-10",
             "recipe": "설계 예제 · 검증 전", "metric": "—",
             "file": "minivit.tfg.json"},
            {"id": "nanogpt", "name": "nanoGPT char / Shakespeare",
             "recipe": "5,000 iter · 6층 384 dim", "metric": "val loss ≈1.47",
             "file": "nanogpt.tfg.json"},
        ]
        roots = [Path.cwd() / "examples", Path(__file__).resolve().parents[3] / "examples"]
        for entry in catalogue:
            found = next((root / entry["file"] for root in roots
                          if (root / entry["file"]).is_file()), None)
            entry["path"] = str(found) if found else ""
            entry["available"] = found is not None
        return catalogue

    def registry(self) -> dict[str, Any]:
        path = self.state_dir / "registry.json"
        if not path.exists():
            return {"blocks": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot_for_kernel(self) -> dict[str, Any]:
        if self.store is None:
            raise RuntimeError("no graph is open")
        snapshot = json.loads(canonical_json(self.store.ir))
        snapshot["rt"] = self.rt
        return snapshot

    def apply_op(self, op: dict[str, Any]) -> int:
        """op를 권위 그래프에 반영하고 상태 머신을 전이시킨다(§5.4)."""
        seq = self.store.apply(op)
        self.engine.rebuild(self.store.ir)
        self.engine.on_edit(op)
        return seq

    @property
    def traced(self) -> bool:
        """hook 트레이스로 만들어진 그래프인가.

        트레이스 결과는 IR을 다시 인스턴스화할 수 있는 명세가 아니다. 모듈의
        생성 인자는 파이썬이 기억하지 않으므로 `extra_repr`에서 건진 표시용
        문자열뿐이고, 그걸로 모듈을 다시 만들면 엉뚱한 크기가 나온다.
        실측값이 이미 노드에 박혀 있으니 그것을 쓴다(§3.5: Attach에는 L0
        반응형 갱신이 없다).
        """
        return bool(self.store and self.store.ir.meta.get("source") == "attach")

    def measured_states(self) -> list[proto.NodeState]:
        """트레이스가 기록해 둔 실측 spec을 그대로 노드 상태로 낸다."""
        states = []
        for node in self.store.ir.graph.nodes:
            measured = getattr(node, "measured", None)
            state = proto.NodeState(seq=self.seq, node=node.id, axis="L0",
                                    state="ok", spec=measured,
                                    badges={"measured": True})
            self.node_states[node.id] = state.model_dump(mode="json")
            states.append(state)
        self.total_params = int(self.store.ir.meta.get("params") or 0)
        return states

    def run_l0(self, node_ids: list[str] | None = None) -> list[proto.NodeState]:
        """L0 정적 패스 1회. 커널이 죽어 있으면 조용히 다시 세운다."""
        if self.traced:
            return self.measured_states()
        self.kernel.ensure()
        batch = [proto.NodeRequest(node_id=n, version=self.engine.version.get(n, 0))
                 for n in (node_ids or [])]
        replies = self.kernel.request(
            proto.RunNodes(req_id=f"r-{self.store.seq}", graph=self.snapshot_for_kernel(), batch=batch)
        )
        summary = next((reply for reply in replies if reply.type == "Progress"), None)
        if summary is not None and summary.total_params is not None:
            self.total_params = summary.total_params
            self.rerun_ratio = summary.rerun_ratio or 0.0
        done = [reply for reply in replies if reply.type == "Done"]
        failure = next((reply for reply in replies if reply.type == "Error"), None)
        error = (
            {"kind": failure.kind, "node_id": failure.node_id, "message": failure.message,
             "mapping": failure.mapping}
            if failure is not None else None
        )

        states: list[proto.NodeState] = []
        for key, fields in self.engine.absorb(done, error).items():
            state = proto.NodeState(seq=self.store.seq, axis="L0", **fields)
            self.node_states[key] = state.model_dump(mode="json")
            states.append(state)
        return states

    async def broadcast(self, message) -> None:
        payload = message.model_dump_json(exclude_none=True)
        for client in list(self.clients):
            try:
                await client.send_text(payload)
            except (WebSocketDisconnect, RuntimeError):
                self.clients.remove(client)


def create_app(
    graph_path: Path | str | None = None,
    *,
    state_dir: Path | str = ".torchflow",
    token: str | None = None,
    rt: dict[str, Any] | None = None,
    allowed_hosts: tuple[str, ...] = DEFAULT_HOSTS,
) -> FastAPI:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    # 그래프 없이 뜨면 첫 화면이 뜬다(§2.2의 5개 진입점).
    hub = Hub(load(graph_path) if graph_path else None, state_dir, rt)
    if graph_path:
        hub.graph_path = Path(graph_path)
    token = token or new_token()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        hub.kernel.stop()
        hub.l1.stop()

    app = FastAPI(
        title="TorchFlow hub", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.hub = hub
    app.state.token = token
    app.state.graph_path = Path(graph_path) if graph_path else None
    app.add_middleware(AuthMiddleware, token=token, allowed_hosts=allowed_hosts)

    def no_graph() -> JSONResponse:
        return JSONResponse({"error": "no graph is open"}, status_code=409)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "seq": hub.seq,
            "graph_open": hub.store is not None,
            "attached": hub.attached,
            "kernel": {"alive": hub.kernel.alive(), "level": hub.kernel.level},
            "l1": {"alive": hub.l1.alive()},
        }

    @app.get("/api/graph")
    def graph() -> JSONResponse:
        return no_graph() if hub.store is None else JSONResponse(hub.store.snapshot())

    @app.get("/api/registry")
    def registry() -> JSONResponse:
        return JSONResponse(hub.registry())

    @app.post("/api/ops")
    def ops(op: dict[str, Any]) -> JSONResponse:
        """WebSocket 폴백 경로. seq 규약은 WS와 동일하다(§8.2.2)."""
        if hub.store is None:
            return no_graph()
        try:
            seq = hub.apply_op(op)
        except OpError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        states = hub.run_l0()
        return JSONResponse(
            {"seq": seq, "node_states": [s.model_dump(mode="json") for s in states]}
        )

    @app.post("/api/shapes")
    def shapes() -> JSONResponse:
        if hub.store is None:
            return no_graph()
        states = hub.run_l0()
        return JSONResponse({
            "node_states": [s.model_dump(mode="json") for s in states],
            "total_params": hub.total_params,
            "rerun_ratio": hub.rerun_ratio,
        })

    @app.get("/api/start")
    def start() -> JSONResponse:
        """첫 화면이 필요로 하는 것 전부 (§2.2의 5개 진입점)."""
        hub.kernel.ensure()
        ready = hub.kernel.ready
        return JSONResponse({
            "graph_open": hub.store is not None,
            "state_dir": str(hub.state_dir.resolve()),
            "torch_version": ready.torch_version if ready else None,
            "devices": hub.devices(),
            "templates": hub.templates(),
        })

    @app.post("/api/inspect")
    def inspect_source(request: dict[str, Any]) -> JSONResponse:
        """드롭된 .py에서 인스턴스화할 수 있는 클래스를 고른다.

        hub는 사용자 코드를 실행하지 않는다. ast로 읽기만 한다.
        """
        try:
            found = candidates(request.get("source") or "")
        except SyntaxError as exc:
            return JSONResponse({"error": f"구문 오류 {exc.lineno}행: {exc.msg}"}, status_code=400)
        return JSONResponse({"candidates": found})

    @app.post("/api/import")
    def import_source(request: dict[str, Any]) -> JSONResponse:
        """사용자 .py를 인스턴스로 만들어 그래프로 편다(§7.4 경로 3)."""
        name = Path(request.get("filename") or "model.py").name
        target = hub.state_dir / "imports" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if request.get("source") is not None:
            target.write_text(request["source"], encoding="utf-8")

        try:
            example = parse_example_spec(request.get("example") or "")
        except ValueError as exc:
            return JSONResponse({"error": f"입력 명세: {exc}"}, status_code=400)

        hub.l1.ensure()
        replies = hub.l1.request(proto.ImportTrace(
            req_id=f"i-{hub.seq}", file=str(target),
            factory=request.get("factory") or "", example_inputs=example))

        failure = next((r for r in replies if r.type == "Error"), None)
        if failure is not None:
            stage = (failure.mapping or {}).get("stage", "import")
            return JSONResponse({"error": failure.message, "stage": stage}, status_code=400)

        imported = next((r for r in replies if r.type == "Imported"), None)
        if imported is None or imported.graph is None:
            return JSONResponse({"error": "커널이 그래프를 돌려주지 않았습니다"}, status_code=502)

        hub.open(ModuleGraph.model_validate(imported.graph), target)
        return JSONResponse({"ok": True, "name": hub.store.ir.graph.name,
                             "report": imported.report})

    @app.post("/api/open")
    def open_graph(request: dict[str, Any]) -> JSONResponse:
        """템플릿 또는 경로를 연다. 첫 화면의 진입점 하나."""
        path = Path(request["path"]).expanduser()
        allowed = {entry["path"] for entry in hub.templates()}
        if str(path) not in allowed and not path.is_file():
            return JSONResponse({"error": f"not found: {path}"}, status_code=404)
        try:
            hub.open(load(path), path)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
        return JSONResponse({"ok": True, "name": hub.store.ir.graph.name})

    @app.get("/api/layout")
    def read_layout() -> JSONResponse:
        return JSONResponse(hub.layout)

    @app.post("/api/layout")
    def write_layout(patch: dict[str, Any]) -> JSONResponse:
        """노드 좌표를 저장한다.

        좌표는 IR이 아니라 ``graph/layout.json``에 산다(§10.1) - 그래서 노드를
        옮겨도 그래프 의미가 바뀌지 않고, git에서 ``merge=ours``로 충돌하지 않는다.
        """
        hub.update_layout(patch.get("positions") or {})
        return JSONResponse({"ok": True, "positions": len(hub.layout["positions"])})

    @app.post("/api/probe")
    async def run_probe(cfg: dict[str, Any] | None = None) -> JSONResponse:
        """IR 모드의 L1 probe. Attach 모드에서는 사용자 프로세스가 대신 밀어 넣는다."""
        if hub.store is None:
            return no_graph()
        if hub.attached:
            return JSONResponse({"ok": False, "error": "attached session drives L1"},
                                status_code=409)
        hub.l1.ensure()
        replies = hub.l1.request(proto.RunClosure(
            req_id=f"p-{hub.seq}", graph=hub.snapshot_for_kernel(), probe_cfg=cfg or {}))

        busy = next((r for r in replies if r.type == "Busy"), None)
        if busy is not None:
            return JSONResponse({"ok": False, "budget": busy.reason})

        states = []
        for reply in (r for r in replies if r.type == "Done"):
            grad = reply.grad or {}
            key = f"{reply.path}/{reply.node_id}" if reply.path else reply.node_id
            numeric = reply.numeric or {}
            state = proto.NodeState(
                seq=hub.seq, node=reply.node_id, path=reply.path, axis="L1bwd",
                state="ok.warn" if (grad.get("warn") or numeric.get("nan") or numeric.get("inf"))
                else "ok",
                spec=reply.spec,
                badges={key: value for key, value in {
                    "grad_norm": grad.get("norm"), "grad_ratio": grad.get("ratio"),
                    "grad_warn": grad.get("warn"), "probe_objective": grad.get("objective"),
                    "histogram": reply.histogram, "device": grad.get("device"),
                }.items() if value is not None},
            )
            hub.node_states[key] = {**(hub.node_states.get(key) or {}),
                                    **state.model_dump(mode="json")}
            states.append(state)

        for state in states:
            await hub.broadcast(state)
        return JSONResponse({"ok": True, "nodes": len(states)})

    @app.post("/api/l1")
    async def ingest_l1(payload: dict[str, Any]) -> JSONResponse:
        """Attach 모드의 사용자 프로세스가 밀어 넣는 L1 probe 결과(§3.5)."""
        hub.attached = True
        pushed = []
        for entry in payload.get("nodes") or []:
            node = entry.get("node")
            if not node:
                continue
            merged = dict(hub.node_states.get(node) or {})
            badges = dict(merged.get("badges") or {})
            for field in ("grad_norm", "grad_ratio", "warn", "histogram"):
                if entry.get(field) is not None:
                    badges[field] = entry[field]
            badges["probe_objective"] = payload.get("objective", "")
            numeric = entry.get("numeric") or {}
            state = proto.NodeState(
                node=node,
                axis="L1bwd",
                # grad 임계 위반과 NaN/Inf는 ok.warn이다(§5.3).
                state="ok.warn" if (entry.get("warn") or numeric.get("nan") or numeric.get("inf"))
                else "ok",
                spec=entry.get("spec") or merged.get("spec"),
                badges=badges,
            )
            hub.node_states[node] = state.model_dump(mode="json")
            pushed.append(state)

        for state in pushed:
            await hub.broadcast(state)
        return JSONResponse({"ok": True, "nodes": len(pushed), "loss": payload.get("loss")})

    @app.post("/api/scalars")
    def ingest_scalars(record: dict[str, Any]) -> JSONResponse:
        run_id = record.pop("run_id", None) or "run-attached"
        step = int(record.pop("step", 0))
        record.pop("wall", None)
        name = record.pop("run_name", None)
        hub.tracker.ensure_run(run_id, name=name)
        return JSONResponse({"ok": True, "written": hub.tracker.log(run_id, step, record)})

    @app.post("/api/runs")
    def create_run(request: dict[str, Any]) -> JSONResponse:
        """manifest와 함께 run을 연다. 관찰 모드의 첫 tf.log가 여기로 온다."""
        run_id = hub.tracker.ensure_run(
            request["run_id"], kind=request.get("kind", "exploratory"),
            name=request.get("name"), parent_run=request.get("parent_run"),
            manifest=request.get("manifest"))
        return JSONResponse({"ok": True, "run_id": run_id})

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        runs = [run.as_dict() for run in hub.tracker.runs()]
        for run in runs:
            run["keys"] = hub.tracker.keys(run["id"])
        return JSONResponse({"runs": runs})

    @app.get("/api/runs/curve")
    def read_curve(key: str, runs: str = "", aggregate: bool = False) -> JSONResponse:
        """곡선 하나 또는 시드 그룹의 평균과 표준편차."""
        ids = [item for item in runs.split(",") if item]
        if not ids:
            ids = [run.id for run in hub.tracker.runs(limit=8)]
        if aggregate:
            return JSONResponse({"key": key, "aggregate": hub.tracker.aggregate(ids, key)})
        return JSONResponse({"key": key, "series": [
            {"run": run_id, "points": hub.tracker.curve(run_id, key)} for run_id in ids
        ]})

    @app.get("/api/logs")
    def read_logs(limit: int = 300) -> JSONResponse:
        return JSONResponse({"logs": hub.tracker.tail(limit)})

    @app.post("/api/probes")
    def add_probe(spec: dict[str, Any]) -> JSONResponse:
        hub.probes.append(spec)
        return JSONResponse({"ok": True, "probes": hub.probes})

    @app.post("/api/memory")
    def memory(batch: int = 64, optimizer: str = "adam", amp: bool = False) -> JSONResponse:
        """메모리 밴드 추정(§6.2). 배치 슬라이더가 이 값을 읽는다."""
        if hub.store is None:
            return no_graph()
        if hub.traced:
            # 트레이스 그래프는 재인스턴스화가 안 되므로 활성값을 셀 수 없다.
            return JSONResponse({"ok": False, "error": "traced graph"})
        hub.kernel.ensure()
        replies = hub.kernel.request(
            proto.EstimateMemory(req_id=f"m-{hub.seq}", graph=hub.snapshot_for_kernel(),
                                 batch=batch, optimizer=optimizer, amp=amp)
        )
        estimate = next((r for r in replies if r.type == "MemoryEstimate"), None)
        if estimate is None:
            return JSONResponse({"ok": False, "error": "kernel returned no estimate"},
                                status_code=502)
        return JSONResponse(estimate.model_dump(mode="json", exclude_none=True))

    @app.websocket("/ws")
    async def websocket(socket: WebSocket) -> None:
        # BaseHTTPMiddleware는 WS scope를 지나치므로 여기서 직접 인증한다.
        given = socket.query_params.get("token") or socket.cookies.get(COOKIE_NAME)
        await socket.accept()
        if not token_matches(token, given):
            try:
                first = proto.decode_from_browser(await socket.receive_text())
            except Exception:
                first = None
            if first is None or first.type != "Auth" or not token_matches(token, first.token):
                await socket.close(code=4401)
                return

        hub.clients.append(socket)
        try:
            await socket.send_text(
                proto.Resync(
                    from_seq=0, to_seq=hub.seq,
                    ops=hub.store.log if hub.store is not None else [],
                    snapshot={"node_states": hub.node_states},
                ).model_dump_json(exclude_none=True)
            )
            while True:
                message = proto.decode_from_browser(await socket.receive_text())
                if message.type == "Op":
                    payload = message.model_dump(mode="json", exclude_none=True)
                    try:
                        seq = hub.apply_op(payload)
                    except OpError as exc:
                        await socket.send_text(
                            proto.Error(req_id=str(message.tmp_seq), kind="kernel", message=str(exc))
                            .model_dump_json(exclude_none=True)
                        )
                        continue
                    await socket.send_text(
                        proto.OpAck(tmp_seq=message.tmp_seq, seq=seq).model_dump_json()
                    )
                    await hub.broadcast(proto.OpBroadcast(seq=seq, op=payload))
                    # ponytail: 즉시 재실행. 80 ms 디바운스·슬라이더 코얼레싱은 M5.
                    for state in hub.run_l0():
                        await hub.broadcast(state)
                elif message.type == "Resync":
                    await socket.send_text(
                        proto.Resync(
                            from_seq=message.from_seq, to_seq=hub.seq,
                            ops=(hub.store.since(message.from_seq)
                                 if hub.store is not None else []),
                            snapshot={"node_states": hub.node_states},
                        ).model_dump_json(exclude_none=True)
                    )
        except WebSocketDisconnect:
            pass
        finally:
            if socket in hub.clients:
                hub.clients.remove(socket)

    # 프론트는 wheel 안에 정적으로 실린다(§9). API 라우트 뒤에 붙여야 가려지지 않는다.
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        @app.get("/")
        def index() -> FileResponse:
            # 에셋 파일명은 해시가 붙어 영구 캐시가 안전하지만 index.html은 아니다.
            # 캐시된 index가 사라진 번들을 가리키면 화면이 통째로 빈다.
            return FileResponse(static_dir / "index.html",
                                headers={"Cache-Control": "no-store"})

        app.mount("/", StaticFiles(directory=static_dir), name="static")

    return app
