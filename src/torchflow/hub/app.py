"""hub: FastAPI 애플리케이션 (기획서 §8.1, §8.2.2).

이 파일과 hub 패키지 전체는 torch를 import하지 않는다. shape 추론은 전부
L0 커널 프로세스에서 돈다. ``tests/test_hub.py``가 이 불변식을 강제한다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__, codegen, datasets, paper, protocol as proto
from ..ir import ModuleGraph, canonical_json, load, validate as ir_problems
from ..pysource import candidates, parse_example_spec
from .auth import DEFAULT_HOSTS, AuthMiddleware, COOKIE_NAME, extract_token, new_token, token_matches
from .engine import Engine
from .graphstore import GraphStore, OpError
from . import runs as l2
from .tracker import Tracker
from .kernels import KernelManager


class Hub:
    """그래프 상태 · 커널 · 접속 클라이언트를 한 곳에서 소유한다."""

    def __init__(self, ir: ModuleGraph | None, state_dir: Path, rt: dict[str, Any] | None = None,
                 path: Path | None = None):
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
        # 프로젝트 폴더. 열려 있으면 run·데이터·좌표·그래프가 전부 그 안에 산다(Figma 00 New project).
        self.project_dir: Path | None = None
        self.last_train_log = 0.0
        # L2 학습 run들. hub는 워커를 띄우고 파일을 읽을 뿐 학습을 돌리지 않는다(§5.5.3).
        self._runs_dir = Path.cwd() / "runs"
        self.l2: dict[str, l2.RunHandle] = {}
        self.testing: set[str] = set()        # 지금 테스트 프로세스가 도는 run - 겹쳐 띄우지 않는다
        # 내려받은 데이터셋 자리. hub는 파일 유무와 다운로드만 알고 적재는 워커가 한다.
        self._data_dir = Path.cwd() / "data"
        self.downloading: set[str] = set()    # 겹쳐 받으면 두 스레드가 같은 .part를 덮어쓴다
        # Attach 모드에서는 사용자 프로세스가 L1 커널이다(§8.1). hub는 커널을 띄우는
        # 대신 그쪽이 밀어 넣는 결과를 받아 브로드캐스트한다.
        self.attached = False
        self._graph_id: str | None = None
        self.total_params = 0
        self.rerun_ratio = 0.0
        self.layout_path = state_dir / "layout.json"
        self.layout = self._load_layout()
        if ir is not None:
            # 경로를 같이 넘긴다. 안 넘기면 CLI로 연 그래프가 저장된 파일이 아닌 것처럼
            # 보여서, 같은 파일을 다시 열 때 identity가 달라진다.
            self.open(ir, path)

    def open(self, ir: ModuleGraph, path: Path | None = None) -> None:
        """그래프를 연다. 첫 화면에서 템플릿을 고르면 이 경로로 들어온다."""
        self._graph_id = (ir.meta or {}).get("id") or _graph_id_for(path)
        self.store = GraphStore(ir, self.state_dir / "journal.jsonl")
        self.engine = Engine(ir)
        self.graph_path = path
        self.node_states.clear()
        self.total_params = 0
        # 마지막으로 저장한 seq. 새로고침해도 "저장 안 됨"이 살아 있어야 편집을 잃지 않는다.
        self.saved_seq = 0

    @property
    def runs_dir(self) -> Path:
        return self.project_dir / "runs" if self.project_dir else self._runs_dir

    @runs_dir.setter
    def runs_dir(self, path: Path) -> None:
        self._runs_dir = Path(path)

    @property
    def data_dir(self) -> Path:
        """데이터 창의 내려받기는 여기로 온다 - 프로젝트가 열려 있으면 그 폴더의 data/."""
        return self.project_dir / "data" if self.project_dir else self._data_dir

    @data_dir.setter
    def data_dir(self, path: Path) -> None:
        self._data_dir = Path(path)

    # 프로젝트 폴더 (Figma 00 New project: "그래프·레이아웃·런 기록은 프로젝트 폴더 하나에 모입니다")

    def save_project(self, target: Path) -> dict[str, Any]:
        """폴더 하나에 그래프·좌표·생성 코드·이 그래프의 run·붙은 데이터를 모은다."""
        assert self.store is not None
        target.mkdir(parents=True, exist_ok=True)
        self.store.save(target / "graph.tfg.json")
        self.graph_path = target / "graph.tfg.json"
        self.saved_seq = self.seq
        (target / "layout.json").write_text(
            json.dumps({"positions": self.positions()}, indent=2), encoding="utf-8")
        if not self.traced:
            try:
                (target / "model.py").write_text(
                    codegen.generate(self.store.ir, version=__version__, source="graph.tfg.json",
                                     specs=self.node_states), encoding="utf-8")
            except codegen.CodegenError:
                pass    # 오류 블록이 있는 그래프도 저장은 된다. 코드는 고친 뒤에 나온다.
        copied = 0
        old_runs, old_data = self.runs_dir, self.data_dir
        self.project_dir = target
        if old_runs != self.runs_dir and old_runs.is_dir():
            for job in old_runs.glob("*/job.json"):
                try:
                    owner = json.loads(job.read_text(encoding="utf-8")).get("graph_id")
                except (OSError, json.JSONDecodeError):
                    continue
                if owner == self.graph_id and not (self.runs_dir / job.parent.name).exists():
                    shutil.copytree(job.parent, self.runs_dir / job.parent.name)
                    copied += 1
        # 붙은 데이터가 밖에 있으면 프로젝트 안으로 가져온다 - 폴더째 옮겨도 학습이 돼야 한다.
        name = ((self.store.ir.experiment or {}).get("data") or {}).get("name")
        if name and (old_data / name).is_dir() and not (self.data_dir / name).exists():
            shutil.copytree(old_data / name, self.data_dir / name)
        self._remember_project(target)
        return {"dir": str(target), "runs": copied}

    def open_project(self, target: Path) -> None:
        ir = load(target / "graph.tfg.json")
        self.project_dir = target
        _fit_input_to_data(ir, self.data_dir)
        self.open(ir, target / "graph.tfg.json")
        try:
            layout = json.loads((target / "layout.json").read_text(encoding="utf-8"))
            self.update_layout(layout.get("positions") or {})
        except (OSError, json.JSONDecodeError):
            pass
        self.recover()
        self._remember_project(target)

    def _remember_project(self, target: Path) -> None:
        path = self.state_dir / "projects.json"
        try:
            known = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            known = []
        known = [entry for entry in known if entry.get("dir") != str(target)]
        known.insert(0, {"dir": str(target), "name": self.store.ir.graph.name if self.store else target.name,
                         "opened": time.time()})
        path.write_text(json.dumps(known[:20], ensure_ascii=False, indent=2), encoding="utf-8")

    def recent_projects(self) -> list[dict[str, Any]]:
        """열 수 있는 프로젝트. 기억해 둔 것과 projects/ 아래 폴더를 합친다."""
        path = self.state_dir / "projects.json"
        try:
            known = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            known = []
        seen = {entry["dir"]: entry for entry in known if Path(entry.get("dir", "")).joinpath("graph.tfg.json").is_file()}
        root = Path.cwd() / "projects"
        if root.is_dir():
            for graph in sorted(root.glob("*/graph.tfg.json")):
                folder = str(graph.parent)
                if folder not in seen:
                    seen[folder] = {"dir": folder, "name": _graph_name(graph), "opened": graph.stat().st_mtime}
        return sorted(seen.values(), key=lambda entry: entry.get("opened", 0), reverse=True)

    def _load_layout(self) -> dict[str, Any]:
        if self.layout_path.exists():
            try:
                return json.loads(self.layout_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass   # 좌표가 깨져도 그래프는 열려야 한다 - 자동 배치로 돌아간다.
        return {"positions": {}}

    def positions(self) -> dict[str, Any]:
        """지금 열린 그래프의 노드 좌표. 그래프마다 따로 둔다 - IN, OUT 같은 id는 그래프마다
        있어서 한 통에 넣으면 다른 그래프의 좌표가 새 그래프에 묻어 나온다(실제로 그랬다).
        예전의 평평한 ``positions``는 읽지 않는다 - 자동 배치로 돌아간다."""
        return self.layout.setdefault("graphs", {}).setdefault(self.graph_id or "_", {})

    def update_layout(self, positions: dict[str, Any]) -> None:
        self.positions().update(positions)
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
             "recipe": "200 ep · SGD 0.1 + cosine", "metric": "문헌값 ≈95.0 % · 앱에서 미검증",
             "file": "resnet18.tfg.json", "rt": {"num_classes": 10}},
            {"id": "minivit", "name": "MiniViT / CIFAR-10",
             "recipe": "설계 예제 · 검증 전", "metric": "—",
             "file": "minivit.tfg.json", "rt": {"num_classes": 10}},
            {"id": "mnist_cnn", "name": "MNIST CNN",
             "recipe": "2,000 step · AdamW 1e-3 · batch 64", "metric": "test 98.8 % · 재현됨",
             "file": "mnist_cnn.tfg.json"},
            {"id": "nanogpt", "name": "nanoGPT char / Shakespeare",
             "recipe": "5,000 iter · 6층 384 dim", "metric": "문헌값 val loss ≈1.47",
             "file": "nanogpt.tfg.json"},
        ]
        roots = [Path.cwd() / "examples", Path(__file__).resolve().parents[3] / "examples"]
        for entry in catalogue:
            found = next((root / entry["file"] for root in roots
                          if (root / entry["file"]).is_file()), None)
            entry["path"] = str(found) if found else ""
            entry["available"] = found is not None
        return catalogue

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        """열 수 있는 내 그래프들. 프로젝트 graph/ 와 작업 폴더를 훑는다(§10.1).

        제목은 파일 이름이 아니라 **그래프 이름**이다 - 이름을 바꿔도 파일 이름은
        따라가지 않으므로, 파일 이름만 보여 주면 바꾼 이름이 어디에도 안 보인다.
        """
        seen: dict[str, dict[str, Any]] = {}
        for root in (Path.cwd() / "graph", Path.cwd(), self.state_dir):
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.tfg.json")):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen[resolved] = {"path": resolved,
                                  "name": _graph_name(path),
                                  "file": path.name,
                                  "modified": path.stat().st_mtime,
                                  "where": str(root.resolve())}
        entries = sorted(seen.values(), key=lambda entry: entry["modified"], reverse=True)
        return entries[:limit]

    def registry(self) -> dict[str, Any]:
        path = self.state_dir / "registry.json"
        if not path.exists():
            return {"blocks": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def composites(self) -> list[dict[str, Any]]:
        """팔레트가 꺼내 쓸 묶음 블록. 템플릿(ResNet BasicBlock, MiniViT Block …)과 지금 그래프의
        컴포지트를 이름으로 모은다 - 이름이 겹치면 지금 그래프의 것이 이긴다."""
        found: dict[str, dict[str, Any]] = {}
        for entry in self.templates():
            if not entry["available"]:
                continue
            try:
                doc = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for name, body in (doc.get("composites") or {}).items():
                found[name] = {"name": name, "source": entry["name"], "doc": body.get("doc"),
                               "params": body.get("params") or {}, "ports": body.get("ports") or {},
                               "composite": body}
        if self.store is not None:
            for name, body in self.store.ir.composites.items():
                dumped = body.model_dump(mode="json", exclude_none=True)
                found[name] = {"name": name, "source": self.store.ir.graph.name, "doc": body.doc,
                               "params": dumped.get("params") or {}, "ports": dumped.get("ports") or {},
                               "composite": dumped}
        return sorted(found.values(), key=lambda item: item["name"])

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
        self.absorb_logs(replies)
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

    def sync_runs(self) -> None:
        """워커가 남긴 이벤트를 트래커로 옮긴다. 곡선을 물어볼 때마다 부른다."""
        # /api/train이 다른 스레드에서 l2에 넣는 동안 돌 수 있다 - 복사본을 돈다.
        for handle in list(self.l2.values()):
            l2.merge(handle, self.tracker)
            # 워커가 "적용했다"고 말한 것만 적는다. hub가 명령을 낸 step이 아니라
            # 실제로 반영된 step이 재현에 쓸 수 있는 숫자다.
            if handle.overrides_dirty:
                handle.overrides_dirty = False
                l2.write_overrides(handle.run_id, handle.extra["overrides"],
                                   self.runs_dir.parent)

    @property
    def graph_id(self) -> str | None:
        """지금 열린 그래프의 identity. run이 어느 그래프 것인지 가르는 값이다.

        IR 해시가 아니라 identity인 이유: 해시는 편집할 때마다 바뀌므로 그것으로
        묶으면 노드 하나 고친 순간 이전 run이 남이 된다. 비교하려고 남기는 것이
        곡선인데 그러면 쓸모가 없다.
        """
        return None if self.store is None else self._graph_id

    def occupied_devices(self) -> list[str]:
        """L2 워커가 잡고 있는 가속기. L1은 여기를 피해 배정한다(§5.1.5)."""
        return sorted({handle.device for handle in self.l2.values()
                       if handle.alive and handle.device and handle.device != "cpu"})

    def recover(self) -> list[str]:
        """hub가 다시 떴을 때 ``runs/``를 훑어 run을 다시 집는다(§5.5.3).

        워커는 hub와 분리된 세션이라 hub가 죽는 동안에도 계속 돌았다.
        ``events.jsonl``이 정본이므로 스칼라를 지우고 파일을 처음부터 다시 읽는다 -
        hub가 죽은 뒤에 쌓인 부분이 그래야 들어온다.
        """
        found = []
        for job in sorted(self.runs_dir.glob("*/job.json")):
            run_id = job.parent.name
            if run_id in self.l2:
                continue
            handle = l2.RunHandle(run_id=run_id, directory=job.parent)
            try:
                handle.extra["graph_id"] = json.loads(job.read_text(encoding="utf-8")).get("graph_id")
            except (OSError, json.JSONDecodeError):
                pass
            self.tracker.ensure_run(run_id, graph_id=handle.extra.get("graph_id"))
            self.tracker.clear_scalars(run_id)
            l2.remember_test(handle, l2.last_test(handle))
            self.l2[run_id] = handle
            l2.merge(handle, self.tracker)
            found.append(run_id)
        return found

    def absorb_logs(self, replies: list) -> None:
        """커널이 올린 노드별 stdout/stderr를 트래커에 남긴다(§5.6.1).

        하단 로그 탭과 Inspector 출력 탭이 같은 테이블을 읽는다.
        """
        for reply in replies:
            if reply.type == "Log" and reply.text:
                self.tracker.log_text(reply.text, node_id=reply.node_id,
                                      stream=reply.level_name)

    async def broadcast(self, message) -> None:
        payload = message.model_dump_json(exclude_none=True)
        for client in list(self.clients):
            try:
                await client.send_text(payload)
            except (WebSocketDisconnect, RuntimeError):
                self.clients.remove(client)


# 재시작이 필요한 변경을 사람 말로 옮긴 것. Phase B는 **표시만** 한다 - 자동
# 재시작은 신뢰를 깨므로 사람이 Stop하고 다시 누른다(ADR-05).
# 트레이스 그래프의 인자는 extra_repr에서 건진 표시용 문자열이라 모듈을 다시 만들 수 없고,
# 함수형 연산이 빠진 엣지로 forward를 쓰면 틀린 모델이 된다. 코드도 학습도 원본 .py의 몫이다.
TRACED_IS_READ_ONLY = "트레이스로 가져온 그래프는 코드를 만들지 않습니다. 원본 .py가 정본입니다"

RESTART_REASON = {
    "scheduler": "스케줄러 변경은 재생성이 필요합니다. Stop 후 다시 Run하세요",
    "restart": "가중치는 유지되지만 재시작이 필요합니다. Stop 후 다시 Run하세요",
    "cold": "데이터·시드 변경은 step 0부터 다시 돌려야 합니다",
}


def _apply_hparam(hub: "Hub", handle, options: dict[str, Any]) -> JSONResponse:
    """학습 중 hparam 변경(§5.7.1, §5.7.2).

    HOT이 아니면 재시작 필요를 알리고 아무것도 하지 않는다. HOT이어도 run이
    ``reported``면 in-place로 못 바꾼다 - 갈라진 run을 새로 만든다.
    """
    path = options.get("path") or "optim.lr"
    value = options["value"]
    change = l2.classify(path)
    if change != "hot":
        return JSONResponse({"ok": False, "restart_required": change, "path": path,
                             "message": RESTART_REASON[change]})

    run = hub.tracker.run(handle.run_id)
    entry = {"step": handle.step, "path": path, "value": value}
    if run is not None and run.kind == "reported":
        child = l2.fork(handle, root=hub.runs_dir, changes=l2.hot_command(path, value))
        child.extra["kind"] = "reported"
        child.extra["graph_id"] = handle.extra.get("graph_id")
        hub.l2[child.run_id] = child
        hub.tracker.ensure_run(child.run_id, kind="reported", parent_run=handle.run_id,
                               name=run.name, graph_id=run.graph_id,
                               manifest={**run.manifest, "parent_run": handle.run_id,
                                         "forked_at_step": handle.step})
        # 갈라진 run의 lr은 job.json에 박혀 있어 워커가 따로 알려 주지 않는다.
        # 갈라진 지점의 변경은 여기서만 기록할 수 있다.
        child.extra["overrides"] = [*handle.extra.get("overrides", []), entry]
        l2.write_overrides(child.run_id, child.extra["overrides"], hub.runs_dir.parent)
        l2.merge(handle, hub.tracker)
        return JSONResponse({"ok": True, "forked_from": handle.run_id, **child.as_dict()})

    # in-place 적용. overrides 파일은 워커가 반영을 알려 오면 sync_runs가 쓴다.
    l2.command(handle, **l2.hot_command(path, value))
    l2.merge(handle, hub.tracker)
    return JSONResponse({"ok": True, **handle.as_dict()})


def _fit_input_to_data(ir: ModuleGraph, data_dir: Path) -> None:
    """붙어 있는 데이터가 Input 규격을 정한다(§5.1.6).

    템플릿은 데이터를 달고 오는데(MNIST, CIFAR-10) Input에 적힌 모양이 다르면 첫 실행이
    "Input을 맞추세요"로 끝난다. 여는 순간 맞춰 두면 사람이 할 일이 없다.
    """
    name = ((ir.experiment or {}).get("data") or {}).get("name")
    spec = datasets.describe(name, data_dir) if name else None
    if not spec:
        return
    for node in ir.graph.nodes:
        if node.type == "torchflow.Input" and node.ports_out:
            port = node.ports_out[0]
            if list(port.shape or [])[1:] != list(spec["shape"]):
                port.shape = ["B", *spec["shape"]]


def _graph_id_for(path: Path | None) -> str:
    """``meta.id``가 없는 그래프의 identity.

    id 없이 저장된 그래프가 이미 있다. 그때 "가르지 않는다"로 떨어지면 그 그래프를
    열 때마다 남의 run이 전부 딸려 온다 - 이 함수가 없을 때 그랬다.

    파일 경로에서 만든다. hub를 다시 띄워도 같은 값이라 곡선이 그 그래프에 계속
    붙어 있는다. 저장 전 그래프는 경로가 없으므로 이 세션 동안만 유효한 값을 준다.
    """
    if path is None:
        return f"session-{uuid4().hex[:8].upper()}"
    digest = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    return f"path-{digest[:12]}"


def _class_name(name: str) -> str:
    from ..codegen import _class_name as convert

    return convert(name)


def _graph_name(path: Path) -> str:
    """파일에서 그래프 이름만 꺼낸다. 깨진 파일이면 파일 이름으로 대신한다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["graph"]["name"]
    except Exception:
        return path.name.removesuffix(".tfg.json")


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
    hub = Hub(load(graph_path) if graph_path else None, state_dir, rt,
              path=Path(graph_path) if graph_path else None)
    token = token or new_token()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # 커널 종료는 동기 호출이고 실패할 수도 있다. 여기서 막히거나 예외가 나면 uvicorn 종료가
        # 영영 끝나지 않으므로 시간을 재고 예외는 삼킨다 - 남은 프로세스는 OS가 거둔다.
        import asyncio

        def stop_kernels() -> None:
            for manager in (hub.kernel, hub.l1):
                try:
                    manager.stop(timeout=2.0)
                except Exception:
                    pass
        try:
            await asyncio.wait_for(asyncio.to_thread(stop_kernels), timeout=6.0)
        except (asyncio.TimeoutError, Exception):
            pass

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
        if hub.store is None:
            return no_graph()
        return JSONResponse({**hub.store.snapshot(), "dirty": hub.seq != hub.saved_seq,
                             "project": str(hub.project_dir) if hub.project_dir else None})

    @app.get("/api/registry")
    def registry() -> JSONResponse:
        # 잎 블록(torch.nn 리플렉션)에 묶음 블록(템플릿과 이 그래프의 컴포지트)을 얹는다.
        return JSONResponse({**hub.registry(), "composites": hub.composites()})

    @app.post("/api/ops")
    async def ops(op: dict[str, Any]) -> JSONResponse:
        """편집 op 하나. seq 규약은 WS와 동일하다(§8.2.2).

        보낸 클라이언트는 응답으로 결과를 받고, 붙어 있는 다른 클라이언트는
        WebSocket 브로드캐스트로 같은 편집을 본다.
        """
        if hub.store is None:
            return no_graph()
        try:
            seq = hub.apply_op(op)
        except OpError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        states = hub.run_l0()
        await hub.broadcast(proto.OpBroadcast(seq=seq, op=op))
        for state in states:
            await hub.broadcast(state)
        # 총계는 L0가 센 값이 정본이다. 편집마다 같이 보내야 상단 바가 다시 열 때까지 0으로 남지 않는다.
        return JSONResponse(
            {"seq": seq, "node_states": [s.model_dump(mode="json") for s in states],
             "total_params": hub.total_params}
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
            "recent": hub.recent(),
            "projects": hub.recent_projects(),
        })

    @app.post("/api/project/save")
    def save_project(request: dict[str, Any] | None = None) -> JSONResponse:
        """프로젝트 폴더에 전부 저장한다. 처음이면 projects/<이름>/ 을 만든다."""
        if hub.store is None:
            return no_graph()
        request = request or {}
        if request.get("dir"):
            target = Path(request["dir"]).expanduser()
        elif hub.project_dir is not None and not request.get("name"):
            target = hub.project_dir
        else:
            name = str(request.get("name") or hub.store.ir.graph.name or "untitled").strip()
            if not name or "/" in name or name.startswith("."):
                return JSONResponse({"error": f"프로젝트 이름으로 쓸 수 없습니다: {name!r}"}, status_code=400)
            target = Path.cwd() / "projects" / name
        if (target / "graph.tfg.json").is_file() and target != hub.project_dir:
            return JSONResponse({"error": f"{target.name}/ 에 이미 다른 프로젝트가 있습니다. 다른 이름을 쓰세요"},
                                status_code=409)
        return JSONResponse({"ok": True, **hub.save_project(target)})

    @app.post("/api/project/open")
    def open_project(request: dict[str, Any]) -> JSONResponse:
        target = Path(request.get("dir") or "").expanduser()
        if not (target / "graph.tfg.json").is_file():
            return JSONResponse({"error": f"프로젝트 폴더가 아닙니다 (graph.tfg.json 없음): {target}"}, status_code=404)
        try:
            hub.open_project(target)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
        return JSONResponse({"ok": True, "name": hub.store.ir.graph.name, "dir": str(target)})

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
        templates = {entry["path"]: entry for entry in hub.templates()}
        if str(path) not in templates and not path.is_file():
            return JSONResponse({"error": f"not found: {path}"}, status_code=404)
        # 템플릿은 런타임 상수를 스스로 들고 온다. IR에는 그 자리가 없고 첫 화면에는
        # --rt를 칠 곳이 없어서, 안 채우면 fc에서 "unresolved rt.num_classes"로 끝난다.
        hub.rt.update((templates.get(str(path)) or {}).get("rt") or {})
        try:
            ir = load(path)
            # 템플릿은 프로젝트 밖의 파일이다. 새로 열면 프로젝트 소속도 풀린다.
            hub.project_dir = None
            _fit_input_to_data(ir, hub.data_dir)
            hub.open(ir, path)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
        return JSONResponse({"ok": True, "name": hub.store.ir.graph.name})

    @app.get("/api/code")
    def read_code() -> JSONResponse:
        """Code 탭(§7.3 model-only). 디스크에 쓰지 않고 메모리에서 렌더한다(§7.6.2)."""
        if hub.store is None:
            return no_graph()
        if hub.traced:
            return JSONResponse({"error": TRACED_IS_READ_ONLY}, status_code=400)
        source = str(hub.graph_path) if hub.graph_path else "graph/model.tfg.json"
        try:
            code = codegen.generate(hub.store.ir, version=__version__, source=source,
                                    specs=hub.node_states)
        except codegen.CodegenError as exc:
            return JSONResponse({"error": str(exc), "node": exc.node_id}, status_code=400)
        return JSONResponse({"code": code, "ir_sha256": codegen.ir_hash(hub.store.ir),
                             "lines": len(code.splitlines()), "source": source})

    @app.get("/api/export")
    def export_figure(format: str = "svg", preset: str = paper.DEFAULT_PRESET,
                      anonymous: bool = False) -> Response:
        """논문용 아키텍처 다이어그램(§6.4). 흑백 안전, IR 해시가 메타데이터에 박힌다."""
        if hub.store is None:
            return no_graph()
        if format not in ("svg", "pdf"):
            return JSONResponse({"error": f"unknown format: {format}"}, status_code=400)

        figure = paper.build(
            hub.store.ir,
            node_states=hub.node_states,
            preset=preset,
            anonymous=anonymous,
            version=__version__,
        )
        name = (hub.store.ir.graph.name or "figure").replace(" ", "-")
        body = paper.to_svg(figure).encode("utf-8") if format == "svg" else paper.to_pdf(figure)
        return Response(
            content=body,
            media_type="image/svg+xml" if format == "svg" else "application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.{format}"'},
        )

    @app.post("/api/new")
    def new_graph(request: dict[str, Any] | None = None) -> JSONResponse:
        """빈 그래프에서 시작한다(§2.2의 진입점 다섯 중 하나)."""
        from ..ir import Graph

        request = request or {}
        # 데이터부터 시작하면 Input과 Output을 그 규격으로 깔아 준다 - 모양을 손으로 옮겨 적지 않는다.
        spec = datasets.describe(request["dataset"], hub.data_dir) if request.get("dataset") else None
        if request.get("dataset") and spec is None:
            return JSONResponse({"error": f"데이터셋을 찾지 못했습니다: {request['dataset']}"},
                                status_code=400)
        if spec is not None:
            spec = datasets.resolve(spec, request.get("recipe"))
        name = (request.get("name") or (spec["label"] if spec else "untitled")).strip() or "untitled"
        nodes = []
        if spec is not None:
            nodes = [{"id": "IN", "label": "x", "type": "torchflow.Input",
                      "ports_out": [{"name": "x", "type": "Tensor", "shape": ["B", *spec["shape"]],
                                     "dtype": "float32"}]},
                     {"id": "OUT", "label": "logits", "type": "torchflow.Output"}]
        # id를 여기서 박아 둔다. 없으면 새 프로젝트마다 run이 섞인다.
        meta = {"app_version": __version__, "id": uuid4().hex[:8].upper()}
        experiment = {"data": {"name": spec["name"], "recipe": spec["recipe"]}} if spec else None
        hub.project_dir = None      # 새 그래프는 아직 어느 폴더에도 속하지 않는다
        hub.open(ModuleGraph(meta=meta, experiment=experiment,
                             graph=Graph.model_validate({"name": name, "nodes": nodes})))
        return JSONResponse({"ok": True, "name": name, "dataset": spec and spec["name"]})

    @app.post("/api/delete")
    def delete_graph(request: dict[str, Any]) -> JSONResponse:
        """첫 화면 "내 그래프"에 뜬 파일만 지운다. 임의 경로는 받지 않는다."""
        path = str(Path(request["path"]).expanduser().resolve())
        if path not in {entry["path"] for entry in hub.recent()}:
            return JSONResponse({"error": f"not found: {path}"}, status_code=404)
        Path(path).unlink()
        return JSONResponse({"ok": True})

    @app.post("/api/close")
    def close_graph() -> JSONResponse:
        """그래프를 닫고 첫 화면으로 돌아간다(§2.2).

        커널은 살려 둔다 - 다음 그래프를 열 때 다시 기동하는 비용(콜드 <5 s)을
        치를 이유가 없다.
        """
        hub.store = None
        hub.engine = None
        hub.graph_path = None
        hub.node_states.clear()
        hub.total_params = 0
        return JSONResponse({"ok": True})

    @app.post("/api/save")
    def save_graph(request: dict[str, Any] | None = None) -> JSONResponse:
        """그래프를 파일로 쓴다. 경로가 없으면 state-dir 안에 만든다."""
        if hub.store is None:
            return no_graph()
        given = (request or {}).get("path")
        # 그래프는 프로젝트의 graph/ 에 산다(§10.1). state-dir는 캐시·journal·runs.db 자리다.
        # 이름을 바꿨으면 바꾼 이름의 파일로 간다 - 열려 있던 파일을 말없이 옮기지는 않는다.
        name = hub.store.ir.graph.name
        keep = hub.graph_path if (hub.graph_path
                                  and hub.graph_path.name == f"{name}.tfg.json") else None
        path = Path(given).expanduser() if given else (
            keep or Path.cwd() / "graph" / f"{name}.tfg.json")
        # 열려 있는 파일이 아닌데 이미 있으면 덮어쓰지 않는다 - 다른 그래프가 지워진다.
        if path.is_file() and (hub.graph_path is None
                               or path.resolve() != hub.graph_path.resolve()):
            return JSONResponse({"error": f"{path.name}이 이미 있습니다. 이름을 바꿔 저장하세요"},
                                status_code=409)
        problems = ir_problems(hub.store.ir)
        path.parent.mkdir(parents=True, exist_ok=True)
        hub.store.save(path)
        hub.graph_path = path
        hub.saved_seq = hub.seq
        return JSONResponse({"ok": True, "path": str(path), "problems": problems})

    @app.get("/api/layout")
    def read_layout() -> JSONResponse:
        return JSONResponse({"positions": hub.positions()})

    @app.post("/api/layout")
    def write_layout(patch: dict[str, Any]) -> JSONResponse:
        """노드 좌표를 저장한다.

        좌표는 IR이 아니라 ``graph/layout.json``에 산다(§10.1) - 그래서 노드를
        옮겨도 그래프 의미가 바뀌지 않고, git에서 ``merge=ours``로 충돌하지 않는다.
        """
        hub.update_layout(patch.get("positions") or {})
        return JSONResponse({"ok": True, "positions": len(hub.positions())})

    @app.post("/api/probe")
    async def run_probe(cfg: dict[str, Any] | None = None) -> JSONResponse:
        """IR 모드의 L1 probe. Attach 모드에서는 사용자 프로세스가 대신 밀어 넣는다."""
        if hub.store is None:
            return no_graph()
        if hub.attached:
            return JSONResponse({"ok": False, "error": "attached session drives L1"},
                                status_code=409)
        hub.l1.ensure()
        hub.sync_runs()
        replies = hub.l1.request(proto.RunClosure(
            req_id=f"p-{hub.seq}", graph=hub.snapshot_for_kernel(), probe_cfg=cfg or {},
            occupied=hub.occupied_devices()))

        hub.absorb_logs(replies)
        busy = next((r for r in replies if r.type == "Busy"), None)
        if busy is not None:
            return JSONResponse({"ok": False, "budget": busy.reason})

        states = []
        for reply in (r for r in replies if r.type == "Done"):
            grad = reply.grad or {}
            key = f"{reply.path}/{reply.node_id}" if reply.path else reply.node_id
            numeric = reply.numeric or {}
            # L0(심볼 shape)와 L1(실측 B)은 다른 축이다(§5.3). L1이 도착해도 노드의
            # shape는 L0의 것을 남기고 실측치는 배지로만 둔다 - 아니면 재접속
            # 스냅샷과 논문용 그림에서 [B, 3, 32, 32]가 [4, 3, 32, 32]로 굳는다.
            known = (hub.node_states.get(key) or {}).get("spec")
            state = proto.NodeState(
                seq=hub.seq, node=reply.node_id, path=reply.path, axis="L1bwd",
                state="ok.warn" if (grad.get("warn") or numeric.get("nan") or numeric.get("inf"))
                else "ok",
                spec=known or reply.spec,
                badges={key: value for key, value in {
                    "grad_norm": grad.get("norm"), "grad_ratio": grad.get("ratio"),
                    "grad_warn": grad.get("warn"), "probe_objective": grad.get("objective"),
                    "histogram": reply.histogram, "feature": reply.feature,
                    "device": grad.get("device"),
                    "measured": reply.spec,
                }.items() if value is not None},
            )
            hub.node_states[key] = {**(hub.node_states.get(key) or {}),
                                    **state.model_dump(mode="json")}
            states.append(state)

        for state in states:
            await hub.broadcast(state)
        # 어느 디바이스에서 돌았는지, 무엇이 실패했는지 돌려준다. 학습이 GPU를 잡고 있으면
        # L1은 CPU forward-only로 내려가고(§5.1.5) backward가 실패해도 forward 결과는
        # 남는데, 그 사실이 화면에 안 보이면 사람은 배지가 사라진 것을 고장으로 읽는다.
        failure = next((r for r in replies if r.type == "Error"), None)
        last = next((r for r in reversed(replies) if r.type == "Done"), None)
        if last is None:
            return JSONResponse({"ok": False, "nodes": 0,
                                 "error": failure.message if failure else "probe returned nothing",
                                 "node": failure.node_id if failure else None})
        badge = last.grad or {}
        return JSONResponse({"ok": True, "nodes": len(states),
                             "device": badge.get("device"),
                             "forward_only": badge.get("forward_only"),
                             **({"error": failure.message} if failure else {})})

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
        hub.tracker.ensure_run(run_id, name=name, graph_id=hub.graph_id)
        return JSONResponse({"ok": True, "written": hub.tracker.log(run_id, step, record)})

    @app.post("/api/runs")
    def create_run(request: dict[str, Any]) -> JSONResponse:
        """manifest와 함께 run을 연다. 관찰 모드의 첫 tf.log가 여기로 온다."""
        run_id = hub.tracker.ensure_run(
            request["run_id"], kind=request.get("kind", "exploratory"),
            name=request.get("name"), parent_run=request.get("parent_run"),
            graph_id=hub.graph_id,
            manifest=request.get("manifest"))
        return JSONResponse({"ok": True, "run_id": run_id})

    @app.get("/api/datasets")
    def list_datasets() -> JSONResponse:
        """내장 데이터셋과 data/ 아래 사용자 데이터. 다운로드는 명시 버튼으로만(§3.1)."""
        return JSONResponse({"datasets": datasets.scan(hub.data_dir)})

    @app.post("/api/datasets/{name}/preview")
    def preview_dataset(name: str, request: dict[str, Any] | None = None) -> JSONResponse:
        """정제 화면: 레시피를 적용한 명세와 미리보기. 적재는 하지 않는다."""
        spec = datasets.describe(name, hub.data_dir)
        if spec is None:
            return JSONResponse({"error": f"데이터셋을 찾지 못했습니다: {name}"}, status_code=404)
        return JSONResponse(datasets.preview(spec, hub.data_dir, (request or {}).get("recipe")))

    @app.post("/api/data")
    async def set_graph_data(request: dict[str, Any]) -> JSONResponse:
        """그래프에 데이터와 레시피를 붙이고 Input 규격을 맞춘다.

        레시피는 IR의 ``experiment.data``에 남아 저장·재현에 따라간다. Input은 평범한
        ``set_ports`` op로 고치므로 실행 취소가 되고 다른 클라이언트도 본다.
        """
        if hub.store is None:
            return no_graph()
        name = request.get("name") or ""
        # 합성 과제로 되돌리기: experiment.data를 떼고 Input 규격은 건드리지 않는다
        # (합성은 Input의 shape를 그대로 쓴다). 데이터 창의 "합성 과제" 항목이 이리로 온다.
        if name in ("teacher", "noise"):
            experiment = dict(hub.store.ir.experiment or {})
            experiment.pop("data", None)
            hub.store.ir.experiment = experiment
            return JSONResponse({"ok": True, "spec": {"name": name, "recipe": {}},
                                 "seq": hub.seq, "node_states": []})
        spec = datasets.describe(name, hub.data_dir)
        if spec is None:
            return JSONResponse({"error": f"데이터셋을 찾지 못했습니다: {name}"}, status_code=404)
        effective = datasets.resolve(spec, request.get("recipe"))
        ir = hub.store.ir
        ir.experiment = {**(ir.experiment or {}),
                         "data": {"name": name, "recipe": effective["recipe"]}}
        entry = next((node for node in ir.graph.nodes if node.type == "torchflow.Input"), None)
        states = []
        if entry is not None and entry.ports_out and request.get("apply_input", True):
            port = entry.ports_out[0]
            wanted = ["B", *effective["shape"]]
            if list(port.shape or []) != wanted:
                op = {"client_id": "hub", "tmp_seq": 0, "kind": "set_ports",
                      "payload": {"node": entry.id,
                                  "ports_out": [{"name": port.name, "type": port.type,
                                                 "shape": wanted, "dtype": port.dtype or "float32"}]}}
                seq = hub.apply_op(op)
                states = hub.run_l0()
                await hub.broadcast(proto.OpBroadcast(seq=seq, op=op))
                for state in states:
                    await hub.broadcast(state)
        return JSONResponse({"ok": True, "spec": effective, "seq": hub.seq,
                             "node_states": [s.model_dump(mode="json") for s in states]})

    @app.put("/api/datasets/{name}/files")
    async def upload_dataset_file(name: str, request: Request, path: str = "") -> JSONResponse:
        """브라우저에서 끌어다 놓은 데이터 파일 하나를 data/<name>/<path>에 쓴다.

        multipart 없이 본문이 곧 파일이다 - 의존성 하나를 아끼고 폴더 하나에 수백 장이어도
        요청 수백 개로 충분하다(로컬이다).
        """
        relative = Path(path)
        if not name or "/" in name or name.startswith(".") or relative.is_absolute() \
                or ".." in relative.parts or not path:
            return JSONResponse({"error": f"쓸 수 없는 경로: {name}/{path}"}, status_code=400)
        target = hub.data_dir / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await request.body())
        return JSONResponse({"ok": True, "path": str(target)})

    @app.post("/api/datasets")
    def add_dataset_folder(request: dict[str, Any]) -> JSONResponse:
        """다른 곳의 폴더를 data/ 에 링크로 등록한다. 복사하지 않는다."""
        source = Path(request.get("path") or "").expanduser()
        if not source.is_dir():
            return JSONResponse({"error": f"폴더가 아닙니다: {source}"}, status_code=400)
        spec = datasets.inspect(source)
        if spec is None:
            return JSONResponse({"error": "이미지 폴더(클래스별 하위 폴더), CSV, x.npy+y.npy 중 "
                                          "무엇도 찾지 못했습니다"}, status_code=400)
        hub.data_dir.mkdir(parents=True, exist_ok=True)
        link = hub.data_dir / source.name
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
        return JSONResponse({"ok": True, **datasets.describe(source.name, hub.data_dir)})

    @app.post("/api/datasets/{name}/download")
    def download_dataset(name: str) -> JSONResponse:
        if name not in datasets.CATALOGUE:
            return JSONResponse({"error": f"unknown dataset {name}"}, status_code=404)
        if name in hub.downloading:
            return JSONResponse({"error": "이미 내려받는 중입니다"}, status_code=409)
        hub.downloading.add(name)
        try:
            written = datasets.download(name, hub.data_dir)
        except OSError as exc:
            return JSONResponse({"error": f"내려받지 못했습니다: {exc}"}, status_code=502)
        finally:
            hub.downloading.discard(name)
        return JSONResponse({"ok": True, "files": [str(path) for path in written]})

    @app.get("/api/datasets/{name}/progress")
    def dataset_progress(name: str) -> JSONResponse:
        """받은 바이트 / 예상 총량. ``active``가 거짓인데 바이트가 남아 있으면 끊긴 다운로드다."""
        if name not in datasets.CATALOGUE:
            return JSONResponse({"error": f"unknown dataset {name}"}, status_code=404)
        return JSONResponse({**datasets.progress(name, hub.data_dir),
                             "active": name in hub.downloading})

    training_lock = threading.Lock()

    @app.post("/api/train")
    def start_training(http: Request, request: dict[str, Any] | None = None) -> JSONResponse:
        # 누가 눌렀는지 터미널에 남기되, 거부되는 반복 요청은 초당 한 줄로 접는다 - 키 자동 반복으로
        # 초당 30번 들어온 요청이 터미널을 도배해 사람이 놀랐다.
        now = time.time()
        if now - hub.last_train_log >= 1.0:
            print(f"[train] 시작 요청 {http.client.host if http.client else '?'} "
                  f"{(http.headers.get('user-agent') or '')[:40]}", flush=True)
            hub.last_train_log = now
        if hub.store is None:
            return no_graph()
        if hub.traced:
            return JSONResponse({"error": TRACED_IS_READ_ONLY}, status_code=400)
        options = request or {}

        graph = hub.store.ir.graph
        # 모양 계산이 실패한 블록이 있으면 워커도 첫 forward에서 같은 자리에서 죽는다. 그러면
        # 사람은 stdout을 뒤져야 한다 - 여기서 블록 이름을 대고 멈춘다.
        broken = [node.label or node.id for node in graph.nodes
                  if (hub.node_states.get(node.id) or {}).get("state") == "error"]
        if broken:
            return JSONResponse({"error": f"모델 오류를 먼저 고치세요: {', '.join(broken)} "
                                          "(블록을 누르면 오른쪽에 진단이 보입니다)",
                                 "broken": broken}, status_code=400)
        # 한 그래프에 학습은 한 번에 하나다. 화면이 잘못 반복해 눌러도 워커가 쌓이면 안 된다 -
        # 실제로 run 200개가 한꺼번에 떠서 기계가 멈출 뻔했다.
        busy = next((handle for handle in list(hub.l2.values())
                     if handle.alive and handle.extra.get("graph_id") == hub.graph_id), None)
        if busy is not None:
            return JSONResponse({"error": f"이미 학습이 돌고 있습니다: {busy.run_id} "
                                          f"(step {busy.step}). 멈추거나 끝난 뒤 시작하세요",
                                 "run_id": busy.run_id}, status_code=409)
        trainer = next((node for node in graph.nodes if node.type == "torchflow.Train"), None)
        if trainer is not None:
            # 학습 블록이 있으면 그 값이 기본이다. 요청이 준 값(smoke의 짧은 steps 등)이 이긴다.
            options = {**{key: value for key, value in (trainer.args or {}).items()
                          if isinstance(value, (int, float, str, bool))}, **options}
        entry = next((node for node in graph.nodes if node.type == "torchflow.Input"), None)
        if entry is None or not entry.ports_out:
            return JSONResponse(
                {"error": "Input 노드와 입력 규격이 있어야 학습할 수 있습니다"}, status_code=400)

        # 실제 데이터셋이면 워커를 띄우기 전에 규격을 맞춰 본다. 워커 안에서 터지면 사람은
        # stdout을 뒤져야 하고, 여기서 말하면 Input 노드를 고치면 된다.
        dataset = options.get("dataset", "teacher")
        spec = datasets.describe(dataset, hub.data_dir) if dataset not in ("teacher", "noise") else None
        if dataset not in ("teacher", "noise") and spec is None:
            return JSONResponse({"error": f"데이터셋을 찾지 못했습니다: {dataset}"}, status_code=400)
        recipe = options.get("recipe")
        attached = ((hub.store.ir.experiment or {}).get("data") or {})
        if recipe is None and attached.get("name") == dataset:
            recipe = attached.get("recipe")
        if spec is not None:
            spec = datasets.resolve(spec, recipe)
            if spec.get("problem"):
                return JSONResponse({"error": f"{spec['label']}: {spec['problem']}"}, status_code=400)
            given = list(entry.ports_out[0].shape or [])
            wanted = ", ".join(str(dim) for dim in ["B", *spec["shape"]])
            if given[1:] != spec["shape"]:
                port = entry.ports_out[0]
                return JSONResponse({"error": f"{spec['label']}의 입력은 [{wanted}]입니다. "
                                              f"Input 노드의 규격을 {wanted}로 바꾸세요 "
                                              f"(지금 [{', '.join(map(str, given))}])",
                                     # 화면이 버튼 하나로 고칠 수 있게 op 재료를 같이 준다.
                                     "fix": {"node": entry.id,
                                             "ports_out": [{"name": port.name, "type": port.type,
                                                            "shape": ["B", *spec["shape"]],
                                                            "dtype": port.dtype or "float32"}]}},
                                    status_code=400)
            sink = next((node for node in graph.nodes if node.type == "torchflow.Output"), None)
            logits = ((hub.node_states.get(sink.id) or {}).get("spec") or {}).get("shape") if sink else None
            if logits and logits[-1] != spec["classes"]:
                return JSONResponse({"error": f"{spec['label']}는 {spec['classes']}개 클래스입니다. "
                                              f"출력이 [B, {spec['classes']}]여야 합니다 (지금 {logits})"},
                                    status_code=400)
            if not spec["available"]:
                return JSONResponse({"error": f"{spec['label']} 파일이 없습니다. 먼저 내려받으세요",
                                     "download": dataset}, status_code=400)

        try:
            code = codegen.generate(hub.store.ir, version=__version__,
                                    source=str(hub.graph_path or "graph/model.tfg.json"))
        except codegen.CodegenError as exc:
            return JSONResponse({"error": str(exc), "node": exc.node_id}, status_code=400)

        # Smoke는 "지금 이 그래프가 학습이 되긴 하는가"를 30초 안에 답하는 것이다
        # (§13.1 M7). 짧고 결정적이라 두 번 돌리면 loss가 bitwise로 같아야 한다.
        smoke = bool(options.get("smoke"))
        if smoke:
            options = {**options, "steps": 20, "batch": 8, "log_every": 1}

        run_id = options.get("run_id") or l2.new_run_id()
        # 생성 코드가 실제로 받는 인자만 넘긴다 - 그래프가 rt.num_classes를 안 쓰면
        # 생성자에도 그 인자가 없다.
        accepted = codegen.model_params(hub.store.ir)
        model_args = {name: spec["default"] for name, spec in accepted.items()
                      if spec.get("default") is not None}
        model_args.update({key: value for key, value in hub.rt.items() if key in accepted})
        model_args.update({key: value for key, value in (options.get("hparams") or {}).items()
                           if key in accepted})
        missing = [name for name in accepted if name not in model_args]
        if missing:
            return JSONResponse(
                {"error": f"값이 없는 인자: {', '.join(missing)} (--rt로 주거나 hparam 기본값을 넣으세요)"},
                status_code=400)
        model_args["seed"] = int(options.get("seed", 0))

        job = {
            "class_name": _class_name(graph.name),
            "model_args": model_args,
            "input_shape": list(entry.ports_out[0].shape or []),
            "num_classes": spec["classes"] if spec else int(hub.rt.get("num_classes", 10)),
            "dataset": dataset,
            "data_dir": str(hub.data_dir),
            "recipe": spec["recipe"] if spec else None,
            "eval_every": int(options.get("eval_every", 100)),
            "batch": int(options.get("batch", 32)),
            "steps": int(options.get("steps", 200)),
            "optimizer": options.get("optimizer", "adamw"),
            # 스케줄러가 있으면 lr은 base다 - 실제 lr은 base x schedule(§5.7.1).
            "scheduler": options.get("scheduler", "none"),
            "warmup_steps": int(options.get("warmup_steps", 0)),
            "lr": float(options.get("lr", 1e-3)),
            "weight_decay": float(options.get("weight_decay", 0.0)),
            "log_every": int(options.get("log_every", 5)),
            "nan_policy": options.get("nan_policy", "pause"),
            "seed": int(options.get("seed", 0)),
            "device": options.get("device", "auto"),
            "smoke": smoke,
            # 복구가 이 파일만 보고 run의 소속을 알 수 있어야 한다.
            "graph_id": hub.graph_id,
        }

        kind = options.get("kind", "exploratory")
        handle = l2.start(run_id=run_id, root=hub.runs_dir, job=job, code=code)
        handle.extra.update({"kind": kind, "smoke": smoke, "graph_id": hub.graph_id})
        hub.l2[run_id] = handle
        hub.tracker.ensure_run(run_id, kind=kind, name=graph.name, graph_id=hub.graph_id,
                               manifest={"job": job, "ir_sha256": codegen.ir_hash(hub.store.ir)})
        return JSONResponse({"ok": True, **handle.as_dict()})

    @app.post("/api/train/{run_id}")
    def control_training(run_id: str, request: dict[str, Any] | None = None) -> JSONResponse:
        """pause · resume · stop · lr 변경(§5.7.1 HOT)."""
        handle = hub.l2.get(run_id)
        if handle is None:
            return JSONResponse({"error": f"unknown run {run_id}"}, status_code=404)
        options = request or {}
        cmd = options.get("cmd", "")
        if cmd == "pause":
            l2.command(handle, pause=True)
        elif cmd == "resume":
            if handle.alive:
                l2.command(handle, pause=False)
            else:
                # 워커가 이미 끝났거나 hub와 함께 죽었다 - 체크포인트에서 다시 띄운다.
                l2.resume(handle, steps=options.get("steps"))
                hub.tracker.ensure_run(run_id)
        elif cmd == "stop":
            l2.command(handle, stop=True)
        elif cmd == "promote":
            # exploratory -> reported. 이 뒤로는 in-place 변경이 fork가 된다(§5.7.2).
            hub.tracker.set_kind(run_id, "reported")
            handle.extra["kind"] = "reported"
        elif cmd in ("set_lr", "set_hparam"):
            return _apply_hparam(hub, handle, options)
        else:
            return JSONResponse({"error": f"unknown cmd {cmd!r}"}, status_code=400)
        l2.merge(handle, hub.tracker)
        return JSONResponse({"ok": True, **handle.as_dict()})

    @app.get("/api/train/{run_id}/stdout")
    def training_stdout(run_id: str, offset: int = 0, tail: int = 40_000) -> JSONResponse:
        """워커 프로세스가 찍은 것 그대로. 파이썬이 낸 출력이 정본이다.

        커서를 돌려주므로 클라이언트는 새로 늘어난 부분만 받아 이어 붙인다.
        """
        handle = hub.l2.get(run_id)
        if handle is None:
            return JSONResponse({"error": f"unknown run {run_id}"}, status_code=404)
        path = handle.directory / "stdout.log"
        if not path.exists():
            return JSONResponse({"text": "", "offset": 0})
        size = path.stat().st_size
        # 처음 열 때 수십 MB를 통째로 보내지 않는다 - 사람이 보는 것은 꼬리다.
        start = max(size - tail, 0) if offset <= 0 else min(offset, size)
        with path.open("rb") as stream:
            stream.seek(start)
            chunk = stream.read()
        return JSONResponse({"text": chunk.decode("utf-8", "replace"),
                             "offset": start + len(chunk)})

    @app.post("/api/test/{run_id}")
    def test_run(run_id: str) -> JSONResponse:
        """run의 체크포인트를 학습이 보지 않은 분할에 돌린다. 끝날 때까지 기다린다(스레드풀)."""
        handle = hub.l2.get(run_id)
        if handle is None:
            return JSONResponse({"error": f"unknown run {run_id}"}, status_code=404)
        if run_id in hub.testing:
            return JSONResponse({"error": "이미 테스트가 돌고 있습니다"}, status_code=409)
        hub.testing.add(run_id)
        try:
            result = l2.test(handle)
        finally:
            hub.testing.discard(run_id)
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @app.get("/api/test/{run_id}")
    def last_test_result(run_id: str) -> JSONResponse:
        """마지막 테스트 결과. 탭을 다시 열어도 다시 돌리지 않는다."""
        handle = hub.l2.get(run_id)
        if handle is None:
            return JSONResponse({"error": f"unknown run {run_id}"}, status_code=404)
        result = l2.last_test(handle)
        if result is None:
            return JSONResponse({"error": "아직 테스트하지 않았습니다"}, status_code=404)
        return JSONResponse(result)

    @app.get("/api/train")
    def training_status() -> JSONResponse:
        """돌고 있는 run들의 상태. 곡선은 /api/runs/curve가 준다.

        다른 그래프의 run은 빼고 준다 - 워커는 계속 돌지만 이 화면의 것이 아니다.
        """
        hub.sync_runs()
        return JSONResponse({"runs": [
            handle.as_dict() for handle in hub.l2.values()
            if hub.graph_id is None or handle.extra.get("graph_id") == hub.graph_id]})

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        hub.sync_runs()
        runs = [run.as_dict() for run in hub.tracker.runs(graph_id=hub.graph_id)]
        for run in runs:
            run["keys"] = hub.tracker.keys(run["id"])
        return JSONResponse({"runs": runs})

    @app.get("/api/runs/curve")
    def read_curve(key: str, runs: str = "", aggregate: bool = False) -> JSONResponse:
        hub.sync_runs()
        """곡선 하나 또는 시드 그룹의 평균과 표준편차."""
        ids = [item for item in runs.split(",") if item]
        if not ids:
            ids = [run.id for run in hub.tracker.runs(limit=8, graph_id=hub.graph_id)]
        if aggregate:
            return JSONResponse({"key": key, "aggregate": hub.tracker.aggregate(ids, key)})
        return JSONResponse({"key": key, "series": [
            {"run": run_id, "points": hub.tracker.curve(run_id, key)} for run_id in ids
        ]})

    @app.get("/api/logs")
    def read_logs(limit: int = 300, node: str = "") -> JSONResponse:
        return JSONResponse({"logs": hub.tracker.tail(limit, node or None)})

    @app.post("/api/detail")
    def node_detail(request: dict[str, Any]) -> JSONResponse:
        """블록 상세 탭(§6.3). L1 커널의 마지막 probe 값으로 그린다."""
        if hub.attached or not hub.l1.alive():
            return JSONResponse({"ok": False, "error": "Probe를 한 번 돌리면 이 블록이 무엇을 했는지 보입니다"})
        node = request.get("node") or ""
        path, _, node_id = node.rpartition("/")
        replies = hub.l1.request(proto.NodeDetail(req_id=f"d-{hub.seq}", node_id=node_id, path=path))
        result = next((r for r in replies if r.type == "NodeDetailResult"), None)
        if result is None:
            return JSONResponse({"ok": False, "error": "커널이 응답하지 않았습니다"}, status_code=502)
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    @app.post("/api/eval")
    def eval_expression(request: dict[str, Any]) -> JSONResponse:
        """Debug Console(§5.6.1). 표현식은 L1 커널의 마지막 probe 위에서 평가된다."""
        if hub.attached:
            return JSONResponse(
                {"ok": False, "error": "Attach 모드의 값은 사용자 프로세스 안에 있습니다 "
                                       "(콘솔은 v1)"}, status_code=409)
        if not hub.l1.alive():
            return JSONResponse({"ok": False, "error": "probe를 한 번 돌려 주세요"})
        node = request.get("node") or ""
        path, _, node_id = node.rpartition("/")
        replies = hub.l1.request(proto.Eval(
            req_id=f"e-{hub.seq}", expr=request.get("expr") or "", node_id=node_id, path=path))
        result = next((r for r in replies if r.type == "EvalResult"), None)
        if result is None:
            return JSONResponse({"ok": False, "error": "커널이 응답하지 않았습니다"}, status_code=502)
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

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
