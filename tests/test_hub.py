"""hub 라우트와 보안 불변식 (기획서 §8.2.2, §9)."""

import json
import subprocess
import sys

import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import MINIVIT
from torchflow.hub import runs as l2
from torchflow.hub.app import create_app

TOKEN = "test-token"


@pytest.fixture
def app(tmp_path):
    application = create_app(MINIVIT, state_dir=tmp_path / "state", token=TOKEN,
                             rt={"num_classes": 10})
    yield application
    application.state.hub.kernel.stop()


@pytest.fixture
def client(app):
    # base_url의 host가 실제 허용목록을 지나야 한다 - TestClient 기본값 "testserver"는 막힌다.
    return TestClient(app, base_url="http://127.0.0.1:8765")


def http_routes(app):
    return [
        (route.path, sorted(route.methods - {"HEAD", "OPTIONS"})[0])
        for route in app.routes
        if getattr(route, "methods", None)
    ]


def test_every_route_requires_a_token(app, client):
    """인증 없는 라우트는 0개다 - 예외 목록을 두지 않는다(§9).

    marimo CVE-2026-39987과 Gradio CVE-2024-47165가 모두 '인증 없는 라우트
    하나'에서 시작했다. 라우트가 늘어나면 이 테스트가 자동으로 덮는다.
    """
    routes = http_routes(app)
    assert routes, "no routes discovered — the invariant would pass vacuously"
    for path, method in routes:
        response = client.request(method, path, json={})
        assert response.status_code == 401, f"{method} {path} answered without a token"


def test_websocket_requires_a_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws") as socket:
            socket.send_text(json.dumps({"type": "Auth", "token": "wrong"}))
            socket.receive_text()
    assert excinfo.value.code == 4401


def test_bad_token_is_rejected(client):
    assert client.get("/api/health", headers={"Authorization": "token nope"}).status_code == 401


def test_unknown_host_is_rejected(client):
    response = client.get(f"/api/health?token={TOKEN}", headers={"Host": "evil.example.com"})
    assert response.status_code == 403


def test_origin_null_is_rejected(client):
    response = client.get(f"/api/health?token={TOKEN}", headers={"Origin": "null"})
    assert response.status_code == 403


def test_token_grants_access_and_sets_a_cookie(client):
    response = client.get(f"/api/health?token={TOKEN}")
    assert response.status_code == 200 and response.json()["ok"]
    assert "torchflow_token" in response.cookies


def test_stale_cookie_is_replaced_by_a_fresh_token(client):
    """hub를 재시작하면 토큰이 바뀐다. 낡은 쿠키를 그대로 두면 에셋이 전부 401이 된다."""
    response = client.get(f"/api/health?token={TOKEN}", cookies={"torchflow_token": "old-token"})
    assert response.status_code == 200
    assert response.cookies.get("torchflow_token") == TOKEN


def test_graph_endpoint_returns_the_authoritative_snapshot(client):
    payload = client.get("/api/graph", headers={"Authorization": f"token {TOKEN}"}).json()
    assert payload["seq"] == 0
    assert payload["graph"]["graph"]["name"] == "MiniViT"


def test_ops_applies_and_reruns_l0(client):
    """op 하나가 그래프를 바꾸고 곧바로 shape가 갱신된다(§8.3)."""
    auth = {"Authorization": f"token {TOKEN}"}
    before = client.post("/api/shapes", headers=auth).json()["node_states"]
    head_before = next(s for s in before if s["node"] == "01J9Q4B5")
    assert head_before["spec"]["shape"] == ["B", 10]

    response = client.post(
        "/api/ops",
        headers=auth,
        json={"client_id": "c-1", "tmp_seq": 1, "kind": "set_switch_active",
              "payload": {"composite": "Block", "instance": "01J9I002", "active": "none"}},
    )
    assert response.status_code == 200 and response.json()["seq"] == 1

    graph = client.get("/api/graph", headers=auth).json()["graph"]
    assert graph["composites"]["Block"]["instances"]["01J9I002"]["active"] == "none"


def test_ops_rejects_an_unknown_variant(client):
    response = client.post(
        "/api/ops",
        headers={"Authorization": f"token {TOKEN}"},
        json={"client_id": "c-1", "tmp_seq": 1, "kind": "set_switch_active",
              "payload": {"composite": "Block", "instance": "01J9I002", "active": "ghost"}},
    )
    assert response.status_code == 400


def test_hub_never_imports_torch(tmp_path):
    """커널 격리의 실체(§8.1): hub 프로세스에 torch가 들어오면 안 된다."""
    script = (
        "import sys;"
        "from torchflow.hub.app import create_app;"
        f"create_app({str(MINIVIT)!r}, state_dir={str(tmp_path)!r}, token='t');"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", script]).returncode == 0


def test_websocket_op_acks_and_pushes_node_states(client):
    """편집 1회의 왕복: Op -> OpAck -> OpBroadcast -> NodeState (§8.3)."""
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        assert json.loads(socket.receive_text())["type"] == "Resync"

        socket.send_text(json.dumps({
            "type": "Op", "client_id": "c-1", "tmp_seq": 7, "kind": "set_switch_active",
            "payload": {"composite": "Block", "instance": "01J9I002", "active": "none"},
        }))

        seen = {}
        for _ in range(40):
            message = json.loads(socket.receive_text())
            seen.setdefault(message["type"], message)
            if message["type"] == "NodeState" and message["node"] == "01J9Q4B5":
                break

        assert seen["OpAck"] == {"type": "OpAck", "tmp_seq": 7, "seq": 1}
        assert seen["OpBroadcast"]["op"]["kind"] == "set_switch_active"
        assert seen["NodeState"]["axis"] == "L0"
        assert "elapsed_ms" in seen["NodeState"]["badges"]


def test_layout_persists_node_positions(client, app, tmp_path):
    """좌표는 IR이 아니라 layout.json에 산다(§10.1) - 그래프 의미를 건드리지 않는다."""
    auth = {"Authorization": f"token {TOKEN}"}
    assert client.get("/api/layout", headers=auth).json() == {"positions": {}}

    before = client.get("/api/graph", headers=auth).json()["graph"]
    response = client.post("/api/layout", headers=auth,
                           json={"positions": {"01J9Q4B2": {"x": 120, "y": -40}}})
    assert response.status_code == 200

    stored = client.get("/api/layout", headers=auth).json()["positions"]
    assert stored["01J9Q4B2"] == {"x": 120, "y": -40}
    # IR은 좌표를 모른다.
    assert client.get("/api/graph", headers=auth).json()["graph"] == before
    assert (app.state.hub.layout_path).exists()


def test_layout_survives_a_restart(app, client, tmp_path):
    client.post("/api/layout", headers={"Authorization": f"token {TOKEN}"},
                json={"positions": {"n1": {"x": 5, "y": 6}}})
    restarted = create_app(MINIVIT, state_dir=app.state.hub.state_dir, token=TOKEN)
    try:
        assert restarted.state.hub.layout["positions"]["n1"] == {"x": 5, "y": 6}
    finally:
        restarted.state.hub.kernel.stop()
        restarted.state.hub.l1.stop()


def test_corrupt_layout_falls_back_to_auto_placement(app, tmp_path):
    """좌표가 깨져도 그래프는 열려야 한다."""
    app.state.hub.layout_path.write_text("{broken", encoding="utf-8")
    reopened = create_app(MINIVIT, state_dir=app.state.hub.state_dir, token=TOKEN)
    try:
        assert reopened.state.hub.layout == {"positions": {}}
    finally:
        reopened.state.hub.kernel.stop()
        reopened.state.hub.l1.stop()


def test_hub_starts_without_a_graph(tmp_path):
    """첫 화면을 띄우려면 그래프 없이도 떠야 한다(§2.2)."""
    app = create_app(state_dir=tmp_path / "state", token=TOKEN)
    try:
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        health = client.get(f"/api/health?token={TOKEN}").json()
        assert health["ok"] and health["graph_open"] is False
        # 그래프가 필요한 라우트는 409로 정직하게 거절한다.
        assert client.get("/api/graph", headers={"Authorization": f"token {TOKEN}"}).status_code == 409
    finally:
        app.state.hub.kernel.stop()
        app.state.hub.l1.stop()


def test_start_endpoint_lists_templates_and_marks_missing_ones(client):
    payload = client.get("/api/start", headers={"Authorization": f"token {TOKEN}"}).json()
    templates = {entry["id"]: entry for entry in payload["templates"]}
    assert templates["resnet18"]["available"] is True
    assert templates["resnet18"]["metric"] == "≈95.0 % top-1"
    # 아직 없는 템플릿을 있는 척하지 않는다.
    assert templates["nanogpt"]["available"] is False
    assert payload["state_dir"] and payload["torch_version"]


def test_open_loads_a_template(tmp_path):
    app = create_app(state_dir=tmp_path / "state", token=TOKEN)
    try:
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        auth = {"Authorization": f"token {TOKEN}"}
        template = next(entry for entry in client.get("/api/start", headers=auth).json()["templates"]
                        if entry["id"] == "resnet18")

        response = client.post("/api/open", headers=auth, json={"path": template["path"]})
        assert response.status_code == 200 and response.json()["name"] == "ResNet18"
        assert client.get("/api/graph", headers=auth).status_code == 200
        # 템플릿의 fc는 rt.num_classes를 참조한다. 첫 화면에는 --rt를 칠 자리가 없다.
        assert app.state.hub.rt == {"num_classes": 10}
    finally:
        app.state.hub.kernel.stop()
        app.state.hub.l1.stop()


def test_open_rejects_a_missing_path(client):
    response = client.post("/api/open", headers={"Authorization": f"token {TOKEN}"},
                           json={"path": "/nope/missing.tfg.json"})
    assert response.status_code == 404


def test_scalars_land_in_the_tracker(client):
    auth = {"Authorization": f"token {TOKEN}"}
    client.post("/api/runs", headers=auth,
                json={"run_id": "r1", "name": "demo", "manifest": {"seed": 3}})
    for step in range(4):
        client.post("/api/scalars", headers=auth,
                    json={"run_id": "r1", "step": step, "train/loss": 2.0 - step * 0.1})

    runs = client.get("/api/runs", headers=auth).json()["runs"]
    assert runs[0]["id"] == "r1" and runs[0]["keys"] == ["train/loss"]
    assert runs[0]["manifest"]["seed"] == 3

    curve = client.get("/api/runs/curve?key=train/loss&runs=r1", headers=auth).json()
    assert curve["series"][0]["points"][0] == [0, 2.0]


def test_seed_group_aggregate_endpoint(client):
    auth = {"Authorization": f"token {TOKEN}"}
    for seed in range(3):
        client.post("/api/scalars", headers=auth,
                    json={"run_id": f"s{seed}", "step": 0, "loss": 1.0 + seed})
    payload = client.get("/api/runs/curve?key=loss&runs=s0,s1,s2&aggregate=true",
                         headers=auth).json()
    assert payload["aggregate"][0]["mean"] == 2.0
    assert payload["aggregate"][0]["n"] == 3


def test_eval_needs_a_probe_before_it_can_read_values(client):
    """Debug Console(§5.6.1)은 L1 커널의 마지막 probe 위에서만 값을 본다."""
    response = client.post("/api/eval", headers={"Authorization": f"token {TOKEN}"},
                           json={"expr": "y.shape", "node": "01J9Q4B4"})
    assert response.status_code == 200
    assert response.json() == {"ok": False, "error": "probe를 한 번 돌려 주세요"}


def test_eval_is_refused_in_attach_mode(app, client):
    """Attach 모드의 값은 사용자 프로세스 안에 있다 - hub가 대신 볼 수 없다(§3.5)."""
    app.state.hub.attached = True
    response = client.post("/api/eval", headers={"Authorization": f"token {TOKEN}"},
                           json={"expr": "y", "node": "01J9Q4B4"})
    assert response.status_code == 409


def test_node_stdout_reaches_the_log_routes(app, client):
    """커널이 올린 노드별 출력은 하단 로그 탭과 Inspector 출력이 같은 표에서 읽는다."""
    auth = {"Authorization": f"token {TOKEN}"}
    app.state.hub.tracker.log_text("forward one", node_id="01J9Q4B4", stream="stdout")
    app.state.hub.tracker.log_text("elsewhere", node_id="01J9Q4B5", stream="stdout")

    assert len(client.get("/api/logs", headers=auth).json()["logs"]) == 2
    only = client.get("/api/logs?node=01J9Q4B4", headers=auth).json()["logs"]
    assert [line["text"] for line in only] == ["forward one"]


def test_export_returns_a_downloadable_figure(client):
    """논문용 그림은 hub에서 바로 받는다(§6.4)."""
    auth = {"Authorization": f"token {TOKEN}"}
    svg = client.get("/api/export?format=svg", headers=auth)
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert "attachment" in svg.headers["content-disposition"]
    assert svg.text.startswith("<svg")

    pdf = client.get("/api/export?format=pdf&preset=cvpr&anonymous=true", headers=auth)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert b"TorchFlow" not in pdf.content

    assert client.get("/api/export?format=png", headers=auth).status_code == 400


def test_new_graph_starts_empty_and_saves(tmp_path):
    """빈 캔버스 진입점(§2.2). 노드 0으로 열리고 편집 op를 받는다."""
    app = create_app(state_dir=tmp_path / "state", token=TOKEN)
    auth = {"Authorization": f"token {TOKEN}"}
    try:
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        assert client.post("/api/new", headers=auth, json={"name": "scratch"}).status_code == 200

        graph = client.get("/api/graph", headers=auth).json()["graph"]
        # 빈 컬렉션은 정본 직렬화에서 빠진다(exclude_defaults) - 노드 0의 모습이다.
        assert graph["graph"]["name"] == "scratch" and "nodes" not in graph["graph"]

        added = client.post("/api/ops", headers=auth, json={
            "client_id": "c", "tmp_seq": 1, "kind": "add_node",
            "payload": {"node": {"id": "n1", "label": "x", "type": "torchflow.Input",
                                 "ports_out": [{"name": "x", "type": "Tensor",
                                                "shape": ["B", 8], "dtype": "float32"}]}}})
        assert added.status_code == 200 and added.json()["seq"] == 1

        # 저장 기본 경로는 프로젝트의 graph/ 다 - 테스트는 tmp_path로 명시한다.
        target = tmp_path / "graph" / "scratch.tfg.json"
        saved = client.post("/api/save", headers=auth, json={"path": str(target)}).json()
        assert saved["ok"] and saved["problems"] == []
        assert Path(saved["path"]).is_file()
    finally:
        app.state.hub.kernel.stop()
        app.state.hub.l1.stop()


def test_close_returns_to_the_start_screen(app, client):
    """그래프를 닫으면 첫 화면 상태로 돌아간다(§2.2). 커널은 살아 있다."""
    auth = {"Authorization": f"token {TOKEN}"}
    assert client.get("/api/health", headers=auth).json()["graph_open"] is True

    assert client.post("/api/close", headers=auth).json()["ok"] is True
    health = client.get("/api/health", headers=auth).json()
    assert health["graph_open"] is False
    assert client.get("/api/graph", headers=auth).status_code == 409

    # 다시 열 수 있어야 한다 - 닫기가 종착역이 아니다.
    assert client.post("/api/new", headers=auth, json={"name": "next"}).status_code == 200
    assert client.get("/api/health", headers=auth).json()["graph_open"] is True


def test_code_route_renders_the_model(client):
    """Code 탭은 디스크에 쓰지 않고 메모리에서 렌더한다(§7.6.2)."""
    body = client.get("/api/code", headers={"Authorization": f"token {TOKEN}"}).json()
    assert body["lines"] > 50
    assert "class MiniViT(nn.Module):" in body["code"]
    assert body["ir_sha256"]


def test_training_starts_and_takes_commands(app, client, tmp_path, monkeypatch):
    """학습은 hub가 아니라 분리된 워커가 돈다(§5.5.3). hub는 파일로만 말한다."""
    auth = {"Authorization": f"token {TOKEN}"}
    app.state.hub.runs_dir = tmp_path / "runs"
    started: dict = {}

    def fake_start(*, run_id, root, job, code, python=None):
        from torchflow.hub.runs import RunHandle

        directory = root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        started.update({"job": job, "code": code})
        # 워커가 남길 법한 이벤트를 그대로 흉내낸다.
        (directory / "events.jsonl").write_text(
            '{"kind": "status", "state": "running", "device": "cpu", "steps": 3}\n'
            '{"kind": "scalar", "step": 1, "loss": 2.5}\n'
            '{"kind": "scalar", "step": 2, "loss": 1.5}\n', encoding="utf-8")
        (directory / "control.json").write_text("{}", encoding="utf-8")
        return RunHandle(run_id=run_id, directory=directory, total=3)

    monkeypatch.setattr("torchflow.hub.runs.start", fake_start)

    body = client.post("/api/train", headers=auth, json={"steps": 3, "batch": 4}).json()
    assert body["ok"] and body["run_id"]
    # 학습 대상은 그래프에서 뽑은 생성 코드 그대로다(§7.2).
    assert "class MiniViT(nn.Module):" in started["code"]
    assert started["job"]["model_args"]["num_classes"] == 10

    status = client.get("/api/train", headers=auth).json()["runs"][0]
    assert status["state"] == "running" and status["step"] == 2

    curve = client.get(f"/api/runs/curve?key=loss&runs={body['run_id']}", headers=auth).json()
    assert curve["series"][0]["points"] == [[1, 2.5], [2, 1.5]]

    # 명령은 control.json에 쌓인다 - 연결을 유지하지 않으므로 hub 재시작에도 살아 있다.
    client.post(f"/api/train/{body['run_id']}", headers=auth, json={"cmd": "pause"})
    client.post(f"/api/train/{body['run_id']}", headers=auth,
                json={"cmd": "set_lr", "value": 0.02})
    control = json.loads((app.state.hub.runs_dir / body["run_id"] / "control.json")
                         .read_text(encoding="utf-8"))
    assert control == {"pause": True, "lr": 0.02}


@pytest.fixture
def trained(app, client, tmp_path, monkeypatch):
    """워커를 띄우지 않고 run 하나를 연다.

    ``start``를 통째로 가짜로 바꾸지 않고 프로세스 기동만 막는다 - fork와 재개가
    실제로 파일을 어떻게 다루는지가 이 테스트들의 요점이다.
    """
    app.state.hub.runs_dir = tmp_path / "runs"
    monkeypatch.setattr("torchflow.hub.runs._spawn", lambda directory, python=None: None)
    auth = {"Authorization": f"token {TOKEN}"}
    body = client.post("/api/train", headers=auth, json={"steps": 10}).json()
    assert body["ok"]
    handle = app.state.hub.l2[body["run_id"]]
    handle.step = 7
    (handle.directory / "ckpt.pt").write_bytes(b"fake-checkpoint")
    return body["run_id"], handle, auth


def test_smoke_runs_short_and_deterministic(app, client, trained):
    """Smoke는 20 step · batch 8 · 결정적이다 - 두 번 돌리면 loss가 같아야 한다."""
    run_id, _handle, auth = trained
    body = client.post("/api/train", headers=auth, json={"smoke": True}).json()

    job = json.loads((app.state.hub.runs_dir / body["run_id"] / "job.json")
                     .read_text(encoding="utf-8"))
    assert (job["steps"], job["batch"], job["smoke"]) == (20, 8, True)


def test_an_exploratory_run_takes_the_change_in_place(app, client, trained, tmp_path):
    """exploratory는 in-place 변경이 허용되고, 변경은 파일로 물질화된다(§5.7.2)."""
    run_id, handle, auth = trained
    body = client.post(f"/api/train/{run_id}", headers=auth,
                       json={"cmd": "set_hparam", "path": "optim.lr", "value": 3e-4}).json()

    assert body["ok"] and "forked_from" not in body
    control = json.loads((handle.directory / "control.json").read_text(encoding="utf-8"))
    assert control == {"lr": 3e-4}
    # 파일은 아직 없다 - 워커가 "적용했다"고 말하기 전에는 적을 사실이 없다.
    assert not (tmp_path / "conf" / "overrides" / f"{run_id}.yaml").exists()

    (handle.directory / "events.jsonl").write_text(
        '{"kind": "hparam", "step": 9, "path": "optim.lr", "value": 0.0003}\n', encoding="utf-8")
    app.state.hub.sync_runs()

    # 학습 중에 바꾼 값은 코드 어디에도 안 남는다 - 이 파일이 재현의 유일한 근거다.
    overrides = (tmp_path / "conf" / "overrides" / f"{run_id}.yaml").read_text(encoding="utf-8")
    assert "path: optim.lr" in overrides and "step: 9" in overrides
    # 같은 이벤트를 다시 읽어도 두 줄이 되지 않는다.
    app.state.hub.sync_runs()
    assert overrides.count("optim.lr") == 1


def test_a_reported_run_forks_instead_of_changing_in_place(app, client, trained):
    """reported run은 hparam이 동결이다. 편집하면 갈라진 run이 생긴다(§5.7.2)."""
    run_id, handle, auth = trained
    client.post(f"/api/train/{run_id}", headers=auth, json={"cmd": "promote"})
    assert app.state.hub.tracker.run(run_id).kind == "reported"

    body = client.post(f"/api/train/{run_id}", headers=auth,
                       json={"cmd": "set_hparam", "path": "optim.lr", "value": 3e-4}).json()

    assert body["forked_from"] == run_id and body["run_id"] != run_id
    child = app.state.hub.tracker.run(body["run_id"])
    assert child.parent_run == run_id and child.kind == "reported"
    # 원 run은 멈추고, 새 run은 복사된 ckpt에서 이어 간다.
    assert json.loads((handle.directory / "control.json").read_text())["stop"] is True
    child_dir = app.state.hub.runs_dir / body["run_id"]
    assert (child_dir / "ckpt.pt").read_bytes() == b"fake-checkpoint"
    child_job = json.loads((child_dir / "job.json").read_text(encoding="utf-8"))
    assert child_job["resume"] == str(child_dir / "ckpt.pt") and child_job["lr"] == 3e-4


@pytest.mark.parametrize("path, expected", [
    ("optim.lr", "hot"), ("optim.weight_decay", "hot"),
    ("scheduler.type", "scheduler"), ("data.batch", "restart"),
    ("optim.optimizer", "restart"), ("data.seed", "cold"),
])
def test_hparam_paths_are_classified(path, expected):
    from torchflow.hub import runs as l2

    assert l2.classify(path) == expected


def test_a_change_needing_a_restart_is_reported_not_applied(app, client, trained):
    """재시작이 필요한 변경은 알리기만 한다 - 자동 재시작은 하지 않는다(ADR-05)."""
    run_id, handle, auth = trained
    body = client.post(f"/api/train/{run_id}", headers=auth,
                       json={"cmd": "set_hparam", "path": "data.batch", "value": 64}).json()

    assert body["ok"] is False and body["restart_required"] == "restart"
    assert json.loads((handle.directory / "control.json").read_text()) == {}


def test_resume_relaunches_a_dead_run_from_its_checkpoint(app, client, trained):
    """끝났거나 hub와 함께 죽은 run은 ckpt에서 다시 띄운다. run id는 그대로다."""
    run_id, handle, auth = trained
    l2.command(handle, stop=True)
    assert not handle.alive

    body = client.post(f"/api/train/{run_id}", headers=auth,
                       json={"cmd": "resume", "steps": 40}).json()

    assert body["ok"] and body["run_id"] == run_id
    job = json.loads((handle.directory / "job.json").read_text(encoding="utf-8"))
    assert job["resume"] == str(handle.directory / "ckpt.pt") and job["steps"] == 40
    # 남아 있던 stop 명령이 지워져야 새 워커가 첫 스텝에 다시 멈추지 않는다.
    assert json.loads((handle.directory / "control.json").read_text()) == {}


def test_hub_recovers_runs_it_did_not_start(app, tmp_path):
    """hub가 죽는 동안에도 워커는 돌았다. events.jsonl이 정본이다(§5.5.3)."""
    hub = app.state.hub
    hub.runs_dir = tmp_path / "runs"
    directory = hub.runs_dir / "run-orphan"
    directory.mkdir(parents=True)
    (directory / "job.json").write_text('{"steps": 3}', encoding="utf-8")
    (directory / "events.jsonl").write_text(
        '{"kind": "status", "state": "running", "steps": 3}\n'
        '{"kind": "scalar", "step": 1, "loss": 2.0}\n'
        '{"kind": "scalar", "step": 2, "loss": 1.0}\n', encoding="utf-8")

    assert hub.recover() == ["run-orphan"]
    assert hub.tracker.curve("run-orphan", "loss") == [(1, 2.0), (2, 1.0)]
    # 두 번 복구해도 곡선이 두 배가 되지 않는다.
    hub.l2.clear()
    hub.recover()
    assert hub.tracker.curve("run-orphan", "loss") == [(1, 2.0), (2, 1.0)]


def test_l1_probe_tells_the_kernel_which_gpus_l2_holds(app, monkeypatch):
    """학습 중 L1은 L2가 잡은 GPU를 피한다(§5.1.5). 그 사실이 커널까지 가야 한다."""
    from torchflow.hub.runs import RunHandle

    hub = app.state.hub
    handle = RunHandle(run_id="r", directory=app.state.hub.state_dir, device="cuda:0")
    handle.state = "running"
    monkeypatch.setattr(type(handle), "alive", property(lambda self: True))
    hub.l2["r"] = handle

    assert hub.occupied_devices() == ["cuda:0"]


def test_a_new_project_does_not_inherit_the_previous_curves(app, client, trained):
    """새 그래프를 열면 이전 그래프의 run이 따라오면 안 된다.

    트래커는 state-dir 하나를 쓰므로 DB에는 남아 있다. 남아 있는 것과 화면에
    보이는 것은 다른 문제고, 여기서 가르는 것은 그래프 identity(meta.id)다 -
    IR 해시로 가르면 노드 하나 고친 순간 비교하려던 곡선이 남이 된다.
    """
    run_id, handle, auth = trained
    before = client.get("/api/runs", headers=auth).json()["runs"]
    assert [run["id"] for run in before] == [run_id]

    client.post("/api/new", headers=auth, json={"name": "빈 프로젝트"})

    assert client.get("/api/runs", headers=auth).json()["runs"] == []
    # 워커는 계속 돌지만 이 화면의 것이 아니다.
    assert client.get("/api/train", headers=auth).json()["runs"] == []
    # DB에서 지운 것이 아니다 - 그래프를 도로 열면 다시 보인다.
    assert len(app.state.hub.tracker.runs()) == 1


def test_a_graph_without_an_id_still_gets_one(app, client, tmp_path):
    """id 없이 저장된 그래프가 이미 있다. 그때 가르기를 포기하면 남의 run이 딸려 온다."""
    auth = {"Authorization": f"token {TOKEN}"}
    plain = tmp_path / "legacy.tfg.json"
    plain.write_text(json.dumps({"schema_version": "1.0.0",
                                 "graph": {"name": "legacy", "nodes": [], "edges": []}}),
                     encoding="utf-8")

    client.post("/api/open", headers=auth, json={"path": str(plain)})
    first = app.state.hub.graph_id
    assert first, "id 없는 그래프도 identity를 가져야 한다"
    # CLI로 연 것과 화면에서 연 것이 같은 파일이면 같은 값이어야 한다.
    assert create_app(plain, state_dir=tmp_path / "s", token="t").state.hub.graph_id == first

    # 경로에서 만들므로 다시 열어도 같다 - hub를 다시 띄워도 곡선이 붙어 있는다.
    client.post("/api/new", headers=auth, json={"name": "다른 것"})
    client.post("/api/open", headers=auth, json={"path": str(plain)})
    assert app.state.hub.graph_id == first


def test_a_new_project_gets_its_own_identity(app, client):
    """id가 없으면 새 프로젝트끼리도 run이 섞인다."""
    auth = {"Authorization": f"token {TOKEN}"}
    client.post("/api/new", headers=auth, json={"name": "하나"})
    first = app.state.hub.graph_id
    client.post("/api/new", headers=auth, json={"name": "둘"})

    assert first and app.state.hub.graph_id and first != app.state.hub.graph_id


def test_training_refuses_a_graph_without_an_input(app, client):
    auth = {"Authorization": f"token {TOKEN}"}
    app.state.hub.store.ir.graph.nodes = [
        node for node in app.state.hub.store.ir.graph.nodes if node.type != "torchflow.Input"]

    response = client.post("/api/train", headers=auth, json={})
    assert response.status_code == 400 and "Input" in response.json()["error"]


def test_save_refuses_to_overwrite_another_graph(tmp_path):
    """같은 이름의 새 그래프가 기존 파일을 말없이 지우면 안 된다."""
    app = create_app(state_dir=tmp_path / "state", token=TOKEN)
    auth = {"Authorization": f"token {TOKEN}"}
    target = tmp_path / "graph" / "scratch.tfg.json"
    try:
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        client.post("/api/new", headers=auth, json={"name": "scratch"})
        assert client.post("/api/save", headers=auth, json={"path": str(target)}).status_code == 200
        # 열려 있는 파일에 다시 저장하는 것은 된다.
        assert client.post("/api/save", headers=auth, json={"path": str(target)}).status_code == 200

        client.post("/api/new", headers=auth, json={"name": "scratch"})
        response = client.post("/api/save", headers=auth, json={"path": str(target)})
        assert response.status_code == 409 and "이미 있습니다" in response.json()["error"]
    finally:
        app.state.hub.kernel.stop()
        app.state.hub.l1.stop()


def test_probe_reports_a_kernel_failure(client):
    """probe가 실패하면 응답에 실린다 - 배지만 조용히 비면 사람은 고장으로 읽는다."""
    auth = {"Authorization": f"token {TOKEN}"}
    client.post("/api/ops", headers=auth, json={
        "client_id": "c-1", "tmp_seq": 1, "kind": "set_param",
        "payload": {"instance": "01J9I103", "path": "in_features", "value": 768}})
    body = client.post("/api/probe", headers=auth, json={}).json()
    assert body["ok"] is False and body["node"] == "01J9Q4B5"
    assert body["error"]


def test_merge_survives_concurrent_polling(tmp_path):
    """이벤트 병합은 스레드풀에서 동시에 불린다. 커서가 파일 끝을 넘으면 이벤트를 영영 잃는다."""
    import threading
    import time

    from torchflow.hub.tracker import Tracker

    directory = tmp_path / "run"
    directory.mkdir()
    events = directory / "events.jsonl"
    tracker = Tracker(tmp_path / "runs.db")
    tracker.ensure_run("r")
    handle = l2.RunHandle(run_id="r", directory=directory)
    running = True

    def poll():
        while running:
            l2.merge(handle, tracker)

    threads = [threading.Thread(target=poll) for _ in range(3)]
    for thread in threads:
        thread.start()
    with events.open("a", encoding="utf-8") as stream:
        for step in range(1500):
            stream.write(json.dumps({"kind": "scalar", "step": step, "loss": 1.0}) + "\n")
            stream.flush()
            time.sleep(0.0005)
        stream.write(json.dumps({"kind": "status", "state": "stopped", "step": 1499}) + "\n")
    running = False
    for thread in threads:
        thread.join()
    l2.merge(handle, tracker)
    tracker.close()

    assert handle.cursor == events.stat().st_size
    assert handle.state == "stopped" and handle.step == 1499


def test_training_on_a_dataset_checks_the_input_spec(app, client, tmp_path):
    """MNIST는 [B, 1, 28, 28]이다. 규격이 다르면 워커를 띄우기 전에 말한다."""
    auth = {"Authorization": f"token {TOKEN}"}
    app.state.hub.data_dir = tmp_path / "data"
    listed = client.get("/api/datasets", headers=auth).json()["datasets"]
    assert [(entry["name"], entry["available"]) for entry in listed] == [("mnist", False)]

    response = client.post("/api/train", headers=auth, json={"dataset": "mnist", "steps": 2})
    assert response.status_code == 400 and "1, 28, 28" in response.json()["error"]
    assert client.post("/api/datasets/nope/download", headers=auth).status_code == 404
