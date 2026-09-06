"""hub 라우트와 보안 불변식 (기획서 §8.2.2, §9)."""

import json
import subprocess
import sys

import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import MINIVIT
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


def test_training_refuses_a_graph_without_an_input(app, client):
    auth = {"Authorization": f"token {TOKEN}"}
    app.state.hub.store.ir.graph.nodes = [
        node for node in app.state.hub.store.ir.graph.nodes if node.type != "torchflow.Input"]

    response = client.post("/api/train", headers=auth, json={})
    assert response.status_code == 400 and "Input" in response.json()["error"]
