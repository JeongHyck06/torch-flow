"""CLI. hub는 프로젝트에 하나만 뜬다."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from torchflow.cli import _running_hub


def test_no_lock_or_dead_pid_means_no_running_hub(tmp_path):
    lock = tmp_path / "hub.json"
    assert _running_hub(lock) is None
    lock.write_text(json.dumps({"pid": 2 ** 22 - 1, "host": "127.0.0.1", "port": 1, "token": "t"}))
    assert _running_hub(lock) is None


def test_a_live_hub_is_reported_by_its_url(tmp_path):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/api/health" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        lock = tmp_path / "hub.json"
        lock.write_text(json.dumps({"pid": os.getpid(), "host": "127.0.0.1",
                                    "port": server.server_port, "token": "t"}))
        assert _running_hub(lock) == f"http://127.0.0.1:{server.server_port}/?token=t"
    finally:
        server.shutdown()


def test_an_unresponsive_hub_stops_the_launch(tmp_path):
    """살아 있는데 대답이 없는 hub 위에 또 띄우면 포트가 겹친다 - 사람에게 넘긴다."""
    lock = tmp_path / "hub.json"
    lock.write_text(json.dumps({"pid": os.getpid(), "host": "127.0.0.1", "port": 9, "token": "t"}))
    with pytest.raises(SystemExit):
        _running_hub(lock)
