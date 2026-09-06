"""실험 트래커 (기획서 §10.3, §11).

state-dir의 ``runs.db`` 하나. 자체 클라우드는 비목표이고 W&B와 MLflow는
어댑터로 붙는다(v1). 여기서 필요한 것은 곡선과 manifest를 잃지 않는 것뿐이다.

torch를 import하지 않는다. hub에서 돈다.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'exploratory',
    parent_run  TEXT,
    created     REAL NOT NULL,
    updated     REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    name        TEXT,
    manifest    TEXT
);
CREATE TABLE IF NOT EXISTS scalars (
    run_id  TEXT NOT NULL,
    key     TEXT NOT NULL,
    step    INTEGER NOT NULL,
    value   REAL NOT NULL,
    wall    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS scalars_lookup ON scalars(run_id, key, step);
CREATE TABLE IF NOT EXISTS logs (
    run_id  TEXT,
    node_id TEXT,
    stream  TEXT NOT NULL,
    text    TEXT NOT NULL,
    wall    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS logs_lookup ON logs(run_id, wall);
"""

# 곡선 하나에 보낼 최대 점 수. 그 이상은 LTTB로 줄인다(§6.3).
MAX_POINTS = 4000


@dataclass
class Run:
    id: str
    kind: str
    parent_run: str | None
    created: float
    updated: float
    status: str
    name: str | None
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


class Tracker:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # hub는 요청을 스레드풀에서 처리한다. 연결을 스레드마다 새로 열지 않고
        # 같은 연결을 공유하되 sqlite에게 그래도 된다고 알린다.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    # ── run ───────────────────────────────────────────────────────────
    def ensure_run(self, run_id: str, *, kind: str = "exploratory", name: str | None = None,
                   parent_run: str | None = None, manifest: dict[str, Any] | None = None) -> str:
        now = time.time()
        self.connection.execute(
            """INSERT INTO runs (id, kind, parent_run, created, updated, status, name, manifest)
               VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   updated = excluded.updated,
                   name = COALESCE(excluded.name, runs.name),
                   manifest = COALESCE(excluded.manifest, runs.manifest)""",
            (run_id, kind, parent_run, now, now, name,
             json.dumps(manifest, ensure_ascii=False) if manifest else None),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "done") -> None:
        self.connection.execute("UPDATE runs SET status = ?, updated = ? WHERE id = ?",
                                (status, time.time(), run_id))
        self.connection.commit()

    def runs(self, limit: int = 100) -> list[Run]:
        rows = self.connection.execute(
            "SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_run(row) for row in rows]

    def run(self, run_id: str) -> Run | None:
        row = self.connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    # ── 스칼라 ────────────────────────────────────────────────────────
    def log(self, run_id: str, step: int, values: dict[str, float], wall: float | None = None) -> int:
        """유한한 수만 기록한다. NaN은 곡선을 통째로 못 쓰게 만든다."""
        wall = wall if wall is not None else time.time()
        rows = [(run_id, key, step, float(value), wall)
                for key, value in values.items()
                if isinstance(value, (int, float)) and math.isfinite(value)]
        if rows:
            self.connection.executemany(
                "INSERT INTO scalars (run_id, key, step, value, wall) VALUES (?, ?, ?, ?, ?)", rows)
            self.connection.execute("UPDATE runs SET updated = ? WHERE id = ?", (wall, run_id))
            self.connection.commit()
        return len(rows)

    def keys(self, run_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT key FROM scalars WHERE run_id = ? ORDER BY key", (run_id,)).fetchall()
        return [row["key"] for row in rows]

    def curve(self, run_id: str, key: str, *, since: int = 0,
              max_points: int = MAX_POINTS) -> list[tuple[int, float]]:
        rows = self.connection.execute(
            "SELECT step, value FROM scalars WHERE run_id = ? AND key = ? AND step >= ?"
            " ORDER BY step", (run_id, key, since)).fetchall()
        return downsample([(row["step"], row["value"]) for row in rows], max_points)

    def aggregate(self, run_ids: Iterable[str], key: str,
                  *, max_points: int = MAX_POINTS) -> list[dict[str, float]]:
        """시드 그룹의 step별 mean과 std (§11 시드 mean+-std)."""
        ids = list(run_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.connection.execute(
            f"SELECT step, AVG(value) AS mean, COUNT(*) AS n,"
            f"       AVG(value * value) - AVG(value) * AVG(value) AS var"
            f"  FROM scalars WHERE run_id IN ({placeholders}) AND key = ?"
            f" GROUP BY step ORDER BY step", (*ids, key)).fetchall()
        points = [{"step": row["step"], "mean": row["mean"], "n": row["n"],
                   "std": math.sqrt(max(row["var"] or 0.0, 0.0))} for row in rows]
        if len(points) <= max_points:
            return points
        stride = math.ceil(len(points) / max_points)
        return points[::stride]

    # ── 로그 ──────────────────────────────────────────────────────────
    def log_text(self, text: str, *, run_id: str | None = None, node_id: str | None = None,
                 stream: str = "stdout") -> None:
        self.connection.execute(
            "INSERT INTO logs (run_id, node_id, stream, text, wall) VALUES (?, ?, ?, ?, ?)",
            (run_id, node_id, stream, text, time.time()))
        self.connection.commit()

    def tail(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM logs ORDER BY wall DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def close(self) -> None:
        self.connection.close()


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"], kind=row["kind"], parent_run=row["parent_run"],
        created=row["created"], updated=row["updated"], status=row["status"],
        name=row["name"], manifest=json.loads(row["manifest"]) if row["manifest"] else {},
    )


def downsample(points: list[tuple[int, float]], threshold: int) -> list[tuple[int, float]]:
    """LTTB. 곡선의 모양을 지키면서 점 수를 줄인다.

    균일 추출은 스파이크를 통째로 놓친다. loss 곡선에서 스파이크는 대개
    보고 싶은 바로 그것이다.
    """
    count = len(points)
    if threshold >= count or threshold < 3:
        return points

    bucket = (count - 2) / (threshold - 2)
    result = [points[0]]
    index = 0

    for i in range(threshold - 2):
        start = int((i + 1) * bucket) + 1
        end = min(int((i + 2) * bucket) + 1, count)
        nxt_start = end
        nxt_end = min(int((i + 3) * bucket) + 1, count)

        window = points[nxt_start:nxt_end] or [points[-1]]
        avg_x = sum(point[0] for point in window) / len(window)
        avg_y = sum(point[1] for point in window) / len(window)

        ax, ay = points[index]
        best, best_area = start, -1.0
        for candidate in range(start, end):
            cx, cy = points[candidate]
            area = abs((ax - avg_x) * (cy - ay) - (ax - cx) * (avg_y - ay))
            if area > best_area:
                best_area, best = area, candidate
        result.append(points[best])
        index = best

    result.append(points[-1])
    return result
