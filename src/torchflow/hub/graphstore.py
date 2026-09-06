"""권위 그래프 상태: op-log · seq · journal (기획서 §8.2.2).

hub가 그래프의 정본을 가진다. 클라이언트는 낙관적으로 먼저 그리고, 서버가
``seq``를 붙여 승인한다. journal은 append-only JSONL이라 hub가 죽어도
재시작 시 재생된다.

지금 적용되는 op는 ``set_param``·``set_switch_active``·``connect``·
``disconnect``뿐이다. M1의 반응형 루프를 돌리는 데 필요한 최소치다.
나머지 op kind는 journal에 기록되지만 그래프에는 적용되지 않는다(M5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ir import ModuleGraph, canonical_json, save

APPLIED_KINDS = {"set_param", "set_switch_active", "connect", "disconnect"}


class OpError(ValueError):
    """op가 그래프에 적용될 수 없다."""


class GraphStore:
    def __init__(self, ir: ModuleGraph, journal_path: Path | str | None = None):
        self.ir = ir
        self.seq = 0
        self.log: list[dict[str, Any]] = []
        self.journal_path = Path(journal_path) if journal_path else None
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    # op 적용
    def apply(self, op: dict[str, Any]) -> int:
        """op를 적용하고 부여된 ``seq``를 돌려준다."""
        kind = op.get("kind")
        if kind == "batch":
            for inner in op.get("ops") or []:
                self._mutate(inner)
        elif kind in APPLIED_KINDS:
            self._mutate(op)
        self.seq += 1
        record = {"seq": self.seq, "op": op}
        self.log.append(record)
        self._journal(record)
        return self.seq

    def _mutate(self, op: dict[str, Any]) -> None:
        kind, payload = op.get("kind"), op.get("payload") or {}
        scope = self._scope(payload.get("composite"))

        if kind == "set_param":
            instance = scope.instances.get(payload["instance"])
            if instance is None:
                raise OpError(f"unknown instance {payload['instance']}")
            instance.args[payload["path"]] = payload["value"]
        elif kind == "set_switch_active":
            instance = scope.instances.get(payload["instance"])
            if instance is None:
                raise OpError(f"unknown instance {payload['instance']}")
            if payload["active"] not in (instance.variants or {}):
                raise OpError(f"unknown variant {payload['active']!r}")
            instance.active = payload["active"]
        elif kind == "connect":
            edge = (payload["src"], payload["dst"])
            if edge not in scope.edges:
                scope.edges.append(edge)
        elif kind == "disconnect":
            edge = (payload["src"], payload["dst"])
            if edge in scope.edges:
                scope.edges.remove(edge)

    def _scope(self, composite: str | None):
        if composite is None:
            return self.ir.graph
        scope = self.ir.composites.get(composite)
        if scope is None:
            raise OpError(f"unknown composite {composite!r}")
        return scope

    # journal
    def _journal(self, record: dict[str, Any]) -> None:
        if self.journal_path is None:
            return
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def replay(self) -> int:
        """journal을 재생한다. 재시작 복구 경로."""
        if self.journal_path is None or not self.journal_path.exists():
            return 0
        records = [
            json.loads(line)
            for line in self.journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.journal_path.write_text("", encoding="utf-8")
        self.seq, self.log = 0, []
        for record in records:
            self.apply(record["op"])
        return len(records)

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [record for record in self.log if record["seq"] > seq]

    def snapshot(self) -> dict[str, Any]:
        return {"seq": self.seq, "graph": json.loads(canonical_json(self.ir))}

    def save(self, path: Path | str) -> None:
        save(self.ir, path)
