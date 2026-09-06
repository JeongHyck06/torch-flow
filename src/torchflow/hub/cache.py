"""디스크 CAS와 쿼터 (기획서 §5.2).

두 개를 분리한다.

* action cache: ``key -> digest``. 키는 ``node_key``/``closure_key``다.
* CAS: ``digest -> 파일``. 같은 내용은 한 번만 저장된다.

분리하는 이유: 같은 결과를 내는 키가 여럿일 수 있고(다른 경로, 같은 출력),
GC는 digest 단위로 돌아야 하기 때문이다.

L1 활성값 원본은 저장하지 않는다(§5.2). 통계·썸네일·다이제스트만 담는다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUOTA_GB = 20.0


@dataclass
class GCReport:
    scanned: int
    removed: int
    freed_bytes: int
    dry_run: bool = False

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / 1024**3


class DiskCAS:
    """``<state-dir>/cache/<sha[:2]>/<sha>``."""

    def __init__(self, root: Path | str, quota_gb: float = DEFAULT_QUOTA_GB):
        self.root = Path(root)
        self.quota_bytes = int(quota_gb * 1024**3)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "actions.json"
        self.actions: dict[str, str] = self._load_actions()

    # CAS
    def _path_of(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put(self, payload: bytes) -> str:
        digest = hashlib.blake2b(payload, digest_size=20).hexdigest()
        target = self._path_of(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # 같은 digest를 두 프로세스가 동시에 써도 부분 파일이 남지 않게.
            scratch = target.with_suffix(".part")
            scratch.write_bytes(payload)
            scratch.replace(target)
        return digest

    def get(self, digest: str) -> bytes | None:
        target = self._path_of(digest)
        return target.read_bytes() if target.exists() else None

    def put_json(self, payload) -> str:
        return self.put(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

    def get_json(self, digest: str):
        raw = self.get(digest)
        return None if raw is None else json.loads(raw)

    # action cache
    def _load_actions(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}   # 인덱스가 깨져도 캐시는 재구축 가능한 파생 상태다.

    def remember(self, key: str, payload) -> str:
        digest = self.put_json(payload)
        self.actions[key] = digest
        return digest

    def recall(self, key: str):
        digest = self.actions.get(key)
        return None if digest is None else self.get_json(digest)

    def flush(self) -> None:
        self.index_path.write_text(json.dumps(self.actions, sort_keys=True), encoding="utf-8")

    # 쿼터와 GC
    def entries(self) -> list[tuple[Path, int, float]]:
        return [
            (path, path.stat().st_size, path.stat().st_atime)
            for path in self.root.glob("*/*")
            if path.is_file() and path.suffix != ".part"
        ]

    def size_bytes(self) -> int:
        return sum(size for _, size, _ in self.entries())

    def gc(self, keep: set[str] | None = None, *, dry_run: bool = False) -> GCReport:
        """참조 카운트 0인 digest부터 LRU로 지운다(§5.2).

        ``keep``에 현재 IR·핀 노드·run manifest가 참조하는 digest를 넘긴다.
        비워 두면 쿼터를 넘긴 만큼만 오래된 것부터 지운다.
        """
        keep = keep or set()
        entries = sorted(self.entries(), key=lambda entry: entry[2])   # atime 오름차순
        total = sum(size for _, size, _ in entries)
        removed = freed = 0

        for path, size, _ in entries:
            if total - freed <= self.quota_bytes:
                break
            if path.name in keep:
                continue
            if not dry_run:
                path.unlink(missing_ok=True)
            removed += 1
            freed += size

        if not dry_run and removed:
            self.actions = {key: digest for key, digest in self.actions.items()
                            if self._path_of(digest).exists()}
            self.flush()
        return GCReport(scanned=len(entries), removed=removed, freed_bytes=freed, dry_run=dry_run)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.actions = {}
