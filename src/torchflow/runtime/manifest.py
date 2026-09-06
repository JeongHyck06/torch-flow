"""run manifest (기획서 §10.3).

run 하나를 다시 돌리는 데 필요한 것 전부를 한 파일에 적는다. Attach Mode도
같은 스키마를 쓰되 IR 해시 자리에 모델 구조 해시가 들어간다.

``--anonymous``는 논문 심사용이다. 도구 이름과 버전, git remote, 사용자명,
절대 경로를 지운다. 익명 심사에서 저자가 드러나는 흔한 통로가 이것들이다.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ANONYMOUS_PLACEHOLDER = "<redacted>"


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def structure_hash(model) -> str:
    """모델 구조 해시. 가중치가 아니라 이름과 모양만 본다.

    같은 아키텍처를 다른 시드로 돌린 run들이 같은 해시를 갖는다. 그래야
    시드 집계가 성립한다.
    """
    entries = [(name, tuple(param.shape), str(param.dtype))
               for name, param in model.named_parameters()]
    entries += [(name, tuple(buffer.shape), str(buffer.dtype))
                for name, buffer in model.named_buffers()]
    return _digest(sorted(map(str, entries)))


def git_state(root: Path | None = None) -> dict[str, Any]:
    """commit과 dirty 여부. 저장소가 아니면 비운다."""
    root = root or Path.cwd()

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(("git", *args), cwd=root, capture_output=True,
                                    text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {}
    return {
        "commit": commit,
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": run("config", "--get", "remote.origin.url"),
    }


def environment() -> dict[str, Any]:
    """재현에 필요한 환경. torch가 없으면 없는 대로 적는다."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    info["cuda"] = getattr(torch.version, "cuda", None)
    info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    info["deterministic"] = torch.are_deterministic_algorithms_enabled()
    info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())]
        info["driver"] = getattr(torch.version, "cuda", None)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        info["gpus"] = ["mps"]
    else:
        info["gpus"] = []
    return info


def frozen_packages(limit: int = 400) -> list[str]:
    """설치된 패키지 목록. pip를 부르지 않고 메타데이터를 직접 읽는다."""
    from importlib import metadata

    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    )
    return packages[:limit]


def build(
    *,
    run_id: str,
    kind: str = "exploratory",
    parent_run: str | None = None,
    ir_hash: str | None = None,
    model=None,
    hparams: dict[str, Any] | None = None,
    hparam_overrides: list[dict[str, Any]] | None = None,
    seed: int = 0,
    probe: dict[str, Any] | None = None,
    data_hash: str | None = None,
    node_hashes: dict[str, str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """manifest 한 장."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "created": time.time(),
        "seed": seed,
        "hparams": dict(hparams or {}),
        "hparam_overrides": list(hparam_overrides or []),
        "env": environment(),
        "git": git_state(root),
        "packages": frozen_packages(),
        "probe": dict(probe or {}),
        "data_hash": data_hash,
        "node_hashes": dict(node_hashes or {}),
    }
    if parent_run:
        manifest["parent_run"] = parent_run
    if ir_hash:
        manifest["ir_hash"] = ir_hash
    if model is not None:
        manifest["structure_hash"] = structure_hash(model)
        manifest["params"] = sum(param.numel() for param in model.parameters())
    return manifest


def anonymize(manifest: dict[str, Any]) -> dict[str, Any]:
    """익명 심사용. 저자가 드러나는 필드를 지운다."""
    clean = json.loads(json.dumps(manifest, default=str))
    clean.pop("packages", None)
    git = clean.get("git") or {}
    git.pop("remote", None)
    git.pop("branch", None)
    if git:
        clean["git"] = git
    clean["tool"] = ANONYMOUS_PLACEHOLDER
    home = str(Path.home())
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""

    def scrub(value):
        if isinstance(value, str):
            value = value.replace(home, "~")
            if user:
                value = value.replace(user, ANONYMOUS_PLACEHOLDER)
            return value
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(clean)


def save(manifest: dict[str, Any], path: Path | str, *, anonymous: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = anonymize(manifest) if anonymous else manifest
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path
