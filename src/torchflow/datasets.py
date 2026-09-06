"""내장 데이터셋 (기획서 §3.1: 다운로드는 명시 버튼으로만).

hub는 torch 없이 파일 유무와 다운로드만 다루고, 텐서 적재는 워커가 한다.
MNIST는 IDX 파일 네 개라 torchvision 없이 stdlib로 읽는다 - 커널·워커는 사용자
인터프리터에서 돌므로(§9) 거기에 torchvision이 있다고 가정할 수 없다.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Any

# torchvision이 쓰는 미러. 원본 yann.lecun.com은 자주 막힌다.
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

CATALOGUE: dict[str, dict[str, Any]] = {
    "mnist": {
        "label": "MNIST", "shape": [1, 28, 28], "classes": 10, "size_mb": 11, "url": MNIST_URL,
        "files": {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))와 같은 값.
        "mean": 0.1307, "std": 0.3081,
    },
}


def folder(name: str, base: str | Path) -> Path:
    return Path(base) / name


def available(name: str, base: str | Path) -> bool:
    return all((folder(name, base) / filename).is_file()
               for filename in CATALOGUE[name]["files"].values())


def download(name: str, base: str | Path) -> list[Path]:
    """없는 파일만 받는다. 임시 이름에 받고 갈아 끼우므로 끊긴 다운로드가 완성본 행세를 못 한다."""
    entry = CATALOGUE[name]
    target = folder(name, base)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for filename in entry["files"].values():
        path = target / filename
        if path.is_file():
            continue
        staging = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(entry["url"] + filename, staging)
        staging.replace(path)
        written.append(path)
    return written


def read_idx(path: Path):
    """IDX(gzip) -> uint8 텐서. 헤더는 magic 4바이트(0, 0, 타입, 차원 수) + 차원별 int32 big-endian."""
    import torch

    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    dims = raw[3]
    shape = struct.unpack(">" + "I" * dims, raw[4:4 + 4 * dims])
    return torch.frombuffer(bytearray(raw[4 + 4 * dims:]), dtype=torch.uint8).reshape(shape)


def load(name: str, base: str | Path) -> dict[str, tuple]:
    """``{"train": (x, y), "test": (x, y)}``. x는 정규화된 float32 ``[N, C, H, W]``, y는 int64."""
    entry = CATALOGUE[name]
    target = folder(name, base)
    splits = {}
    for split in ("train", "test"):
        images = read_idx(target / entry["files"][f"{split}_images"]).float().div_(255)
        images = images.sub_(entry["mean"]).div_(entry["std"]).reshape(-1, *entry["shape"])
        labels = read_idx(target / entry["files"][f"{split}_labels"]).long()
        splits[split] = (images, labels)
    return splits
