"""내장 데이터셋: IDX 읽기와 파일 유무 (기획서 §3.1)."""

import gzip
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torchflow import datasets  # noqa: E402


def write_fake_mnist(base: Path, count: int = 8) -> Path:
    """진짜와 같은 IDX gzip 형식의 아주 작은 MNIST. 픽셀은 0, 1, 2, ... 순서다."""
    target = base / "mnist"
    target.mkdir(parents=True)
    files = datasets.CATALOGUE["mnist"]["files"]
    pixels = (bytes(range(256)) * (count * 784 // 256 + 1))[:count * 784]
    for split in ("train", "test"):
        with gzip.open(target / files[f"{split}_images"], "wb") as stream:
            stream.write(struct.pack(">IIII", 0x803, count, 28, 28) + pixels)
        with gzip.open(target / files[f"{split}_labels"], "wb") as stream:
            stream.write(struct.pack(">II", 0x801, count) + bytes(index % 10 for index in range(count)))
    return target


def test_mnist_loads_normalised_tensors(tmp_path):
    assert not datasets.available("mnist", tmp_path)
    write_fake_mnist(tmp_path)
    assert datasets.available("mnist", tmp_path)

    x, y = datasets.load("mnist", tmp_path)["train"]
    assert x.shape == (8, 1, 28, 28) and x.dtype == torch.float32
    assert y.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    # 픽셀 0은 정규화 뒤 -mean/std다.
    assert abs(float(x[0, 0, 0, 0]) - (-0.1307 / 0.3081)) < 1e-4
