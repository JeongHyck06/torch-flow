"""데이터셋: 내장(MNIST)과 ``data/`` 아래 사용자 데이터 (기획서 §3.1).

hub는 torch 없이 **명세**만 본다 - 어떤 종류인지, 입력 모양과 클래스 수가 무엇인지,
내려받았는지. 텐서 적재는 워커가 한다. 사용자 데이터는 ``data/<이름>/`` 폴더 하나가
데이터셋 하나이고 안을 보고 종류를 정한다:

* 클래스별 하위 폴더에 이미지 -> ``image_folder``  (data/pets/cat/*.jpg)
* CSV 하나 -> ``csv``  (열이 특징, ``label``/``target``/``y``/``class`` 열 또는 마지막 열이 정답)
* ``x.npy`` + ``y.npy`` 또는 ``*.npz``(x, y) -> ``arrays``

MNIST는 IDX 파일 네 개라 torchvision 없이 stdlib로 읽는다 - 커널·워커는 사용자
인터프리터에서 돌므로(§9) 거기에 torchvision이 있다고 가정할 수 없다.
"""

from __future__ import annotations

import array
import ast
import csv
import gzip
import struct
import urllib.request
import zipfile
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


# 사용자 데이터 (data/<이름>/)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
LABEL_COLUMNS = ("label", "target", "y", "class")
MAX_SIDE = 64          # ponytail: 이미지는 긴 변 64 상한으로 정사각 리사이즈. 크기 옵션은 다음에.
VAL_FRACTION = 0.2     # test 분할이 없는 사용자 데이터는 시드 0으로 8:2 나눈다.


def describe(name: str, base: str | Path) -> dict[str, Any] | None:
    """데이터셋 하나의 명세. 내장이면 카탈로그에서, 아니면 폴더를 보고. 없으면 None."""
    if name in CATALOGUE:
        entry = CATALOGUE[name]
        return {"name": name, "label": entry["label"], "source": "builtin", "kind": "builtin",
                "shape": entry["shape"], "classes": entry["classes"], "size_mb": entry["size_mb"],
                "available": available(name, base)}
    folder = Path(base) / name
    return inspect(folder) if folder.is_dir() else None


def scan(base: str | Path) -> list[dict[str, Any]]:
    """내장 + data/ 아래 폴더들. 정체를 알 수 없는 폴더는 조용히 건너뛴다."""
    found = [describe(name, base) for name in CATALOGUE]
    root = Path(base)
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and folder.name not in CATALOGUE and not folder.name.startswith("."):
                spec = inspect(folder)
                if spec is not None:
                    found.append(spec)
    return found


def inspect(folder: Path) -> dict[str, Any] | None:
    for probe in (_inspect_images, _inspect_csv, _inspect_arrays):
        spec = probe(folder)
        if spec is not None:
            return {"name": folder.name, "label": folder.name, "source": "user",
                    "available": True, **spec}
    return None


def _image_files(folder: Path) -> dict[str, list[Path]]:
    classes = sorted(child.name for child in folder.iterdir()
                     if child.is_dir() and not child.name.startswith("."))
    return {name: sorted(path for path in (folder / name).iterdir()
                         if path.suffix.lower() in IMAGE_SUFFIXES)
            for name in classes}


def _inspect_images(folder: Path) -> dict[str, Any] | None:
    files = _image_files(folder)
    if not files or not any(files.values()):
        return None
    first = next(path for paths in files.values() for path in paths)
    channels, side = 3, MAX_SIDE
    try:
        from PIL import Image

        with Image.open(first) as image:
            channels = 1 if image.mode in ("1", "L", "I", "I;16") else 3
            side = min(max(image.size), MAX_SIDE)
    except ImportError:
        pass   # PIL이 없으면 기본값으로 보여 주고, 적재 때 워커가 분명히 말한다.
    return {"kind": "image_folder", "shape": [channels, side, side], "classes": len(files),
            "class_names": list(files), "count": sum(len(paths) for paths in files.values())}


def _csv_table(folder: Path):
    """(파일, 헤더, 행, 정답 열 번호). CSV가 없으면 None."""
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        return None
    with paths[0].open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if not header or not rows:
        return None
    label = next((index for index, name in enumerate(header)
                  if name.strip().lower() in LABEL_COLUMNS), len(header) - 1)
    return paths[0], header, rows, label


def _inspect_csv(folder: Path) -> dict[str, Any] | None:
    table = _csv_table(folder)
    if table is None:
        return None
    path, header, rows, label = table
    names = sorted({row[label] for row in rows})
    return {"kind": "csv", "shape": [len(header) - 1], "classes": len(names), "class_names": names,
            "count": len(rows), "file": path.name, "label_column": header[label]}


def _npy_header(stream) -> tuple[list[int], str]:
    """``.npy`` 헤더만 읽는다 - numpy 없이 (모양, dtype 문자열)."""
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = stream.read(1)[0]
    stream.read(1)
    length = struct.unpack("<H", stream.read(2))[0] if major == 1 else struct.unpack("<I", stream.read(4))[0]
    header = ast.literal_eval(stream.read(length).decode("latin1"))
    return list(header["shape"]), header["descr"]


_TYPECODES = {"i1": "b", "u1": "B", "i2": "h", "u2": "H", "i4": "i", "u4": "I", "i8": "q", "u8": "Q",
              "f4": "f", "f8": "d"}


def _read_labels(stream) -> list:
    shape, descr = _npy_header(stream)
    values = array.array(_TYPECODES[descr[1:]])
    values.frombytes(stream.read())
    if descr[0] == ">":
        values.byteswap()
    return values.tolist()


def _array_streams(folder: Path):
    """``x``·``y``를 여는 함수 둘. npz 하나 또는 x.npy + y.npy."""
    bundles = sorted(folder.glob("*.npz"))
    if bundles:
        archive = zipfile.ZipFile(bundles[0])
        if {"x.npy", "y.npy"} <= set(archive.namelist()):
            return (lambda: archive.open("x.npy")), (lambda: archive.open("y.npy")), bundles[0].name
        return None
    if (folder / "x.npy").is_file() and (folder / "y.npy").is_file():
        return (lambda: (folder / "x.npy").open("rb")), (lambda: (folder / "y.npy").open("rb")), "x.npy + y.npy"
    return None


def _inspect_arrays(folder: Path) -> dict[str, Any] | None:
    streams = _array_streams(folder)
    if streams is None:
        return None
    open_x, open_y, file = streams
    with open_x() as stream:
        shape, _ = _npy_header(stream)
    with open_y() as stream:
        labels = _read_labels(stream)
    names = sorted(set(labels))
    return {"kind": "arrays", "shape": [int(dim) for dim in shape[1:]], "classes": len(names),
            "class_names": [str(name) for name in names], "count": int(shape[0]), "file": file}


# 적재 (워커 쪽, torch 필요)


def load_any(name: str, base: str | Path) -> dict[str, tuple]:
    """내장이든 사용자 데이터든 ``{"train": (x, y), "test": (x, y)}``. 없으면 ValueError."""
    if name in CATALOGUE:
        return load(name, base)
    folder = Path(base) / name
    spec = inspect(folder) if folder.is_dir() else None
    if spec is None:
        raise ValueError(f"data/{name}에서 데이터셋을 찾지 못했습니다 (이미지 폴더, CSV, x.npy+y.npy 중 하나)")
    loader = {"image_folder": _load_images, "csv": _load_csv, "arrays": _load_arrays}[spec["kind"]]
    x, y = loader(folder, spec)
    return _split(x, y)


def _split(x, y) -> dict[str, tuple]:
    """시드 0으로 섞어 8:2. 같은 데이터면 언제나 같은 분할이라 run끼리 비교할 수 있다."""
    import torch

    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    held = max(1, int(len(y) * VAL_FRACTION))
    test, train = order[:held], order[held:]
    return {"train": (x[train], y[train]), "test": (x[test], y[test])}


def _standardize(train_x, x):
    """train 분할의 평균·표준편차로 맞춘다. 채널(또는 열)마다."""
    dims = [0] + list(range(2, x.dim()))
    mean = train_x.mean(dim=dims, keepdim=True)
    std = train_x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _load_images(folder: Path, spec: dict[str, Any]):
    """이미지를 전부 메모리에 올린다. ponytail: 수만 장까지. 그 위는 디스크 캐시가 필요하다."""
    import torch
    from PIL import Image

    channels, side, _ = spec["shape"]
    mode = "L" if channels == 1 else "RGB"
    images, labels = [], []
    for index, (name, paths) in enumerate(_image_files(folder).items()):
        for path in paths:
            with Image.open(path) as image:
                pixels = image.convert(mode).resize((side, side))
                images.append(torch.frombuffer(bytearray(pixels.tobytes()), dtype=torch.uint8)
                              .reshape(side, side, channels).permute(2, 0, 1).clone())
            labels.append(index)
    x = torch.stack(images).float().div_(255)
    y = torch.tensor(labels, dtype=torch.int64)
    # 정규화 통계는 test 분할을 보면 안 된다 - 분할과 같은 순서로 train만 쓴다.
    return _standardize(_split(x, y)["train"][0], x), y


def _load_csv(folder: Path, spec: dict[str, Any]):
    import torch

    _, header, rows, label = _csv_table(folder)
    names = {name: index for index, name in enumerate(spec["class_names"])}
    features = [[float(value) for column, value in enumerate(row) if column != label] for row in rows]
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([names[row[label]] for row in rows], dtype=torch.int64)
    return _standardize(_split(x, y)["train"][0], x), y


def _load_arrays(folder: Path, spec: dict[str, Any]):
    import numpy
    import torch

    open_x, open_y, _ = _array_streams(folder)
    with open_x() as stream:
        x = torch.from_numpy(numpy.load(stream))
    with open_y() as stream:
        y = torch.from_numpy(numpy.load(stream)).long()
    x = x.float().div_(255) if x.dtype == torch.uint8 else x.float()   # uint8은 픽셀로 본다
    return x, y
