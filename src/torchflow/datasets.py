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
import base64
import csv
import gzip
import io
import math
import struct
import pickle
import tarfile
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
    # ResNet-18·MiniViT 템플릿이 배우는 데이터. 없으면 두 템플릿은 무작위 합성 과제로 돌아
    # 정확도가 10 % 근처에서 끝난다 - 사람은 그것을 "모델이 고장났다"로 읽는다.
    "cifar10": {
        "label": "CIFAR-10", "shape": [3, 32, 32], "classes": 10, "size_mb": 163,
        "url": "https://www.cs.toronto.edu/~kriz/",
        "files": {"archive": "cifar-10-python.tar.gz"},
        "format": "cifar",
        "names": ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"],
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616],
    },
}

CIFAR_TRAIN = [f"cifar-10-batches-py/data_batch_{index}" for index in range(1, 6)]
CIFAR_TEST = ["cifar-10-batches-py/test_batch"]


def _cifar_batches(archive: Path, members: list[str]) -> tuple[bytes, list[int]]:
    """tar 안의 pickle 배치들 -> (픽셀 바이트, 라벨). 픽셀은 이미지마다 R·G·B 평면 1024바이트씩."""
    pixels, labels = [], []
    with tarfile.open(archive, "r:gz") as tar:
        for member in members:
            stream = tar.extractfile(member)
            if stream is None:
                raise OSError(f"{archive.name}에 {member}가 없습니다")
            batch = pickle.load(stream, encoding="bytes")
            pixels.append(bytes(batch[b"data"]))
            labels.extend(int(label) for label in batch[b"labels"])
    return b"".join(pixels), labels


def folder(name: str, base: str | Path) -> Path:
    return Path(base) / name


def available(name: str, base: str | Path) -> bool:
    return all((folder(name, base) / filename).is_file()
               for filename in CATALOGUE[name]["files"].values())


def progress(name: str, base: str | Path) -> dict[str, int]:
    """받은 바이트와 예상 총량. 받는 중인 파일은 ``.part``라 디스크만 보면 진행률이 나온다 -
    다운로드 스레드가 따로 보고하지 않아도 되고, 다른 창이 시작한 다운로드도 보인다."""
    entry = CATALOGUE[name]
    target = folder(name, base)
    got = 0
    for filename in entry["files"].values():
        for path in (target / filename, target / f"{filename}.part"):
            if path.is_file():
                got += path.stat().st_size
                break
    return {"bytes": got, "total": int(entry["size_mb"] * 1024 * 1024)}


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


def _idx(path: Path) -> tuple[tuple[int, ...], bytes]:
    """IDX(gzip) -> (모양, 원시 바이트). 헤더는 magic 4바이트(0, 0, 타입, 차원 수) + 차원별 int32 big-endian."""
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    dims = raw[3]
    return struct.unpack(">" + "I" * dims, raw[4:4 + 4 * dims]), raw[4 + 4 * dims:]


def read_idx(path: Path):
    """IDX(gzip) -> uint8 텐서."""
    import torch

    shape, raw = _idx(path)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)


def load(name: str, base: str | Path) -> dict[str, tuple]:
    """``{"train": (x, y), "test": (x, y)}``. x는 정규화된 float32 ``[N, C, H, W]``, y는 int64."""
    entry = CATALOGUE[name]
    target = folder(name, base)
    if entry.get("format") == "cifar":
        return _load_cifar(target / entry["files"]["archive"], entry)
    splits = {}
    for split in ("train", "test"):
        images = read_idx(target / entry["files"][f"{split}_images"]).float().div_(255)
        images = images.sub_(entry["mean"]).div_(entry["std"]).reshape(-1, *entry["shape"])
        labels = read_idx(target / entry["files"][f"{split}_labels"]).long()
        splits[split] = (images, labels)
    return splits


def _load_cifar(archive: Path, entry: dict[str, Any]) -> dict[str, tuple]:
    # ponytail: 5만 장을 float32로 통째로 올린다(약 600 MB). 그 위는 디스크 캐시가 필요하다.
    import torch

    mean = torch.tensor(entry["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(entry["std"]).view(1, 3, 1, 1)
    splits = {}
    for split, members in (("train", CIFAR_TRAIN), ("test", CIFAR_TEST)):
        raw, labels = _cifar_batches(archive, members)
        images = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, *entry["shape"])
        images = images.float().div_(255).sub_(mean).div_(std)
        splits[split] = (images, torch.tensor(labels, dtype=torch.int64))
    return splits


# 사용자 데이터 (data/<이름>/)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
LABEL_COLUMNS = ("label", "target", "y", "class")
MISSING = {"", "na", "n/a", "nan", "null", "none", "?"}
MAX_SIDE = 64          # 자동 인식 기본값. 레시피의 size가 이긴다.
MAX_UNIQUES = 100      # 고유값이 이보다 많은 열은 정답(클래스) 열로 쓰지 않는다.
MAX_PREVIEW_CLASSES = 10   # 미리보기 격자는 이 수까지만 - 클래스 100개면 격자가 화면을 넘긴다.
VAL_FRACTION = 0.2     # test 분할이 없는 사용자 데이터는 시드 0으로 8:2 나눈다.
SIZES = (28, 32, 48, 64, 96, 128)


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
    counts = {name: len(paths) for name, paths in files.items()}
    return {"kind": "image_folder", "shape": [channels, side, side], "classes": len(files),
            "class_names": list(files), "class_counts": counts, "count": sum(counts.values())}


def _csv_table(folder: Path):
    """(파일, 헤더, 행). CSV가 없으면 None. 헤더보다 짧은 행은 빈 칸으로 채운다."""
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        return None
    with paths[0].open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [name.strip() for name in (next(reader, None) or [])]
        rows = [[value.strip() for value in row] + [""] * (len(header) - len(row))
                for row in reader if any(value.strip() for value in row)]
    if not header or not rows:
        return None
    return paths[0], header, rows


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING


def _column_profile(header: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    """열마다 종류(numeric/text), 결측 수, 고유값(적을 때만)."""
    profile = []
    for index, name in enumerate(header):
        values = [row[index] for row in rows]
        present = [value for value in values if not _is_missing(value)]
        numeric = all(_as_float(value) is not None for value in present)
        uniques = sorted(set(present), key=lambda v: (float(v) if numeric else 0, v))
        profile.append({"name": name, "kind": "numeric" if numeric else "text",
                        "missing": len(values) - len(present),
                        "uniques": uniques if len(uniques) <= MAX_UNIQUES else None})
    return profile


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _default_label(header: list[str]) -> str:
    return next((name for name in header if name.lower() in LABEL_COLUMNS), header[-1])


def _inspect_csv(folder: Path) -> dict[str, Any] | None:
    table = _csv_table(folder)
    if table is None:
        return None
    path, header, rows = table
    columns = _column_profile(header, rows)
    label = _default_label(header)
    uniques = next(column["uniques"] for column in columns if column["name"] == label) or []
    counts = {name: 0 for name in uniques}
    label_index = header.index(label)
    for row in rows:
        if row[label_index] in counts:
            counts[row[label_index]] += 1
    return {"kind": "csv", "shape": [len(header) - 1], "classes": len(uniques), "class_names": uniques,
            "class_counts": counts, "count": len(rows), "file": path.name,
            "label_column": label, "columns": columns}


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
        shape, descr = _npy_header(stream)
    with open_y() as stream:
        labels = _read_labels(stream)
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    names = sorted(counts, key=lambda name: float(name))
    return {"kind": "arrays", "shape": [int(dim) for dim in shape[1:]], "classes": len(names),
            "class_names": names, "class_counts": {name: counts[name] for name in names},
            "count": int(shape[0]), "file": file, "dtype": descr}


# 레시피: 사람이 정제한 설정. 자동 인식 위에 얹히고 그래프의 experiment.data에 남는다.


def recipe_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    recipe: dict[str, Any] = {"val_fraction": VAL_FRACTION, "seed": 0, "classes": None, "limit": None}
    if spec["kind"] == "csv":
        recipe.update(label_column=spec["label_column"], features=None, missing="drop",
                      onehot=True, normalize="standard")
    elif spec["kind"] == "image_folder":
        recipe.update(size=spec["shape"][1], channels=spec["shape"][0], normalize="standard")
    elif spec["kind"] == "arrays":
        recipe.update(scale="auto", normalize="none")
    return recipe


def resolve(spec: dict[str, Any], recipe: dict[str, Any] | None) -> dict[str, Any]:
    """레시피를 적용한 뒤의 명세 - 입력 모양, 클래스, 개수, 분할. 적재 없이 셀 수 있는 것만.

    내장 데이터셋은 레시피가 없다(그대로 돌려준다).
    """
    if spec["kind"] == "builtin":
        return {**spec, "recipe": {}}
    full = {**recipe_defaults(spec), **{key: value for key, value in (recipe or {}).items()
                                       if value is not None}}
    out = {**spec, "recipe": full}
    counts = dict(spec.get("class_counts") or {})
    names = list(spec.get("class_names") or [])

    if spec["kind"] == "csv":
        columns = {column["name"]: column for column in spec["columns"]}
        label = full["label_column"] if full["label_column"] in columns else spec["label_column"]
        full["label_column"] = label
        uniques = columns[label]["uniques"]
        if uniques is None:
            out.update(classes=0, class_names=[], class_counts={}, count=spec["count"],
                       shape=[0], problem=f"'{label}' 열은 고유값이 {MAX_UNIQUES}개를 넘어 정답 열로 쓸 수 없습니다")
            return out
        # 정답 열이 바뀌면 클래스별 개수는 미리보기가 다시 센다 - 여기서는 이름만 안다.
        names = list(uniques)
        counts = dict(spec["class_counts"]) if label == spec["label_column"] else None
        features = [name for name in (full["features"] or list(columns)) if name in columns and name != label]
        full["features"] = features
        width = sum(len(columns[name]["uniques"] or []) if columns[name]["kind"] == "text" and full["onehot"]
                    else 1 for name in features)
        out["shape"] = [width]
        if not features:
            out["problem"] = "특징 열이 하나도 없습니다"
    elif spec["kind"] == "image_folder":
        out["shape"] = [int(full["channels"]), int(full["size"]), int(full["size"])]

    if full.get("classes"):
        chosen = [name for name in names if name in set(full["classes"])]
        counts = {name: counts.get(name, 0) for name in chosen} if counts is not None else None
        names = chosen
    total = sum(counts.values()) if counts is not None else spec.get("count", 0)
    if full.get("limit"):
        total = min(total, int(full["limit"]))
    held = max(1, int(total * float(full["val_fraction"]))) if total > 1 else 0
    out.update(classes=len(names), class_names=names, class_counts=counts or {}, count=total,
               split={"train": total - held, "val": held})
    if len(names) < 2 and "problem" not in out:
        out["problem"] = "클래스가 둘 이상이어야 분류를 배웁니다"
    return out


def preview(spec: dict[str, Any], base: str | Path, recipe: dict[str, Any] | None) -> dict[str, Any]:
    """정제 화면이 보여 줄 것: 레시피 적용 후 명세와 종류별 미리보기. torch 없이."""
    effective = resolve(spec, recipe)
    folder = Path(base) / spec["name"]
    extra: dict[str, Any] = {}
    if spec["kind"] == "csv":
        _, header, rows = _csv_table(folder)
        label = effective["recipe"]["label_column"]
        index = header.index(label)
        counts = {name: 0 for name in effective["class_names"]}
        for row in rows:
            if row[index] in counts:
                counts[row[index]] += 1
        effective["class_counts"] = counts
        extra = {"columns": spec["columns"], "rows": rows[:5], "header": header}
    elif spec["kind"] == "image_folder":
        extra = {"thumbnails": _safe_thumbnails(lambda: _folder_samples(folder, effective))}
    elif spec["kind"] == "builtin" and spec["available"]:
        # 내장 데이터는 레시피가 없지만 무엇이 들었는지는 보여야 한다. 라벨 파일은 작다.
        entry = CATALOGUE[spec["name"]]
        if entry.get("format") == "cifar":
            archive = folder / entry["files"]["archive"]
            labels = bytes(_cifar_batches(archive, CIFAR_TRAIN)[1])
            held = len(_cifar_batches(archive, CIFAR_TEST)[1])
        else:
            labels = _idx(folder / entry["files"]["train_labels"])[1]
            held = _idx(folder / entry["files"]["test_labels"])[0][0]
        names = entry.get("names") or [str(label) for label in range(entry["classes"])]
        counts = {name: labels.count(label) for label, name in enumerate(names)}
        effective.update(class_names=names, class_counts=counts, count=len(labels) + held,
                         split={"train": len(labels), "val": held})
        extra = {"thumbnails": _safe_thumbnails(lambda: _builtin_samples(folder, spec["name"], labels))}
    return {"base": spec, "spec": effective, "preview": extra}


def _safe_thumbnails(build) -> dict | None:
    """PIL이 없으면 격자 없이 간다 - 적재 때 워커가 분명히 말한다."""
    try:
        return _thumbnails(build())
    except ImportError:
        return None


def _folder_samples(folder: Path, effective: dict[str, Any], per_class: int = 6) -> dict[str, list]:
    from PIL import Image

    files = _image_files(folder)
    mode = "L" if effective["shape"][0] == 1 else "RGB"
    samples: dict[str, list] = {}
    for name in effective["class_names"][:MAX_PREVIEW_CLASSES]:
        samples[name] = []
        for path in files.get(name, [])[:per_class]:
            with Image.open(path) as image:
                samples[name].append(image.convert(mode))
    return samples


def _builtin_samples(folder: Path, name: str, labels: bytes, per_class: int = 6) -> dict[str, list]:
    """IDX 이미지에서 클래스마다 앞쪽 몇 장. torch 없이 바이트를 그대로 PIL에 넣는다."""
    from PIL import Image

    entry = CATALOGUE[name]
    names = entry.get("names") or [str(label) for label in range(entry["classes"])]
    if entry.get("format") == "cifar":
        # 미리보기는 첫 배치(1만 장)면 충분하다. 평면(RRR…GGG…BBB)을 채널로 합친다.
        pixels, first = _cifar_batches(folder / entry["files"]["archive"], CIFAR_TRAIN[:1])
        labels = bytes(first)
        rows = cols = entry["shape"][1]
        plane = rows * cols

        def tile(at: int):
            start = at * 3 * plane
            return Image.merge("RGB", [Image.frombytes("L", (cols, rows),
                                                       pixels[start + channel * plane:start + (channel + 1) * plane])
                                       for channel in range(3)])
    else:
        (_, rows, cols), pixels = _idx(folder / entry["files"]["train_images"])
        size = rows * cols

        def tile(at: int):
            return Image.frombytes("L", (cols, rows), pixels[at * size:(at + 1) * size])
    samples: dict[str, list] = {}
    for label in range(min(entry["classes"], MAX_PREVIEW_CLASSES)):
        images, at = [], -1
        while len(images) < per_class and (at := labels.find(bytes([label]), at + 1)) >= 0:
            images.append(tile(at))
        samples[names[label]] = images
    return samples


def _thumbnails(samples: dict[str, list], tile: int = 40) -> dict:
    """{클래스: [PIL 이미지]} -> 클래스마다 한 줄인 격자 PNG 하나."""
    from PIL import Image

    names = list(samples)[:MAX_PREVIEW_CLASSES]
    per_class = max((len(samples[name]) for name in names), default=0)
    sheet = Image.new("RGB", (max(per_class, 1) * tile, max(len(names), 1) * tile), (249, 250, 251))
    for row, name in enumerate(names):
        for column, image in enumerate(samples[name]):
            sheet.paste(image.resize((tile, tile)).convert("RGB"), (column * tile, row * tile))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return {"png": base64.b64encode(buffer.getvalue()).decode(), "labels": names,
            "tile": tile, "per_class": per_class}


# 적재 (워커 쪽, torch 필요)


def load_any(name: str, base: str | Path, recipe: dict[str, Any] | None = None) -> dict[str, tuple]:
    """내장이든 사용자 데이터든 ``{"train": (x, y), "test": (x, y)}``. 없으면 ValueError."""
    if name in CATALOGUE:
        return load(name, base)
    folder = Path(base) / name
    spec = inspect(folder) if folder.is_dir() else None
    if spec is None:
        raise ValueError(f"data/{name}에서 데이터셋을 찾지 못했습니다 (이미지 폴더, CSV, x.npy+y.npy 중 하나)")
    effective = resolve(spec, recipe)
    if effective.get("problem"):
        raise ValueError(effective["problem"])
    loader = {"image_folder": _load_images, "csv": _load_csv, "arrays": _load_arrays}[spec["kind"]]
    x, y = loader(folder, spec, effective)
    x, y = _limit(x, y, effective["recipe"])
    splits = _split(x, y, effective["recipe"])
    return _normalize(splits, effective)


def _limit(x, y, recipe):
    """샘플 수 제한은 시드로 섞어 뽑는다 - 앞에서 자르면 한 클래스만 남는다."""
    import torch

    limit = recipe.get("limit")
    if limit and int(limit) < len(y):
        keep = torch.randperm(len(y), generator=torch.Generator().manual_seed(int(recipe["seed"])))[:int(limit)]
        return x[keep], y[keep]
    return x, y


def _split(x, y, recipe) -> dict[str, tuple]:
    """시드로 섞어 val 비율만큼 떼어 둔다. 같은 레시피면 언제나 같은 분할이다."""
    import torch

    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(int(recipe["seed"])))
    held = max(1, int(len(y) * float(recipe["val_fraction"]))) if len(y) > 1 else 0
    test, train = order[:held], order[held:]
    return {"train": (x[train], y[train]), "test": (x[test], y[test])}


def _normalize(splits: dict[str, tuple], effective: dict[str, Any]) -> dict[str, tuple]:
    """정규화 통계는 train 분할에서만 낸다 - test를 보면 검증이 거짓말을 한다."""
    import torch

    how = effective["recipe"].get("normalize", "none")
    train_x = splits["train"][0]
    if how == "standard":
        dims = [0] + list(range(2, train_x.dim()))
        mean = train_x.mean(dim=dims, keepdim=True)
        std = train_x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        apply = lambda x: (x - mean) / std
    elif how == "minmax":
        low = train_x.amin(dim=0, keepdim=True)
        span = (train_x.amax(dim=0, keepdim=True) - low).clamp_min(1e-6)
        apply = lambda x: (x - low) / span
    elif how == "fixed":
        apply = lambda x: (x - 0.5) / 0.5
    else:
        return splits
    return {split: (apply(x), y) for split, (x, y) in splits.items()}


def _class_index(effective: dict[str, Any]) -> dict[str, int]:
    return {name: index for index, name in enumerate(effective["class_names"])}


def _load_images(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    """이미지를 전부 메모리에 올린다. ponytail: 수만 장까지. 그 위는 디스크 캐시가 필요하다."""
    import torch
    from PIL import Image

    channels, side, _ = effective["shape"]
    mode = "L" if channels == 1 else "RGB"
    index = _class_index(effective)
    images, labels = [], []
    for name, paths in _image_files(folder).items():
        if name not in index:
            continue
        for path in paths:
            with Image.open(path) as image:
                pixels = image.convert(mode).resize((side, side))
                images.append(torch.frombuffer(bytearray(pixels.tobytes()), dtype=torch.uint8)
                              .reshape(side, side, channels).permute(2, 0, 1).clone())
            labels.append(index[name])
    return torch.stack(images).float().div_(255), torch.tensor(labels, dtype=torch.int64)


def _load_csv(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    import torch

    _, header, rows = _csv_table(folder)
    recipe = effective["recipe"]
    columns = {column["name"]: column for column in spec["columns"]}
    label = header.index(recipe["label_column"])
    index = _class_index(effective)
    features = recipe["features"]

    def encode(name: str, value: str) -> list[float]:
        column = columns[name]
        if column["kind"] == "text":
            if recipe["onehot"]:
                return [1.0 if value == choice else 0.0 for choice in column["uniques"] or []]
            return [float((column["uniques"] or []).index(value)) if value in (column["uniques"] or []) else math.nan]
        number = _as_float(value)
        return [math.nan if number is None else number]

    x_rows, labels = [], []
    for row in rows:
        if row[label] not in index:
            continue
        x_rows.append([value for name in features for value in encode(name, row[header.index(name)])])
        labels.append(index[row[label]])
    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.int64)
    nan = torch.isnan(x)
    if nan.any():
        if recipe["missing"] == "mean":
            means = torch.nan_to_num(x).sum(dim=0) / (~nan).sum(dim=0).clamp_min(1)
            x = torch.where(nan, means.expand_as(x), x)
        else:   # drop: 결측이 하나라도 있는 행을 버린다
            keep = ~nan.any(dim=1)
            x, y = x[keep], y[keep]
    return x, y


def _load_arrays(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    import numpy
    import torch

    open_x, open_y, _ = _array_streams(folder)
    with open_x() as stream:
        x = torch.from_numpy(numpy.load(stream))
    with open_y() as stream:
        y = torch.from_numpy(numpy.load(stream)).long()
    scale = effective["recipe"].get("scale", "auto")
    x = x.float().div_(255) if (x.dtype == torch.uint8 and scale == "auto") else x.float()
    index = _class_index(effective)
    remap = torch.full((int(y.max()) + 1,), -1, dtype=torch.int64)
    for name, position in index.items():
        remap[int(float(name))] = position
    y = remap[y]
    keep = y >= 0
    return x[keep], y[keep]
