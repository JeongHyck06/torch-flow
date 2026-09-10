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
import re
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
        entry = {"name": name, "kind": "numeric" if numeric else "text",
                 "missing": len(values) - len(present),
                 "uniques": uniques if len(uniques) <= MAX_UNIQUES else None}
        if numeric and present:
            # 고유값이 많아 uniques가 None인 열도 범위는 알아야 한다 - 회귀 정답이 바로 그런 열이다.
            numbers = [_as_float(value) for value in present]
            entry.update(min=min(numbers), max=max(numbers),
                         mean=sum(numbers) / len(numbers), distinct=len(uniques))
        elif present:
            # 자유 텍스트인지 범주인지 가른다. "positive"/"negative"는 범주고, 리뷰 문장은 텍스트다.
            entry.update(distinct=len(uniques),
                         words=sum(len(value.split()) for value in present) / len(present))
        profile.append(entry)
    return profile


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _default_label(header: list[str]) -> str:
    return next((name for name in header if name.lower() in LABEL_COLUMNS), header[-1])


# 수치 정답 열의 고유값이 이보다 많으면 회귀로 짐작한다. 등급(1~5)까지는 분류다.
CLASS_LIMIT = 20

# 한 칸의 평균 낱말 수가 이보다 많으면 범주가 아니라 문장으로 본다.
MIN_TEXT_WORDS = 3.0


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
            "label_column": label, "columns": columns,
            "suggested_task": suggest_task(columns, label),
            "suggested_text_column": suggest_text_column(columns, label)}


def suggest_text_column(columns: list[dict[str, Any]], label: str) -> str | None:
    """자유 텍스트 열 하나. 있으면 그 열이 곧 입력이 된다(특징 열 대신).

    "positive"/"negative" 같은 범주 열과 갈라야 한다 - 둘 다 kind가 text다. 기준은
    **낱말 수**다. 한 칸에 두 낱말 넘게 들어 있으면 문장으로 본다.
    """
    for column in columns:
        if column["name"] == label or column["kind"] != "text":
            continue
        if column.get("words", 0) >= MIN_TEXT_WORDS:
            return column["name"]
    return None


def suggest_task(columns: list[dict[str, Any]], label: str) -> str:
    """정답 열을 보고 분류인지 회귀인지 짐작한다. 사용자가 화면에서 바꿀 수 있다.

    수치이면서 고유값이 많으면 회귀다 - 집값이나 온도 열은 행마다 값이 다르다.
    글자 열이나 고유값이 몇 개뿐인 수치 열(0/1, 1~3 등급)은 분류로 본다.
    """
    column = next((one for one in columns if one["name"] == label), None)
    if column is None or column["kind"] != "numeric":
        return "classification"
    # distinct는 uniques가 잘려도 남는다 - 고유값이 많은 것이 곧 연속형의 신호다.
    distinct = column.get("distinct", len(column["uniques"] or []))
    return "regression" if distinct > CLASS_LIMIT else "classification"


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
                      onehot=True, normalize="standard",
                      task=spec.get("suggested_task", "classification"),
                      target_normalize="standard",
                      text_column=spec.get("suggested_text_column"),
                      max_len=64, vocab_size=8000,
                      series_column=None, window=24, horizon=1, split_mode="random")
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
        if full.get("task") == "language_modeling":
            # 다음 토큰을 맞히므로 라벨 열이 필요 없다. 글은 시간 순이라 분할도 앞뒤로 나눈다.
            text_column = full.get("text_column") or spec.get("suggested_text_column")
            length = int(full.get("max_len") or 64)
            full.update(text_column=text_column, features=[], normalize="none",
                        split_mode="time", series_column=None)
            out["shape"] = [length]
            out["dtype"] = "int64"
            if not text_column:
                out["problem"] = "언어 모델은 문장이 든 열이 있어야 합니다"
            # 표본은 행이 아니라 **토큰 창**이다. 정확한 토큰 수는 열을 다시 읽어야 알지만
            # resolve는 spec만 본다 - 평균 낱말 수로 어림한다. 실제 창은 _load_lm이 센다.
            tokens = int(spec["count"] * ((columns.get(text_column) or {}).get("words") or 0))
            usable = max(0, tokens - length - 1)
            held = max(1, int(usable * float(full["val_fraction"]))) if usable > 1 else 0
            out.update(classes=int(full.get("vocab_size") or 8000), class_names=[],
                       class_counts={}, count=usable,
                       split={"train": usable - held, "val": held})
            return out

        series_column = full.get("series_column")
        if series_column and series_column in columns:
            # 시계열은 과제가 회귀로 못박힌다. 창을 겹쳐 만들기 때문에 분할도 시간 순이다.
            window, horizon = int(full.get("window") or 24), int(full.get("horizon") or 1)
            full.update(task="regression", split_mode="time", features=[], text_column=None)
            out["shape"] = [window]
            usable = max(0, spec["count"] - window - horizon + 1)
            if columns[series_column]["kind"] != "numeric":
                out["problem"] = f"'{series_column}' 열은 수치가 아니라 시계열로 쓸 수 없습니다"
            elif not usable:
                out["problem"] = (f"창 {window} + 예측 {horizon}을 만들려면 값이 최소 "
                                  f"{window + horizon}개 필요합니다 (지금 {spec['count']}개)")
            held = max(1, int(usable * float(full["val_fraction"]))) if usable > 1 else 0
            out.update(classes=0, class_names=[], class_counts={}, count=usable,
                       split={"train": usable - held, "val": held},
                       target=_target_stats(spec, series_column), horizon=horizon)
            return out

        text_column = full.get("text_column")
        if text_column not in columns or text_column == label:
            text_column = full["text_column"] = None
        if text_column:
            # 텍스트는 그 열 하나가 곧 입력이다. 특징 열 조합은 쓰지 않는다.
            full["features"] = []
            out["shape"] = [int(full.get("max_len") or 64)]
            out["dtype"] = "int64"
            # 토큰 번호에 표준화를 걸면 Embedding에 넣을 정수가 아니게 된다.
            full["normalize"] = "none"
            features = []
        else:
            features = [name for name in (full["features"] or list(columns))
                        if name in columns and name != label]
            full["features"] = features
            width = sum(len(columns[name]["uniques"] or []) if columns[name]["kind"] == "text" and full["onehot"]
                        else 1 for name in features)
            out["shape"] = [width]

        if full.get("task") == "regression":
            # 회귀는 클래스가 없다. 정답 열이 수치이기만 하면 된다 - 고유값이 많은 것이
            # 오히려 정상이다(예전에는 바로 그 이유로 막았다).
            if columns[label]["kind"] != "numeric":
                out["problem"] = f"'{label}' 열은 수치가 아니라 회귀 정답으로 쓸 수 없습니다"
            total = spec["count"]
            if full.get("limit"):
                total = min(total, int(full["limit"]))
            held = max(1, int(total * float(full["val_fraction"]))) if total > 1 else 0
            out.update(classes=0, class_names=[], class_counts={}, count=total,
                       split={"train": total - held, "val": held},
                       target=_target_stats(spec, label))
            if not features and not text_column:
                out["problem"] = "특징 열이 하나도 없습니다"
            return out

        if uniques is None:
            out.update(classes=0, class_names=[], class_counts={}, count=spec["count"],
                       shape=[0], problem=f"'{label}' 열은 고유값이 {MAX_UNIQUES}개를 넘습니다"
                                          " - 회귀로 바꾸거나 다른 열을 정답으로 고르세요")
            return out
        # 정답 열이 바뀌면 클래스별 개수는 미리보기가 다시 센다 - 여기서는 이름만 안다.
        names = list(uniques)
        counts = dict(spec["class_counts"]) if label == spec["label_column"] else None
        if not features and not text_column:
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


def _target_stats(spec: dict[str, Any], label: str) -> dict[str, Any] | None:
    """회귀 정답 열의 범위. 화면이 "무엇을 맞히려는지"를 숫자로 보여 준다."""
    column = next((one for one in spec["columns"] if one["name"] == label), None)
    if column is None or "min" not in column:
        return None
    return {key: column[key] for key in ("min", "max", "mean", "distinct") if key in column}


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
    """내장이든 사용자 데이터든 ``{"train": (x, y), "test": (x, y)}``. 없으면 ValueError.

    회귀는 ``target_scale``이 함께 온다(``{"mean", "std"}`` 또는 ``None``) - 정답을
    표준화했으므로 지표를 원래 단위로 되돌리려면 이 값이 필요하다.
    """
    if name in CATALOGUE:
        return {**load(name, base), "target_scale": None}
    folder = Path(base) / name
    spec = inspect(folder) if folder.is_dir() else None
    if spec is None:
        raise ValueError(f"data/{name}에서 데이터셋을 찾지 못했습니다 (이미지 폴더, CSV, x.npy+y.npy 중 하나)")
    effective = resolve(spec, recipe)
    if effective.get("problem"):
        raise ValueError(effective["problem"])
    loader = {"image_folder": _load_images, "csv": _load_csv, "arrays": _load_arrays}[spec["kind"]]
    if spec["kind"] == "csv" and effective["recipe"].get("task") == "language_modeling":
        loader = _load_lm
    elif spec["kind"] == "csv" and effective["recipe"].get("series_column"):
        loader = _load_series
    elif spec["kind"] == "csv" and effective["recipe"].get("text_column"):
        loader = _load_text
    x, y = loader(folder, spec, effective)
    x, y = _limit(x, y, effective["recipe"])
    splits = _split(x, y, effective["recipe"])
    splits, scale = _target_normalize(splits, effective)
    return {**_normalize(splits, effective), "target_scale": scale}


def _limit(x, y, recipe):
    """샘플 수 제한은 시드로 섞어 뽑는다 - 앞에서 자르면 한 클래스만 남는다."""
    import torch

    limit = recipe.get("limit")
    if limit and int(limit) < len(y):
        keep = torch.randperm(len(y), generator=torch.Generator().manual_seed(int(recipe["seed"])))[:int(limit)]
        return x[keep], y[keep]
    return x, y


def split_indices(count: int, recipe: dict[str, Any]):
    """``(train, test)`` 인덱스. 분할을 정하는 곳은 여기 하나다.

    텍스트 어휘를 train 분할에서만 만들려면 **적재 도중에** 어느 행이 train인지 알아야
    한다. 그때와 `_split`이 각자 섞으면 어휘가 검증 문장을 보게 된다 - 누수다.

    ``split_mode``가 ``"time"``이면 섞지 않고 **뒤쪽**을 검증으로 뗀다. 시계열을
    무작위로 나누면 검증 구간의 앞뒤 값이 학습에 들어가 미래를 보고 미래를 맞히는
    셈이 된다 - 점수는 좋아지고 실제로는 아무 쓸모가 없다.
    """
    import torch

    held = max(1, int(count * float(recipe["val_fraction"]))) if count > 1 else 0
    if recipe.get("split_mode") == "time":
        order = torch.arange(count)
        return order[:count - held], order[count - held:]
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(recipe["seed"])))
    return order[held:], order[:held]


def _split(x, y, recipe) -> dict[str, tuple]:
    """시드로 섞어 val 비율만큼 떼어 둔다. 같은 레시피면 언제나 같은 분할이다."""
    train, test = split_indices(len(y), recipe)
    return {"train": (x[train], y[train]), "test": (x[test], y[test])}


def _target_normalize(splits: dict[str, tuple], effective: dict[str, Any]):
    """회귀 정답을 train 분할 통계로 표준화하고 되돌릴 값을 준다.

    집값처럼 정답이 수백 단위면 MSE가 수만에서 시작해 기본 lr로는 사실상 학습되지
    않는다. 그렇다고 표준화한 채로 RMSE를 보고하면 "0.3"이 얼마나 좋은지 아무도 모른다.
    그래서 **배우는 것은 표준화한 값, 보고하는 것은 원래 단위**로 나눈다 - 되돌리는 데
    필요한 (mean, std)를 함께 돌려준다.
    """
    recipe = effective["recipe"]
    if recipe.get("task") != "regression" or recipe.get("target_normalize") != "standard":
        return splits, None
    train_y = splits["train"][1]
    mean = float(train_y.mean())
    std = max(float(train_y.std()), 1e-6)
    return ({split: (x, (y - mean) / std) for split, (x, y) in splits.items()},
            {"mean": mean, "std": std})


def _normalize(splits: dict[str, tuple], effective: dict[str, Any]) -> dict[str, tuple]:
    """정규화 통계는 train 분할에서만 낸다 - test를 보면 검증이 거짓말을 한다."""
    import torch

    how = effective["recipe"].get("normalize", "none")
    train_x = splits["train"][0]
    # 토큰 번호는 크기에 뜻이 없다(어휘 500번이 250번의 두 배가 아니다). 표준화하면
    # Embedding에 넣을 정수가 아니게 되고 그 자리에서 죽는다.
    if not train_x.is_floating_point():
        return splits
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

    regression = recipe.get("task") == "regression"
    x_rows, labels = [], []
    for row in rows:
        if regression:
            # 회귀는 정답이 수치 그대로다. 읽을 수 없는 칸은 그 행을 버린다 -
            # 정답을 평균으로 채우면 모델이 채워 넣은 값을 배운다.
            number = _as_float(row[label])
            if number is None:
                continue
            labels.append(number)
        else:
            if row[label] not in index:
                continue
            labels.append(index[row[label]])
        x_rows.append([value for name in features for value in encode(name, row[header.index(name)])])
    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32 if regression else torch.int64)
    nan = torch.isnan(x)
    if nan.any():
        if recipe["missing"] == "mean":
            means = torch.nan_to_num(x).sum(dim=0) / (~nan).sum(dim=0).clamp_min(1)
            x = torch.where(nan, means.expand_as(x), x)
        else:   # drop: 결측이 하나라도 있는 행을 버린다
            keep = ~nan.any(dim=1)
            x, y = x[keep], y[keep]
    return x, y


TOKEN = re.compile(r"[0-9a-z가-힣]+")
PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    """소문자 낱말. 형태소 분석기 없이 공백·문장부호로만 자른다.

    ponytail: 영어와 한글 어절 수준. 한국어를 제대로 하려면 형태소 분석기가 필요한데,
    그건 무거운 의존성이라 사용자 코드(Code Cell)로 붙이는 편이 낫다.
    """
    return TOKEN.findall(text.lower())


def build_vocab(texts: list[str], limit: int) -> dict[str, int]:
    """자주 나온 낱말부터 어휘에. 0은 padding, 1은 모르는 낱말로 비워 둔다.

    **train 분할의 문장만** 넣어야 한다. 검증 문장의 낱말이 어휘에 있으면 모델이
    그 낱말을 이미 아는 상태로 평가된다.
    """
    counts: dict[str, int] = {}
    for text in texts:
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {word: index for index, (word, _) in enumerate(ordered[:max(0, limit - 2)], start=2)}


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """낱말을 번호로. 짧으면 0으로 채우고 길면 자른다."""
    ids = [vocab.get(word, UNK) for word in tokenize(text)][:max_len]
    return ids + [PAD] * (max_len - len(ids))


def _load_text(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    """텍스트 열 하나를 토큰 번호 행렬로. ``x``는 ``[N, max_len]`` int64다."""
    import torch

    _, header, rows = _csv_table(folder)
    recipe = effective["recipe"]
    column = header.index(recipe["text_column"])
    label = header.index(recipe["label_column"])
    index = _class_index(effective)
    regression = recipe.get("task") == "regression"

    kept, targets = [], []
    for row in rows:
        text = row[column]
        if _is_missing(text):
            continue
        if regression:
            number = _as_float(row[label])
            if number is None:
                continue
            targets.append(number)
        else:
            if row[label] not in index:
                continue
            targets.append(index[row[label]])
        kept.append(text)

    max_len = int(recipe.get("max_len") or 64)
    train, _ = split_indices(len(kept), recipe)
    vocab = build_vocab([kept[int(i)] for i in train], int(recipe.get("vocab_size") or 8000))
    x = torch.tensor([encode_text(text, vocab, max_len) for text in kept], dtype=torch.int64)
    y = torch.tensor(targets, dtype=torch.float32 if regression else torch.int64)
    return x, y


def _load_series(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    """한 열의 시간 순 값을 ``(과거 window, 다음 horizon)`` 쌍으로 자른다.

    창은 한 칸씩 밀며 만든다. 창 i의 입력은 ``값[i : i+window]``이고 정답은
    ``값[i+window : i+window+horizon]``이다. 창끼리 겹치므로 무작위로 나누면 검증 창의
    값이 학습 창에도 들어간다 - 그래서 이 경로는 ``split_mode``를 ``"time"``으로 못박는다.
    """
    import torch

    _, header, rows = _csv_table(folder)
    recipe = effective["recipe"]
    column = header.index(recipe["series_column"])
    window = int(recipe.get("window") or 24)
    horizon = int(recipe.get("horizon") or 1)

    values = []
    for row in rows:
        number = _as_float(row[column])
        if number is not None:
            values.append(number)
    if len(values) < window + horizon + 1:
        raise ValueError(f"창 {window} + 예측 {horizon}을 만들려면 값이 최소 "
                         f"{window + horizon + 1}개 필요합니다 (지금 {len(values)}개)")

    series = torch.tensor(values, dtype=torch.float32)
    count = len(series) - window - horizon + 1
    x = torch.stack([series[i:i + window] for i in range(count)])
    y = torch.stack([series[i + window:i + window + horizon] for i in range(count)])
    return x, y.squeeze(1) if horizon == 1 else y


def _load_lm(folder: Path, spec: dict[str, Any], effective: dict[str, Any]):
    """텍스트 열 전체를 토큰 한 줄로 이어 붙이고 ``(x, 한 칸 민 x)`` 창으로 자른다.

    라벨 열이 필요 없다 - 정답이 입력 안에 있다. 어휘는 시계열과 같은 이유로 **앞쪽**
    (train 구간)에서만 만든다. 뒤쪽 글의 낱말이 어휘에 있으면 모델이 그 낱말을 이미
    아는 상태로 평가된다.
    """
    import torch

    _, header, rows = _csv_table(folder)
    recipe = effective["recipe"]
    column = header.index(recipe["text_column"])
    length = int(recipe.get("max_len") or 64)

    words: list[str] = []
    for row in rows:
        if not _is_missing(row[column]):
            words.extend(tokenize(row[column]))
    # 어휘 경계는 글 개수가 아니라 **토큰 위치**로 잡는다. 글 길이가 제각각이면
    # 글 수로 자른 경계가 실제 학습 구간과 어긋나 뒤쪽 낱말이 어휘에 샌다.
    cut = max(1, int(len(words) * (1 - float(recipe["val_fraction"]))))
    vocab = build_vocab([" ".join(words[:cut])], int(recipe.get("vocab_size") or 8000))
    stream = [vocab.get(word, UNK) for word in words]
    if len(stream) < length + 2:
        raise ValueError(f"길이 {length} 창을 만들려면 토큰이 최소 {length + 2}개 필요합니다 "
                         f"(지금 {len(stream)}개)")

    tokens = torch.tensor(stream, dtype=torch.int64)
    count = len(tokens) - length - 1
    x = torch.stack([tokens[i:i + length] for i in range(count)])
    y = torch.stack([tokens[i + 1:i + length + 1] for i in range(count)])
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
