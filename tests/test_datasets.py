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


def write_image_folder(base: Path, side: int = 20, per_class: int = 3) -> Path:
    from PIL import Image

    folder = base / "shapes"
    for label, color in (("red", (220, 40, 40)), ("blue", (40, 40, 220))):
        (folder / label).mkdir(parents=True)
        for index in range(per_class):
            Image.new("RGB", (side + index, side), color).save(folder / label / f"{index}.png")
    return folder


def test_image_folder_is_recognised_and_loaded(tmp_path):
    folder = write_image_folder(tmp_path)
    spec = datasets.inspect(folder)
    assert spec["kind"] == "image_folder" and spec["shape"] == [3, 20, 20]
    assert spec["classes"] == 2 and spec["class_names"] == ["blue", "red"] and spec["count"] == 6

    splits = datasets.load_any("shapes", tmp_path)
    x, y = splits["train"]
    assert x.shape[1:] == (3, 20, 20) and len(x) + len(splits["test"][0]) == 6
    assert set(y.tolist()) <= {0, 1}


def test_csv_table_is_recognised_and_standardised(tmp_path):
    folder = tmp_path / "table"
    folder.mkdir()
    rows = ["height,weight,label"] + [f"{170 + i},{60 + 2 * i},{'a' if i % 2 else 'b'}" for i in range(10)]
    (folder / "people.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = datasets.inspect(folder)
    assert spec["kind"] == "csv" and spec["shape"] == [2] and spec["class_names"] == ["a", "b"]
    assert spec["label_column"] == "label" and spec["count"] == 10

    x, y = datasets.load_any("table", tmp_path)["train"]
    assert x.shape[1] == 2 and abs(float(x.mean())) < 0.5   # train 통계로 표준화됐다
    assert y.dtype == torch.int64


def test_npy_pair_is_recognised_without_numpy_in_the_hub(tmp_path):
    import numpy

    folder = tmp_path / "arrays"
    folder.mkdir()
    numpy.save(folder / "x.npy", numpy.zeros((8, 4), dtype=numpy.float32))
    numpy.save(folder / "y.npy", numpy.array([0, 1, 2] * 2 + [0, 1], dtype=numpy.int64))
    spec = datasets.inspect(folder)
    assert spec["kind"] == "arrays" and spec["shape"] == [4] and spec["classes"] == 3 and spec["count"] == 8

    listed = [entry["name"] for entry in datasets.scan(tmp_path)]
    assert listed == ["mnist", "cifar10", "arrays"]
    assert datasets.load_any("arrays", tmp_path)["train"][0].shape[1] == 4


def test_recipe_changes_label_column_features_and_missing_values(tmp_path):
    """정제: 정답 열을 바꾸고, 특징 열을 고르고, 범주형은 원-핫, 결측은 평균으로."""
    folder = tmp_path / "table"
    folder.mkdir()
    (folder / "t.csv").write_text(
        "age,city,score,label\n"
        "30,seoul,1.0,a\n40,busan,,b\n50,seoul,3.0,a\n60,busan,4.0,b\n70,seoul,5.0,a\n",
        encoding="utf-8")
    spec = datasets.inspect(folder)
    assert [c["kind"] for c in spec["columns"]] == ["numeric", "text", "numeric", "text"]
    assert spec["columns"][2]["missing"] == 1

    recipe = {"label_column": "city", "features": ["age", "score"], "missing": "mean",
              "normalize": "none", "val_fraction": 0.2}
    effective = datasets.resolve(spec, recipe)
    assert effective["class_names"] == ["busan", "seoul"] and effective["shape"] == [2]
    assert effective["split"] == {"train": 4, "val": 1}

    splits = datasets.load_any("table", tmp_path, recipe)
    x = torch.cat([splits["train"][0], splits["test"][0]])
    assert x.shape == (5, 2) and not torch.isnan(x).any()
    assert abs(float(x[:, 1].sum()) - (1 + 3.25 + 3 + 4 + 5)) < 1e-4   # 결측 하나가 평균 3.25로

    onehot = datasets.resolve(spec, {"features": ["city", "age"]})
    assert onehot["shape"] == [3]   # city 원-핫 2 + age 1


def test_recipe_filters_classes_limits_samples_and_resizes_images(tmp_path):
    write_image_folder(tmp_path, side=20, per_class=3)
    spec = datasets.inspect(tmp_path / "shapes")
    recipe = {"classes": ["red"], "size": 8, "channels": 1, "limit": 2, "normalize": "fixed"}
    effective = datasets.resolve(spec, recipe)
    assert effective["shape"] == [1, 8, 8] and effective["class_names"] == ["red"]
    assert effective["count"] == 2 and effective["problem"]   # 클래스 하나로는 분류가 안 된다

    splits = datasets.load_any("shapes", tmp_path, {"size": 8, "channels": 1, "normalize": "fixed",
                                                    "val_fraction": 0.5})
    x, y = splits["train"]
    assert x.shape[1:] == (1, 8, 8) and len(x) == 3 and len(splits["test"][0]) == 3
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0

    shown = datasets.preview(spec, tmp_path, None)
    assert shown["preview"]["thumbnails"]["labels"] == ["blue", "red"]
    assert shown["spec"]["split"] == {"train": 5, "val": 1}


def test_builtin_preview_shows_samples_without_torch(tmp_path):
    """내장 데이터도 미리보기가 있다 - 클래스별 개수·분할·격자. hub 쪽이므로 torch 없이."""
    write_fake_mnist(tmp_path)
    shown = datasets.preview(datasets.describe("mnist", tmp_path), tmp_path, None)
    assert shown["spec"]["class_counts"]["0"] == 1 and shown["spec"]["class_counts"]["9"] == 0
    assert shown["spec"]["split"] == {"train": 8, "val": 8}
    assert shown["preview"]["thumbnails"]["labels"] == [str(label) for label in range(10)]
    assert shown["preview"]["thumbnails"]["per_class"] == 1


def write_fake_cifar(base: Path, count: int = 8) -> Path:
    """진짜와 같은 tar.gz + pickle 배치. 픽셀은 이미지마다 라벨값으로 채운다."""
    import io
    import pickle
    import tarfile

    target = base / "cifar10"
    target.mkdir(parents=True)
    archive = target / datasets.CATALOGUE["cifar10"]["files"]["archive"]
    with tarfile.open(archive, "w:gz") as tar:
        for member in datasets.CIFAR_TRAIN + datasets.CIFAR_TEST:
            labels = [index % 10 for index in range(count)]
            data = b"".join(bytes([label]) * 3072 for label in labels)
            blob = pickle.dumps({b"data": data, b"labels": labels})
            info = tarfile.TarInfo(member)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return target


def test_cifar10_loads_from_the_python_tarball(tmp_path):
    assert not datasets.available("cifar10", tmp_path)
    write_fake_cifar(tmp_path)
    assert datasets.available("cifar10", tmp_path)

    splits = datasets.load("cifar10", tmp_path)
    x, y = splits["train"]
    # 배치 5개가 각각 라벨 0..7을 담으므로 이어 붙이면 0..7이 다섯 번 반복된다.
    assert x.shape == (40, 3, 32, 32) and y.tolist() == [index % 8 for index in range(40)]
    assert splits["test"][0].shape == (8, 3, 32, 32)
    mean, std = datasets.CATALOGUE["cifar10"]["mean"], datasets.CATALOGUE["cifar10"]["std"]
    assert abs(float(x[1, 0, 0, 0]) - ((1 / 255 - mean[0]) / std[0])) < 1e-4

    spec = datasets.describe("cifar10", tmp_path)
    preview = datasets.preview(spec, tmp_path, None)
    assert preview["spec"]["class_names"][0] == "airplane"
    assert preview["spec"]["class_counts"]["airplane"] == 5
    assert preview["preview"]["thumbnails"]["labels"][:2] == ["airplane", "automobile"]


def test_progress_counts_finished_files_and_the_partial_one(tmp_path):
    target = tmp_path / "mnist"
    target.mkdir()
    files = datasets.CATALOGUE["mnist"]["files"]
    (target / files["train_images"]).write_bytes(b"x" * 100)
    (target / f'{files["train_labels"]}.part').write_bytes(b"x" * 30)

    got = datasets.progress("mnist", tmp_path)
    # 다 받은 파일과 받는 중인 .part를 함께 센다. 아직 시작 안 한 두 파일은 0이다.
    assert got["bytes"] == 130
    assert got["total"] == 11 * 1024 * 1024
    assert datasets.progress("mnist", tmp_path / "nowhere")["bytes"] == 0
