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
    assert listed == ["mnist", "arrays"]
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
