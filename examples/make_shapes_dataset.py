"""'내 데이터' 예제: 도형 그림 세 종류를 data/shapes/<클래스>/*.png 로 만든다.

클래스별 하위 폴더에 이미지를 두는 흔한 구조 그대로다 - 첫 화면의 "내 데이터"에
shapes가 뜨고, 그걸로 새 그래프를 열면 Input이 [B, 3, 64, 64]로 깔린다.

    python examples/make_shapes_dataset.py            # data/shapes/ 에 600장
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

SIDE = 64
PER_CLASS = 200


def draw(kind: str, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (SIDE, SIDE), tuple(rng.randint(200, 255) for _ in range(3)))
    pen = ImageDraw.Draw(image)
    size = rng.randint(18, 40)
    x = rng.randint(4, SIDE - size - 4)
    y = rng.randint(4, SIDE - size - 4)
    color = tuple(rng.randint(0, 160) for _ in range(3))
    if kind == "circle":
        pen.ellipse([x, y, x + size, y + size], fill=color)
    elif kind == "square":
        pen.rectangle([x, y, x + size, y + size], fill=color)
    else:
        pen.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)], fill=color)
    return image


def main(root: Path = Path("data") / "shapes") -> None:
    rng = random.Random(0)
    for kind in ("circle", "square", "triangle"):
        folder = root / kind
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(PER_CLASS):
            draw(kind, rng).save(folder / f"{index:03d}.png")
    print(f"wrote {3 * PER_CLASS} images under {root}")


if __name__ == "__main__":
    main()
