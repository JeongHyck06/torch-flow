"""논문용 아키텍처 다이어그램 SVG/PDF (기획서 §6.4).

캔버스와 같은 배치를 쓰되 인쇄용으로 다시 그린다: **흑백 안전**(색 없이 선과
글자만), 스타일 프리셋의 폭에 맞춰 통째로 스케일, 산출물 메타데이터에 IR 해시.
``--anonymous``는 도구 이름과 버전을 지운다(이중 블라인드 supplementary).

torch를 import하지 않는다 - hub 프로세스에서 그대로 돈다(§5.8).

장면(:class:`Figure`)을 한 번 만들고 두 백엔드가 같은 좌표를 쓴다. PDF는 사각형과
선과 글자뿐이라 base-14 폰트만으로 충분하다.
``ponytail: PDF 직접 생성(약 60줄). cairosvg는 네이티브 cairo를 끌고 오고
matplotlib은 곡선 export(§6.4 두 번째 항목) 전까지는 필요 없다.``
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ir import Graph, ModuleGraph, canonical_json, split_endpoint

# 스타일 프리셋(§6.4). 값은 본문 1단 폭(inch).
PRESETS = {"neurips": 5.5, "iclr": 5.5, "cvpr": 3.25}
DEFAULT_PRESET = "neurips"

# 좌표 단위는 포인트다. 캔버스 좌표(216 px 노드)를 그대로 축소하면 5.5 in 폭에서
# 라벨이 1 pt가 되어 못 읽는다 - 인쇄 배치는 처음부터 인쇄 치수로 잡고, 열 수는
# 프리셋 폭이 정한다. 그래서 layout.json은 쓰지 않는다(캔버스 배치 != 지면 배치).
NODE_W, NODE_H = 84.0, 30.0
GAP_X, GAP_Y = 16.0, 14.0
MARGIN = 4.0
LABEL_PT, DETAIL_PT, FOOTER_PT = 6.5, 5.5, 5.0


@dataclass
class Box:
    x: float
    y: float
    label: str
    sublabel: str = ""
    note: str = ""


@dataclass
class Figure:
    boxes: list[Box] = field(default_factory=list)
    edges: list[tuple[float, float, float, float]] = field(default_factory=list)
    caption: str = ""
    ir_hash: str = ""
    generator: str | None = None      # 익명 export에서는 None
    width: float = 0.0                # pt
    height: float = 0.0

    @property
    def page(self) -> tuple[float, float]:
        return self.width, self.height


def build(
    ir: ModuleGraph,
    *,
    node_states: dict[str, Any] | None = None,
    preset: str = DEFAULT_PRESET,
    anonymous: bool = False,
    version: str | None = None,
) -> Figure:
    """IR + shape 상태 -> 인쇄용 장면."""
    graph = ir.graph
    states = node_states or {}
    order = flow_order(graph)
    columns = max(1, int((PRESETS.get(preset, PRESETS[DEFAULT_PRESET]) * 72 - 2 * MARGIN + GAP_X)
                         // (NODE_W + GAP_X)))

    boxes: dict[str, Box] = {}
    for index, node in enumerate(order):
        spec = (states.get(node.id) or {}).get("spec") or {}
        instance = graph.instances.get(node.call) if node.call else None
        kind = (node.type or (instance.type if instance else "") or "").split("@")[0]
        boxes[node.id] = Box(
            x=MARGIN + (index % columns) * (NODE_W + GAP_X),
            y=MARGIN + (index // columns) * (NODE_H + GAP_Y),
            label=node.label,
            sublabel=_short(kind),
            note=_clip(_shape_text(spec.get("shape"))),
        )

    edges = []
    for src, dst in graph.edges:
        source, _ = split_endpoint(src)
        target, _ = split_endpoint(dst)
        if source in boxes and target in boxes:
            edges.append(_route(boxes[source], boxes[target]))

    rows = (len(order) + columns - 1) // columns
    figure = Figure(
        boxes=list(boxes.values()), edges=edges,
        caption=graph.name,
        ir_hash=hashlib.sha256(canonical_json(ir).encode()).hexdigest()[:16],
        generator=None if anonymous else f"TorchFlow {version or ''}".strip(),
        width=2 * MARGIN + columns * NODE_W + (columns - 1) * GAP_X,
        height=2 * MARGIN + rows * NODE_H + (rows - 1) * GAP_Y + 10,
    )
    return figure


def flow_order(graph: Graph) -> list:
    """위상 순. 층이 같으면 선언 순 - 캔버스의 키보드 탐색 순서와 같다(§2.2)."""
    ids = {node.id for node in graph.nodes}
    indegree = {node.id: 0 for node in graph.nodes}
    successors: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for src, dst in graph.edges:
        source, _ = split_endpoint(src)
        target, _ = split_endpoint(dst)
        if source in ids and target in ids:
            successors[source].append(target)
            indegree[target] += 1

    depth = {node_id: 0 for node_id, count in indegree.items() if count == 0}
    queue = list(depth)
    while queue:
        current = queue.pop(0)
        for following in successors[current]:
            depth[following] = max(depth.get(following, 0), depth[current] + 1)
            indegree[following] -= 1
            if indegree[following] == 0:
                queue.append(following)
    return sorted(graph.nodes, key=lambda node: depth.get(node.id, 0))


def _route(tail: Box, head: Box) -> tuple[float, float, float, float]:
    """줄 안이면 오른쪽 -> 왼쪽, 줄이 바뀌면 아래 -> 위로 잇는다."""
    if head.y > tail.y:
        return (tail.x + NODE_W / 2, tail.y + NODE_H, head.x + NODE_W / 2, head.y)
    return (tail.x + NODE_W, tail.y + NODE_H / 2, head.x, head.y + NODE_H / 2)


def _elbow(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    """직교 꺾임 경로. 줄이 바뀌는 엣지는 상자 사이 여백으로 돌아 내려간다."""
    if y1 > y0:   # 다음 줄로 넘어간다
        mid = (y0 + y1) / 2
        return [(x0, y0), (x0, mid), (x1, mid), (x1, y1)]
    mid = (x0 + x1) / 2
    return [(x0, y0), (mid, y0), (mid, y1), (x1, y1)]


def _arrow_head(x0: float, y0: float, x1: float, y1: float,
                size: float = 3.0) -> list[tuple[float, float]]:
    """마지막 구간의 방향으로 삼각형을 세운다."""
    if y1 > y0:   # 아래로 들어온다
        return [(x1, y1), (x1 - size / 1.6, y1 - size), (x1 + size / 1.6, y1 - size)]
    return [(x1, y1), (x1 - size, y1 - size / 1.6), (x1 - size, y1 + size / 1.6)]


def _shape_text(shape: list[Any] | None) -> str:
    return f"[{', '.join(str(dim) for dim in shape)}]" if shape else ""


def _short(kind: str) -> str:
    """지면에서는 네임스페이스가 자리만 먹는다. ``torch.nn.Conv2d`` -> ``Conv2d``."""
    for prefix in ("torchflow.", "composite:", "torch.nn.functional.", "torch.nn.", "torch."):
        kind = kind.removeprefix(prefix)
    return _clip(kind)


def _clip(text: str, width: float = NODE_W - 8) -> str:
    """Courier 5.5 pt는 글자당 0.6 em이다. 넘치면 잘라서 상자 밖으로 안 나가게 한다."""
    limit = int(width / (DETAIL_PT * 0.6))
    return text if len(text) <= limit else text[:limit - 2] + ".."


# SVG


def to_svg(figure: Figure) -> str:
    width_pt, height_pt = figure.page
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_pt:.1f}pt" '
        f'height="{height_pt:.1f}pt" viewBox="0 0 {figure.width:.1f} {figure.height:.1f}">',
        f"<desc>ir_sha256={figure.ir_hash}"
        f"{'' if figure.generator is None else ' generator=' + figure.generator}</desc>",
        '<g fill="none" stroke="#000" stroke-width="0.5">',
    ]
    for edge in figure.edges:
        points = _elbow(*edge)
        parts.append('<path d="M' + " L".join(f"{x:.1f} {y:.1f}" for x, y in points) + '"/>')
        parts.append('<path d="M' + " L".join(f"{x:.1f} {y:.1f}"
                                              for x, y in _arrow_head(*edge)) + ' z" fill="#000"/>')
    for box in figure.boxes:
        parts.append(
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="2" fill="#fff"/>')
        # 캔버스의 헤더 스트립 - 인쇄에서는 색 대신 얇은 구분선이다.
        parts.append(f'<path d="M{box.x:.1f} {box.y + 10:.1f} h{NODE_W}"/>')
    parts.append("</g>")
    for box in figure.boxes:
        parts.append(_svg_text(box.x + 4, box.y + 7.5, box.label, LABEL_PT, weight="600"))
        if box.sublabel:
            parts.append(_svg_text(box.x + 4, box.y + 18, box.sublabel, DETAIL_PT, mono=True))
        if box.note:
            parts.append(_svg_text(box.x + 4, box.y + 26, box.note, DETAIL_PT, mono=True))
    parts.append(_svg_text(MARGIN, figure.height - 2, _footer(figure), FOOTER_PT, mono=True))
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_text(x: float, y: float, text: str, size: float, *, weight: str = "400",
              mono: bool = False) -> str:
    family = "IBM Plex Mono, monospace" if mono else "IBM Plex Sans KR, sans-serif"
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="#000">{_escape(text)}</text>')


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _footer(figure: Figure) -> str:
    """산출물에서 원본 그래프로 돌아가는 길(§6.4 마지막 항목)."""
    parts = [f"ir {figure.ir_hash}"]
    if figure.generator:
        parts.append(figure.generator)
    return "  ·  ".join(parts)


# PDF


def to_pdf(figure: Figure) -> bytes:
    width_pt, height_pt = figure.page
    # PDF의 y축은 위로 자란다. 장면을 뒤집어 좌표를 그대로 쓴다.
    ops = [f"1 0 0 -1 0 {height_pt:.2f} cm", "0 G", "0 g", "0.5 w"]

    for edge in figure.edges:
        points = _elbow(*edge)
        ops.append(" ".join([f"{points[0][0]:.1f} {points[0][1]:.1f} m"]
                            + [f"{x:.1f} {y:.1f} l" for x, y in points[1:]] + ["S"]))
        head = _arrow_head(*edge)
        ops.append(" ".join([f"{head[0][0]:.1f} {head[0][1]:.1f} m"]
                            + [f"{x:.1f} {y:.1f} l" for x, y in head[1:]] + ["f"]))
    for box in figure.boxes:
        ops.append(f"1 g {box.x:.1f} {box.y:.1f} {NODE_W} {NODE_H} re f 0 g")
        ops.append(f"{box.x:.1f} {box.y:.1f} {NODE_W} {NODE_H} re S")
        ops.append(f"{box.x:.1f} {box.y + 10:.1f} m {box.x + NODE_W:.1f} {box.y + 10:.1f} l S")
        ops.append(_pdf_text(box.x + 4, box.y + 7.5, box.label, LABEL_PT, "F2"))
        if box.sublabel:
            ops.append(_pdf_text(box.x + 4, box.y + 18, box.sublabel, DETAIL_PT, "F3"))
        if box.note:
            ops.append(_pdf_text(box.x + 4, box.y + 26, box.note, DETAIL_PT, "F3"))
    ops.append(_pdf_text(MARGIN, figure.height - 2, _footer(figure), FOOTER_PT, "F3"))

    stream = "\n".join(ops).encode("latin-1", "replace")
    info = {"Subject": f"ir_sha256={figure.ir_hash}", "Title": figure.caption}
    if figure.generator:
        info["Creator"] = figure.generator
        info["CreationDate"] = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")
    return _pdf_document(width_pt, height_pt, stream, info)


def _pdf_text(x: float, y: float, text: str, size: float, font: str) -> str:
    # 텍스트는 뒤집힌 좌표계 안에서 다시 뒤집어야 바로 선다.
    body = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return (f"BT /{font} {size} Tf 1 0 0 -1 {x:.1f} {y:.1f} Tm ({body}) Tj ET")


def _pdf_document(width: float, height: float, stream: bytes, info: dict[str, str]) -> bytes:
    """객체 7개짜리 최소 PDF. 폰트는 base-14라 임베딩이 필요 없다."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
        f"/Resources << /Font << /F2 5 0 R /F3 6 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
        ("<< " + " ".join(f"/{key} ({_pdf_string(value)})" for key, value in info.items())
         + " >>").encode("latin-1", "replace"),
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {len(objects)} 0 R >>\n"
            f"startxref\n{start}\n%%EOF\n").encode()
    return bytes(out)


def _pdf_string(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
