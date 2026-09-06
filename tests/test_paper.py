"""논문용 그림 내보내기 (기획서 §6.4)."""

from conftest import MINIVIT
from torchflow import paper
from torchflow.ir import load


def figure(**kwargs):
    return paper.build(load(MINIVIT), version="0.0.1", **kwargs)


def test_svg_carries_every_node_and_the_ir_hash():
    fig = figure()
    svg = paper.to_svg(fig)

    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    for label in ("x", "patch", "blocks", "pool", "head", "logits"):
        assert f">{label}<" in svg
    assert f"ir_sha256={fig.ir_hash}" in svg
    assert "TorchFlow 0.0.1" in svg


def test_anonymous_export_drops_the_tool_name():
    """이중 블라인드 supplementary - 도구 이름과 버전이 저자를 드러낸다(§6.4)."""
    fig = figure(anonymous=True)
    svg = paper.to_svg(fig)
    pdf = paper.to_pdf(fig)

    assert "TorchFlow" not in svg and b"TorchFlow" not in pdf
    # 해시는 남는다 - 그림에서 그래프로 돌아가는 길이다.
    assert fig.ir_hash in svg and fig.ir_hash.encode() in pdf


def test_pdf_is_well_formed_and_points_at_its_xref():
    pdf = paper.to_pdf(figure())

    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    start = int(pdf.rsplit(b"startxref", 1)[1].split(b"%%EOF")[0].strip())
    assert pdf[start:start + 4] == b"xref"
    assert pdf.count(b" obj") == 7


def test_preset_decides_the_column_count():
    """좁은 단은 열이 줄고 대신 길어진다 - 폭은 프리셋이 정한다."""
    wide = figure(preset="neurips")
    narrow = figure(preset="cvpr")

    assert wide.width > narrow.width
    assert wide.width <= paper.PRESETS["neurips"] * 72
    assert narrow.width <= paper.PRESETS["cvpr"] * 72
    assert narrow.height >= wide.height


def test_long_type_names_stay_inside_the_box():
    limit = int((paper.NODE_W - 8) / (paper.DETAIL_PT * 0.6))
    assert paper._short("torch.nn.AdaptiveAvgPool2d") == "AdaptiveAvgPool2d"
    assert len(paper._short("torch.nn.SomeVeryLongModuleNameIndeed")) <= limit


def test_shapes_come_from_node_states():
    fig = figure(node_states={"01J9Q4B4": {"spec": {"shape": ["B", 192]}}})
    assert any(box.note == "[B, 192]" for box in fig.boxes)
