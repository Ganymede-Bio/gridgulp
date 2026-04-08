"""Tests for normalized sheet rendering used by multimodal fallback."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from gridgulp.ai.rendering import SheetImageRenderer
from gridgulp.models.sheet_data import CellData, SheetData
from gridgulp.models.table import TableInfo, TableRange


def _build_render_sheet() -> tuple[SheetData, list[TableInfo]]:
    sheet = SheetData(name="RenderedSheet")
    values = {
        "A1": CellData(value="Region", data_type="text", is_bold=True, row=0, column=0),
        "B1": CellData(value="Sales", data_type="text", is_bold=True, row=0, column=1),
        "A2": CellData(value="West", data_type="text", row=1, column=0),
        "B2": CellData(value=10, data_type="number", row=1, column=1),
        "D1": CellData(value="Owner", data_type="text", is_bold=True, row=0, column=3),
        "E1": CellData(value="Notes", data_type="text", is_bold=True, row=0, column=4),
        "D2": CellData(value="Ops", data_type="text", row=1, column=3),
        "E2": CellData(value="Follow up", data_type="text", row=1, column=4),
    }
    sheet.cells.update(values)
    sheet.max_row = 1
    sheet.max_column = 4

    proposals = [
        TableInfo(
            id="sales_table",
            range=TableRange.from_excel("A1:B2"),
            confidence=0.9,
            detection_method="simple_case",
        ),
        TableInfo(
            id="notes_table",
            range=TableRange.from_excel("D1:E2"),
            confidence=0.8,
            detection_method="simple_case",
        ),
    ]
    return sheet, proposals


def test_sheet_image_renderer_emits_overview_and_crops() -> None:
    """Renderer should emit one overview plus proposal crops."""
    sheet, proposals = _build_render_sheet()
    renderer = SheetImageRenderer(max_proposal_crops=2, crop_margin_cells=1)

    images = renderer.render_bundle(sheet, proposals)

    assert [image.kind for image in images] == ["overview", "proposal_crop", "proposal_crop"]
    assert all(image.png_bytes.startswith(b"\x89PNG") for image in images)

    overview = Image.open(BytesIO(images[0].png_bytes))
    crop = Image.open(BytesIO(images[1].png_bytes))
    assert overview.size[0] > crop.size[0]
    assert overview.size[1] >= crop.size[1]


def test_sheet_image_renderer_saves_debug_images(tmp_path: Path) -> None:
    """Renderer should persist PNGs when debug output is requested."""
    sheet, proposals = _build_render_sheet()
    renderer = SheetImageRenderer(max_proposal_crops=1, crop_margin_cells=1)

    images = renderer.render_bundle(sheet, proposals)
    saved_paths = renderer.save_bundle(images, tmp_path)

    assert len(saved_paths) == 2
    assert all(Path(path).exists() for path in saved_paths)
