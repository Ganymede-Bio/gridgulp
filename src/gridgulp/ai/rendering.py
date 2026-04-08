"""Normalized sheet rendering utilities for multimodal Gemma reasoning."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from gridgulp.models.sheet_data import CellData, SheetData
from gridgulp.models.table import TableInfo, TableRange

_HIGHLIGHT_PALETTE = (
    "#D7263D",
    "#1B998B",
    "#F46036",
    "#2E294E",
    "#C5D86D",
)


@dataclass(slots=True)
class RenderedPromptImage:
    """Image artifact prepared for multimodal prompts."""

    label: str
    kind: str
    png_bytes: bytes
    width: int
    height: int
    excel_range: str | None = None
    debug_name: str | None = None

    @property
    def data_uri(self) -> str:
        """Encode the PNG payload as a data URI."""
        encoded = base64.b64encode(self.png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"


class SheetImageRenderer:
    """Render SheetData into normalized PNG images for Gemma vision fallback."""

    def __init__(
        self,
        max_proposal_crops: int = 4,
        crop_margin_cells: int = 1,
    ) -> None:
        self.max_proposal_crops = max_proposal_crops
        self.crop_margin_cells = crop_margin_cells
        self.font = ImageFont.load_default()

    def render_bundle(
        self,
        sheet_data: SheetData,
        proposals: list[TableInfo],
    ) -> list[RenderedPromptImage]:
        """Render the overview image plus proposal crops."""
        selected_proposals = sorted(
            proposals,
            key=lambda proposal: (
                -proposal.confidence,
                proposal.range.start_row,
                proposal.range.start_col,
            ),
        )[: self.max_proposal_crops]

        images = [
            self._render_region(
                sheet_data=sheet_data,
                region=TableRange(
                    start_row=0,
                    start_col=0,
                    end_row=max(sheet_data.max_row, 0),
                    end_col=max(sheet_data.max_column, 0),
                ),
                label=f"{sheet_data.name} overview",
                kind="overview",
                highlight_ranges=[
                    (
                        proposal.range,
                        _HIGHLIGHT_PALETTE[index % len(_HIGHLIGHT_PALETTE)],
                        proposal.id,
                    )
                    for index, proposal in enumerate(selected_proposals)
                ],
                max_pixels=(1800, 1400),
                debug_name=f"{self._sanitize_filename(sheet_data.name)}_overview.png",
            )
        ]

        for index, proposal in enumerate(selected_proposals, start=1):
            crop_region = self._expand_range(
                proposal.range,
                sheet_data,
                margin=self.crop_margin_cells,
            )
            images.append(
                self._render_region(
                    sheet_data=sheet_data,
                    region=crop_region,
                    label=f"{sheet_data.name} crop {index}",
                    kind="proposal_crop",
                    highlight_ranges=[
                        (
                            proposal.range,
                            _HIGHLIGHT_PALETTE[(index - 1) % len(_HIGHLIGHT_PALETTE)],
                            proposal.id,
                        )
                    ],
                    max_pixels=(1400, 1000),
                    debug_name=(
                        f"{self._sanitize_filename(sheet_data.name)}_"
                        f"crop_{index}_{proposal.range.excel_range.replace(':', '_')}.png"
                    ),
                    excel_range=proposal.range.excel_range,
                )
            )

        return images

    def save_bundle(self, images: list[RenderedPromptImage], output_dir: Path) -> list[str]:
        """Persist rendered images for debugging."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for image in images:
            filename = image.debug_name or f"{image.kind}.png"
            output_path = output_dir / filename
            output_path.write_bytes(image.png_bytes)
            saved_paths.append(str(output_path))
        return saved_paths

    def _render_region(
        self,
        *,
        sheet_data: SheetData,
        region: TableRange,
        label: str,
        kind: str,
        highlight_ranges: list[tuple[TableRange, str, str]],
        max_pixels: tuple[int, int],
        debug_name: str,
        excel_range: str | None = None,
    ) -> RenderedPromptImage:
        region = self._clip_range(region, sheet_data)
        rows = max(region.row_count, 1)
        cols = max(region.col_count, 1)

        left_gutter = 44
        top_gutter = 28
        cell_width = self._choose_cell_size(
            cols, max_pixels[0] - left_gutter, preferred=110, minimum=24
        )
        cell_height = self._choose_cell_size(
            rows, max_pixels[1] - top_gutter, preferred=30, minimum=16
        )

        width = left_gutter + cols * cell_width + 1
        height = top_gutter + rows * cell_height + 1

        image = Image.new("RGB", (width, height), "#FFFFFF")
        draw = ImageDraw.Draw(image)

        draw.rectangle((0, 0, width, top_gutter), fill="#F2F4F7")
        draw.rectangle((0, 0, left_gutter, height), fill="#F2F4F7")
        draw.text((6, 7), self._truncate(label, 36), fill="#111827", font=self.font)

        self._draw_headers(draw, region, left_gutter, top_gutter, cell_width, cell_height)
        self._draw_cells(draw, sheet_data, region, left_gutter, top_gutter, cell_width, cell_height)
        self._draw_merged_outlines(
            draw,
            sheet_data,
            region,
            left_gutter,
            top_gutter,
            cell_width,
            cell_height,
        )
        self._draw_highlights(
            draw,
            region,
            highlight_ranges,
            left_gutter,
            top_gutter,
            cell_width,
            cell_height,
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return RenderedPromptImage(
            label=label,
            kind=kind,
            png_bytes=buffer.getvalue(),
            width=width,
            height=height,
            excel_range=excel_range or region.excel_range,
            debug_name=debug_name,
        )

    def _draw_headers(
        self,
        draw: ImageDraw.ImageDraw,
        region: TableRange,
        left_gutter: int,
        top_gutter: int,
        cell_width: int,
        cell_height: int,
    ) -> None:
        for col in range(region.start_col, region.end_col + 1):
            x0 = left_gutter + (col - region.start_col) * cell_width
            x1 = x0 + cell_width
            draw.rectangle((x0, 0, x1, top_gutter), outline="#CBD5E1", fill="#F8FAFC")
            label = self._column_letter(col)
            draw.text((x0 + 4, 7), label, fill="#334155", font=self.font)

        for row in range(region.start_row, region.end_row + 1):
            y0 = top_gutter + (row - region.start_row) * cell_height
            y1 = y0 + cell_height
            draw.rectangle((0, y0, left_gutter, y1), outline="#CBD5E1", fill="#F8FAFC")
            draw.text((6, y0 + 4), str(row + 1), fill="#334155", font=self.font)

    def _draw_cells(
        self,
        draw: ImageDraw.ImageDraw,
        sheet_data: SheetData,
        region: TableRange,
        left_gutter: int,
        top_gutter: int,
        cell_width: int,
        cell_height: int,
    ) -> None:
        render_text = cell_width >= 38 and cell_height >= 18

        for row in range(region.start_row, region.end_row + 1):
            for col in range(region.start_col, region.end_col + 1):
                x0 = left_gutter + (col - region.start_col) * cell_width
                y0 = top_gutter + (row - region.start_row) * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                cell = sheet_data.get_cell(row, col)

                fill_color = self._cell_fill_color(cell)
                draw.rectangle((x0, y0, x1, y1), fill=fill_color, outline="#D0D7DE")

                if cell is not None:
                    self._draw_cell_borders(draw, cell, x0, y0, x1, y1)
                    if render_text and not cell.is_empty:
                        self._draw_cell_text(draw, cell, x0, y0, x1, y1)

    def _draw_merged_outlines(
        self,
        draw: ImageDraw.ImageDraw,
        sheet_data: SheetData,
        region: TableRange,
        left_gutter: int,
        top_gutter: int,
        cell_width: int,
        cell_height: int,
    ) -> None:
        seen_ranges: set[str] = set()
        for cell in sheet_data.cells.values():
            if not cell.is_merged or not cell.merge_range or cell.merge_range in seen_ranges:
                continue
            seen_ranges.add(cell.merge_range)
            try:
                merged_range = TableRange.from_excel(cell.merge_range)
            except ValueError:
                continue

            clipped = self._intersect_ranges(region, merged_range)
            if clipped is None:
                continue

            x0 = left_gutter + (clipped.start_col - region.start_col) * cell_width
            y0 = top_gutter + (clipped.start_row - region.start_row) * cell_height
            x1 = left_gutter + (clipped.end_col - region.start_col + 1) * cell_width
            y1 = top_gutter + (clipped.end_row - region.start_row + 1) * cell_height
            draw.rectangle((x0, y0, x1, y1), outline="#6D28D9", width=2)

    def _draw_highlights(
        self,
        draw: ImageDraw.ImageDraw,
        region: TableRange,
        highlight_ranges: list[tuple[TableRange, str, str]],
        left_gutter: int,
        top_gutter: int,
        cell_width: int,
        cell_height: int,
    ) -> None:
        for table_range, color, label in highlight_ranges:
            clipped = self._intersect_ranges(region, table_range)
            if clipped is None:
                continue
            x0 = left_gutter + (clipped.start_col - region.start_col) * cell_width
            y0 = top_gutter + (clipped.start_row - region.start_row) * cell_height
            x1 = left_gutter + (clipped.end_col - region.start_col + 1) * cell_width
            y1 = top_gutter + (clipped.end_row - region.start_row + 1) * cell_height
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            tag = self._truncate(label, 16)
            draw.rectangle(
                (x0, max(top_gutter, y0 - 14), x0 + 8 + len(tag) * 6, max(top_gutter, y0 - 1)),
                fill=color,
            )
            draw.text((x0 + 4, max(top_gutter, y0 - 13)), tag, fill="#FFFFFF", font=self.font)

    def _draw_cell_text(
        self,
        draw: ImageDraw.ImageDraw,
        cell: CellData,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> None:
        raw_value = cell.formatted_value if cell.formatted_value is not None else cell.value
        text = self._truncate(
            str(raw_value).replace("\n", " / ").strip(), max(4, (x1 - x0 - 8) // 6)
        )
        if not text:
            return

        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 4
        if cell.alignment == "right":
            text_x = max(x0 + padding, x1 - text_width - padding)
        elif cell.alignment == "center":
            text_x = x0 + max((x1 - x0 - text_width) // 2, padding)
        else:
            text_x = x0 + padding
        text_y = y0 + max((y1 - y0 - text_height) // 2, 2)

        fill = self._cell_text_color(cell)
        if cell.is_bold:
            draw.text((text_x + 1, text_y), text, fill=fill, font=self.font)
        draw.text((text_x, text_y), text, fill=fill, font=self.font)

    def _draw_cell_borders(
        self,
        draw: ImageDraw.ImageDraw,
        cell: CellData,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> None:
        border_widths = {
            "thin": 1,
            "hair": 1,
            "dashed": 1,
            "dotted": 1,
            "medium": 2,
            "double": 2,
            "thick": 3,
        }
        color = "#475569"

        if cell.border_top:
            draw.line((x0, y0, x1, y0), fill=color, width=border_widths.get(cell.border_top, 1))
        if cell.border_bottom:
            draw.line((x0, y1, x1, y1), fill=color, width=border_widths.get(cell.border_bottom, 1))
        if cell.border_left:
            draw.line((x0, y0, x0, y1), fill=color, width=border_widths.get(cell.border_left, 1))
        if cell.border_right:
            draw.line((x1, y0, x1, y1), fill=color, width=border_widths.get(cell.border_right, 1))

    def _cell_fill_color(self, cell: CellData | None) -> str:
        if cell is None or not cell.background_color:
            return "#FFFFFF"
        return self._normalize_hex_color(cell.background_color, fallback="#FFFFFF")

    def _cell_text_color(self, cell: CellData) -> str:
        if cell.font_color:
            return self._normalize_hex_color(cell.font_color, fallback="#111827")
        return "#111827"

    def _clip_range(self, table_range: TableRange, sheet_data: SheetData) -> TableRange:
        return TableRange(
            start_row=max(0, min(table_range.start_row, sheet_data.max_row)),
            start_col=max(0, min(table_range.start_col, sheet_data.max_column)),
            end_row=max(0, min(table_range.end_row, sheet_data.max_row)),
            end_col=max(0, min(table_range.end_col, sheet_data.max_column)),
        )

    def _expand_range(
        self, table_range: TableRange, sheet_data: SheetData, margin: int
    ) -> TableRange:
        return TableRange(
            start_row=max(0, table_range.start_row - margin),
            start_col=max(0, table_range.start_col - margin),
            end_row=min(sheet_data.max_row, table_range.end_row + margin),
            end_col=min(sheet_data.max_column, table_range.end_col + margin),
        )

    def _intersect_ranges(self, left: TableRange, right: TableRange) -> TableRange | None:
        start_row = max(left.start_row, right.start_row)
        start_col = max(left.start_col, right.start_col)
        end_row = min(left.end_row, right.end_row)
        end_col = min(left.end_col, right.end_col)
        if end_row < start_row or end_col < start_col:
            return None
        return TableRange(
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
        )

    def _choose_cell_size(
        self, count: int, max_pixels: int, *, preferred: int, minimum: int
    ) -> int:
        if count <= 0:
            return preferred
        return max(minimum, min(preferred, max_pixels // count if max_pixels > 0 else minimum))

    def _normalize_hex_color(self, value: str, *, fallback: str) -> str:
        stripped = value.strip().lstrip("#")
        if len(stripped) == 8:
            stripped = stripped[-6:]
        if len(stripped) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", stripped):
            return fallback
        try:
            ImageColor.getrgb(f"#{stripped}")
        except ValueError:
            return fallback
        return f"#{stripped.upper()}"

    def _column_letter(self, column: int) -> str:
        result = ""
        index = column
        while index >= 0:
            result = chr(index % 26 + ord("A")) + result
            index = index // 26 - 1
        return result

    def _truncate(self, text: str, limit: int) -> str:
        if limit <= 1:
            return text[:limit]
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}…"

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "sheet"
