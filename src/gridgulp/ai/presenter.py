"""Structured text presentation for Gemma-first reasoning."""

from __future__ import annotations

from typing import Any

from gridgulp.detectors.format_analyzer import SemanticFormatAnalyzer
from gridgulp.detectors.merged_cell_analyzer import MergedCellAnalyzer
from gridgulp.detectors.multi_header_detector import MultiHeaderDetector
from gridgulp.models.sheet_data import SheetData
from gridgulp.models.table import TableInfo, TableRange


class StructuredTextPresenter:
    """Render sheet and proposal context into a compact text prompt."""

    def __init__(self, max_rows: int = 18, max_cols: int = 10) -> None:
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.merged_cell_analyzer = MergedCellAnalyzer()
        self.multi_header_detector = MultiHeaderDetector()
        self.semantic_analyzer = SemanticFormatAnalyzer()

    def build_context(self, sheet_data: SheetData, proposals: list[TableInfo]) -> dict[str, Any]:
        """Build structured context for prompt rendering."""
        return {
            "sheet": self._sheet_summary(sheet_data),
            "density_map": self._density_map(sheet_data),
            "row_samples": self._row_samples(sheet_data),
            "proposals": [self._proposal_summary(sheet_data, proposal) for proposal in proposals],
        }

    def format_for_prompt(self, context: dict[str, Any]) -> str:
        """Format structured context into a concise prompt section."""
        sheet = context["sheet"]
        lines = [
            "## Sheet Summary",
            f"- Name: {sheet['name']}",
            f"- Dimensions: {sheet['rows']} rows x {sheet['cols']} cols",
            f"- Non-empty cells: {sheet['non_empty_cells']}",
            f"- Density: {sheet['density']:.3f}",
            "",
            "## Density Map",
            context["density_map"],
            "",
            "## Row Samples",
        ]

        lines.extend(f"- {sample}" for sample in context["row_samples"])

        lines.append("")
        lines.append("## Heuristic Proposals")
        if not context["proposals"]:
            lines.append("- None")
        else:
            for proposal in context["proposals"]:
                lines.append(
                    "- "
                    f"{proposal['id']} | range={proposal['range']} | confidence={proposal['confidence']:.2f} "
                    f"| method={proposal['detection_method']} | headers={proposal['headers']}"
                )
                lines.append(f"  context={proposal['neighbor_rows']}")
                lines.append(f"  features={proposal['features']}")

        return "\n".join(lines)

    def _sheet_summary(self, sheet_data: SheetData) -> dict[str, Any]:
        rows = sheet_data.max_row + 1 if sheet_data.has_data() else 0
        cols = sheet_data.max_column + 1 if sheet_data.has_data() else 0
        non_empty_cells = len(sheet_data.get_non_empty_cells())
        area = rows * cols if rows and cols else 0
        density = non_empty_cells / area if area else 0.0
        return {
            "name": sheet_data.name,
            "rows": rows,
            "cols": cols,
            "non_empty_cells": non_empty_cells,
            "density": density,
        }

    def _density_map(self, sheet_data: SheetData) -> str:
        rows = min(sheet_data.max_row + 1, self.max_rows)
        cols = min(sheet_data.max_column + 1, self.max_cols)
        if rows == 0 or cols == 0:
            return "(empty sheet)"

        lines = []
        for row in range(rows):
            row_markers = []
            for col in range(cols):
                cell = sheet_data.get_cell(row, col)
                row_markers.append(self._cell_marker(cell))
            lines.append(f"r{row + 1:02d}: {''.join(row_markers)}")

        if sheet_data.max_row + 1 > rows:
            lines.append("...")
        return "\n".join(lines)

    def _row_samples(self, sheet_data: SheetData) -> list[str]:
        samples = []
        max_rows = min(sheet_data.max_row + 1, self.max_rows)
        for row in range(max_rows):
            cells = []
            for col in range(min(sheet_data.max_column + 1, self.max_cols)):
                cell = sheet_data.get_cell(row, col)
                if cell is None or cell.value is None:
                    continue
                cells.append(f"{cell.excel_address}={self._stringify_value(cell.value)}")
            if cells:
                samples.append(" | ".join(cells[:6]))
        return samples[:8]

    def _proposal_summary(self, sheet_data: SheetData, proposal: TableInfo) -> dict[str, Any]:
        multi_header = None
        semantic_summary: dict[str, Any] | None = None

        try:
            multi_header_result = self.multi_header_detector.detect_multi_row_headers(
                sheet_data,
                proposal.range,
            )
            if multi_header_result is not None:
                multi_header = {
                    "row_count": multi_header_result.end_row - multi_header_result.start_row + 1,
                    "column_mappings": multi_header_result.column_mappings,
                }
        except Exception:
            multi_header = None

        try:
            structure = self.semantic_analyzer.analyze_table_structure(
                sheet_data,
                proposal.range,
                header_rows=1,
            )
            semantic_summary = {
                "sections": len(structure.sections),
                "has_subtotals": structure.has_subtotals,
                "has_grand_total": structure.has_grand_total,
                "preserve_blank_rows": structure.preserve_blank_rows[:5],
            }
        except Exception:
            semantic_summary = None

        merged_cells = self.merged_cell_analyzer.analyze_merged_cells(sheet_data, proposal.range)
        header_merged = self.merged_cell_analyzer.find_header_merged_cells(merged_cells)

        return {
            "id": proposal.id,
            "range": proposal.range.excel_range,
            "confidence": proposal.confidence,
            "detection_method": proposal.detection_method,
            "headers": proposal.headers[:6] if proposal.headers else [],
            "neighbor_rows": self._neighbor_rows(sheet_data, proposal.range),
            "features": {
                "multi_header": multi_header,
                "semantic_structure": semantic_summary,
                "merged_cell_count": len(merged_cells),
                "header_merged_count": len(header_merged),
            },
        }

    def _neighbor_rows(
        self, sheet_data: SheetData, table_range: TableRange
    ) -> dict[str, list[str]]:
        ranges = {
            "above": [table_range.start_row - 1] if table_range.start_row > 0 else [],
            "inside": list(
                range(
                    table_range.start_row,
                    min(table_range.end_row + 1, table_range.start_row + 2),
                )
            ),
            "below": [table_range.end_row + 1] if table_range.end_row < sheet_data.max_row else [],
        }
        rendered: dict[str, list[str]] = {}
        for key, rows in ranges.items():
            rendered[key] = [self._render_row(sheet_data, row, table_range) for row in rows]
        return rendered

    def _render_row(self, sheet_data: SheetData, row: int, table_range: TableRange) -> str:
        parts = []
        for col in range(
            table_range.start_col, min(table_range.end_col + 1, table_range.start_col + 6)
        ):
            cell = sheet_data.get_cell(row, col)
            if cell is None or cell.value is None:
                continue
            parts.append(f"{cell.excel_address}={self._stringify_value(cell.value)}")
        return " | ".join(parts) if parts else "(empty)"

    def _cell_marker(self, cell: Any) -> str:
        if cell is None or cell.value is None:
            return "."
        if getattr(cell, "has_formula", False):
            return "F"
        if isinstance(cell.value, int | float):
            return "#"
        if isinstance(cell.value, bool):
            return "B"
        return "T"

    def _stringify_value(self, value: Any) -> str:
        text = str(value).replace("\n", " ").strip()
        return text[:40]
