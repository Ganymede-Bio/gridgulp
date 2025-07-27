"""Multi-row header detection for complex tables."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..models.sheet_data import SheetData
from ..models.table import TableRange
from .merged_cell_analyzer import MergedCell, MergedCellAnalyzer

logger = logging.getLogger(__name__)


class HeaderCell(BaseModel):
    """Represents a single cell in a multi-row header."""

    model_config = ConfigDict(strict=True)

    row: int = Field(..., ge=0, description="Row index (0-based)")
    col: int = Field(..., ge=0, description="Column index (0-based)")
    value: str = Field(..., description="Cell value")
    row_span: int = Field(1, ge=1, description="Number of rows this cell spans")
    col_span: int = Field(1, ge=1, description="Number of columns this cell spans")
    is_merged: bool = Field(False, description="Whether this is a merged cell")
    formatting: dict[str, Any] | None = Field(None, description="Cell formatting info")


class MultiRowHeader(BaseModel):
    """Represents a multi-row header structure."""

    model_config = ConfigDict(strict=True)

    start_row: int = Field(..., ge=0, description="First row of header")
    end_row: int = Field(..., ge=0, description="Last row of header (inclusive)")
    start_col: int = Field(..., ge=0, description="First column of header")
    end_col: int = Field(..., ge=0, description="Last column of header (inclusive)")
    cells: list[HeaderCell] = Field(..., description="All cells in the header")
    column_mappings: dict[int, list[str]] = Field(
        ..., description="Maps data column index to header hierarchy"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


@dataclass
class MergedCellInfo:
    """Information about a merged cell."""

    start_row: int
    start_col: int
    end_row: int
    end_col: int
    value: Any
    formatting: dict[str, Any] | None = None


class MultiHeaderDetector:
    """Detects multi-row headers in spreadsheets."""

    def __init__(self):
        self.min_header_rows = 1
        self.max_header_rows = 10
        self.merged_cell_analyzer = MergedCellAnalyzer()

    def detect_multi_row_headers(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        data: pd.DataFrame | None = None,
    ) -> MultiRowHeader | None:
        """
        Detect multi-row headers in a table.

        Args:
            sheet_data: The sheet data containing cell information
            table_range: The range of the table
            data: Optional DataFrame representation of the data

        Returns:
            MultiRowHeader if detected, None otherwise
        """
        logger.info(f"Detecting multi-row headers for range {table_range.excel_range}")

        # First, analyze merged cells in the table range
        merged_cells = self.merged_cell_analyzer.analyze_merged_cells(sheet_data, table_range)

        # Find header merged cells
        header_merged_cells = self.merged_cell_analyzer.find_header_merged_cells(
            merged_cells, self.max_header_rows
        )

        # Analyze potential header rows
        header_row_count = self._estimate_header_rows_from_sheet(
            sheet_data, table_range, header_merged_cells
        )

        if header_row_count <= 1:
            logger.debug("No multi-row header detected")
            return None

        # Extract header cells with merged cell awareness
        header_cells = self._extract_header_cells_from_sheet(
            sheet_data, header_row_count, table_range, header_merged_cells
        )

        # Build column mappings using merged cell information
        column_mappings = self.merged_cell_analyzer.get_column_header_mapping(
            header_merged_cells, table_range.col_count, table_range.start_col
        )

        # Add non-merged header values to mappings
        self._enhance_column_mappings(column_mappings, header_cells, header_row_count, table_range)

        # Detect column spans
        column_spans = self.merged_cell_analyzer.build_column_spans(
            header_merged_cells, table_range
        )

        # Calculate confidence
        confidence = self._calculate_confidence_from_sheet(
            header_cells, column_mappings, sheet_data, header_merged_cells
        )

        return MultiRowHeader(
            start_row=table_range.start_row,
            end_row=table_range.start_row + header_row_count - 1,
            start_col=table_range.start_col,
            end_col=table_range.end_col,
            cells=header_cells,
            column_mappings=column_mappings,
            confidence=confidence,
        )

    def _estimate_header_rows_from_sheet(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        header_merged_cells: list[MergedCell],
    ) -> int:
        """Estimate the number of header rows from sheet data."""
        if not sheet_data or table_range.row_count == 0:
            return 0

        # Check for merged cells in top rows
        initial_header_rows = 0
        if header_merged_cells:
            max_merged_row = 0
            for cell in header_merged_cells:
                max_merged_row = max(max_merged_row, cell.end_row - table_range.start_row)
            # Start checking from after merged cells
            initial_header_rows = min(max_merged_row + 1, self.max_header_rows)

        # Analyze data patterns to find where headers end
        # Start from initial_header_rows to check for additional non-merged headers
        for row_offset in range(
            initial_header_rows, min(self.max_header_rows, table_range.row_count)
        ):
            row_idx = table_range.start_row + row_offset

            # Check if this row looks like data
            numeric_count = 0
            non_empty_count = 0
            bold_count = 0

            for col_offset in range(table_range.col_count):
                col_idx = table_range.start_col + col_offset
                cell = sheet_data.get_cell(row_idx, col_idx)

                if cell and cell.value is not None:
                    non_empty_count += 1
                    if cell.data_type == "number":
                        numeric_count += 1
                    if cell.is_bold:
                        bold_count += 1

            # If majority of cells are bold and non-numeric, likely still headers
            if non_empty_count > 0:
                if bold_count / non_empty_count > 0.5 and numeric_count / non_empty_count < 0.2:
                    continue  # This is likely still a header row

            # If 80% of non-empty cells are numeric, likely data row
            if non_empty_count > 0 and numeric_count / non_empty_count > 0.8:
                return row_offset

            # Check for significant formatting changes
            if row_offset > 0:
                if self._has_format_boundary_in_sheet(
                    sheet_data, table_range, row_idx - 1, row_idx
                ):
                    return row_offset

        # Return at least initial_header_rows if we found merged cells
        if initial_header_rows > 0:
            return initial_header_rows

        # Default to single row if no multi-row pattern found
        return 1

    def _extract_header_cells_from_sheet(
        self,
        sheet_data: SheetData,
        header_row_count: int,
        table_range: TableRange,
        header_merged_cells: list[MergedCell],
    ) -> list[HeaderCell]:
        """Extract all header cells from sheet data."""
        cells = []
        processed_positions = set()

        # Process merged cells first
        for merged in header_merged_cells:
            if merged.start_row < table_range.start_row + header_row_count:
                cell = HeaderCell(
                    row=merged.start_row - table_range.start_row,
                    col=merged.start_col - table_range.start_col,
                    value=merged.value,
                    row_span=merged.row_span,
                    col_span=merged.col_span,
                    is_merged=True,
                    formatting=None,  # Could extract from sheet_data if needed
                )
                cells.append(cell)

                # Mark all positions covered by this merged cell
                for r in range(merged.start_row, merged.end_row + 1):
                    for c in range(merged.start_col, merged.end_col + 1):
                        if (
                            r < table_range.start_row + header_row_count
                            and r >= table_range.start_row
                        ):
                            processed_positions.add((r, c))

        # Process regular cells
        for row_offset in range(header_row_count):
            row_idx = table_range.start_row + row_offset
            for col_offset in range(table_range.col_count):
                col_idx = table_range.start_col + col_offset

                if (row_idx, col_idx) not in processed_positions:
                    cell_data = sheet_data.get_cell(row_idx, col_idx)
                    if cell_data:
                        cell = HeaderCell(
                            row=row_offset,
                            col=col_offset,
                            value=(str(cell_data.value) if cell_data.value is not None else ""),
                            formatting=(
                                {
                                    "bold": cell_data.is_bold,
                                    "background_color": cell_data.background_color,
                                    "font_size": cell_data.font_size,
                                }
                                if cell_data
                                else None
                            ),
                        )
                        cells.append(cell)

        return cells

    def _enhance_column_mappings(
        self,
        column_mappings: dict[int, list[str]],
        header_cells: list[HeaderCell],
        header_row_count: int,
        table_range: TableRange,
    ) -> None:
        """Enhance column mappings with non-merged header values."""
        # Add regular (non-merged) header values
        for cell in header_cells:
            if not cell.is_merged and cell.value:
                if cell.col < len(column_mappings):
                    # Insert at appropriate position based on row
                    existing = column_mappings[cell.col]
                    if len(existing) <= cell.row:
                        # Extend list if needed
                        while len(existing) < cell.row:
                            existing.append("")
                        existing.append(cell.value)
                    elif not existing[cell.row]:
                        # Replace empty value
                        existing[cell.row] = cell.value

    def _has_format_boundary_in_sheet(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        prev_row: int,
        curr_row: int,
    ) -> bool:
        """Check if there's a significant formatting change between rows in sheet data."""
        differences = 0
        cells_checked = 0

        for col_offset in range(table_range.col_count):
            col_idx = table_range.start_col + col_offset
            prev_cell = sheet_data.get_cell(prev_row, col_idx)
            curr_cell = sheet_data.get_cell(curr_row, col_idx)

            if prev_cell and curr_cell:
                cells_checked += 1

                # Check formatting differences
                if prev_cell.is_bold != curr_cell.is_bold:
                    differences += 1
                if prev_cell.background_color != curr_cell.background_color:
                    differences += 1
                # Border detection would require additional cell properties

        # Significant change if more than 50% of cells have different formatting
        return cells_checked > 0 and differences > cells_checked * 0.5

    def _calculate_confidence_from_sheet(
        self,
        header_cells: list[HeaderCell],
        column_mappings: dict[int, list[str]],
        sheet_data: SheetData,
        header_merged_cells: list[MergedCell],
    ) -> float:
        """Calculate confidence score for multi-row header detection using sheet data."""
        scores = []

        # Score 1: Presence of merged cells
        if header_cells:
            merged_ratio = len(header_merged_cells) / len(header_cells)
            scores.append(min(merged_ratio * 2, 1.0))

        # Score 2: Hierarchy consistency
        hierarchy_depths = [len(h) for h in column_mappings.values() if h]
        if hierarchy_depths:
            avg_depth = sum(hierarchy_depths) / len(hierarchy_depths)
            depth_variance = np.var(hierarchy_depths) if len(hierarchy_depths) > 1 else 0
            consistency_score = 1.0 - min(depth_variance / avg_depth, 1.0) if avg_depth > 0 else 0
            scores.append(consistency_score)

        # Score 3: Non-empty header values
        non_empty = sum(1 for cell in header_cells if cell.value.strip())
        if header_cells:
            scores.append(non_empty / len(header_cells))

        # Score 4: Formatting consistency in headers
        bold_headers = sum(
            1 for cell in header_cells if cell.formatting and cell.formatting.get("bold")
        )
        if header_cells:
            scores.append(bold_headers / len(header_cells))

        # Return average of scores
        return sum(scores) / len(scores) if scores else 0.0

    def _estimate_header_rows(
        self,
        data: pd.DataFrame,
        merged_cells: list[MergedCellInfo] | None,
        formatting_info: dict[tuple[int, int], dict[str, Any]] | None,
    ) -> int:
        """Estimate the number of header rows."""
        if data.empty:
            return 0

        # Check for merged cells in top rows
        if merged_cells:
            max_merged_row = 0
            for cell in merged_cells:
                if cell.start_row < self.max_header_rows:
                    max_merged_row = max(max_merged_row, cell.end_row)
            if max_merged_row > 0:
                return min(max_merged_row + 1, self.max_header_rows)

        # Analyze data patterns to find where headers end
        for i in range(min(self.max_header_rows, len(data))):
            row = data.iloc[i]

            # Check if this row looks like data (all numeric, dates, etc.)
            numeric_count = sum(1 for val in row if self._is_numeric_value(val))
            if numeric_count > len(row) * 0.8:  # 80% numeric threshold
                return i

            # Check for significant formatting changes
            if formatting_info and i > 0:
                current_formats = [formatting_info.get((i, j), {}) for j in range(len(row))]
                prev_formats = [formatting_info.get((i - 1, j), {}) for j in range(len(row))]
                if self._has_format_boundary(prev_formats, current_formats):
                    return i

        # Default to single row if no multi-row pattern found
        return 1

    def _extract_header_cells(
        self,
        data: pd.DataFrame,
        header_row_count: int,
        table_range: TableRange,
        merged_cells: list[MergedCellInfo] | None,
        formatting_info: dict[tuple[int, int], dict[str, Any]] | None,
    ) -> list[HeaderCell]:
        """Extract all header cells with their properties."""
        cells = []
        processed_positions = set()

        # Process merged cells first
        if merged_cells:
            for merged in merged_cells:
                if merged.start_row < header_row_count:
                    cell = HeaderCell(
                        row=merged.start_row,
                        col=merged.start_col,
                        value=str(merged.value) if merged.value else "",
                        row_span=merged.end_row - merged.start_row + 1,
                        col_span=merged.end_col - merged.start_col + 1,
                        is_merged=True,
                        formatting=merged.formatting,
                    )
                    cells.append(cell)

                    # Mark all positions covered by this merged cell
                    for r in range(merged.start_row, merged.end_row + 1):
                        for c in range(merged.start_col, merged.end_col + 1):
                            processed_positions.add((r, c))

        # Process regular cells
        for row_idx in range(header_row_count):
            for col_idx in range(table_range.col_count):
                if (row_idx, col_idx) not in processed_positions:
                    value = data.iloc[row_idx, col_idx]
                    cell = HeaderCell(
                        row=row_idx,
                        col=col_idx,
                        value=str(value) if pd.notna(value) else "",
                        formatting=(
                            formatting_info.get((row_idx, col_idx)) if formatting_info else None
                        ),
                    )
                    cells.append(cell)

        return cells

    def _build_column_mappings(
        self,
        header_cells: list[HeaderCell],
        header_row_count: int,
        table_range: TableRange,
    ) -> dict[int, list[str]]:
        """Build mappings from data columns to header hierarchies."""
        mappings = {}

        for col_idx in range(table_range.col_count):
            hierarchy = []

            # Build hierarchy from top to bottom
            for row_idx in range(header_row_count):
                # Find the cell that covers this position
                covering_cell = None
                for cell in header_cells:
                    if (
                        cell.row <= row_idx < cell.row + cell.row_span
                        and cell.col <= col_idx < cell.col + cell.col_span
                    ):
                        covering_cell = cell
                        break

                if covering_cell and covering_cell.value:
                    # Only add to hierarchy if not already present (from row span)
                    if not hierarchy or hierarchy[-1] != covering_cell.value:
                        hierarchy.append(covering_cell.value)

            mappings[col_idx] = hierarchy

        return mappings

    def _calculate_confidence(
        self,
        header_cells: list[HeaderCell],
        column_mappings: dict[int, list[str]],
        data: pd.DataFrame,
    ) -> float:
        """Calculate confidence score for multi-row header detection."""
        scores = []

        # Score 1: Presence of merged cells
        merged_count = sum(1 for cell in header_cells if cell.is_merged)
        if header_cells:
            scores.append(min(merged_count / len(header_cells) * 2, 1.0))

        # Score 2: Hierarchy consistency
        hierarchy_depths = [len(h) for h in column_mappings.values()]
        if hierarchy_depths:
            avg_depth = sum(hierarchy_depths) / len(hierarchy_depths)
            depth_variance = np.var(hierarchy_depths)
            consistency_score = 1.0 - min(depth_variance / avg_depth, 1.0) if avg_depth > 0 else 0
            scores.append(consistency_score)

        # Score 3: Non-empty header values
        non_empty = sum(1 for cell in header_cells if cell.value.strip())
        if header_cells:
            scores.append(non_empty / len(header_cells))

        # Return average of scores
        return sum(scores) / len(scores) if scores else 0.0

    def _is_numeric_value(self, value: Any) -> bool:
        """Check if a value is numeric."""
        if pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value.replace(",", "").replace("$", "").strip())
                return True
            except ValueError:
                return False
        return False

    def _has_format_boundary(
        self, prev_formats: list[dict[str, Any]], current_formats: list[dict[str, Any]]
    ) -> bool:
        """Check if there's a significant formatting change between rows."""
        if not prev_formats or not current_formats:
            return False

        # Count formatting differences
        differences = 0
        for prev, curr in zip(prev_formats, current_formats, strict=False):
            if prev.get("bold") != curr.get("bold"):
                differences += 1
            if prev.get("background_color") != curr.get("background_color"):
                differences += 1
            if prev.get("border_bottom") != curr.get("border_bottom"):
                differences += 2  # Borders are stronger indicators

        # Significant change if more than 50% of cells have different formatting
        return differences > len(prev_formats) * 0.5
