"""Simple case detector for identifying single-table sheets.

This module provides fast detection of sheets that contain only a single table,
allowing the system to avoid expensive vision processing for simple cases.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models.table import TableInfo, TableRange
from ..utils.excel_utils import get_column_letter

if TYPE_CHECKING:
    from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


@dataclass
class SimpleTableResult:
    """Result of simple table detection."""

    is_simple_table: bool
    table_range: str | None = None
    confidence: float = 0.0
    has_headers: bool = True
    reason: str = ""


class SimpleCaseDetector:
    """Detects sheets containing a single, continuous table starting from A1."""

    def __init__(self, max_empty_threshold: int = 3):
        """Initialize the simple case detector.

        Args:
            max_empty_threshold: Maximum consecutive empty rows/cols allowed within table
        """
        self.logger = logger
        self.max_empty_threshold = max_empty_threshold

    def detect_simple_table(self, sheet_data: "SheetData") -> SimpleTableResult:
        """Detect if sheet contains a simple single table case.

        A simple table case is defined as:
        1. Data starts at or near A1 (within first 3 rows/cols)
        2. Continuous data region with no empty rows/columns
        3. Rectangular shape
        4. Optional headers in first row

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            SimpleTableResult with detection outcome
        """
        # Quick check: if no data, not a simple table
        if not sheet_data.has_data():
            return SimpleTableResult(is_simple_table=False, reason="Sheet has no data")

        # Find the bounds of all data
        min_row, max_row, min_col, max_col = self._find_data_bounds(sheet_data)

        # Check if data starts near A1
        if min_row > 2 or min_col > 2:  # Allow for small offsets
            return SimpleTableResult(
                is_simple_table=False,
                reason=f"Data doesn't start near A1 (starts at row {min_row + 1}, col {get_column_letter(min_col)})",
            )

        # Check for continuity - no empty rows or columns within the data region
        empty_rows = self._find_empty_rows(sheet_data, min_row, max_row, min_col, max_col)
        empty_cols = self._find_empty_columns(sheet_data, min_row, max_row, min_col, max_col)

        if empty_rows:
            return SimpleTableResult(
                is_simple_table=False,
                reason=f"Found {len(empty_rows)} empty rows within data region",
            )

        if empty_cols:
            return SimpleTableResult(
                is_simple_table=False,
                reason=f"Found {len(empty_cols)} empty columns within data region",
            )

        # Check density - ensure reasonable data density
        total_cells = (max_row - min_row + 1) * (max_col - min_col + 1)
        filled_cells = self._count_filled_cells(sheet_data, min_row, max_row, min_col, max_col)
        density = filled_cells / total_cells if total_cells > 0 else 0

        if density < 0.5:  # At least 50% of cells should have data
            return SimpleTableResult(
                is_simple_table=False, reason=f"Low data density: {density:.1%}"
            )

        # Check if first row looks like headers
        has_headers = self._detect_headers(sheet_data, min_row, min_col, max_col)

        # Calculate confidence based on various factors
        confidence = self._calculate_confidence(
            min_row, min_col, density, has_headers, max_row - min_row + 1
        )

        # Build the range string
        start_cell = f"{get_column_letter(min_col)}{min_row + 1}"
        end_cell = f"{get_column_letter(max_col)}{max_row + 1}"
        table_range = f"{start_cell}:{end_cell}"

        return SimpleTableResult(
            is_simple_table=True,
            table_range=table_range,
            confidence=confidence,
            has_headers=has_headers,
            reason="Detected simple continuous table",
        )

    def _find_data_bounds(self, sheet_data: "SheetData") -> tuple[int, int, int, int]:
        """Find the bounding box of all data in the sheet - OPTIMIZED for performance.

        Returns:
            Tuple of (min_row, max_row, min_col, max_col) in 0-based indices
        """
        # PERFORMANCE OPTIMIZATION: Use sheet_data's pre-computed bounds if available
        if hasattr(sheet_data, "max_row") and hasattr(sheet_data, "max_column"):
            # For dense tables starting at A1, use the sheet bounds directly
            non_empty_cells = sheet_data.get_non_empty_cells()
            if non_empty_cells:
                # Quick check: if we have data at (0,0) and sheet is reasonably dense,
                # use sheet bounds to avoid cell iteration
                first_cell = sheet_data.get_cell(0, 0)
                if first_cell and first_cell.value is not None:
                    total_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
                    if len(non_empty_cells) / total_cells > 0.3:  # 30%+ density
                        return 0, sheet_data.max_row, 0, sheet_data.max_column

        # Fallback to cell iteration for sparse or complex layouts
        min_row = float("inf")
        max_row = -1
        min_col = float("inf")
        max_col = -1

        # Use get_non_empty_cells() which is more efficient
        for cell_data in sheet_data.get_non_empty_cells():
            min_row = min(min_row, cell_data.row)
            max_row = max(max_row, cell_data.row)
            min_col = min(min_col, cell_data.column)
            max_col = max(max_col, cell_data.column)

        # Handle edge case of no data
        if min_row == float("inf"):
            return 0, 0, 0, 0

        return int(min_row), max_row, int(min_col), max_col

    def _find_empty_rows(
        self,
        sheet_data: "SheetData",
        min_row: int,
        max_row: int,
        min_col: int,
        max_col: int,
    ) -> list[int]:
        """Find empty rows within the data bounds - OPTIMIZED for performance.

        Returns:
            List of 0-based row indices that are empty
        """
        # PERFORMANCE OPTIMIZATION: Use set lookup instead of nested loops
        non_empty_cells = sheet_data.get_non_empty_cells()
        cell_positions = set()
        for cell_data in non_empty_cells:
            cell_positions.add((cell_data.row, cell_data.column))

        empty_rows = []
        for row in range(min_row, max_row + 1):
            # Check if any cell in this row has data
            has_data = any((row, col) in cell_positions for col in range(min_col, max_col + 1))
            if not has_data:
                empty_rows.append(row)

        return empty_rows

    def _find_empty_columns(
        self,
        sheet_data: "SheetData",
        min_row: int,
        max_row: int,
        min_col: int,
        max_col: int,
    ) -> list[int]:
        """Find empty columns within the data bounds - OPTIMIZED for performance.

        Returns:
            List of 0-based column indices that are empty
        """
        # PERFORMANCE OPTIMIZATION: Reuse cell_positions from memory if possible
        if not hasattr(self, "_cached_cell_positions"):
            non_empty_cells = sheet_data.get_non_empty_cells()
            self._cached_cell_positions = set()
            for cell_data in non_empty_cells:
                self._cached_cell_positions.add((cell_data.row, cell_data.column))

        empty_cols = []
        for col in range(min_col, max_col + 1):
            # Check if any cell in this column has data
            has_data = any(
                (row, col) in self._cached_cell_positions for row in range(min_row, max_row + 1)
            )
            if not has_data:
                empty_cols.append(col)

        return empty_cols

    def _count_filled_cells(
        self,
        sheet_data: "SheetData",
        min_row: int,
        max_row: int,
        min_col: int,
        max_col: int,
    ) -> int:
        """Count the number of filled cells in the region.

        Returns:
            Number of cells with non-None values
        """
        count = 0
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    count += 1
        return count

    def _detect_headers(
        self, sheet_data: "SheetData", first_row: int, min_col: int, max_col: int
    ) -> bool:
        """Detect if the first row contains headers.

        Headers are detected by:
        1. All cells in first row are text
        2. Different data types in subsequent rows
        3. Bold formatting in first row

        Returns:
            True if headers are likely present
        """
        # Check first row characteristics
        first_row_cells = []
        for col in range(min_col, max_col + 1):
            cell = sheet_data.get_cell(first_row, col)
            if cell and cell.value is not None:
                first_row_cells.append(cell)

        if not first_row_cells:
            return False

        # Check if all first row cells are strings
        all_strings = all(cell.data_type == "string" for cell in first_row_cells)

        # Check if any are bold (common header formatting)
        any_bold = any(cell.is_bold for cell in first_row_cells)

        # Check second row for different data types
        has_different_types = False
        if first_row + 1 <= sheet_data.max_row:
            for col in range(min_col, max_col + 1):
                cell = sheet_data.get_cell(first_row + 1, col)
                if cell and cell.value is not None and cell.data_type != "string":
                    has_different_types = True
                    break

        # Headers likely if: all strings in first row AND (bold OR different types below)
        return all_strings and (any_bold or has_different_types)

    def _calculate_confidence(
        self,
        min_row: int,
        min_col: int,
        density: float,
        has_headers: bool,
        row_count: int,
    ) -> float:
        """Calculate confidence score for simple table detection.

        Args:
            min_row: Starting row (0-based)
            min_col: Starting column (0-based)
            density: Data density ratio
            has_headers: Whether headers were detected
            row_count: Number of rows in table

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Boost for starting at A1
        if min_row == 0 and min_col == 0:
            confidence += 0.2
        elif min_row <= 1 and min_col <= 1:
            confidence += 0.1

        # Boost for high density
        if density > 0.9:
            confidence += 0.2
        elif density > 0.7:
            confidence += 0.1

        # Boost for detected headers
        if has_headers:
            confidence += 0.1

        # Small penalty for very small tables
        if row_count < 3:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def convert_to_table_info(self, result: SimpleTableResult, sheet_name: str) -> TableInfo | None:
        """Convert simple table result to TableInfo.

        Args:
            result: SimpleTableResult from detection
            sheet_name: Name of the sheet

        Returns:
            TableInfo if simple table detected, None otherwise
        """
        if not result.is_simple_table or not result.table_range:
            return None

        # Parse the range string to create TableRange
        if ":" in result.table_range:
            from ..utils.excel_utils import cell_to_indices

            start, end = result.table_range.split(":")
            start_row, start_col = cell_to_indices(start)
            end_row, end_col = cell_to_indices(end)

            table_range = TableRange(
                start_row=start_row,
                start_col=start_col,
                end_row=end_row,
                end_col=end_col,
            )

            return TableInfo(
                id=f"simple_{start_row}_{start_col}",
                range=table_range,
                suggested_name=f"{sheet_name}_table",
                confidence=result.confidence,
                detection_method="simple_case",
                headers=None,  # Would need to extract if needed
                data_preview=None,  # Would need to extract if needed
            )

        return None
