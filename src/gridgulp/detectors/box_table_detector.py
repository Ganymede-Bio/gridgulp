"""Box table detector for finding tables with complete borders.

This detector specializes in finding tables that have borders on all four sides,
which are very likely to be actual tables with high confidence.
"""

import logging
from typing import TYPE_CHECKING, Any

from ..models.table import TableInfo, TableRange
from ..utils.excel_utils import get_column_letter

if TYPE_CHECKING:
    from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


class BoxTableDetector:
    """Detects tables that have complete borders (all four sides)."""

    def __init__(self, min_table_size: tuple[int, int] = (2, 2), box_confidence: float = 0.95):
        """Initialize the box table detector.

        Args:
            min_table_size: Minimum (rows, cols) for a valid table
            box_confidence: Confidence score for tables with complete borders
        """
        self.min_rows, self.min_cols = min_table_size
        self.box_confidence = box_confidence
        self.logger = logger

    def detect_box_tables(self, sheet_data: "SheetData") -> list[TableInfo]:
        """Detect tables with complete borders.

        This is a fast-path detector that looks for regions where:
        1. All cells have top borders in the first row
        2. All cells have bottom borders in the last row
        3. All cells have left borders in the first column
        4. All cells have right borders in the last column
        5. The region forms a contiguous rectangle

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of high-confidence TableInfo objects
        """
        if not sheet_data.has_data():
            return []

        tables = []
        processed_cells = set()

        # Iterate through all cells looking for top-left corners of box tables
        for row in range(sheet_data.max_row + 1):
            for col in range(sheet_data.max_column + 1):
                if (row, col) in processed_cells:
                    continue

                # Check if this could be the start of a box table
                cell = sheet_data.get_cell(row, col)
                if not cell or not self._has_top_left_corner(cell):
                    continue

                # Try to find the complete box
                box_range = self._find_box_extent(sheet_data, row, col)
                # Verify it's a complete box
                if box_range and self._verify_complete_box(sheet_data, box_range):
                    # Mark all cells as processed
                    for r in range(box_range.start_row, box_range.end_row + 1):
                        for c in range(box_range.start_col, box_range.end_col + 1):
                            processed_cells.add((r, c))

                    # Check if first row looks like headers
                    has_headers = self._detect_headers(sheet_data, box_range)

                    # Extract headers if present
                    headers = None
                    if has_headers:
                        headers = self._extract_headers(sheet_data, box_range)

                    # Create TableInfo with high confidence
                    table = TableInfo(
                        id=f"box_{box_range.start_row}_{box_range.start_col}",
                        range=box_range,
                        confidence=self.box_confidence,
                        detection_method="box_table",
                        has_headers=has_headers,
                        headers=headers,
                        metadata={"detection_type": "complete_borders", "border_type": "box"},
                    )
                    tables.append(table)

                    self.logger.info(
                        f"Detected box table at {box_range.excel_range} "
                        f"with confidence {self.box_confidence:.2f}"
                    )

        return tables

    def _has_top_left_corner(self, cell: Any) -> bool:
        """Check if cell has borders suggesting top-left corner of a table."""
        has_top = cell.border_top is not None and cell.border_top != "none"
        has_left = cell.border_left is not None and cell.border_left != "none"
        return has_top and has_left

    def _find_box_extent(
        self, sheet_data: "SheetData", start_row: int, start_col: int
    ) -> TableRange | None:
        """Find the extent of a potential box table.

        Args:
            sheet_data: Sheet data
            start_row: Starting row
            start_col: Starting column

        Returns:
            TableRange if a valid box is found, None otherwise
        """
        # Find right edge - cells should have top border
        end_col = start_col
        for col in range(start_col + 1, sheet_data.max_column + 1):
            cell = sheet_data.get_cell(start_row, col)
            if not cell or not (cell.border_top and cell.border_top != "none"):
                break
            end_col = col

        # Find bottom edge - cells should have left border
        end_row = start_row
        for row in range(start_row + 1, sheet_data.max_row + 1):
            cell = sheet_data.get_cell(row, start_col)
            if not cell or not (cell.border_left and cell.border_left != "none"):
                break
            end_row = row

        # Check minimum size
        if end_row - start_row + 1 < self.min_rows or end_col - start_col + 1 < self.min_cols:
            return None

        return TableRange(
            start_row=start_row, start_col=start_col, end_row=end_row, end_col=end_col
        )

    def _verify_complete_box(self, sheet_data: "SheetData", box_range: TableRange) -> bool:
        """Verify that the range has complete borders on all sides.

        Args:
            sheet_data: Sheet data
            box_range: Range to verify

        Returns:
            True if all borders are present
        """
        # Check top row - all cells should have top border
        for col in range(box_range.start_col, box_range.end_col + 1):
            cell = sheet_data.get_cell(box_range.start_row, col)
            if not cell or not (cell.border_top and cell.border_top != "none"):
                return False

        # Check bottom row - all cells should have bottom border
        for col in range(box_range.start_col, box_range.end_col + 1):
            cell = sheet_data.get_cell(box_range.end_row, col)
            if not cell or not (cell.border_bottom and cell.border_bottom != "none"):
                return False

        # Check left column - all cells should have left border
        for row in range(box_range.start_row, box_range.end_row + 1):
            cell = sheet_data.get_cell(row, box_range.start_col)
            if not cell or not (cell.border_left and cell.border_left != "none"):
                return False

        # Check right column - all cells should have right border
        for row in range(box_range.start_row, box_range.end_row + 1):
            cell = sheet_data.get_cell(row, box_range.end_col)
            if not cell or not (cell.border_right and cell.border_right != "none"):
                return False

        # Verify reasonable data density (not just empty borders)
        non_empty_cells = 0
        total_cells = box_range.row_count * box_range.col_count

        for row in range(box_range.start_row, box_range.end_row + 1):
            for col in range(box_range.start_col, box_range.end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    non_empty_cells += 1

        # Require at least 30% non-empty cells
        return non_empty_cells / total_cells >= 0.3

    def _detect_headers(self, sheet_data: "SheetData", table_range: TableRange) -> bool:
        """Detect if the first row contains headers.

        Args:
            sheet_data: Sheet data
            table_range: Table range

        Returns:
            True if headers are detected
        """
        first_row_cells = []
        for col in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(table_range.start_row, col)
            if cell and cell.value is not None:
                first_row_cells.append(cell)

        if not first_row_cells:
            return False

        # Check if all first row cells are strings
        all_strings = all(cell.data_type == "string" for cell in first_row_cells)

        # Check if any are bold
        any_bold = any(cell.is_bold for cell in first_row_cells)

        # Check if second row has different data types
        has_different_types = False
        if table_range.start_row + 1 <= table_range.end_row:
            for col in range(table_range.start_col, table_range.end_col + 1):
                cell = sheet_data.get_cell(table_range.start_row + 1, col)
                if cell and cell.value is not None and cell.data_type != "string":
                    has_different_types = True
                    break

        # Headers likely if all strings AND (bold OR different types below)
        return all_strings and (any_bold or has_different_types)

    def _extract_headers(self, sheet_data: "SheetData", table_range: TableRange) -> list[str]:
        """Extract header values from the first row.

        Args:
            sheet_data: Sheet data
            table_range: Table range

        Returns:
            List of header strings
        """
        headers = []

        for col in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(table_range.start_row, col)
            if cell and cell.value is not None:
                header_val = str(cell.value).strip()
                headers.append(header_val)
            else:
                # Use column letter as fallback
                headers.append(get_column_letter(col))

        return headers
