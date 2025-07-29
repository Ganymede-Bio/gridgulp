"""Merged cell analysis for complex table headers."""

import logging
from dataclasses import dataclass

from ..models.sheet_data import CellData, SheetData
from ..models.table import TableRange

logger = logging.getLogger(__name__)


@dataclass
class MergedCell:
    """Represents a merged cell region."""

    start_row: int
    start_col: int
    end_row: int
    end_col: int
    value: str
    is_header: bool = False
    spans_columns: bool = False
    spans_rows: bool = False

    @property
    def row_span(self) -> int:
        """Number of rows this cell spans."""
        return self.end_row - self.start_row + 1

    @property
    def col_span(self) -> int:
        """Number of columns this cell spans."""
        return self.end_col - self.start_col + 1

    @property
    def area(self) -> int:
        """Total area covered by the merged cell."""
        return self.row_span * self.col_span

    def contains(self, row: int, col: int) -> bool:
        """Check if a position is within this merged cell."""
        return self.start_row <= row <= self.end_row and self.start_col <= col <= self.end_col

    def overlaps_row(self, row: int) -> bool:
        """Check if this merged cell overlaps with a given row."""
        return self.start_row <= row <= self.end_row


class MergedCellAnalyzer:
    """Analyzes merged cells in spreadsheets for header detection."""

    def __init__(self) -> None:
        self.header_row_threshold = 10  # Max rows to consider for headers

    def analyze_merged_cells(
        self, sheet_data: SheetData, table_range: TableRange | None = None
    ) -> list[MergedCell]:
        """
        Analyze merged cells in a sheet or table range.

        Args:
            sheet_data: The sheet data containing cells
            table_range: Optional range to limit analysis

        Returns:
            List of MergedCell objects
        """
        if sheet_data is None:
            raise ValueError("sheet_data cannot be None")

        logger.info("Analyzing merged cells in sheet")

        # Extract merged cells from sheet data
        merged_cells = self._extract_merged_cells(sheet_data, table_range)

        # Analyze each merged cell
        for cell in merged_cells:
            self._analyze_merged_cell(cell, sheet_data)

        # Sort by position (top-left to bottom-right)
        merged_cells.sort(key=lambda c: (c.start_row, c.start_col))

        return merged_cells

    def find_header_merged_cells(
        self, merged_cells: list[MergedCell], max_header_row: int | None = None
    ) -> list[MergedCell]:
        """
        Find merged cells that are likely to be headers.

        Args:
            merged_cells: List of all merged cells
            max_header_row: Maximum row to consider for headers

        Returns:
            List of merged cells that are likely headers
        """
        if max_header_row is None:
            max_header_row = self.header_row_threshold

        header_cells = []

        for cell in merged_cells:
            if cell.start_row < max_header_row and self._is_likely_header(cell):
                cell.is_header = True
                header_cells.append(cell)

        return header_cells

    def build_column_spans(
        self, merged_cells: list[MergedCell], _table_range: TableRange
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Build column span information from merged cells.

        Args:
            merged_cells: List of merged cells
            _table_range: Table range

        Returns:
            Dict mapping row index to list of (start_col, end_col) spans
        """
        from collections import defaultdict

        spans_by_row = defaultdict(list)

        for cell in merged_cells:
            if cell.spans_columns:
                for row in range(cell.start_row, cell.end_row + 1):
                    spans_by_row[row].append((cell.start_col, cell.end_col))

        # Sort spans within each row and convert to regular dict
        return {row: sorted(spans) for row, spans in spans_by_row.items()}

    def _extract_merged_cells(
        self, sheet_data: SheetData, table_range: TableRange | None
    ) -> list[MergedCell]:
        """Extract merged cells from sheet data."""
        merged_cells = []
        processed_positions = set()

        # Iterate through all cells
        for row_idx in range(sheet_data.max_row + 1):
            for col_idx in range(sheet_data.max_column + 1):
                if (row_idx, col_idx) in processed_positions:
                    continue

                cell = sheet_data.get_cell(row_idx, col_idx)
                if cell and cell.is_merged and cell.merge_range:
                    # Parse merge range and create MergedCell
                    merged = self._parse_merge_range(cell, row_idx, col_idx)
                    if merged:
                        # Check if within table range
                        if table_range and not self._is_in_range(merged, table_range):
                            continue

                        merged_cells.append(merged)

                        # Mark all positions as processed
                        for r in range(merged.start_row, merged.end_row + 1):
                            for c in range(merged.start_col, merged.end_col + 1):
                                processed_positions.add((r, c))

        return merged_cells

    def _parse_merge_range(self, cell: CellData, _row_idx: int, _col_idx: int) -> MergedCell | None:
        """Parse merge range string to create MergedCell object."""
        if not cell.merge_range:
            return None

        try:
            # Parse Excel-style range (e.g., "A1:C3" or "B1:G1")
            parts = cell.merge_range.split(":")
            if len(parts) != 2:
                return None

            # Parse start and end positions
            start_addr = parts[0].strip()
            end_addr = parts[1].strip()

            # Parse column letters and row numbers
            def parse_excel_address(addr: str) -> tuple[int, int]:
                """Parse Excel address like 'A1' to (row, col)."""
                col_str = ""
                row_str = ""

                for char in addr:
                    if char.isalpha():
                        col_str += char
                    elif char.isdigit():
                        row_str += char

                # Convert column letters to index
                col = 0
                for char in col_str:
                    col = col * 26 + (ord(char.upper()) - ord("A") + 1)
                col -= 1  # Convert to 0-based

                # Convert row to 0-based
                row = int(row_str) - 1 if row_str else 0

                return row, col

            start_row, start_col = parse_excel_address(start_addr)
            end_row, end_col = parse_excel_address(end_addr)

            # Check if cell value exists
            value = str(cell.value) if cell.value is not None else ""

            merged = MergedCell(
                start_row=start_row,
                start_col=start_col,
                end_row=end_row,
                end_col=end_col,
                value=value,
                spans_columns=end_col > start_col,
                spans_rows=end_row > start_row,
            )

            return merged

        except Exception as e:
            logger.warning(f"Failed to parse merge range {cell.merge_range}: {e}")
            return None

    def _analyze_merged_cell(self, cell: MergedCell, sheet_data: SheetData) -> None:
        """Analyze properties of a merged cell."""
        # Determine if this merged cell spans multiple columns or rows
        cell.spans_columns = cell.col_span > 1
        cell.spans_rows = cell.row_span > 1

        # Check formatting to help identify headers
        first_cell = sheet_data.get_cell(cell.start_row, cell.start_col)
        if first_cell:
            # Headers often have specific formatting
            is_bold = first_cell.is_bold
            has_background = first_cell.background_color is not None

            # Simple heuristic for header detection
            if is_bold or has_background:
                cell.is_header = True

    def _is_likely_header(self, cell: MergedCell) -> bool:
        """Check if a merged cell is likely to be a header."""
        # Headers typically:
        # 1. Are in the top rows
        # 2. Span multiple columns
        # 3. Have non-empty values
        # 4. May span rows for hierarchical headers

        value = cell.value
        if not value or not value.strip():
            return False

        # If it spans columns, it's likely a header
        if cell.spans_columns:
            return True

        # If it's in the first few rows and spans rows, might be hierarchical
        return cell.start_row < 5 and cell.spans_rows

    def _is_in_range(self, merged: MergedCell, table_range: TableRange) -> bool:
        """Check if merged cell is within table range."""
        return (
            merged.start_row >= table_range.start_row
            and merged.end_row <= table_range.end_row
            and merged.start_col >= table_range.start_col
            and merged.end_col <= table_range.end_col
        )

    def detect_hierarchical_headers(
        self, merged_cells: list[MergedCell]
    ) -> dict[int, list[MergedCell]]:
        """
        Detect hierarchical header structure from merged cells.

        Args:
            merged_cells: List of merged cells

        Returns:
            Dict mapping header level (row) to merged cells at that level
        """
        from collections import defaultdict

        hierarchy = defaultdict(list)

        # Group by starting row
        for cell in merged_cells:
            if cell.is_header:
                hierarchy[cell.start_row].append(cell)

        # Sort cells within each level by column and convert to regular dict
        return {row: sorted(cells, key=lambda c: c.start_col) for row, cells in hierarchy.items()}

    def get_column_header_mapping(
        self, merged_cells: list[MergedCell], total_columns: int, table_start_col: int = 0
    ) -> dict[int, list[str]]:
        """
        Map each data column to its header hierarchy.

        Args:
            merged_cells: List of header merged cells
            total_columns: Total number of columns in the table
            table_start_col: Starting column of the table (for proper offset)

        Returns:
            Dict mapping column index to list of header values (top to bottom)
        """
        mapping: dict[int, list[str]] = {col: [] for col in range(total_columns)}

        # Get hierarchical structure
        hierarchy = self.detect_hierarchical_headers(merged_cells)

        # Process each level from top to bottom
        for row in sorted(hierarchy.keys()):
            cells_at_level = hierarchy[row]

            for cell in cells_at_level:
                # Apply this header to all columns it spans
                for col in range(cell.start_col, cell.end_col + 1):
                    # Convert absolute column to table-relative column
                    table_col = col - table_start_col
                    if 0 <= table_col < total_columns and cell.value:
                        mapping[table_col].append(cell.value)

        return mapping
