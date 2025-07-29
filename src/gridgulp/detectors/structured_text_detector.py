"""Specialized detector for structured text files (TSV/CSV from instruments).

This module provides detection capabilities specifically designed for
instrument output files and other structured text formats that may contain
multiple tables with different formats.
"""

import logging
from typing import TYPE_CHECKING, Any

from ..models.table import TableInfo, TableRange
from .island_detector import IslandDetector

if TYPE_CHECKING:
    from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


class StructuredTextDetector:
    """Detects tables in structured text files using specialized heuristics."""

    def __init__(self) -> None:
        """Initialize the structured text detector."""
        self.logger = logger
        # Use structural analysis with reasonable parameters for text files
        self.island_detector = IslandDetector(
            max_gap=2,  # Allow 2-row gaps to properly separate tables
            min_island_size=2,  # Lower threshold for smaller tables
            include_diagonal=False,  # No diagonal connections for text files
            use_structural_analysis=True,  # Enable column-based analysis
        )

    def detect_tables(self, sheet_data: "SheetData") -> list[TableInfo]:
        """Detect all tables in the structured text sheet.

        Uses column consistency analysis and special handling for:
        - Plate map formats (96-well, 384-well, etc.)
        - Instrument output with metadata sections
        - Multiple table formats in single file

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of detected tables
        """
        if not sheet_data or not sheet_data.has_data():
            return []

        # First, detect islands using structural analysis
        islands = self.island_detector.detect_islands(sheet_data)

        # Log for debugging
        self.logger.debug(f"Detected {len(islands)} islands in {sheet_data.name}")
        for island in islands:
            self.logger.debug(
                f"Island: {island.min_row},{island.min_col} to {island.max_row},{island.max_col}"
            )

        # Also check for wide single-row or few-row tables that might be plate data
        wide_tables = self._detect_wide_tables(sheet_data)

        # Check each island for special formats
        tables = []
        processed_ranges = set()

        for island in islands:
            island_range = (island.min_row, island.max_row, island.min_col, island.max_col)
            if island_range in processed_ranges:
                continue

            # Check if it's a plate map
            self.logger.debug(f"Checking plate format for island {island_range}")
            plate_table = self._check_plate_format(sheet_data, island)
            if plate_table:
                self.logger.debug(f"Found plate table: {plate_table.metadata}")
                tables.append(plate_table)
                processed_ranges.add(island_range)
                continue

            # Check if this is a wide table that should be handled by wide table detector
            if island.max_col is not None and island.min_col is not None:
                col_count = island.max_col - island.min_col + 1
                if col_count >= 50:
                    # Skip this island - let wide table detector handle it
                    continue

            # Convert regular island to table
            table_infos = self.island_detector.convert_to_table_infos([island], sheet_data.name)
            if table_infos:
                table = table_infos[0]
                # Enhance with metadata detection (this may set has_headers)
                self._enhance_table_metadata(sheet_data, table)
                # Extract headers if present
                if table.has_headers:
                    headers = self._extract_headers(sheet_data, table.range)
                    table.headers = headers
                tables.append(table)
                processed_ranges.add(island_range)

        # Add wide tables that weren't captured by island detection
        for table in wide_tables:
            table_range = (
                table.range.start_row,
                table.range.end_row,
                table.range.start_col,
                table.range.end_col,
            )
            if table_range not in processed_ranges:
                tables.append(table)

        # Sort tables by position
        tables.sort(key=lambda t: (t.range.start_row, t.range.start_col))

        self.logger.info(f"Detected {len(tables)} tables in structured text file")
        return tables

    def _check_plate_format(self, sheet_data: "SheetData", island: Any) -> TableInfo | None:
        """Check if an island represents a plate map format.

        Args:
            sheet_data: Sheet data
            island: Island to check

        Returns:
            TableInfo if plate format detected, None otherwise
        """
        if (
            island.min_row is None
            or island.max_row is None
            or island.min_col is None
            or island.max_col is None
        ):
            return None

        # Standard plate formats: wells -> (rows, cols)
        PLATE_FORMATS = {
            6: [(2, 3), (3, 2)],
            24: [(4, 6), (6, 4)],
            96: [(8, 12), (12, 8)],
            384: [(16, 24), (24, 16)],
            1536: [(32, 48), (48, 32)],
        }

        # Check dimensions (accounting for row/col headers)
        data_rows = island.max_row - island.min_row + 1
        data_cols = island.max_col - island.min_col + 1

        # Log for debugging
        self.logger.debug(
            f"Checking plate format for island {island.min_row},{island.min_col} to {island.max_row},{island.max_col}"
        )
        self.logger.debug(f"Island dimensions: {data_rows}x{data_cols}")

        for wells, dimensions in PLATE_FORMATS.items():
            for expected_rows, expected_cols in dimensions:
                # Check if dimensions match (with tolerance for headers)
                # A 96-well plate (8x12) would be 9x13 with headers (row labels + col numbers)
                self.logger.debug(
                    f"Checking {wells}-well plate: expected {expected_rows}+1 x {expected_cols}+1, got {data_rows}x{data_cols}"
                )
                if (
                    data_rows == expected_rows + 1  # +1 for column headers
                    and data_cols == expected_cols + 1  # +1 for row labels
                ):
                    # Verify row headers
                    headers_valid = self._verify_plate_row_headers(
                        sheet_data, island, expected_rows
                    )
                    self.logger.debug(f"Row headers valid: {headers_valid}")
                    if headers_valid:
                        # Found a plate map!
                        table_range = TableRange(
                            start_row=island.min_row,
                            start_col=island.min_col,
                            end_row=island.max_row,
                            end_col=island.max_col,
                        )

                        return TableInfo(
                            id=f"plate_{wells}well_{island.min_row}_{island.min_col}",
                            range=table_range,
                            suggested_name=f"{wells}_well_plate",
                            confidence=0.95,
                            detection_method="plate_format_detection",
                            metadata={
                                "plate_format": f"{wells}-well",
                                "plate_dimensions": f"{expected_rows}x{expected_cols}",
                            },
                        )

        return None

    def _verify_plate_row_headers(
        self, sheet_data: "SheetData", island: Any, expected_rows: int
    ) -> bool:
        """Verify if the first column contains plate row headers (A, B, C, etc.).

        Args:
            sheet_data: Sheet data
            island: Island to check
            expected_rows: Expected number of data rows

        Returns:
            True if plate row headers found
        """
        if island.min_row is None or island.min_col is None:
            return False

        # Check first column for row labels
        found_labels = 0
        for i in range(expected_rows):
            row = island.min_row + i + 1  # Skip header row
            cell = sheet_data.get_cell(row, island.min_col)

            if cell and cell.value:
                expected_label = chr(ord("A") + i)
                if str(cell.value).strip().upper() == expected_label:
                    found_labels += 1

        # Accept if we found at least 75% of expected labels
        return found_labels >= expected_rows * 0.75

    def _extract_headers(self, sheet_data: "SheetData", table_range: TableRange) -> list[str]:
        """Extract headers from the first row of the table range.

        Args:
            sheet_data: Sheet data
            table_range: Table range

        Returns:
            List of header strings
        """
        headers = []
        for col in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(table_range.start_row, col)
            if cell and cell.value:
                headers.append(str(cell.value))
            else:
                headers.append(f"Column_{col + 1}")
        return headers

    def _enhance_table_metadata(self, sheet_data: "SheetData", table: TableInfo) -> None:
        """Enhance table with metadata specific to instrument output.

        Args:
            sheet_data: Sheet data
            table: Table to enhance
        """
        # Check if first row looks like headers
        first_row_cells = []
        for col in range(table.range.start_col, table.range.end_col + 1):
            cell = sheet_data.get_cell(table.range.start_row, col)
            if cell and cell.value:
                first_row_cells.append(str(cell.value))

        # Common instrument output headers
        instrument_keywords = [
            "sample",
            "well",
            "name",
            "value",
            "result",
            "concentration",
            "absorbance",
            "fluorescence",
            "mean",
            "std",
            "cv",
            "temperature",
        ]

        header_score = sum(
            1
            for header in first_row_cells
            if any(keyword in header.lower() for keyword in instrument_keywords)
        )

        if header_score >= len(first_row_cells) * 0.3:  # 30% match
            table.has_headers = True
            table.headers = first_row_cells

            # Update metadata
            if not table.metadata:
                table.metadata = {}
            table.metadata["instrument_output"] = True
            table.metadata["header_keywords"] = header_score

    def _detect_wide_tables(self, sheet_data: "SheetData") -> list[TableInfo]:
        """Detect wide tables that span many columns (like plate readings).

        These tables often have:
        - Very few rows (1-4)
        - Many columns (50+)
        - Row headers in first column
        - Column headers in first row

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of detected wide tables
        """
        tables = []

        # Get bounds
        max_row = 10  # Default
        max_col = 200  # Default
        if hasattr(sheet_data, "cells") and sheet_data.cells:
            max_row = max(cell.row for cell in sheet_data.cells.values()) + 1
            max_col = max(cell.column for cell in sheet_data.cells.values()) + 1

        # Look for rows with many columns of data
        processed_rows = set()
        for row in range(min(10, max_row)):  # Check first 10 rows
            if row in processed_rows:
                continue

            cols_with_data = []
            for col in range(min(200, max_col)):  # Check up to 200 columns
                cell = sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    cols_with_data.append(col)

            # If this row has 50+ columns, it might be a wide table
            if len(cols_with_data) >= 50:
                # Find the extent of this table
                min_col = min(cols_with_data)
                max_col = max(cols_with_data)

                # Check how many rows have similar width
                end_row = row
                for next_row in range(row + 1, min(row + 5, max_row)):
                    next_cols = []
                    for col in range(min_col, max_col + 1):
                        cell = sheet_data.get_cell(next_row, col)
                        if cell and cell.value is not None:
                            next_cols.append(col)

                    # If next row has significantly fewer columns, stop
                    if len(next_cols) < len(cols_with_data) * 0.3:
                        break
                    end_row = next_row
                    processed_rows.add(next_row)

                # Create table if it's wide enough
                if max_col - min_col >= 50:
                    table_range = TableRange(
                        start_row=row,
                        start_col=min_col,
                        end_row=end_row,
                        end_col=max_col,
                    )

                    # Check if it's a plate format
                    plate_format = self._check_wide_plate_format(sheet_data, table_range)

                    table = TableInfo(
                        id=f"wide_table_{row}_{min_col}",
                        range=table_range,
                        suggested_name=plate_format or "wide_data_table",
                        confidence=0.85,
                        detection_method="wide_table_detection",
                        has_headers=True,
                        headers=self._extract_headers(sheet_data, table_range),
                        metadata={
                            "table_type": "wide_table",
                            "width": max_col - min_col + 1,
                            "plate_format": plate_format,
                        },
                    )
                    tables.append(table)

        return tables

    def _check_wide_plate_format(
        self, sheet_data: "SheetData", table_range: TableRange
    ) -> str | None:
        """Check if a wide table is a plate format based on dimensions.

        Args:
            sheet_data: Sheet data
            table_range: Range of the wide table

        Returns:
            Plate format string if detected, None otherwise
        """
        width = table_range.col_count
        height = table_range.row_count

        # Check for 96-well plate (8x12 or 12x8)
        if 90 <= width <= 100 and 1 <= height <= 4:
            # Check if first row has column numbers 1-12
            has_col_numbers = 0
            for i in range(1, 13):
                cell = sheet_data.get_cell(table_range.start_row, table_range.start_col + i)
                if cell and str(cell.value) == str(i):
                    has_col_numbers += 1

            if has_col_numbers >= 10:  # At least 10 of 12 column numbers
                return "96-well-plate"

        # Check for 384-well plate (16x24 or 24x16)
        if 370 <= width <= 400 and 1 <= height <= 4:
            return "384-well-plate"

        return None
