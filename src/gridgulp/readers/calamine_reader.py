"""Fast Excel/ODS reader using python-calamine (Rust-based)."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from python_calamine import CalamineWorkbook

from ..models.file_info import FileInfo, FileType
from ..models.sheet_data import CellData, FileData, SheetData
from .base_reader import ReaderError, SyncBaseReader

logger = logging.getLogger(__name__)


class CalamineReader(SyncBaseReader):
    """High-performance reader for Excel/ODS files using Calamine (Rust backend).

    Supports:
    - Excel: .xlsx, .xlsm, .xlsb, .xls
    - OpenDocument: .ods

    Features:
    - Very fast reading (10-100x faster than openpyxl)
    - Low memory usage
    - Streaming support for large files
    """

    def __init__(self, file_path: Path, file_info: FileInfo):
        """Initialize Calamine reader.

        Args:
            file_path: Path to Excel/ODS file
            file_info: File information
        """
        super().__init__(file_path, file_info)
        self._workbook: CalamineWorkbook | None = None

    def can_read(self) -> bool:
        """Check if can read Excel/ODS files."""
        return self.file_info.type in {
            FileType.XLSX,
            FileType.XLS,
            FileType.XLSM,
            FileType.XLSB,
        }

    def get_supported_formats(self) -> list[str]:
        """Get supported formats."""
        return ["xlsx", "xls", "xlsm", "xlsb"]

    def read_sync(self) -> FileData:
        """Read file synchronously.

        Returns:
            FileData with all sheets

        Raises:
            ReaderError: If file cannot be read
        """
        self._log_read_start()
        self.validate_file()

        try:
            # Open workbook
            self._workbook = CalamineWorkbook.from_path(str(self.file_path))

            # Read all sheets
            sheets = []
            sheet_names = self._workbook.sheet_names

            for sheet_name in sheet_names:
                try:
                    sheet_data = self._read_sheet(sheet_name)
                    if sheet_data:
                        sheets.append(sheet_data)
                except Exception as e:
                    logger.warning(f"Failed to read sheet '{sheet_name}': {e}")
                    # Create empty sheet for failed reads
                    sheets.append(self._create_empty_sheet(sheet_name))

            # Create file data
            file_data = FileData(
                sheets=sheets,
                file_format=self.file_info.type.value,
                metadata={
                    "sheet_count": len(sheet_names),
                    "reader": "calamine",
                },
            )

            self._log_read_complete(file_data)
            return file_data

        except Exception as e:
            logger.error(f"Failed to read file with Calamine: {e}")
            raise ReaderError(f"Failed to read {self.file_path}: {str(e)}") from e
        finally:
            self._workbook = None

    def read_to_polars(self) -> list[pl.DataFrame]:
        """Read file directly to Polars DataFrames for maximum performance.

        This method bypasses the CellData model for better performance
        when you just need the data in a DataFrame.

        Returns:
            List of Polars DataFrames, one per sheet
        """
        self.validate_file()

        try:
            workbook = CalamineWorkbook.from_path(str(self.file_path))
            dataframes = []

            for sheet_name in workbook.sheet_names:
                # Get data as list of lists
                data = workbook.get_sheet_by_name(sheet_name).to_python()

                if not data:
                    # Empty sheet
                    dataframes.append(pl.DataFrame())
                    continue

                # Convert to Polars DataFrame
                # First row might be headers
                if len(data) > 1:
                    # Try to detect if first row is headers
                    first_row = data[0]
                    second_row = data[1] if len(data) > 1 else []

                    # Simple heuristic: if first row has all strings and second has mixed types
                    if all(isinstance(cell, str) for cell in first_row if cell is not None) and any(
                        isinstance(cell, int | float) for cell in second_row if cell is not None
                    ):
                        # Use first row as headers
                        # Convert headers to strings
                        headers = [
                            str(h) if h is not None else f"Column_{i}"
                            for i, h in enumerate(first_row)
                        ]
                        df = pl.DataFrame(data[1:], schema=headers, orient="row")
                    else:
                        # No headers detected
                        df = pl.DataFrame(data, orient="row")
                else:
                    df = pl.DataFrame(data, orient="row")

                # Add sheet name as metadata
                df = df.with_columns(pl.lit(sheet_name).alias("__sheet_name__"))
                dataframes.append(df)

            return dataframes

        except Exception as e:
            logger.error(f"Failed to read file to Polars: {e}")
            raise ReaderError(f"Failed to read {self.file_path}: {str(e)}") from e

    def _read_sheet(self, sheet_name: str) -> SheetData | None:
        """Read a single sheet.

        Args:
            sheet_name: Name of the sheet to read

        Returns:
            SheetData or None if sheet is empty
        """
        if self._workbook is None:
            return self._create_empty_sheet(sheet_name)

        try:
            # Get sheet data
            sheet = self._workbook.get_sheet_by_name(sheet_name)
            data = sheet.to_python()

            if not data:
                return self._create_empty_sheet(sheet_name)

            # Create sheet data
            sheet_data = SheetData(name=sheet_name)
            max_row = len(data) - 1
            max_col = 0

            # Process each row
            for row_idx, row in enumerate(data):
                if not row:
                    continue

                max_col = max(max_col, len(row) - 1)

                # Process each cell
                for col_idx, value in enumerate(row):
                    # Create cell data (handles None values)
                    cell = self._create_cell(value, row_idx, col_idx)
                    sheet_data.set_cell(row_idx, col_idx, cell)

            # Update dimensions
            sheet_data.max_row = max_row
            sheet_data.max_column = max_col

            return sheet_data

        except Exception as e:
            logger.error(f"Error reading sheet '{sheet_name}': {e}")
            return None

    def _create_cell(self, value: Any, row: int, col: int) -> CellData:
        """Create CellData from a raw value.

        Args:
            value: Cell value from Calamine
            row: Row index
            col: Column index

        Returns:
            CellData object
        """
        # Determine data type
        data_type = self._get_data_type(value)

        # Convert value if needed
        cell_value: Any
        formatted_value: str | None

        if isinstance(value, datetime):
            # Keep as datetime
            cell_value = value
            formatted_value = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, bool | int | float):
            cell_value = value
            formatted_value = str(value)
        else:
            # String or other
            cell_value = str(value) if value is not None else None
            formatted_value = cell_value

        return CellData(
            value=cell_value,
            formatted_value=formatted_value,
            data_type=data_type,
            row=row,
            column=col,
            # Calamine doesn't provide formatting info, so we use defaults
            is_bold=False,
            is_italic=False,
            is_underline=False,
        )

    def _get_data_type(self, value: Any) -> str:
        """Determine data type from value.

        Args:
            value: Cell value

        Returns:
            Data type string
        """
        if value is None:
            return "empty"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, datetime):
            return "datetime"
        elif isinstance(value, str):
            # Could be formula, but Calamine evaluates formulas
            return "string"
        else:
            return "unknown"
