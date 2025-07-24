"""Data models for representing sheet content."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CellData(BaseModel):
    """Represents a single cell with its value and formatting information."""

    model_config = ConfigDict(strict=True)

    value: str | int | float | bool | datetime | Decimal | None = Field(
        None, description="Cell value"
    )
    formatted_value: str | None = Field(None, description="Formatted string representation")
    data_type: str = Field("string", description="Detected data type")

    # Formatting information
    is_bold: bool = Field(False, description="Text is bold")
    is_italic: bool = Field(False, description="Text is italic")
    is_underline: bool = Field(False, description="Text is underlined")
    font_size: float | None = Field(None, description="Font size")
    font_color: str | None = Field(None, description="Font color (hex)")
    background_color: str | None = Field(None, description="Background color (hex)")

    # Cell properties
    is_merged: bool = Field(False, description="Cell is part of merged range")
    merge_range: str | None = Field(None, description="Merge range if applicable")
    has_formula: bool = Field(False, description="Cell contains formula")
    formula: str | None = Field(None, description="Formula text")

    # Position information
    row: int = Field(..., ge=0, description="Row index (0-based)")
    column: int = Field(..., ge=0, description="Column index (0-based)")

    @property
    def is_empty(self) -> bool:
        """Check if cell is effectively empty."""
        return self.value is None or (isinstance(self.value, str) and not self.value.strip())

    @property
    def excel_address(self) -> str:
        """Get Excel-style address (e.g., 'A1')."""

        def col_to_letter(col: int) -> str:
            result = ""
            while col >= 0:
                result = chr(col % 26 + ord("A")) + result
                col = col // 26 - 1
            return result

        return f"{col_to_letter(self.column)}{self.row + 1}"


class SheetData(BaseModel):
    """Represents a complete sheet with all its data and metadata."""

    model_config = ConfigDict(strict=True)

    name: str = Field(..., description="Sheet name")
    cells: dict[str, CellData] = Field(
        default_factory=dict, description="Cells indexed by Excel address (e.g., 'A1')"
    )
    max_row: int = Field(0, ge=0, description="Maximum row index with data")
    max_column: int = Field(0, ge=0, description="Maximum column index with data")

    # Sheet properties
    is_visible: bool = Field(True, description="Sheet is visible")
    sheet_type: str = Field("worksheet", description="Type of sheet")

    # Metadata
    creation_time: datetime | None = Field(None, description="Sheet creation time")
    modification_time: datetime | None = Field(None, description="Last modification")

    def get_cell(self, row: int, column: int) -> CellData | None:
        """Get cell data by row and column indices."""
        address = self._get_address(row, column)
        return self.cells.get(address)

    def set_cell(self, row: int, column: int, cell_data: CellData) -> None:
        """Set cell data at specific position."""
        address = self._get_address(row, column)
        cell_data.row = row
        cell_data.column = column
        self.cells[address] = cell_data

        # Update max dimensions
        self.max_row = max(self.max_row, row)
        self.max_column = max(self.max_column, column)

    def get_row_data(self, row: int) -> list[CellData | None]:
        """Get all cells in a specific row."""
        return [self.get_cell(row, col) for col in range(self.max_column + 1)]

    def get_column_data(self, column: int) -> list[CellData | None]:
        """Get all cells in a specific column."""
        return [self.get_cell(row, column) for row in range(self.max_row + 1)]

    def get_range_data(
        self, start_row: int, start_col: int, end_row: int, end_col: int
    ) -> list[list[CellData | None]]:
        """Get cells in a specific range."""
        result = []
        for row in range(start_row, end_row + 1):
            row_data = []
            for col in range(start_col, end_col + 1):
                row_data.append(self.get_cell(row, col))
            result.append(row_data)
        return result

    def get_non_empty_cells(self) -> list[CellData]:
        """Get all non-empty cells."""
        return [cell for cell in self.cells.values() if not cell.is_empty]

    def get_dimensions(self) -> tuple[int, int]:
        """Get sheet dimensions as (rows, columns)."""
        return (self.max_row + 1, self.max_column + 1)

    def _get_address(self, row: int, column: int) -> str:
        """Convert row/column to Excel address."""

        def col_to_letter(col: int) -> str:
            result = ""
            while col >= 0:
                result = chr(col % 26 + ord("A")) + result
                col = col // 26 - 1
            return result

        return f"{col_to_letter(column)}{row + 1}"


class FileData(BaseModel):
    """Represents complete file data with all sheets."""

    model_config = ConfigDict(strict=True)

    sheets: list[SheetData] = Field(default_factory=list, description="All sheets in file")
    metadata: dict[str, Any] = Field(default_factory=dict, description="File metadata")

    # File properties
    file_format: str = Field(..., description="File format (xlsx, xls, csv, etc.)")
    application: str | None = Field(None, description="Creating application")
    version: str | None = Field(None, description="File format version")

    def get_sheet_by_name(self, name: str) -> SheetData | None:
        """Get sheet by name."""
        for sheet in self.sheets:
            if sheet.name == name:
                return sheet
        return None

    def get_sheet_names(self) -> list[str]:
        """Get list of all sheet names."""
        return [sheet.name for sheet in self.sheets]

    @property
    def sheet_count(self) -> int:
        """Number of sheets in file."""
        return len(self.sheets)
