"""Data models for representing sheet content."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CellDataType(str, Enum):
    """Cell data types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    EMPTY = "empty"
    FORMULA = "formula"
    ERROR = "error"


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

    # Border information
    border_top: str | None = Field(None, description="Top border style (none/thin/medium/thick)")
    border_bottom: str | None = Field(
        None, description="Bottom border style (none/thin/medium/thick)"
    )
    border_left: str | None = Field(None, description="Left border style (none/thin/medium/thick)")
    border_right: str | None = Field(
        None, description="Right border style (none/thin/medium/thick)"
    )

    # Cell properties
    is_merged: bool = Field(False, description="Cell is part of merged range")
    merge_range: str | None = Field(None, description="Merge range if applicable")
    has_formula: bool = Field(False, description="Cell contains formula")
    formula: str | None = Field(None, description="Formula text")

    # Hierarchical/alignment information
    indentation_level: int = Field(0, ge=0, description="Indentation level (0 = no indent)")
    alignment: str | None = Field(None, description="Horizontal alignment (left/center/right)")

    # Position information
    row: int = Field(0, ge=0, description="Row index (0-based)")
    column: int = Field(0, ge=0, description="Column index (0-based)")

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

    @property
    def value_type(self) -> str:
        """Alias for data_type for better clarity and pandas/polars compatibility."""
        return self.data_type

    @property
    def formatting(self) -> dict[str, Any]:
        """Return all formatting information as a dictionary."""
        return {
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "is_underline": self.is_underline,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "background_color": self.background_color,
            "border_top": self.border_top,
            "border_bottom": self.border_bottom,
            "border_left": self.border_left,
            "border_right": self.border_right,
            "indentation_level": self.indentation_level,
            "alignment": self.alignment,
        }


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

    # Optional fields for feature collection tracking
    file_path: str | None = Field(None, description="Source file path (for tracking)")
    file_type: str | None = Field(None, description="File type (xlsx, csv, etc.)")

    def has_data(self) -> bool:
        """Check if sheet has any data."""
        return len(self.cells) > 0

    def get_cell(self, row: int, column: int) -> CellData | None:
        """Get cell data by row and column indices."""
        address = self._get_address(row, column)
        # Support both string addresses and tuple keys for backward compatibility
        cell = self.cells.get(address)
        if cell is None:
            # Try tuple key as fallback (for tests)
            # Try tuple key as fallback (for tests) - cast to Any to avoid type error
            from typing import Any, cast

            cell = cast(dict[Any, CellData], self.cells).get((row, column))
        return cell

    def __setitem__(self, address: str, cell_data: CellData) -> None:
        """Allow sheet["A1"] = CellData(...) syntax."""
        # Parse Excel-style address to get row and column
        col_str = "".join(c for c in address if c.isalpha())
        row_str = "".join(c for c in address if c.isdigit())

        if not col_str or not row_str:
            raise ValueError(f"Invalid cell address: {address}")

        # Convert column letters to index (A=0, B=1, ..., Z=25, AA=26, ...)
        col = 0
        for char in col_str:
            col = col * 26 + (ord(char.upper()) - ord("A") + 1)
        col -= 1  # Convert to 0-based

        # Convert row number to 0-based index
        row = int(row_str) - 1

        # Use set_cell which handles row/column assignment
        self.set_cell(row, col, cell_data)

    def set_cell(self, row: int, column: int, cell_data: CellData) -> None:
        """Set cell data at specific position."""
        if row < 0:
            raise ValueError(f"Row index must be non-negative, got {row}")
        if column < 0:
            raise ValueError(f"Column index must be non-negative, got {column}")
        if cell_data is None:
            raise ValueError("cell_data cannot be None")

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
        if start_row < 0 or start_col < 0:
            raise ValueError("Start indices must be non-negative")
        if end_row < start_row or end_col < start_col:
            raise ValueError("End indices must be greater than or equal to start indices")

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

    def get_filled_cells_arrays(self) -> tuple[list[int], list[int], list[CellData]]:
        """Get all filled cells as parallel arrays for efficient processing.

        Returns:
            Tuple of (row_indices, col_indices, cell_data_list)
        """
        rows = []
        cols = []
        cells = []

        for _address, cell in self.cells.items():
            if cell and not cell.is_empty:
                rows.append(cell.row)
                cols.append(cell.column)
                cells.append(cell)

        return rows, cols, cells

    def get_cells_in_region(
        self, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> list[CellData]:
        """Get all non-empty cells within a region.

        Args:
            start_row: Starting row (inclusive)
            end_row: Ending row (inclusive)
            start_col: Starting column (inclusive)
            end_col: Ending column (inclusive)

        Returns:
            List of CellData objects in the region
        """
        cells_in_region = []

        for _address, cell in self.cells.items():
            if (
                cell
                and not cell.is_empty
                and start_row <= cell.row <= end_row
                and start_col <= cell.column <= end_col
            ):
                cells_in_region.append(cell)

        return cells_in_region

    def get_cells_batch(self, positions: list[tuple[int, int]]) -> list[CellData | None]:
        """Get multiple cells by their positions in a single batch operation.

        Args:
            positions: List of (row, col) tuples

        Returns:
            List of CellData objects or None for each position
        """
        return [self.get_cell(row, col) for row, col in positions]

    def set_cells_batch(self, cells: list[tuple[int, int, CellData]]) -> None:
        """Set multiple cells in a single batch operation.

        Args:
            cells: List of (row, col, cell_data) tuples
        """
        for row, col, cell_data in cells:
            self.set_cell(row, col, cell_data)

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

    @property
    def data(self) -> list[list[CellData | None]]:
        """Return sheet data as 2D list for easier access (pandas/polars style).

        Returns a list of rows, where each row is a list of cells.
        Missing cells are represented as None.
        """
        if self.max_row < 0 or self.max_column < 0:
            return []

        rows = []
        for row_idx in range(self.max_row + 1):
            row_data = []
            for col_idx in range(self.max_column + 1):
                cell = self.get_cell(row_idx, col_idx)
                row_data.append(cell)
            rows.append(row_data)
        return rows

    @property
    def merged_cells(self) -> list[str]:
        """Return list of unique merged cell ranges in the sheet."""
        merge_ranges = set()
        for cell in self.cells.values():
            if cell.is_merged and cell.merge_range:
                merge_ranges.add(cell.merge_range)
        return sorted(merge_ranges)


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
