"""Table-related models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TableRange(BaseModel):
    """Represents a table's location in a spreadsheet."""

    model_config = ConfigDict(strict=True)

    start_row: int = Field(..., ge=0, description="Starting row (0-indexed)")
    start_col: int = Field(..., ge=0, description="Starting column (0-indexed)")
    end_row: int = Field(..., ge=0, description="Ending row (inclusive)")
    end_col: int = Field(..., ge=0, description="Ending column (inclusive)")

    @property
    def excel_range(self) -> str:
        """Convert to Excel-style range (e.g., 'A1:D10')."""

        # Convert column indices to letters
        def col_to_letter(col: int) -> str:
            result = ""
            while col >= 0:
                result = chr(col % 26 + ord("A")) + result
                col = col // 26 - 1
            return result

        start_col_letter = col_to_letter(self.start_col)
        end_col_letter = col_to_letter(self.end_col)
        return f"{start_col_letter}{self.start_row + 1}:{end_col_letter}{self.end_row + 1}"

    @property
    def row_count(self) -> int:
        """Number of rows in the range."""
        return self.end_row - self.start_row + 1

    @property
    def col_count(self) -> int:
        """Number of columns in the range."""
        return self.end_col - self.start_col + 1


class TableInfo(BaseModel):
    """Information about a detected table."""

    model_config = ConfigDict(strict=True)

    range: TableRange = Field(..., description="Table location")
    suggested_name: str | None = Field(None, description="LLM-suggested name for the table")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    detection_method: str = Field(..., description="Method used to detect this table")
    headers: list[str] | None = Field(None, description="Detected header row values")
    data_preview: list[dict[str, Any]] | None = Field(
        None, description="Preview of table data (first few rows)"
    )
    has_headers: bool = Field(True, description="Whether table has headers")
    data_types: dict[str, str] | None = Field(
        None, description="Inferred data types for each column"
    )

    @property
    def shape(self) -> tuple[int, int]:
        """Table shape as (rows, columns)."""
        return (self.range.row_count, self.range.col_count)
