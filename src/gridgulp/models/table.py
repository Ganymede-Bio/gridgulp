"""Table-related models."""

from typing import Any, TypeAlias

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

    def to_excel(self) -> str:
        """Alias for excel_range property."""
        return self.excel_range

    @classmethod
    def from_excel(cls, excel_range: str) -> "TableRange":
        """Create TableRange from Excel-style range string."""
        # Parse Excel range (e.g., "A1:D10")
        parts = excel_range.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid Excel range: {excel_range}")

        def parse_cell(cell: str) -> tuple[int, int]:
            # Extract column letters and row number
            import re

            match = re.match(r"([A-Z]+)(\d+)", cell.upper())
            if not match:
                raise ValueError(f"Invalid cell reference: {cell}")

            col_str, row_str = match.groups()

            # Convert column letters to index
            col = 0
            for char in col_str:
                col = col * 26 + (ord(char) - ord("A") + 1)
            col -= 1  # 0-indexed

            row = int(row_str) - 1  # 0-indexed
            return row, col

        start_row, start_col = parse_cell(parts[0])
        end_row, end_col = parse_cell(parts[1])

        return cls(start_row=start_row, start_col=start_col, end_row=end_row, end_col=end_col)

    @property
    def row_count(self) -> int:
        """Number of rows in the range."""
        return self.end_row - self.start_row + 1

    @property
    def col_count(self) -> int:
        """Number of columns in the range."""
        return self.end_col - self.start_col + 1


class HeaderInfo(BaseModel):
    """Information about table headers."""

    model_config = ConfigDict(strict=True)

    row_count: int = Field(..., ge=1, description="Number of header rows")
    headers: list[str] | None = Field(
        None, description="Simple header values (for single-row headers)"
    )
    multi_row_headers: dict[int, list[str]] | None = Field(
        None, description="Column index to header hierarchy mapping (for multi-row headers)"
    )
    merged_regions: list[dict[str, Any]] | None = Field(
        None, description="Merged cell regions in headers"
    )

    @property
    def is_multi_row(self) -> bool:
        """Check if headers span multiple rows."""
        return self.row_count > 1


class TableInfo(BaseModel):
    """Information about a detected table."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique identifier for the table")
    range: TableRange = Field(..., description="Table location")
    suggested_name: str | None = Field(None, description="LLM-suggested name for the table")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    detection_method: str = Field(..., description="Method used to detect this table")
    sheet_name: str | None = Field(None, description="Sheet containing this table")
    headers: list[str] | None = Field(
        None, description="Detected header row values (deprecated, use header_info)"
    )
    header_info: HeaderInfo | None = Field(None, description="Detailed header information")
    data_preview: list[dict[str, Any]] | None = Field(
        None, description="Preview of table data (first few rows)"
    )
    has_headers: bool = Field(True, description="Whether table has headers")
    data_types: dict[str, str] | None = Field(
        None, description="Inferred data types for each column"
    )
    semantic_structure: dict[str, Any] | None = Field(
        None, description="Semantic structure information (sections, totals, etc.)"
    )
    format_preservation: dict[str, Any] | None = Field(
        None, description="Formatting that should be preserved"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata from detection/analysis"
    )

    @property
    def shape(self) -> tuple[int, int]:
        """Table shape as (rows, columns)."""
        return (self.range.row_count, self.range.col_count)

    @property
    def row_count(self) -> int:
        """Number of data rows (excluding headers)."""
        header_rows = 1
        if self.header_info and self.header_info.row_count:
            header_rows = self.header_info.row_count
        return self.range.row_count - header_rows

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return self.range.col_count


class ExtractedTable(BaseModel):
    """A table with extracted data and metadata."""

    model_config = ConfigDict(strict=True)

    info: TableInfo = Field(..., description="Table detection information")
    headers: list[str] = Field(..., description="Column headers")
    data: list[list[Any]] = Field(..., description="Table data rows")
    column_types: dict[str, str] = Field(..., description="Inferred column data types")
    extraction_method: str = Field(..., description="Method used for extraction")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")
    pandas_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for pandas ingestion"
    )


# Set alias for backward compatibility
CellRange: TypeAlias = TableRange
