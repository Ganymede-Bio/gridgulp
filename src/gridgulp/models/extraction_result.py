"""Data models for DataFrame extraction results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .table import CellRange


class ExtractedTable(BaseModel):
    """Represents an extracted table with DataFrame and metadata."""

    model_config = ConfigDict(strict=True)

    # Original detection info
    range: CellRange = Field(..., description="Original detected range")
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Original detection confidence"
    )
    detection_method: str = Field(..., description="Detection method used")

    # Extraction results
    extraction_status: str = Field(..., description="success, failed, or skipped")
    error_message: str | None = Field(None, description="Error message if extraction failed")

    # Header information
    has_headers: bool = Field(False, description="Whether headers were detected")
    header_rows: int = Field(0, ge=0, description="Number of header rows")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    orientation: str = Field("vertical", description="Table orientation: vertical or horizontal")

    # Table metrics
    data_rows: int = Field(0, ge=0, description="Number of data rows (excluding headers)")
    data_columns: int = Field(0, ge=0, description="Number of data columns")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall table quality score")
    data_density: float = Field(0.0, ge=0.0, le=1.0, description="Ratio of non-empty cells")

    # The extracted data
    dataframe_dict: dict[str, list[Any]] | None = Field(
        None, description="DataFrame as column-oriented dict (for JSON serialization)"
    )

    @property
    def is_valid(self) -> bool:
        """Check if extraction was successful."""
        return self.extraction_status == "success" and self.dataframe_dict is not None

    @property
    def shape(self) -> tuple[int, int]:
        """Get table shape (rows, columns)."""
        return (self.data_rows, self.data_columns)


class SheetExtractionResult(BaseModel):
    """Extraction results for a single sheet."""

    model_config = ConfigDict(strict=True)

    sheet_name: str = Field(..., description="Sheet name")
    total_tables_detected: int = Field(0, ge=0, description="Number of tables detected")
    tables_extracted: int = Field(0, ge=0, description="Number of tables successfully extracted")
    tables_failed: int = Field(0, ge=0, description="Number of failed extractions")

    extracted_tables: list[ExtractedTable] = Field(
        default_factory=list, description="All extraction attempts"
    )

    @property
    def success_rate(self) -> float:
        """Calculate extraction success rate."""
        if self.total_tables_detected == 0:
            return 0.0
        return self.tables_extracted / self.total_tables_detected


class FileExtractionResult(BaseModel):
    """Complete extraction results for a file."""

    model_config = ConfigDict(strict=True)

    file_path: str = Field(..., description="Path to the source file")
    file_type: str = Field(..., description="File type (xlsx, csv, etc.)")
    timestamp: str = Field(..., description="Extraction timestamp")

    # Summary statistics
    total_sheets: int = Field(0, ge=0, description="Total number of sheets")
    total_tables_detected: int = Field(
        0, ge=0, description="Total tables detected across all sheets"
    )
    total_tables_extracted: int = Field(0, ge=0, description="Total tables successfully extracted")
    total_tables_failed: int = Field(0, ge=0, description="Total failed extractions")

    # Processing metadata
    detection_time: float = Field(0.0, ge=0.0, description="Time for detection (seconds)")
    extraction_time: float = Field(0.0, ge=0.0, description="Time for extraction (seconds)")

    # Sheet results
    sheets: list[SheetExtractionResult] = Field(
        default_factory=list, description="Extraction results by sheet"
    )

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall extraction success rate."""
        if self.total_tables_detected == 0:
            return 0.0
        return self.total_tables_extracted / self.total_tables_detected

    @property
    def high_quality_tables(self) -> list[ExtractedTable]:
        """Get all tables with quality score > 0.7."""
        tables = []
        for sheet in self.sheets:
            tables.extend(
                [t for t in sheet.extracted_tables if t.is_valid and t.quality_score > 0.7]
            )
        return tables
