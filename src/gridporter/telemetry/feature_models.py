"""Feature models for table detection telemetry."""

from pydantic import BaseModel, ConfigDict, Field


class DetectionFeatures(BaseModel):
    """Comprehensive feature vector for table detection analysis."""

    model_config = ConfigDict(strict=True)

    # Identification
    file_path: str = Field(..., description="Path to the processed file")
    file_type: str = Field(..., description="File type (xlsx, csv, etc.)")
    sheet_name: str | None = Field(None, description="Sheet name for Excel files")
    table_range: str = Field(..., description="Cell range of detected table")
    detection_method: str = Field(..., description="Method used for detection")

    # Geometric features (from RegionVerifier)
    rectangularness: float | None = Field(
        None, ge=0.0, le=1.0, description="How rectangular the region is"
    )
    filledness: float | None = Field(None, ge=0.0, le=1.0, description="Ratio of filled cells")
    density: float | None = Field(
        None, ge=0.0, le=1.0, description="Data density considering sparsity"
    )
    contiguity: float | None = Field(None, ge=0.0, le=1.0, description="How connected the data is")
    edge_quality: float | None = Field(
        None, ge=0.0, le=1.0, description="Quality of region boundaries"
    )
    aspect_ratio: float | None = Field(None, gt=0.0, description="Width/height ratio")
    size_ratio: float | None = Field(
        None, ge=0.0, le=1.0, description="Region size relative to sheet"
    )

    # Pattern features (from PatternDetector)
    pattern_type: str | None = Field(None, description="Detected pattern type")
    header_density: float | None = Field(None, ge=0.0, le=1.0, description="Density of header rows")
    has_multi_headers: bool | None = Field(None, description="Whether multi-row headers detected")
    orientation: str | None = Field(
        None, description="Table orientation (horizontal/vertical/matrix)"
    )
    fill_ratio: float | None = Field(None, ge=0.0, le=1.0, description="Overall fill ratio")

    # Format features (from FormatAnalyzer)
    header_row_count: int | None = Field(None, ge=0, description="Number of header rows")
    has_bold_headers: bool | None = Field(None, description="Whether headers are bold")
    has_totals: bool | None = Field(None, description="Whether total rows detected")
    has_subtotals: bool | None = Field(None, description="Whether subtotal rows detected")
    section_count: int | None = Field(None, ge=0, description="Number of sections detected")
    separator_row_count: int | None = Field(None, ge=0, description="Number of separator rows")

    # Content features
    total_cells: int | None = Field(None, gt=0, description="Total cells in region")
    filled_cells: int | None = Field(None, ge=0, description="Number of filled cells")
    numeric_ratio: float | None = Field(None, ge=0.0, le=1.0, description="Ratio of numeric cells")
    date_columns: int | None = Field(None, ge=0, description="Number of date columns")
    text_columns: int | None = Field(None, ge=0, description="Number of text columns")
    empty_cell_ratio: float | None = Field(None, ge=0.0, le=1.0, description="Ratio of empty cells")

    # Hierarchical features (from HierarchicalDetector)
    max_hierarchy_depth: int | None = Field(None, ge=0, description="Maximum indentation depth")
    has_indentation: bool | None = Field(None, description="Whether indentation detected")
    subtotal_count: int | None = Field(None, ge=0, description="Number of subtotal rows")

    # Results
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    detection_success: bool = Field(..., description="Whether detection succeeded")
    error_message: str | None = Field(None, description="Error message if failed")
    processing_time_ms: int | None = Field(
        None, ge=0, description="Processing time in milliseconds"
    )

    def to_db_dict(self) -> dict:
        """Convert to dictionary suitable for database insertion."""
        data = self.model_dump()
        # Convert booleans to integers for SQLite
        for key, value in data.items():
            if isinstance(value, bool):
                data[key] = 1 if value else 0
        return data

    @classmethod
    def from_db_row(cls, row: dict) -> "DetectionFeatures":
        """Create instance from database row."""
        # Convert SQLite integers back to booleans
        bool_fields = {
            "has_multi_headers",
            "has_bold_headers",
            "has_totals",
            "has_subtotals",
            "has_indentation",
            "detection_success",
        }
        for field in bool_fields:
            if field in row and row[field] is not None:
                row[field] = bool(row[field])
        return cls(**row)
