"""Vision analysis result models for GridPorter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .table import TableRange


class VisionRegion(BaseModel):
    """A table region detected by vision analysis."""

    model_config = ConfigDict(strict=True)

    # Pixel coordinates in the bitmap
    pixel_bounds: dict[str, int] = Field(
        ..., description="Pixel coordinates {x1, y1, x2, y2} in the bitmap"
    )

    # Cell coordinates in the spreadsheet
    cell_bounds: dict[str, int] = Field(
        ..., description="Cell coordinates {start_row, start_col, end_row, end_col}"
    )

    # Excel-style range
    range: str = Field(..., description="Excel-style range (e.g., 'A1:D10')")

    # Confidence score from vision model
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")

    # Detection method used
    detection_method: str = Field(default="vision", description="Method used for detection")

    # Characteristics identified by vision model
    characteristics: dict[str, Any] = Field(
        default_factory=dict, description="Table characteristics (headers, formatting, etc.)"
    )

    def to_table_range(self) -> TableRange:
        """Convert to standard TableRange model.

        Returns:
            TableRange instance for compatibility with existing detection pipeline
        """
        return TableRange(
            range=self.range,
            start_row=self.cell_bounds["start_row"],
            start_col=self.cell_bounds["start_col"],
            end_row=self.cell_bounds["end_row"],
            end_col=self.cell_bounds["end_col"],
        )


class VisionAnalysisResult(BaseModel):
    """Result from vision-based table detection analysis."""

    model_config = ConfigDict(strict=True)

    # List of detected regions
    regions: list[VisionRegion] = Field(
        default_factory=list, description="List of detected table regions"
    )

    # Metadata about the analysis
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Analysis metadata (model used, processing time, etc.)"
    )

    # Bitmap information
    bitmap_info: dict[str, Any] = Field(
        default_factory=dict, description="Information about the generated bitmap"
    )

    # Raw model response for debugging
    raw_response: str | None = Field(None, description="Raw response from vision model")

    # Whether result was retrieved from cache
    cached: bool = Field(default=False, description="Whether result was cached")

    def high_confidence_regions(self, threshold: float = 0.7) -> list[VisionRegion]:
        """Get regions above confidence threshold.

        Args:
            threshold: Minimum confidence score

        Returns:
            List of high-confidence regions
        """
        return [region for region in self.regions if region.confidence >= threshold]

    def to_table_ranges(self) -> list[TableRange]:
        """Convert all regions to TableRange objects.

        Returns:
            List of TableRange objects for integration with existing pipeline
        """
        return [region.to_table_range() for region in self.regions]


class VisionDetectionMetrics(BaseModel):
    """Metrics for vision detection performance."""

    model_config = ConfigDict(strict=True)

    # Processing times
    bitmap_generation_time: float = Field(..., description="Time to generate bitmap (seconds)")
    vision_analysis_time: float = Field(..., description="Time for vision model analysis (seconds)")
    parsing_time: float = Field(..., description="Time to parse response (seconds)")
    total_time: float = Field(..., description="Total processing time (seconds)")

    # Token usage (if applicable)
    tokens_used: dict[str, int] = Field(
        default_factory=dict, description="Token usage information from model"
    )

    # Detection stats
    regions_detected: int = Field(..., description="Number of regions detected")
    regions_filtered: int = Field(..., description="Number of regions after filtering")
    confidence_scores: list[float] = Field(
        default_factory=list, description="All confidence scores"
    )

    # Bitmap information
    bitmap_dimensions: dict[str, int] = Field(
        default_factory=dict, description="Bitmap width/height information"
    )

    def average_confidence(self) -> float:
        """Calculate average confidence score.

        Returns:
            Average confidence, or 0.0 if no scores
        """
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    def max_confidence(self) -> float:
        """Get maximum confidence score.

        Returns:
            Maximum confidence, or 0.0 if no scores
        """
        return max(self.confidence_scores) if self.confidence_scores else 0.0

    def min_confidence(self) -> float:
        """Get minimum confidence score.

        Returns:
            Minimum confidence, or 0.0 if no scores
        """
        return min(self.confidence_scores) if self.confidence_scores else 0.0
