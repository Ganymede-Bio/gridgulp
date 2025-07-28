"""Data models for multi-scale bitmap generation and vision processing."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CompressionLevel(IntEnum):
    """Excel-proportioned compression levels.

    Each level maintains Excel's natural 64:1 row:column ratio where applicable.
    """

    NONE = 0  # 1×1 - No compression (1 pixel = 1 cell)
    MILD = 1  # 16×1 - Mild compression
    EXCEL_RATIO = 2  # 64×1 - Excel's natural ratio
    LARGE = 3  # 256×4 - Maintains 64:1
    HUGE = 4  # 1024×16 - Maintains 64:1
    MAXIMUM = 5  # 4096×64 - Maintains 64:1

    @property
    def row_block(self) -> int:
        """Number of rows per pixel at this compression level."""
        blocks = {0: 1, 1: 16, 2: 64, 3: 256, 4: 1024, 5: 4096}
        return blocks[self.value]

    @property
    def col_block(self) -> int:
        """Number of columns per pixel at this compression level."""
        blocks = {0: 1, 1: 1, 2: 1, 3: 4, 4: 16, 5: 64}
        return blocks[self.value]

    @property
    def description(self) -> str:
        """Human-readable description of compression level."""
        descriptions = {
            0: "No compression (1 pixel = 1 cell)",
            1: "16:1 ratio",
            2: "64:1 ratio (Excel-proportioned)",
            3: "256:4 ratio (maintains 64:1)",
            4: "1024:16 ratio (maintains 64:1)",
            5: "4096:64 ratio (maintains 64:1)",
        }
        return descriptions[self.value]

    @property
    def max_cells(self) -> int:
        """Maximum number of cells this level can handle efficiently."""
        max_cells = {
            0: 100_000,  # 316×316
            1: 1_600_000,  # 1.6M cells
            2: 6_400_000,  # 6.4M cells
            3: 100_000_000,  # 100M cells
            4: 1_600_000_000,  # 1.6B cells
            5: 16_000_000_000,  # 16B cells (Excel max)
        }
        return max_cells[self.value]


@dataclass
class DataRegion:
    """A region of data detected in the spreadsheet."""

    region_id: str
    bounds: dict[str, int]  # top, left, bottom, right
    cell_count: int
    density: float  # 0.0 to 1.0
    characteristics: dict[str, Any] = field(default_factory=dict)
    skip: bool = False
    skip_reason: str | None = None

    @property
    def rows(self) -> int:
        """Number of rows in this region."""
        return self.bounds["bottom"] - self.bounds["top"] + 1

    @property
    def cols(self) -> int:
        """Number of columns in this region."""
        return self.bounds["right"] - self.bounds["left"] + 1

    @property
    def total_cells(self) -> int:
        """Total cells in the region bounds."""
        return self.rows * self.cols


class VisionImage(BaseModel):
    """An image prepared for vision model analysis."""

    model_config = ConfigDict(strict=True)

    image_id: str = Field(..., description="Unique identifier for this image")
    image_data: str = Field(..., description="Base64 encoded PNG data")
    compression_level: int = Field(..., ge=0, le=5, description="Compression level used")
    block_size: list[int] = Field(..., description="[row_block, col_block] size")
    description: str = Field(..., description="Explicit description for vision model")
    purpose: str = Field(..., description="Why this image is included")
    covers_cells: str = Field(..., description="Excel range covered (e.g., A1:Z100)")
    size_bytes: int = Field(..., gt=0, description="Size of the image in bytes")

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)


class MultiScaleBitmaps(BaseModel):
    """Collection of bitmaps at multiple scales for a sheet."""

    model_config = ConfigDict(strict=True)

    sheet_name: str = Field(..., description="Name of the sheet")
    sheet_dimensions: dict[str, int] = Field(..., description="Total rows and cols")
    data_bounds: dict[str, int] = Field(..., description="Bounds of actual data")

    overview: VisionImage | None = Field(None, description="Compressed overview image")
    detail_views: list[VisionImage] = Field(
        default_factory=list, description="Full resolution views of data regions"
    )
    compression_strategy: Literal["single_image", "multi_scale", "progressive"] = Field(
        ..., description="Strategy used for compression"
    )

    total_size_mb: float = Field(..., description="Total size of all images")
    generation_time_ms: int = Field(..., description="Time to generate all bitmaps")

    def add_image(self, image: VisionImage) -> None:
        """Add an image and update total size."""
        if image.compression_level == 0:
            self.detail_views.append(image)
        else:
            self.overview = image
        self.total_size_mb = self._calculate_total_size()

    def _calculate_total_size(self) -> float:
        """Calculate total size of all images."""
        total = 0.0
        if self.overview:
            total += self.overview.size_mb
        for detail in self.detail_views:
            total += detail.size_mb
        return total


class VisionRequest(BaseModel):
    """A request prepared for vision model analysis."""

    model_config = ConfigDict(strict=True)

    images: list[VisionImage] = Field(..., description="All images to analyze")
    prompt_template: Literal["EXPLICIT_MULTI_SCALE", "SINGLE_IMAGE", "PROGRESSIVE"] = Field(
        ..., description="Prompt template to use"
    )
    total_images: int = Field(..., description="Number of images")
    total_size_mb: float = Field(..., description="Total size of all images")

    def validate_size_limit(self, limit_mb: float = 20.0) -> bool:
        """Check if request is within size limits."""
        return self.total_size_mb <= limit_mb


class ProgressiveRefinementPhase(BaseModel):
    """A phase in progressive refinement for large sheets."""

    model_config = ConfigDict(strict=True)

    phase: Literal["overview", "refinement", "verification"] = Field(
        ..., description="Current phase"
    )
    strategy: str = Field(..., description="Strategy for this phase")
    compression_level: int = Field(..., description="Compression level to use")
    focus_regions: list[dict[str, int]] = Field(
        default_factory=list, description="Regions to focus on"
    )
    purpose: str = Field(..., description="Purpose of this phase")


class ExactBounds(BaseModel):
    """Exact cell boundaries detected by vision model."""

    model_config = ConfigDict(strict=True)

    top_row: int = Field(..., ge=0, description="Top row (0-indexed)")
    left_col: int = Field(..., ge=0, description="Left column (0-indexed)")
    bottom_row: int = Field(..., ge=0, description="Bottom row (0-indexed)")
    right_col: int = Field(..., ge=0, description="Right column (0-indexed)")

    @property
    def excel_range(self) -> str:
        """Convert to Excel A1 notation."""
        from ..utils.excel_utils import to_excel_range

        return to_excel_range(self.top_row, self.left_col, self.bottom_row, self.right_col)

    @property
    def total_cells(self) -> int:
        """Total number of cells in these bounds."""
        return (self.bottom_row - self.top_row + 1) * (self.right_col - self.left_col + 1)
