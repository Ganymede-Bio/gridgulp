"""Generate bitmap representations of spreadsheet data for vision model analysis."""

import io
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from PIL import Image

from ..models.sheet_data import SheetData
from .quadtree import QuadTreeBounds, VisualizationRegion

# Optional telemetry import
try:
    from ..telemetry.metrics import get_metrics_collector

    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)

# Excel size limits
XLSX_MAX_ROWS = 1048576
XLSX_MAX_COLS = 16384
XLS_MAX_ROWS = 65536
XLS_MAX_COLS = 256


class CompressionMode(Enum):
    """Compression modes for cell representation."""

    FULL = "full"  # 8 bits per cell
    COMPRESSED_4 = "compressed_4"  # 4 bits per cell
    COMPRESSED_2 = "compressed_2"  # 2 bits per cell
    SAMPLED = "sampled"  # Downsampled for very large regions


@dataclass
class BitmapMetadata:
    """Metadata for generated bitmap images."""

    width: int
    height: int
    cell_width: int
    cell_height: int
    total_rows: int
    total_cols: int
    scale_factor: float
    mode: str
    compression: CompressionMode = CompressionMode.FULL
    region_bounds: QuadTreeBounds | None = None
    file_size_bytes: int | None = None


@dataclass
class BitmapResult:
    """Result of bitmap generation."""

    image_data: bytes
    metadata: BitmapMetadata
    compression_info: dict


class BitmapGenerator:
    """Convert spreadsheet data to bitmap representation for vision analysis."""

    # Default cell dimensions in pixels
    DEFAULT_CELL_WIDTH = 10
    DEFAULT_CELL_HEIGHT = 10

    # Maximum image dimensions to avoid memory issues
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048

    # Minimum cell size to ensure visibility
    MIN_CELL_SIZE = 3

    # GPT-4o constraints
    MAX_IMAGE_SIZE_MB = 20

    # Size limits are defined at module level

    def __init__(
        self,
        cell_width: int = DEFAULT_CELL_WIDTH,
        cell_height: int = DEFAULT_CELL_HEIGHT,
        mode: Literal["binary", "grayscale", "color"] = "binary",
        compression_level: int = 6,
        auto_compress: bool = True,
    ):
        """Initialize bitmap generator.

        Args:
            cell_width: Width of each cell in pixels
            cell_height: Height of each cell in pixels
            mode: Visualization mode - binary (filled/empty), grayscale (with formatting hints), or color
            compression_level: PNG compression level (0-9)
            auto_compress: Automatically select compression mode based on size
        """
        self.cell_width = max(cell_width, self.MIN_CELL_SIZE)
        self.cell_height = max(cell_height, self.MIN_CELL_SIZE)
        self.mode = mode
        self.compression_level = compression_level
        self.auto_compress = auto_compress

    def generate(
        self, sheet_data: SheetData, region: QuadTreeBounds | None = None
    ) -> tuple[bytes, BitmapMetadata]:
        """Generate bitmap from sheet data.

        Args:
            sheet_data: Sheet data to visualize
            region: Optional region bounds to visualize (None for entire sheet)

        Returns:
            Tuple of (PNG image bytes, metadata)
        """
        # Track timing if telemetry available
        if HAS_TELEMETRY:
            metrics = get_metrics_collector()
            start_time = time.time()

        # Use sheet data directly
        sheet = sheet_data

        # Calculate dimensions
        if region:
            rows = region.height
            cols = region.width
        else:
            rows = sheet.max_row + 1 if sheet.max_row >= 0 else 0
            cols = sheet.max_column + 1 if sheet.max_column >= 0 else 0

        if rows == 0 or cols == 0:
            # Empty sheet - return minimal white image
            img = Image.new("L", (10, 10), 255)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", compress_level=self.compression_level)
            img_data = buffer.getvalue()
            metadata = BitmapMetadata(
                width=10,
                height=10,
                cell_width=10,
                cell_height=10,
                total_rows=0,
                total_cols=0,
                scale_factor=1.0,
                mode=self.mode,
                compression=CompressionMode.FULL,
                region_bounds=region,
                file_size_bytes=len(img_data),
            )

            if HAS_TELEMETRY:
                metrics.record_duration("bitmap_generation", time.time() - start_time)
                metrics.increment("bitmaps_generated")

            return img_data, metadata

        # Determine compression mode based on size
        compression_mode = self._determine_compression_mode(rows, cols)

        # Generate bitmap based on compression mode
        if compression_mode == CompressionMode.SAMPLED:
            result = self._generate_sampled(sheet, region, rows, cols)
        elif compression_mode == CompressionMode.COMPRESSED_2:
            result = self._generate_compressed_2bit(sheet, region, rows, cols)
        elif compression_mode == CompressionMode.COMPRESSED_4:
            result = self._generate_compressed_4bit(sheet, region, rows, cols)
        else:
            result = self._generate_full(sheet, region, rows, cols)

        if HAS_TELEMETRY:
            duration = time.time() - start_time
            metrics.record_duration("bitmap_generation", duration)
            metrics.increment("bitmaps_generated")
            metrics.record_value("bitmap_size_bytes", result[1].file_size_bytes)
            metrics.record_value("bitmap_cells", rows * cols)
            logger.info(f"Bitmap generation took {duration:.3f}s")

        return result

    def _determine_compression_mode(self, rows: int, cols: int) -> CompressionMode:
        """Determine appropriate compression mode based on size."""
        total_cells = rows * cols

        if not self.auto_compress:
            return CompressionMode.FULL

        # Use sampling for extremely large regions
        if total_cells > 5_000_000:
            return CompressionMode.SAMPLED

        # Use 2-bit compression for large regions
        elif total_cells > 1_000_000:
            return CompressionMode.COMPRESSED_2

        # Use 4-bit compression for medium regions
        elif total_cells > 100_000:
            return CompressionMode.COMPRESSED_4

        # Full quality for small regions
        else:
            return CompressionMode.FULL

    def _generate_full(
        self,
        sheet_data: SheetData,
        region: QuadTreeBounds | None,
        rows: int,
        cols: int,
    ) -> tuple[bytes, BitmapMetadata]:
        """Generate full-quality bitmap."""
        start_row = region.min_row if region else 0
        start_col = region.min_col if region else 0

        # Calculate scale factor if image would be too large
        scale_factor = 1.0
        target_width = cols * self.cell_width
        target_height = rows * self.cell_height

        if target_width > self.MAX_WIDTH or target_height > self.MAX_HEIGHT:
            scale_factor = min(self.MAX_WIDTH / target_width, self.MAX_HEIGHT / target_height)
            logger.info(
                f"Scaling down image by {scale_factor:.2f} to fit within {self.MAX_WIDTH}x{self.MAX_HEIGHT}"
            )

        # Calculate actual cell dimensions
        actual_cell_width = max(int(self.cell_width * scale_factor), self.MIN_CELL_SIZE)
        actual_cell_height = max(int(self.cell_height * scale_factor), self.MIN_CELL_SIZE)

        # Create image array
        width = cols * actual_cell_width
        height = rows * actual_cell_height

        # Ensure final dimensions don't exceed limits even with MIN_CELL_SIZE
        if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
            final_scale = min(self.MAX_WIDTH / width, self.MAX_HEIGHT / height)
            width = int(width * final_scale)
            height = int(height * final_scale)
            actual_cell_width = max(width // cols, 1)
            actual_cell_height = max(height // rows, 1)

        if self.mode == "binary":
            # Binary mode: white background (255), black filled cells (0)
            img_array = np.full((height, width), 255, dtype=np.uint8)
        elif self.mode == "grayscale":
            # Grayscale mode: white background, varying gray levels for formatting
            img_array = np.full((height, width), 255, dtype=np.uint8)
        else:  # color
            # Color mode: RGB image
            img_array = np.full((height, width, 3), 255, dtype=np.uint8)

        # Fill in cells
        for row in range(rows):
            for col in range(cols):
                cell = sheet_data.get_cell(start_row + row, start_col + col)
                if cell and not cell.is_empty:
                    # Calculate pixel boundaries for this cell
                    y_start = row * actual_cell_height
                    y_end = (row + 1) * actual_cell_height
                    x_start = col * actual_cell_width
                    x_end = (col + 1) * actual_cell_width

                    if self.mode == "binary":
                        # Binary: filled cells are black
                        img_array[y_start:y_end, x_start:x_end] = 0
                    elif self.mode == "grayscale":
                        # Grayscale: use different gray levels for formatting
                        if cell.is_bold:
                            gray_level = 0  # Black for bold
                        elif cell.has_formula:
                            gray_level = 64  # Dark gray for formulas
                        elif cell.is_merged:
                            gray_level = 128  # Medium gray for merged
                        else:
                            gray_level = 192  # Light gray for regular filled cells
                        img_array[y_start:y_end, x_start:x_end] = gray_level
                    else:  # color
                        # Color mode: use different colors for different cell types
                        if cell.is_bold:
                            color = [0, 0, 0]  # Black for bold/headers
                        elif cell.has_formula:
                            color = [0, 0, 255]  # Blue for formulas
                        elif cell.is_merged:
                            color = [255, 165, 0]  # Orange for merged
                        elif cell.data_type == "number":
                            color = [0, 128, 0]  # Green for numbers
                        elif cell.data_type == "date":
                            color = [128, 0, 128]  # Purple for dates
                        else:
                            color = [64, 64, 64]  # Dark gray for text
                        img_array[y_start:y_end, x_start:x_end] = color

        # Convert to PIL Image
        if self.mode == "color":
            img = Image.fromarray(img_array, mode="RGB")
        else:
            img = Image.fromarray(img_array, mode="L")

        # Save as PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=self.compression_level)
        img_data = buffer.getvalue()

        metadata = BitmapMetadata(
            width=width,
            height=height,
            cell_width=actual_cell_width,
            cell_height=actual_cell_height,
            total_rows=rows,
            total_cols=cols,
            scale_factor=scale_factor,
            mode=self.mode,
            compression=CompressionMode.FULL,
            region_bounds=region,
            file_size_bytes=len(img_data),
        )

        logger.info(
            f"Generated {width}x{height} bitmap for {rows}x{cols} region "
            f"(cell size: {actual_cell_width}x{actual_cell_height}, size: {len(img_data)/1024:.1f}KB)"
        )

        return img_data, metadata

    def _generate_compressed_2bit(
        self,
        sheet_data: SheetData,
        region: QuadTreeBounds | None,
        rows: int,
        cols: int,
    ) -> tuple[bytes, BitmapMetadata]:
        """Generate 2-bit compressed bitmap.

        2-bit encoding:
        - 00: Empty cell
        - 01: Filled cell
        - 10: Header/bold cell
        - 11: Special cell (formula/merged)
        """
        start_row = region.min_row if region else 0
        start_col = region.min_col if region else 0

        # Create array to hold cell values
        cell_array = np.zeros((rows, cols), dtype=np.uint8)

        # Collect cells by type for vectorized encoding
        filled_positions = []
        header_positions = []
        special_positions = []

        # Get non-empty cells
        for cell in sheet_data.get_non_empty_cells():
            # Check if cell is within region bounds
            if (
                start_row <= cell.row < start_row + rows
                and start_col <= cell.column < start_col + cols
            ):
                local_row = cell.row - start_row
                local_col = cell.column - start_col

                if cell.is_bold:
                    header_positions.append((local_row, local_col))
                elif cell.has_formula or cell.is_merged:
                    special_positions.append((local_row, local_col))
                else:
                    filled_positions.append((local_row, local_col))

        # Vectorized assignments
        if filled_positions:
            positions = np.array(filled_positions)
            cell_array[positions[:, 0], positions[:, 1]] = 1

        if header_positions:
            positions = np.array(header_positions)
            cell_array[positions[:, 0], positions[:, 1]] = 2

        if special_positions:
            positions = np.array(special_positions)
            cell_array[positions[:, 0], positions[:, 1]] = 3

        # Convert to grayscale image using vectorized lookup
        # Map 2-bit values to grayscale: 0->255, 1->192, 2->64, 3->0
        gray_lookup = np.array([255, 192, 64, 0], dtype=np.uint8)
        grayscale_cells = gray_lookup[cell_array]

        # Use small cell size for compressed view
        cell_size = max(2, min(self.cell_width // 2, self.cell_height // 2))

        width = cols * cell_size
        height = rows * cell_size

        # Ensure dimensions fit within limits
        if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
            scale = min(self.MAX_WIDTH / width, self.MAX_HEIGHT / height)
            width = int(width * scale)
            height = int(height * scale)
            cell_size = max(1, min(width // cols, height // rows))

        # Create image array using NumPy repeat for efficient scaling
        if cell_size == 1:
            # Direct mapping when cell_size is 1
            img_array = grayscale_cells[:height, :width]
        else:
            # Use repeat for efficient upscaling
            img_array = np.repeat(np.repeat(grayscale_cells, cell_size, axis=0), cell_size, axis=1)
            # Trim to exact dimensions
            img_array = img_array[:height, :width]

        # Convert to PIL Image
        img = Image.fromarray(img_array, mode="L")

        # Save with high compression
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=9)
        img_data = buffer.getvalue()

        metadata = BitmapMetadata(
            width=width,
            height=height,
            cell_width=cell_size,
            cell_height=cell_size,
            total_rows=rows,
            total_cols=cols,
            scale_factor=cell_size / self.cell_width,
            mode="compressed_2bit",
            compression=CompressionMode.COMPRESSED_2,
            region_bounds=region,
            file_size_bytes=len(img_data),
        )

        logger.info(
            f"Generated 2-bit compressed {width}x{height} bitmap for {rows}x{cols} region "
            f"(size: {len(img_data)/1024:.1f}KB)"
        )

        return img_data, metadata

    def _generate_compressed_4bit(
        self,
        sheet_data: SheetData,
        region: QuadTreeBounds | None,
        rows: int,
        cols: int,
    ) -> tuple[bytes, BitmapMetadata]:
        """Generate 4-bit compressed bitmap with more nuanced representation."""
        # Similar to 2-bit but with more categories
        # For now, use smaller cell size with full grayscale
        start_row = region.min_row if region else 0
        start_col = region.min_col if region else 0

        # Use medium cell size
        cell_size = max(3, min(self.cell_width * 3 // 4, self.cell_height * 3 // 4))

        width = cols * cell_size
        height = rows * cell_size

        # Ensure dimensions fit
        if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
            scale = min(self.MAX_WIDTH / width, self.MAX_HEIGHT / height)
            width = int(width * scale)
            height = int(height * scale)
            cell_size = max(1, min(width // cols, height // rows))

        # Create image array
        img_array = np.full((height, width), 255, dtype=np.uint8)

        # Fill cells with more nuanced grayscale
        for row in range(rows):
            for col in range(cols):
                cell = sheet_data.get_cell(start_row + row, start_col + col)
                if cell and not cell.is_empty:
                    y_start = row * cell_size
                    y_end = min((row + 1) * cell_size, height)
                    x_start = col * cell_size
                    x_end = min((col + 1) * cell_size, width)

                    if y_end > y_start and x_end > x_start:
                        # More nuanced grayscale based on cell properties
                        if cell.is_bold:
                            gray = 0
                        elif cell.has_formula:
                            gray = 48
                        elif cell.is_merged:
                            gray = 96
                        elif cell.data_type == "number":
                            gray = 144
                        elif cell.data_type == "date":
                            gray = 192
                        else:
                            gray = 224

                        img_array[y_start:y_end, x_start:x_end] = gray

        # Convert to PIL Image
        img = Image.fromarray(img_array, mode="L")

        # Save with medium compression
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=self.compression_level)
        img_data = buffer.getvalue()

        metadata = BitmapMetadata(
            width=width,
            height=height,
            cell_width=cell_size,
            cell_height=cell_size,
            total_rows=rows,
            total_cols=cols,
            scale_factor=cell_size / self.cell_width,
            mode="compressed_4bit",
            compression=CompressionMode.COMPRESSED_4,
            region_bounds=region,
            file_size_bytes=len(img_data),
        )

        logger.info(
            f"Generated 4-bit compressed {width}x{height} bitmap for {rows}x{cols} region "
            f"(size: {len(img_data)/1024:.1f}KB)"
        )

        return img_data, metadata

    def _generate_sampled(
        self,
        sheet_data: SheetData,
        region: QuadTreeBounds | None,
        rows: int,
        cols: int,
    ) -> tuple[bytes, BitmapMetadata]:
        """Generate downsampled bitmap for very large regions."""
        start_row = region.min_row if region else 0
        start_col = region.min_col if region else 0

        # Calculate sampling rate to fit within reasonable size
        max_samples = 500  # Maximum cells in each dimension
        row_sample = max(1, rows // max_samples)
        col_sample = max(1, cols // max_samples)

        sampled_rows = (rows + row_sample - 1) // row_sample
        sampled_cols = (cols + col_sample - 1) // col_sample

        # Use tiny cell size
        cell_size = 2

        width = sampled_cols * cell_size
        height = sampled_rows * cell_size

        # Create image array
        img_array = np.full((height, width), 255, dtype=np.uint8)

        # Sample cells
        for i in range(sampled_rows):
            for j in range(sampled_cols):
                # Check if any cell in the sample region is filled
                has_data = False
                is_header = False

                for r in range(i * row_sample, min((i + 1) * row_sample, rows)):
                    for c in range(j * col_sample, min((j + 1) * col_sample, cols)):
                        cell = sheet_data.get_cell(start_row + r, start_col + c)
                        if cell and not cell.is_empty:
                            has_data = True
                            if cell.is_bold:
                                is_header = True
                                break
                    if is_header:
                        break

                if has_data:
                    y_start = i * cell_size
                    y_end = (i + 1) * cell_size
                    x_start = j * cell_size
                    x_end = (j + 1) * cell_size

                    gray = 0 if is_header else 128
                    img_array[y_start:y_end, x_start:x_end] = gray

        # Convert to PIL Image
        img = Image.fromarray(img_array, mode="L")

        # Save with high compression
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=9)
        img_data = buffer.getvalue()

        metadata = BitmapMetadata(
            width=width,
            height=height,
            cell_width=cell_size,
            cell_height=cell_size,
            total_rows=rows,
            total_cols=cols,
            scale_factor=1.0 / row_sample,  # Approximate scale
            mode="sampled",
            compression=CompressionMode.SAMPLED,
            region_bounds=region,
            file_size_bytes=len(img_data),
        )

        logger.info(
            f"Generated sampled {width}x{height} bitmap for {rows}x{cols} region "
            f"(sampling: {row_sample}x{col_sample}, size: {len(img_data)/1024:.1f}KB)"
        )

        return img_data, metadata

    def generate_from_visualization_plan(
        self, sheet_data: SheetData, regions: list[VisualizationRegion]
    ) -> list[BitmapResult]:
        """Generate bitmaps for a list of visualization regions.

        Args:
            sheet_data: Sheet data to visualize
            regions: List of regions from quadtree analysis

        Returns:
            List of bitmap results
        """
        results = []

        for region in regions:
            # Generate bitmap for this region
            image_data, metadata = self.generate(sheet_data, region.bounds)

            # Check size constraint
            if metadata.file_size_bytes > self.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(
                    f"Region bitmap exceeds size limit ({metadata.file_size_bytes / 1024 / 1024:.1f}MB), "
                    "regenerating with higher compression"
                )
                # Try with maximum compression
                old_level = self.compression_level
                self.compression_level = 9
                image_data, metadata = self.generate(sheet_data, region.bounds)
                self.compression_level = old_level

            compression_info = {
                "mode": metadata.compression.value,
                "original_cells": region.bounds.area,
                "image_dimensions": f"{metadata.width}x{metadata.height}",
                "file_size_mb": metadata.file_size_bytes / 1024 / 1024,
                "compression_ratio": region.bounds.area / (metadata.width * metadata.height),
            }

            results.append(
                BitmapResult(
                    image_data=image_data,
                    metadata=metadata,
                    compression_info=compression_info,
                )
            )

        return results

    def generate_analysis_bitmap(
        self,
        sheet_data: SheetData,
        bits_per_cell: int = 1,
        region: QuadTreeBounds | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Generate bitmap for local analysis without size constraints.

        This method creates full-resolution bitmaps for pattern detection,
        preserving the actual sheet dimensions for accurate analysis.

        Args:
            sheet_data: Sheet data to visualize
            bits_per_cell: Bits per cell (1=binary, 2=detailed)
            region: Optional region bounds (None for entire sheet)

        Returns:
            Tuple of (bitmap array, metadata dict)
        """
        # Determine bounds
        if region:
            rows = region.height
            cols = region.width
            start_row = region.min_row
            start_col = region.min_col
        else:
            # Use sheet data directly
            sheet = sheet_data
            rows = sheet.max_row + 1 if sheet.max_row >= 0 else 0
            cols = sheet.max_column + 1 if sheet.max_column >= 0 else 0
            start_row = 0
            start_col = 0

        if rows == 0 or cols == 0:
            return np.array([[]], dtype=np.uint8), {
                "rows": 0,
                "cols": 0,
                "bits_per_cell": bits_per_cell,
                "region": region,
            }

        # Create bitmap array
        bitmap = np.zeros((rows, cols), dtype=np.uint8)

        # Collect filled cells for vectorized operations
        if bits_per_cell == 1:
            # Simple binary: 0=empty, 1=filled
            filled_positions = []
            # Get non-empty cells
            for cell in sheet_data.get_non_empty_cells():
                # Check if cell is within region bounds
                if (
                    start_row <= cell.row < start_row + rows
                    and start_col <= cell.column < start_col + cols
                ):
                    filled_positions.append((cell.row - start_row, cell.column - start_col))

            # Vectorized assignment
            if filled_positions:
                positions = np.array(filled_positions)
                bitmap[positions[:, 0], positions[:, 1]] = 1

        elif bits_per_cell == 2:
            # 2-bit representation: collect by type
            value_positions = []
            formula_positions = []
            header_positions = []

            # Get non-empty cells
            for cell in sheet_data.get_non_empty_cells():
                # Check if cell is within region bounds
                if (
                    start_row <= cell.row < start_row + rows
                    and start_col <= cell.column < start_col + cols
                ):
                    local_row = cell.row - start_row
                    local_col = cell.column - start_col

                    if cell.is_bold:
                        header_positions.append((local_row, local_col))
                    elif cell.has_formula:
                        formula_positions.append((local_row, local_col))
                    else:
                        value_positions.append((local_row, local_col))

            # Vectorized assignments
            if value_positions:
                positions = np.array(value_positions)
                bitmap[positions[:, 0], positions[:, 1]] = 1

            if formula_positions:
                positions = np.array(formula_positions)
                bitmap[positions[:, 0], positions[:, 1]] = 2

            if header_positions:
                positions = np.array(header_positions)
                bitmap[positions[:, 0], positions[:, 1]] = 3
        else:
            raise ValueError(f"Unsupported bits_per_cell: {bits_per_cell}")

        # Calculate metadata
        filled_cells = np.sum(bitmap > 0)
        metadata = {
            "rows": rows,
            "cols": cols,
            "bits_per_cell": bits_per_cell,
            "filled_cells": int(filled_cells),
            "density": float(filled_cells) / (rows * cols) if rows * cols > 0 else 0.0,
            "region": region,
            "memory_bytes": bitmap.nbytes,
        }

        logger.info(
            f"Generated {rows}x{cols} analysis bitmap "
            f"({bits_per_cell}-bit, {metadata['memory_bytes']/1024:.1f}KB, "
            f"{metadata['density']:.1%} density)"
        )

        return bitmap, metadata

    def pixel_to_cell(self, x: int, y: int, metadata: BitmapMetadata) -> tuple[int, int]:
        """Convert pixel coordinates to cell row/column.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            metadata: Bitmap metadata

        Returns:
            Tuple of (row, column)
        """
        row = y // metadata.cell_height
        col = x // metadata.cell_width
        return row, col

    def cell_to_pixel(
        self, row: int, col: int, metadata: BitmapMetadata
    ) -> tuple[int, int, int, int]:
        """Convert cell coordinates to pixel bounding box.

        Args:
            row: Cell row index
            col: Cell column index
            metadata: Bitmap metadata

        Returns:
            Tuple of (x1, y1, x2, y2) pixel coordinates
        """
        x1 = col * metadata.cell_width
        y1 = row * metadata.cell_height
        x2 = (col + 1) * metadata.cell_width
        y2 = (row + 1) * metadata.cell_height
        return x1, y1, x2, y2

    def create_prompt_description(self, metadata: BitmapMetadata) -> str:
        """Create a clear description of the bitmap format for the LLM.

        Args:
            metadata: Bitmap metadata

        Returns:
            Description text to include in the prompt
        """
        # Determine cell description based on mode and compression
        if metadata.compression == CompressionMode.COMPRESSED_2:
            cell_desc = (
                "This is a 2-bit compressed view where:\n"
                "- White (255) = empty cells\n"
                "- Light gray (192) = filled cells\n"
                "- Dark gray (64) = header/bold cells\n"
                "- Black (0) = special cells (formulas/merged)"
            )
        elif metadata.compression == CompressionMode.COMPRESSED_4:
            cell_desc = (
                "This is a 4-bit compressed view with nuanced grayscale:\n"
                "- White (255) = empty cells\n"
                "- Black (0) = bold/header cells\n"
                "- Dark shades = formulas and merged cells\n"
                "- Medium shades = numeric data\n"
                "- Light shades = text data"
            )
        elif metadata.compression == CompressionMode.SAMPLED:
            cell_desc = (
                f"This is a heavily downsampled view (sampling rate ~{1/metadata.scale_factor:.0f}x).\n"
                "Each pixel represents multiple cells:\n"
                "- White = empty region\n"
                "- Black = header region\n"
                "- Gray = data region"
            )
        elif self.mode == "binary":
            cell_desc = "Black pixels represent filled cells, white pixels represent empty cells."
        elif self.mode == "grayscale":
            cell_desc = (
                "Black pixels (0) represent bold/header cells, "
                "dark gray (64) represents formula cells, "
                "medium gray (128) represents merged cells, "
                "light gray (192) represents regular filled cells, "
                "and white (255) represents empty cells."
            )
        else:  # color
            cell_desc = (
                "Black pixels represent bold/header cells, "
                "blue represents formula cells, "
                "orange represents merged cells, "
                "green represents numeric data, "
                "purple represents dates, "
                "dark gray represents text, "
                "and white represents empty cells."
            )

        # Add region info if available
        region_info = ""
        if metadata.region_bounds:
            region_info = (
                f"\nThis image shows a specific region of the spreadsheet: "
                f"rows {metadata.region_bounds.min_row}-{metadata.region_bounds.max_row}, "
                f"columns {metadata.region_bounds.min_col}-{metadata.region_bounds.max_col}."
            )

        return f"""This is a bitmap representation of a spreadsheet with {metadata.total_rows} rows and {metadata.total_cols} columns.
Each cell is represented by a {metadata.cell_width}x{metadata.cell_height} pixel rectangle.
{cell_desc}{region_info}

Please analyze this image and identify all distinct table regions. For each table found, provide:
1. The bounding box in pixel coordinates (x1, y1, x2, y2)
2. The estimated cell range (e.g., "A1:D10")
3. A confidence score (0-1)
4. Any notable characteristics (headers, merged cells, etc.)

Focus on rectangular regions of filled cells that form logical tables, considering:
- Natural boundaries formed by empty rows/columns
- Visual patterns suggesting headers (e.g., bold cells at the top)
- Groupings that appear to represent related data"""
