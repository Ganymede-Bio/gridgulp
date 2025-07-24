"""Generate bitmap representations of spreadsheet data for vision model analysis."""

import io
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        cell_width: int = DEFAULT_CELL_WIDTH,
        cell_height: int = DEFAULT_CELL_HEIGHT,
        mode: Literal["binary", "grayscale", "color"] = "binary",
    ):
        """Initialize bitmap generator.

        Args:
            cell_width: Width of each cell in pixels
            cell_height: Height of each cell in pixels
            mode: Visualization mode - binary (filled/empty), grayscale (with formatting hints), or color
        """
        self.cell_width = max(cell_width, self.MIN_CELL_SIZE)
        self.cell_height = max(cell_height, self.MIN_CELL_SIZE)
        self.mode = mode

    def generate(self, sheet_data: SheetData) -> tuple[bytes, BitmapMetadata]:
        """Generate bitmap from sheet data.

        Args:
            sheet_data: Sheet data to visualize

        Returns:
            Tuple of (PNG image bytes, metadata)
        """
        # Calculate dimensions
        rows = sheet_data.max_row + 1 if sheet_data.max_row >= 0 else 0
        cols = sheet_data.max_column + 1 if sheet_data.max_column >= 0 else 0

        if rows == 0 or cols == 0:
            # Empty sheet - return minimal white image
            img = Image.new("L", (10, 10), 255)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", compress_level=0)  # No compression for lossless
            return buffer.getvalue(), BitmapMetadata(
                width=10,
                height=10,
                cell_width=10,
                cell_height=10,
                total_rows=0,
                total_cols=0,
                scale_factor=1.0,
                mode=self.mode,
            )

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
                cell = sheet_data.get_cell(row, col)
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

        # Save as PNG (lossless)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=0)  # No compression for perfect lossless

        metadata = BitmapMetadata(
            width=width,
            height=height,
            cell_width=actual_cell_width,
            cell_height=actual_cell_height,
            total_rows=rows,
            total_cols=cols,
            scale_factor=scale_factor,
            mode=self.mode,
        )

        logger.info(
            f"Generated {width}x{height} bitmap for {rows}x{cols} sheet "
            f"(cell size: {actual_cell_width}x{actual_cell_height})"
        )

        return buffer.getvalue(), metadata

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
        if self.mode == "binary":
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

        return f"""This is a bitmap representation of a spreadsheet with {metadata.total_rows} rows and {metadata.total_cols} columns.
Each cell is represented by a {metadata.cell_width}x{metadata.cell_height} pixel rectangle.
{cell_desc}

Please analyze this image and identify all distinct table regions. For each table found, provide:
1. The bounding box in pixel coordinates (x1, y1, x2, y2)
2. The estimated cell range (e.g., "A1:D10")
3. A confidence score (0-1)
4. Any notable characteristics (headers, merged cells, etc.)

Focus on rectangular regions of filled cells that form logical tables, considering:
- Natural boundaries formed by empty rows/columns
- Visual patterns suggesting headers (e.g., bold cells at the top)
- Groupings that appear to represent related data"""
