"""Analyze spreadsheet patterns using bitmap representations."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import filters

from ..models.sheet_data import SheetData

# Optional telemetry import
try:
    from ..telemetry.metrics import get_metrics_collector

    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)


class TableOrientation(Enum):
    """Orientation of a detected table."""

    HORIZONTAL = "horizontal"  # Headers on top (typical)
    VERTICAL = "vertical"  # Headers on left (transposed)
    MATRIX = "matrix"  # Headers on both axes
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class BitmapRegion:
    """A connected region in the bitmap."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    pixel_count: int
    label: int

    @property
    def width(self) -> int:
        """Width of the region in cells."""
        return self.col_end - self.col_start + 1

    @property
    def height(self) -> int:
        """Height of the region in cells."""
        return self.row_end - self.row_start + 1

    @property
    def area(self) -> int:
        """Total area of the bounding box."""
        return self.width * self.height

    @property
    def density(self) -> float:
        """Density of filled cells in the region."""
        return self.pixel_count / self.area if self.area > 0 else 0.0

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        return self.width / self.height if self.height > 0 else float("inf")

    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is within this region."""
        return self.row_start <= row <= self.row_end and self.col_start <= col <= self.col_end


class BitmapAnalyzer:
    """Analyze spreadsheet patterns using bitmap representations."""

    def __init__(
        self,
        min_region_size: tuple[int, int] = (2, 2),
        min_density: float = 0.1,
        header_density_threshold: float = 0.6,
    ):
        """Initialize bitmap analyzer.

        Args:
            min_region_size: Minimum (height, width) to consider as a table
            min_density: Minimum density to consider a region
            header_density_threshold: Density threshold for header detection
        """
        self.min_region_size = min_region_size
        self.min_density = min_density
        self.header_density_threshold = header_density_threshold

    def generate_binary_bitmap(self, sheet_data: SheetData) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate 1-bit bitmap at full resolution.

        Args:
            sheet_data: Sheet data to convert (SheetData or PolarsSheetData)

        Returns:
            Tuple of (bitmap array, metadata dict)
        """
        # Track timing if telemetry available
        if HAS_TELEMETRY:
            metrics = get_metrics_collector()
            start_time = time.time()

        # Use sheet data directly
        sheet = sheet_data
        # Get sheet dimensions
        rows = sheet.max_row + 1 if sheet.max_row >= 0 else 0
        cols = sheet.max_column + 1 if sheet.max_column >= 0 else 0

        if rows == 0 or cols == 0:
            return np.array([[]], dtype=np.uint8), {
                "rows": 0,
                "cols": 0,
                "filled_cells": 0,
            }

        # Create bitmap
        bitmap = np.zeros((rows, cols), dtype=np.uint8)

        # Efficient approach: collect all filled cells first
        filled_cells = []
        # Iterate through cells
        for _cell_ref, cell in sheet.cells.items():
            # Use cell's row/column attributes
            if not cell.is_empty and cell.row < rows and cell.column < cols:
                filled_cells.append((cell.row, cell.column))

        # Vectorized assignment
        if filled_cells:
            filled_array = np.array(filled_cells)
            bitmap[filled_array[:, 0], filled_array[:, 1]] = 1

        # Use NumPy for counting
        filled_count = np.sum(bitmap)

        metadata = {
            "rows": rows,
            "cols": cols,
            "filled_cells": int(filled_count),
            "density": float(filled_count) / (rows * cols) if rows * cols > 0 else 0.0,
        }

        logger.info(
            f"Generated {rows}x{cols} binary bitmap with {filled_count} filled cells "
            f"({metadata['density']:.1%} density)"
        )

        if HAS_TELEMETRY:
            duration = time.time() - start_time
            metrics.record_duration("bitmap_analysis.generate_binary", duration)
            metrics.record_value("bitmap_analysis.cells", rows * cols)
            metrics.record_value("bitmap_analysis.filled_cells", filled_count)

        return bitmap, metadata

    def generate_detailed_bitmap(self, sheet_data: SheetData) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate 2-bit bitmap for more detail.

        Cell values:
        - 0: Empty
        - 1: Value
        - 2: Formula
        - 3: Header (bold/styled)

        Args:
            sheet_data: Sheet data to convert (SheetData or PolarsSheetData)

        Returns:
            Tuple of (bitmap array, metadata dict)
        """
        # Track timing if telemetry available
        if HAS_TELEMETRY:
            metrics = get_metrics_collector()
            start_time = time.time()

        # Use sheet data directly
        sheet = sheet_data
        # Get sheet dimensions
        rows = sheet.max_row + 1 if sheet.max_row >= 0 else 0
        cols = sheet.max_column + 1 if sheet.max_column >= 0 else 0

        if rows == 0 or cols == 0:
            return np.array([[]], dtype=np.uint8), {"rows": 0, "cols": 0}

        # Create bitmap - default is 0 (empty)
        bitmap = np.zeros((rows, cols), dtype=np.uint8)

        # Collect cells by type for vectorized assignment
        value_cells = []
        formula_cells = []
        header_cells = []

        # Get non-empty cells
        for cell in sheet.get_non_empty_cells():
            # Use cell's row/column attributes
            if cell.row < rows and cell.column < cols:
                if cell.is_bold:
                    header_cells.append((cell.row, cell.column))
                elif cell.has_formula:
                    formula_cells.append((cell.row, cell.column))
                else:
                    value_cells.append((cell.row, cell.column))

        # Vectorized assignments
        if value_cells:
            value_array = np.array(value_cells)
            bitmap[value_array[:, 0], value_array[:, 1]] = 1

        if formula_cells:
            formula_array = np.array(formula_cells)
            bitmap[formula_array[:, 0], formula_array[:, 1]] = 2

        if header_cells:
            header_array = np.array(header_cells)
            bitmap[header_array[:, 0], header_array[:, 1]] = 3

        # Use NumPy to count each type
        unique, counts = np.unique(bitmap, return_counts=True)
        count_dict = dict(zip(unique, counts, strict=False))

        counts = {
            "empty": count_dict.get(0, 0),
            "value": count_dict.get(1, 0),
            "formula": count_dict.get(2, 0),
            "header": count_dict.get(3, 0),
        }

        metadata = {
            "rows": rows,
            "cols": cols,
            "counts": counts,
            "filled_cells": int(np.sum(bitmap > 0)),
        }

        if HAS_TELEMETRY:
            duration = time.time() - start_time
            metrics.record_duration("bitmap_analysis.generate_detailed", duration)
            metrics.record_value("bitmap_analysis.detailed_cells", rows * cols)

        return bitmap, metadata

    def detect_connected_regions(
        self, bitmap: np.ndarray, min_size: tuple[int, int] | None = None
    ) -> list[BitmapRegion]:
        """Find connected components (potential tables).

        Args:
            bitmap: Binary bitmap (1=filled, 0=empty)
            min_size: Minimum (height, width) for regions

        Returns:
            List of detected regions
        """
        if min_size is None:
            min_size = self.min_region_size

        # Find connected components
        labeled_array, num_features = ndimage.label(bitmap)

        regions = []

        # Analyze each component
        for label in range(1, num_features + 1):
            # Get coordinates of this component
            coords = np.argwhere(labeled_array == label)
            if len(coords) == 0:
                continue

            # Get bounding box
            row_min, col_min = coords.min(axis=0)
            row_max, col_max = coords.max(axis=0)

            # Create region
            region = BitmapRegion(
                row_start=row_min,
                row_end=row_max,
                col_start=col_min,
                col_end=col_max,
                pixel_count=len(coords),
                label=label,
            )

            # Check size and density constraints
            if (
                region.height >= min_size[0]
                and region.width >= min_size[1]
                and region.density >= self.min_density
            ):
                regions.append(region)

        logger.info(f"Found {len(regions)} regions from {num_features} connected components")
        return regions

    def detect_table_edges(self, bitmap: np.ndarray) -> np.ndarray:
        """Detect edges in bitmap using Sobel filters.

        Args:
            bitmap: Binary bitmap

        Returns:
            Edge magnitude map
        """
        # Convert to float for edge detection
        bitmap_float = bitmap.astype(np.float32)

        # Sobel edge detection
        edges_h = filters.sobel_h(bitmap_float)
        edges_v = filters.sobel_v(bitmap_float)

        # Magnitude
        edge_magnitude = np.sqrt(edges_h**2 + edges_v**2)

        return edge_magnitude

    def analyze_region_pattern(self, bitmap: np.ndarray, region: BitmapRegion) -> dict[str, Any]:
        """Analyze a region to determine pattern characteristics.

        Args:
            bitmap: Full bitmap or region bitmap
            region: Region to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        # Extract region from bitmap using NumPy slicing
        region_bitmap = bitmap[
            region.row_start : region.row_end + 1, region.col_start : region.col_end + 1
        ]

        # Vectorized density calculations
        row_density = np.mean(region_bitmap, axis=1)  # Already efficient
        col_density = np.mean(region_bitmap, axis=0)  # Already efficient

        # Use NumPy operations for analysis
        first_row_density = row_density[0] if row_density.size > 0 else 0.0
        first_col_density = col_density[0] if col_density.size > 0 else 0.0

        # Vectorized empty detection
        empty_rows = np.where(row_density == 0)[0]
        empty_cols = np.where(col_density == 0)[0]

        # Check for patterns using NumPy
        # Gradient analysis for detecting headers
        if len(row_density) > 1:
            row_density_gradient = np.diff(row_density)
            has_sharp_drop_after_first = (
                len(row_density_gradient) > 0 and row_density_gradient[0] < -0.3
            )
        else:
            has_sharp_drop_after_first = False

        analysis = {
            "region": region,
            "aspect_ratio": region.aspect_ratio,
            "density": region.density,
            "first_row_density": float(first_row_density),
            "first_col_density": float(first_col_density),
            "row_density_std": float(np.std(row_density)),
            "col_density_std": float(np.std(col_density)),
            "has_dense_first_row": first_row_density >= self.header_density_threshold,
            "has_dense_first_col": first_col_density >= self.header_density_threshold,
            "empty_rows": empty_rows.tolist(),
            "empty_cols": empty_cols.tolist(),
            "has_sharp_drop_after_first": has_sharp_drop_after_first,
            "avg_row_density": float(np.mean(row_density)),
            "avg_col_density": float(np.mean(col_density)),
        }

        return analysis

    def detect_orientation(self, analysis: dict[str, Any]) -> TableOrientation:
        """Detect if table is horizontal or vertical based on analysis.

        Args:
            analysis: Region analysis from analyze_region_pattern

        Returns:
            Detected orientation
        """
        aspect_ratio = analysis["aspect_ratio"]
        has_dense_first_row = analysis["has_dense_first_row"]
        has_dense_first_col = analysis["has_dense_first_col"]

        # Matrix pattern: dense first row AND column
        if has_dense_first_row and has_dense_first_col:
            return TableOrientation.MATRIX

        # Horizontal: dense first row + wide aspect ratio
        if has_dense_first_row and aspect_ratio > 1.2:
            return TableOrientation.HORIZONTAL

        # Vertical: dense first column + tall aspect ratio
        if has_dense_first_col and aspect_ratio < 0.8:
            return TableOrientation.VERTICAL

        # Ambiguous cases
        if has_dense_first_row:
            return TableOrientation.HORIZONTAL
        elif has_dense_first_col:
            return TableOrientation.VERTICAL

        # Default based on aspect ratio
        if aspect_ratio > 1.5:
            return TableOrientation.HORIZONTAL
        elif aspect_ratio < 0.67:
            return TableOrientation.VERTICAL
        else:
            return TableOrientation.UNKNOWN

    def classify_pattern_type(self, analysis: dict[str, Any], orientation: TableOrientation) -> str:
        """Classify the pattern type based on density distribution.

        Args:
            analysis: Region analysis
            orientation: Detected orientation

        Returns:
            Pattern type string
        """
        region = analysis["region"]

        # Get density characteristics
        row_density_std = analysis["row_density_std"]
        col_density_std = analysis["col_density_std"]

        if orientation == TableOrientation.MATRIX:
            return "matrix"

        elif orientation == TableOrientation.HORIZONTAL:
            # High first row density + lower data density = header-data
            if analysis["has_dense_first_row"] and region.density < 0.6:
                return "header_data"
            # Uniform density = simple table
            elif row_density_std < 0.2:
                return "simple_table"
            # Regular gaps might indicate time series
            elif len(analysis["empty_rows"]) > region.height * 0.2:
                return "time_series"
            else:
                return "header_data"

        elif orientation == TableOrientation.VERTICAL:
            # Dense first column = form/checklist pattern
            if analysis["has_dense_first_col"] and col_density_std > 0.3:
                return "form"
            else:
                return "header_data"

        else:
            return "simple_table"

    def find_table_regions(self, sheet_data: SheetData) -> list[dict[str, Any]]:
        """Find and analyze all table regions in a sheet.

        Args:
            sheet_data: Sheet to analyze

        Returns:
            List of analyzed regions with pattern information
        """
        # Generate bitmap
        bitmap, metadata = self.generate_binary_bitmap(sheet_data)

        if metadata["filled_cells"] == 0:
            return []

        # Use connected component approach
        regions = self.detect_connected_regions(bitmap)

        analyzed_regions = []
        for region in regions:
            analysis = self.analyze_region_pattern(bitmap, region)
            orientation = self.detect_orientation(analysis)
            pattern_type = self.classify_pattern_type(analysis, orientation)

            analysis["orientation"] = orientation
            analysis["pattern_type"] = pattern_type
            analyzed_regions.append(analysis)

        return analyzed_regions

    def visualize_regions(self, bitmap: np.ndarray, regions: list[BitmapRegion]) -> np.ndarray:
        """Create a visualization of detected regions (for debugging).

        Args:
            bitmap: Original bitmap
            regions: Detected regions

        Returns:
            Labeled array where each region has a unique value
        """
        labeled = np.zeros_like(bitmap, dtype=np.int32)

        for i, region in enumerate(regions, 1):
            labeled[
                region.row_start : region.row_end + 1,
                region.col_start : region.col_end + 1,
            ] = i

        return labeled

    def find_table_regions_with_edges(self, sheet_data: SheetData) -> list[dict[str, Any]]:
        """Find table regions using edge detection approach.

        Args:
            sheet_data: Sheet to analyze

        Returns:
            List of detected regions with edge information
        """
        # Generate bitmap
        bitmap, metadata = self.generate_binary_bitmap(sheet_data)

        if metadata["filled_cells"] == 0:
            return []

        # Detect edges
        edge_map = self.detect_table_edges(bitmap)

        # Threshold edges
        edge_threshold = np.mean(edge_map) + np.std(edge_map)
        strong_edges = edge_map > edge_threshold

        # Find regions bounded by edges
        # Fill regions between edges
        filled = ndimage.binary_fill_holes(strong_edges)

        # Label regions
        labeled, num_features = ndimage.label(filled)

        analyzed_regions = []
        for label in range(1, num_features + 1):
            # Get region mask
            region_mask = labeled == label

            # Find bounds
            rows, cols = np.where(region_mask)
            if len(rows) == 0:
                continue

            region = BitmapRegion(
                row_start=rows.min(),
                row_end=rows.max(),
                col_start=cols.min(),
                col_end=cols.max(),
                pixel_count=len(rows),
                label=label,
            )

            # Check size constraints
            if (
                region.height >= self.min_region_size[0]
                and region.width >= self.min_region_size[1]
                and region.density >= self.min_density
            ):
                analysis = self.analyze_region_pattern(bitmap, region)
                orientation = self.detect_orientation(analysis)
                pattern_type = self.classify_pattern_type(analysis, orientation)

                analysis["orientation"] = orientation
                analysis["pattern_type"] = pattern_type
                analysis["detection_method"] = "edge_based"

                analyzed_regions.append(analysis)

        return analyzed_regions
