"""Fast pre-processing to detect data regions in spreadsheets."""

import logging
import time

import numpy as np
from scipy import ndimage

from ..models.multi_scale import DataRegion
from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


class DataRegionPreprocessor:
    """Pre-process spreadsheets to quickly identify regions containing data.

    This preprocessor performs a fast scan to find all regions with data,
    enabling optimization of bitmap generation and vision analysis.
    Never skips regions with data.
    """

    def __init__(self, gap_threshold: int = 5, min_region_size: int = 4):
        """Initialize the preprocessor.

        Args:
            gap_threshold: Maximum gap between cells to consider them part of same region
            min_region_size: Minimum cells for a valid region (to filter noise)
        """
        self.gap_threshold = gap_threshold
        self.min_region_size = min_region_size

    def detect_data_regions(self, sheet_data: SheetData) -> list[DataRegion]:
        """Detect all regions containing data in the sheet.

        Args:
            sheet_data: The sheet data to analyze

        Returns:
            List of data regions found in the sheet
        """
        start_time = time.time()

        # Create binary mask of data presence
        has_data = self._create_data_mask(sheet_data)

        if not has_data.any():
            logger.info("No data found in sheet")
            return []

        # Find connected components
        raw_regions = self._find_connected_regions(has_data)

        # Merge nearby regions
        merged_regions = self._merge_nearby_regions(raw_regions)

        # Calculate metrics for each region
        final_regions = []
        for idx, region in enumerate(merged_regions):
            data_region = self._analyze_region(sheet_data, region, idx)
            final_regions.append(data_region)

        # Also identify empty regions for context
        empty_regions = self._find_empty_regions(has_data, final_regions)

        elapsed = time.time() - start_time
        logger.info(
            f"Found {len(final_regions)} data regions and {len(empty_regions)} empty regions "
            f"in {elapsed:.2f}s"
        )

        return final_regions

    def _create_data_mask(self, sheet_data: SheetData) -> np.ndarray:
        """Create binary mask where True indicates cell has data.

        Args:
            sheet_data: The sheet data

        Returns:
            Boolean numpy array of shape (max_row+1, max_col+1)
        """
        if sheet_data.max_row < 0 or sheet_data.max_column < 0:
            return np.array([[]], dtype=bool)

        # Create mask
        mask = np.zeros((sheet_data.max_row + 1, sheet_data.max_column + 1), dtype=bool)

        # Mark cells with data
        for row_idx in range(sheet_data.max_row + 1):
            for col_idx in range(sheet_data.max_column + 1):
                cell = sheet_data.get_cell(row_idx, col_idx)
                if cell and not cell.is_empty:
                    mask[row_idx, col_idx] = True

        return mask

    def _find_connected_regions(self, mask: np.ndarray) -> list[dict[str, int]]:
        """Find connected components in the data mask.

        Args:
            mask: Binary mask of data presence

        Returns:
            List of region bounds
        """
        # Dilate to connect nearby cells
        if self.gap_threshold > 0:
            structure = np.ones((self.gap_threshold * 2 + 1, self.gap_threshold * 2 + 1))
            dilated = ndimage.binary_dilation(mask, structure=structure)
        else:
            dilated = mask

        # Find connected components
        labeled, num_features = ndimage.label(dilated)

        # Extract region bounds
        regions = []
        for label_id in range(1, num_features + 1):
            rows, cols = np.where(labeled == label_id)

            if len(rows) < self.min_region_size:
                continue

            # Use original mask to get actual data bounds (not dilated)
            data_rows, data_cols = np.where(mask & (labeled == label_id))

            if len(data_rows) == 0:
                continue

            regions.append(
                {
                    "top": int(data_rows.min()),
                    "left": int(data_cols.min()),
                    "bottom": int(data_rows.max()),
                    "right": int(data_cols.max()),
                }
            )

        return regions

    def _merge_nearby_regions(self, regions: list[dict[str, int]]) -> list[dict[str, int]]:
        """Merge regions that are close to each other.

        Args:
            regions: List of region bounds

        Returns:
            List of merged region bounds
        """
        if len(regions) <= 1:
            return regions

        # Sort by top-left corner
        regions.sort(key=lambda r: (r["top"], r["left"]))

        merged = []
        current = regions[0].copy()

        for region in regions[1:]:
            # Check if regions overlap or are nearby
            if self._should_merge(current, region):
                # Expand current region
                current["top"] = min(current["top"], region["top"])
                current["left"] = min(current["left"], region["left"])
                current["bottom"] = max(current["bottom"], region["bottom"])
                current["right"] = max(current["right"], region["right"])
            else:
                # Save current and start new
                merged.append(current)
                current = region.copy()

        merged.append(current)
        return merged

    def _should_merge(self, region1: dict[str, int], region2: dict[str, int]) -> bool:
        """Check if two regions should be merged.

        Args:
            region1: First region bounds
            region2: Second region bounds

        Returns:
            True if regions should be merged
        """
        # Check vertical gap
        v_gap = max(0, region2["top"] - region1["bottom"] - 1)
        if v_gap > self.gap_threshold * 2:
            return False

        # Check horizontal gap
        h_gap = max(
            0, min(region2["left"] - region1["right"] - 1, region1["left"] - region2["right"] - 1)
        )
        if h_gap > self.gap_threshold * 2:
            return False

        # Check for overlap
        h_overlap = min(region1["right"], region2["right"]) >= max(region1["left"], region2["left"])
        v_overlap = min(region1["bottom"], region2["bottom"]) >= max(region1["top"], region2["top"])

        return h_overlap and v_overlap

    def _analyze_region(
        self, sheet_data: SheetData, bounds: dict[str, int], idx: int
    ) -> DataRegion:
        """Analyze a region to extract characteristics.

        Args:
            sheet_data: The sheet data
            bounds: Region bounds
            idx: Region index

        Returns:
            DataRegion with analysis results
        """
        # Count cells and calculate density
        cell_count = 0
        has_text = False
        has_numbers = False
        has_formatting = False
        likely_headers = False

        total_cells = (bounds["bottom"] - bounds["top"] + 1) * (
            bounds["right"] - bounds["left"] + 1
        )

        # Check first few rows for header patterns
        for row in range(bounds["top"], min(bounds["top"] + 3, bounds["bottom"] + 1)):
            for col in range(bounds["left"], bounds["right"] + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    cell_count += 1

                    if cell.data_type == "text":
                        has_text = True
                    elif cell.data_type in ("number", "currency", "percentage"):
                        has_numbers = True

                    if cell.is_bold or cell.background_color:
                        has_formatting = True
                        if row == bounds["top"]:
                            likely_headers = True

        # Sample the rest of the region
        if bounds["bottom"] - bounds["top"] > 10:
            sample_rows = np.linspace(
                bounds["top"] + 3,
                bounds["bottom"],
                min(20, bounds["bottom"] - bounds["top"] - 3),
                dtype=int,
            )
            for row in sample_rows:
                for col in range(bounds["left"], bounds["right"] + 1):
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        cell_count += 1

                        if cell.data_type == "text":
                            has_text = True
                        elif cell.data_type in ("number", "currency", "percentage"):
                            has_numbers = True
        else:
            # Small region, check all cells
            for row in range(bounds["top"] + 3, bounds["bottom"] + 1):
                for col in range(bounds["left"], bounds["right"] + 1):
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        cell_count += 1

                        if cell.data_type == "text":
                            has_text = True
                        elif cell.data_type in ("number", "currency", "percentage"):
                            has_numbers = True

        density = cell_count / total_cells if total_cells > 0 else 0

        characteristics = {
            "likely_headers": likely_headers,
            "mostly_text": has_text and not has_numbers,
            "mostly_numbers": has_numbers and not has_text,
            "mixed_types": has_text and has_numbers,
            "has_formatting": has_formatting,
            "likely_data": has_numbers or (has_text and density > 0.3),
        }

        return DataRegion(
            region_id=f"region_{idx + 1}",
            bounds=bounds,
            cell_count=cell_count,
            density=density,
            characteristics=characteristics,
            skip=False,  # Never skip regions with data
        )

    def _find_empty_regions(
        self, mask: np.ndarray, data_regions: list[DataRegion]
    ) -> list[dict[str, int]]:
        """Identify significant empty regions for context.

        Args:
            mask: Binary mask of data presence
            data_regions: List of detected data regions

        Returns:
            List of empty region bounds
        """
        if len(data_regions) == 0:
            return []

        # Find overall data bounds
        min_row = min(r.bounds["top"] for r in data_regions)
        max_row = max(r.bounds["bottom"] for r in data_regions)
        min_col = min(r.bounds["left"] for r in data_regions)
        max_col = max(r.bounds["right"] for r in data_regions)

        empty_regions = []

        # Check for large empty area to the right
        if max_col < mask.shape[1] - 1:
            empty_regions.append(
                {
                    "bounds": {
                        "top": min_row,
                        "left": max_col + 1,
                        "bottom": max_row,
                        "right": mask.shape[1] - 1,
                    }
                }
            )

        # Check for large empty area below
        if max_row < mask.shape[0] - 1:
            empty_regions.append(
                {
                    "bounds": {
                        "top": max_row + 1,
                        "left": min_col,
                        "bottom": mask.shape[0] - 1,
                        "right": max_col,
                    }
                }
            )

        return empty_regions
