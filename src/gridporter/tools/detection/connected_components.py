"""Tool for finding connected components in spreadsheet data."""

from typing import Any

import numpy as np
from scipy import ndimage


class DataRegion:
    """Represents a connected region of data."""

    def __init__(
        self,
        bounds: dict[str, int],
        cell_count: int,
        density: float,
        characteristics: dict[str, Any],
    ):
        self.bounds = bounds
        self.cell_count = cell_count
        self.density = density
        self.characteristics = characteristics


def find_connected_components(
    sheet_data: Any, gap_threshold: int = 3, min_region_size: int = 4
) -> list[DataRegion]:
    """
    Find connected regions of data using flood fill algorithm.

    Args:
        sheet_data: Sheet data object with cell information
        gap_threshold: Maximum gap between cells to consider connected
        min_region_size: Minimum cells for a valid region

    Returns:
        List of connected data regions
    """
    # Create binary mask of data presence
    mask = _create_data_mask(sheet_data)

    if not mask.any():
        return []

    # Dilate mask to connect nearby cells
    if gap_threshold > 0:
        structure = np.ones((gap_threshold * 2 + 1, gap_threshold * 2 + 1))
        dilated_mask = ndimage.binary_dilation(mask, structure=structure)
    else:
        dilated_mask = mask

    # Find connected components
    labeled, num_components = ndimage.label(dilated_mask)

    regions = []
    for label_id in range(1, num_components + 1):
        # Get original cells in this component
        component_mask = labeled == label_id
        original_cells = mask & component_mask

        if original_cells.sum() < min_region_size:
            continue

        # Find bounds
        rows, cols = np.where(original_cells)
        bounds = {
            "top": int(rows.min()),
            "left": int(cols.min()),
            "bottom": int(rows.max()),
            "right": int(cols.max()),
        }

        # Calculate metrics
        area = (bounds["bottom"] - bounds["top"] + 1) * (bounds["right"] - bounds["left"] + 1)
        cell_count = int(original_cells.sum())
        density = cell_count / area if area > 0 else 0

        # Analyze characteristics
        characteristics = _analyze_region_characteristics(sheet_data, bounds, original_cells)

        region = DataRegion(
            bounds=bounds, cell_count=cell_count, density=density, characteristics=characteristics
        )
        regions.append(region)

    return regions


def _create_data_mask(sheet_data: Any) -> np.ndarray:
    """Create binary mask where True indicates cell has data."""
    if sheet_data.max_row < 0 or sheet_data.max_column < 0:
        return np.array([[]], dtype=bool)

    mask = np.zeros((sheet_data.max_row + 1, sheet_data.max_column + 1), dtype=bool)

    for row_idx in range(sheet_data.max_row + 1):
        for col_idx in range(sheet_data.max_column + 1):
            cell = sheet_data.get_cell(row_idx, col_idx)
            if cell and hasattr(cell, "value") and cell.value is not None:
                mask[row_idx, col_idx] = True

    return mask


def _analyze_region_characteristics(
    sheet_data: Any, bounds: dict[str, int], cell_mask: np.ndarray
) -> dict[str, Any]:
    """Analyze characteristics of a region."""
    characteristics = {
        "likely_headers": False,
        "mostly_text": False,
        "mostly_numbers": False,
        "has_formatting": False,
        "mixed_types": False,
        "rectangularness": 0.0,
    }

    # Check first row for headers
    first_row_cells = []
    for col in range(bounds["left"], bounds["right"] + 1):
        if cell_mask[bounds["top"], col]:
            cell = sheet_data.get_cell(bounds["top"], col)
            if cell:
                first_row_cells.append(cell)

    # Header detection heuristics
    if first_row_cells:
        text_count = sum(
            1 for c in first_row_cells if hasattr(c, "data_type") and c.data_type == "text"
        )
        bold_count = sum(1 for c in first_row_cells if hasattr(c, "is_bold") and c.is_bold)

        if text_count > len(first_row_cells) * 0.8 or bold_count > len(first_row_cells) * 0.5:
            characteristics["likely_headers"] = True

    # Data type analysis
    text_cells = 0
    number_cells = 0
    formatted_cells = 0
    total_cells = 0

    for row in range(bounds["top"], bounds["bottom"] + 1):
        for col in range(bounds["left"], bounds["right"] + 1):
            if cell_mask[row, col]:
                cell = sheet_data.get_cell(row, col)
                if cell:
                    total_cells += 1
                    if hasattr(cell, "data_type"):
                        if cell.data_type == "text":
                            text_cells += 1
                        elif cell.data_type in ["number", "float", "int"]:
                            number_cells += 1
                    if hasattr(cell, "background_color") and cell.background_color:
                        formatted_cells += 1

    if total_cells > 0:
        text_ratio = text_cells / total_cells
        number_ratio = number_cells / total_cells

        characteristics["mostly_text"] = text_ratio > 0.7
        characteristics["mostly_numbers"] = number_ratio > 0.7
        characteristics["mixed_types"] = 0.3 < text_ratio < 0.7
        characteristics["has_formatting"] = formatted_cells > 0

    # Calculate rectangularness
    filled_cells = cell_mask[
        bounds["top"] : bounds["bottom"] + 1, bounds["left"] : bounds["right"] + 1
    ].sum()
    total_area = (bounds["bottom"] - bounds["top"] + 1) * (bounds["right"] - bounds["left"] + 1)
    characteristics["rectangularness"] = filled_cells / total_area if total_area > 0 else 0

    return characteristics
