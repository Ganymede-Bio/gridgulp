"""Tool for preprocessing data regions in spreadsheets."""

from typing import Any, NamedTuple


class DataRegionSummary(NamedTuple):
    """Summary of data regions in a sheet."""

    data_regions: list
    sheet_utilization: float
    data_density: float
    disconnected_regions: int
    has_mixed_patterns: bool
    total_cells: int
    empty_regions: list


def preprocess_data_regions(sheet_data: Any, min_cells: int = 4) -> DataRegionSummary:
    """
    Fast scan to identify regions with actual data.

    This tool provides a quick overview of data distribution without
    deep analysis, helping agents decide on detection strategies.

    Args:
        sheet_data: Sheet data to analyze
        min_cells: Minimum cells for a region to be considered

    Returns:
        Summary of data regions found
    """
    from .connected_components import find_connected_components

    # Find all data regions
    regions = find_connected_components(sheet_data, gap_threshold=5, min_region_size=min_cells)

    # Calculate sheet utilization
    total_possible_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
    total_data_cells = sum(r.cell_count for r in regions)
    sheet_utilization = total_data_cells / total_possible_cells if total_possible_cells > 0 else 0

    # Calculate average density
    avg_density = sum(r.density for r in regions) / len(regions) if regions else 0

    # Check for mixed patterns
    has_text_regions = any(r.characteristics.get("mostly_text", False) for r in regions)
    has_number_regions = any(r.characteristics.get("mostly_numbers", False) for r in regions)
    has_mixed_patterns = has_text_regions and has_number_regions

    # Find empty regions (simplified - just major gaps)
    empty_regions = _find_major_empty_regions(sheet_data, regions)

    return DataRegionSummary(
        data_regions=regions,
        sheet_utilization=sheet_utilization,
        data_density=avg_density,
        disconnected_regions=len(regions),
        has_mixed_patterns=has_mixed_patterns,
        total_cells=total_data_cells,
        empty_regions=empty_regions,
    )


def _find_major_empty_regions(sheet_data: Any, data_regions: list) -> list:
    """Find major empty regions between data regions."""
    if not data_regions:
        return []

    empty_regions = []

    # Sort regions by position
    sorted_regions = sorted(data_regions, key=lambda r: (r.bounds["top"], r.bounds["left"]))

    # Check vertical gaps
    for i in range(len(sorted_regions) - 1):
        current = sorted_regions[i]
        next_region = sorted_regions[i + 1]

        gap = next_region.bounds["top"] - current.bounds["bottom"] - 1
        if gap > 10:  # Significant gap
            empty_regions.append({"type": "vertical_gap", "size": gap, "after_region": i})

    return empty_regions
