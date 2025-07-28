"""Tool for calculating optimal compression strategies."""

from typing import NamedTuple


class CompressionStrategy(NamedTuple):
    """Optimal compression strategy for a sheet."""

    compression_levels: list[int]
    estimated_size_mb: float
    strategy_name: str
    reasoning: str


class DataBounds(NamedTuple):
    """Actual data bounds in a sheet."""

    min_row: int
    max_row: int
    min_col: int
    max_col: int
    populated_cells: int
    sheet_utilization: float


def calculate_optimal_compression(
    data_bounds: DataBounds, target_size_mb: float = 20.0
) -> CompressionStrategy:
    """
    Calculate optimal compression levels based on actual data size.

    This function determines the best compression strategy to stay within
    size limits while preserving as much detail as possible.

    Args:
        data_bounds: Actual bounds of data in the sheet
        target_size_mb: Target total size in MB

    Returns:
        Optimal compression strategy
    """
    populated_cells = data_bounds.populated_cells

    # Estimate bytes per pixel (conservative estimate)
    bytes_per_pixel = 3  # RGB
    overhead_factor = 1.2  # PNG compression overhead

    # No compression needed for small sheets
    if populated_cells < 10_000:
        return CompressionStrategy(
            compression_levels=[0],  # No compression
            estimated_size_mb=populated_cells * bytes_per_pixel * overhead_factor / 1_000_000,
            strategy_name="single_full_resolution",
            reasoning="Sheet is small enough for full resolution",
        )

    # Single compressed image for medium sheets
    elif populated_cells < 100_000:
        # Calculate compression needed
        uncompressed_size_mb = populated_cells * bytes_per_pixel * overhead_factor / 1_000_000

        if uncompressed_size_mb < target_size_mb:
            compression = 0
        elif uncompressed_size_mb < target_size_mb * 16:
            compression = 1  # 16:1
        else:
            compression = 2  # 64:1

        return CompressionStrategy(
            compression_levels=[compression],
            estimated_size_mb=min(uncompressed_size_mb, target_size_mb),
            strategy_name="single_compressed",
            reasoning=f"Medium sheet compressed to level {compression}",
        )

    # Multi-scale for large sheets
    elif populated_cells < 1_000_000:
        # Overview + detail views
        overview_compression = 3  # 256:4
        detail_compression = 0  # Full resolution for key areas

        # Estimate we'll capture 20% of sheet in detail views
        detail_cells = populated_cells * 0.2
        overview_cells = populated_cells / (256 * 4)

        estimated_size = (
            (overview_cells + detail_cells) * bytes_per_pixel * overhead_factor / 1_000_000
        )

        return CompressionStrategy(
            compression_levels=[overview_compression, detail_compression],
            estimated_size_mb=min(estimated_size, target_size_mb),
            strategy_name="multi_scale",
            reasoning="Large sheet with overview and detail views",
        )

    # Progressive refinement for huge sheets
    else:
        # Three levels: maximum compression, medium, and detail
        return CompressionStrategy(
            compression_levels=[5, 3, 1],  # Maximum, large, mild
            estimated_size_mb=target_size_mb * 0.9,  # Use most of budget
            strategy_name="progressive_refinement",
            reasoning="Huge sheet requiring progressive analysis",
        )
