"""Tool for generating multi-scale bitmaps from sheet data."""

from typing import Any

from gridporter.models.multi_scale import MultiScaleBitmaps, VisionImage


def generate_multi_scale_bitmaps(
    sheet_data: Any, regions: list[dict[str, Any]], compression_levels: list[int]
) -> MultiScaleBitmaps:
    """
    Generate bitmaps at multiple compression levels.

    This tool creates visual representations of the spreadsheet at different
    scales, optimized for vision model analysis.

    Args:
        sheet_data: The sheet data to visualize
        regions: Data regions to focus on
        compression_levels: List of compression levels to generate

    Returns:
        MultiScaleBitmaps object with generated images
    """
    images = []

    # This would integrate with the existing BitmapGenerator
    # For now, creating placeholder
    for i, level in enumerate(compression_levels):
        if level == 0:
            image_id = f"detail_{i}"
            description = "Full resolution (1 pixel = 1 cell)"
        else:
            compression_blocks = _get_compression_blocks(level)
            image_id = f"compressed_L{level}"
            description = f"Compressed {compression_blocks[0]}:{compression_blocks[1]}"

        image = VisionImage(
            image_id=image_id,
            image_data="<base64_placeholder>",
            compression_level=level,
            block_size=_get_compression_blocks(level),
            description=description,
            purpose="table_detection",
            covers_cells="A1:Z100",  # Would be calculated
            size_bytes=100000,  # Would be actual size
        )
        images.append(image)

    return MultiScaleBitmaps(images=images)


def _get_compression_blocks(level: int) -> list[int]:
    """Get compression block sizes for a level."""
    compression_map = {0: [1, 1], 1: [16, 1], 2: [64, 1], 3: [256, 4], 4: [1024, 16], 5: [4096, 64]}
    return compression_map.get(level, [1, 1])
