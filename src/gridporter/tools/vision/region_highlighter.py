"""Tool for highlighting regions with context in images."""

from PIL import Image, ImageDraw


def highlight_region_with_context(
    image: Image.Image, region: tuple[int, int, int, int], context_cells: int = 10
) -> Image.Image:
    """
    Add visual indicators for region and context.

    This tool adds visual highlighting to show:
    - The proposed table region (red border)
    - Context cells around it (yellow highlight)

    Args:
        image: PIL Image to modify
        region: (x1, y1, x2, y2) coordinates
        context_cells: Number of context cells to show

    Returns:
        Modified image with highlights
    """
    # Create a copy to avoid modifying original
    highlighted = image.copy()
    draw = ImageDraw.Draw(highlighted)

    x1, y1, x2, y2 = region

    # Draw context area (yellow, semi-transparent)
    context_bounds = (
        max(0, x1 - context_cells),
        max(0, y1 - context_cells),
        min(image.width, x2 + context_cells),
        min(image.height, y2 + context_cells),
    )

    # Draw context rectangle
    draw.rectangle(
        context_bounds,
        outline=(255, 255, 0, 128),  # Yellow
        width=2,
    )

    # Draw main region (red, solid)
    draw.rectangle(
        region,
        outline=(255, 0, 0, 255),  # Red
        width=3,
    )

    # Add labels
    draw.text((x1, y1 - 20), "Proposed Table", fill=(255, 0, 0, 255))

    draw.text((context_bounds[0], context_bounds[1] - 20), "Context Area", fill=(255, 255, 0, 255))

    return highlighted
