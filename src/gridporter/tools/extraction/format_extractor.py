"""Tool for extracting cell formatting information."""

from typing import Any


def extract_cell_formats(sheet_data: Any, table_range: Any) -> list[list[dict[str, Any]]]:
    """
    Extract formatting information for cells in a range.

    This includes colors, fonts, borders, and other visual properties
    that might need to be preserved.

    Args:
        sheet_data: Sheet containing the table
        table_range: Table boundaries

    Returns:
        2D list of format dictionaries
    """
    formats = []

    for row_idx in range(table_range.start_row, table_range.end_row + 1):
        row_formats = []

        for col_idx in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(row_idx, col_idx)

            if cell:
                cell_format = {
                    "bold": getattr(cell, "is_bold", False),
                    "italic": getattr(cell, "is_italic", False),
                    "underline": getattr(cell, "is_underline", False),
                    "font_size": getattr(cell, "font_size", None),
                    "font_color": getattr(cell, "font_color", None),
                    "background_color": getattr(cell, "background_color", None),
                    "horizontal_alignment": getattr(cell, "horizontal_alignment", None),
                    "vertical_alignment": getattr(cell, "vertical_alignment", None),
                    "number_format": getattr(cell, "number_format", None),
                    "has_border": getattr(cell, "has_border", False),
                }
            else:
                cell_format = {}

            row_formats.append(cell_format)

        formats.append(row_formats)

    return formats
