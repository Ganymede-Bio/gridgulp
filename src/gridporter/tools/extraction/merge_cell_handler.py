"""Tool for handling merged cells in spreadsheets."""

from typing import Any


def resolve_merged_cells(sheet_data: Any, table_range: Any) -> list[dict[str, Any]]:
    """
    Find and resolve merged cells within a table range.

    Merged cells can affect table structure and need special handling
    during extraction.

    Args:
        sheet_data: Sheet containing the table
        table_range: Table boundaries

    Returns:
        List of merged cell regions with their properties
    """
    merged_regions = []

    # Check if sheet has merged cells info
    if hasattr(sheet_data, "merged_cells"):
        for merged_range in sheet_data.merged_cells:
            # Parse merged range (e.g., "A1:B3")
            try:
                start_cell, end_cell = merged_range.split(":")
                start_row, start_col = _parse_cell_reference(start_cell)
                end_row, end_col = _parse_cell_reference(end_cell)

                # Check if merged region overlaps with table
                if _ranges_overlap(
                    start_row,
                    start_col,
                    end_row,
                    end_col,
                    table_range.start_row,
                    table_range.start_col,
                    table_range.end_row,
                    table_range.end_col,
                ):
                    merged_regions.append(
                        {
                            "start_row": start_row,
                            "start_col": start_col,
                            "end_row": end_row,
                            "end_col": end_col,
                            "range": merged_range,
                        }
                    )
            except (ValueError, AttributeError):
                continue

    return merged_regions


def _parse_cell_reference(cell_ref: str) -> tuple[int, int]:
    """Parse Excel cell reference to row, col indices."""
    import re

    match = re.match(r"([A-Z]+)(\d+)", cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")

    col_str, row_str = match.groups()

    # Convert column letters to index
    col = 0
    for char in col_str:
        col = col * 26 + (ord(char) - ord("A") + 1)
    col -= 1  # 0-indexed

    row = int(row_str) - 1  # 0-indexed

    return row, col


def _ranges_overlap(
    r1_start_row: int,
    r1_start_col: int,
    r1_end_row: int,
    r1_end_col: int,
    r2_start_row: int,
    r2_start_col: int,
    r2_end_row: int,
    r2_end_col: int,
) -> bool:
    """Check if two ranges overlap."""
    return not (
        r1_end_row < r2_start_row
        or r1_start_row > r2_end_row
        or r1_end_col < r2_start_col
        or r1_start_col > r2_end_col
    )
