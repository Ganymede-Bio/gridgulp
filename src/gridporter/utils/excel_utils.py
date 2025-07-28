"""Excel utility functions for working with cell references and ranges."""

import re


def get_column_letter(col_index: int) -> str:
    """Convert 0-based column index to Excel column letter.

    Args:
        col_index: 0-based column index (0 = A, 1 = B, etc.)

    Returns:
        Excel column letter (A, B, ..., Z, AA, AB, ...)
    """
    result = ""
    col_index += 1  # Convert to 1-based for Excel

    while col_index > 0:
        col_index, remainder = divmod(col_index - 1, 26)
        result = chr(65 + remainder) + result

    return result


def column_letter_to_index(column: str) -> int:
    """Convert Excel column letter to 0-based index.

    Args:
        column: Excel column letter (A, B, ..., Z, AA, AB, ...)

    Returns:
        0-based column index (A = 0, B = 1, etc.)
    """
    result = 0
    for char in column.upper():
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1  # Convert to 0-based


def cell_to_indices(cell_ref: str) -> tuple[int, int]:
    """Convert Excel cell reference to 0-based row and column indices.

    Args:
        cell_ref: Excel cell reference (e.g., "A1", "AB123")

    Returns:
        Tuple of (row_index, col_index) in 0-based indexing

    Raises:
        ValueError: If cell reference is invalid
    """
    match = re.match(r"^([A-Z]+)(\d+)$", cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")

    col_letters, row_num = match.groups()
    col_index = column_letter_to_index(col_letters)
    row_index = int(row_num) - 1  # Convert to 0-based

    return row_index, col_index


def indices_to_cell(row: int, col: int) -> str:
    """Convert 0-based indices to Excel cell reference.

    Args:
        row: 0-based row index
        col: 0-based column index

    Returns:
        Excel cell reference (e.g., "A1", "AB123")
    """
    return f"{get_column_letter(col)}{row + 1}"


def parse_range(range_ref: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse Excel range reference to start and end indices.

    Args:
        range_ref: Excel range reference (e.g., "A1:B10", "Sheet1!A1:B10")

    Returns:
        Tuple of ((start_row, start_col), (end_row, end_col)) in 0-based indexing

    Raises:
        ValueError: If range reference is invalid
    """
    # Remove sheet name if present
    if "!" in range_ref:
        range_ref = range_ref.split("!")[-1]

    # Split range
    if ":" not in range_ref:
        raise ValueError(f"Invalid range reference: {range_ref}")

    start_cell, end_cell = range_ref.split(":")
    start_indices = cell_to_indices(start_cell)
    end_indices = cell_to_indices(end_cell)

    return start_indices, end_indices


def format_range(start_row: int, start_col: int, end_row: int, end_col: int) -> str:
    """Format 0-based indices as Excel range.

    Args:
        start_row: 0-based start row index
        start_col: 0-based start column index
        end_row: 0-based end row index
        end_col: 0-based end column index

    Returns:
        Excel range reference (e.g., "A1:B10")
    """
    start_cell = indices_to_cell(start_row, start_col)
    end_cell = indices_to_cell(end_row, end_col)
    return f"{start_cell}:{end_cell}"


def to_excel_range(top_row: int, left_col: int, bottom_row: int, right_col: int) -> str:
    """Convert cell bounds to Excel A1 notation range.

    Args:
        top_row: Top row (0-indexed)
        left_col: Left column (0-indexed)
        bottom_row: Bottom row (0-indexed)
        right_col: Right column (0-indexed)

    Returns:
        Excel range string (e.g., 'A1:Z100')
    """
    # Alias to format_range for compatibility
    return format_range(top_row, left_col, bottom_row, right_col)
