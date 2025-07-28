"""Tool for extracting raw cell data from a table range."""

from typing import Any


def extract_cell_data(sheet_data: Any, table_range: Any) -> list[list[Any]]:
    """
    Extract raw cell values from a table range.

    This is a pure function that extracts data without any interpretation
    or modification.

    Args:
        sheet_data: Sheet containing the data
        table_range: TableRange object defining boundaries

    Returns:
        2D list of cell values
    """
    rows = []

    for row_idx in range(table_range.start_row, table_range.end_row + 1):
        row_data = []

        for col_idx in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(row_idx, col_idx)

            if cell and hasattr(cell, "value"):
                value = cell.value
            else:
                value = None

            row_data.append(value)

        rows.append(row_data)

    return rows
