"""Tool for handling formulas in spreadsheets."""

from typing import Any


def handle_formulas(sheet_data: Any, table_range: Any) -> dict[str, Any]:
    """
    Extract and process formulas within a table range.

    This tool identifies cells containing formulas and extracts
    both the formula and calculated value.

    Args:
        sheet_data: Sheet containing the table
        table_range: Table boundaries

    Returns:
        Dictionary mapping cell coordinates to formula info
    """
    formula_map = {}

    for row_idx in range(table_range.start_row, table_range.end_row + 1):
        for col_idx in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(row_idx, col_idx)

            if cell and hasattr(cell, "has_formula") and cell.has_formula:
                # Create cell reference
                cell_ref = f"R{row_idx}C{col_idx}"

                formula_map[cell_ref] = {
                    "formula": getattr(cell, "formula", ""),
                    "value": getattr(cell, "value", None),
                    "row": row_idx - table_range.start_row,
                    "col": col_idx - table_range.start_col,
                }

    return formula_map
