"""Tool for parsing table headers, including multi-row headers."""

from typing import Any, NamedTuple


class HeaderStructure(NamedTuple):
    """Structure of parsed headers."""

    row_count: int
    unified_headers: list[str]
    header_rows: list[list[str]]
    data_start_row: int
    spans: list[dict[str, Any]]


def parse_headers(sheet_data: Any, table_range: Any, max_header_rows: int = 5) -> HeaderStructure:
    """
    Analyze and parse headers, including multi-row headers.

    This tool detects header structure by analyzing formatting,
    data types, and patterns in the first rows.

    Args:
        sheet_data: Sheet containing the table
        table_range: Table boundaries
        max_header_rows: Maximum rows to consider as headers

    Returns:
        Parsed header structure
    """
    # Extract first N rows for analysis
    header_candidates = []
    for row_idx in range(
        table_range.start_row, min(table_range.start_row + max_header_rows, table_range.end_row + 1)
    ):
        row_data = []
        row_formats = []

        for col_idx in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(row_idx, col_idx)

            if cell:
                row_data.append(cell.value if hasattr(cell, "value") else None)
                row_formats.append(
                    {
                        "is_bold": getattr(cell, "is_bold", False),
                        "data_type": getattr(cell, "data_type", "unknown"),
                        "background_color": getattr(cell, "background_color", None),
                    }
                )
            else:
                row_data.append(None)
                row_formats.append({})

        header_candidates.append(
            {"data": row_data, "formats": row_formats, "row_idx": row_idx - table_range.start_row}
        )

    # Determine header row count
    header_row_count = _detect_header_rows(header_candidates)

    # Extract header rows
    header_rows = [candidate["data"] for candidate in header_candidates[:header_row_count]]

    # Create unified headers
    if header_row_count == 1:
        unified_headers = header_rows[0]
    else:
        unified_headers = _merge_multi_row_headers(header_rows)

    # Detect spans in multi-row headers
    spans = _detect_header_spans(header_rows) if header_row_count > 1 else []

    return HeaderStructure(
        row_count=header_row_count,
        unified_headers=unified_headers,
        header_rows=header_rows,
        data_start_row=table_range.start_row + header_row_count,
        spans=spans,
    )


def _detect_header_rows(candidates: list[dict[str, Any]]) -> int:
    """Detect how many rows are headers."""
    if not candidates:
        return 0

    header_count = 1  # At least first row

    for i in range(1, len(candidates)):
        row = candidates[i]

        # Check if this row looks like headers
        text_count = sum(1 for fmt in row["formats"] if fmt.get("data_type") == "text")
        bold_count = sum(1 for fmt in row["formats"] if fmt.get("is_bold", False))

        # If mostly text or bold, likely still headers
        total_cells = len(row["formats"])
        if text_count > total_cells * 0.7 or bold_count > total_cells * 0.5:
            header_count = i + 1
        else:
            break

    return header_count


def _merge_multi_row_headers(header_rows: list[list[Any]]) -> list[str]:
    """Merge multi-row headers into single row."""
    if not header_rows:
        return []

    num_cols = len(header_rows[0])
    unified = []

    for col_idx in range(num_cols):
        # Collect non-empty values from this column
        col_values = []
        for row in header_rows:
            if col_idx < len(row) and row[col_idx]:
                col_values.append(str(row[col_idx]))

        # Join with separator
        unified_header = " - ".join(col_values) if col_values else f"Column_{col_idx + 1}"
        unified.append(unified_header)

    return unified


def _detect_header_spans(header_rows: list[list[Any]]) -> list[dict[str, Any]]:
    """Detect spanning headers in multi-row headers."""
    spans = []

    for row_idx, row in enumerate(header_rows[:-1]):  # Don't check last row
        current_span = None

        for col_idx, value in enumerate(row):
            if value and (col_idx + 1 >= len(row) or row[col_idx + 1] != value):
                # End of a potential span
                if current_span and current_span["value"] == value:
                    current_span["end_col"] = col_idx
                    if current_span["end_col"] > current_span["start_col"]:
                        spans.append(current_span)
                    current_span = None
                elif value:
                    # Single cell with value
                    current_span = {
                        "row": row_idx,
                        "start_col": col_idx,
                        "end_col": col_idx,
                        "value": value,
                    }
            elif value and current_span and current_span["value"] == value:
                # Continue span
                pass
            elif value:
                # Start new span
                if current_span:
                    current_span["end_col"] = col_idx - 1
                    if current_span["end_col"] > current_span["start_col"]:
                        spans.append(current_span)

                current_span = {"row": row_idx, "start_col": col_idx, "value": value}

    return spans
