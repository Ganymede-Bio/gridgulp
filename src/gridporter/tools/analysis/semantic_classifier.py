"""Tool for classifying table types based on content."""

import re
from typing import Any


def classify_table_type(data: list[list[Any]], headers: list[str]) -> str:
    """
    Classify the type of table based on its content and structure.

    This tool analyzes headers and data patterns to determine the
    table's purpose (financial, inventory, time series, etc.).

    Args:
        data: Table data rows
        headers: Column headers

    Returns:
        Table type classification
    """
    # Analyze headers for clues
    header_text = " ".join(str(h).lower() for h in headers if h)

    # Financial indicators
    financial_keywords = [
        "revenue",
        "profit",
        "cost",
        "expense",
        "income",
        "balance",
        "debit",
        "credit",
        "total",
        "subtotal",
        "tax",
        "gross",
        "net",
        "q1",
        "q2",
        "q3",
        "q4",
        "quarter",
        "fiscal",
        "budget",
    ]

    # Time series indicators
    time_keywords = [
        "date",
        "time",
        "year",
        "month",
        "day",
        "week",
        "period",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    # Inventory/catalog indicators
    inventory_keywords = [
        "product",
        "item",
        "sku",
        "inventory",
        "stock",
        "quantity",
        "price",
        "description",
        "category",
        "unit",
        "in stock",
    ]

    # Personnel/HR indicators
    hr_keywords = [
        "employee",
        "name",
        "department",
        "salary",
        "position",
        "hire",
        "manager",
        "email",
        "phone",
        "id",
        "staff",
    ]

    # Count keyword matches
    financial_score = sum(1 for kw in financial_keywords if kw in header_text)
    time_score = sum(1 for kw in time_keywords if kw in header_text)
    inventory_score = sum(1 for kw in inventory_keywords if kw in header_text)
    hr_score = sum(1 for kw in hr_keywords if kw in header_text)

    # Analyze data patterns
    if data and len(data) > 0:
        # Check for numeric patterns
        numeric_columns = _count_numeric_columns(data)
        date_columns = _count_date_columns(data)

        # Adjust scores based on data
        if numeric_columns > len(headers) * 0.5:
            financial_score += 2
        if date_columns > 0:
            time_score += 2

    # Determine primary type
    scores = {
        "financial": financial_score,
        "time_series": time_score,
        "inventory": inventory_score,
        "personnel": hr_score,
    }

    # Get highest scoring type
    max_score = max(scores.values())
    if max_score > 0:
        for table_type, score in scores.items():
            if score == max_score:
                return table_type

    # Check for other patterns
    if _is_hierarchical(data, headers):
        return "hierarchical"

    if _is_matrix(data, headers):
        return "matrix"

    # Default
    return "data_table"


def _count_numeric_columns(data: list[list[Any]]) -> int:
    """Count columns that are primarily numeric."""
    if not data or not data[0]:
        return 0

    numeric_cols = 0
    num_cols = len(data[0])

    for col_idx in range(num_cols):
        numeric_count = 0
        total_count = 0

        for row in data:
            if col_idx < len(row) and row[col_idx] is not None:
                total_count += 1
                try:
                    float(str(row[col_idx]).replace(",", "").replace("$", ""))
                    numeric_count += 1
                except ValueError:
                    pass

        if total_count > 0 and numeric_count / total_count > 0.8:
            numeric_cols += 1

    return numeric_cols


def _count_date_columns(data: list[list[Any]]) -> int:
    """Count columns that appear to contain dates."""
    if not data or not data[0]:
        return 0

    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",  # ISO date
        r"\d{1,2}/\d{1,2}/\d{2,4}",  # US date
        r"\d{1,2}-\d{1,2}-\d{2,4}",  # Other date
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",  # Month names
    ]

    date_cols = 0
    num_cols = len(data[0])

    for col_idx in range(num_cols):
        date_count = 0
        total_count = 0

        for row in data[:10]:  # Sample first 10 rows
            if col_idx < len(row) and row[col_idx]:
                total_count += 1
                value_str = str(row[col_idx])

                if any(re.search(pattern, value_str, re.I) for pattern in date_patterns):
                    date_count += 1

        if total_count > 0 and date_count / total_count > 0.5:
            date_cols += 1

    return date_cols


def _is_hierarchical(data: list[list[Any]], headers: list[str]) -> bool:
    """Check if table has hierarchical structure."""
    if not data or not data[0]:
        return False

    # Check first column for indentation patterns
    first_col_values = [row[0] for row in data if row and row[0]]

    indented_count = sum(
        1 for val in first_col_values if isinstance(val, str) and val.startswith((" ", "\t"))
    )

    return indented_count > len(first_col_values) * 0.2


def _is_matrix(data: list[list[Any]], headers: list[str]) -> bool:
    """Check if table is a matrix (e.g., correlation matrix)."""
    if not data or not headers:
        return False

    # Check if row headers match column headers
    if len(data) == len(headers) - 1:
        row_headers = [row[0] for row in data if row and row[0]]
        col_headers = headers[1:]  # Skip first header

        matches = sum(
            1 for rh, ch in zip(row_headers, col_headers, strict=False) if str(rh) == str(ch)
        )

        return matches > len(row_headers) * 0.8

    return False
