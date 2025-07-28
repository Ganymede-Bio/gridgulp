"""Tool for analyzing individual field semantics."""

import re
from collections import Counter
from typing import Any


def analyze_field_semantics(column_data: list[Any], header: str, table_type: str) -> dict[str, Any]:
    """
    Analyze the semantic meaning of a field/column.

    This tool examines the data patterns, header text, and table context
    to understand what a field represents.

    Args:
        column_data: Values from the column
        header: Column header text
        table_type: Type of table for context

    Returns:
        Semantic analysis of the field
    """
    analysis = {
        "field_name": header,
        "data_type": _infer_data_type(column_data),
        "semantic_type": "",
        "role": "",
        "unit": None,
        "cardinality": 0,
        "null_rate": 0.0,
        "patterns": [],
        "statistics": {},
    }

    # Filter out None values
    non_null_data = [v for v in column_data if v is not None and str(v).strip()]

    # Calculate null rate
    analysis["null_rate"] = (
        (len(column_data) - len(non_null_data)) / len(column_data) if column_data else 0
    )

    # Calculate cardinality
    analysis["cardinality"] = len(set(str(v) for v in non_null_data))

    # Analyze based on data type
    if analysis["data_type"] == "numeric":
        analysis.update(_analyze_numeric_field(non_null_data, header, table_type))
    elif analysis["data_type"] == "date":
        analysis.update(_analyze_date_field(non_null_data, header))
    elif analysis["data_type"] == "text":
        analysis.update(_analyze_text_field(non_null_data, header, analysis["cardinality"]))

    # Determine role based on position and characteristics
    analysis["role"] = _determine_field_role(analysis, header, table_type)

    return analysis


def _infer_data_type(values: list[Any]) -> str:
    """Infer the primary data type of values."""
    type_counts = Counter()

    for value in values[:100]:  # Sample first 100
        if value is None or str(value).strip() == "":
            continue

        value_str = str(value)

        # Check for dates
        if _is_date(value_str):
            type_counts["date"] += 1
        # Check for numbers
        elif _is_numeric(value_str):
            type_counts["numeric"] += 1
        else:
            type_counts["text"] += 1

    if not type_counts:
        return "text"

    # Return most common type
    return type_counts.most_common(1)[0][0]


def _is_date(value: str) -> bool:
    """Check if value looks like a date."""
    date_patterns = [
        r"^\d{4}-\d{2}-\d{2}",
        r"^\d{1,2}/\d{1,2}/\d{2,4}",
        r"^\d{1,2}-\d{1,2}-\d{2,4}",
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    ]
    return any(re.match(pattern, value, re.I) for pattern in date_patterns)


def _is_numeric(value: str) -> bool:
    """Check if value is numeric."""
    try:
        # Remove common formatting
        cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
        float(cleaned)
        return True
    except ValueError:
        return False


def _analyze_numeric_field(values: list[Any], header: str, table_type: str) -> dict[str, Any]:
    """Analyze numeric field characteristics."""
    result = {"semantic_type": "numeric"}

    # Convert to floats
    numeric_values = []
    for v in values:
        try:
            cleaned = str(v).replace(",", "").replace("$", "").replace("%", "").strip()
            numeric_values.append(float(cleaned))
        except ValueError:
            continue

    if not numeric_values:
        return result

    # Calculate statistics
    result["statistics"] = {
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "has_decimals": any(v % 1 != 0 for v in numeric_values),
        "all_positive": all(v >= 0 for v in numeric_values),
    }

    # Detect units/format
    header_lower = header.lower()
    sample_values = [str(v) for v in values[:10] if v is not None]

    if any("$" in v for v in sample_values) or "price" in header_lower or "cost" in header_lower:
        result["unit"] = "currency_USD"
        result["semantic_type"] = "currency"
    elif (
        any("%" in v for v in sample_values) or "percent" in header_lower or "rate" in header_lower
    ):
        result["unit"] = "percentage"
        result["semantic_type"] = "percentage"
    elif "quantity" in header_lower or "count" in header_lower or "qty" in header_lower:
        result["semantic_type"] = "quantity"
    elif table_type == "financial" and any(
        kw in header_lower for kw in ["revenue", "profit", "expense"]
    ):
        result["unit"] = "currency"
        result["semantic_type"] = "financial_metric"

    return result


def _analyze_date_field(values: list[Any], header: str) -> dict[str, Any]:
    """Analyze date field characteristics."""
    result = {"semantic_type": "date", "patterns": []}

    # Detect date patterns
    pattern_counts = Counter()
    for v in values[:20]:  # Sample
        if _is_date(str(v)):
            if re.match(r"^\d{4}-\d{2}-\d{2}", str(v)):
                pattern_counts["ISO"] += 1
            elif re.match(r"^\d{1,2}/\d{1,2}/\d{4}", str(v)):
                pattern_counts["US"] += 1
            elif re.match(r"^\d{1,2}-\d{1,2}-\d{4}", str(v)):
                pattern_counts["EU"] += 1

    if pattern_counts:
        result["patterns"] = [pattern_counts.most_common(1)[0][0]]

    # Check if it's a time series
    header_lower = header.lower()
    if any(period in header_lower for period in ["month", "quarter", "year", "date", "day"]):
        result["semantic_type"] = "time_dimension"

    return result


def _analyze_text_field(values: list[Any], header: str, cardinality: int) -> dict[str, Any]:
    """Analyze text field characteristics."""
    result = {"semantic_type": "text"}

    header_lower = header.lower()

    # Check for categories
    if cardinality < len(values) * 0.5 and cardinality < 100:
        result["semantic_type"] = "category"

    # Check for identifiers
    if any(id_kw in header_lower for id_kw in ["id", "code", "number", "key"]):
        result["semantic_type"] = "identifier"

    # Check for names
    if any(name_kw in header_lower for name_kw in ["name", "title", "description"]):
        result["semantic_type"] = "name"

    # Check for emails
    if any(re.match(r"^[^@]+@[^@]+\.[^@]+$", str(v)) for v in values[:10]):
        result["semantic_type"] = "email"

    # Check for URLs
    if any(str(v).startswith(("http://", "https://", "www.")) for v in values[:10]):
        result["semantic_type"] = "url"

    return result


def _determine_field_role(analysis: dict[str, Any], header: str, table_type: str) -> str:
    """Determine the role of a field in the table."""
    semantic_type = analysis.get("semantic_type", "")

    # Dimensions vs Measures
    if semantic_type in ["category", "identifier", "name", "time_dimension"]:
        return "dimension"
    elif semantic_type in ["numeric", "currency", "percentage", "quantity"]:
        return "measure"

    # Special roles
    if semantic_type == "identifier" and analysis.get("cardinality", 0) == len(
        analysis.get("values", [])
    ):
        return "primary_key"

    if table_type == "time_series" and semantic_type == "time_dimension":
        return "time_index"

    return "attribute"
