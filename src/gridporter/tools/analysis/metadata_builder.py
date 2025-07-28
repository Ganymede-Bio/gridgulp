"""Tool for building comprehensive metadata."""

from typing import Any


def build_metadata(
    table: Any, table_type: str, field_analysis: dict[str, Any], field_descriptions: dict[str, str]
) -> dict[str, Any]:
    """
    Build comprehensive metadata for a table.

    This tool aggregates all analysis results into a structured
    metadata object that can guide data processing.

    Args:
        table: Extracted table data
        table_type: Classification of table type
        field_analysis: Semantic analysis of each field
        field_descriptions: Human-readable descriptions

    Returns:
        Comprehensive metadata dictionary
    """
    metadata = {
        "table_characteristics": {
            "type": table_type,
            "row_count": len(table.data) if hasattr(table, "data") else 0,
            "column_count": len(table.headers) if hasattr(table, "headers") else 0,
            "has_headers": True,
            "header_row_count": table.metadata.get("header_row_count", 1)
            if hasattr(table, "metadata")
            else 1,
        },
        "data_quality": {
            "completeness": _calculate_completeness(table, field_analysis),
            "consistency": _assess_consistency(field_analysis),
            "validity": _assess_validity(field_analysis),
        },
        "field_metadata": {},
        "processing_hints": {
            "recommended_dtypes": {},
            "parse_dates": [],
            "categorical_columns": [],
            "index_columns": [],
            "skip_columns": [],
        },
        "semantic_summary": {
            "purpose": _infer_table_purpose(table_type, field_analysis),
            "key_metrics": [],
            "dimensions": [],
            "time_granularity": None,
        },
    }

    # Build field metadata
    for field_name, analysis in field_analysis.items():
        metadata["field_metadata"][field_name] = {
            "data_type": analysis.get("data_type"),
            "semantic_type": analysis.get("semantic_type"),
            "role": analysis.get("role"),
            "description": field_descriptions.get(field_name, ""),
            "statistics": analysis.get("statistics", {}),
            "quality": {
                "null_rate": analysis.get("null_rate", 0),
                "cardinality": analysis.get("cardinality", 0),
                "patterns": analysis.get("patterns", []),
            },
        }

        # Update processing hints
        if analysis.get("semantic_type") == "date":
            metadata["processing_hints"]["parse_dates"].append(field_name)

        if analysis.get("semantic_type") == "category" and analysis.get("cardinality", 999) < 50:
            metadata["processing_hints"]["categorical_columns"].append(field_name)

        if analysis.get("role") == "primary_key" or analysis.get("role") == "time_index":
            metadata["processing_hints"]["index_columns"].append(field_name)

        # Recommended dtypes
        dtype = _recommend_dtype(analysis)
        if dtype:
            metadata["processing_hints"]["recommended_dtypes"][field_name] = dtype

        # Semantic summary
        if analysis.get("role") == "measure":
            metadata["semantic_summary"]["key_metrics"].append(field_name)
        elif analysis.get("role") == "dimension":
            metadata["semantic_summary"]["dimensions"].append(field_name)

    # Detect time granularity
    time_fields = [
        name
        for name, analysis in field_analysis.items()
        if analysis.get("semantic_type") == "time_dimension"
    ]
    if time_fields:
        metadata["semantic_summary"]["time_granularity"] = _detect_time_granularity(
            table, time_fields[0]
        )

    return metadata


def _calculate_completeness(table: Any, field_analysis: dict[str, Any]) -> float:
    """Calculate overall data completeness."""
    if not field_analysis:
        return 1.0

    null_rates = [analysis.get("null_rate", 0) for analysis in field_analysis.values()]

    avg_null_rate = sum(null_rates) / len(null_rates)
    return 1.0 - avg_null_rate


def _assess_consistency(field_analysis: dict[str, Any]) -> float:
    """Assess data consistency."""
    # Simple heuristic: fields with expected types are consistent
    consistency_scores = []

    for analysis in field_analysis.values():
        if analysis.get("data_type") == analysis.get("semantic_type"):
            consistency_scores.append(1.0)
        elif analysis.get("patterns"):
            # Has identifiable patterns
            consistency_scores.append(0.8)
        else:
            consistency_scores.append(0.5)

    return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5


def _assess_validity(field_analysis: dict[str, Any]) -> float:
    """Assess data validity."""
    # Check for valid ranges, formats, etc.
    validity_scores = []

    for analysis in field_analysis.values():
        score = 1.0

        # Penalize high null rates
        null_rate = analysis.get("null_rate", 0)
        if null_rate > 0.5:
            score *= 0.5
        elif null_rate > 0.2:
            score *= 0.8

        # Check for outliers in numeric fields
        if analysis.get("data_type") == "numeric":
            stats = analysis.get("statistics", {})
            if "has_outliers" in stats:
                score *= 0.9

        validity_scores.append(score)

    return sum(validity_scores) / len(validity_scores) if validity_scores else 0.5


def _recommend_dtype(analysis: dict[str, Any]) -> str | None:
    """Recommend pandas dtype based on analysis."""
    data_type = analysis.get("data_type")
    semantic_type = analysis.get("semantic_type")

    if semantic_type == "identifier":
        return "string"
    elif semantic_type == "category":
        return "category"
    elif data_type == "numeric":
        stats = analysis.get("statistics", {})
        if stats.get("has_decimals", True):
            return "float64"
        else:
            return "Int64"  # Nullable integer
    elif semantic_type == "date":
        return "datetime64[ns]"

    return None


def _infer_table_purpose(table_type: str, field_analysis: dict[str, Any]) -> str:
    """Infer the purpose of the table."""
    purposes = {
        "financial": "Financial data tracking monetary values over time or categories",
        "inventory": "Inventory or catalog data tracking items and quantities",
        "personnel": "Personnel or HR data containing employee information",
        "time_series": "Time series data tracking metrics over time periods",
        "hierarchical": "Hierarchical data with parent-child relationships",
        "matrix": "Matrix data showing relationships between row and column dimensions",
    }

    return purposes.get(table_type, "General data table")


def _detect_time_granularity(table: Any, time_field: str) -> str | None:
    """Detect the time granularity of data."""
    # This would analyze actual time values
    # For now, return a placeholder
    return "daily"
