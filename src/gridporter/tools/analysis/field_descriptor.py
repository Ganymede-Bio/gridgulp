"""Tool for generating human-readable field descriptions."""

from typing import Any


def generate_field_descriptions(
    header: str, semantics: dict[str, Any], sample_data: list[Any]
) -> str:
    """
    Generate a human-readable description of what a field contains.

    This tool creates descriptions that help users understand the
    purpose and content of each field.

    Args:
        header: Field header/name
        semantics: Semantic analysis results
        sample_data: Sample values from the field

    Returns:
        Human-readable field description
    """
    # Start with basic info
    data_type = semantics.get("data_type", "unknown")
    semantic_type = semantics.get("semantic_type", data_type)
    role = semantics.get("role", "attribute")

    # Build description based on semantic type
    if semantic_type == "identifier":
        description = "Unique identifier for each record"

    elif semantic_type == "currency":
        unit = semantics.get("unit", "currency")
        if "USD" in unit:
            currency = "USD"
        else:
            currency = "currency"

        # Try to infer meaning from header
        header_lower = header.lower()
        if "revenue" in header_lower:
            description = f"Revenue amount in {currency}"
        elif "cost" in header_lower or "expense" in header_lower:
            description = f"Cost/expense amount in {currency}"
        elif "price" in header_lower:
            description = f"Price per unit in {currency}"
        elif "total" in header_lower:
            description = f"Total amount in {currency}"
        else:
            description = f"Monetary value in {currency}"

    elif semantic_type == "percentage":
        header_lower = header.lower()
        if "rate" in header_lower:
            description = "Rate expressed as a percentage"
        elif "growth" in header_lower:
            description = "Growth percentage"
        elif "margin" in header_lower:
            description = "Margin percentage"
        else:
            description = "Percentage value"

    elif semantic_type == "quantity":
        description = "Quantity or count of items"

    elif semantic_type == "category":
        cardinality = semantics.get("cardinality", 0)
        description = f"Categorical field with {cardinality} unique values"

        # Add examples if few categories
        if sample_data and cardinality <= 10:
            unique_values = list(set(str(v) for v in sample_data if v is not None))[:5]
            if unique_values:
                description += f" ({', '.join(unique_values)})"

    elif semantic_type == "time_dimension":
        patterns = semantics.get("patterns", [])
        if patterns:
            description = f"Date/time field in {patterns[0]} format"
        else:
            description = "Date/time field"

    elif semantic_type == "name":
        header_lower = header.lower()
        if "product" in header_lower:
            description = "Product name or description"
        elif "customer" in header_lower:
            description = "Customer name"
        elif "employee" in header_lower:
            description = "Employee name"
        else:
            description = "Name or title field"

    elif semantic_type == "email":
        description = "Email address"

    elif semantic_type == "url":
        description = "Web URL or link"

    else:
        # Generic description
        if data_type == "numeric":
            stats = semantics.get("statistics", {})
            if stats:
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)
                description = f"Numeric field ranging from {min_val:.2f} to {max_val:.2f}"
            else:
                description = "Numeric field"
        else:
            description = f"{data_type.capitalize()} field"

    # Add null rate info if significant
    null_rate = semantics.get("null_rate", 0)
    if null_rate > 0.1:
        description += f" ({null_rate*100:.0f}% missing values)"

    # Add role information
    if role == "dimension":
        description = f"{description} (used for grouping/filtering)"
    elif role == "measure":
        description = f"{description} (aggregatable metric)"
    elif role == "primary_key":
        description = f"{description} (primary key)"
    elif role == "time_index":
        description = f"{description} (time series index)"

    return description
