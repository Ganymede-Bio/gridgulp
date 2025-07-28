"""Tool for generating meaningful table names."""

import re
from typing import Any


def generate_table_name(
    headers: list[str], table_type: str, context: dict[str, Any], use_llm: bool = False
) -> str:
    """
    Generate a meaningful name for a table.

    This tool creates descriptive names based on table content,
    either using heuristics or LLM assistance.

    Args:
        headers: Column headers
        table_type: Classified table type
        context: Additional context (sheet name, file name, etc.)
        use_llm: Whether to use LLM for naming

    Returns:
        Suggested table name
    """
    if use_llm:
        # This would call an LLM with table info
        # For now, falling back to heuristics
        pass

    # Heuristic-based naming
    sheet_name = context.get("sheet_name", "").lower()
    file_name = context.get("file_name", "").lower()

    # Clean headers for analysis
    clean_headers = [str(h).lower() for h in headers if h]
    header_text = " ".join(clean_headers)

    # Start with table type
    base_name = table_type

    # Enhance based on content
    if table_type == "financial":
        if "revenue" in header_text:
            base_name = "revenue"
        elif "expense" in header_text or "cost" in header_text:
            base_name = "expenses"
        elif "profit" in header_text:
            base_name = "profit_loss"
        elif "balance" in header_text:
            base_name = "balance_sheet"

        # Add time period if found
        if "quarterly" in header_text or all(f"q{i}" in header_text for i in range(1, 5)):
            base_name += "_quarterly"
        elif "monthly" in header_text:
            base_name += "_monthly"
        elif "annual" in header_text or "yearly" in header_text:
            base_name += "_annual"

    elif table_type == "inventory":
        if "product" in header_text:
            base_name = "product"
        elif "item" in header_text:
            base_name = "item"
        else:
            base_name = "inventory"

        if "catalog" in sheet_name or "catalog" in header_text:
            base_name += "_catalog"
        elif "stock" in header_text:
            base_name += "_stock"

    elif table_type == "personnel":
        if "employee" in header_text:
            base_name = "employee"
        elif "staff" in header_text:
            base_name = "staff"
        else:
            base_name = "personnel"

        if "contact" in header_text:
            base_name += "_contact"
        elif "salary" in header_text:
            base_name += "_compensation"
        else:
            base_name += "_list"

    elif table_type == "time_series":
        # Look for what's being tracked
        measure_found = False
        for header in clean_headers:
            if header not in ["date", "time", "month", "year", "quarter"]:
                # This might be what we're tracking
                base_name = f"{header}_time_series"
                measure_found = True
                break

        if not measure_found:
            base_name = "time_series_data"

    elif table_type == "hierarchical":
        # Try to identify what the hierarchy represents
        if "account" in header_text:
            base_name = "account_hierarchy"
        elif "category" in header_text:
            base_name = "category_hierarchy"
        elif "org" in header_text or "department" in header_text:
            base_name = "org_hierarchy"
        else:
            base_name = "hierarchical_data"

    # Add sheet context if meaningful
    if sheet_name and sheet_name not in ["sheet1", "data", "table"]:
        # Extract meaningful parts from sheet name
        sheet_words = re.findall(r"\w+", sheet_name)
        meaningful_words = [
            w for w in sheet_words if len(w) > 2 and w not in ["the", "and", "for", "with"]
        ]

        if meaningful_words and meaningful_words[0] not in base_name:
            base_name = f"{meaningful_words[0]}_{base_name}"

    # Ensure valid Python identifier
    base_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name)
    base_name = re.sub(r"^[0-9]+", "", base_name)
    base_name = re.sub(r"_+", "_", base_name)
    base_name = base_name.strip("_").lower()

    # Add suffix if needed to avoid conflicts
    if context.get("existing_names") and base_name in context["existing_names"]:
        suffix = 2
        while f"{base_name}_{suffix}" in context["existing_names"]:
            suffix += 1
        base_name = f"{base_name}_{suffix}"

    return base_name or "data_table"
