"""Tool for detecting hierarchical structures in data."""

from typing import Any, NamedTuple


class HierarchyInfo(NamedTuple):
    """Information about detected hierarchy."""

    levels: int
    indent_pattern: str
    parent_child_map: dict[int, list[int]]
    level_indicators: dict[int, str]


def detect_hierarchy(sheet_data: Any, table_range: Any) -> HierarchyInfo:
    """
    Detect hierarchical structure in table data.

    This tool identifies parent-child relationships based on:
    - Indentation patterns
    - Formatting differences
    - Empty cells indicating grouping

    Args:
        sheet_data: Sheet containing the table
        table_range: Table boundaries

    Returns:
        Hierarchy information
    """
    # Analyze first column for hierarchy patterns
    hierarchy_levels = []
    indent_counts = []

    for row_idx in range(table_range.start_row, table_range.end_row + 1):
        cell = sheet_data.get_cell(row_idx, table_range.start_col)

        if cell and hasattr(cell, "value") and cell.value:
            value = str(cell.value)

            # Count leading spaces
            indent = len(value) - len(value.lstrip())
            indent_counts.append(indent)

            # Check for other hierarchy indicators
            is_bold = getattr(cell, "is_bold", False)
            has_background = bool(getattr(cell, "background_color", None))

            hierarchy_levels.append(
                {
                    "row": row_idx - table_range.start_row,
                    "indent": indent,
                    "is_bold": is_bold,
                    "has_background": has_background,
                    "value": value.strip(),
                }
            )

    # Determine number of hierarchy levels
    unique_indents = sorted(set(indent_counts))
    num_levels = len(unique_indents) if unique_indents else 1

    # Build parent-child relationships
    parent_child_map = _build_parent_child_map(hierarchy_levels)

    # Determine indent pattern
    indent_pattern = "spaces"
    if unique_indents and len(unique_indents) > 1:
        indent_diff = unique_indents[1] - unique_indents[0]
        if indent_diff == 2:
            indent_pattern = "2_spaces"
        elif indent_diff == 4:
            indent_pattern = "4_spaces"
        elif indent_diff == 1:
            indent_pattern = "1_space"

    # Identify level indicators
    level_indicators = _identify_level_indicators(hierarchy_levels, unique_indents)

    return HierarchyInfo(
        levels=num_levels,
        indent_pattern=indent_pattern,
        parent_child_map=parent_child_map,
        level_indicators=level_indicators,
    )


def _build_parent_child_map(hierarchy_levels: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Build parent-child relationships based on indentation."""
    parent_child_map = {}

    # Stack to track current parent at each level
    parent_stack = []

    for item in hierarchy_levels:
        row = item["row"]
        indent = item["indent"]

        # Pop stack until we find appropriate parent level
        while parent_stack and parent_stack[-1]["indent"] >= indent:
            parent_stack.pop()

        # Current item's parent is top of stack
        if parent_stack:
            parent_row = parent_stack[-1]["row"]
            if parent_row not in parent_child_map:
                parent_child_map[parent_row] = []
            parent_child_map[parent_row].append(row)

        # Push current item as potential parent
        parent_stack.append(item)

    return parent_child_map


def _identify_level_indicators(
    hierarchy_levels: list[dict[str, Any]], unique_indents: list[int]
) -> dict[int, str]:
    """Identify visual indicators for each hierarchy level."""
    level_indicators = {}

    for level, indent in enumerate(unique_indents):
        # Find all items at this level
        level_items = [item for item in hierarchy_levels if item["indent"] == indent]

        if not level_items:
            continue

        # Check common characteristics
        mostly_bold = sum(1 for item in level_items if item["is_bold"]) > len(level_items) * 0.7
        mostly_background = (
            sum(1 for item in level_items if item["has_background"]) > len(level_items) * 0.7
        )

        indicators = []
        if indent > 0:
            indicators.append(f"{indent}_space_indent")
        if mostly_bold:
            indicators.append("bold")
        if mostly_background:
            indicators.append("background_color")

        level_indicators[level] = "_".join(indicators) if indicators else "none"

    return level_indicators
