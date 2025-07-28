"""Tool for parsing vision model responses."""

import json
import re
from typing import Any


def parse_vision_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Parse response from vision model into structured format.

    This tool handles various response formats and extracts table
    information reliably.

    Args:
        response: Raw response from vision API

    Returns:
        Parsed table information
    """
    # Extract content from response
    if "choices" in response and response["choices"]:
        content = response["choices"][0]["message"]["content"]
    elif "content" in response:
        content = response["content"]
    else:
        raise ValueError("Unable to extract content from response")

    # Try to parse as JSON first
    try:
        # Look for JSON in the response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return _validate_parsed_response(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback: parse structured text
    return _parse_text_response(content)


def _validate_parsed_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize parsed response."""
    # Ensure required fields
    if "tables" not in parsed:
        parsed["tables"] = []

    # Normalize table entries
    normalized_tables = []
    for table in parsed.get("tables", []):
        normalized = {
            "id": table.get("id", f"table_{len(normalized_tables)}"),
            "bounds": _normalize_bounds(table.get("bounds", {})),
            "confidence": float(table.get("confidence", 0.5)),
            "evidence": table.get("evidence", {}),
            "context_analysis": table.get("context_analysis", {}),
            "description": table.get("description", ""),
        }
        normalized_tables.append(normalized)

    return {"tables": normalized_tables, "analysis_notes": parsed.get("analysis_notes", "")}


def _normalize_bounds(bounds: dict[str, Any]) -> dict[str, int]:
    """Normalize bounds to consistent format."""
    # Handle different key names
    key_mappings = {
        "top": ["top", "top_row", "start_row", "y1"],
        "left": ["left", "left_col", "start_col", "x1"],
        "bottom": ["bottom", "bottom_row", "end_row", "y2"],
        "right": ["right", "right_col", "end_col", "x2"],
    }

    normalized = {}
    for target_key, possible_keys in key_mappings.items():
        for key in possible_keys:
            if key in bounds:
                normalized[target_key] = int(bounds[key])
                break

    # Ensure we have all required bounds
    if len(normalized) != 4:
        raise ValueError(f"Invalid bounds: {bounds}")

    return normalized


def _parse_text_response(content: str) -> dict[str, Any]:
    """Parse structured text response as fallback."""
    tables = []

    # Look for table descriptions
    table_pattern = r"Table\s+(\d+).*?(?:bounds|range).*?(\d+),\s*(\d+).*?to.*?(\d+),\s*(\d+)"

    for match in re.finditer(table_pattern, content, re.IGNORECASE | re.DOTALL):
        table_id, top, left, bottom, right = match.groups()

        tables.append(
            {
                "id": f"table_{table_id}",
                "bounds": {
                    "top": int(top),
                    "left": int(left),
                    "bottom": int(bottom),
                    "right": int(right),
                },
                "confidence": 0.7,  # Default confidence for text parsing
                "evidence": {},
                "description": "",
            }
        )

    return {"tables": tables}
