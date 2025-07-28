"""Tool for detecting named ranges in Excel workbooks."""

from typing import NamedTuple


class NamedRange(NamedTuple):
    """Represents a named range in Excel."""

    name: str
    sheet: str
    range: str
    scope: str = "Workbook"


def detect_named_ranges(workbook) -> list[NamedRange]:
    """
    Extract all named ranges from an Excel workbook.

    This is a pure function that extracts defined names from the workbook
    without making any decisions about whether they represent tables.

    Args:
        workbook: The Excel workbook object (openpyxl.Workbook)

    Returns:
        List of NamedRange objects with name and cell references
    """
    named_ranges = []

    if not hasattr(workbook, "defined_names"):
        return named_ranges

    for defined_name in workbook.defined_names.definedName:
        # Skip print areas and other special ranges
        if defined_name.name.startswith("_"):
            continue

        # Skip invalid or malformed names
        if not defined_name.value:
            continue

        # Parse the range reference
        try:
            # Handle references like 'Sheet1!$A$1:$D$10'
            if "!" in defined_name.value:
                sheet_name, cell_range = defined_name.value.split("!", 1)
                sheet_name = sheet_name.strip("'\"")
                # Remove absolute references ($)
                cell_range = cell_range.replace("$", "")
            else:
                # Workbook-level range
                sheet_name = ""
                cell_range = defined_name.value.replace("$", "")

            named_ranges.append(
                NamedRange(
                    name=defined_name.name,
                    sheet=sheet_name,
                    range=cell_range,
                    scope=defined_name.scope or "Workbook",
                )
            )
        except (ValueError, AttributeError):
            # Skip malformed references
            continue

    return named_ranges
