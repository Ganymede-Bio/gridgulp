"""Tool for extracting Excel ListObjects (formal tables)."""

from typing import NamedTuple


class ListObject(NamedTuple):
    """Represents an Excel ListObject (formal table)."""

    name: str
    range: str
    has_headers: bool
    style: str = ""
    totals_row: bool = False


def extract_list_objects(worksheet) -> list[ListObject]:
    """
    Extract Excel ListObjects (formal tables) from a worksheet.

    ListObjects are Excel's built-in table feature where users explicitly
    define a range as a table with formatting and functionality.

    Args:
        worksheet: The Excel worksheet object (openpyxl.worksheet)

    Returns:
        List of ListObject information
    """
    list_objects = []

    # Check if worksheet has tables attribute
    if not hasattr(worksheet, "tables") or not worksheet.tables:
        return list_objects

    for table_name, table in worksheet.tables.items():
        try:
            list_object = ListObject(
                name=table_name,
                range=table.ref,  # e.g., "A1:E10"
                has_headers=table.headerRowCount > 0 if hasattr(table, "headerRowCount") else True,
                style=table.tableStyleInfo.name
                if hasattr(table, "tableStyleInfo") and table.tableStyleInfo
                else "",
                totals_row=table.totalsRowCount > 0 if hasattr(table, "totalsRowCount") else False,
            )
            list_objects.append(list_object)
        except AttributeError:
            # Skip if table object is malformed
            continue

    return list_objects
