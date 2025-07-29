"""Excel metadata extractor for ListObjects, named ranges, and print areas.

This module provides extraction of Excel-specific metadata that can be used
as hints for table detection and verification.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExcelTableInfo:
    """Information about an Excel ListObject (native table)."""

    name: str
    display_name: str
    range_address: str
    has_headers: bool = True
    has_totals: bool = False
    table_style: str | None = None


@dataclass
class NamedRangeInfo:
    """Information about an Excel named range."""

    name: str
    refers_to: str
    scope: str = "Workbook"  # Can be "Workbook" or sheet name
    is_hidden: bool = False
    comment: str | None = None


@dataclass
class PrintAreaInfo:
    """Information about print areas in a sheet."""

    sheet_name: str
    print_area: str | None = None
    print_titles_rows: str | None = None
    print_titles_cols: str | None = None


@dataclass
class ExcelMetadata:
    """Container for all Excel-specific metadata."""

    list_objects: list[ExcelTableInfo] = field(default_factory=list)
    named_ranges: list[NamedRangeInfo] = field(default_factory=list)
    print_areas: list[PrintAreaInfo] = field(default_factory=list)
    has_data_model: bool = False
    has_pivot_tables: bool = False


class ExcelMetadataExtractor:
    """Extracts Excel-specific metadata for table detection hints."""

    def __init__(self) -> None:
        """Initialize the metadata extractor."""
        self.logger = logger

    def extract_metadata_openpyxl(self, workbook: Any) -> ExcelMetadata:
        """Extract metadata from an openpyxl workbook.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            ExcelMetadata containing all extracted information
        """
        metadata = ExcelMetadata()

        # Extract ListObjects (Tables)
        metadata.list_objects = self._extract_list_objects_openpyxl(workbook)

        # Extract Named Ranges
        metadata.named_ranges = self._extract_named_ranges_openpyxl(workbook)

        # Extract Print Areas
        metadata.print_areas = self._extract_print_areas_openpyxl(workbook)

        # Check for advanced features
        metadata.has_pivot_tables = self._check_pivot_tables_openpyxl(workbook)

        return metadata

    def extract_metadata_xlrd(self, workbook: Any) -> ExcelMetadata:
        """Extract metadata from an xlrd workbook.

        Note: xlrd has limited support for Excel metadata, especially
        for .xls files. Most advanced features are not accessible.

        Args:
            workbook: xlrd Book object

        Returns:
            ExcelMetadata with limited information
        """
        metadata = ExcelMetadata()

        # Extract named ranges (limited support in xlrd)
        metadata.named_ranges = self._extract_named_ranges_xlrd(workbook)

        # xlrd doesn't support ListObjects or print areas for .xls files
        self.logger.debug("xlrd has limited metadata support for .xls files")

        return metadata

    def _extract_list_objects_openpyxl(self, workbook: Any) -> list[ExcelTableInfo]:
        """Extract ListObjects (native Excel tables) from openpyxl workbook.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            List of ExcelTableInfo objects
        """
        tables = []

        try:
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]

                # Check if worksheet has tables
                if hasattr(worksheet, "tables") and worksheet.tables:
                    for table_name, table in worksheet.tables.items():
                        try:
                            table_info = ExcelTableInfo(
                                name=table_name,
                                display_name=table.displayName or table_name,
                                range_address=f"{sheet_name}!{table.ref}",
                                has_headers=table.headerRowCount > 0
                                if hasattr(table, "headerRowCount")
                                else True,
                                has_totals=table.totalsRowCount > 0
                                if hasattr(table, "totalsRowCount")
                                else False,
                                table_style=table.tableStyleInfo.name
                                if hasattr(table, "tableStyleInfo") and table.tableStyleInfo
                                else None,
                            )
                            tables.append(table_info)
                            self.logger.debug(
                                f"Found table: {table_info.display_name} at {table_info.range_address}"
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to extract table {table_name}: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to extract ListObjects: {e}")

        return tables

    def _extract_named_ranges_openpyxl(self, workbook: Any) -> list[NamedRangeInfo]:
        """Extract named ranges from openpyxl workbook.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            List of NamedRangeInfo objects
        """
        named_ranges = []

        try:
            # Workbook-level named ranges
            if hasattr(workbook, "defined_names"):
                for defn in workbook.defined_names.definedName:
                    try:
                        # Skip built-in names like _xlnm.Print_Area
                        if defn.name.startswith("_xlnm."):
                            continue

                        range_info = NamedRangeInfo(
                            name=defn.name,
                            refers_to=defn.value,
                            scope=defn.localSheetId
                            if hasattr(defn, "localSheetId") and defn.localSheetId is not None
                            else "Workbook",
                            is_hidden=defn.hidden if hasattr(defn, "hidden") else False,
                            comment=defn.comment if hasattr(defn, "comment") else None,
                        )
                        named_ranges.append(range_info)
                        self.logger.debug(
                            f"Found named range: {range_info.name} -> {range_info.refers_to}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to extract named range: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to extract named ranges: {e}")

        return named_ranges

    def _extract_print_areas_openpyxl(self, workbook: Any) -> list[PrintAreaInfo]:
        """Extract print areas from openpyxl workbook.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            List of PrintAreaInfo objects
        """
        print_areas = []

        try:
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                print_info = PrintAreaInfo(sheet_name=sheet_name)

                # Extract print area
                if hasattr(worksheet, "print_area") and worksheet.print_area:
                    print_info.print_area = str(worksheet.print_area)
                    self.logger.debug(f"Found print area in {sheet_name}: {print_info.print_area}")

                # Extract print titles (repeated rows/columns)
                if hasattr(worksheet, "print_title_rows") and worksheet.print_title_rows:
                    print_info.print_titles_rows = worksheet.print_title_rows

                if hasattr(worksheet, "print_title_cols") and worksheet.print_title_cols:
                    print_info.print_titles_cols = worksheet.print_title_cols

                # Only add if we found any print settings
                if (
                    print_info.print_area
                    or print_info.print_titles_rows
                    or print_info.print_titles_cols
                ):
                    print_areas.append(print_info)

        except Exception as e:
            self.logger.warning(f"Failed to extract print areas: {e}")

        return print_areas

    def _check_pivot_tables_openpyxl(self, workbook: Any) -> bool:
        """Check if workbook contains pivot tables.

        Args:
            workbook: openpyxl Workbook object

        Returns:
            True if pivot tables are found
        """
        try:
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                if hasattr(worksheet, "_pivots") and worksheet._pivots:
                    return True
        except Exception as e:
            self.logger.debug(f"Failed to check for pivot tables: {e}")

        return False

    def _extract_named_ranges_xlrd(self, workbook: Any) -> list[NamedRangeInfo]:
        """Extract named ranges from xlrd workbook (limited support).

        Args:
            workbook: xlrd Book object

        Returns:
            List of NamedRangeInfo objects (may be empty)
        """
        named_ranges = []

        try:
            if hasattr(workbook, "name_map"):
                for name, name_obj_list in workbook.name_map.items():
                    for name_obj in name_obj_list:
                        try:
                            # xlrd provides limited information about names
                            range_info = NamedRangeInfo(
                                name=name,
                                refers_to=str(name_obj.formula_text)
                                if hasattr(name_obj, "formula_text")
                                else "Unknown",
                                scope="Workbook",  # xlrd doesn't clearly distinguish scope
                                is_hidden=name_obj.hidden if hasattr(name_obj, "hidden") else False,
                            )
                            named_ranges.append(range_info)
                        except Exception as e:
                            self.logger.warning(f"Failed to extract xlrd named range {name}: {e}")

        except Exception as e:
            self.logger.debug(f"xlrd named range extraction not supported: {e}")

        return named_ranges

    def convert_to_detection_hints(self, metadata: ExcelMetadata) -> list[dict[str, Any]]:
        """Convert Excel metadata to detection hints for table detection.

        Args:
            metadata: ExcelMetadata object

        Returns:
            List of detection hints with range and confidence information
        """
        hints = []

        # Convert ListObjects with high confidence
        for table in metadata.list_objects:
            hints.append(
                {
                    "source": "excel_table",
                    "range": table.range_address,
                    "name": table.display_name,
                    "confidence": 0.95,  # Native Excel tables are highly reliable
                    "has_headers": table.has_headers,
                    "metadata": {
                        "table_style": table.table_style,
                        "has_totals": table.has_totals,
                    },
                }
            )

        # Convert named ranges with medium confidence
        for named_range in metadata.named_ranges:
            # Filter out non-data ranges (formulas, single cells, etc.)
            if self._is_likely_data_range(named_range.refers_to):
                hints.append(
                    {
                        "source": "named_range",
                        "range": named_range.refers_to,
                        "name": named_range.name,
                        "confidence": 0.7,  # Named ranges might not always be tables
                        "scope": named_range.scope,
                    }
                )

        # Convert print areas with lower confidence
        for print_area in metadata.print_areas:
            if print_area.print_area:
                hints.append(
                    {
                        "source": "print_area",
                        "range": f"{print_area.sheet_name}!{print_area.print_area}",
                        "sheet": print_area.sheet_name,
                        "confidence": 0.5,  # Print areas are just hints
                    }
                )

        return hints

    def _is_likely_data_range(self, refers_to: str) -> bool:
        """Check if a named range reference likely points to a data range.

        Args:
            refers_to: Excel reference string

        Returns:
            True if likely a data range (not a formula or single cell)
        """
        try:
            # Remove sheet name if present
            if "!" in refers_to:
                refers_to = refers_to.split("!")[-1]

            # Check if it's a range (contains colon)
            return ":" in refers_to

        except Exception:
            return False
