"""Extraction Agent - Responsible for extracting data from detected tables."""

from typing import Any, Optional

from gridporter.agents.base_agent import BaseAgent
from gridporter.config import Config
from gridporter.models.file_data import FileData
from gridporter.models.table import ExtractedTable, TableInfo
from gridporter.tools.extraction import (
    detect_hierarchy,
    extract_cell_data,
    extract_cell_formats,
    handle_formulas,
    parse_headers,
    resolve_merged_cells,
)


class ExtractionAgent(BaseAgent):
    """
    Agent responsible for extracting data from detected tables.

    This agent determines extraction complexity, handles special structures
    like merged cells and formulas, and preserves formatting when needed.
    """

    def __init__(self, config: Config):
        """Initialize the extraction agent."""
        super().__init__(config)

    async def execute(
        self, file_data: FileData, table_proposals: list[TableInfo]
    ) -> list[ExtractedTable]:
        """
        Extract data from all detected tables.

        Args:
            file_data: The file containing the tables
            table_proposals: Verified table boundaries

        Returns:
            List of extracted tables with data and metadata
        """
        extracted_tables = []

        for proposal in table_proposals:
            sheet_data = file_data.get_sheet(proposal.sheet_name)

            # Determine extraction complexity
            complexity = self._assess_complexity(sheet_data, proposal)

            # Choose extraction strategy
            if complexity["is_complex"]:
                extracted = await self._complex_extraction(sheet_data, proposal, complexity)
            else:
                extracted = await self._simple_extraction(sheet_data, proposal)

            # Add pandas ingestion parameters
            extracted.pandas_params = self._generate_pandas_params(extracted, file_data.format)

            extracted_tables.append(extracted)

        return extracted_tables

    def _assess_complexity(self, sheet_data: Any, proposal: TableInfo) -> dict[str, Any]:
        """Assess the complexity of table extraction."""
        complexity = {
            "is_complex": False,
            "has_merged_cells": False,
            "has_multi_row_headers": False,
            "has_formulas": False,
            "has_hierarchy": False,
            "has_special_formatting": False,
        }

        # Check for merged cells
        merged_cells = resolve_merged_cells(sheet_data, proposal.range)
        if merged_cells:
            complexity["has_merged_cells"] = True
            complexity["is_complex"] = True
            complexity["merged_regions"] = merged_cells

        # Quick header analysis
        header_structure = parse_headers(sheet_data, proposal.range, max_header_rows=5)

        if header_structure.row_count > 1:
            complexity["has_multi_row_headers"] = True
            complexity["is_complex"] = True
            complexity["header_rows"] = header_structure.row_count

        # Check for formulas
        if self._has_formulas(sheet_data, proposal.range):
            complexity["has_formulas"] = True
            complexity["is_complex"] = True

        # Check for hierarchical data
        hierarchy = detect_hierarchy(sheet_data, proposal.range)
        if hierarchy.levels > 1:
            complexity["has_hierarchy"] = True
            complexity["is_complex"] = True
            complexity["hierarchy_info"] = hierarchy

        return complexity

    async def _simple_extraction(self, sheet_data: Any, proposal: TableInfo) -> ExtractedTable:
        """Simple extraction for basic tables."""
        # Extract raw data
        raw_data = extract_cell_data(sheet_data, proposal.range)

        # Simple header detection (first row)
        headers = raw_data[0] if raw_data else []
        data_rows = raw_data[1:] if len(raw_data) > 1 else []

        # Basic type inference
        column_types = self._infer_column_types(data_rows)

        return ExtractedTable(
            info=proposal,
            headers=headers,
            data=data_rows,
            column_types=column_types,
            extraction_method="simple",
            metadata={"row_count": len(data_rows), "column_count": len(headers)},
        )

    async def _complex_extraction(
        self, sheet_data: Any, proposal: TableInfo, complexity: dict[str, Any]
    ) -> ExtractedTable:
        """Complex extraction handling special structures."""
        # Extract raw data with metadata
        raw_data = extract_cell_data(sheet_data, proposal.range)

        # Extract formatting if needed
        formats = None
        if complexity.get("has_special_formatting"):
            formats = extract_cell_formats(sheet_data, proposal.range)

        # Handle multi-row headers
        if complexity.get("has_multi_row_headers"):
            header_info = parse_headers(
                sheet_data, proposal.range, max_header_rows=complexity.get("header_rows", 5)
            )
            headers = header_info.unified_headers
            data_start_row = header_info.data_start_row
        else:
            headers = raw_data[0] if raw_data else []
            data_start_row = 1

        data_rows = raw_data[data_start_row:] if len(raw_data) > data_start_row else []

        # Handle formulas
        if complexity.get("has_formulas"):
            formula_map = handle_formulas(sheet_data, proposal.range)
        else:
            formula_map = {}

        # Handle hierarchy
        hierarchy_metadata = {}
        if complexity.get("has_hierarchy"):
            hierarchy_info = complexity["hierarchy_info"]
            hierarchy_metadata = {
                "hierarchy_levels": hierarchy_info.levels,
                "indent_pattern": hierarchy_info.indent_pattern,
                "parent_child_map": hierarchy_info.parent_child_map,
            }

        # Infer column types considering complexity
        column_types = self._infer_column_types(data_rows, headers)

        return ExtractedTable(
            info=proposal,
            headers=headers,
            data=data_rows,
            column_types=column_types,
            extraction_method="complex",
            metadata={
                "row_count": len(data_rows),
                "column_count": len(headers),
                "complexity": complexity,
                "merged_cells": complexity.get("merged_regions", []),
                "formula_cells": formula_map,
                "hierarchy": hierarchy_metadata,
                "formats": formats,
                "header_row_count": complexity.get("header_rows", 1),
            },
        )

    def _has_formulas(self, sheet_data: Any, range: Any) -> bool:
        """Check if the range contains formulas."""
        # This would check for formula cells
        # Placeholder implementation
        return False

    def _infer_column_types(
        self, data_rows: list[list[Any]], headers: Optional[list[str]] = None
    ) -> dict[str, str]:
        """Infer column data types."""
        if not data_rows or not data_rows[0]:
            return {}

        column_types = {}
        num_cols = len(data_rows[0])

        for col_idx in range(num_cols):
            col_name = headers[col_idx] if headers and col_idx < len(headers) else f"col_{col_idx}"
            col_values = [row[col_idx] for row in data_rows if col_idx < len(row)]

            # Infer type from values
            col_type = self._infer_single_column_type(col_values)
            column_types[col_name] = col_type

        return column_types

    def _infer_single_column_type(self, values: list[Any]) -> str:
        """Infer type for a single column."""
        # Filter out None/empty values
        non_empty = [v for v in values if v is not None and str(v).strip()]

        if not non_empty:
            return "object"  # Default to object for empty columns

        # Check if all numeric
        try:
            numeric_values = [float(str(v).replace(",", "")) for v in non_empty]

            # Check if all integers
            if all(v.is_integer() for v in numeric_values):
                return "int64"
            else:
                return "float64"
        except ValueError:
            pass

        # Check for dates (simple heuristic)
        date_patterns = ["/", "-", "2020", "2021", "2022", "2023", "2024"]
        if any(pattern in str(non_empty[0]) for pattern in date_patterns):
            # Could be dates
            return "datetime64"

        # Default to string/object
        return "object"

    def _generate_pandas_params(
        self, extracted_table: ExtractedTable, file_format: str
    ) -> dict[str, Any]:
        """Generate pandas parameters for reading this specific table."""
        params = {}

        # Determine read function
        if file_format in ["xlsx", "xls", "xlsm", "xlsb"]:
            params["read_function"] = "read_excel"
        elif file_format == "csv":
            params["read_function"] = "read_csv"
        elif file_format == "tsv":
            params["read_function"] = "read_csv"
            params["sep"] = "\t"
        else:
            params["read_function"] = "read_excel"  # Default

        # Common parameters
        table_range = extracted_table.info.range

        # For read_excel
        if params["read_function"] == "read_excel":
            params["sheet_name"] = extracted_table.info.sheet_name

            # Convert to Excel-style range
            excel_range = table_range.to_excel()
            params["usecols"] = excel_range

            # Handle headers
            if extracted_table.metadata.get("header_row_count", 1) > 1:
                # Multi-row headers
                header_rows = list(
                    range(
                        table_range.start_row,
                        table_range.start_row + extracted_table.metadata["header_row_count"],
                    )
                )
                params["header"] = header_rows
            else:
                params["header"] = table_range.start_row

            # Skip rows before table
            if table_range.start_row > 0:
                params["skiprows"] = list(range(table_range.start_row))

            # Number of rows to read
            params["nrows"] = table_range.end_row - table_range.start_row + 1

        # For read_csv
        elif params["read_function"] == "read_csv":
            # Skip rows before table
            params["skiprows"] = table_range.start_row

            # Use column positions
            params["usecols"] = list(range(table_range.start_col, table_range.end_col + 1))

            # Number of rows
            params["nrows"] = table_range.end_row - table_range.start_row + 1

            # Header handling
            if extracted_table.metadata.get("header_row_count", 1) > 1:
                # For CSV, we need to handle multi-row headers differently
                params["header"] = list(range(extracted_table.metadata["header_row_count"]))
            else:
                params["header"] = 0

        # Data type hints
        dtype_map = {}
        for col_name, col_type in extracted_table.column_types.items():
            if col_type == "int64":
                dtype_map[col_name] = "Int64"  # Nullable integer
            elif col_type == "float64":
                dtype_map[col_name] = "float64"
            elif col_type == "datetime64":
                # Add to parse_dates instead
                if "parse_dates" not in params:
                    params["parse_dates"] = []
                params["parse_dates"].append(col_name)
            else:
                dtype_map[col_name] = "object"

        if dtype_map:
            params["dtype"] = dtype_map

        # Additional parameters based on data characteristics

        # Missing value handling
        params["na_values"] = ["", "N/A", "NA", "null", "NULL", "-", "--"]

        # For numeric data with thousands separators
        if any(t in ["int64", "float64"] for t in extracted_table.column_types.values()):
            params["thousands"] = ","

        # Hierarchy handling
        if extracted_table.metadata.get("hierarchy"):
            hierarchy_info = extracted_table.metadata["hierarchy"]
            params["index_col"] = 0  # First column usually contains hierarchy

            # Add post-processing instructions
            params["post_processing"] = {
                "convert_hierarchy": True,
                "hierarchy_levels": hierarchy_info["hierarchy_levels"],
                "method": "indentation_to_multiindex",
            }

        # Add table-specific metadata
        params["table_metadata"] = {
            "table_id": extracted_table.info.id,
            "table_name": extracted_table.info.suggested_name,
            "confidence": extracted_table.info.confidence,
            "has_merged_cells": bool(extracted_table.metadata.get("merged_cells")),
            "has_formulas": bool(extracted_table.metadata.get("formula_cells")),
        }

        return params
