"""Analysis Agent - Responsible for semantic analysis and metadata generation."""

from typing import Any

from gridporter.agents.base_agent import BaseAgent
from gridporter.config import Config
from gridporter.models.file_data import FileData
from gridporter.models.table import ExtractedTable
from gridporter.tools.analysis import (
    analyze_field_semantics,
    build_metadata,
    classify_table_type,
    detect_relationships,
    generate_field_descriptions,
    generate_table_name,
)


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for understanding table semantics and generating metadata.

    This agent analyzes table purpose and structure, generates meaningful names,
    creates field descriptions, builds pandas import configurations, and provides
    business context.
    """

    def __init__(self, config: Config):
        """Initialize the analysis agent."""
        super().__init__(config)
        self.use_llm_for_naming = config.use_llm_for_table_naming

    async def execute(
        self, extracted_tables: list[ExtractedTable], file_data: FileData
    ) -> list[ExtractedTable]:
        """
        Analyze extracted tables to add semantic understanding.

        Args:
            extracted_tables: Tables with extracted data
            file_data: Original file data for context

        Returns:
            Tables enriched with semantic analysis and metadata
        """
        analyzed_tables = []

        # First pass: individual table analysis
        for table in extracted_tables:
            analyzed = await self._analyze_single_table(table, file_data)
            analyzed_tables.append(analyzed)

        # Second pass: inter-table relationships
        if len(analyzed_tables) > 1:
            relationships = detect_relationships(analyzed_tables)

            # Add relationship info to tables
            for table in analyzed_tables:
                table_relationships = [r for r in relationships if r.involves_table(table.info.id)]
                if table_relationships:
                    table.metadata["relationships"] = table_relationships

        return analyzed_tables

    async def _analyze_single_table(
        self, table: ExtractedTable, file_data: FileData
    ) -> ExtractedTable:
        """Analyze a single table."""
        # Classify table type
        table_type = classify_table_type(table.data, table.headers)

        # Analyze each field
        field_analysis = {}
        field_descriptions = {}

        for i, header in enumerate(table.headers):
            # Get column data
            column_data = [row[i] for row in table.data if i < len(row)]

            # Analyze field semantics
            semantics = analyze_field_semantics(column_data, header, table_type)

            field_analysis[header] = semantics

            # Generate human-readable description
            description = generate_field_descriptions(
                header,
                semantics,
                column_data[:5],  # Sample data
            )

            field_descriptions[header] = description

        # Generate table name
        suggested_name = await self._generate_table_name(table, table_type, file_data)

        # Build comprehensive metadata
        metadata = build_metadata(table, table_type, field_analysis, field_descriptions)

        # Update table with analysis results
        table.info.suggested_name = suggested_name
        table.metadata.update(metadata)
        table.metadata["table_type"] = table_type
        table.metadata["field_analysis"] = field_analysis
        table.metadata["field_descriptions"] = field_descriptions

        # Enhance pandas parameters with semantic understanding
        table.pandas_params = self._enhance_pandas_params(
            table.pandas_params, field_analysis, table_type
        )

        return table

    async def _generate_table_name(
        self, table: ExtractedTable, table_type: str, file_data: FileData
    ) -> str:
        """Generate a meaningful table name."""
        context = {
            "sheet_name": table.info.sheet_name,
            "table_type": table_type,
            "headers": table.headers[:10],  # First 10 headers
            "row_count": len(table.data),
            "file_name": file_data.filename,
        }

        if self.use_llm_for_naming:
            # Use LLM for more creative naming
            # This would call an LLM with naming prompt
            suggested_name = generate_table_name(table.headers, table_type, context, use_llm=True)
        else:
            # Use heuristic naming
            suggested_name = generate_table_name(table.headers, table_type, context, use_llm=False)

        # Ensure valid Python identifier
        suggested_name = self._sanitize_table_name(suggested_name)

        return suggested_name

    def _sanitize_table_name(self, name: str) -> str:
        """Ensure table name is a valid Python identifier."""
        # Replace spaces and special chars with underscore
        import re

        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Remove leading numbers
        name = re.sub(r"^[0-9]+", "", name)

        # Ensure not empty
        if not name:
            name = "table"

        # Convert to snake_case
        name = name.lower()

        # Remove duplicate underscores
        name = re.sub(r"_+", "_", name)

        # Remove trailing underscores
        name = name.strip("_")

        return name

    def _enhance_pandas_params(
        self, params: dict[str, Any], field_analysis: dict[str, Any], table_type: str
    ) -> dict[str, Any]:
        """Enhance pandas parameters with semantic understanding."""
        # Add semantic-based enhancements

        # Date parsing based on field analysis
        date_columns = []
        for field, analysis in field_analysis.items():
            if analysis.get("data_type") == "date" or analysis.get("semantic_type") == "date":
                date_columns.append(field)

        if date_columns:
            params["parse_dates"] = date_columns

            # Try to infer date format
            date_formats = {"iso": "%Y-%m-%d", "us": "%m/%d/%Y", "eu": "%d/%m/%Y"}

            # Add infer_datetime_format for flexibility
            params["infer_datetime_format"] = True

        # Category columns for memory efficiency
        category_columns = []
        for field, analysis in field_analysis.items():
            if (
                analysis.get("semantic_type") == "category"
                and analysis.get("cardinality", 999) < 50
            ):
                category_columns.append(field)

        if category_columns and "dtype" in params:
            for col in category_columns:
                if col in params["dtype"]:
                    params["dtype"][col] = "category"

        # Special handling for financial data
        if table_type in ["financial", "accounting"]:
            # Decimal precision for financial data
            params["float_precision"] = "high"

            # Look for currency symbols
            currency_columns = []
            for field, analysis in field_analysis.items():
                if analysis.get("unit") == "currency" or "$" in str(field):
                    currency_columns.append(field)

            if currency_columns:
                # Add converters to strip currency symbols
                if "converters" not in params:
                    params["converters"] = {}

                for col in currency_columns:
                    params["converters"][col] = lambda x: (
                        float(str(x).replace("$", "").replace(",", "")) if x else 0.0
                    )

        # Special handling for time series data
        if table_type == "time_series":
            # Look for index column
            time_column = None
            for field, analysis in field_analysis.items():
                if analysis.get("semantic_type") in ["date", "datetime", "timestamp"]:
                    time_column = field
                    break

            if time_column:
                params["index_col"] = time_column
                params["parse_dates"] = True

        # Add semantic metadata for downstream processing
        params["semantic_metadata"] = {
            "table_type": table_type,
            "detected_categories": category_columns,
            "detected_dates": date_columns,
            "field_roles": {
                field: analysis.get("role", "unknown") for field, analysis in field_analysis.items()
            },
        }

        # Add data quality parameters
        quality_params = self._assess_data_quality(field_analysis)
        params["quality_hints"] = quality_params

        return params

    def _assess_data_quality(self, field_analysis: dict[str, Any]) -> dict[str, Any]:
        """Assess data quality and provide hints."""
        quality = {
            "has_missing_values": False,
            "missing_columns": [],
            "sparse_columns": [],
            "high_cardinality_columns": [],
            "recommended_cleaning": [],
        }

        for field, analysis in field_analysis.items():
            # Check for missing values
            if analysis.get("null_rate", 0) > 0:
                quality["has_missing_values"] = True
                if analysis["null_rate"] > 0.5:
                    quality["sparse_columns"].append(field)

            # Check cardinality
            if analysis.get("cardinality", 0) > 1000:
                quality["high_cardinality_columns"].append(field)

            # Cleaning recommendations
            if analysis.get("has_outliers"):
                quality["recommended_cleaning"].append(
                    {
                        "column": field,
                        "issue": "outliers",
                        "suggestion": "Consider outlier removal or capping",
                    }
                )

            if analysis.get("has_mixed_types"):
                quality["recommended_cleaning"].append(
                    {
                        "column": field,
                        "issue": "mixed_types",
                        "suggestion": "Standardize data types before analysis",
                    }
                )

        return quality
