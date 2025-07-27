"""Complex table detection agent using multi-agent architecture."""

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..detectors.format_analyzer import SemanticFormatAnalyzer, TableStructure
from ..detectors.merged_cell_analyzer import MergedCellAnalyzer
from ..detectors.multi_header_detector import MultiHeaderDetector, MultiRowHeader
from ..models.sheet_data import SheetData
from ..models.table import HeaderInfo, TableInfo, TableRange
from ..models.vision_result import VisionResult
from ..vision.pattern_detector import SparsePatternDetector

logger = logging.getLogger(__name__)

# Optional feature collection import
try:
    from ..telemetry import get_feature_collector

    HAS_FEATURE_COLLECTION = True
except ImportError:
    HAS_FEATURE_COLLECTION = False


class ComplexTableDetectionResult(BaseModel):
    """Result from complex table detection."""

    model_config = ConfigDict(strict=True)

    tables: list[TableInfo] = Field(..., description="Detected tables with full metadata")
    detection_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the detection process"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall detection confidence")


class ComplexTableAgent:
    """Agent for detecting complex tables with multi-row headers and semantic structure."""

    def __init__(self, config: Config):
        self.config = config
        self.multi_header_detector = MultiHeaderDetector()
        self.format_analyzer = SemanticFormatAnalyzer()
        self.merged_cell_analyzer = MergedCellAnalyzer()
        self.pattern_detector = SparsePatternDetector()

        # Initialize feature collector if enabled
        if HAS_FEATURE_COLLECTION and config.enable_feature_collection:
            feature_collector = get_feature_collector()
            if not feature_collector.enabled:
                feature_collector.initialize(enabled=True, db_path=config.feature_db_path)

    async def detect_complex_tables(
        self,
        sheet_data: SheetData,
        vision_result: VisionResult | None = None,
        simple_tables: list[TableRange] | None = None,
    ) -> ComplexTableDetectionResult:
        """
        Detect complex tables with multi-row headers and semantic structure.

        Args:
            sheet_data: Sheet data to analyze
            vision_result: Optional vision analysis results
            simple_tables: Optional list of simple table ranges to enhance

        Returns:
            ComplexTableDetectionResult with enhanced table information
        """
        logger.info("Starting complex table detection")

        # If we have simple tables, enhance them
        if simple_tables:
            tables = await self._enhance_simple_tables(sheet_data, simple_tables, vision_result)
        else:
            # Otherwise, detect tables from scratch
            tables = await self._detect_tables_from_scratch(sheet_data, vision_result)

        # Calculate overall confidence
        overall_confidence = sum(t.confidence for t in tables) / len(tables) if tables else 0.0

        return ComplexTableDetectionResult(
            tables=tables,
            detection_metadata={
                "methods_used": ["multi_header", "format_analysis", "merged_cell"],
                "vision_used": vision_result is not None,
                "table_count": len(tables),
            },
            confidence=overall_confidence,
        )

    async def _enhance_simple_tables(
        self,
        sheet_data: SheetData,
        simple_tables: list[TableRange],
        vision_result: VisionResult | None,
    ) -> list[TableInfo]:
        """Enhance simple table ranges with complex detection."""
        tasks = []
        for table_range in simple_tables:
            task = self._analyze_single_table(sheet_data, table_range, vision_result)
            tasks.append(task)

        # Run all analyses in parallel
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def _detect_tables_from_scratch(
        self, sheet_data: SheetData, vision_result: VisionResult | None
    ) -> list[TableInfo]:
        """Detect tables from scratch using vision and local analysis."""
        tables = []

        # Use vision results if available
        if vision_result and vision_result.regions:
            for region in vision_result.regions:
                table_range = TableRange(
                    start_row=region.start_row,
                    start_col=region.start_col,
                    end_row=region.end_row,
                    end_col=region.end_col,
                )
                table = await self._analyze_single_table(sheet_data, table_range, vision_result)
                if table:
                    tables.append(table)
        else:
            # Fallback to heuristic detection
            detected_ranges = self._detect_table_ranges_heuristically(sheet_data)
            for table_range in detected_ranges:
                table = await self._analyze_single_table(sheet_data, table_range, None)
                if table:
                    tables.append(table)

        return tables

    async def _analyze_single_table(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        vision_result: VisionResult | None,
    ) -> TableInfo | None:
        """Analyze a single table for complex structures."""
        try:
            # Detect multi-row headers
            multi_header = self.multi_header_detector.detect_multi_row_headers(
                sheet_data, table_range
            )

            # Determine header info
            if multi_header:
                header_info = self._create_header_info(multi_header)
                header_row_count = multi_header.end_row - multi_header.start_row + 1
            else:
                # Single row header
                header_info = self._create_simple_header_info(sheet_data, table_range)
                header_row_count = 1 if header_info else 0

            # Analyze semantic structure
            structure = self.format_analyzer.analyze_table_structure(
                sheet_data, table_range, header_row_count
            )

            # Extract pattern information if available
            pattern_info = self._extract_pattern_info(table_range, vision_result)

            # Build table info
            table = TableInfo(
                range=table_range,
                confidence=self._calculate_table_confidence(multi_header, structure, pattern_info),
                detection_method="complex_detection",
                header_info=header_info,
                has_headers=header_info is not None,
                semantic_structure=self._structure_to_dict(structure),
                format_preservation=self._extract_format_preservation(structure),
                suggested_name=(
                    await self._suggest_table_name(sheet_data, table_range, header_info)
                    if self.config.use_vision
                    else None
                ),
            )

            # Add data preview
            table.data_preview = self._generate_data_preview(
                sheet_data, table_range, header_row_count
            )

            # Infer data types
            table.data_types = self._infer_data_types(sheet_data, table_range, header_row_count)

            # Record features if collection is enabled
            if HAS_FEATURE_COLLECTION:
                try:
                    start_time = time.time()
                    feature_collector = get_feature_collector()
                    if feature_collector.enabled:
                        logger.debug(f"Recording features for table at {table.range.excel_range}")
                        # Build format features
                        format_features = {
                            "header_row_count": header_row_count,
                            "has_bold_headers": self._check_bold_headers(
                                sheet_data, table_range, header_row_count
                            ),
                            "has_totals": structure.has_grand_total,
                            "has_subtotals": structure.has_subtotals,
                            "section_count": len(structure.sections),
                            "separator_row_count": sum(
                                1
                                for r in structure.semantic_rows
                                if r.row_type.value == "separator"
                            ),
                        }

                        # Build hierarchical features
                        hierarchical_features = {
                            "has_multi_headers": multi_header is not None,
                            "max_hierarchy_depth": (
                                multi_header.end_row - multi_header.start_row + 1
                            )
                            if multi_header
                            else header_row_count,
                            "has_indentation": False,  # Multi-header is different from indentation
                        }

                        # Build content features
                        content_features = {
                            "total_cells": table_range.row_count * table_range.col_count,
                            "filled_cells": len([c for c in table.data_preview or [] if c]),
                            "numeric_ratio": sum(
                                1 for dt in (table.data_types or {}).values() if dt == "numeric"
                            )
                            / max(len(table.data_types or {"x": 1}), 1),
                            "date_columns": sum(
                                1 for dt in (table.data_types or {}).values() if dt == "date"
                            ),
                            "text_columns": sum(
                                1 for dt in (table.data_types or {}).values() if dt == "text"
                            ),
                        }

                        # Record the detection
                        processing_time = int((time.time() - start_time) * 1000)
                        feature_collector.record_detection(
                            file_path=getattr(sheet_data, "file_path", "unknown"),
                            file_type=getattr(sheet_data, "file_type", "unknown"),
                            sheet_name=getattr(sheet_data, "name", None),
                            table_range=table.range.excel_range,
                            detection_method="complex_detection",
                            confidence=table.confidence,
                            success=True,
                            format_features=format_features,
                            hierarchical_features=hierarchical_features,
                            content_features=content_features,
                            processing_time_ms=processing_time,
                        )
                except Exception as e:
                    logger.debug(f"Failed to record complex table features: {e}")

            return table

        except Exception as e:
            logger.error(f"Error analyzing table at {table_range.excel_range}: {e}")
            return None

    def _create_header_info(self, multi_header: MultiRowHeader) -> HeaderInfo:
        """Create HeaderInfo from multi-row header detection."""
        # Extract merged regions
        merged_regions = []
        for cell in multi_header.cells:
            if cell.is_merged:
                merged_regions.append(
                    {
                        "start_row": cell.row,
                        "start_col": cell.col,
                        "row_span": cell.row_span,
                        "col_span": cell.col_span,
                        "value": cell.value,
                    }
                )

        return HeaderInfo(
            row_count=multi_header.end_row - multi_header.start_row + 1,
            multi_row_headers=multi_header.column_mappings,
            merged_regions=merged_regions if merged_regions else None,
        )

    def _create_simple_header_info(
        self, sheet_data: SheetData, table_range: TableRange
    ) -> HeaderInfo | None:
        """Create HeaderInfo for single-row headers."""
        headers = []
        for col_offset in range(table_range.col_count):
            col_idx = table_range.start_col + col_offset
            cell = sheet_data.get_cell(table_range.start_row, col_idx)
            if cell and cell.value:
                headers.append(str(cell.value))
            else:
                headers.append(f"Column {col_offset + 1}")

        # Check if this looks like headers
        if all(h.startswith("Column ") for h in headers):
            return None  # No headers detected

        return HeaderInfo(row_count=1, headers=headers)

    def _calculate_table_confidence(
        self,
        multi_header: MultiRowHeader | None,
        structure: TableStructure,
        pattern_info: dict[str, Any] | None,
    ) -> float:
        """Calculate overall confidence for table detection."""
        scores = []

        # Multi-header confidence
        if multi_header:
            scores.append(multi_header.confidence)
        else:
            scores.append(0.7)  # Default for simple headers

        # Structure confidence
        semantic_rows = [r for r in structure.semantic_rows if r.row_type != "data"]
        structure_score = min(len(semantic_rows) / 5, 1.0)  # More semantic rows = higher confidence
        scores.append(structure_score)

        # Pattern confidence
        if pattern_info and pattern_info.get("confidence"):
            scores.append(pattern_info["confidence"])

        return sum(scores) / len(scores)

    def _structure_to_dict(self, structure: TableStructure) -> dict[str, Any]:
        """Convert TableStructure to dictionary."""
        return {
            "sections": structure.sections,
            "has_subtotals": structure.has_subtotals,
            "has_grand_total": structure.has_grand_total,
            "preserve_blank_rows": structure.preserve_blank_rows,
            "row_types": {row.row_index: row.row_type.value for row in structure.semantic_rows},
        }

    def _extract_format_preservation(self, structure: TableStructure) -> dict[str, Any]:
        """Extract formatting that should be preserved."""
        preservation = {
            "preserve_blank_rows": structure.preserve_blank_rows,
            "format_patterns": [],
        }

        for pattern in structure.format_patterns:
            preservation["format_patterns"].append(
                {
                    "type": pattern.pattern_type,
                    "rows": pattern.rows,
                    "cols": pattern.cols,
                    "value": pattern.value,
                }
            )

        return preservation

    def _extract_pattern_info(
        self, table_range: TableRange, vision_result: VisionResult | None
    ) -> dict[str, Any] | None:
        """Extract pattern information from vision results."""
        if not vision_result:
            return None

        # Find matching region
        for region in vision_result.regions:
            if (
                region.start_row == table_range.start_row
                and region.start_col == table_range.start_col
            ):
                return {
                    "pattern": region.pattern,
                    "confidence": region.confidence,
                    "description": region.description,
                }

        return None

    async def _suggest_table_name(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        header_info: HeaderInfo | None,
    ) -> str | None:
        """Suggest a name for the table based on its content."""
        # This would typically use an LLM
        # For now, return a simple name based on headers
        if header_info:
            if header_info.headers:
                # Use first few headers
                name_parts = header_info.headers[:3]
                return "_".join(name_parts).lower().replace(" ", "_")
            elif header_info.multi_row_headers:
                # Use bottom-level headers
                first_col_headers = header_info.multi_row_headers.get(0, [])
                if first_col_headers:
                    return first_col_headers[-1].lower().replace(" ", "_") + "_table"

        return None

    def _generate_data_preview(
        self, sheet_data: SheetData, table_range: TableRange, header_rows: int
    ) -> list[dict[str, Any]]:
        """Generate a preview of table data."""
        preview = []
        preview_rows = min(5, table_range.row_count - header_rows)

        # Get headers for column names
        headers = []
        if header_rows > 0:
            for col_offset in range(table_range.col_count):
                col_idx = table_range.start_col + col_offset
                # Use last header row for column names
                header_row_idx = table_range.start_row + header_rows - 1
                cell = sheet_data.get_cell(header_row_idx, col_idx)
                if cell and cell.value:
                    headers.append(str(cell.value))
                else:
                    headers.append(f"Column {col_offset + 1}")
        else:
            headers = [f"Column {i + 1}" for i in range(table_range.col_count)]

        # Get data rows
        for row_offset in range(header_rows, header_rows + preview_rows):
            if row_offset >= table_range.row_count:
                break

            row_idx = table_range.start_row + row_offset
            row_data = {}

            for col_offset in range(table_range.col_count):
                col_idx = table_range.start_col + col_offset
                cell = sheet_data.get_cell(row_idx, col_idx)
                value = cell.value if cell else None
                row_data[headers[col_offset]] = value

            preview.append(row_data)

        return preview

    def _infer_data_types(
        self, sheet_data: SheetData, table_range: TableRange, header_rows: int
    ) -> dict[str, str]:
        """Infer data types for each column."""
        data_types = {}

        # Sample rows for type inference
        sample_size = min(20, table_range.row_count - header_rows)

        for col_offset in range(table_range.col_count):
            col_idx = table_range.start_col + col_offset
            col_types = []

            # Sample data from this column
            for row_offset in range(header_rows, header_rows + sample_size):
                if row_offset >= table_range.row_count:
                    break

                row_idx = table_range.start_row + row_offset
                cell = sheet_data.get_cell(row_idx, col_idx)
                if cell and cell.value is not None:
                    col_types.append(cell.data_type)

            # Determine predominant type
            if col_types:
                type_counts = {}
                for t in col_types:
                    type_counts[t] = type_counts.get(t, 0) + 1

                # Get most common type
                predominant_type = max(type_counts, key=type_counts.get)

                # Normalize type names to match expected format
                type_mapping = {
                    "number": "numeric",
                    "string": "text",
                    "boolean": "text",  # Treat boolean as text for now
                    "datetime": "date",
                }
                predominant_type = type_mapping.get(predominant_type, predominant_type)

                # Use column name from headers
                if header_rows > 0:
                    header_cell = sheet_data.get_cell(
                        table_range.start_row + header_rows - 1, col_idx
                    )
                    col_name = (
                        str(header_cell.value)
                        if header_cell and header_cell.value
                        else f"Column {col_offset + 1}"
                    )
                else:
                    col_name = f"Column {col_offset + 1}"

                data_types[col_name] = predominant_type

        return data_types

    def _check_bold_headers(
        self, sheet_data: SheetData, table_range: TableRange, header_rows: int
    ) -> bool:
        """Check if headers contain bold formatting."""
        if header_rows == 0:
            return False

        bold_count = 0
        total_count = 0

        for row_offset in range(header_rows):
            row_idx = table_range.start_row + row_offset
            for col_offset in range(table_range.col_count):
                col_idx = table_range.start_col + col_offset
                cell = sheet_data.get_cell(row_idx, col_idx)
                if cell and cell.value is not None:
                    total_count += 1
                    if cell.is_bold:
                        bold_count += 1

        return bold_count > 0 and bold_count >= total_count * 0.5  # At least 50% bold

    def _detect_table_ranges_heuristically(self, sheet_data: SheetData) -> list[TableRange]:
        """Detect table ranges using heuristics when vision is not available."""
        # Simple heuristic: look for contiguous regions of data
        ranges = []

        # This is a simplified implementation
        # In practice, would use more sophisticated algorithms
        if sheet_data.max_row > 0 and sheet_data.max_column > 0:
            # For now, assume the whole sheet is one table
            ranges.append(
                TableRange(
                    start_row=0,
                    start_col=0,
                    end_row=sheet_data.max_row,
                    end_col=sheet_data.max_column,
                )
            )

        return ranges
