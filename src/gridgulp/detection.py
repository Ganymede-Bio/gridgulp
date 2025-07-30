"""Simplified table detection module implementing Option 3 (Hybrid Approach).

This module provides a lightweight interface that uses proven detection
algorithms (SimpleCaseDetector, BoxTableDetector, and IslandDetector) that handle most real-world cases.
"""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core.constants import ISLAND_DETECTION
from .detectors.box_table_detector import BoxTableDetector
from .detectors.island_detector import IslandDetector
from .detectors.simple_case_detector import SimpleCaseDetector
from .detectors.structured_text_detector import StructuredTextDetector
from .models.file_info import FileType
from .models.sheet_data import SheetData
from .models.table import TableInfo
from .utils.logging_context import get_contextual_logger

logger = get_contextual_logger(__name__)


class DetectionResult(BaseModel):
    """Result from table detection."""

    model_config = ConfigDict(strict=True)

    tables: list[TableInfo] = Field(..., description="Detected tables")
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata"
    )


class TableDetectionAgent:
    """Simplified table detection with minimal overhead."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        file_type: FileType | None = None,
        config: Any | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.file_type = file_type
        self.simple_detector = SimpleCaseDetector()
        self.box_detector = BoxTableDetector()
        self.structured_text_detector: StructuredTextDetector | None = None

        # Extract config values
        adaptive_thresholds = True
        min_table_percentage = 0.005
        prefer_large_tables = True
        if config:
            adaptive_thresholds = getattr(config, "adaptive_thresholds", True)
            min_table_percentage = getattr(config, "min_table_percentage", 0.005)
            prefer_large_tables = getattr(config, "prefer_large_tables", True)

        self.adaptive_thresholds = adaptive_thresholds
        self.min_table_percentage = min_table_percentage
        self.prefer_large_tables = prefer_large_tables

        # Configure island detector based on file type
        if file_type in (FileType.TXT, FileType.TSV):
            # Use stricter settings for text files
            self.island_detector = IslandDetector(
                max_gap=ISLAND_DETECTION.TEXT_FILE_MAX_GAP,
                use_structural_analysis=True,
                adaptive_thresholds=adaptive_thresholds,
                empty_row_tolerance=config.empty_row_tolerance if config else 1,
                use_formatting_boundaries=config.use_border_detection if config else True,
            )
            self.structured_text_detector = StructuredTextDetector()
        else:
            # Use default settings for Excel and other files
            self.island_detector = IslandDetector(
                max_gap=ISLAND_DETECTION.EXCEL_FILE_MAX_GAP,
                adaptive_thresholds=adaptive_thresholds,
                empty_row_tolerance=config.empty_row_tolerance if config else 1,
                use_formatting_boundaries=config.use_border_detection if config else True,
            )

    async def detect_tables(self, sheet_data: SheetData) -> DetectionResult:
        """Detect tables using fast-path algorithms."""
        start_time = time.time()

        # Calculate sheet statistics
        total_sheet_cells = len(sheet_data.cells)
        sheet_area = (
            (sheet_data.max_row + 1) * (sheet_data.max_column + 1) if sheet_data.has_data() else 0
        )
        sheet_density = total_sheet_cells / sheet_area if sheet_area > 0 else 0

        # Try fast paths in order of effectiveness
        tables = []
        method_used = "none"

        # Simple case (23% success rate)
        simple_result = self.simple_detector.detect_simple_table(sheet_data)

        # ULTRA-FAST path: For very large dense tables, skip all heavy processing
        cell_count = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
        if simple_result.confidence >= 0.89 and cell_count > 10000:
            logger.info(
                f"Ultra-fast path: Large dense table ({cell_count} cells) with perfect confidence"
            )
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                # Always extract headers from first row
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                table = TableInfo(
                    id=f"ultra_fast_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="ultra_fast",
                    has_headers=simple_result.has_headers,
                    headers=headers,
                )
                tables = [table]
                method_used = "ultra_fast"

        # High confidence simple case
        elif simple_result.confidence >= 0.95:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                # Always extract headers from first row
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                table = TableInfo(
                    id=f"simple_case_fast_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="simple_case_fast",
                    has_headers=simple_result.has_headers,
                    headers=headers,
                )
                tables = [table]
                method_used = "simple_case_fast"

        # Box table detection (high confidence border-based)
        if not tables and self.file_type not in (FileType.TXT, FileType.TSV, FileType.CSV):
            # Box table detector is primarily for Excel files with formatting
            box_tables = self.box_detector.detect_box_tables(sheet_data)
            if box_tables:
                logger.info(f"Box table detector found {len(box_tables)} tables")
                tables = box_tables
                method_used = "box_table_detection"

        # Multi-table detection (74% success rate)
        if not tables:
            # Use structured text detector for TSV/TXT files
            if self.structured_text_detector and self.file_type in (
                FileType.TXT,
                FileType.TSV,
            ):
                tables = self.structured_text_detector.detect_tables(sheet_data)
                method_used = "structured_text_detection"
            else:
                # Use regular island detection for other file types
                islands = self.island_detector.detect_islands(sheet_data)
                good_islands = [i for i in islands if i.confidence >= self.confidence_threshold]

                if good_islands:
                    # Convert islands to TableInfo objects
                    tables = self.island_detector.convert_to_table_infos(
                        good_islands, sheet_data.name, self.confidence_threshold, sheet_data
                    )
                    method_used = "island_detection_fast"

        # Fallback (3% success rate)
        if not tables and simple_result.confidence >= self.confidence_threshold:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                # Always extract headers from first row
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                table = TableInfo(
                    id=f"simple_case_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="simple_case",
                    has_headers=simple_result.has_headers,
                    headers=headers,
                )
                tables = [table]
                method_used = "simple_case"

        # Filter tables by relative size if adaptive thresholds are enabled
        if self.adaptive_thresholds and self.min_table_percentage > 0 and total_sheet_cells > 0:
            min_cells = int(total_sheet_cells * self.min_table_percentage)
            original_count = len(tables)
            tables = [t for t in tables if self._get_table_cell_count(t, sheet_data) >= min_cells]
            if original_count != len(tables):
                logger.info(
                    f"Filtered {original_count - len(tables)} small tables below {self.min_table_percentage:.1%} threshold"
                )

        # Sort tables by size if prefer_large_tables is enabled
        if self.prefer_large_tables and len(tables) > 1:
            tables.sort(key=lambda t: -self._get_table_cell_count(t, sheet_data))

        processing_time = time.time() - start_time

        # Add table statistics to metadata
        table_sizes = [self._get_table_cell_count(t, sheet_data) for t in tables]
        large_tables = (
            sum(1 for s in table_sizes if s >= total_sheet_cells * 0.05)
            if total_sheet_cells > 0
            else 0
        )
        medium_tables = (
            sum(1 for s in table_sizes if total_sheet_cells * 0.01 <= s < total_sheet_cells * 0.05)
            if total_sheet_cells > 0
            else 0
        )
        small_tables = len(tables) - large_tables - medium_tables

        return DetectionResult(
            tables=tables,
            processing_metadata={
                "method_used": method_used,
                "processing_time": processing_time,
                "sheet_cells": total_sheet_cells,
                "sheet_density": sheet_density,
                "table_count": len(tables),
                "large_tables": large_tables,
                "medium_tables": medium_tables,
                "small_tables": small_tables,
                "cell_count": cell_count,
                "performance": len(tables) > 0,
            },
        )

    def _parse_range(self, range_str: str | None) -> Any:
        """Parse range string into TableRange object."""
        if not range_str:
            return None

        # Import here to avoid circular imports
        from .models.table import TableRange

        try:
            # Parse Excel-style range like "A1:D10"
            if ":" in range_str:
                start_cell, end_cell = range_str.split(":")
                start_row, start_col = self._parse_cell(start_cell)
                end_row, end_col = self._parse_cell(end_cell)

                return TableRange(
                    start_row=start_row,
                    start_col=start_col,
                    end_row=end_row,
                    end_col=end_col,
                )
        except Exception as e:
            logger.warning(f"Failed to parse range {range_str}: {e}")

        return None

    def _parse_cell(self, cell_str: str) -> tuple[int, int]:
        """Parse cell string like 'A1' into (row, col) indices."""
        col_str = ""
        row_str = ""

        for char in cell_str:
            if char.isalpha():
                col_str += char
            else:
                row_str += char

        # Convert column letters to number (A=0, B=1, etc.)
        col = 0
        for char in col_str:
            col = col * 26 + (ord(char.upper()) - ord("A") + 1)
        col -= 1  # Convert to 0-based indexing

        # Convert row to number (1-based to 0-based)
        row = int(row_str) - 1

        return row, col

    def _get_table_cell_count(self, table: TableInfo, sheet_data: SheetData) -> int:
        """Calculate the number of non-empty cells in a table."""
        count = 0
        for row in range(table.range.start_row, table.range.end_row + 1):
            for col in range(table.range.start_col, table.range.end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    count += 1
        return count


# Convenience function for direct API usage
def detect_tables(
    sheet_data: SheetData,
    confidence_threshold: float = 0.6,
    file_type: FileType | None = None,
) -> list[TableInfo]:
    """Direct table detection function.

    This replaces the complex agent orchestration with direct
    algorithm calls that handle most real-world cases.

    Args:
        sheet_data: The sheet data to analyze
        confidence_threshold: Minimum confidence for table detection
        file_type: Type of file being processed (for strategy selection)

    Returns:
        List of detected tables
    """
    agent = TableDetectionAgent(confidence_threshold, file_type)
    import asyncio

    # Run detection
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to create a new loop
            import concurrent.futures

            def _run_async() -> DetectionResult:
                return asyncio.run(agent.detect_tables(sheet_data))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async)
                result = future.result()
        else:
            result = asyncio.run(agent.detect_tables(sheet_data))
    except RuntimeError:
        # No event loop
        result = asyncio.run(agent.detect_tables(sheet_data))

    return result.tables
