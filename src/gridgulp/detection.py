"""Simplified table detection module implementing Option 3 (Hybrid Approach).

This module provides a lightweight interface that uses only the proven detection
algorithms (SimpleCaseDetector and IslandDetector) that handle most real-world cases.
"""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core.constants import ISLAND_DETECTION
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

    def __init__(self, confidence_threshold: float = 0.6, file_type: FileType | None = None):
        self.confidence_threshold = confidence_threshold
        self.file_type = file_type
        self.simple_detector = SimpleCaseDetector()
        self.structured_text_detector: StructuredTextDetector | None = None

        # Configure island detector based on file type
        if file_type in (FileType.TXT, FileType.TSV):
            # Use stricter settings for text files
            self.island_detector = IslandDetector(
                max_gap=ISLAND_DETECTION.TEXT_FILE_MAX_GAP, use_structural_analysis=True
            )
            self.structured_text_detector = StructuredTextDetector()
        else:
            # Use default settings for Excel and other files
            self.island_detector = IslandDetector(max_gap=ISLAND_DETECTION.EXCEL_FILE_MAX_GAP)

    async def detect_tables(self, sheet_data: SheetData) -> DetectionResult:
        """Detect tables using fast-path algorithms."""
        start_time = time.time()

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
                table = TableInfo(
                    id=f"ultra_fast_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="ultra_fast",
                    has_headers=simple_result.has_headers,
                )
                tables = [table]
                method_used = "ultra_fast"

        # High confidence simple case
        elif simple_result.confidence >= 0.95:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                table = TableInfo(
                    id=f"simple_case_fast_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="simple_case_fast",
                    has_headers=simple_result.has_headers,
                )
                tables = [table]
                method_used = "simple_case_fast"

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
                    tables = []
                    for _i, island in enumerate(good_islands):
                        range_str = (
                            island.to_range()
                        )  # Use to_range() method instead of table_range attribute
                        table_range = self._parse_range(range_str)
                        if table_range:
                            table = TableInfo(
                                id=f"island_detection_fast_{table_range.start_row}_{table_range.start_col}",
                                range=table_range,
                                confidence=island.confidence,
                                detection_method="island_detection_fast",
                                has_headers=island.has_headers,
                            )
                            tables.append(table)
                    method_used = "island_detection_fast"

        # Fallback (3% success rate)
        if not tables and simple_result.confidence >= self.confidence_threshold:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                table = TableInfo(
                    id=f"simple_case_{table_range.start_row}_{table_range.start_col}",
                    range=table_range,
                    confidence=simple_result.confidence,
                    detection_method="simple_case",
                    has_headers=simple_result.has_headers,
                )
                tables = [table]
                method_used = "simple_case"

        processing_time = time.time() - start_time

        return DetectionResult(
            tables=tables,
            processing_metadata={
                "method_used": method_used,
                "processing_time": processing_time,
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
