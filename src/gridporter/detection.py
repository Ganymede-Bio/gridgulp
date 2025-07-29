"""Simplified table detection module implementing Option 3 (Hybrid Approach).

This module provides a lightweight interface that uses only the proven detection
algorithms (SimpleCaseDetector and IslandDetector) that handle 97% of real-world cases.
"""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .detectors.island_detector import IslandDetector
from .detectors.simple_case_detector import SimpleCaseDetector
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

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.simple_detector = SimpleCaseDetector()
        self.island_detector = IslandDetector()

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
                    start_row=start_row, start_col=start_col, end_row=end_row, end_col=end_col
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
def detect_tables(sheet_data: SheetData, confidence_threshold: float = 0.6) -> list[TableInfo]:
    """Direct table detection function.

    This replaces the complex agent orchestration with direct
    algorithm calls that handle 97% of real-world cases.
    """
    agent = TableDetectionAgent(confidence_threshold)
    import asyncio

    # Run detection
    if asyncio.get_event_loop().is_running():
        # If we're already in an async context, we need to create a new loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, agent.detect_tables(sheet_data))
            result = future.result()
    else:
        result = asyncio.run(agent.detect_tables(sheet_data))

    return result.tables
