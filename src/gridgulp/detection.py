"""Table detection orchestration for heuristic and Gemma-first flows."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ai import GemmaTableReasoner
from .config import DetectionStrategy
from .core.constants import ISLAND_DETECTION
from .detectors.box_table_detector import BoxTableDetector
from .detectors.island_detector import IslandDetector
from .detectors.simple_case_detector import SimpleCaseDetector
from .detectors.structured_text_detector import StructuredTextDetector
from .models.file_info import FileType
from .models.sheet_data import SheetData
from .models.table import TableInfo, TableRange
from .utils.logging_context import get_contextual_logger

logger = get_contextual_logger(__name__)


class DetectionResult(BaseModel):
    """Result from table detection."""

    model_config = ConfigDict(strict=True)

    tables: list[TableInfo] = Field(..., description="Detected tables")
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata",
    )


class TableDetectionAgent:
    """Hybrid heuristic and Gemma-first table detection orchestrator."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        file_type: FileType | None = None,
        config: Any | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.file_type = file_type
        self.config = config
        self.simple_detector = SimpleCaseDetector()
        self.box_detector = BoxTableDetector()
        self.structured_text_detector: StructuredTextDetector | None = None
        self.detection_strategy = (
            getattr(config, "detection_strategy", DetectionStrategy.HEURISTIC_ONLY)
            if config is not None
            else DetectionStrategy.HEURISTIC_ONLY
        )

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

        if file_type in (FileType.TXT, FileType.TSV):
            self.island_detector = IslandDetector(
                max_gap=ISLAND_DETECTION.TEXT_FILE_MAX_GAP,
                use_structural_analysis=True,
                adaptive_thresholds=adaptive_thresholds,
                empty_row_tolerance=config.empty_row_tolerance if config else 1,
                use_formatting_boundaries=config.use_border_detection if config else True,
            )
            self.structured_text_detector = StructuredTextDetector()
        else:
            self.island_detector = IslandDetector(
                max_gap=ISLAND_DETECTION.EXCEL_FILE_MAX_GAP,
                adaptive_thresholds=adaptive_thresholds,
                empty_row_tolerance=config.empty_row_tolerance if config else 1,
                use_formatting_boundaries=config.use_border_detection if config else True,
            )

        self.reasoner = None
        if self.detection_strategy == DetectionStrategy.GEMMA_FIRST and config is not None:
            self.reasoner = GemmaTableReasoner(config=config, file_type=file_type)

    async def detect_tables(self, sheet_data: SheetData) -> DetectionResult:
        """Detect tables using heuristics alone or heuristics plus Gemma refinement."""
        start_time = time.time()
        heuristic_tables, heuristic_metadata = self._generate_heuristic_proposals(sheet_data)

        tables = heuristic_tables
        methods_used = list(heuristic_metadata.get("methods_used", []))
        processing_metadata = {
            **heuristic_metadata,
            "detection_strategy": self.detection_strategy.value,
        }

        if self.reasoner is not None:
            gemma_tables, gemma_metadata = await self.reasoner.reason_about_tables(
                sheet_data,
                heuristic_tables,
            )
            tables = gemma_tables
            processing_metadata.update(gemma_metadata)
            methods_used.append("gemma_first")
            if gemma_metadata.get("vision_fallback_triggered"):
                methods_used.append("vision_fallback")
            if gemma_metadata.get("final_reasoner") == "multimodal":
                methods_used.append("gemma_multimodal")

        processing_metadata["processing_time"] = time.time() - start_time
        processing_metadata["table_count"] = len(tables)
        processing_metadata["methods_used"] = list(dict.fromkeys(methods_used))

        return DetectionResult(tables=tables, processing_metadata=processing_metadata)

    def _generate_heuristic_proposals(
        self,
        sheet_data: SheetData,
    ) -> tuple[list[TableInfo], dict[str, Any]]:
        total_sheet_cells = len(sheet_data.cells)
        sheet_area = (
            (sheet_data.max_row + 1) * (sheet_data.max_column + 1) if sheet_data.has_data() else 0
        )
        sheet_density = total_sheet_cells / sheet_area if sheet_area > 0 else 0.0
        cell_count = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

        tables: list[TableInfo] = []
        method_used = "none"
        simple_result = self.simple_detector.detect_simple_table(sheet_data)

        if simple_result.confidence >= 0.89 and cell_count > 10000:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                tables = [
                    TableInfo(
                        id=f"ultra_fast_{table_range.start_row}_{table_range.start_col}",
                        range=table_range,
                        confidence=simple_result.confidence,
                        detection_method="ultra_fast",
                        has_headers=simple_result.has_headers,
                        headers=headers,
                        sheet_name=sheet_data.name,
                    )
                ]
                method_used = "ultra_fast"

        elif simple_result.confidence >= 0.95:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                tables = [
                    TableInfo(
                        id=f"simple_case_fast_{table_range.start_row}_{table_range.start_col}",
                        range=table_range,
                        confidence=simple_result.confidence,
                        detection_method="simple_case_fast",
                        has_headers=simple_result.has_headers,
                        headers=headers,
                        sheet_name=sheet_data.name,
                    )
                ]
                method_used = "simple_case_fast"

        if not tables and self.file_type not in (FileType.TXT, FileType.TSV, FileType.CSV):
            box_tables = self.box_detector.detect_box_tables(sheet_data)
            if box_tables:
                for table in box_tables:
                    table.sheet_name = sheet_data.name
                tables = box_tables
                method_used = "box_table_detection"

        if not tables:
            if self.structured_text_detector and self.file_type in (FileType.TXT, FileType.TSV):
                tables = self.structured_text_detector.detect_tables(sheet_data)
                for table in tables:
                    table.sheet_name = sheet_data.name
                method_used = "structured_text_detection"
            else:
                islands = self.island_detector.detect_islands(sheet_data)
                good_islands = [i for i in islands if i.confidence >= self.confidence_threshold]
                if good_islands:
                    tables = self.island_detector.convert_to_table_infos(
                        good_islands,
                        sheet_data.name,
                        self.confidence_threshold,
                        sheet_data,
                    )
                    method_used = "island_detection_fast"

        if not tables and simple_result.confidence >= self.confidence_threshold:
            table_range = self._parse_range(simple_result.table_range)
            if table_range:
                headers = self.simple_detector._extract_headers(sheet_data, table_range)
                tables = [
                    TableInfo(
                        id=f"simple_case_{table_range.start_row}_{table_range.start_col}",
                        range=table_range,
                        confidence=simple_result.confidence,
                        detection_method="simple_case",
                        has_headers=simple_result.has_headers,
                        headers=headers,
                        sheet_name=sheet_data.name,
                    )
                ]
                method_used = "simple_case"

        if self.adaptive_thresholds and self.min_table_percentage > 0 and total_sheet_cells > 0:
            min_cells = int(total_sheet_cells * self.min_table_percentage)
            tables = [
                table
                for table in tables
                if self._get_table_cell_count(table, sheet_data) >= min_cells
            ]

        if self.prefer_large_tables and len(tables) > 1:
            tables.sort(key=lambda table: -self._get_table_cell_count(table, sheet_data))

        table_sizes = [self._get_table_cell_count(table, sheet_data) for table in tables]
        large_tables = (
            sum(1 for size in table_sizes if size >= total_sheet_cells * 0.05)
            if total_sheet_cells > 0
            else 0
        )
        medium_tables = (
            sum(
                1
                for size in table_sizes
                if total_sheet_cells * 0.01 <= size < total_sheet_cells * 0.05
            )
            if total_sheet_cells > 0
            else 0
        )
        small_tables = len(tables) - large_tables - medium_tables

        return tables, {
            "method_used": method_used,
            "methods_used": [method_used] if method_used != "none" else [],
            "sheet_cells": total_sheet_cells,
            "sheet_density": sheet_density,
            "large_tables": large_tables,
            "medium_tables": medium_tables,
            "small_tables": small_tables,
            "cell_count": cell_count,
            "heuristic_table_count": len(tables),
        }

    def _parse_range(self, range_str: str | None) -> TableRange | None:
        """Parse a range string into a TableRange."""
        if not range_str:
            return None

        try:
            return TableRange.from_excel(range_str)
        except Exception as exc:
            logger.warning(f"Failed to parse range {range_str}: {exc}")
            return None

    def _get_table_cell_count(self, table: TableInfo, sheet_data: SheetData) -> int:
        """Calculate the number of non-empty cells in a table."""
        count = 0
        for row in range(table.range.start_row, table.range.end_row + 1):
            for col in range(table.range.start_col, table.range.end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    count += 1
        return count


def detect_tables(
    sheet_data: SheetData,
    confidence_threshold: float = 0.6,
    file_type: FileType | None = None,
) -> list[TableInfo]:
    """Direct table detection function for convenience API usage."""
    import asyncio

    agent = TableDetectionAgent(confidence_threshold, file_type)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            def _run_async() -> DetectionResult:
                return asyncio.run(agent.detect_tables(sheet_data))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async)
                return future.result().tables
    except RuntimeError:
        pass

    return asyncio.run(agent.detect_tables(sheet_data)).tables
