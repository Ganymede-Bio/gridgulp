"""Pipeline Orchestrator Agent - Coordinates the entire table detection pipeline."""

import time
from typing import Any, Optional

from gridporter.agents.analysis_agent import AnalysisAgent
from gridporter.agents.base_agent import BaseAgent
from gridporter.agents.detection_agent import DetectionAgent
from gridporter.agents.extraction_agent import ExtractionAgent
from gridporter.agents.vision_agent import VisionAgent
from gridporter.config import Config
from gridporter.models.file_data import FileData
from gridporter.models.table import TableInfo


class PipelineOrchestrator(BaseAgent):
    """
    Orchestrates the entire table detection pipeline.

    This agent makes high-level decisions about workflow, manages costs,
    coordinates other agents, and handles global error recovery.
    """

    def __init__(self, config: Config):
        """Initialize the orchestrator with all sub-agents."""
        super().__init__(config)

        # Initialize sub-agents
        self.detection_agent = DetectionAgent(config)
        self.vision_agent = VisionAgent(config)
        self.extraction_agent = ExtractionAgent(config)
        self.analysis_agent = AnalysisAgent(config)

        # Pipeline state
        self.total_cost = 0.0
        self.total_tables_detected = 0
        self.processing_times: dict[str, float] = {}

    async def execute(
        self, file_path: str, options: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Execute the complete table detection pipeline.

        Args:
            file_path: Path to the spreadsheet file
            options: Processing options

        Returns:
            Complete extraction result with all tables and metadata
        """
        start_time = time.time()
        options = options or {}

        try:
            # Phase 1: Detection
            self.logger.info(f"Starting pipeline for {file_path}")
            detection_start = time.time()

            file_data = await self._load_file(file_path, options)
            self.update_state("file_data", file_data)

            # Detect tables using appropriate strategy
            table_proposals = await self.detection_agent.execute(file_data, options)

            self.processing_times["detection"] = time.time() - detection_start
            self.logger.info(f"Detected {len(table_proposals)} potential tables")

            # Phase 2: Vision Analysis (if needed)
            if self._should_use_vision(table_proposals, options):
                vision_start = time.time()

                refined_proposals = await self.vision_agent.execute(
                    file_data, table_proposals, options
                )

                self.processing_times["vision"] = time.time() - vision_start
                self.total_cost += self.vision_agent.get_state("api_cost", 0.0)

                table_proposals = refined_proposals

            # Phase 3: Extraction
            extraction_start = time.time()

            extracted_tables = await self.extraction_agent.execute(file_data, table_proposals)

            self.processing_times["extraction"] = time.time() - extraction_start

            # Phase 4: Analysis
            analysis_start = time.time()

            analyzed_tables = await self.analysis_agent.execute(extracted_tables, file_data)

            self.processing_times["analysis"] = time.time() - analysis_start

            # Build final result
            result = self._build_final_result(file_data, analyzed_tables, time.time() - start_time)

            self.total_tables_detected += len(analyzed_tables)

            return result

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")

            # Attempt fallback strategy
            if self.should_use_fallback():
                return await self._fallback_extraction(file_path, options)
            else:
                raise

    async def _load_file(self, file_path: str, options: dict[str, Any]) -> FileData:
        """Load and validate the input file."""
        # This would use file loading tools
        # For now, returning a placeholder
        return FileData(path=file_path, format="xlsx")

    def _should_use_vision(self, proposals: list[TableInfo], options: dict[str, Any]) -> bool:
        """Decide whether vision analysis is needed."""
        # Skip vision if:
        # 1. Vision is disabled
        if not options.get("enable_vision", True):
            return False

        # 2. All proposals have high confidence
        if all(p.confidence > 0.9 for p in proposals):
            return False

        # 3. Cost limit would be exceeded
        estimated_cost = len(proposals) * 0.01  # Rough estimate
        if self.total_cost + estimated_cost > self.config.max_cost_per_session:
            self.logger.warning("Skipping vision due to cost limit")
            return False

        return True

    def _build_final_result(
        self, file_data: FileData, tables: list[TableInfo], total_time: float
    ) -> dict[str, Any]:
        """Build the final extraction result."""
        return {
            "status": "success",
            "file_info": {
                "path": file_data.path,
                "format": file_data.format,
                "sheets_processed": len(file_data.sheets) if hasattr(file_data, "sheets") else 1,
            },
            "tables": [self._serialize_table(t) for t in tables],
            "performance_metrics": {
                "total_time_seconds": total_time,
                "stages": self.processing_times,
                "total_cost_usd": self.total_cost,
            },
            "detection_summary": {
                "total_tables": len(tables),
                "methods_used": self._get_methods_used(),
                "vision_used": "vision" in self.processing_times,
            },
        }

    def _serialize_table(self, table: TableInfo) -> dict[str, Any]:
        """Serialize a table for output."""
        return {
            "id": table.id,
            "name": table.suggested_name,
            "range": table.range.to_excel(),
            "headers": table.headers,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "confidence": table.confidence,
            "pandas_config": table.metadata.get("pandas_config", {}),
            "field_descriptions": table.metadata.get("field_descriptions", {}),
        }

    def _get_methods_used(self) -> list[str]:
        """Get list of detection methods used."""
        methods = []

        if self.detection_agent.get_state("used_named_ranges"):
            methods.append("named_ranges")
        if self.detection_agent.get_state("used_list_objects"):
            methods.append("excel_tables")
        if self.vision_agent.get_state("vision_completed"):
            methods.append("vision_analysis")
        if self.detection_agent.get_state("used_heuristics"):
            methods.append("heuristic_detection")

        return methods

    async def _fallback_extraction(self, file_path: str, options: dict[str, Any]) -> dict[str, Any]:
        """Fallback extraction strategy when main pipeline fails."""
        self.logger.info("Using fallback extraction strategy")

        # Simple fallback: extract entire sheets as tables
        # This would be implemented with basic tools
        return {
            "status": "partial_success",
            "warning": "Used fallback extraction due to errors",
            "tables": [],
            "errors": [str(e) for e in self.state.get("errors", [])],
        }

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get statistics about pipeline execution."""
        return {
            "total_files_processed": self.state.get("files_processed", 0),
            "total_tables_detected": self.total_tables_detected,
            "total_cost": self.total_cost,
            "average_processing_time": sum(self.processing_times.values())
            / len(self.processing_times)
            if self.processing_times
            else 0,
            "error_rate": self.error_count / max(1, self.state.get("files_processed", 1)),
        }
