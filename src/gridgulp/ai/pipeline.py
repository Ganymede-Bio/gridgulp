"""Gemma-first table reasoning pipeline with optional multimodal fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gridgulp.ai.client import GemmaClient
from gridgulp.ai.parser import (
    GemmaDecision,
    GemmaDecisionAction,
    GemmaReasoningResult,
    GemmaResponseParser,
)
from gridgulp.ai.presenter import StructuredTextPresenter
from gridgulp.ai.prompts import GemmaMultimodalPromptBuilder, GemmaPromptBuilder
from gridgulp.ai.rendering import RenderedPromptImage, SheetImageRenderer
from gridgulp.config import Config
from gridgulp.detectors.format_analyzer import SemanticFormatAnalyzer
from gridgulp.detectors.merged_cell_analyzer import MergedCellAnalyzer
from gridgulp.detectors.multi_header_detector import MultiHeaderDetector
from gridgulp.models.file_info import FileType
from gridgulp.models.sheet_data import SheetData
from gridgulp.models.table import HeaderInfo, TableInfo, TableRange


class GemmaReasonerError(RuntimeError):
    """Raised when Gemma reasoning fails."""


class GemmaTableReasoner:
    """Run Gemma over structured sheet context and vision fallback when needed."""

    def __init__(self, config: Config, file_type: FileType | None = None) -> None:
        self.config = config
        self.file_type = file_type
        self.client = GemmaClient(config.gemma)
        self.presenter = StructuredTextPresenter()
        self.prompt_builder = GemmaPromptBuilder()
        self.multimodal_prompt_builder = GemmaMultimodalPromptBuilder()
        self.parser = GemmaResponseParser()
        self.multi_header_detector = MultiHeaderDetector()
        self.merged_cell_analyzer = MergedCellAnalyzer()
        self.semantic_analyzer = SemanticFormatAnalyzer()
        self.image_renderer = SheetImageRenderer(
            max_proposal_crops=config.vision.max_proposal_crops,
            crop_margin_cells=config.vision.crop_margin_cells,
        )

    async def reason_about_tables(
        self,
        sheet_data: SheetData,
        proposals: list[TableInfo],
    ) -> tuple[list[TableInfo], dict[str, Any]]:
        """Finalize tables using text-first Gemma reasoning with image fallback."""
        context = self.presenter.build_context(sheet_data, proposals)
        formatted_context = self.presenter.format_for_prompt(context)

        text_reasoning = await self._run_text_reasoning(formatted_context)
        text_tables = self._materialize_tables(sheet_data, text_reasoning)

        vision_trigger_reasons = self._collect_vision_trigger_reasons(
            sheet_data=sheet_data,
            proposals=proposals,
            text_reasoning=text_reasoning,
            text_tables=text_tables,
        )
        should_render_images = (
            bool(vision_trigger_reasons) or self.config.vision.debug_image_dir is not None
        )

        rendered_images: list[RenderedPromptImage] = []
        saved_debug_images: list[str] = []
        if should_render_images:
            rendered_images = self.image_renderer.render_bundle(sheet_data, proposals)
            if self.config.vision.debug_image_dir is not None:
                saved_debug_images = self.image_renderer.save_bundle(
                    rendered_images,
                    self._sheet_debug_dir(sheet_data),
                )

        final_reasoning = text_reasoning
        final_tables = text_tables
        final_reasoner = "text"
        vision_error: str | None = None

        if vision_trigger_reasons:
            try:
                final_reasoning = await self._run_multimodal_reasoning(
                    formatted_context=formatted_context,
                    rendered_images=rendered_images,
                )
                final_tables = self._materialize_tables(sheet_data, final_reasoning)
                final_reasoner = "multimodal"
            except Exception as exc:
                vision_error = str(exc)

        metadata = {
            "gemma": {
                "model_id": self.config.gemma.model_id,
                "decision_count": len(final_reasoning.decisions),
                "analysis_summary": final_reasoning.analysis_summary.model_dump(mode="json"),
                "text_analysis_summary": text_reasoning.analysis_summary.model_dump(mode="json"),
                "final_reasoner": final_reasoner,
            },
            "raw_prompt_context_length": len(formatted_context),
            "final_reasoner": final_reasoner,
            "vision_fallback_triggered": bool(vision_trigger_reasons),
            "vision_trigger_reason": ", ".join(vision_trigger_reasons)
            if vision_trigger_reasons
            else None,
            "vision_trigger_reasons": vision_trigger_reasons,
            "vision_image_count": len(rendered_images) if vision_trigger_reasons else 0,
            "vision_debug_images": saved_debug_images,
        }
        if vision_error is not None:
            metadata["vision_error"] = vision_error

        return final_tables, metadata

    async def _run_text_reasoning(self, formatted_context: str) -> GemmaReasoningResult:
        system_prompt, user_prompt = self.prompt_builder.build(formatted_context)
        try:
            raw_response = await self.client.chat_completion(system_prompt, user_prompt)
            return self.parser.parse(raw_response)
        except Exception as exc:
            raise GemmaReasonerError(f"Gemma reasoning failed: {exc}") from exc

    async def _run_multimodal_reasoning(
        self,
        *,
        formatted_context: str,
        rendered_images: list[RenderedPromptImage],
    ) -> GemmaReasoningResult:
        image_notes = [
            f"{image.kind}: {image.label} ({image.excel_range or 'sheet overview'})"
            for image in rendered_images
        ]
        system_prompt, user_prompt = self.multimodal_prompt_builder.build(
            formatted_context,
            image_notes,
        )
        try:
            raw_response = await self.client.multimodal_chat_completion(
                system_prompt,
                user_prompt,
                [image.data_uri for image in rendered_images],
            )
            return self.parser.parse(raw_response)
        except Exception as exc:
            raise GemmaReasonerError(f"Gemma multimodal reasoning failed: {exc}") from exc

    def _collect_vision_trigger_reasons(
        self,
        *,
        sheet_data: SheetData,
        proposals: list[TableInfo],
        text_reasoning: GemmaReasoningResult,
        text_tables: list[TableInfo],
    ) -> list[str]:
        if not self.config.vision.is_active():
            return []

        reasons: list[str] = []
        if sheet_data.has_data() and not text_tables:
            reasons.append("no_text_tables")
        if text_reasoning.analysis_summary.confidence < 0.85:
            reasons.append("low_text_confidence")
        if self._has_material_overlap(proposals):
            reasons.append("overlapping_proposals")
        if self._has_visual_header_complexity(sheet_data, proposals):
            reasons.append("header_complexity")
        if self._has_adjacent_blocks(proposals):
            reasons.append("adjacent_blocks")

        return list(dict.fromkeys(reasons))

    def _has_material_overlap(self, proposals: list[TableInfo]) -> bool:
        for index, left in enumerate(proposals):
            for right in proposals[index + 1 :]:
                if self._overlap_ratio(left.range, right.range) >= 0.25:
                    return True
        return False

    def _has_visual_header_complexity(
        self, sheet_data: SheetData, proposals: list[TableInfo]
    ) -> bool:
        for proposal in proposals:
            try:
                multi_header = self.multi_header_detector.detect_multi_row_headers(
                    sheet_data, proposal.range
                )
                if multi_header is not None and multi_header.end_row > multi_header.start_row:
                    return True
            except Exception:
                pass

            try:
                merged_cells = self.merged_cell_analyzer.analyze_merged_cells(
                    sheet_data, proposal.range
                )
                if self.merged_cell_analyzer.find_header_merged_cells(merged_cells):
                    return True
            except Exception:
                pass

        return False

    def _has_adjacent_blocks(self, proposals: list[TableInfo]) -> bool:
        if len(proposals) < 2:
            return False

        for index, left in enumerate(proposals):
            for right in proposals[index + 1 :]:
                row_gap = self._axis_gap(
                    left.range.start_row,
                    left.range.end_row,
                    right.range.start_row,
                    right.range.end_row,
                )
                col_gap = self._axis_gap(
                    left.range.start_col,
                    left.range.end_col,
                    right.range.start_col,
                    right.range.end_col,
                )
                row_overlap = self._axis_overlap_ratio(
                    left.range.start_row,
                    left.range.end_row,
                    right.range.start_row,
                    right.range.end_row,
                )
                col_overlap = self._axis_overlap_ratio(
                    left.range.start_col,
                    left.range.end_col,
                    right.range.start_col,
                    right.range.end_col,
                )

                if col_gap <= 2 and row_overlap >= 0.4:
                    return True
                if row_gap <= 1 and col_overlap >= 0.4:
                    return True

        return False

    def _materialize_tables(
        self,
        sheet_data: SheetData,
        reasoning: GemmaReasoningResult,
    ) -> list[TableInfo]:
        built_tables: list[TableInfo] = []

        for decision in reasoning.decisions:
            if decision.action == GemmaDecisionAction.REJECT:
                continue
            if decision.range is None:
                continue

            table_range = self._clip_range(TableRange.from_excel(decision.range), sheet_data)
            if table_range is None:
                continue
            if self._non_empty_cells(sheet_data, table_range) == 0:
                continue

            base_headers = self._extract_headers(sheet_data, table_range)
            header_info = self._extract_header_info(sheet_data, table_range, base_headers)
            semantic_structure = self._extract_semantic_structure(sheet_data, table_range)

            built_tables.append(
                TableInfo(
                    id=self._build_table_id(decision, table_range),
                    range=table_range,
                    suggested_name=decision.suggested_name,
                    confidence=decision.confidence,
                    detection_method=f"gemma_{decision.action.value}",
                    sheet_name=sheet_data.name,
                    headers=base_headers,
                    header_info=header_info,
                    has_headers=bool(base_headers),
                    semantic_structure=semantic_structure,
                    metadata={
                        "gemma_action": decision.action.value,
                        "proposal_ids": decision.proposal_ids,
                        "reason": decision.reason,
                        "file_type": self.file_type.value if self.file_type else None,
                    },
                )
            )

        return self._deduplicate_tables(built_tables)

    def _build_table_id(self, decision: GemmaDecision, table_range: TableRange) -> str:
        proposal_segment = "_".join(decision.proposal_ids) if decision.proposal_ids else "new"
        return (
            f"gemma_{decision.action.value}_{proposal_segment}_"
            f"{table_range.start_row}_{table_range.start_col}"
        )

    def _clip_range(self, table_range: TableRange, sheet_data: SheetData) -> TableRange | None:
        max_row = sheet_data.max_row
        max_col = sheet_data.max_column
        if max_row < 0 or max_col < 0:
            return None

        start_row = min(max(table_range.start_row, 0), max_row)
        start_col = min(max(table_range.start_col, 0), max_col)
        end_row = min(max(table_range.end_row, start_row), max_row)
        end_col = min(max(table_range.end_col, start_col), max_col)
        if end_row < start_row or end_col < start_col:
            return None
        return TableRange(
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
        )

    def _extract_headers(self, sheet_data: SheetData, table_range: TableRange) -> list[str] | None:
        headers = []
        for col in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(table_range.start_row, col)
            headers.append("" if cell is None or cell.value is None else str(cell.value).strip())
        return headers if any(header for header in headers) else None

    def _extract_header_info(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        headers: list[str] | None,
    ) -> HeaderInfo | None:
        try:
            multi_header = self.multi_header_detector.detect_multi_row_headers(
                sheet_data, table_range
            )
        except Exception:
            multi_header = None

        if multi_header is None:
            return HeaderInfo(row_count=1, headers=headers)

        return HeaderInfo(
            row_count=multi_header.end_row - multi_header.start_row + 1,
            headers=headers,
            multi_row_headers=multi_header.column_mappings,
            merged_regions=[
                {
                    "row": cell.row,
                    "col": cell.col,
                    "row_span": cell.row_span,
                    "col_span": cell.col_span,
                    "value": cell.value,
                }
                for cell in multi_header.cells
                if cell.is_merged
            ],
        )

    def _extract_semantic_structure(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
    ) -> dict[str, Any] | None:
        try:
            structure = self.semantic_analyzer.analyze_table_structure(sheet_data, table_range)
        except Exception:
            return None

        return {
            "sections": structure.sections,
            "has_subtotals": structure.has_subtotals,
            "has_grand_total": structure.has_grand_total,
            "preserve_blank_rows": structure.preserve_blank_rows,
        }

    def _deduplicate_tables(self, tables: list[TableInfo]) -> list[TableInfo]:
        deduplicated: list[TableInfo] = []
        ordered_tables = sorted(
            tables,
            key=lambda table: (-table.confidence, table.range.start_row, table.range.start_col),
        )
        for candidate in ordered_tables:
            if any(
                self._overlap_ratio(candidate.range, existing.range) >= 0.9
                for existing in deduplicated
            ):
                continue
            deduplicated.append(candidate)
        return sorted(
            deduplicated, key=lambda table: (table.range.start_row, table.range.start_col)
        )

    def _overlap_ratio(self, left: TableRange, right: TableRange) -> float:
        start_row = max(left.start_row, right.start_row)
        start_col = max(left.start_col, right.start_col)
        end_row = min(left.end_row, right.end_row)
        end_col = min(left.end_col, right.end_col)
        if end_row < start_row or end_col < start_col:
            return 0.0

        intersection = (end_row - start_row + 1) * (end_col - start_col + 1)
        left_area = left.row_count * left.col_count
        right_area = right.row_count * right.col_count
        return intersection / min(left_area, right_area)

    def _axis_gap(
        self,
        left_start: int,
        left_end: int,
        right_start: int,
        right_end: int,
    ) -> int:
        if left_end >= right_start and right_end >= left_start:
            return 0
        if left_end < right_start:
            return right_start - left_end - 1
        return left_start - right_end - 1

    def _axis_overlap_ratio(
        self,
        left_start: int,
        left_end: int,
        right_start: int,
        right_end: int,
    ) -> float:
        overlap = min(left_end, right_end) - max(left_start, right_start) + 1
        if overlap <= 0:
            return 0.0
        left_span = left_end - left_start + 1
        right_span = right_end - right_start + 1
        return overlap / min(left_span, right_span)

    def _non_empty_cells(self, sheet_data: SheetData, table_range: TableRange) -> int:
        count = 0
        for row in range(table_range.start_row, table_range.end_row + 1):
            for col in range(table_range.start_col, table_range.end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell is not None and cell.value is not None:
                    count += 1
        return count

    def _sheet_debug_dir(self, sheet_data: SheetData) -> Path:
        debug_image_dir = self.config.vision.debug_image_dir
        if debug_image_dir is None:
            raise ValueError("Vision debug image directory is not configured.")

        base_dir = debug_image_dir.expanduser()
        file_stem = "sheet"
        if sheet_data.file_path:
            file_stem = Path(sheet_data.file_path).stem
        sheet_name = self._sanitize_path_component(sheet_data.name)
        return base_dir / file_stem / sheet_name

    def _sanitize_path_component(self, value: str) -> str:
        cleaned = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value
        )
        return cleaned.strip("_") or "sheet"
