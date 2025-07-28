"""Vision Agent - Manages all vision model interactions for table detection."""

import json
from typing import Any, Optional

from gridporter.agents.base_agent import BaseAgent
from gridporter.config import Config
from gridporter.models.file_data import FileData
from gridporter.models.multi_scale import VisionImage, VisionRequest
from gridporter.models.table import TableInfo, TableRange
from gridporter.tools.vision import (
    build_vision_prompt,
    calculate_optimal_compression,
    generate_multi_scale_bitmaps,
    parse_vision_response,
)


class VisionAgent(BaseAgent):
    """
    Agent responsible for managing vision model interactions.

    This agent builds prompts, handles multi-image requests, implements
    retry logic, and manages vision API costs.
    """

    def __init__(self, config: Config):
        """Initialize the vision agent."""
        super().__init__(config)
        self.api_cost = 0.0
        self.total_tokens_used = 0

    async def execute(
        self,
        file_data: FileData,
        proposals: list[TableInfo],
        options: Optional[dict[str, Any]] = None,
    ) -> list[TableInfo]:
        """
        Refine table proposals using vision analysis.

        Args:
            file_data: The file being processed
            proposals: Initial table proposals to refine
            options: Processing options

        Returns:
            Refined table proposals with higher confidence
        """
        options = options or {}
        refined_proposals = []

        # Group proposals by sheet for efficient processing
        proposals_by_sheet = self._group_by_sheet(proposals)

        for sheet_name, sheet_proposals in proposals_by_sheet.items():
            sheet_data = file_data.get_sheet(sheet_name)

            # Build vision request for this sheet
            vision_request = await self._build_vision_request(sheet_data, sheet_proposals, options)

            # Execute vision analysis
            vision_results = await self._analyze_with_vision(vision_request, sheet_proposals)

            # Handle low confidence results
            if self._needs_refinement(vision_results):
                vision_results = await self._refine_with_feedback(
                    sheet_data, vision_results, vision_request
                )

            refined_proposals.extend(vision_results)

        self.update_state("vision_completed", True)
        self.update_state("api_cost", self.api_cost)

        return refined_proposals

    def _group_by_sheet(self, proposals: list[TableInfo]) -> dict[str, list[TableInfo]]:
        """Group proposals by sheet name."""
        grouped = {}
        for proposal in proposals:
            sheet = proposal.sheet_name or "Sheet1"
            if sheet not in grouped:
                grouped[sheet] = []
            grouped[sheet].append(proposal)
        return grouped

    async def _build_vision_request(
        self, sheet_data: Any, proposals: list[TableInfo], options: dict[str, Any]
    ) -> VisionRequest:
        """Build a vision request with appropriate images and prompts."""
        # Calculate optimal compression based on sheet size
        compression_strategy = calculate_optimal_compression(
            sheet_data.actual_bounds, target_size_mb=self.config.vision_max_image_size_mb
        )

        # Generate multi-scale bitmaps
        bitmaps = generate_multi_scale_bitmaps(
            sheet_data,
            regions=[self._proposal_to_region(p) for p in proposals],
            compression_levels=compression_strategy.compression_levels,
        )

        # Add context visualization to detail images
        context_margin = options.get("context_margin", 10)
        enhanced_images = []

        for image in bitmaps.images:
            if image.compression_level == 0:  # Full resolution
                # Add context highlighting
                for proposal in proposals:
                    image = self._add_context_to_image(image, proposal, context_margin)
            enhanced_images.append(image)

        # Build vision request
        return VisionRequest(
            images=enhanced_images,
            prompt_template=self._select_prompt_template(sheet_data, proposals),
            context_info={
                "sheet_name": sheet_data.name,
                "total_proposals": len(proposals),
                "context_margin": context_margin,
            },
        )

    def _proposal_to_region(self, proposal: TableInfo) -> dict[str, Any]:
        """Convert a table proposal to a region dict."""
        return {
            "id": proposal.id,
            "bounds": {
                "top": proposal.range.start_row,
                "left": proposal.range.start_col,
                "bottom": proposal.range.end_row,
                "right": proposal.range.end_col,
            },
            "confidence": proposal.confidence,
            "characteristics": proposal.metadata.get("characteristics", {}),
        }

    def _add_context_to_image(
        self, image: VisionImage, proposal: TableInfo, context_margin: int
    ) -> VisionImage:
        """Add context visualization to an image."""
        # This would use the highlight_region_with_context tool
        # For now, we'll just update the description
        image.context_info = {
            "highlighted_region": proposal.range.to_excel(),
            "context_cells": context_margin,
            "visualization": "red_border_with_yellow_context",
        }
        return image

    def _select_prompt_template(self, sheet_data: Any, proposals: list[TableInfo]) -> str:
        """Select appropriate prompt template based on context."""
        # Multiple proposals need different handling
        if len(proposals) > 3:
            return "MULTI_TABLE_DETECTION"

        # Large sheet needs progressive approach
        if sheet_data.actual_bounds.populated_cells > 1_000_000:
            return "PROGRESSIVE_REFINEMENT"

        # Standard multi-scale detection
        return "EXPLICIT_MULTI_SCALE"

    async def _analyze_with_vision(
        self, vision_request: VisionRequest, original_proposals: list[TableInfo]
    ) -> list[TableInfo]:
        """Execute vision analysis and parse results."""
        # Build the complete prompt
        prompt = build_vision_prompt(
            template=vision_request.prompt_template,
            images=vision_request.images,
            context={
                "proposals": [self._serialize_proposal(p) for p in original_proposals],
                "definitions": self._get_definitions(),
                "expected_output": self._get_output_schema(),
            },
        )

        # Call vision model (this would use actual API)
        response = await self._call_vision_api(prompt, vision_request.images)

        # Parse response
        parsed_results = parse_vision_response(response)

        # Convert to TableInfo objects
        refined_tables = []
        for result in parsed_results["tables"]:
            table = TableInfo(
                id=result["id"],
                range=TableRange(
                    start_row=result["bounds"]["top_row"],
                    start_col=result["bounds"]["left_col"],
                    end_row=result["bounds"]["bottom_row"],
                    end_col=result["bounds"]["right_col"],
                ),
                confidence=result["confidence"],
                detection_method="vision_analysis",
                suggested_name=result.get("description", ""),
                metadata={
                    "vision_evidence": result.get("evidence", {}),
                    "context_analysis": result.get("context_analysis", {}),
                },
            )
            refined_tables.append(table)

        return refined_tables

    def _serialize_proposal(self, proposal: TableInfo) -> dict[str, Any]:
        """Serialize a proposal for the prompt."""
        return {
            "id": proposal.id,
            "bounds": {
                "top": proposal.range.start_row,
                "left": proposal.range.start_col,
                "bottom": proposal.range.end_row,
                "right": proposal.range.end_col,
            },
            "initial_confidence": proposal.confidence,
            "detection_method": proposal.detection_method,
        }

    def _get_definitions(self) -> dict[str, str]:
        """Get definitions for the vision prompt."""
        return {
            "rectangularness": "The ratio of filled cells to total cells in the bounding box (0.0-1.0)",
            "consistency": "Degree to which cells in same column have same data type",
            "alignment": "Whether cells form straight edges without scattered data",
            "table": "A rectangular region with consistent structure, typically with headers",
            "context_area": "The cells surrounding a proposed table boundary (yellow highlight)",
            "natural_break": "Empty rows/columns or format changes indicating table boundary",
        }

    def _get_output_schema(self) -> dict[str, Any]:
        """Get expected output schema for vision model."""
        return {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "bounds", "confidence"],
                        "properties": {
                            "id": {"type": "string"},
                            "bounds": {
                                "type": "object",
                                "properties": {
                                    "top_row": {"type": "integer"},
                                    "left_col": {"type": "integer"},
                                    "bottom_row": {"type": "integer"},
                                    "right_col": {"type": "integer"},
                                },
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence": {"type": "object"},
                            "context_analysis": {"type": "object"},
                            "description": {"type": "string"},
                        },
                    },
                }
            },
        }

    async def _call_vision_api(self, prompt: str, images: list[VisionImage]) -> dict[str, Any]:
        """Call the vision API (placeholder for actual implementation)."""
        # This would call the actual vision model
        # For now, return a mock response

        # Track costs
        self.api_cost += 0.01 * len(images)
        self.total_tokens_used += 1000

        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tables": [
                                    {
                                        "id": "table_1",
                                        "bounds": {
                                            "top_row": 5,
                                            "left_col": 2,
                                            "bottom_row": 100,
                                            "right_col": 10,
                                        },
                                        "confidence": 0.95,
                                        "evidence": {"rectangularness": 0.92, "has_headers": True},
                                        "description": "Sales data table",
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 1000},
        }

    def _needs_refinement(self, results: list[TableInfo]) -> bool:
        """Check if results need refinement."""
        # Refine if any table has low confidence
        if any(t.confidence < 0.8 for t in results):
            return True

        # Refine if we got fewer tables than expected
        expected_count = self.get_state("expected_table_count", 0)
        if expected_count > 0 and len(results) < expected_count * 0.8:
            return True

        return False

    async def _refine_with_feedback(
        self, sheet_data: Any, initial_results: list[TableInfo], original_request: VisionRequest
    ) -> list[TableInfo]:
        """Refine results using feedback loop."""
        self.logger.info("Refining vision results with feedback")

        # Build feedback prompt
        feedback_prompt = self._build_feedback_prompt(initial_results)

        # Create focused images for problem areas
        problem_areas = [t for t in initial_results if t.confidence < 0.8]
        focused_images = []

        for table in problem_areas:
            # Generate detailed image of this specific area
            focused_image = self._create_focused_image(sheet_data, table)
            focused_images.append(focused_image)

        # Call vision again with feedback
        refined_request = VisionRequest(
            images=focused_images,
            prompt_template="REFINEMENT_WITH_FEEDBACK",
            context_info={"feedback": feedback_prompt, "original_results": initial_results},
        )

        # Get refined results
        refined_results = await self._analyze_with_vision(refined_request, initial_results)

        # Merge with high-confidence original results
        final_results = []
        for original in initial_results:
            if original.confidence >= 0.8:
                final_results.append(original)
            else:
                # Find refined version
                refined = next((r for r in refined_results if r.id == original.id), original)
                final_results.append(refined)

        return final_results

    def _build_feedback_prompt(self, results: list[TableInfo]) -> str:
        """Build feedback prompt for refinement."""
        issues = []

        for table in results:
            if table.confidence < 0.8:
                issues.append(
                    f"Table {table.id} has low confidence ({table.confidence:.2f}). "
                    f"Please verify boundaries and check for natural breaks."
                )

        return "\n".join(issues)

    def _create_focused_image(self, sheet_data: Any, table: TableInfo) -> VisionImage:
        """Create a focused image for a specific table area."""
        # This would generate a detailed image of just this table area
        # with extra context
        return VisionImage(
            image_id=f"focused_{table.id}",
            image_data="<base64_encoded_image>",
            compression_level=0,
            block_size=[1, 1],
            description=f"Detailed view of {table.id} with 20 cell context",
            purpose="boundary_refinement",
            covers_cells=table.range.to_excel(),
            size_bytes=50000,
        )
