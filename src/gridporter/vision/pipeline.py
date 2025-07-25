"""Vision pipeline coordinator for table detection."""

import hashlib
import logging
from pathlib import Path

from ..config import GridPorterConfig
from ..models.sheet_data import SheetData
from ..models.vision_result import VisionAnalysisResult, VisionRegion
from .bitmap_generator import BitmapGenerator, BitmapMetadata
from .region_proposer import RegionProposal, RegionProposer
from .vision_models import VisionModel, create_vision_model

logger = logging.getLogger(__name__)


def _convert_proposals_to_regions(proposals: list[RegionProposal]) -> list[VisionRegion]:
    """Convert RegionProposal objects to VisionRegion models.

    Args:
        proposals: List of RegionProposal objects

    Returns:
        List of VisionRegion model instances
    """
    regions = []
    for proposal in proposals:
        region = VisionRegion(
            pixel_bounds={
                "x1": proposal.x1,
                "y1": proposal.y1,
                "x2": proposal.x2,
                "y2": proposal.y2,
            },
            cell_bounds={
                "start_row": proposal.start_row,
                "start_col": proposal.start_col,
                "end_row": proposal.end_row,
                "end_col": proposal.end_col,
            },
            range=proposal.excel_range,
            confidence=proposal.confidence,
            detection_method="vision",
            characteristics=proposal.characteristics,
        )
        regions.append(region)
    return regions


class VisionPipeline:
    """Orchestrates vision-based table detection pipeline."""

    def __init__(self, config: GridPorterConfig):
        """Initialize vision pipeline.

        Args:
            config: GridPorter configuration
        """
        self.config = config
        self.bitmap_generator = BitmapGenerator(
            cell_width=config.vision_cell_width,
            cell_height=config.vision_cell_height,
            mode=config.vision_mode,
        )
        self.region_proposer = RegionProposer()
        self._vision_model: VisionModel | None = None
        self._cache: dict[str, VisionAnalysisResult] = {}

    async def analyze_sheet(self, sheet_data: SheetData) -> VisionAnalysisResult:
        """Analyze a sheet using vision-based detection.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            Pipeline result with region proposals
        """
        # Generate cache key based on sheet content
        cache_key = self._generate_cache_key(sheet_data)

        # Check cache if enabled
        if self.config.enable_cache and cache_key in self._cache:
            logger.info("Using cached vision analysis result")
            result = self._cache[cache_key]
            # Create a copy with cached=True
            return VisionAnalysisResult(
                regions=result.regions,
                bitmap_info=result.bitmap_info,
                raw_response=result.raw_response,
                cached=True,
                metadata=result.metadata,
            )

        # Generate bitmap
        logger.info(f"Generating bitmap for sheet '{sheet_data.name}'")
        image_bytes, bitmap_metadata = self.bitmap_generator.generate(sheet_data)

        # Create prompt
        base_prompt = self.bitmap_generator.create_prompt_description(bitmap_metadata)
        full_prompt = self._create_analysis_prompt(base_prompt, sheet_data)

        # Get vision model
        if not self._vision_model:
            self._vision_model = create_vision_model(self.config)

        # Analyze with vision model
        logger.info(f"Analyzing bitmap with {self._vision_model.name}")
        model_response = await self._vision_model.analyze_image(image_bytes, full_prompt)

        # Parse response
        proposals = self.region_proposer.parse_response(model_response.content, bitmap_metadata)

        # Filter proposals
        proposals = self.region_proposer.filter_proposals(
            proposals,
            min_confidence=self.config.confidence_threshold,
            min_size=self.config.min_table_size,
        )

        # Convert proposals to VisionRegion objects
        regions = _convert_proposals_to_regions(proposals)

        # Create result
        result = VisionAnalysisResult(
            regions=regions,
            bitmap_info={
                "width": bitmap_metadata.width,
                "height": bitmap_metadata.height,
                "cell_width": bitmap_metadata.cell_width,
                "cell_height": bitmap_metadata.cell_height,
                "total_rows": bitmap_metadata.total_rows,
                "total_cols": bitmap_metadata.total_cols,
                "scale_factor": bitmap_metadata.scale_factor,
                "mode": bitmap_metadata.mode,
            },
            raw_response=model_response.content,
            cached=False,
        )

        # Cache result if enabled
        if self.config.enable_cache:
            self._cache[cache_key] = result

        logger.info(f"Vision analysis completed: {len(regions)} tables detected")
        return result

    def _create_analysis_prompt(self, base_prompt: str, sheet_data: SheetData) -> str:
        """Create comprehensive analysis prompt.

        Args:
            base_prompt: Base prompt from bitmap generator
            sheet_data: Sheet data for context

        Returns:
            Complete analysis prompt
        """
        # Add sheet-specific context
        context = []

        if sheet_data.name and sheet_data.name != "Sheet1":
            context.append(f"This is sheet '{sheet_data.name}' from a spreadsheet.")

        # Add any additional instructions
        instructions = [
            "Return results in JSON format with this structure:",
            '{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}, "confidence": 0.9, "characteristics": {"has_headers": true}}]}',
            "",
            "Be precise with pixel coordinates and conservative with confidence scores.",
            "Only identify regions that clearly form rectangular tables with distinct boundaries.",
        ]

        full_prompt = "\n".join([base_prompt] + context + [""] + instructions)
        return full_prompt

    def _generate_cache_key(self, sheet_data: SheetData) -> str:
        """Generate cache key for sheet data.

        Args:
            sheet_data: Sheet data

        Returns:
            Cache key string
        """
        # Create hash based on sheet content
        content_str = f"{sheet_data.name}_{sheet_data.max_row}_{sheet_data.max_column}"

        # Include cell data
        cell_data = []
        for address, cell in sorted(sheet_data.cells.items()):
            cell_str = f"{address}:{cell.value}:{cell.data_type}:{cell.is_bold}"
            cell_data.append(cell_str)

        content_str += "_" + "|".join(cell_data[:100])  # Limit to avoid huge keys

        # Add bitmap generation settings
        content_str += f"_{self.bitmap_generator.cell_width}_{self.bitmap_generator.cell_height}_{self.bitmap_generator.mode}"

        # Generate hash
        return hashlib.md5(content_str.encode()).hexdigest()

    async def batch_analyze(self, sheets: list[SheetData]) -> list[VisionAnalysisResult]:
        """Analyze multiple sheets, potentially in batch if model supports it.

        Args:
            sheets: List of sheet data to analyze

        Returns:
            List of pipeline results
        """
        results = []

        # Get vision model
        if not self._vision_model:
            self._vision_model = create_vision_model(self.config)

        if self._vision_model.supports_batch and len(sheets) > 1:
            # Model supports batch processing
            logger.info(f"Batch analyzing {len(sheets)} sheets")
            # TODO: Implement batch processing for models that support it
            # For now, process sequentially
            for sheet in sheets:
                result = await self.analyze_sheet(sheet)
                results.append(result)
        else:
            # Process sheets sequentially
            for sheet in sheets:
                result = await self.analyze_sheet(sheet)
                results.append(result)

        return results

    def save_debug_bitmap(self, sheet_data: SheetData, output_path: Path) -> BitmapMetadata:
        """Save bitmap for debugging purposes.

        Args:
            sheet_data: Sheet data to visualize
            output_path: Path to save PNG file

        Returns:
            Bitmap metadata
        """
        image_bytes, metadata = self.bitmap_generator.generate(sheet_data)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Debug bitmap saved to {output_path}")
        return metadata

    def clear_cache(self) -> None:
        """Clear the vision analysis cache."""
        self._cache.clear()
        logger.info("Vision pipeline cache cleared")

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "hits": 0,  # TODO: Track cache hits
            "misses": 0,  # TODO: Track cache misses
            "cache_size": len(self._cache),
            "cache_enabled": self.config.enable_cache,
        }
