"""Integrated vision pipeline using bitmap-first approach."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..models.sheet_data import SheetData
from .bitmap_analyzer import BitmapAnalyzer
from .bitmap_generator import BitmapGenerator
from .pattern_detector import (
    PatternType,
    SparsePatternDetector,
    TableBounds,
    TablePattern,
)
from .quadtree import QuadtreeAnalyzer, VisualizationRegion

# Optional telemetry import
try:
    from ..telemetry.metrics import get_metrics_collector

    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from the integrated pipeline."""

    detected_tables: list[TablePattern]
    visualization_regions: list[VisualizationRegion]
    analysis_metadata: dict[str, Any]


class IntegratedVisionPipeline:
    """Multi-scale vision pipeline using bitmap-first detection."""

    def __init__(
        self,
        min_table_size: tuple[int, int] = (2, 2),
        min_density: float = 0.1,
        max_regions_for_llm: int = 10,
    ):
        """Initialize integrated pipeline.

        Args:
            min_table_size: Minimum (rows, cols) to consider as table
            min_density: Minimum density for regions
            max_regions_for_llm: Maximum regions to send to LLM
        """
        self.bitmap_analyzer = BitmapAnalyzer(
            min_region_size=min_table_size, min_density=min_density
        )
        self.bitmap_generator = BitmapGenerator(auto_compress=True)
        self.pattern_detector = SparsePatternDetector(
            min_filled_ratio=min_density, min_table_size=min_table_size
        )
        self.quadtree_analyzer = QuadtreeAnalyzer(pattern_aware=True)
        self.max_regions_for_llm = max_regions_for_llm

    def process_sheet(self, sheet_data: SheetData) -> PipelineResult:
        """Process a sheet using the bitmap-first approach.

        The pipeline follows these steps:
        1. Generate full-resolution binary bitmap
        2. Detect connected regions using image processing
        3. Analyze regions for patterns and orientation
        4. Build quadtree for efficient visualization
        5. Generate optimized bitmaps for LLM submission

        Args:
            sheet_data: Sheet to process

        Returns:
            Pipeline processing result
        """
        if HAS_TELEMETRY:
            start_time = time.time()
            metrics = get_metrics_collector()

        logger.info(
            f"Processing sheet with bitmap-first pipeline "
            f"({sheet_data.max_row+1}x{sheet_data.max_column+1} cells)"
        )

        # Phase 1: Initial bitmap analysis
        logger.info("Phase 1: Bitmap analysis")
        bitmap_regions = self._phase1_bitmap_analysis(sheet_data)

        # Phase 2: Pattern detection on regions
        logger.info("Phase 2: Pattern detection")
        patterns = self._phase2_pattern_detection(sheet_data, bitmap_regions)

        # Phase 3: Quadtree optimization
        logger.info("Phase 3: Quadtree optimization")
        quadtree = self._phase3_quadtree_analysis(sheet_data, patterns)

        # Phase 4: Visualization planning
        logger.info("Phase 4: Visualization planning")
        visualization_regions = self._phase4_visualization_planning(quadtree)

        # Compile results
        metadata = {
            "sheet_dimensions": (sheet_data.max_row + 1, sheet_data.max_column + 1),
            "bitmap_regions_found": len(bitmap_regions),
            "patterns_detected": len(patterns),
            "visualization_regions": len(visualization_regions),
            "pattern_types": self._count_pattern_types(patterns),
            "orientations": self._count_orientations(patterns),
        }

        result = PipelineResult(
            detected_tables=patterns,
            visualization_regions=visualization_regions,
            analysis_metadata=metadata,
        )

        if HAS_TELEMETRY:
            duration = time.time() - start_time
            metrics.record_duration("vision_pipeline.process_sheet", duration)
            metrics.record_value("vision_pipeline.patterns_detected", len(patterns))
            metrics.record_value("vision_pipeline.regions_generated", len(visualization_regions))
            logger.info(f"Vision pipeline completed in {duration:.3f}s")

        return result

    def _phase1_bitmap_analysis(self, sheet_data: SheetData) -> list[dict[str, Any]]:
        """Phase 1: Analyze sheet using bitmap representation.

        Returns:
            List of analyzed regions from bitmap
        """
        # Generate full-resolution binary bitmap
        bitmap, metadata = self.bitmap_analyzer.generate_binary_bitmap(sheet_data)

        logger.info(
            f"Generated {metadata['rows']}x{metadata['cols']} bitmap "
            f"with {metadata['filled_cells']} filled cells "
            f"({metadata['density']:.1%} density)"
        )

        if metadata["filled_cells"] == 0:
            return []

        # Find connected regions
        regions = self.bitmap_analyzer.detect_connected_regions(bitmap)

        # Analyze each region
        analyzed_regions = []
        for region in regions:
            analysis = self.bitmap_analyzer.analyze_region_pattern(bitmap, region)
            orientation = self.bitmap_analyzer.detect_orientation(analysis)
            pattern_type = self.bitmap_analyzer.classify_pattern_type(analysis, orientation)

            analysis["orientation"] = orientation
            analysis["pattern_type"] = pattern_type
            analyzed_regions.append(analysis)

        logger.info(f"Found {len(analyzed_regions)} potential table regions")
        return analyzed_regions

    def _phase2_pattern_detection(
        self, sheet_data: SheetData, bitmap_regions: list[dict[str, Any]]
    ) -> list[TablePattern]:
        """Phase 2: Run pattern detection on identified regions.

        Args:
            sheet_data: Sheet data
            bitmap_regions: Regions from bitmap analysis

        Returns:
            List of detected table patterns
        """
        all_patterns = []

        for region_analysis in bitmap_regions:
            region = region_analysis["region"]

            # Run pattern detection on this specific region
            region_patterns = self.pattern_detector.detect_patterns_in_region(
                sheet_data,
                start_row=region.row_start,
                end_row=region.row_end,
                start_col=region.col_start,
                end_col=region.col_end,
            )

            # Enhance patterns with bitmap analysis results
            for pattern in region_patterns:
                # Update orientation from bitmap analysis
                pattern.orientation = region_analysis["orientation"]

                # Add bitmap-derived characteristics
                pattern.characteristics.update(
                    {
                        "bitmap_density": region.density,
                        "bitmap_aspect_ratio": region.aspect_ratio,
                        "detected_via": "bitmap_analysis",
                    }
                )

            # If no patterns detected but region looks like a table, create one
            if not region_patterns and region.density >= self.pattern_detector.min_filled_ratio:
                pattern = self._create_pattern_from_bitmap_region(sheet_data, region_analysis)
                if pattern:
                    all_patterns.append(pattern)
            else:
                all_patterns.extend(region_patterns)

        logger.info(f"Detected {len(all_patterns)} table patterns")
        return all_patterns

    def _phase3_quadtree_analysis(self, sheet_data: SheetData, patterns: list[TablePattern]) -> Any:
        """Phase 3: Build quadtree respecting detected patterns.

        Returns:
            Quadtree root node
        """
        quadtree = self.quadtree_analyzer.analyze(sheet_data, patterns)
        stats = self.quadtree_analyzer.get_coverage_stats(quadtree)

        logger.info(
            f"Built quadtree: {stats['total_nodes']} nodes, " f"max depth {stats['max_depth']}"
        )

        return quadtree

    def _phase4_visualization_planning(self, quadtree: Any) -> list[VisualizationRegion]:
        """Phase 4: Plan visualization regions for LLM submission.

        Returns:
            List of regions to visualize
        """
        regions = self.quadtree_analyzer.plan_visualization(
            quadtree,
            max_regions=self.max_regions_for_llm,
            max_total_size_mb=20.0,  # GPT-4o limit
        )

        total_size = sum(r.estimated_size_mb for r in regions)
        logger.info(
            f"Planned {len(regions)} visualization regions " f"(total size: {total_size:.1f}MB)"
        )

        return regions

    def _create_pattern_from_bitmap_region(
        self, _sheet_data: SheetData, region_analysis: dict[str, Any]
    ) -> TablePattern | None:
        """Create a table pattern from bitmap region analysis.

        Args:
            sheet_data: Sheet data
            region_analysis: Analysis results from bitmap analyzer

        Returns:
            TablePattern or None
        """
        region = region_analysis["region"]

        # Create bounds
        bounds = TableBounds(
            start_row=region.row_start,
            start_col=region.col_start,
            end_row=region.row_end,
            end_col=region.col_end,
        )

        # Map bitmap pattern type to PatternType enum
        pattern_type_map = {
            "header_data": PatternType.HEADER_DATA,
            "matrix": PatternType.MATRIX,
            "form": PatternType.FORM,
            "time_series": PatternType.TIME_SERIES,
            "simple_table": PatternType.SIMPLE_TABLE,
        }

        pattern_type = pattern_type_map.get(
            region_analysis["pattern_type"], PatternType.SIMPLE_TABLE
        )

        # Determine headers based on analysis
        header_rows = []
        header_cols = []

        if region_analysis.get("has_dense_first_row"):
            header_rows = [region.row_start]
        if region_analysis.get("has_dense_first_col"):
            header_cols = [region.col_start]

        # Create pattern
        pattern = TablePattern(
            pattern_type=pattern_type,
            bounds=bounds,
            confidence=0.7,  # Base confidence from bitmap detection
            header_rows=header_rows,
            header_cols=header_cols,
            orientation=region_analysis["orientation"],
            characteristics={
                "density": region.density,
                "aspect_ratio": region.aspect_ratio,
                "detected_via": "bitmap_only",
                "has_dense_first_row": region_analysis.get("has_dense_first_row", False),
                "has_dense_first_col": region_analysis.get("has_dense_first_col", False),
            },
        )

        return pattern

    def _count_pattern_types(self, patterns: list[TablePattern]) -> dict[str, int]:
        """Count patterns by type."""
        counts = {}
        for pattern in patterns:
            type_name = pattern.pattern_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _count_orientations(self, patterns: list[TablePattern]) -> dict[str, int]:
        """Count patterns by orientation."""
        counts = {}
        for pattern in patterns:
            orientation = pattern.orientation.value
            counts[orientation] = counts.get(orientation, 0) + 1
        return counts

    def generate_visualization_bitmaps(
        self, sheet_data: SheetData, regions: list[VisualizationRegion]
    ) -> list[Any]:
        """Generate compressed bitmaps for LLM submission.

        Args:
            sheet_data: Sheet data
            regions: Regions to visualize

        Returns:
            List of bitmap results
        """
        return self.bitmap_generator.generate_from_visualization_plan(sheet_data, regions)
