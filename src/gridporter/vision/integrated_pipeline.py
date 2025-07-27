"""Integrated vision pipeline using bitmap-first approach."""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..models.sheet_data import SheetData

if TYPE_CHECKING:
    from ..config import GridPorterConfig
from ..detectors.format_analyzer import SemanticFormatAnalyzer, TableStructure
from ..detectors.multi_header_detector import MultiHeaderDetector
from .bitmap_analyzer import BitmapAnalyzer
from .bitmap_generator import BitmapGenerator
from .pattern_detector import (
    PatternType,
    SparsePatternDetector,
    TableBounds,
    TablePattern,
)
from .quadtree import QuadtreeAnalyzer, VisualizationRegion
from .region_verifier import RegionVerifier, VerificationResult

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
    verification_results: dict[str, VerificationResult] | None = None
    semantic_structures: dict[str, TableStructure] | None = None
    multi_row_headers: dict[str, Any] | None = None


class IntegratedVisionPipeline:
    """Multi-scale vision pipeline using bitmap-first detection."""

    def __init__(
        self,
        min_table_size: tuple[int, int] = (2, 2),
        min_density: float = 0.1,
        max_regions_for_llm: int = 10,
        enable_verification: bool = True,
        verification_strict: bool = False,
    ):
        """Initialize integrated pipeline.

        Args:
            min_table_size: Minimum (rows, cols) to consider as table
            min_density: Minimum density for regions
            max_regions_for_llm: Maximum regions to send to LLM
            enable_verification: Whether to enable region verification
            verification_strict: Whether to use strict verification rules
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

        # Semantic analysis components
        self.format_analyzer = SemanticFormatAnalyzer()
        self.multi_header_detector = MultiHeaderDetector()

        # Region verification
        self.enable_verification = enable_verification
        self.verification_strict = verification_strict
        if enable_verification:
            self.region_verifier = RegionVerifier(
                min_filledness=min_density,
                min_rectangularness=0.7,
                min_contiguity=0.5,
            )

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

        # Phase 5: Semantic analysis (Week 5 enhancement)
        logger.info("Phase 5: Semantic analysis")
        semantic_structures, multi_headers = self._phase5_semantic_analysis(sheet_data, patterns)

        # Compile results
        metadata = {
            "sheet_dimensions": (sheet_data.max_row + 1, sheet_data.max_column + 1),
            "bitmap_regions_found": len(bitmap_regions),
            "patterns_detected": len(patterns),
            "visualization_regions": len(visualization_regions),
            "pattern_types": self._count_pattern_types(patterns),
            "orientations": self._count_orientations(patterns),
            "has_multi_row_headers": bool(multi_headers),
            "semantic_tables_found": len(semantic_structures),
        }

        result = PipelineResult(
            detected_tables=patterns,
            visualization_regions=visualization_regions,
            analysis_metadata=metadata,
            verification_results=getattr(self, "_verification_results", None),
            semantic_structures=semantic_structures,
            multi_row_headers=multi_headers,
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

        # Phase 2.5: Verify detected patterns
        if self.enable_verification:
            verified_patterns = self._verify_patterns(sheet_data, all_patterns)
            logger.info(f"Verified {len(verified_patterns)} of {len(all_patterns)} patterns")
            return verified_patterns

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

    def _verify_patterns(
        self, sheet_data: SheetData, patterns: list[TablePattern]
    ) -> list[TablePattern]:
        """Verify detected patterns using geometry analysis.

        Args:
            sheet_data: Sheet data
            patterns: Patterns to verify

        Returns:
            List of verified patterns
        """
        verified_patterns = []
        self._verification_results = {}

        for pattern in patterns:
            # Verify the pattern
            result = self.region_verifier.verify_pattern(sheet_data, pattern)

            # Store verification result
            pattern_id = (
                f"{pattern.start_row}_{pattern.start_col}_{pattern.end_row}_{pattern.end_col}"
            )
            self._verification_results[pattern_id] = result

            if result.valid:
                # Update pattern confidence based on verification
                pattern.confidence = min(pattern.confidence, result.confidence)
                pattern.characteristics["verification_score"] = result.confidence
                pattern.characteristics["verification_metrics"] = result.metrics
                verified_patterns.append(pattern)
            else:
                logger.debug(f"Pattern at {pattern.range} failed verification: {result.reason}")
                if result.feedback:
                    logger.debug(f"Feedback: {result.feedback}")

        return verified_patterns

    def _generate_feedback_for_invalid_regions(
        self, invalid_results: list[tuple[TablePattern, VerificationResult]]
    ) -> dict[str, Any]:
        """Generate feedback for regions that failed verification.

        Args:
            invalid_results: List of (pattern, verification_result) tuples

        Returns:
            Feedback dictionary for re-querying vision model
        """
        feedback = {
            "failed_regions": [],
            "suggestions": [],
        }

        for pattern, result in invalid_results:
            region_feedback = {
                "range": pattern.range,
                "reason": result.reason,
                "metrics": result.metrics,
                "feedback": result.feedback,
            }
            feedback["failed_regions"].append(region_feedback)

            # Generate specific suggestions based on failure reason
            if "Low filledness" in result.reason:
                feedback["suggestions"].append(
                    f"Region {pattern.range} appears too sparse. "
                    "Look for denser data concentrations within this area."
                )
            elif "Low rectangularness" in result.reason:
                feedback["suggestions"].append(
                    f"Region {pattern.range} has irregular shape. "
                    "Try to identify more rectangular sub-regions."
                )
            elif "Low contiguity" in result.reason:
                feedback["suggestions"].append(
                    f"Region {pattern.range} has disconnected data. "
                    "Consider splitting into multiple smaller tables."
                )

        return feedback

    def _phase5_semantic_analysis(
        self, sheet_data: SheetData, patterns: list[TablePattern]
    ) -> tuple[dict[str, TableStructure], dict[str, Any]]:
        """Phase 5: Perform semantic analysis on detected patterns.

        Args:
            sheet_data: Sheet data
            patterns: Detected table patterns

        Returns:
            Tuple of (semantic_structures, multi_row_headers)
        """
        semantic_structures = {}
        multi_headers = {}

        for i, pattern in enumerate(patterns):
            pattern_id = f"table_{i}"

            # Convert pattern bounds to TableRange
            from ..models.table import TableRange

            table_range = TableRange(
                start_row=pattern.bounds.start_row,
                start_col=pattern.bounds.start_col,
                end_row=pattern.bounds.end_row,
                end_col=pattern.bounds.end_col,
            )

            # Detect multi-row headers if pattern indicates headers
            if pattern.multi_row_headers or len(pattern.header_rows) > 1:
                multi_header = self.multi_header_detector.detect_multi_row_headers(
                    sheet_data, table_range
                )
                if multi_header:
                    multi_headers[pattern_id] = {
                        "header_rows": list(
                            range(multi_header.start_row, multi_header.end_row + 1)
                        ),
                        "column_mappings": multi_header.column_mappings,
                        "confidence": multi_header.confidence,
                    }
                    header_row_count = multi_header.end_row - multi_header.start_row + 1
                else:
                    header_row_count = len(pattern.header_rows) if pattern.header_rows else 1
            else:
                header_row_count = len(pattern.header_rows) if pattern.header_rows else 1

            # Analyze semantic structure
            structure = self.format_analyzer.analyze_table_structure(
                sheet_data, table_range, header_row_count
            )
            semantic_structures[pattern_id] = structure

            # Update pattern characteristics with semantic info
            pattern.characteristics.update(
                {
                    "has_subtotals": structure.has_subtotals,
                    "has_grand_total": structure.has_grand_total,
                    "section_count": len(structure.sections),
                    "semantic_blank_rows": len(structure.preserve_blank_rows),
                }
            )

        logger.info(
            f"Semantic analysis complete: {len(multi_headers)} multi-row headers, "
            f"{sum(1 for s in semantic_structures.values() if s.has_subtotals)} tables with subtotals"
        )

        return semantic_structures, multi_headers

    @classmethod
    def from_config(cls, config: "GridPorterConfig") -> "IntegratedVisionPipeline":
        """Create pipeline from GridPorter configuration.

        Args:
            config: GridPorter configuration

        Returns:
            Configured pipeline instance
        """
        pipeline = cls(
            min_table_size=config.min_table_size,
            min_density=config.min_region_filledness,
            max_regions_for_llm=config.max_tables_per_sheet,
            enable_verification=config.enable_region_verification,
            verification_strict=config.verification_strict_mode,
        )

        # Update region verifier with config values if enabled
        if config.enable_region_verification and pipeline.region_verifier:
            pipeline.region_verifier = RegionVerifier(
                min_filledness=config.min_region_filledness,
                min_rectangularness=config.min_region_rectangularness,
                min_contiguity=config.min_region_contiguity,
            )

        return pipeline
