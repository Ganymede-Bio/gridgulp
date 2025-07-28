"""Progressive refinement for very large spreadsheets."""

import logging
from typing import Any

from ..models.multi_scale import (
    CompressionLevel,
    DataRegion,
    ProgressiveRefinementPhase,
    VisionImage,
)
from ..models.sheet_data import SheetData
from ..models.table import TableInfo, TableRange
from ..utils.excel_utils import to_excel_range
from .bitmap_generator import BitmapGenerator
from .data_region_preprocessor import DataRegionPreprocessor
from .quadtree import QuadTreeBounds

logger = logging.getLogger(__name__)


class ProgressiveRefiner:
    """Handle progressive refinement for very large sheets.

    This refiner implements a three-phase approach:
    1. Overview: Highly compressed view to identify major regions
    2. Refinement: Medium compression on identified regions
    3. Verification: Full resolution on final boundaries
    """

    # Size thresholds
    PHASE_1_MAX_PIXELS = 50_000  # ~224x224
    PHASE_2_MAX_PIXELS = 200_000  # ~447x447
    PHASE_3_MAX_PIXELS = 500_000  # ~707x707

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70

    # Budget limits
    MAX_IMAGES_PER_PHASE = 5
    MAX_TOTAL_SIZE_MB = 18.0  # Leave room for overhead

    def __init__(self):
        """Initialize the progressive refiner."""
        self.preprocessor = DataRegionPreprocessor()
        self.bitmap_gen = BitmapGenerator(auto_compress=True)
        self.phases_completed = []
        self.total_size_mb = 0.0

    def refine_sheet(
        self, sheet_data: SheetData, initial_regions: list[DataRegion]
    ) -> tuple[list[VisionImage], list[ProgressiveRefinementPhase]]:
        """Progressively refine table detection for a very large sheet.

        Args:
            sheet_data: The sheet data to analyze
            initial_regions: Data regions from preprocessing

        Returns:
            Tuple of (images for vision analysis, phases executed)
        """
        logger.info(
            f"Starting progressive refinement for sheet with "
            f"{sheet_data.max_row + 1}×{sheet_data.max_column + 1} cells"
        )

        all_images = []
        self.phases_completed = []
        self.total_size_mb = 0.0

        # Phase 1: Overview
        phase1_images = self._phase_1_overview(sheet_data, initial_regions)
        all_images.extend(phase1_images)

        # Early exit if we've used most of our budget
        if self.total_size_mb > self.MAX_TOTAL_SIZE_MB * 0.7:
            logger.warning(
                f"Budget mostly consumed after phase 1 ({self.total_size_mb:.1f}MB), "
                "skipping further refinement"
            )
            return all_images, self.phases_completed

        # Phase 2: Refinement (if needed)
        high_priority_regions = self._identify_high_priority_regions(initial_regions)
        if high_priority_regions:
            phase2_images = self._phase_2_refinement(sheet_data, high_priority_regions)
            all_images.extend(phase2_images)

        # Phase 3: Verification (only if budget allows)
        if self.total_size_mb < self.MAX_TOTAL_SIZE_MB * 0.85:
            # Focus on most promising regions for final verification
            critical_regions = self._identify_critical_regions(high_priority_regions)
            if critical_regions:
                phase3_images = self._phase_3_verification(sheet_data, critical_regions)
                all_images.extend(phase3_images)

        logger.info(
            f"Progressive refinement complete: {len(self.phases_completed)} phases, "
            f"{len(all_images)} images, {self.total_size_mb:.1f}MB total"
        )

        return all_images, self.phases_completed

    def _phase_1_overview(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Phase 1: Generate highly compressed overview.

        Args:
            sheet_data: The sheet data
            regions: Initial data regions

        Returns:
            List of overview images
        """
        phase = ProgressiveRefinementPhase(
            phase="overview",
            strategy="maximum_compression",
            compression_level=CompressionLevel.MAXIMUM.value,
            focus_regions=[r.bounds for r in regions[:5]],  # Top 5 regions
            purpose="Identify major table regions in very large sheet",
        )
        self.phases_completed.append(phase)

        images = []

        # Generate maximum compression overview
        self.bitmap_gen.auto_compress = False
        old_compression = self.bitmap_gen.compression_level
        self.bitmap_gen.compression_level = 9

        try:
            # Full sheet overview
            img_data, metadata = self.bitmap_gen.generate(sheet_data)

            overview = VisionImage(
                image_id="phase1_overview",
                image_data=img_data.hex(),  # Convert bytes to hex string
                compression_level=CompressionLevel.MAXIMUM.value,
                block_size=[CompressionLevel.MAXIMUM.row_block, CompressionLevel.MAXIMUM.col_block],
                description=(
                    f"Phase 1: Maximum compression overview of entire sheet. "
                    f"Each pixel represents {CompressionLevel.MAXIMUM.row_block}×"
                    f"{CompressionLevel.MAXIMUM.col_block} cells."
                ),
                purpose="Identify approximate locations of all tables",
                covers_cells=to_excel_range(0, 0, sheet_data.max_row, sheet_data.max_column),
                size_bytes=len(img_data),
            )
            images.append(overview)
            self.total_size_mb += overview.size_mb

            # If sheet is extremely large, also generate regional overviews
            if (sheet_data.max_row + 1) * (sheet_data.max_column + 1) > 10_000_000:
                # Quadrant views
                mid_row = sheet_data.max_row // 2
                mid_col = sheet_data.max_column // 2

                quadrants = [
                    ("top_left", 0, 0, mid_row, mid_col),
                    ("top_right", 0, mid_col + 1, mid_row, sheet_data.max_column),
                    ("bottom_left", mid_row + 1, 0, sheet_data.max_row, mid_col),
                    (
                        "bottom_right",
                        mid_row + 1,
                        mid_col + 1,
                        sheet_data.max_row,
                        sheet_data.max_column,
                    ),
                ]

                for quad_name, min_row, min_col, max_row, max_col in quadrants:
                    if self.total_size_mb > self.MAX_TOTAL_SIZE_MB * 0.5:
                        break

                    # Check if quadrant has data
                    has_data = any(
                        r
                        for r in regions
                        if (
                            r.bounds["top"] >= min_row
                            and r.bounds["bottom"] <= max_row
                            and r.bounds["left"] >= min_col
                            and r.bounds["right"] <= max_col
                        )
                    )

                    if has_data:
                        bounds = QuadTreeBounds(min_row, min_col, max_row, max_col)
                        quad_data, quad_meta = self.bitmap_gen.generate(sheet_data, bounds)

                        quad_image = VisionImage(
                            image_id=f"phase1_quadrant_{quad_name}",
                            image_data=quad_data.hex(),
                            compression_level=CompressionLevel.MAXIMUM.value,
                            block_size=[
                                CompressionLevel.MAXIMUM.row_block,
                                CompressionLevel.MAXIMUM.col_block,
                            ],
                            description=f"Phase 1: {quad_name} quadrant overview",
                            purpose=f"Identify tables in {quad_name} section",
                            covers_cells=to_excel_range(min_row, min_col, max_row, max_col),
                            size_bytes=len(quad_data),
                        )
                        images.append(quad_image)
                        self.total_size_mb += quad_image.size_mb

        finally:
            self.bitmap_gen.compression_level = old_compression
            self.bitmap_gen.auto_compress = True

        return images

    def _phase_2_refinement(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Phase 2: Refine high-priority regions with better resolution.

        Args:
            sheet_data: The sheet data
            regions: High-priority regions to refine

        Returns:
            List of refinement images
        """
        phase = ProgressiveRefinementPhase(
            phase="refinement",
            strategy="targeted_medium_compression",
            compression_level=CompressionLevel.LARGE.value,
            focus_regions=[r.bounds for r in regions],
            purpose="Refine table boundaries in promising regions",
        )
        self.phases_completed.append(phase)

        images = []

        for idx, region in enumerate(regions[: self.MAX_IMAGES_PER_PHASE]):
            if self.total_size_mb > self.MAX_TOTAL_SIZE_MB * 0.8:
                logger.info(f"Budget limit approaching, stopping phase 2 at region {idx + 1}")
                break

            # Use LARGE compression for refinement
            bounds = QuadTreeBounds(
                region.bounds["top"],
                region.bounds["left"],
                region.bounds["bottom"],
                region.bounds["right"],
            )

            # Force specific compression level
            self.bitmap_gen.auto_compress = False
            old_compression = self.bitmap_gen.compression_level
            self.bitmap_gen.compression_level = 7

            try:
                img_data, metadata = self.bitmap_gen.generate(sheet_data, bounds)

                # Override compression level in metadata
                metadata.compression_level = CompressionLevel.LARGE
                metadata.block_size = [
                    CompressionLevel.LARGE.row_block,
                    CompressionLevel.LARGE.col_block,
                ]

                refinement = VisionImage(
                    image_id=f"phase2_region_{idx + 1}",
                    image_data=img_data.hex(),
                    compression_level=CompressionLevel.LARGE.value,
                    block_size=metadata.block_size,
                    description=(
                        f"Phase 2: Refined view of region {idx + 1} using {CompressionLevel.LARGE.description}. "
                        f"Each pixel represents {metadata.block_size[0]}×{metadata.block_size[1]} cells."
                    ),
                    purpose=f"Identify precise table boundaries in region {region.region_id}",
                    covers_cells=to_excel_range(
                        region.bounds["top"],
                        region.bounds["left"],
                        region.bounds["bottom"],
                        region.bounds["right"],
                    ),
                    size_bytes=len(img_data),
                )
                images.append(refinement)
                self.total_size_mb += refinement.size_mb

            finally:
                self.bitmap_gen.compression_level = old_compression
                self.bitmap_gen.auto_compress = True

        return images

    def _phase_3_verification(
        self, sheet_data: SheetData, regions: list[DataRegion]
    ) -> list[VisionImage]:
        """Phase 3: Final verification with minimal compression.

        Args:
            sheet_data: The sheet data
            regions: Critical regions for final verification

        Returns:
            List of verification images
        """
        phase = ProgressiveRefinementPhase(
            phase="verification",
            strategy="minimal_compression",
            compression_level=CompressionLevel.MILD.value,
            focus_regions=[r.bounds for r in regions],
            purpose="Verify exact table boundaries",
        )
        self.phases_completed.append(phase)

        images = []

        for idx, region in enumerate(regions[:3]):  # Max 3 verification images
            if self.total_size_mb > self.MAX_TOTAL_SIZE_MB * 0.95:
                logger.info("Near budget limit, stopping phase 3")
                break

            # Use MILD compression for final verification
            bounds = QuadTreeBounds(
                region.bounds["top"],
                region.bounds["left"],
                region.bounds["bottom"],
                region.bounds["right"],
            )

            # Check if region is small enough for no compression
            region_cells = region.total_cells
            if region_cells < self.PHASE_3_MAX_PIXELS:
                compression = CompressionLevel.NONE
            else:
                compression = CompressionLevel.MILD

            # Force specific compression
            self.bitmap_gen.auto_compress = False
            old_compression = self.bitmap_gen.compression_level
            self.bitmap_gen.compression_level = 3 if compression == CompressionLevel.MILD else 0

            try:
                img_data, metadata = self.bitmap_gen.generate(sheet_data, bounds)

                # Override compression level in metadata
                metadata.compression_level = compression
                metadata.block_size = [compression.row_block, compression.col_block]

                verification = VisionImage(
                    image_id=f"phase3_verify_{idx + 1}",
                    image_data=img_data.hex(),
                    compression_level=compression.value,
                    block_size=metadata.block_size,
                    description=(
                        f"Phase 3: Final verification of critical region {idx + 1}. "
                        f"{'No compression - full detail visible.' if compression == CompressionLevel.NONE else f'Minimal compression ({compression.description}).'}"
                    ),
                    purpose="Verify exact cell-level table boundaries",
                    covers_cells=to_excel_range(
                        region.bounds["top"],
                        region.bounds["left"],
                        region.bounds["bottom"],
                        region.bounds["right"],
                    ),
                    size_bytes=len(img_data),
                )
                images.append(verification)
                self.total_size_mb += verification.size_mb

            finally:
                self.bitmap_gen.compression_level = old_compression
                self.bitmap_gen.auto_compress = True

        return images

    def _identify_high_priority_regions(self, regions: list[DataRegion]) -> list[DataRegion]:
        """Identify regions that need refinement.

        Args:
            regions: All data regions

        Returns:
            High-priority regions for phase 2
        """
        # Prioritize by:
        # 1. Density (likely tables have higher density)
        # 2. Has headers
        # 3. Size (not too small, not too large)

        scored_regions = []
        for region in regions:
            score = 0.0

            # Density score
            if region.density > 0.3:
                score += 0.4
            elif region.density > 0.1:
                score += 0.2

            # Header score
            if region.characteristics.get("likely_headers", False):
                score += 0.3

            # Size score (prefer medium-sized regions)
            if 100 < region.total_cells < 10000:
                score += 0.3
            elif 10 < region.total_cells <= 100:
                score += 0.2

            # Mixed data types indicate structured data
            if region.characteristics.get("mixed_types", False):
                score += 0.2

            scored_regions.append((score, region))

        # Sort by score and return top regions
        scored_regions.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored_regions[:10]]  # Top 10 regions

    def _identify_critical_regions(self, regions: list[DataRegion]) -> list[DataRegion]:
        """Identify most critical regions for final verification.

        Args:
            regions: High-priority regions from phase 2

        Returns:
            Critical regions for phase 3
        """
        # Further filter to most critical regions
        critical = []

        for region in regions:
            # Must have headers and good density
            if (
                region.characteristics.get("likely_headers", False)
                and region.density > 0.4
                and region.total_cells < self.PHASE_3_MAX_PIXELS
            ):
                critical.append(region)

        # Return top 3 by density
        critical.sort(key=lambda r: r.density, reverse=True)
        return critical[:3]

    def merge_progressive_results(self, phase_results: list[dict[str, Any]]) -> list[TableInfo]:
        """Merge results from multiple phases into final table list.

        Args:
            phase_results: Results from each phase of analysis

        Returns:
            Merged and deduplicated table information
        """
        all_tables = []
        seen_ranges = set()

        # Process results from each phase (later phases override earlier ones)
        for phase_idx, result in enumerate(phase_results):
            phase = (
                self.phases_completed[phase_idx] if phase_idx < len(self.phases_completed) else None
            )

            for table_data in result.get("tables", []):
                # Extract bounds
                bounds = table_data.get("bounds", {})
                range_key = (
                    bounds.get("top", 0),
                    bounds.get("left", 0),
                    bounds.get("bottom", 0),
                    bounds.get("right", 0),
                )

                # Skip if we've seen this exact range before
                if range_key in seen_ranges:
                    continue

                seen_ranges.add(range_key)

                # Create TableInfo
                table_range = TableRange(
                    start_row=bounds.get("top", 0),
                    start_col=bounds.get("left", 0),
                    end_row=bounds.get("bottom", 0),
                    end_col=bounds.get("right", 0),
                )

                # Adjust confidence based on phase
                base_confidence = table_data.get("confidence", 0.5)
                if phase and phase.phase == "verification":
                    confidence = base_confidence  # Trust verification phase
                elif phase and phase.phase == "refinement":
                    confidence = base_confidence * 0.9  # Slightly lower for refinement
                else:
                    confidence = base_confidence * 0.8  # Lower for overview

                table_info = TableInfo(
                    range=table_range,
                    confidence=confidence,
                    detection_method=f"progressive_phase_{phase_idx + 1}",
                    suggested_name=table_data.get("suggested_name"),
                    headers=table_data.get("headers"),
                    has_headers=table_data.get("has_headers", True),
                )

                all_tables.append(table_info)

        # Sort by confidence (highest first)
        all_tables.sort(key=lambda t: t.confidence, reverse=True)

        return all_tables
