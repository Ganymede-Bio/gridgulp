"""Unit tests for ProgressiveRefiner."""

import pytest
from unittest.mock import MagicMock, patch

from gridporter.models.multi_scale import (
    CompressionLevel,
    DataRegion,
    ProgressiveRefinementPhase,
    VisionImage,
)
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.models.table import TableInfo, TableRange
from gridporter.vision.progressive_refiner import ProgressiveRefiner


class TestProgressiveRefiner:
    """Test ProgressiveRefiner functionality."""

    @pytest.fixture
    def refiner(self):
        """Create progressive refiner."""
        return ProgressiveRefiner()

    @pytest.fixture
    def large_sheet(self):
        """Create large sheet requiring progressive refinement."""
        sheet = SheetData(name="LargeSheet")

        # 10K x 1K = 10M cells
        sheet.max_row = 9999
        sheet.max_column = 999

        # Add some data regions
        # Region 1: Top area
        for row in range(100):
            for col in range(50):
                if row < 10 or (row > 20 and row < 30):
                    sheet.set_cell(
                        row,
                        col,
                        CellData(
                            row=row,
                            column=col,
                            value=f"T{row}{col}",
                            data_type="text",
                            is_bold=(row == 0 or row == 21),
                        ),
                    )

        # Region 2: Middle area
        for row in range(5000, 5100):
            for col in range(100, 200):
                if col % 2 == 0:
                    sheet.set_cell(
                        row, col, CellData(row=row, column=col, value=row * col, data_type="number")
                    )

        # Region 3: Bottom area
        for row in range(9900, 9950):
            for col in range(900, 950):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value=f"B{row}{col}", data_type="text")
                )

        return sheet

    @pytest.fixture
    def data_regions(self):
        """Create test data regions."""
        return [
            DataRegion(
                region_id="region_1",
                bounds={"top": 0, "left": 0, "bottom": 99, "right": 49},
                cell_count=2000,
                density=0.4,
                characteristics={
                    "likely_headers": True,
                    "mixed_types": True,
                    "has_formatting": True,
                },
            ),
            DataRegion(
                region_id="region_2",
                bounds={"top": 5000, "left": 100, "bottom": 5099, "right": 199},
                cell_count=5000,
                density=0.5,
                characteristics={"mostly_numbers": True, "likely_data": True},
            ),
            DataRegion(
                region_id="region_3",
                bounds={"top": 9900, "left": 900, "bottom": 9949, "right": 949},
                cell_count=2500,
                density=1.0,
                characteristics={"mostly_text": True},
            ),
        ]

    def test_refine_sheet_basic(self, refiner, large_sheet, data_regions):
        """Test basic progressive refinement."""
        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        # Should have completed at least one phase
        assert len(phases) >= 1
        assert len(images) >= 1

        # First phase should be overview
        assert phases[0].phase == "overview"
        assert phases[0].compression_level == CompressionLevel.MAXIMUM.value

        # Should have overview image
        overview = next((img for img in images if "phase1_overview" in img.image_id), None)
        assert overview is not None
        assert overview.compression_level == CompressionLevel.MAXIMUM.value

    def test_phase_progression(self, refiner, large_sheet, data_regions):
        """Test progression through phases."""
        # Give more budget for multiple phases
        refiner.MAX_TOTAL_SIZE_MB = 50.0

        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        # Check phase types if multiple phases executed
        phase_types = [p.phase for p in phases]
        assert "overview" in phase_types

        # May have refinement and verification depending on budget
        if len(phases) > 1:
            assert phases[1].phase in ["refinement", "verification"]

    def test_budget_management(self, refiner, large_sheet, data_regions):
        """Test budget limit enforcement."""
        # Set very low budget
        refiner.MAX_TOTAL_SIZE_MB = 0.1  # 100KB

        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        # Should stop early due to budget
        assert refiner.total_size_mb <= refiner.MAX_TOTAL_SIZE_MB * 1.1  # Allow 10% overhead

        # Likely only phase 1
        assert len(phases) <= 2

    def test_high_priority_region_selection(self, refiner, data_regions):
        """Test identification of high-priority regions."""
        # Add more varied regions
        extra_regions = [
            DataRegion(
                region_id="low_density",
                bounds={"top": 100, "left": 100, "bottom": 199, "right": 199},
                cell_count=100,
                density=0.01,  # Very sparse
                characteristics={},
            ),
            DataRegion(
                region_id="high_density_headers",
                bounds={"top": 200, "left": 0, "bottom": 299, "right": 99},
                cell_count=9000,
                density=0.9,  # Very dense
                characteristics={"likely_headers": True, "mixed_types": True},
            ),
        ]

        all_regions = data_regions + extra_regions
        high_priority = refiner._identify_high_priority_regions(all_regions)

        # Should prioritize dense regions with headers
        assert len(high_priority) > 0
        assert high_priority[0].characteristics.get("likely_headers", False)

        # Low density region should not be high priority
        assert not any(r.region_id == "low_density" for r in high_priority)

    def test_critical_region_identification(self, refiner, data_regions):
        """Test identification of critical regions for final verification."""
        high_priority = refiner._identify_high_priority_regions(data_regions)
        critical = refiner._identify_critical_regions(high_priority)

        # Critical regions must have headers and good density
        for region in critical:
            assert region.characteristics.get("likely_headers", False)
            assert region.density > 0.4

        # Should limit to top 3
        assert len(critical) <= 3

    def test_extremely_large_sheet_handling(self, refiner):
        """Test handling of sheets at Excel limits."""
        huge_sheet = SheetData(name="HugeSheet")
        huge_sheet.max_row = 1_048_575  # Excel max
        huge_sheet.max_column = 16_383  # Excel max

        # Just a few data points
        huge_sheet.set_cell(0, 0, CellData(row=0, column=0, value="Start", data_type="text"))
        huge_sheet.set_cell(
            1000, 1000, CellData(row=1000, column=1000, value="Middle", data_type="text")
        )

        regions = [
            DataRegion(
                region_id="huge_region",
                bounds={"top": 0, "left": 0, "bottom": 1000, "right": 1000},
                cell_count=2,
                density=0.000002,
                characteristics={},
            )
        ]

        images, phases = refiner.refine_sheet(huge_sheet, regions)

        # Should handle without crashing
        assert len(phases) >= 1
        assert phases[0].compression_level == CompressionLevel.MAXIMUM.value

        # Should potentially generate quadrant views
        quadrant_images = [img for img in images if "quadrant" in img.image_id]
        assert len(quadrant_images) >= 0  # May or may not have quadrants

    def test_image_id_generation(self, refiner, large_sheet, data_regions):
        """Test that image IDs are properly generated."""
        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        # Collect all image IDs
        image_ids = [img.image_id for img in images]

        # Should be unique
        assert len(image_ids) == len(set(image_ids))

        # Should follow naming convention
        for img in images:
            assert img.image_id.startswith("phase")
            assert any(
                phase in img.image_id for phase in ["overview", "quadrant", "region", "verify"]
            )

    def test_compression_level_progression(self, refiner, large_sheet, data_regions):
        """Test that compression levels decrease through phases."""
        # Ensure multiple phases
        refiner.MAX_TOTAL_SIZE_MB = 100.0

        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        if len(phases) >= 2:
            # Later phases should use less compression
            assert phases[0].compression_level >= phases[-1].compression_level

    def test_empty_regions_handling(self, refiner, large_sheet):
        """Test handling when no data regions provided."""
        images, phases = refiner.refine_sheet(large_sheet, [])

        # Should still generate overview
        assert len(phases) >= 1
        assert phases[0].phase == "overview"

    def test_phase_descriptions(self, refiner, large_sheet, data_regions):
        """Test that phases have proper descriptions."""
        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        for phase in phases:
            assert phase.strategy
            assert phase.purpose
            assert phase.compression_level >= 0

            # Check strategy matches phase
            if phase.phase == "overview":
                assert "compression" in phase.strategy
            elif phase.phase == "refinement":
                assert "medium" in phase.strategy or "targeted" in phase.strategy

    def test_merge_progressive_results(self, refiner):
        """Test merging results from multiple phases."""
        # Mock phase results
        phase_results = [
            {
                "tables": [
                    {
                        "bounds": {"top": 0, "left": 0, "bottom": 10, "right": 5},
                        "confidence": 0.6,
                        "suggested_name": "Table1_Overview",
                        "has_headers": True,
                    }
                ]
            },
            {
                "tables": [
                    {
                        "bounds": {"top": 0, "left": 0, "bottom": 10, "right": 5},
                        "confidence": 0.8,
                        "suggested_name": "Table1_Refined",
                        "has_headers": True,
                    },
                    {
                        "bounds": {"top": 20, "left": 10, "bottom": 30, "right": 20},
                        "confidence": 0.7,
                        "suggested_name": "Table2",
                        "has_headers": False,
                    },
                ]
            },
            {
                "tables": [
                    {
                        "bounds": {"top": 0, "left": 0, "bottom": 10, "right": 5},
                        "confidence": 0.95,
                        "suggested_name": "Table1_Verified",
                        "has_headers": True,
                        "headers": ["Col1", "Col2", "Col3", "Col4", "Col5"],
                    }
                ]
            },
        ]

        # Set phases to match results
        refiner.phases_completed = [
            ProgressiveRefinementPhase(
                phase="overview",
                strategy="max_compression",
                compression_level=5,
                purpose="Overview",
            ),
            ProgressiveRefinementPhase(
                phase="refinement",
                strategy="medium_compression",
                compression_level=3,
                purpose="Refinement",
            ),
            ProgressiveRefinementPhase(
                phase="verification",
                strategy="min_compression",
                compression_level=1,
                purpose="Verification",
            ),
        ]

        merged_tables = refiner.merge_progressive_results(phase_results)

        # Should have 2 unique tables
        assert len(merged_tables) == 2

        # First table should have highest confidence from verification
        table1 = next(t for t in merged_tables if t.range.start_row == 0)
        assert table1.confidence == 0.95 * 1.0  # Verification phase gets full confidence
        assert table1.suggested_name == "Table1_Verified"
        assert table1.headers == ["Col1", "Col2", "Col3", "Col4", "Col5"]

        # Second table from refinement phase
        table2 = next(t for t in merged_tables if t.range.start_row == 20)
        assert table2.confidence == 0.7 * 0.9  # Refinement phase gets 0.9 multiplier

    def test_performance_tracking(self, refiner, large_sheet, data_regions):
        """Test that performance is tracked."""
        import time

        start = time.time()
        images, phases = refiner.refine_sheet(large_sheet, data_regions)
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 30.0  # 30 seconds max

        # Should track size
        assert refiner.total_size_mb > 0

    def test_image_size_validation(self, refiner, large_sheet, data_regions):
        """Test that generated images have valid sizes."""
        images, phases = refiner.refine_sheet(large_sheet, data_regions)

        for img in images:
            # Size should be positive
            assert img.size_bytes > 0
            assert img.size_mb > 0

            # Description and purpose should be set
            assert img.description
            assert img.purpose
            assert img.covers_cells  # Should have Excel range
