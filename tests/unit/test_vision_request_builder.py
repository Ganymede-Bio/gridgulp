"""Unit tests for VisionRequestBuilder."""

import base64
import pytest

from gridporter.models.multi_scale import CompressionLevel, DataRegion
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.vision.vision_request_builder import VisionRequestBuilder


class TestVisionRequestBuilder:
    """Test VisionRequestBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create request builder."""
        return VisionRequestBuilder()

    @pytest.fixture
    def small_sheet(self):
        """Create small sheet (< 100K cells)."""
        sheet = SheetData(name="SmallSheet")

        # 100x100 = 10K cells
        for row in range(0, 100, 10):
            for col in range(0, 100, 10):
                sheet.set_cell(
                    row,
                    col,
                    CellData(
                        row=row,
                        column=col,
                        value=f"S_{row}_{col}",
                        data_type="text",
                        is_bold=(row == 0),
                    ),
                )

        return sheet

    @pytest.fixture
    def medium_sheet(self):
        """Create medium sheet (100K - 1M cells)."""
        sheet = SheetData(name="MediumSheet")

        # Set bounds for 500K cells but sparse data
        sheet.max_row = 999
        sheet.max_column = 499

        # Add data in regions
        # Region 1: Top-left
        for row in range(50):
            for col in range(100):
                if row < 10 or col < 10:
                    sheet.set_cell(
                        row,
                        col,
                        CellData(
                            row=row, column=col, value="TL", data_type="text", is_bold=(row == 0)
                        ),
                    )

        # Region 2: Bottom-right
        for row in range(900, 950):
            for col in range(400, 450):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="BR", data_type="number")
                )

        return sheet

    @pytest.fixture
    def large_sheet(self):
        """Create large sheet (> 1M cells)."""
        sheet = SheetData(name="LargeSheet")

        # 2000x1000 = 2M cells
        sheet.max_row = 1999
        sheet.max_column = 999

        # Sparse data
        for row in range(0, 2000, 100):
            for col in range(0, 1000, 50):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value=row + col, data_type="number")
                )

        return sheet

    def test_single_image_strategy(self, builder, small_sheet):
        """Test single image strategy for small sheets."""
        request = builder.build_request(small_sheet, "SmallSheet")

        assert request.prompt_template == "SINGLE_IMAGE"
        assert request.total_images == 1
        assert len(request.images) == 1

        image = request.images[0]
        assert image.image_id == "full_sheet"
        assert image.compression_level == 0  # No compression for small sheet
        assert image.purpose == "Identify all table boundaries in this sheet"
        assert "No compression applied" in image.description

    def test_multi_scale_strategy(self, builder, medium_sheet):
        """Test multi-scale strategy for medium sheets."""
        request = builder.build_request(medium_sheet, "MediumSheet")

        assert request.prompt_template == "EXPLICIT_MULTI_SCALE"
        assert request.total_images >= 2  # Overview + at least one detail

        # Check for overview image
        overview = next((img for img in request.images if img.image_id == "overview"), None)
        assert overview is not None
        assert overview.purpose == "Identify approximate table regions"

        # Check for detail images
        detail_images = [img for img in request.images if img.image_id.startswith("detail_")]
        assert len(detail_images) >= 1

    def test_progressive_strategy(self, builder, large_sheet):
        """Test progressive strategy for very large sheets."""
        request = builder.build_request(large_sheet, "LargeSheet")

        assert request.prompt_template == "PROGRESSIVE"
        assert request.total_images >= 1

        # Should have phase indicators in image IDs
        phase_images = [
            img
            for img in request.images
            if any(phase in img.image_id for phase in ["phase1", "phase2", "phase3"])
        ]
        assert len(phase_images) >= 1

    def test_empty_sheet_handling(self, builder):
        """Test handling of empty sheets."""
        empty_sheet = SheetData(name="EmptySheet")
        request = builder.build_request(empty_sheet, "EmptySheet")

        assert request.prompt_template == "SINGLE_IMAGE"
        assert request.total_images == 0
        assert len(request.images) == 0
        assert request.total_size_mb == 0.0

    def test_size_limit_enforcement(self, builder, large_sheet):
        """Test that size limits are enforced."""
        request = builder.build_request(large_sheet, "LargeSheet")

        # Total size should be under limit
        assert request.validate_size_limit(builder.MAX_TOTAL_SIZE_MB)
        assert request.total_size_mb <= builder.MAX_TOTAL_SIZE_MB

    def test_cost_estimation(self, builder):
        """Test cost estimation for different sheet sizes."""
        # Small sheet
        small = SheetData(name="Small")
        small.max_row = 99
        small.max_column = 9
        small_request = builder.build_request(small, "Small")

        # Medium sheet
        medium = SheetData(name="Medium")
        medium.max_row = 999
        medium.max_column = 99
        medium_request = builder.build_request(medium, "Medium")

        # Costs should increase with size
        assert medium_request.total_size_mb > small_request.total_size_mb

    def test_prompt_generation_single_image(self, builder, small_sheet):
        """Test prompt generation for single image."""
        request = builder.build_request(small_sheet, "SmallSheet")
        prompt = builder.create_explicit_prompt(request)

        assert "Analyze this spreadsheet image" in prompt
        assert "identify ALL distinct table regions" in prompt
        assert "Exact cell range" in prompt

    def test_prompt_generation_multi_scale(self, builder, medium_sheet):
        """Test prompt generation for multi-scale."""
        request = builder.build_request(medium_sheet, "MediumSheet")
        prompt = builder.create_explicit_prompt(request)

        assert "multiple scales" in prompt
        assert "Overview image" in prompt
        assert "Detail image" in prompt
        assert "original sheet coordinates" in prompt

    def test_prompt_generation_progressive(self, builder, large_sheet):
        """Test prompt generation for progressive refinement."""
        request = builder.build_request(large_sheet, "LargeSheet")
        prompt = builder.create_explicit_prompt(request)

        assert "highly compressed" in prompt or "large size" in prompt
        assert "Major table regions" in prompt
        assert "approximate boundaries" in prompt.lower()

    def test_aggressive_compression_fallback(self, builder):
        """Test aggressive compression when approaching limits."""
        # Create sheet that would generate very large images
        huge_sheet = SheetData(name="HugeSheet")
        huge_sheet.max_row = 10000
        huge_sheet.max_column = 1000

        # Fill with enough data to trigger compression
        for i in range(0, 10000, 1000):
            for j in range(0, 1000, 100):
                huge_sheet.set_cell(i, j, CellData(row=i, column=j, value="X", data_type="text"))

        request = builder.build_request(huge_sheet, "HugeSheet")

        # Should use compression
        assert any(img.compression_level >= CompressionLevel.LARGE.value for img in request.images)

    def test_data_region_integration(self, builder):
        """Test integration with DataRegionPreprocessor."""
        sheet = SheetData(name="RegionTest")

        # Create distinct regions
        # Region 1
        for row in range(10):
            for col in range(5):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="R1", data_type="text")
                )

        # Region 2 (separated)
        for row in range(20, 25):
            for col in range(10, 20):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="R2", data_type="text")
                )

        request = builder.build_request(sheet, "RegionTest")

        # For small sheet, still single image, but regions were detected
        assert request.total_images >= 1

    def test_image_encoding(self, builder, small_sheet):
        """Test that images are properly base64 encoded."""
        request = builder.build_request(small_sheet, "SmallSheet")

        for image in request.images:
            # Should be valid base64
            try:
                decoded = base64.b64decode(image.image_data)
                assert len(decoded) > 0
                assert len(decoded) == image.size_bytes
            except Exception as e:
                pytest.fail(f"Invalid base64 encoding: {e}")

    def test_compression_level_metadata(self, builder, medium_sheet):
        """Test compression level metadata in images."""
        request = builder.build_request(medium_sheet, "MediumSheet")

        for image in request.images:
            # Should have valid compression level
            assert 0 <= image.compression_level <= 5

            # Block size should match compression level
            if image.compression_level > 0:
                level = CompressionLevel(image.compression_level)
                assert image.block_size[0] == level.row_block
                assert image.block_size[1] == level.col_block

    def test_excel_range_coverage(self, builder):
        """Test Excel range coverage in image metadata."""
        sheet = SheetData(name="RangeTest")

        # Data in specific range
        for row in range(5, 15):
            for col in range(2, 8):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="Data", data_type="text")
                )

        request = builder.build_request(sheet, "RangeTest")

        for image in request.images:
            # Should have valid Excel range
            assert image.covers_cells
            assert ":" in image.covers_cells  # Format like "A1:H15"

    def test_strategy_selection_edge_cases(self, builder):
        """Test strategy selection at boundaries."""
        # Exactly at single image threshold
        sheet1 = SheetData(name="Boundary1")
        sheet1.max_row = 315  # ~100K cells with 316 cols
        sheet1.max_column = 315

        request1 = builder.build_request(sheet1, "Boundary1")
        assert request1.prompt_template in ["SINGLE_IMAGE", "EXPLICIT_MULTI_SCALE"]

        # Just over multi-scale threshold
        sheet2 = SheetData(name="Boundary2")
        sheet2.max_row = 1000
        sheet2.max_column = 1000  # Just over 1M cells

        request2 = builder.build_request(sheet2, "Boundary2")
        assert request2.prompt_template in ["EXPLICIT_MULTI_SCALE", "PROGRESSIVE"]
