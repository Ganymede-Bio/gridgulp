"""Unit tests for BitmapGenerator compression levels."""

import pytest
from PIL import Image
import numpy as np
import io

from gridporter.models.multi_scale import CompressionLevel
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.vision.bitmap_generator import BitmapGenerator
from gridporter.vision.quadtree import QuadTreeBounds


class TestBitmapGeneratorCompression:
    """Test BitmapGenerator with new compression levels."""

    @pytest.fixture
    def generator(self):
        """Create bitmap generator."""
        return BitmapGenerator(cell_width=10, cell_height=10, mode="binary", auto_compress=True)

    @pytest.fixture
    def small_sheet(self):
        """Create small sheet for testing."""
        sheet = SheetData(name="SmallSheet")

        # 10x10 grid with some data
        for row in range(10):
            for col in range(10):
                if row < 5 and col < 5:
                    sheet.set_cell(
                        row,
                        col,
                        CellData(
                            row=row,
                            column=col,
                            value=f"Cell_{row}_{col}",
                            data_type="text",
                            is_bold=(row == 0),
                        ),
                    )

        return sheet

    @pytest.fixture
    def large_sheet(self):
        """Create large sheet for compression testing."""
        sheet = SheetData(name="LargeSheet")

        # 1000x100 grid with sparse data
        for row in range(0, 1000, 10):
            for col in range(0, 100, 2):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value=row * col, data_type="number")
                )

        return sheet

    @pytest.fixture
    def excel_limit_sheet(self):
        """Create sheet at Excel limits."""
        sheet = SheetData(name="ExcelLimitSheet")

        # Don't actually fill 1M rows, just set bounds
        sheet.max_row = 1_048_575  # Excel max
        sheet.max_column = 16_383  # Excel max

        # Add some sparse data
        for row in range(0, 1000, 100):
            for col in range(0, 100, 10):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="Sparse", data_type="text")
                )

        return sheet

    def test_compression_level_none(self, generator, small_sheet):
        """Test no compression (level 0)."""
        generator.auto_compress = False
        img_data, metadata = generator.generate(small_sheet)

        # With no compression, each cell should be clearly represented
        assert metadata.compression_level == CompressionLevel.NONE
        assert metadata.block_size == [1, 1]
        assert metadata.total_rows == 10
        assert metadata.total_cols == 10

        # Image dimensions should be cells * cell_size
        assert metadata.width == 10 * generator.cell_width
        assert metadata.height == 10 * generator.cell_height

        # Verify image is valid PNG
        img = Image.open(io.BytesIO(img_data))
        assert img.format == "PNG"
        assert img.mode == "L"  # Grayscale

    def test_compression_level_selection(self, generator):
        """Test automatic compression level selection."""
        # Test different sizes
        test_cases = [
            (100, 100, CompressionLevel.NONE),  # 10K cells
            (500, 200, CompressionLevel.NONE),  # 100K cells
            (2000, 100, CompressionLevel.MILD),  # 200K cells
            (5000, 200, CompressionLevel.EXCEL_RATIO),  # 1M cells
            (10000, 500, CompressionLevel.LARGE),  # 5M cells
        ]

        for rows, cols, expected_level in test_cases:
            level = generator._select_compression_level(rows, cols)
            assert (
                level == expected_level
            ), f"Expected {expected_level} for {rows}x{cols}, got {level}"

    def test_excel_proportions(self, generator, large_sheet):
        """Test Excel proportion maintenance in compression."""
        # Force Excel ratio compression
        generator.auto_compress = False
        generator.compression_level = 7

        # Manually set compression level
        result = generator._generate_at_level(
            large_sheet, None, 1000, 100, CompressionLevel.EXCEL_RATIO
        )
        img_data, metadata = result

        assert metadata.compression_level == CompressionLevel.EXCEL_RATIO
        assert metadata.block_size == [64, 1]

        # Check output dimensions
        expected_rows = (1000 + 63) // 64  # Ceiling division
        expected_cols = 100  # No column compression at this level

        # Actual pixel dimensions depend on visualization
        assert metadata.total_rows == 1000
        assert metadata.total_cols == 100

    def test_large_compression_maintains_ratio(self, generator):
        """Test LARGE compression maintains 64:1 ratio."""
        rows, cols = 10000, 1000
        result = generator._generate_compressed(
            SheetData(name="Test"), None, rows, cols, CompressionLevel.LARGE
        )
        img_data, metadata = result

        assert metadata.compression_level == CompressionLevel.LARGE
        assert metadata.block_size == [256, 4]
        assert metadata.block_size[0] / metadata.block_size[1] == 64

    def test_maximum_compression(self, generator, excel_limit_sheet):
        """Test maximum compression for Excel-limit sheets."""
        # This should trigger maximum compression
        img_data, metadata = generator.generate(excel_limit_sheet)

        # Should use maximum compression for such a large sheet
        assert metadata.compression_level in [CompressionLevel.MAXIMUM, CompressionLevel.HUGE]

    def test_block_analysis_accuracy(self, generator):
        """Test that block-based analysis correctly identifies content."""
        sheet = SheetData(name="BlockTest")

        # Create distinct regions with different densities
        # Dense region (0,0) to (63,0) - exactly one 64x1 block
        for row in range(64):
            sheet.set_cell(row, 0, CellData(row=row, column=0, value="Dense", data_type="text"))

        # Sparse region (64,0) to (127,0) - one cell per 16 rows
        for row in range(64, 128, 16):
            sheet.set_cell(row, 0, CellData(row=row, column=0, value="Sparse", data_type="text"))

        # Header region (128,0) to (191,0) - bold cells
        for row in range(128, 192, 8):
            sheet.set_cell(
                row, 0, CellData(row=row, column=0, value="Header", data_type="text", is_bold=True)
            )

        # Generate with EXCEL_RATIO compression
        result = generator._generate_compressed(sheet, None, 192, 1, CompressionLevel.EXCEL_RATIO)
        img_data, metadata = result

        # Load image and check pixel values
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)

        # Should have different gray values for different block types
        # Due to compression, we should see variation in the output

    def test_size_constraints(self, generator):
        """Test that size constraints are enforced."""
        # Create sheet that would generate huge uncompressed image
        sheet = SheetData(name="HugeSheet")
        sheet.max_row = 10000
        sheet.max_column = 1000

        # Add minimal data
        sheet.set_cell(0, 0, CellData(row=0, column=0, value="A", data_type="text"))
        sheet.set_cell(9999, 999, CellData(row=9999, column=999, value="Z", data_type="text"))

        # Generate - should auto-compress
        img_data, metadata = generator.generate(sheet)

        # Check that output dimensions are reasonable
        assert metadata.width <= generator.MAX_WIDTH
        assert metadata.height <= generator.MAX_HEIGHT

        # Should have selected appropriate compression
        assert metadata.compression_level.value >= CompressionLevel.LARGE.value

    def test_edge_cases(self, generator):
        """Test edge cases for compression."""
        # Single cell sheet
        single_cell = SheetData(name="SingleCell")
        single_cell.set_cell(0, 0, CellData(row=0, column=0, value="Only", data_type="text"))

        img_data, metadata = generator.generate(single_cell)
        assert metadata.compression_level == CompressionLevel.NONE
        assert metadata.total_rows == 1
        assert metadata.total_cols == 1

        # Empty sheet
        empty = SheetData(name="Empty")
        img_data, metadata = generator.generate(empty)
        assert metadata.total_rows == 0
        assert metadata.total_cols == 0

    def test_compression_with_regions(self, generator):
        """Test compression with specific regions."""
        sheet = SheetData(name="RegionTest")

        # Create data in specific region
        for row in range(100, 200):
            for col in range(50, 100):
                if (row + col) % 5 == 0:
                    sheet.set_cell(
                        row, col, CellData(row=row, column=col, value="R", data_type="text")
                    )

        # Generate bitmap for just the region
        region = QuadTreeBounds(100, 50, 199, 99)
        img_data, metadata = generator.generate(sheet, region)

        assert metadata.region_bounds == region
        assert metadata.total_rows == 100
        assert metadata.total_cols == 50

    def test_compression_visual_quality(self, generator):
        """Test that compressed images maintain visual distinction."""
        sheet = SheetData(name="VisualTest")

        # Create pattern that should be visible even compressed
        # Horizontal stripes
        for row in range(0, 256, 32):
            for col in range(64):
                sheet.set_cell(row, col, CellData(row=row, column=col, value="=", data_type="text"))

        # Vertical stripes
        for row in range(256):
            for col in range(0, 64, 8):
                sheet.set_cell(row, col, CellData(row=row, column=col, value="|", data_type="text"))

        # Generate with LARGE compression
        result = generator._generate_compressed(sheet, None, 256, 64, CompressionLevel.LARGE)
        img_data, metadata = result

        # Pattern should still be distinguishable in compressed output
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)

        # Should have variation (not all same color)
        assert img_array.min() < img_array.max()

    def test_mode_compatibility(self):
        """Test compression works with different visualization modes."""
        sheet = SheetData(name="ModeTest")

        # Add variety of cell types
        sheet.set_cell(
            0, 0, CellData(row=0, column=0, value="Bold", is_bold=True, data_type="text")
        )
        sheet.set_cell(
            1, 0, CellData(row=1, column=0, value="=A1", has_formula=True, data_type="formula")
        )
        sheet.set_cell(
            2, 0, CellData(row=2, column=0, value="Merged", is_merged=True, data_type="text")
        )

        for mode in ["binary", "grayscale", "color"]:
            gen = BitmapGenerator(mode=mode)
            img_data, metadata = gen.generate(sheet)

            assert (
                metadata.mode.startswith(mode)
                or metadata.mode == f"compressed_{metadata.compression_level.value}"
            )

            # Verify image format matches mode
            img = Image.open(io.BytesIO(img_data))
            if mode == "color":
                assert img.mode == "RGB"
            else:
                assert img.mode == "L"
