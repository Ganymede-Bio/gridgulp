"""Tests for vision-based table detection components."""

import io
from pathlib import Path

import pytest
from PIL import Image

from gridporter.config import Config
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.vision.bitmap_generator import BitmapGenerator, BitmapMetadata
from gridporter.vision.region_proposer import RegionProposal, RegionProposer


class TestBitmapGenerator:
    """Test bitmap generation functionality."""

    def test_init_defaults(self):
        """Test bitmap generator initialization with defaults."""
        generator = BitmapGenerator()
        assert generator.cell_width == 10
        assert generator.cell_height == 10
        assert generator.mode == "binary"

    def test_init_custom_params(self):
        """Test bitmap generator initialization with custom parameters."""
        generator = BitmapGenerator(cell_width=20, cell_height=15, mode="grayscale")
        assert generator.cell_width == 20
        assert generator.cell_height == 15
        assert generator.mode == "grayscale"

    def test_init_minimum_cell_size(self):
        """Test that cell size is enforced to be at least minimum."""
        generator = BitmapGenerator(cell_width=1, cell_height=1)
        assert generator.cell_width == 3  # MIN_CELL_SIZE
        assert generator.cell_height == 3  # MIN_CELL_SIZE

    def test_generate_empty_sheet(self):
        """Test generating bitmap from empty sheet."""
        sheet_data = SheetData(name="Empty")
        generator = BitmapGenerator()

        image_bytes, metadata = generator.generate(sheet_data)

        assert isinstance(image_bytes, bytes)
        assert isinstance(metadata, BitmapMetadata)
        assert metadata.total_rows == 0
        assert metadata.total_cols == 0
        assert metadata.width == 10
        assert metadata.height == 10

    def test_generate_simple_sheet(self):
        """Test generating bitmap from simple sheet with data."""
        sheet_data = SheetData(name="Test")
        # Add some cells
        sheet_data.cells["A1"] = CellData(value="Header", data_type="text", is_bold=True)
        sheet_data.cells["A2"] = CellData(value="Data1", data_type="text")
        sheet_data.cells["B1"] = CellData(value="Count", data_type="text", is_bold=True)
        sheet_data.cells["B2"] = CellData(value=42, data_type="number")
        sheet_data.max_row = 1
        sheet_data.max_column = 1

        generator = BitmapGenerator()
        image_bytes, metadata = generator.generate(sheet_data)

        assert isinstance(image_bytes, bytes)
        assert metadata.total_rows == 2
        assert metadata.total_cols == 2
        assert metadata.width == 20  # 2 cols * 10px each
        assert metadata.height == 20  # 2 rows * 10px each
        assert metadata.mode == "binary"

        # Verify it's a valid PNG
        image = Image.open(io.BytesIO(image_bytes))
        assert image.format == "PNG"
        assert image.size == (20, 20)

    def test_generate_color_mode(self):
        """Test generating bitmap in color mode."""
        sheet_data = SheetData(name="Color")
        sheet_data.cells["A1"] = CellData(value="Bold", data_type="text", is_bold=True)
        sheet_data.cells["A2"] = CellData(value=123, data_type="number")
        sheet_data.max_row = 1
        sheet_data.max_column = 0

        generator = BitmapGenerator(mode="color")
        image_bytes, metadata = generator.generate(sheet_data)

        # Verify it's a valid color PNG
        image = Image.open(io.BytesIO(image_bytes))
        assert image.mode == "RGB"

    def test_pixel_to_cell_conversion(self):
        """Test pixel to cell coordinate conversion."""
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=10,
            total_cols=10,
            scale_factor=1.0,
            mode="binary",
        )
        generator = BitmapGenerator()

        # Test various pixel coordinates
        assert generator.pixel_to_cell(0, 0, metadata) == (0, 0)
        assert generator.pixel_to_cell(15, 12, metadata) == (1, 1)
        assert generator.pixel_to_cell(9, 7, metadata) == (0, 0)  # Still in first cell

    def test_cell_to_pixel_conversion(self):
        """Test cell to pixel coordinate conversion."""
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=10,
            total_cols=10,
            scale_factor=1.0,
            mode="binary",
        )
        generator = BitmapGenerator()

        # Test various cell coordinates
        assert generator.cell_to_pixel(0, 0, metadata) == (0, 0, 10, 8)
        assert generator.cell_to_pixel(1, 1, metadata) == (10, 8, 20, 16)
        assert generator.cell_to_pixel(2, 3, metadata) == (30, 16, 40, 24)

    def test_create_prompt_description(self):
        """Test prompt description generation."""
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=10,
            total_cols=10,
            scale_factor=1.0,
            mode="binary",
        )
        generator = BitmapGenerator(mode="binary")

        description = generator.create_prompt_description(metadata)

        assert "10 rows and 10 columns" in description
        assert "10x8 pixel rectangle" in description
        assert "Black pixels represent filled cells" in description
        assert "bounding box in pixel coordinates" in description


class TestRegionProposer:
    """Test region proposal parsing."""

    def test_init(self):
        """Test region proposer initialization."""
        proposer = RegionProposer()
        assert proposer.pixel_pattern is not None
        assert proposer.range_pattern is not None
        assert proposer.confidence_pattern is not None

    def test_parse_json_response(self):
        """Test parsing structured JSON response."""
        proposer = RegionProposer()
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=10,
            total_cols=10,
            scale_factor=1.0,
            mode="binary",
        )

        json_response = """
        {
            "tables": [
                {
                    "bounds": {"x1": 0, "y1": 0, "x2": 30, "y2": 16},
                    "confidence": 0.9,
                    "characteristics": {"has_headers": true}
                }
            ]
        }
        """

        proposals = proposer.parse_response(json_response, metadata)

        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.x1 == 0
        assert proposal.y1 == 0
        assert proposal.x2 == 30
        assert proposal.y2 == 16
        assert proposal.confidence == 0.9
        assert proposal.characteristics["has_headers"] is True
        assert proposal.excel_range == "A1:C2"

    def test_parse_text_response(self):
        """Test parsing unstructured text response."""
        proposer = RegionProposer()
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=10,
            total_cols=10,
            scale_factor=1.0,
            mode="binary",
        )

        text_response = """
        Table 1: I found a table at coordinates (10, 8, 40, 32) with confidence 0.85.
        This region appears to have headers in the first row.
        """

        proposals = proposer.parse_response(text_response, metadata)

        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.x1 == 10
        assert proposal.y1 == 8
        assert proposal.x2 == 40
        assert proposal.y2 == 32
        assert proposal.confidence == 0.85
        assert proposal.characteristics.get("has_headers") is True

    def test_filter_proposals(self):
        """Test proposal filtering by confidence and size."""
        proposer = RegionProposer()

        # Create test proposals
        proposals = [
            RegionProposal(
                x1=0,
                y1=0,
                x2=30,
                y2=16,
                start_row=0,
                start_col=0,
                end_row=1,
                end_col=2,
                excel_range="A1:C2",
                confidence=0.9,
                characteristics={},
            ),
            RegionProposal(
                x1=40,
                y1=0,
                x2=50,
                y2=8,
                start_row=0,
                start_col=4,
                end_row=0,
                end_col=4,
                excel_range="E1:E1",
                confidence=0.3,
                characteristics={},
            ),
            RegionProposal(
                x1=0,
                y1=20,
                x2=20,
                y2=28,
                start_row=2,
                start_col=0,
                end_row=2,
                end_col=1,
                excel_range="A3:B3",
                confidence=0.8,
                characteristics={},
            ),
        ]

        # Filter with default thresholds
        filtered = proposer.filter_proposals(proposals)

        # Should keep proposals with confidence >= 0.5 and size >= 2x2
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9
        assert filtered[1].confidence == 0.8

    def test_col_to_letter(self):
        """Test column index to Excel letter conversion."""
        proposer = RegionProposer()

        assert proposer._col_to_letter(0) == "A"
        assert proposer._col_to_letter(1) == "B"
        assert proposer._col_to_letter(25) == "Z"
        assert proposer._col_to_letter(26) == "AA"
        assert proposer._col_to_letter(27) == "AB"
        assert proposer._col_to_letter(701) == "ZZ"


@pytest.mark.asyncio
class TestVisionIntegration:
    """Integration tests for vision components."""

    def test_vision_config_integration(self):
        """Test that vision configuration works with main config."""
        config = Config(vision_cell_width=15, vision_cell_height=12, vision_mode="grayscale")

        assert config.vision_cell_width == 15
        assert config.vision_cell_height == 12
        assert config.vision_mode == "grayscale"

    def test_vision_models_import(self):
        """Test that vision models can be imported."""
        from gridporter.models.vision_result import (
            VisionAnalysisResult,
            VisionDetectionMetrics,
            VisionRegion,
        )

        # Test creating instances
        region = VisionRegion(
            pixel_bounds={"x1": 0, "y1": 0, "x2": 100, "y2": 50},
            cell_bounds={"start_row": 0, "start_col": 0, "end_row": 4, "end_col": 9},
            range="A1:J5",
            confidence=0.9,
        )

        result = VisionAnalysisResult(regions=[region], bitmap_info={"width": 100, "height": 50})

        assert len(result.regions) == 1
        assert result.regions[0].confidence == 0.9

    def test_bitmap_generation_end_to_end(self):
        """Test complete bitmap generation workflow."""
        # Create test sheet
        sheet_data = SheetData(name="Test")
        sheet_data.cells["A1"] = CellData(value="Name", data_type="text", is_bold=True)
        sheet_data.cells["B1"] = CellData(value="Age", data_type="text", is_bold=True)
        sheet_data.cells["A2"] = CellData(value="Alice", data_type="text")
        sheet_data.cells["B2"] = CellData(value=25, data_type="number")
        sheet_data.cells["A3"] = CellData(value="Bob", data_type="text")
        sheet_data.cells["B3"] = CellData(value=30, data_type="number")
        sheet_data.max_row = 2
        sheet_data.max_column = 1

        # Generate bitmap
        generator = BitmapGenerator(cell_width=15, cell_height=12, mode="color")
        image_bytes, metadata = generator.generate(sheet_data)

        # Verify bitmap properties
        assert metadata.total_rows == 3
        assert metadata.total_cols == 2
        assert metadata.cell_width == 15
        assert metadata.cell_height == 12

        # Verify image
        image = Image.open(io.BytesIO(image_bytes))
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (30, 36)  # 2*15, 3*12

        # Test coordinate conversions
        assert generator.pixel_to_cell(0, 0, metadata) == (0, 0)
        assert generator.pixel_to_cell(20, 25, metadata) == (2, 1)

        x1, y1, x2, y2 = generator.cell_to_pixel(1, 0, metadata)
        assert (x1, y1, x2, y2) == (0, 12, 15, 24)
