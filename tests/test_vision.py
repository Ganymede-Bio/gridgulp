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
        assert metadata.total_rows == 1  # Minimum 1x1 grid
        assert metadata.total_cols == 1  # Minimum 1x1 grid
        assert metadata.width == 10
        assert metadata.height == 10

    def test_generate_simple_sheet(self):
        """Test generating bitmap from simple sheet with data."""
        sheet_data = SheetData(name="Test")
        # Add some cells
        sheet_data.cells["A1"] = CellData(
            value="Header", data_type="text", is_bold=True, row=0, column=0
        )
        sheet_data.cells["A2"] = CellData(value="Data1", data_type="text", row=1, column=0)
        sheet_data.cells["B1"] = CellData(
            value="Count", data_type="text", is_bold=True, row=0, column=1
        )
        sheet_data.cells["B2"] = CellData(value=42, data_type="number", row=1, column=1)
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
        sheet_data.cells["A1"] = CellData(
            value="Bold", data_type="text", is_bold=True, row=0, column=0
        )
        sheet_data.cells["A2"] = CellData(value=123, data_type="number", row=1, column=0)
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
        assert proposal.characteristics.get("has_headers") is True
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
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_col_to_letter(self):
        """Test column index to Excel letter conversion."""
        proposer = RegionProposer()

        assert proposer._col_to_letter(0) == "A"
        assert proposer._col_to_letter(1) == "B"
        assert proposer._col_to_letter(25) == "Z"
        assert proposer._col_to_letter(26) == "AA"
        assert proposer._col_to_letter(27) == "AB"
        assert proposer._col_to_letter(701) == "ZZ"


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
        sheet_data.cells["A1"] = CellData(
            value="Name", data_type="text", is_bold=True, row=0, column=0
        )
        sheet_data.cells["B1"] = CellData(
            value="Age", data_type="text", is_bold=True, row=0, column=1
        )
        sheet_data.cells["A2"] = CellData(value="Alice", data_type="text", row=1, column=0)
        sheet_data.cells["B2"] = CellData(value=25, data_type="number", row=1, column=1)
        sheet_data.cells["A3"] = CellData(value="Bob", data_type="text", row=2, column=0)
        sheet_data.cells["B3"] = CellData(value=30, data_type="number", row=2, column=1)
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


class TestBitmapGeneratorErrorHandling:
    """Test error handling and edge cases for bitmap generation."""

    def test_generate_with_scaling_edge_case(self):
        """Test bitmap generation with extreme scaling requirements."""
        sheet_data = SheetData(name="Large")
        # Create a very large sheet that will trigger scaling
        sheet_data.max_row = 1000
        sheet_data.max_column = 1000

        generator = BitmapGenerator(cell_width=50, cell_height=50)
        image_bytes, metadata = generator.generate(sheet_data)

        # Should be scaled down
        assert metadata.scale_factor < 1.0
        assert metadata.width <= generator.MAX_WIDTH
        assert metadata.height <= generator.MAX_HEIGHT

    def test_generate_with_merged_cells(self):
        """Test bitmap generation with merged cells."""
        sheet_data = SheetData(name="Merged")
        sheet_data.cells["A1"] = CellData(
            value="Merged", data_type="text", is_merged=True, row=0, column=0
        )
        sheet_data.max_row = 0
        sheet_data.max_column = 0

        generator = BitmapGenerator(mode="grayscale")
        image_bytes, metadata = generator.generate(sheet_data)

        assert isinstance(image_bytes, bytes)
        assert metadata.total_rows == 1
        assert metadata.total_cols == 1

    def test_generate_with_formulas(self):
        """Test bitmap generation with formula cells."""
        sheet_data = SheetData(name="Formulas")
        sheet_data.cells["A1"] = CellData(
            value="=SUM(B1:B10)", data_type="formula", has_formula=True, row=0, column=0
        )
        sheet_data.max_row = 0
        sheet_data.max_column = 0

        generator = BitmapGenerator(mode="color")
        image_bytes, metadata = generator.generate(sheet_data)

        assert isinstance(image_bytes, bytes)
        # Verify PNG format
        image = Image.open(io.BytesIO(image_bytes))
        assert image.format == "PNG"
        assert image.mode == "RGB"

    def test_pixel_to_cell_edge_cases(self):
        """Test pixel to cell conversion edge cases."""
        metadata = BitmapMetadata(
            width=30,
            height=20,
            cell_width=10,
            cell_height=10,
            total_rows=2,
            total_cols=3,
            scale_factor=1.0,
            mode="binary",
        )
        generator = BitmapGenerator()

        # Test boundary cases
        assert generator.pixel_to_cell(-1, -1, metadata) == (-1, -1)  # Negative coords
        assert generator.pixel_to_cell(0, 0, metadata) == (0, 0)  # Origin
        assert generator.pixel_to_cell(29, 19, metadata) == (1, 2)  # Max valid coords
        assert generator.pixel_to_cell(100, 100, metadata) == (10, 10)  # Beyond bounds

    def test_cell_to_pixel_edge_cases(self):
        """Test cell to pixel conversion edge cases."""
        metadata = BitmapMetadata(
            width=30,
            height=20,
            cell_width=10,
            cell_height=10,
            total_rows=2,
            total_cols=3,
            scale_factor=1.0,
            mode="binary",
        )
        generator = BitmapGenerator()

        # Test boundary cases
        assert generator.cell_to_pixel(-1, -1, metadata) == (-10, -10, 0, 0)
        assert generator.cell_to_pixel(0, 0, metadata) == (0, 0, 10, 10)
        assert generator.cell_to_pixel(10, 10, metadata) == (100, 100, 110, 110)

    def test_create_prompt_description_all_modes(self):
        """Test prompt description for all visualization modes."""
        metadata = BitmapMetadata(
            width=100,
            height=80,
            cell_width=10,
            cell_height=8,
            total_rows=8,
            total_cols=10,
            scale_factor=1.0,
            mode="test",
        )

        # Test binary mode
        binary_gen = BitmapGenerator(mode="binary")
        binary_desc = binary_gen.create_prompt_description(metadata)
        assert "Black pixels represent filled cells" in binary_desc
        assert "white pixels represent empty cells" in binary_desc

        # Test grayscale mode
        gray_gen = BitmapGenerator(mode="grayscale")
        gray_desc = gray_gen.create_prompt_description(metadata)
        assert "Black pixels (0) represent bold/header cells" in gray_desc
        assert "dark gray (64) represents formula cells" in gray_desc

        # Test color mode
        color_gen = BitmapGenerator(mode="color")
        color_desc = color_gen.create_prompt_description(metadata)
        assert "Black pixels represent bold/header cells" in color_desc
        assert "blue represents formula cells" in color_desc
        assert "green represents numeric data" in color_desc


class TestRegionProposerErrorHandling:
    """Test error handling and edge cases for region proposal parsing."""

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON response."""
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

        malformed_json = '{"tables": [{"bounds": {"x1": 0, "y1": 0'  # Incomplete JSON

        proposals = proposer.parse_response(malformed_json, metadata)

        # Should gracefully fall back to text parsing (and find nothing)
        assert len(proposals) == 0

    def test_parse_json_missing_fields(self):
        """Test parsing JSON with missing required fields."""
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

        incomplete_json = """
        {
            "tables": [
                {"confidence": 0.9},
                {"bounds": {"x1": 0}, "confidence": 0.8}
            ]
        }
        """

        proposals = proposer.parse_response(incomplete_json, metadata)

        # Should skip invalid entries
        assert len(proposals) == 0

    def test_parse_text_no_coordinates(self):
        """Test parsing text response without coordinate information."""
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
        I can see some tables but cannot determine exact coordinates.
        There might be data here somewhere.
        """

        proposals = proposer.parse_response(text_response, metadata)

        # Should find no valid proposals without coordinates
        assert len(proposals) == 0

    def test_parse_text_invalid_coordinates(self):
        """Test parsing text with invalid coordinate formats."""
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
        Table at coordinates (abc, def, ghi, jkl) with confidence 0.9.
        Another table at (10, 20) incomplete coordinates.
        """

        proposals = proposer.parse_response(text_response, metadata)

        # Should find no valid proposals with invalid coordinates
        assert len(proposals) == 0

    def test_parse_confidence_percentage(self):
        """Test parsing confidence as percentage (>1.0)."""
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
        Table at coordinates (10, 10, 50, 50) with confidence 85%.
        """

        proposals = proposer.parse_response(text_response, metadata)

        assert len(proposals) == 1
        # Should convert percentage to decimal
        assert proposals[0].confidence == 0.85

    def test_filter_proposals_edge_cases(self):
        """Test proposal filtering with edge cases."""
        proposer = RegionProposer()

        # Create proposals with edge case values
        proposals = [
            RegionProposal(
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                start_row=0,
                start_col=0,
                end_row=0,
                end_col=0,  # 1x1 table
                excel_range="A1:A1",
                confidence=1.0,
                characteristics={},
            ),
            RegionProposal(
                x1=0,
                y1=0,
                x2=20,
                y2=10,
                start_row=0,
                start_col=0,
                end_row=0,
                end_col=1,  # 1x2 table
                excel_range="A1:B1",
                confidence=0.0,
                characteristics={},
            ),
        ]

        # Filter with strict size requirements
        filtered = proposer.filter_proposals(proposals, min_confidence=0.5, min_size=(2, 2))
        assert len(filtered) == 0  # Both should be filtered out

        # Filter with relaxed size requirements
        filtered = proposer.filter_proposals(proposals, min_confidence=0.5, min_size=(1, 1))
        assert len(filtered) == 1  # Only high confidence one remains

    def test_col_to_letter_edge_cases(self):
        """Test column letter conversion edge cases."""
        proposer = RegionProposer()

        # Test extreme values
        assert proposer._col_to_letter(0) == "A"
        assert proposer._col_to_letter(25) == "Z"
        assert proposer._col_to_letter(26) == "AA"
        assert proposer._col_to_letter(51) == "AZ"
        assert proposer._col_to_letter(52) == "BA"

        # Test very large column numbers
        assert proposer._col_to_letter(16383) == "XFD"  # Excel's max column


class TestVisionResultModels:
    """Test vision result Pydantic models."""

    def test_vision_region_creation(self):
        """Test VisionRegion model creation and validation."""
        from gridporter.models.vision_result import VisionRegion

        region = VisionRegion(
            pixel_bounds={"x1": 10, "y1": 20, "x2": 100, "y2": 80},
            cell_bounds={"start_row": 2, "start_col": 1, "end_row": 7, "end_col": 9},
            range="B3:J8",
            confidence=0.92,
            characteristics={"has_headers": True, "type": "data_table"},
        )

        assert region.pixel_bounds["x1"] == 10
        assert region.cell_bounds["start_row"] == 2
        assert region.range == "B3:J8"
        assert region.confidence == 0.92
        assert region.detection_method == "vision"  # Default value
        assert region.characteristics["has_headers"] is True

    def test_vision_region_to_table_range(self):
        """Test VisionRegion conversion to TableRange."""
        from gridporter.models.vision_result import VisionRegion

        region = VisionRegion(
            pixel_bounds={"x1": 0, "y1": 0, "x2": 50, "y2": 30},
            cell_bounds={"start_row": 0, "start_col": 0, "end_row": 2, "end_col": 4},
            range="A1:E3",
            confidence=0.85,
        )

        table_range = region.to_table_range()

        assert table_range.excel_range == "A1:E3"
        assert table_range.start_row == 0
        assert table_range.start_col == 0
        assert table_range.end_row == 2
        assert table_range.end_col == 4

    def test_vision_analysis_result_operations(self):
        """Test VisionAnalysisResult operations."""
        from gridporter.models.vision_result import VisionAnalysisResult, VisionRegion

        regions = [
            VisionRegion(
                pixel_bounds={"x1": 0, "y1": 0, "x2": 50, "y2": 30},
                cell_bounds={"start_row": 0, "start_col": 0, "end_row": 2, "end_col": 4},
                range="A1:E3",
                confidence=0.9,
            ),
            VisionRegion(
                pixel_bounds={"x1": 60, "y1": 0, "x2": 120, "y2": 40},
                cell_bounds={"start_row": 0, "start_col": 6, "end_row": 3, "end_col": 11},
                range="G1:L4",
                confidence=0.6,
            ),
            VisionRegion(
                pixel_bounds={"x1": 0, "y1": 50, "x2": 40, "y2": 70},
                cell_bounds={"start_row": 5, "start_col": 0, "end_row": 6, "end_col": 3},
                range="A6:D7",
                confidence=0.4,
            ),
        ]

        result = VisionAnalysisResult(
            regions=regions, bitmap_info={"width": 200, "height": 100, "mode": "binary"}
        )

        # Test high confidence filtering
        high_conf = result.high_confidence_regions(threshold=0.7)
        assert len(high_conf) == 1
        assert high_conf[0].confidence == 0.9

        # Test conversion to table ranges
        table_ranges = result.to_table_ranges()
        assert len(table_ranges) == 3
        assert table_ranges[0].excel_range == "A1:E3"
        assert table_ranges[1].excel_range == "G1:L4"
        assert table_ranges[2].excel_range == "A6:D7"

    def test_vision_detection_metrics(self):
        """Test VisionDetectionMetrics operations."""
        from gridporter.models.vision_result import VisionDetectionMetrics

        metrics = VisionDetectionMetrics(
            bitmap_generation_time=0.15,
            vision_analysis_time=2.4,
            parsing_time=0.05,
            total_time=2.6,
            regions_detected=5,
            regions_filtered=3,
            confidence_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            tokens_used={"total_tokens": 150},
        )

        assert metrics.average_confidence() == pytest.approx(0.7)
        assert metrics.max_confidence() == 0.9
        assert metrics.min_confidence() == 0.5

        # Test with empty scores
        empty_metrics = VisionDetectionMetrics(
            bitmap_generation_time=0.1,
            vision_analysis_time=1.0,
            parsing_time=0.02,
            total_time=1.12,
            regions_detected=0,
            regions_filtered=0,
            confidence_scores=[],
        )

        assert empty_metrics.average_confidence() == 0.0
        assert empty_metrics.max_confidence() == 0.0
        assert empty_metrics.min_confidence() == 0.0


class TestVisionIntegrationAsync:
    """Async integration tests for vision components."""

    @pytest.mark.asyncio
    async def test_vision_pipeline_analyze_sheet_async(self):
        """Test async vision pipeline analysis."""
        from unittest.mock import AsyncMock, patch
        from gridporter.vision.pipeline import VisionPipeline
        from gridporter.vision.vision_models import VisionModelResponse

        config = Config(confidence_threshold=0.5, min_table_size=(1, 1))
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="AsyncTest")
        sheet_data.cells["A1"] = CellData(value="Test", data_type="text", row=0, column=0)
        sheet_data.max_row = 0
        sheet_data.max_column = 0

        # Mock async vision model
        mock_model = AsyncMock()
        mock_response = VisionModelResponse(
            content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": 0.8}]}',
            model="async-test-model",
        )
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            result = await pipeline.analyze_sheet(sheet_data)

        assert len(result.regions) == 1
        assert result.regions[0].confidence == 0.8
        mock_model.analyze_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_vision_pipeline_batch_analyze_async(self):
        """Test async batch analysis."""
        from unittest.mock import AsyncMock, patch
        from gridporter.vision.pipeline import VisionPipeline
        from gridporter.vision.vision_models import VisionModelResponse

        config = Config()
        pipeline = VisionPipeline(config)

        sheets = [SheetData(name="Sheet1"), SheetData(name="Sheet2")]

        mock_model = AsyncMock()
        mock_model.supports_batch = False
        mock_response = VisionModelResponse(content='{"tables": []}', model="test")
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            results = await pipeline.batch_analyze(sheets)

        assert len(results) == 2
        assert mock_model.analyze_image.call_count == 2
