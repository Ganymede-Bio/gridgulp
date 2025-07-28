"""Unit tests for multi-scale data models."""

import pytest
from pydantic import ValidationError

from gridporter.models.multi_scale import (
    CompressionLevel,
    DataRegion,
    ExactBounds,
    MultiScaleBitmaps,
    ProgressiveRefinementPhase,
    VisionImage,
    VisionRequest,
)


class TestCompressionLevel:
    """Test CompressionLevel enum."""

    def test_compression_levels(self):
        """Test all compression levels have correct values."""
        assert CompressionLevel.NONE.value == 0
        assert CompressionLevel.MILD.value == 1
        assert CompressionLevel.EXCEL_RATIO.value == 2
        assert CompressionLevel.LARGE.value == 3
        assert CompressionLevel.HUGE.value == 4
        assert CompressionLevel.MAXIMUM.value == 5

    def test_row_blocks(self):
        """Test row block sizes maintain Excel proportions."""
        assert CompressionLevel.NONE.row_block == 1
        assert CompressionLevel.MILD.row_block == 16
        assert CompressionLevel.EXCEL_RATIO.row_block == 64
        assert CompressionLevel.LARGE.row_block == 256
        assert CompressionLevel.HUGE.row_block == 1024
        assert CompressionLevel.MAXIMUM.row_block == 4096

    def test_col_blocks(self):
        """Test column block sizes maintain Excel proportions."""
        assert CompressionLevel.NONE.col_block == 1
        assert CompressionLevel.MILD.col_block == 1
        assert CompressionLevel.EXCEL_RATIO.col_block == 1
        assert CompressionLevel.LARGE.col_block == 4
        assert CompressionLevel.HUGE.col_block == 16
        assert CompressionLevel.MAXIMUM.col_block == 64

    def test_excel_ratio_maintained(self):
        """Test that 64:1 ratio is maintained for larger compressions."""
        # Check ratio for levels that should maintain 64:1
        assert CompressionLevel.LARGE.row_block / CompressionLevel.LARGE.col_block == 64
        assert CompressionLevel.HUGE.row_block / CompressionLevel.HUGE.col_block == 64
        assert CompressionLevel.MAXIMUM.row_block / CompressionLevel.MAXIMUM.col_block == 64

    def test_descriptions(self):
        """Test compression level descriptions."""
        assert "No compression" in CompressionLevel.NONE.description
        assert "16:1" in CompressionLevel.MILD.description
        assert "64:1" in CompressionLevel.EXCEL_RATIO.description
        assert "Excel-proportioned" in CompressionLevel.EXCEL_RATIO.description

    def test_max_cells(self):
        """Test max cells recommendations."""
        assert CompressionLevel.NONE.max_cells == 100_000
        assert CompressionLevel.MILD.max_cells == 1_600_000
        assert CompressionLevel.EXCEL_RATIO.max_cells == 6_400_000
        assert CompressionLevel.LARGE.max_cells == 100_000_000
        assert CompressionLevel.HUGE.max_cells == 1_600_000_000
        assert CompressionLevel.MAXIMUM.max_cells == 16_000_000_000


class TestDataRegion:
    """Test DataRegion model."""

    def test_data_region_creation(self):
        """Test creating a data region."""
        region = DataRegion(
            region_id="region_1",
            bounds={"top": 0, "left": 0, "bottom": 10, "right": 5},
            cell_count=30,
            density=0.6,
            characteristics={"likely_headers": True},
            skip=False,
        )

        assert region.region_id == "region_1"
        assert region.rows == 11  # 0 to 10 inclusive
        assert region.cols == 6  # 0 to 5 inclusive
        assert region.total_cells == 66
        assert region.density == 0.6
        assert region.characteristics["likely_headers"] is True

    def test_skip_region(self):
        """Test skip functionality."""
        region = DataRegion(
            region_id="skip_me",
            bounds={"top": 0, "left": 0, "bottom": 0, "right": 0},
            cell_count=0,
            density=0.0,
            skip=True,
            skip_reason="Empty region",
        )

        assert region.skip is True
        assert region.skip_reason == "Empty region"


class TestVisionImage:
    """Test VisionImage model."""

    def test_vision_image_creation(self):
        """Test creating a vision image."""
        image = VisionImage(
            image_id="test_image",
            image_data="base64encodeddata",
            compression_level=2,
            block_size=[64, 1],
            description="Test image",
            purpose="Testing",
            covers_cells="A1:Z100",
            size_bytes=1024,
        )

        assert image.image_id == "test_image"
        assert image.compression_level == 2
        assert image.block_size == [64, 1]
        assert image.size_mb == pytest.approx(0.0009765625, rel=1e-6)

    def test_vision_image_validation(self):
        """Test vision image validation."""
        with pytest.raises(ValidationError) as excinfo:
            VisionImage(
                image_id="bad",
                image_data="data",
                compression_level=10,  # Invalid: > 5
                block_size=[1, 1],
                description="desc",
                purpose="purpose",
                covers_cells="A1:B2",
                size_bytes=100,
            )
        assert "less than or equal to 5" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo:
            VisionImage(
                image_id="bad",
                image_data="data",
                compression_level=2,
                block_size=[1, 1],
                description="desc",
                purpose="purpose",
                covers_cells="A1:B2",
                size_bytes=0,  # Invalid: must be > 0
            )
        assert "greater than 0" in str(excinfo.value)


class TestMultiScaleBitmaps:
    """Test MultiScaleBitmaps collection."""

    def test_multi_scale_creation(self):
        """Test creating multi-scale bitmap collection."""
        bitmaps = MultiScaleBitmaps(
            sheet_name="TestSheet",
            sheet_dimensions={"rows": 1000, "cols": 50},
            data_bounds={"top": 0, "left": 0, "bottom": 100, "right": 20},
            compression_strategy="multi_scale",
            total_size_mb=2.5,
            generation_time_ms=150,
        )

        assert bitmaps.sheet_name == "TestSheet"
        assert bitmaps.compression_strategy == "multi_scale"
        assert bitmaps.total_size_mb == 2.5

    def test_add_image(self):
        """Test adding images to collection."""
        bitmaps = MultiScaleBitmaps(
            sheet_name="TestSheet",
            sheet_dimensions={"rows": 100, "cols": 10},
            data_bounds={"top": 0, "left": 0, "bottom": 50, "right": 5},
            compression_strategy="single_image",
            total_size_mb=0.0,
            generation_time_ms=10,
        )

        # Add overview image (compressed)
        overview = VisionImage(
            image_id="overview",
            image_data="data1",
            compression_level=3,
            block_size=[256, 4],
            description="Overview",
            purpose="Overview",
            covers_cells="A1:J100",
            size_bytes=1024 * 512,  # 0.5 MB
        )
        bitmaps.add_image(overview)
        assert bitmaps.overview == overview
        assert len(bitmaps.detail_views) == 0

        # Add detail image (uncompressed)
        detail = VisionImage(
            image_id="detail1",
            image_data="data2",
            compression_level=0,
            block_size=[1, 1],
            description="Detail",
            purpose="Detail",
            covers_cells="A1:E50",
            size_bytes=1024 * 256,  # 0.25 MB
        )
        bitmaps.add_image(detail)
        assert len(bitmaps.detail_views) == 1
        assert bitmaps.detail_views[0] == detail

        # Check total size calculation
        assert bitmaps.total_size_mb == pytest.approx(0.75, rel=1e-3)


class TestVisionRequest:
    """Test VisionRequest model."""

    def test_vision_request_creation(self):
        """Test creating vision request."""
        image1 = VisionImage(
            image_id="img1",
            image_data="data1",
            compression_level=0,
            block_size=[1, 1],
            description="Image 1",
            purpose="Test",
            covers_cells="A1:B2",
            size_bytes=1024 * 1024,  # 1 MB
        )

        image2 = VisionImage(
            image_id="img2",
            image_data="data2",
            compression_level=2,
            block_size=[64, 1],
            description="Image 2",
            purpose="Test",
            covers_cells="C1:D2",
            size_bytes=1024 * 512,  # 0.5 MB
        )

        request = VisionRequest(
            images=[image1, image2],
            prompt_template="EXPLICIT_MULTI_SCALE",
            total_images=2,
            total_size_mb=1.5,
        )

        assert len(request.images) == 2
        assert request.prompt_template == "EXPLICIT_MULTI_SCALE"
        assert request.total_images == 2
        assert request.total_size_mb == 1.5

    def test_size_limit_validation(self):
        """Test request size limit validation."""
        small_image = VisionImage(
            image_id="small",
            image_data="data",
            compression_level=0,
            block_size=[1, 1],
            description="Small",
            purpose="Test",
            covers_cells="A1:B2",
            size_bytes=1024 * 1024 * 5,  # 5 MB
        )

        request = VisionRequest(
            images=[small_image],
            prompt_template="SINGLE_IMAGE",
            total_images=1,
            total_size_mb=5.0,
        )

        # Should pass 20MB limit
        assert request.validate_size_limit(20.0) is True

        # Should fail 4MB limit
        assert request.validate_size_limit(4.0) is False


class TestProgressiveRefinementPhase:
    """Test ProgressiveRefinementPhase model."""

    def test_phase_creation(self):
        """Test creating refinement phase."""
        phase = ProgressiveRefinementPhase(
            phase="overview",
            strategy="maximum_compression",
            compression_level=5,
            focus_regions=[
                {"top": 0, "left": 0, "bottom": 100, "right": 50},
                {"top": 200, "left": 0, "bottom": 300, "right": 50},
            ],
            purpose="Initial overview scan",
        )

        assert phase.phase == "overview"
        assert phase.compression_level == 5
        assert len(phase.focus_regions) == 2

    def test_phase_types(self):
        """Test different phase types."""
        phases = ["overview", "refinement", "verification"]

        for phase_type in phases:
            phase = ProgressiveRefinementPhase(
                phase=phase_type,
                strategy=f"{phase_type}_strategy",
                compression_level=2,
                purpose=f"Testing {phase_type}",
            )
            assert phase.phase == phase_type


class TestExactBounds:
    """Test ExactBounds model."""

    def test_exact_bounds_creation(self):
        """Test creating exact bounds."""
        bounds = ExactBounds(
            top_row=5,
            left_col=2,
            bottom_row=15,
            right_col=8,
        )

        assert bounds.top_row == 5
        assert bounds.left_col == 2
        assert bounds.bottom_row == 15
        assert bounds.right_col == 8
        assert bounds.total_cells == 11 * 7  # (15-5+1) * (8-2+1)

    def test_excel_range_conversion(self):
        """Test conversion to Excel range."""
        bounds = ExactBounds(
            top_row=0,
            left_col=0,
            bottom_row=9,
            right_col=3,
        )

        assert bounds.excel_range == "A1:D10"

    def test_exact_bounds_validation(self):
        """Test bounds validation."""
        with pytest.raises(ValidationError) as excinfo:
            ExactBounds(
                top_row=-1,  # Invalid: must be >= 0
                left_col=0,
                bottom_row=10,
                right_col=5,
            )
        assert "greater than or equal to 0" in str(excinfo.value)
