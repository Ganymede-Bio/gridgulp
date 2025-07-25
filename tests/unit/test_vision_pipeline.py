"""Unit tests for VisionPipeline class."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gridporter.config import Config
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.models.vision_result import VisionAnalysisResult, VisionRegion
from gridporter.vision.pipeline import VisionPipeline, _convert_proposals_to_regions
from gridporter.vision.region_proposer import RegionProposal
from gridporter.vision.vision_models import VisionModelError, VisionModelResponse


class TestVisionPipeline:
    """Test VisionPipeline functionality."""

    def test_init(self):
        """Test VisionPipeline initialization."""
        config = Config(
            vision_cell_width=15,
            vision_cell_height=12,
            vision_mode="grayscale",
            confidence_threshold=0.8,
            enable_cache=True,
        )

        pipeline = VisionPipeline(config)

        assert pipeline.config == config
        assert pipeline.bitmap_generator.cell_width == 15
        assert pipeline.bitmap_generator.cell_height == 12
        assert pipeline.bitmap_generator.mode == "grayscale"
        assert pipeline._vision_model is None
        assert pipeline._cache == {}

    def test_init_default_config(self):
        """Test VisionPipeline with default config values."""
        config = Config()
        pipeline = VisionPipeline(config)

        assert pipeline.bitmap_generator.cell_width == 10
        assert pipeline.bitmap_generator.cell_height == 10
        assert pipeline.bitmap_generator.mode == "binary"

    @pytest.mark.asyncio
    async def test_analyze_sheet_basic(self):
        """Test basic analyze_sheet functionality."""
        config = Config(confidence_threshold=0.6, min_table_size=(2, 2))
        pipeline = VisionPipeline(config)

        # Create test sheet data
        sheet_data = SheetData(name="Test")
        sheet_data.cells["A1"] = CellData(
            value="Header", data_type="text", is_bold=True, row=0, column=0
        )
        sheet_data.cells["A2"] = CellData(value="Data", data_type="text", row=1, column=0)
        sheet_data.max_row = 1
        sheet_data.max_column = 0

        # Mock vision model
        mock_model = AsyncMock()
        mock_response = VisionModelResponse(
            content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, "confidence": 0.9}]}',
            model="test-model",
            usage={"total_tokens": 100},
        )
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            result = await pipeline.analyze_sheet(sheet_data)

        assert isinstance(result, VisionAnalysisResult)
        assert len(result.regions) == 1
        assert result.regions[0].confidence == 0.9
        assert result.cached is False
        assert mock_model.analyze_image.called

    @pytest.mark.asyncio
    async def test_analyze_sheet_with_cache_hit(self):
        """Test analyze_sheet with cache hit."""
        config = Config(enable_cache=True)
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Test")

        # Pre-populate cache
        cached_result = VisionAnalysisResult(
            regions=[
                VisionRegion(
                    pixel_bounds={"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                    cell_bounds={"start_row": 0, "start_col": 0, "end_row": 0, "end_col": 0},
                    range="A1:A1",
                    confidence=0.8,
                )
            ],
            bitmap_info={"width": 10, "height": 10},
        )
        cache_key = pipeline._generate_cache_key(sheet_data)
        pipeline._cache[cache_key] = cached_result

        result = await pipeline.analyze_sheet(sheet_data)

        assert result.cached is True
        assert len(result.regions) == 1
        assert result.regions[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_analyze_sheet_with_cache_disabled(self):
        """Test analyze_sheet with cache disabled."""
        config = Config(enable_cache=False)
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Test")

        mock_model = AsyncMock()
        mock_response = VisionModelResponse(content='{"tables": []}', model="test-model")
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            result = await pipeline.analyze_sheet(sheet_data)

        assert result.cached is False
        assert len(pipeline._cache) == 0  # No caching occurred

    @pytest.mark.asyncio
    async def test_analyze_sheet_filters_low_confidence(self):
        """Test that low confidence proposals are filtered out."""
        config = Config(confidence_threshold=0.8, min_table_size=(2, 2))
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Test")

        mock_model = AsyncMock()
        mock_response = VisionModelResponse(
            content='{"tables": [{"bounds": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, "confidence": 0.5}]}',
            model="test-model",
        )
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            result = await pipeline.analyze_sheet(sheet_data)

        # Should be filtered out due to low confidence
        assert len(result.regions) == 0

    @pytest.mark.asyncio
    async def test_analyze_sheet_vision_model_error(self):
        """Test analyze_sheet handling vision model errors."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Test")

        mock_model = AsyncMock()
        mock_model.analyze_image.side_effect = VisionModelError("API error")

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            with pytest.raises(VisionModelError, match="API error"):
                await pipeline.analyze_sheet(sheet_data)

    @pytest.mark.asyncio
    async def test_batch_analyze_sequential(self):
        """Test batch_analyze with sequential processing."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheets = [SheetData(name="Sheet1"), SheetData(name="Sheet2")]

        mock_model = AsyncMock()
        mock_model.supports_batch = False
        mock_response = VisionModelResponse(content='{"tables": []}', model="test-model")
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            results = await pipeline.batch_analyze(sheets)

        assert len(results) == 2
        assert all(isinstance(r, VisionAnalysisResult) for r in results)
        assert mock_model.analyze_image.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_analyze_with_batch_support(self):
        """Test batch_analyze with batch-supporting model."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheets = [SheetData(name="Sheet1"), SheetData(name="Sheet2")]

        mock_model = AsyncMock()
        mock_model.supports_batch = True
        mock_response = VisionModelResponse(content='{"tables": []}', model="test-model")
        mock_model.analyze_image.return_value = mock_response

        with patch("gridporter.vision.pipeline.create_vision_model", return_value=mock_model):
            results = await pipeline.batch_analyze(sheets)

        # Currently processes sequentially even with batch support
        assert len(results) == 2
        assert mock_model.analyze_image.call_count == 2

    def test_generate_cache_key(self):
        """Test cache key generation."""
        pipeline = VisionPipeline(Config())

        sheet_data = SheetData(name="Test")
        sheet_data.cells["A1"] = CellData(value="Hello", data_type="text", row=0, column=0)
        sheet_data.max_row = 0
        sheet_data.max_column = 0

        key1 = pipeline._generate_cache_key(sheet_data)
        key2 = pipeline._generate_cache_key(sheet_data)

        # Same sheet should produce same key
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

        # Different sheet should produce different key
        sheet_data.cells["A1"].value = "World"
        key3 = pipeline._generate_cache_key(sheet_data)
        assert key1 != key3

    def test_save_debug_bitmap(self):
        """Test saving debug bitmap."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Debug")
        sheet_data.cells["A1"] = CellData(value="Test", data_type="text", row=0, column=0)
        sheet_data.max_row = 0
        sheet_data.max_column = 0

        with patch("builtins.open", create=True) as mock_open:
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                output_path = Path("/tmp/debug.png")
                metadata = pipeline.save_debug_bitmap(sheet_data, output_path)

        assert metadata.total_rows == 1
        assert metadata.total_cols == 1
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(output_path, "wb")

    def test_clear_cache(self):
        """Test cache clearing."""
        config = Config()
        pipeline = VisionPipeline(config)

        # Add some cache entries
        pipeline._cache["key1"] = VisionAnalysisResult(regions=[], bitmap_info={})
        pipeline._cache["key2"] = VisionAnalysisResult(regions=[], bitmap_info={})

        assert len(pipeline._cache) == 2

        pipeline.clear_cache()

        assert len(pipeline._cache) == 0

    def test_get_cache_stats(self):
        """Test cache statistics."""
        config = Config(enable_cache=True)
        pipeline = VisionPipeline(config)

        # Empty cache
        stats = pipeline.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_enabled"] is True

        # Add cache entries
        pipeline._cache["key1"] = VisionAnalysisResult(regions=[], bitmap_info={})
        pipeline._cache["key2"] = VisionAnalysisResult(regions=[], bitmap_info={})

        stats = pipeline.get_cache_stats()
        assert stats["cache_size"] == 2

    def test_get_cache_stats_disabled(self):
        """Test cache statistics with cache disabled."""
        config = Config(enable_cache=False)
        pipeline = VisionPipeline(config)

        stats = pipeline.get_cache_stats()
        assert stats["cache_enabled"] is False

    def test_create_analysis_prompt(self):
        """Test analysis prompt creation."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="TestSheet")
        base_prompt = "This is a test prompt"

        prompt = pipeline._create_analysis_prompt(base_prompt, sheet_data)

        assert "This is a test prompt" in prompt
        assert "TestSheet" in prompt
        assert "JSON format" in prompt
        assert "confidence score" in prompt
        assert "pixel coordinates" in prompt

    def test_create_analysis_prompt_default_sheet_name(self):
        """Test analysis prompt with default sheet name."""
        config = Config()
        pipeline = VisionPipeline(config)

        sheet_data = SheetData(name="Sheet1")  # Default name
        base_prompt = "Test prompt"

        prompt = pipeline._create_analysis_prompt(base_prompt, sheet_data)

        # Should not mention the default sheet name
        assert "Sheet1" not in prompt or "sheet 'Sheet1'" not in prompt


class TestConvertProposalsToRegions:
    """Test the conversion utility function."""

    def test_convert_empty_list(self):
        """Test converting empty proposal list."""
        result = _convert_proposals_to_regions([])
        assert result == []

    def test_convert_single_proposal(self):
        """Test converting single proposal."""
        proposal = RegionProposal(
            x1=10,
            y1=20,
            x2=30,
            y2=40,
            start_row=2,
            start_col=1,
            end_row=3,
            end_col=2,
            excel_range="B3:C4",
            confidence=0.85,
            characteristics={"has_headers": True},
        )

        regions = _convert_proposals_to_regions([proposal])

        assert len(regions) == 1
        region = regions[0]
        assert isinstance(region, VisionRegion)
        assert region.pixel_bounds == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
        assert region.cell_bounds == {"start_row": 2, "start_col": 1, "end_row": 3, "end_col": 2}
        assert region.range == "B3:C4"
        assert region.confidence == 0.85
        assert region.detection_method == "vision"
        assert region.characteristics == {"has_headers": True}

    def test_convert_multiple_proposals(self):
        """Test converting multiple proposals."""
        proposals = [
            RegionProposal(
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                start_row=0,
                start_col=0,
                end_row=0,
                end_col=0,
                excel_range="A1:A1",
                confidence=0.9,
                characteristics={},
            ),
            RegionProposal(
                x1=20,
                y1=20,
                x2=40,
                y2=40,
                start_row=2,
                start_col=2,
                end_row=3,
                end_col=3,
                excel_range="C3:D4",
                confidence=0.7,
                characteristics={"type": "data"},
            ),
        ]

        regions = _convert_proposals_to_regions(proposals)

        assert len(regions) == 2
        assert all(isinstance(r, VisionRegion) for r in regions)
        assert regions[0].confidence == 0.9
        assert regions[1].confidence == 0.7
        assert regions[1].characteristics == {"type": "data"}
