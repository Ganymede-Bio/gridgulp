"""Tests for VisionOrchestratorAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gridporter.agents.vision_orchestrator_agent import (
    VisionOrchestratorAgent,
    ComplexityAssessment,
    OrchestratorDecision,
    VisionOrchestratorResult,
)
from gridporter.config import Config
from gridporter.models.sheet_data import SheetData, CellData
from gridporter.models.table import TableInfo, TableRange
from gridporter.models.vision_result import VisionResult, VisionRegion
from gridporter.vision.vision_request_builder import VisionRequestBuilder


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        use_vision=True,
        openai_api_key="test-key",
        confidence_threshold=0.7,
        max_cost_per_session=1.0,
        max_cost_per_file=0.1,
    )


@pytest.fixture
def simple_sheet_data():
    """Create simple sheet data for testing."""
    # Create a simple 5x3 table
    cells = {}
    for i in range(5):
        for j in range(3):
            if i == 0:
                # Header row
                value = f"Header{j+1}"
                data_type = "string"
            else:
                # Data rows
                value = f"Data{i}-{j+1}" if j == 0 else i * j
                data_type = "string" if j == 0 else "number"

            cell = CellData(
                value=value,
                data_type=data_type,
                is_bold=(i == 0),  # Headers are bold
            )

            # Convert row,col to Excel address
            col_letter = chr(ord("A") + j)
            address = f"{col_letter}{i + 1}"
            cells[address] = cell

    return SheetData(
        name="TestSheet",
        cells=cells,
        max_row=4,
        max_column=2,
    )


@pytest.fixture
def complex_sheet_data():
    """Create complex sheet data for testing."""
    # Create a more complex sheet with multiple regions and merged cells
    cells = {}
    for i in range(20):
        for j in range(10):
            # Create gaps to simulate multiple tables
            if 5 <= i <= 7 and j < 5:
                # Empty region - skip creating cells
                continue
            elif i < 10 and j < 5:
                # First table
                value = f"T1-{i}-{j}" if i > 0 else f"Header{j}"
            elif i >= 12 and j >= 6:
                # Second table
                value = f"T2-{i}-{j}" if i > 12 else f"Col{j}"
            else:
                # Empty region - skip creating cells
                continue

            if value:  # Only create cells with values
                cell = CellData(
                    value=value,
                    data_type="string",
                    is_bold=(i == 0 or i == 12),  # Headers are bold
                    is_merged=(i == 0 and j < 2),  # Some merged cells in headers
                )

                # Convert row,col to Excel address
                col_letter = chr(ord("A") + j)
                address = f"{col_letter}{i + 1}"
                cells[address] = cell

    return SheetData(
        name="ComplexSheet",
        cells=cells,
        max_row=19,
        max_column=9,
    )


class TestVisionOrchestratorAgent:
    """Test cases for VisionOrchestratorAgent."""

    def test_init_with_vision(self, config):
        """Test initialization with vision enabled."""
        with patch(
            "gridporter.agents.vision_orchestrator_agent.create_vision_model"
        ) as mock_create:
            mock_vision_model = MagicMock()
            mock_vision_model.name = "gpt-4o"
            mock_create.return_value = mock_vision_model

            agent = VisionOrchestratorAgent(config)

            assert agent.config == config
            assert agent.vision_model == mock_vision_model
            assert agent.complex_table_agent is not None
            assert agent.cost_optimizer is not None

    def test_init_without_vision(self):
        """Test initialization without vision."""
        config = Config(use_vision=False)
        agent = VisionOrchestratorAgent(config)

        assert agent.vision_model is None
        assert agent.complex_table_agent is not None

    @pytest.mark.asyncio
    async def test_assess_complexity_simple_sheet(self, config, simple_sheet_data):
        """Test complexity assessment for simple sheet."""
        agent = VisionOrchestratorAgent(config)

        with patch.object(agent.sparse_pattern_detector, "detect_patterns") as mock_detect:
            mock_detect.return_value = []  # No patterns found

            assessment = await agent._assess_complexity(simple_sheet_data)

            assert isinstance(assessment, ComplexityAssessment)
            assert assessment.complexity_score <= 0.6  # Should be low complexity
            assert not assessment.requires_vision  # Simple sheet doesn't need vision
            assert assessment.confidence > 0.0
            assert "Simple" in assessment.reasoning

    @pytest.mark.asyncio
    async def test_assess_complexity_complex_sheet(self, config, complex_sheet_data):
        """Test complexity assessment for complex sheet."""
        agent = VisionOrchestratorAgent(config)

        # Mock pattern detection to return multiple patterns
        mock_patterns = [MagicMock(), MagicMock(), MagicMock()]

        with patch.object(agent.sparse_pattern_detector, "detect_patterns") as mock_detect:
            mock_detect.return_value = mock_patterns

            assessment = await agent._assess_complexity(complex_sheet_data)

            assert isinstance(assessment, ComplexityAssessment)
            assert assessment.complexity_score > 0.3  # Should be higher complexity
            assert assessment.requires_vision  # Complex sheet needs vision
            assert "multiple data regions" in assessment.reasoning.lower()

    @pytest.mark.asyncio
    async def test_make_orchestration_decision_simple(self, config, simple_sheet_data):
        """Test orchestration decision for simple sheet."""
        agent = VisionOrchestratorAgent(config)

        # Create low complexity assessment
        complexity = ComplexityAssessment(
            complexity_score=0.3,
            requires_vision=False,
            assessment_factors={"sparsity": 0.1},
            confidence=0.8,
            reasoning="Simple sheet",
        )

        decision = await agent._make_orchestration_decision(
            simple_sheet_data, complexity, excel_metadata=None
        )

        assert isinstance(decision, OrchestratorDecision)
        assert decision.detection_strategy in ["hybrid_traditional", "hybrid_excel_metadata"]
        assert not decision.use_vision
        assert decision.cost_estimate == 0.0

    @pytest.mark.asyncio
    async def test_make_orchestration_decision_complex(self, config, complex_sheet_data):
        """Test orchestration decision for complex sheet."""
        agent = VisionOrchestratorAgent(config)

        # Create high complexity assessment
        complexity = ComplexityAssessment(
            complexity_score=0.8,
            requires_vision=True,
            assessment_factors={"sparsity": 0.8, "pattern_complexity": 0.9},
            confidence=0.9,
            reasoning="Complex sheet with multiple regions",
        )

        decision = await agent._make_orchestration_decision(
            complex_sheet_data, complexity, excel_metadata=None
        )

        assert isinstance(decision, OrchestratorDecision)
        assert decision.use_vision
        assert decision.cost_estimate > 0.0
        assert len(decision.fallback_strategies) > 0

    @pytest.mark.asyncio
    async def test_execute_detection_strategy_success(self, config, simple_sheet_data):
        """Test successful strategy execution."""
        agent = VisionOrchestratorAgent(config)

        # Mock complex table agent to return results
        mock_table = TableInfo(
            id="test_table_1",
            range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
            confidence=0.9,
            detection_method="test",
        )

        with patch.object(agent.complex_table_agent, "detect_complex_tables") as mock_detect:
            mock_result = MagicMock()
            mock_result.tables = [mock_table]
            mock_detect.return_value = mock_result

            decision = OrchestratorDecision(
                detection_strategy="hybrid_traditional",
                use_vision=False,
                fallback_strategies=["traditional_fallback"],
                cost_estimate=0.0,
                confidence=0.8,
            )

            complexity = ComplexityAssessment(
                complexity_score=0.3,
                requires_vision=False,
                assessment_factors={},
                confidence=0.8,
                reasoning="Simple",
            )

            tables = await agent._execute_detection_strategy(
                simple_sheet_data, decision, complexity, excel_metadata=None
            )

            assert len(tables) == 1
            assert tables[0] == mock_table

    @pytest.mark.asyncio
    async def test_execute_detection_strategy_with_fallback(self, config, simple_sheet_data):
        """Test strategy execution with fallback."""
        agent = VisionOrchestratorAgent(config)

        # Mock first strategy to fail, second to succeed
        mock_table = TableInfo(
            id="test_table_2",
            range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
            confidence=0.8,
            detection_method="fallback",
        )

        call_count = 0

        def mock_strategy_side_effect(strategy, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if strategy == "full_vision":
                # First strategy fails
                raise Exception("Primary strategy failed")
            else:
                # Fallback succeeds
                return [mock_table]

        with patch.object(agent, "_execute_single_strategy", side_effect=mock_strategy_side_effect):
            decision = OrchestratorDecision(
                detection_strategy="full_vision",
                use_vision=True,
                fallback_strategies=["hybrid_traditional"],
                cost_estimate=0.05,
                confidence=0.9,
            )

            complexity = ComplexityAssessment(
                complexity_score=0.8,
                requires_vision=True,
                assessment_factors={},
                confidence=0.9,
                reasoning="Complex",
            )

            tables = await agent._execute_detection_strategy(
                simple_sheet_data, decision, complexity, excel_metadata=None
            )

            assert len(tables) == 1
            assert tables[0] == mock_table
            assert call_count == 2  # Tried primary + fallback

    @pytest.mark.asyncio
    async def test_orchestrate_detection_full_pipeline(self, config, simple_sheet_data):
        """Test the full orchestration pipeline."""
        with patch(
            "gridporter.agents.vision_orchestrator_agent.create_vision_model"
        ) as mock_create:
            mock_vision_model = MagicMock()
            mock_vision_model.name = "gpt-4o"
            mock_create.return_value = mock_vision_model

            agent = VisionOrchestratorAgent(config)

            # Mock complex table agent
            mock_table = TableInfo(
                id="test_table_3",
                range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
                confidence=0.85,
                detection_method="hybrid_traditional",
            )

            with patch.object(agent.complex_table_agent, "detect_complex_tables") as mock_detect:
                mock_result = MagicMock()
                mock_result.tables = [mock_table]
                mock_detect.return_value = mock_result

                with patch.object(
                    agent.sparse_pattern_detector, "detect_patterns"
                ) as mock_patterns:
                    mock_patterns.return_value = []  # Simple case

                    result = await agent.orchestrate_detection(simple_sheet_data)

                    assert isinstance(result, VisionOrchestratorResult)
                    assert len(result.tables) == 1
                    assert result.tables[0] == mock_table
                    assert result.complexity_assessment.complexity_score <= 0.6
                    assert not result.orchestrator_decision.use_vision
                    assert "processing_time_seconds" in result.processing_metadata

    @pytest.mark.asyncio
    async def test_orchestrate_detection_error_handling(self, config, simple_sheet_data):
        """Test error handling in orchestration."""
        agent = VisionOrchestratorAgent(config)

        # Mock complexity assessment to fail
        with patch.object(agent, "_assess_complexity", side_effect=Exception("Assessment failed")):
            result = await agent.orchestrate_detection(simple_sheet_data)

            assert isinstance(result, VisionOrchestratorResult)
            assert len(result.tables) == 0
            assert result.complexity_assessment.complexity_score == 0.0
            assert result.orchestrator_decision.detection_strategy == "error_fallback"
            assert "error" in result.processing_metadata

    def test_estimate_vision_cost(self, config, simple_sheet_data, complex_sheet_data):
        """Test vision cost estimation with new request builder."""
        agent = VisionOrchestratorAgent(config)

        # Mock vision request builder response
        with patch.object(agent.vision_request_builder, "build_request") as mock_build:
            # Test small sheet - no images generated (empty)
            mock_request = MagicMock()
            mock_request.total_size_mb = 0
            mock_request.images = []
            mock_build.return_value = mock_request

            cost = agent._estimate_vision_cost(simple_sheet_data)
            assert cost == 0.01  # Falls back to SMALL_SHEET_COST

            # Test with actual image data
            mock_request.total_size_mb = 0.5
            mock_request.images = [MagicMock()]
            mock_build.return_value = mock_request

            cost = agent._estimate_vision_cost(simple_sheet_data)
            assert cost == 0.005  # 0.5 MB * $0.01/MB

            # Test with multiple images (20% overhead)
            mock_request.total_size_mb = 2.0
            mock_request.images = [MagicMock(), MagicMock(), MagicMock()]
            mock_build.return_value = mock_request

            cost = agent._estimate_vision_cost(complex_sheet_data)
            assert cost == 0.024  # 2.0 MB * $0.01/MB * 1.2

    def test_validate_results(self, config):
        """Test result validation."""
        agent = VisionOrchestratorAgent(config)

        # Valid results
        valid_table = TableInfo(
            id="valid_table_1",
            range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
            confidence=0.8,
            detection_method="test",
        )
        assert agent._validate_results([valid_table])

        # Empty results
        assert not agent._validate_results([])

        # Low confidence
        low_confidence_table = TableInfo(
            id="low_conf_table",
            range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
            confidence=0.5,  # Below threshold
            detection_method="test",
        )
        assert not agent._validate_results([low_confidence_table])

        # Invalid size
        small_table = TableInfo(
            id="small_table",
            range=TableRange(start_row=0, start_col=0, end_row=0, end_col=0),  # Too small
            confidence=0.9,
            detection_method="test",
        )
        assert not agent._validate_results([small_table])

    @pytest.mark.asyncio
    async def test_get_model_status(self, config):
        """Test model status reporting."""
        with patch(
            "gridporter.agents.vision_orchestrator_agent.create_vision_model"
        ) as mock_create:
            mock_vision_model = MagicMock()
            mock_vision_model.name = "gpt-4o"
            mock_vision_model.check_model_available = AsyncMock(return_value=True)
            mock_create.return_value = mock_vision_model

            agent = VisionOrchestratorAgent(config)

            status = await agent.get_model_status()

            assert status["vision_model_available"]
            assert status["vision_model_name"] == "gpt-4o"
            assert status["vision_model_reachable"]
            assert "cost_limits" in status
            assert status["cost_optimizer_enabled"]

    @pytest.mark.asyncio
    async def test_get_model_status_no_vision(self):
        """Test model status without vision."""
        config = Config(use_vision=False)
        agent = VisionOrchestratorAgent(config)

        status = await agent.get_model_status()

        assert not status["vision_model_available"]
        assert status["vision_model_name"] is None
        assert status["cost_optimizer_enabled"]

    @pytest.mark.asyncio
    async def test_execute_full_vision_strategy(self, config, simple_sheet_data):
        """Test full vision strategy execution with multi-scale approach."""
        agent = VisionOrchestratorAgent(config)

        # Mock vision request builder
        mock_request = MagicMock()
        mock_request.images = [MagicMock(image_id="overview"), MagicMock(image_id="detail_1")]
        mock_request.prompt_template = "MULTI_SCALE"
        mock_request.total_images = 2

        with patch.object(agent.vision_request_builder, "build_request") as mock_build:
            mock_build.return_value = mock_request

            with patch.object(
                agent.vision_request_builder, "create_explicit_prompt"
            ) as mock_prompt:
                mock_prompt.return_value = "Analyze these multi-scale images"

                # Mock integrated pipeline
                mock_pipeline_result = MagicMock()
                mock_pipeline_result.detected_tables = []

                with patch.object(agent.integrated_pipeline, "process_sheet") as mock_pipeline:
                    mock_pipeline.return_value = mock_pipeline_result

                    tables = await agent._execute_full_vision_strategy(simple_sheet_data)

                    # Verify vision request was built
                    mock_build.assert_called_once_with(simple_sheet_data, simple_sheet_data.name)

                    # Verify prompt was created
                    mock_prompt.assert_called_once_with(mock_request)

                    # Should process with pipeline
                    mock_pipeline.assert_called_once_with(simple_sheet_data)

    @pytest.mark.asyncio
    async def test_execute_hybrid_vision_strategy(self, config, simple_sheet_data):
        """Test hybrid vision strategy execution."""
        agent = VisionOrchestratorAgent(config)

        # Mock complex table agent
        mock_result = MagicMock()
        mock_result.tables = [
            TableInfo(
                id="hybrid_table_1",
                range=TableRange(start_row=0, start_col=0, end_row=4, end_col=2),
                confidence=0.85,
                detection_method="hybrid_vision",
            )
        ]

        with patch.object(agent.complex_table_agent, "detect_complex_tables") as mock_detect:
            mock_detect.return_value = mock_result

            tables = await agent._execute_hybrid_vision_strategy(
                simple_sheet_data, excel_metadata=None
            )

            assert len(tables) == 1
            assert tables[0].detection_method == "hybrid_vision"

            # Verify complex table agent was called with correct params
            mock_detect.assert_called_once_with(simple_sheet_data, excel_metadata=None)

    @pytest.mark.asyncio
    async def test_multi_scale_vision_request_empty_sheet(self, config):
        """Test vision request building for empty sheet."""
        agent = VisionOrchestratorAgent(config)

        # Create empty sheet
        empty_sheet = SheetData(name="Empty", cells={}, max_row=0, max_column=0)

        # Mock vision request builder to return no images
        mock_request = MagicMock()
        mock_request.images = []
        mock_request.prompt_template = "SINGLE_IMAGE"

        with patch.object(agent.vision_request_builder, "build_request") as mock_build:
            mock_build.return_value = mock_request

            tables = await agent._execute_full_vision_strategy(empty_sheet)

            # Should return empty list for empty sheet
            assert tables == []

            # Verify request was attempted
            mock_build.assert_called_once_with(empty_sheet, empty_sheet.name)

    def test_get_methods_attempted(self, config):
        """Test tracking of attempted detection methods."""
        agent = VisionOrchestratorAgent(config)

        methods = agent._get_methods_attempted()

        assert isinstance(methods, list)
        assert "complexity_assessment" in methods
        assert "orchestration_decision" in methods
        assert "strategy_execution" in methods
