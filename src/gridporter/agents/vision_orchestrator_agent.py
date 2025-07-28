"""Vision orchestrator agent for central AI coordination of table detection."""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..core.constants import COMPLEXITY_ASSESSMENT, VISION_ORCHESTRATOR
from ..core.exceptions import GridPorterError
from ..models.sheet_data import SheetData
from ..models.table import TableInfo
from ..utils.cost_optimizer import CostOptimizer
from ..utils.logging_context import OperationContext, get_contextual_logger
from ..vision.bitmap_generator import BitmapGenerator
from ..vision.integrated_pipeline import IntegratedVisionPipeline
from ..vision.pattern_detector import SparsePatternDetector
from ..vision.region_verifier import RegionVerifier
from ..vision.vision_models import VisionModel, create_vision_model
from ..vision.vision_request_builder import VisionRequestBuilder
from .complex_table_agent import ComplexTableAgent

logger = get_contextual_logger(__name__)


class ComplexityAssessment(BaseModel):
    """Assessment of sheet complexity for routing decisions."""

    model_config = ConfigDict(strict=True)

    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Overall complexity score")
    requires_vision: bool = Field(..., description="Whether vision processing is recommended")
    assessment_factors: dict[str, float] = Field(
        default_factory=dict, description="Individual assessment factors"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    reasoning: str = Field(..., description="Human-readable reasoning for the assessment")


class OrchestratorDecision(BaseModel):
    """Decision made by the orchestrator for detection strategy."""

    model_config = ConfigDict(strict=True)

    detection_strategy: str = Field(..., description="Chosen detection strategy")
    use_vision: bool = Field(..., description="Whether to use vision models")
    vision_model: str | None = Field(None, description="Selected vision model")
    fallback_strategies: list[str] = Field(
        default_factory=list, description="Fallback strategies if primary fails"
    )
    cost_estimate: float = Field(..., ge=0.0, description="Estimated cost in USD")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")


class VisionOrchestratorResult(BaseModel):
    """Result from vision orchestrator coordination."""

    model_config = ConfigDict(strict=True)

    tables: list[TableInfo] = Field(..., description="Detected tables")
    complexity_assessment: ComplexityAssessment
    orchestrator_decision: OrchestratorDecision
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Processing metadata and timing"
    )
    cost_report: dict[str, Any] = Field(default_factory=dict, description="Detailed cost breakdown")


class VisionOrchestratorAgent:
    """Central AI orchestrator for intelligent table detection coordination.

    This agent serves as the brain of the GridPorter system, making intelligent
    decisions about which detection methods to use based on sheet complexity,
    cost constraints, and confidence requirements.
    """

    def __init__(self, config: Config):
        """Initialize the vision orchestrator agent.

        Args:
            config: GridPorter configuration
        """
        self.config = config

        # Initialize vision model if available
        self.vision_model: VisionModel | None = None
        if config.use_vision:
            try:
                self.vision_model = create_vision_model(config)
                logger.info(f"Initialized vision model: {self.vision_model.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize vision model: {e}")

        # Initialize sub-agents and tools
        self.complex_table_agent = ComplexTableAgent(config)
        self.bitmap_generator = BitmapGenerator()
        self.vision_request_builder = VisionRequestBuilder()
        # Pattern detector is handled by integrated pipeline
        self.sparse_pattern_detector = SparsePatternDetector()
        self.region_verifier = RegionVerifier()
        self.integrated_pipeline = IntegratedVisionPipeline(
            min_table_size=config.min_table_size,
            enable_verification=config.enable_region_verification,
            verification_strict=config.verification_strict_mode,
        )

        # Initialize cost optimizer
        self.cost_optimizer = CostOptimizer(
            max_cost_per_session=getattr(config, "max_cost_per_session", 1.0),
            max_cost_per_file=getattr(config, "max_cost_per_file", 0.1),
            confidence_threshold=config.confidence_threshold,
            enable_caching=getattr(config, "enable_cache", True),
        )

    async def orchestrate_detection(
        self,
        sheet_data: SheetData,
        excel_metadata: Any = None,
    ) -> VisionOrchestratorResult:
        """Orchestrate the entire table detection process.

        This is the main entry point that coordinates all detection methods,
        makes intelligent routing decisions, and manages fallback strategies.

        Args:
            sheet_data: Sheet data to analyze
            excel_metadata: Optional Excel metadata for optimization

        Returns:
            VisionOrchestratorResult with detected tables and metadata
        """
        start_time = time.time()
        logger.info("Starting vision orchestrator coordination")

        with OperationContext("orchestrator_coordination"):
            try:
                # Step 1: Assess sheet complexity
                complexity = await self._assess_complexity(sheet_data)
                logger.info(
                    f"Complexity assessment: {complexity.complexity_score:.2f} "
                    f"(requires_vision: {complexity.requires_vision})"
                )

                # Step 2: Make orchestration decision
                decision = await self._make_orchestration_decision(
                    sheet_data, complexity, excel_metadata
                )
                logger.info(
                    f"Orchestration decision: {decision.detection_strategy} "
                    f"(vision: {decision.use_vision}, cost: ${decision.cost_estimate:.4f})"
                )

                # Step 3: Execute detection strategy
                tables = await self._execute_detection_strategy(
                    sheet_data, decision, complexity, excel_metadata
                )

                # Step 4: Gather metadata
                processing_time = time.time() - start_time
                processing_metadata = {
                    "processing_time_seconds": processing_time,
                    "sheet_size": (sheet_data.max_row + 1, sheet_data.max_column + 1),
                    "total_cells": (sheet_data.max_row + 1) * (sheet_data.max_column + 1),
                    "vision_model_used": (self.vision_model.name if self.vision_model else None),
                    "methods_attempted": self._get_methods_attempted(),
                }

                cost_report = self.cost_optimizer.get_cost_report()

                logger.info(
                    f"Orchestration completed in {processing_time:.2f}s, "
                    f"detected {len(tables)} tables, cost: ${cost_report.get('total_cost_usd', 0):.4f}"
                )

                return VisionOrchestratorResult(
                    tables=tables,
                    complexity_assessment=complexity,
                    orchestrator_decision=decision,
                    processing_metadata=processing_metadata,
                    cost_report=cost_report,
                )

            except Exception as e:
                logger.error(f"Orchestration failed: {e}")
                # Return minimal result with error information
                return VisionOrchestratorResult(
                    tables=[],
                    complexity_assessment=ComplexityAssessment(
                        complexity_score=0.0,
                        requires_vision=False,
                        confidence=0.0,
                        reasoning=f"Assessment failed: {str(e)}",
                    ),
                    orchestrator_decision=OrchestratorDecision(
                        detection_strategy="error_fallback",
                        use_vision=False,
                        cost_estimate=0.0,
                        confidence=0.0,
                    ),
                    processing_metadata={"error": str(e)},
                    cost_report=self.cost_optimizer.get_cost_report(),
                )

    async def _assess_complexity(self, sheet_data: SheetData) -> ComplexityAssessment:
        """Assess the complexity of a sheet to guide detection strategy.

        Args:
            sheet_data: Sheet data to assess

        Returns:
            ComplexityAssessment with scoring and recommendations
        """
        with OperationContext("complexity_assessment"):
            factors = {}

            # Calculate basic metrics
            total_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
            filled_cells = len(sheet_data.get_non_empty_cells())
            sparsity = 1.0 - (filled_cells / max(total_cells, 1))

            factors["sparsity"] = sparsity
            factors["size_complexity"] = min(
                total_cells / COMPLEXITY_ASSESSMENT.LARGE_SHEET_THRESHOLD, 1.0
            )

            # Detect data islands for complexity assessment
            try:
                patterns = self.sparse_pattern_detector.detect_patterns(sheet_data)
                factors["pattern_complexity"] = min(
                    len(patterns) / COMPLEXITY_ASSESSMENT.MULTI_TABLE_THRESHOLD, 1.0
                )
                factors["has_multiple_regions"] = 1.0 if len(patterns) > 1 else 0.0
            except Exception as e:
                logger.debug(f"Pattern detection failed during complexity assessment: {e}")
                factors["pattern_complexity"] = 0.5  # Medium complexity if we can't assess
                factors["has_multiple_regions"] = 0.0

            # Check for merged cells and formatting complexity
            merged_cell_count = 0
            format_variety = set()

            non_empty_cells = sheet_data.get_non_empty_cells()
            for cell in non_empty_cells:
                if getattr(cell, "is_merged", False):
                    merged_cell_count += 1

                # Collect format indicators
                format_key = (
                    getattr(cell, "is_bold", False),
                    getattr(cell, "data_type", "unknown"),
                )
                format_variety.add(format_key)

            factors["merged_cell_ratio"] = merged_cell_count / max(filled_cells, 1)
            factors["format_complexity"] = min(
                len(format_variety) / COMPLEXITY_ASSESSMENT.FORMAT_VARIETY_THRESHOLD,
                1.0,
            )

            # Calculate overall complexity score
            weights = {
                "sparsity": 0.3,
                "size_complexity": 0.2,
                "pattern_complexity": 0.25,
                "merged_cell_ratio": 0.15,
                "format_complexity": 0.1,
            }

            complexity_score = sum(
                factors.get(factor, 0) * weight for factor, weight in weights.items()
            )

            # Determine if vision is recommended
            requires_vision = (
                complexity_score > COMPLEXITY_ASSESSMENT.VISION_THRESHOLD
                or factors.get("has_multiple_regions", 0.0) > 0.5
                or factors.get("merged_cell_ratio", 0) > COMPLEXITY_ASSESSMENT.MERGED_CELL_THRESHOLD
            )

            # Generate reasoning
            reasoning_parts = []
            if sparsity > 0.7:
                reasoning_parts.append("high sparsity detected")
            if factors.get("has_multiple_regions", 0.0) > 0.5:
                reasoning_parts.append("multiple data regions found")
            if factors.get("merged_cell_ratio", 0) > 0.1:
                reasoning_parts.append("significant merged cell usage")
            if total_cells > COMPLEXITY_ASSESSMENT.LARGE_SHEET_THRESHOLD:
                reasoning_parts.append("large sheet size")

            if not reasoning_parts:
                reasoning = "Simple, well-structured sheet suitable for traditional detection"
            else:
                reasoning = f"Complex sheet: {', '.join(reasoning_parts)}"

            # Confidence based on data availability
            confidence = min(filled_cells / max(total_cells * 0.1, 1), 1.0)

            return ComplexityAssessment(
                complexity_score=complexity_score,
                requires_vision=requires_vision,
                assessment_factors=factors,
                confidence=confidence,
                reasoning=reasoning,
            )

    async def _make_orchestration_decision(
        self,
        sheet_data: SheetData,
        complexity: ComplexityAssessment,
        excel_metadata: Any = None,
    ) -> OrchestratorDecision:
        """Make an intelligent decision about detection strategy.

        Args:
            sheet_data: Sheet data
            complexity: Complexity assessment
            excel_metadata: Optional Excel metadata

        Returns:
            OrchestratorDecision with strategy and parameters
        """
        with OperationContext("orchestration_decision"):
            # Check cost constraints
            current_cost = self.cost_optimizer.get_session_cost()
            cost_budget = self.cost_optimizer.max_cost_per_file

            # Estimate costs for different strategies
            vision_cost_estimate = self._estimate_vision_cost(sheet_data)
            hybrid_cost_estimate = vision_cost_estimate * 0.3  # Hybrid uses vision selectively

            # Decision matrix based on complexity and constraints
            if not complexity.requires_vision and excel_metadata:
                # Simple case with good metadata
                strategy = "hybrid_excel_metadata"
                use_vision = False
                cost_estimate = 0.0
                confidence = 0.9
            elif not complexity.requires_vision:
                # Simple case, use hybrid with traditional methods
                strategy = "hybrid_traditional"
                use_vision = False
                cost_estimate = 0.0
                confidence = 0.8
            elif current_cost + vision_cost_estimate > self.cost_optimizer.max_cost_per_session:
                # Over budget, use traditional methods
                strategy = "traditional_fallback"
                use_vision = False
                cost_estimate = 0.0
                confidence = 0.6
            elif vision_cost_estimate > cost_budget:
                # Too expensive for single file, use hybrid
                strategy = "hybrid_selective_vision"
                use_vision = True
                cost_estimate = hybrid_cost_estimate
                confidence = 0.75
            else:
                # Full vision processing
                strategy = "full_vision"
                use_vision = True
                cost_estimate = vision_cost_estimate
                confidence = 0.9

            # Select vision model if needed
            vision_model = None
            if use_vision and self.vision_model:
                vision_model = self.vision_model.name

            # Define fallback strategies
            fallback_strategies = []
            if strategy == "full_vision":
                fallback_strategies = [
                    "hybrid_selective_vision",
                    "traditional_fallback",
                ]
            elif strategy == "hybrid_selective_vision":
                fallback_strategies = ["traditional_fallback"]
            elif strategy in ["hybrid_excel_metadata", "hybrid_traditional"]:
                fallback_strategies = ["traditional_fallback"]

            return OrchestratorDecision(
                detection_strategy=strategy,
                use_vision=use_vision,
                vision_model=vision_model,
                fallback_strategies=fallback_strategies,
                cost_estimate=cost_estimate,
                confidence=confidence,
            )

    async def _execute_detection_strategy(
        self,
        sheet_data: SheetData,
        decision: OrchestratorDecision,
        complexity: ComplexityAssessment,
        excel_metadata: Any = None,
    ) -> list[TableInfo]:
        """Execute the chosen detection strategy with fallbacks.

        Args:
            sheet_data: Sheet data
            decision: Orchestration decision
            complexity: Complexity assessment
            excel_metadata: Optional Excel metadata

        Returns:
            List of detected tables
        """
        with OperationContext(f"execute_{decision.detection_strategy}"):
            strategies_to_try = [decision.detection_strategy] + decision.fallback_strategies

            for strategy in strategies_to_try:
                try:
                    logger.info(f"Attempting strategy: {strategy}")

                    tables = await self._execute_single_strategy(
                        strategy, sheet_data, decision, complexity, excel_metadata
                    )

                    if tables and self._validate_results(tables):
                        logger.info(f"Strategy {strategy} succeeded with {len(tables)} tables")
                        return tables
                    else:
                        logger.info(f"Strategy {strategy} produced no valid results")

                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed: {e}")
                    continue

            # If all strategies fail, return empty result
            logger.warning("All detection strategies failed")
            return []

    async def _execute_single_strategy(
        self,
        strategy: str,
        sheet_data: SheetData,
        decision: OrchestratorDecision,
        complexity: ComplexityAssessment,
        excel_metadata: Any = None,
    ) -> list[TableInfo]:
        """Execute a single detection strategy.

        Args:
            strategy: Strategy name
            sheet_data: Sheet data
            decision: Orchestration decision
            complexity: Complexity assessment
            excel_metadata: Optional Excel metadata

        Returns:
            List of detected tables
        """
        if strategy == "full_vision":
            return await self._execute_full_vision_strategy(sheet_data)
        elif strategy == "hybrid_selective_vision":
            return await self._execute_hybrid_vision_strategy(sheet_data, excel_metadata)
        elif strategy in ["hybrid_excel_metadata", "hybrid_traditional"]:
            return await self._execute_hybrid_strategy(sheet_data, excel_metadata)
        elif strategy == "traditional_fallback":
            return await self._execute_traditional_strategy(sheet_data, excel_metadata)
        else:
            raise GridPorterError(f"Unknown detection strategy: {strategy}")

    async def _execute_full_vision_strategy(self, sheet_data: SheetData) -> list[TableInfo]:
        """Execute full vision-based detection with multi-scale approach."""
        if not self.vision_model:
            raise GridPorterError("Vision model not available for full vision strategy")

        # Build multi-scale vision request
        vision_request = self.vision_request_builder.build_request(sheet_data, sheet_data.name)

        # If no images generated (empty sheet), return empty results
        if not vision_request.images:
            logger.info(f"No data found in sheet '{sheet_data.name}' for vision processing")
            return []

        # Create explicit prompt for the vision model
        prompt = self.vision_request_builder.create_explicit_prompt(vision_request)

        # Process with vision model using multi-scale images
        logger.info(
            f"Processing {len(vision_request.images)} images with {vision_request.prompt_template} strategy"
        )

        # For now, still use integrated pipeline but with enhanced context
        # TODO: Update integrated pipeline to handle multi-scale requests directly
        pipeline_result = self.integrated_pipeline.process_sheet(sheet_data)

        # Convert pipeline results to simple table ranges for complex table agent
        simple_tables = []
        if pipeline_result and pipeline_result.detected_tables:
            for pattern in pipeline_result.detected_tables:
                # Convert TablePattern to TableRange
                from ..models.table import TableRange

                table_range = TableRange(
                    start_row=pattern.bounds.min_row,
                    start_col=pattern.bounds.min_col,
                    end_row=pattern.bounds.max_row,
                    end_col=pattern.bounds.max_col,
                )
                simple_tables.append(table_range)

        # Use complex table agent to enhance the detected ranges
        if simple_tables:
            result = await self.complex_table_agent.detect_complex_tables(
                sheet_data, simple_tables=simple_tables
            )
            return result.tables

        return []

    async def _execute_hybrid_vision_strategy(
        self, sheet_data: SheetData, excel_metadata: Any = None
    ) -> list[TableInfo]:
        """Execute hybrid strategy with selective vision usage."""
        # Let the complex table agent handle hybrid detection
        result = await self.complex_table_agent.detect_complex_tables(
            sheet_data, excel_metadata=excel_metadata
        )
        return result.tables

    async def _execute_hybrid_strategy(
        self, sheet_data: SheetData, excel_metadata: Any = None
    ) -> list[TableInfo]:
        """Execute hybrid strategy without vision."""
        # Use complex table agent but disable vision
        result = await self.complex_table_agent.detect_complex_tables(
            sheet_data, vision_result=None, excel_metadata=excel_metadata
        )
        return result.tables

    async def _execute_traditional_strategy(
        self, sheet_data: SheetData, excel_metadata: Any = None
    ) -> list[TableInfo]:
        """Execute traditional algorithm-only strategy."""
        # Use only traditional detection methods
        result = await self.complex_table_agent.detect_complex_tables(
            sheet_data, vision_result=None, excel_metadata=excel_metadata
        )
        return result.tables

    def _estimate_vision_cost(self, sheet_data: SheetData) -> float:
        """Estimate the cost of vision processing for this sheet.

        Args:
            sheet_data: Sheet data

        Returns:
            Estimated cost in USD
        """
        # Build a vision request to get accurate cost estimate
        vision_request = self.vision_request_builder.build_request(sheet_data, sheet_data.name)

        # Estimate based on total image size and model pricing
        total_mb = vision_request.total_size_mb

        # Rough estimate: $0.01 per MB processed (adjust based on actual model pricing)
        base_cost = total_mb * 0.01

        # Add overhead for multiple images
        if len(vision_request.images) > 1:
            base_cost *= 1.2  # 20% overhead for multi-image processing

        # Return calculated cost or minimum threshold
        if base_cost > 0:
            return base_cost

        # Fallback to old logic if no images were generated
        total_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
        if total_cells < VISION_ORCHESTRATOR.SMALL_SHEET_THRESHOLD:
            return VISION_ORCHESTRATOR.SMALL_SHEET_COST
        elif total_cells < VISION_ORCHESTRATOR.MEDIUM_SHEET_THRESHOLD:
            return VISION_ORCHESTRATOR.MEDIUM_SHEET_COST
        else:
            return VISION_ORCHESTRATOR.LARGE_SHEET_COST

    def _validate_results(self, tables: list[TableInfo]) -> bool:
        """Validate that detection results meet quality criteria.

        Args:
            tables: Detected tables

        Returns:
            True if results are valid
        """
        if not tables:
            return False

        # Check minimum confidence
        avg_confidence = sum(t.confidence for t in tables) / len(tables)
        if avg_confidence < self.config.confidence_threshold:
            return False

        # Check for reasonable table sizes
        for table in tables:
            if table.range.row_count < 2 or table.range.col_count < 1:
                return False

        return True

    def _get_methods_attempted(self) -> list[str]:
        """Get list of detection methods that were attempted."""
        # This would be populated during execution
        return ["complexity_assessment", "orchestration_decision", "strategy_execution"]

    async def get_model_status(self) -> dict[str, Any]:
        """Get status of available models and capabilities.

        Returns:
            Status information
        """
        status = {
            "vision_model_available": self.vision_model is not None,
            "vision_model_name": self.vision_model.name if self.vision_model else None,
            "cost_optimizer_enabled": True,
            "session_cost": self.cost_optimizer.get_session_cost(),
            "cost_limits": {
                "per_session": self.cost_optimizer.max_cost_per_session,
                "per_file": self.cost_optimizer.max_cost_per_file,
            },
        }

        # Check vision model availability
        if self.vision_model:
            try:
                # For Ollama models, check if they're actually available
                if hasattr(self.vision_model, "check_model_available"):
                    status[
                        "vision_model_reachable"
                    ] = await self.vision_model.check_model_available()
                else:
                    status["vision_model_reachable"] = True
            except Exception as e:
                status["vision_model_reachable"] = False
                status["vision_model_error"] = str(e)

        return status
