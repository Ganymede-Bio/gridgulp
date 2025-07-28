"""Cost optimization manager for intelligent routing between detection methods.

This module manages cost-aware decisions about when to use vision-based detection
versus traditional algorithms, tracks cumulative costs, and implements batch
processing strategies.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..core.constants import COST_OPTIMIZATION, METHOD_COSTS, METHOD_TIMES
from .openai_pricing import get_pricing_instance

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Available detection methods with associated costs."""

    SIMPLE_CASE = "simple_case"  # Free - no API calls
    ISLAND_DETECTION = "island_detection"  # Free - no API calls
    EXCEL_METADATA = "excel_metadata"  # Free - no API calls
    VISION_BASIC = "vision_basic"  # Low cost - single vision call
    VISION_FULL = "vision_full"  # High cost - multiple vision calls with refinement


@dataclass
class CostEstimate:
    """Estimated cost for a detection operation."""

    method: DetectionMethod
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_time_seconds: float = 0.0
    confidence_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class CostTracker:
    """Tracks cumulative costs for a session."""

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0
    method_counts: dict[DetectionMethod, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def add_usage(self, method: DetectionMethod, tokens: int = 0, cost: float = 0.0) -> None:
        """Add usage for a detection method."""
        self.total_tokens += tokens
        self.total_cost_usd += cost
        if tokens > 0 or cost > 0:
            self.total_api_calls += 1
        self.method_counts[method] = self.method_counts.get(method, 0) + 1

    def get_session_duration(self) -> timedelta:
        """Get the duration of the current session."""
        return datetime.now() - self.start_time


@dataclass
class BatchStrategy:
    """Strategy for batch processing multiple sheets."""

    max_batch_size: int = COST_OPTIMIZATION.MAX_BATCH_SIZE
    prefer_simple_first: bool = True
    stop_on_complex: bool = False
    max_vision_per_batch: int = COST_OPTIMIZATION.MAX_VISION_PER_BATCH_DEFAULT


class CostOptimizer:
    """Manages cost-aware routing between detection methods."""

    def __init__(
        self,
        max_cost_per_session: float = COST_OPTIMIZATION.DEFAULT_MAX_COST_PER_SESSION,
        max_cost_per_file: float = COST_OPTIMIZATION.DEFAULT_MAX_COST_PER_FILE,
        confidence_threshold: float = COST_OPTIMIZATION.DEFAULT_CONFIDENCE_THRESHOLD,
        enable_caching: bool = True,
    ):
        """Initialize the cost optimizer.

        Args:
            max_cost_per_session: Maximum cost allowed per session in USD
            max_cost_per_file: Maximum cost allowed per file in USD
            confidence_threshold: Minimum confidence to avoid vision processing
            enable_caching: Whether to use caching for cost savings
        """
        self.logger = logger
        self.max_cost_per_session = max_cost_per_session
        self.max_cost_per_file = max_cost_per_file
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching
        self.tracker = CostTracker()
        self._cache: dict[str, Any] = {}

    def select_detection_strategy(
        self,
        sheet_complexity: dict[str, Any],
        available_hints: list[dict[str, Any]],
        current_file_cost: float = 0.0,
    ) -> list[DetectionMethod]:
        """Select optimal detection strategy based on complexity and cost.

        Args:
            sheet_complexity: Complexity metrics for the sheet
            available_hints: Available hints from Excel metadata
            current_file_cost: Current cost for this file

        Returns:
            Ordered list of detection methods to try
        """
        if sheet_complexity is None:
            raise ValueError("sheet_complexity cannot be None")
        if available_hints is None:
            available_hints = []
        if current_file_cost < 0:
            raise ValueError(f"current_file_cost must be non-negative, got {current_file_cost}")

        strategies = []

        # Always try simple case first (free)
        strategies.append(DetectionMethod.SIMPLE_CASE)

        # If we have high-confidence Excel metadata, use it
        if available_hints:
            max_hint_confidence = max(
                (hint.get("confidence", 0) for hint in available_hints), default=0
            )
            if max_hint_confidence >= self.confidence_threshold:
                strategies.append(DetectionMethod.EXCEL_METADATA)
                # May not need vision at all
                return strategies

        # Try island detection (free)
        strategies.append(DetectionMethod.ISLAND_DETECTION)

        # Check if we should use vision based on complexity and budget
        remaining_budget = min(
            self.max_cost_per_file - current_file_cost,
            self.max_cost_per_session - self.tracker.total_cost_usd,
        )

        # Estimate complexity from metrics
        is_complex = self._estimate_complexity(sheet_complexity)

        if remaining_budget > METHOD_COSTS[DetectionMethod.VISION_BASIC.value]:
            if is_complex and remaining_budget > METHOD_COSTS[DetectionMethod.VISION_FULL.value]:
                strategies.append(DetectionMethod.VISION_FULL)
            else:
                strategies.append(DetectionMethod.VISION_BASIC)

        return strategies

    def estimate_cost(self, method: DetectionMethod, sheet_metrics: dict[str, Any]) -> CostEstimate:
        """Estimate the cost of using a specific detection method.

        Args:
            method: Detection method to estimate
            sheet_metrics: Metrics about the sheet (size, density, etc.)

        Returns:
            CostEstimate with predicted costs
        """
        if method is None:
            raise ValueError("method cannot be None")
        if sheet_metrics is None:
            raise ValueError("sheet_metrics cannot be None")

        base_cost = METHOD_COSTS.get(method.value, 0.0)
        base_time = METHOD_TIMES.get(method.value, 0.0)
        base_tokens = 0

        # Adjust for sheet size for vision methods
        if method in [DetectionMethod.VISION_BASIC, DetectionMethod.VISION_FULL]:
            size_factor = self._calculate_size_factor(sheet_metrics)
            base_cost *= size_factor
            base_tokens = int(base_cost * COST_OPTIMIZATION.VISION_TOKEN_MULTIPLIER)

        # Estimate confidence range based on method
        confidence_ranges = {
            DetectionMethod.SIMPLE_CASE: (0.5, 0.9),
            DetectionMethod.ISLAND_DETECTION: (0.4, 0.8),
            DetectionMethod.EXCEL_METADATA: (0.7, 0.95),
            DetectionMethod.VISION_BASIC: (0.6, 0.9),
            DetectionMethod.VISION_FULL: (0.8, 0.95),
        }

        return CostEstimate(
            method=method,
            estimated_tokens=base_tokens,
            estimated_cost_usd=base_cost,
            estimated_time_seconds=base_time,
            confidence_range=confidence_ranges.get(method, (0.0, 1.0)),
        )

    def optimize_batch_processing(
        self, sheets: list[dict[str, Any]], total_budget: float | None = None
    ) -> BatchStrategy:
        """Optimize processing strategy for multiple sheets.

        Args:
            sheets: List of sheet metadata
            total_budget: Total budget for batch (uses session max if None)

        Returns:
            BatchStrategy for processing the sheets
        """
        if total_budget is None:
            total_budget = self.max_cost_per_session - self.tracker.total_cost_usd

        # Sort sheets by estimated complexity (not used currently but may be useful later)
        # sorted_sheets = sorted(sheets, key=lambda s: self._estimate_complexity(s))

        # Calculate optimal batch size
        avg_cost_per_sheet = COST_OPTIMIZATION.AVERAGE_COST_PER_SHEET
        max_sheets_in_budget = int(total_budget / avg_cost_per_sheet)
        optimal_batch_size = min(
            len(sheets), max_sheets_in_budget, COST_OPTIMIZATION.MAX_BATCH_SIZE
        )

        # Determine strategy
        has_complex_sheets = any(self._estimate_complexity(s) for s in sheets)

        return BatchStrategy(
            max_batch_size=optimal_batch_size,
            prefer_simple_first=True,
            stop_on_complex=has_complex_sheets
            and len(sheets) > COST_OPTIMIZATION.SHEETS_THRESHOLD_FOR_STOP_ON_COMPLEX,
            max_vision_per_batch=min(
                3, int(total_budget / METHOD_COSTS[DetectionMethod.VISION_BASIC.value])
            ),
        )

    def should_use_cache(self, cache_key: str) -> bool:
        """Check if we should use cached results.

        Args:
            cache_key: Key for cache lookup

        Returns:
            True if cache should be used
        """
        if not self.enable_caching:
            return False

        return cache_key in self._cache

    def update_cache(
        self,
        cache_key: str,
        result: Any,
        ttl_seconds: int = COST_OPTIMIZATION.DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Update cache with detection results.

        Args:
            cache_key: Key for caching
            result: Result to cache
            ttl_seconds: Time to live in seconds
        """
        if self.enable_caching:
            self._cache[cache_key] = {
                "result": result,
                "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
            }

    def get_cached_result(self, cache_key: str) -> Any | None:
        """Get cached result if available and not expired.

        Args:
            cache_key: Key for cache lookup

        Returns:
            Cached result or None
        """
        if not self.enable_caching or cache_key not in self._cache:
            return None

        cached = self._cache[cache_key]
        if datetime.now() > cached["expires_at"]:
            del self._cache[cache_key]
            return None

        return cached["result"]

    def get_cost_report(self) -> dict[str, Any]:
        """Get a report of current costs and usage.

        Returns:
            Dictionary with cost and usage statistics
        """
        return {
            "total_cost_usd": self.tracker.total_cost_usd,
            "total_tokens": self.tracker.total_tokens,
            "total_api_calls": self.tracker.total_api_calls,
            "session_duration": str(self.tracker.get_session_duration()),
            "method_usage": dict(self.tracker.method_counts),
            "remaining_budget": self.max_cost_per_session - self.tracker.total_cost_usd,
            "cache_hits": len(self._cache),
        }

    def _estimate_complexity(self, sheet_metrics: dict[str, Any]) -> bool:
        """Estimate if a sheet is complex based on metrics.

        Args:
            sheet_metrics: Metrics about the sheet

        Returns:
            True if sheet appears complex
        """
        # Check various complexity indicators
        indicators = [
            sheet_metrics.get("has_merged_cells", False),
            sheet_metrics.get("table_count", 1) > 1,
            sheet_metrics.get("has_sparse_data", False),
            sheet_metrics.get("has_multiple_formats", False),
            sheet_metrics.get("row_count", 0) > 1000,
        ]

        # Complex if threshold or more indicators are true
        return sum(indicators) >= COST_OPTIMIZATION.COMPLEXITY_INDICATORS_THRESHOLD

    def _calculate_size_factor(self, sheet_metrics: dict[str, Any]) -> float:
        """Calculate size factor for cost adjustment.

        Args:
            sheet_metrics: Metrics about the sheet

        Returns:
            Size factor multiplier (1.0 = normal, >1.0 = larger/more expensive)
        """
        row_count = sheet_metrics.get("row_count", 100)
        col_count = sheet_metrics.get("col_count", 10)

        # Base size for calculations
        base_size = COST_OPTIMIZATION.BASE_CELL_COUNT
        actual_size = row_count * col_count

        # Logarithmic scaling to avoid huge costs for large sheets
        if actual_size <= base_size:
            return 1.0
        else:
            import math

            return 1.0 + math.log10(actual_size / base_size)

    def reset_session(self) -> None:
        """Reset session tracking and cache."""
        self.tracker = CostTracker()
        self._cache.clear()
        self.logger.info("Cost optimizer session reset")

    def update_with_actual_usage(
        self,
        method: DetectionMethod,
        model_id: str,
        usage: dict[str, Any],
    ) -> float:
        """Update tracker with actual token usage and costs.

        Args:
            method: Detection method used
            model_id: Model ID (e.g., "openai/gpt-4o-mini")
            usage: Usage dictionary with token counts

        Returns:
            Actual cost in USD
        """
        # Extract token counts
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        # image_tokens = usage.get("image_tokens", 0)  # Not used currently

        # Check if cost is already calculated
        if "cost_usd" in usage:
            actual_cost = usage["cost_usd"]
        else:
            # Calculate actual cost using OpenAI pricing
            try:
                pricing = get_pricing_instance()
                actual_cost = pricing.calculate_cost(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate cost for {model_id}: {e}")
                # Fall back to estimates
                actual_cost = self._estimate_cost_from_tokens(model_id, total_tokens)

        # Update tracker with actual values
        self.tracker.add_usage(method, tokens=total_tokens, cost=actual_cost)

        self.logger.info(f"{method.value} used {total_tokens} tokens, cost: ${actual_cost:.6f}")

        return actual_cost

    def _estimate_cost_from_tokens(self, model_id: str, total_tokens: int) -> float:
        """Estimate cost when actual pricing unavailable.

        Args:
            model_id: Model identifier
            total_tokens: Total token count

        Returns:
            Estimated cost in USD
        """
        # Rough estimates based on model type
        if "gpt-4o" in model_id.lower() and "mini" not in model_id.lower():
            return total_tokens * COST_OPTIMIZATION.LARGE_MODEL_COST_PER_TOKEN
        elif "gpt-4o-mini" in model_id.lower():
            return total_tokens * COST_OPTIMIZATION.MINI_MODEL_COST_PER_TOKEN
        elif "claude" in model_id.lower():
            return total_tokens * COST_OPTIMIZATION.CLAUDE_COST_PER_TOKEN
        else:
            return total_tokens * COST_OPTIMIZATION.DEFAULT_COST_PER_TOKEN

    def get_model_recommendation(
        self,
        required_features: set[str],
        max_cost_per_request: float | None = None,
    ) -> str | None:
        """Recommend a model based on features and budget.

        Args:
            required_features: Set of required features (e.g., {"vision", "long_context"})
            max_cost_per_request: Maximum cost per request (uses file limit if None)

        Returns:
            Recommended model ID or None
        """
        if max_cost_per_request is None:
            max_cost_per_request = self.max_cost_per_file

        # This would integrate with model pricing to find best model
        # For now, return a sensible default
        if "vision" in required_features:
            if max_cost_per_request < 0.01:
                return "gpt-4o-mini"  # Cheapest vision model
            else:
                return "gpt-4o"  # Better vision model

        return None
