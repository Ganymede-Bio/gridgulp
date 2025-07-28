"""Centralized constants for GridPorter.

This module contains all constants used throughout the GridPorter codebase,
organized by category for easy access and maintenance.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class IslandDetectionConstants:
    """Constants for island detection algorithm."""

    # Minimum cell counts for confidence levels
    MIN_CELLS_GOOD: Final[int] = 20
    MIN_CELLS_MEDIUM: Final[int] = 10
    MIN_CELLS_SMALL: Final[int] = 4

    # Density thresholds
    DENSITY_HIGH: Final[float] = 0.8
    DENSITY_MEDIUM: Final[float] = 0.6
    DENSITY_LOW: Final[float] = 0.3

    # Aspect ratio limits
    ASPECT_RATIO_MIN: Final[float] = 0.1
    ASPECT_RATIO_MAX: Final[float] = 10.0

    # Confidence scoring
    BASE_CONFIDENCE: Final[float] = 0.5
    CONFIDENCE_BOOST_LARGE: Final[float] = 0.2
    CONFIDENCE_BOOST_MEDIUM: Final[float] = 0.1
    CONFIDENCE_PENALTY_SMALL: Final[float] = 0.2
    CONFIDENCE_PENALTY_LOW_DENSITY: Final[float] = 0.2


@dataclass(frozen=True)
class FormatAnalysisConstants:
    """Constants for semantic format analysis."""

    # Thresholds
    BLANK_ROW_THRESHOLD: Final[float] = 0.9  # % of cells that must be empty
    TOTAL_FORMATTING_THRESHOLD: Final[float] = 0.5  # % of cells that must be bold
    CONSISTENT_COLUMN_THRESHOLD: Final[float] = 0.8  # % of cells that must match

    # Pattern detection
    MIN_DATA_ROWS_FOR_PATTERN: Final[int] = 4
    MAX_ROWS_TO_SAMPLE: Final[int] = 20
    FIRST_ROWS_TO_CHECK: Final[int] = 10
    SECTION_BOUNDARY_MIN_ROWS: Final[int] = 2


@dataclass(frozen=True)
class ComplexTableConstants:
    """Constants for complex table detection."""

    # Confidence thresholds
    DEFAULT_SIMPLE_HEADER_CONFIDENCE: Final[float] = 0.7
    MIN_CONFIDENCE_FOR_ISLAND: Final[float] = 0.5
    MIN_CONFIDENCE_FOR_GOOD_ISLAND: Final[float] = 0.7

    # Analysis parameters
    SEMANTIC_ROW_SCORE_DIVISOR: Final[int] = 5
    PREVIEW_ROW_COUNT: Final[int] = 5
    DATA_TYPE_SAMPLE_SIZE: Final[int] = 20
    BOLD_HEADER_THRESHOLD: Final[float] = 0.5

    # Vision estimates
    VISION_FULL_TOKEN_ESTIMATE: Final[int] = 5000
    VISION_FULL_COST_ESTIMATE: Final[float] = 0.05
    MIN_PROCESSING_TIME_MS: Final[int] = 1


@dataclass(frozen=True)
class CostOptimizationConstants:
    """Constants for cost optimization."""

    # Cost limits
    DEFAULT_MAX_COST_PER_SESSION: Final[float] = 1.0  # USD
    DEFAULT_MAX_COST_PER_FILE: Final[float] = 0.1  # USD
    DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.8

    # Caching
    DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour

    # Batch processing
    MAX_BATCH_SIZE: Final[int] = 10
    MAX_VISION_PER_BATCH_DEFAULT: Final[int] = 3
    SHEETS_THRESHOLD_FOR_STOP_ON_COMPLEX: Final[int] = 5

    # Cost calculation
    BASE_CELL_COUNT: Final[int] = 1000  # 100 rows x 10 cols
    AVERAGE_COST_PER_SHEET: Final[float] = 0.02
    COMPLEXITY_INDICATORS_THRESHOLD: Final[int] = 2

    # Token estimates
    VISION_TOKEN_MULTIPLIER: Final[int] = 100000
    MINI_MODEL_COST_PER_TOKEN: Final[float] = 0.0000004
    LARGE_MODEL_COST_PER_TOKEN: Final[float] = 0.00001
    CLAUDE_COST_PER_TOKEN: Final[float] = 0.000008
    DEFAULT_COST_PER_TOKEN: Final[float] = 0.000001


@dataclass(frozen=True)
class ExcelLimits:
    """Excel format limitations."""

    # XLSX limits
    XLSX_MAX_ROWS: Final[int] = 1048576
    XLSX_MAX_COLS: Final[int] = 16384

    # XLS limits
    XLS_MAX_ROWS: Final[int] = 65536
    XLS_MAX_COLS: Final[int] = 256


@dataclass(frozen=True)
class Keywords:
    """Keywords for pattern detection (multi-language support)."""

    # Subtotal keywords
    SUBTOTAL_KEYWORDS: Final[tuple[str, ...]] = ("subtotal", "sub-total", "подытог")

    # Grand total keywords
    GRAND_TOTAL_KEYWORDS: Final[tuple[str, ...]] = ("grand total", "total", "sum", "итого", "всего")

    # Section keywords
    SECTION_KEYWORDS: Final[tuple[str, ...]] = (
        "section",
        "category",
        "group",
        "раздел",
        "категория",
    )

    # Hierarchical subtotal keywords (extended set)
    HIERARCHICAL_SUBTOTAL_KEYWORDS: Final[tuple[str, ...]] = (
        "total",
        "subtotal",
        "sum",
        "sub-total",
        "grand total",
        "net",
        "gross",
        "overall",
    )


# Method costs for cost optimizer (in USD per operation)
METHOD_COSTS: Final[dict[str, float]] = {
    "simple_case": 0.0,
    "island_detection": 0.0,
    "excel_metadata": 0.0,
    "vision_basic": 0.01,  # ~1000 tokens
    "vision_full": 0.05,  # ~5000 tokens with refinement
}

# Method processing times (in seconds)
METHOD_TIMES: Final[dict[str, float]] = {
    "simple_case": 0.01,
    "island_detection": 0.1,
    "excel_metadata": 0.05,
    "vision_basic": 2.0,
    "vision_full": 5.0,
}


# Create singleton instances for easy access
ISLAND_DETECTION = IslandDetectionConstants()
FORMAT_ANALYSIS = FormatAnalysisConstants()
COMPLEX_TABLE = ComplexTableConstants()
COST_OPTIMIZATION = CostOptimizationConstants()
EXCEL_LIMITS = ExcelLimits()
KEYWORDS = Keywords()
