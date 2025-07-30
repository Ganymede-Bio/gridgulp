"""Centralized constants for GridGulp.

This module contains all constants used throughout the GridGulp codebase,
organized by category for easy access and maintenance.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class IslandDetectionConstants:
    """Constants for island detection algorithm."""

    # Minimum cell counts for confidence levels (used as baseline, adjusted by sheet size)
    MIN_CELLS_GOOD: Final[int] = 20  # Islands with 20+ cells get good confidence boost
    MIN_CELLS_MEDIUM: Final[int] = 10  # Islands with 10-19 cells get medium confidence boost
    MIN_CELLS_SMALL: Final[int] = 4  # Islands with 4-9 cells get small/no confidence boost

    # Density thresholds
    DENSITY_HIGH: Final[float] = 0.8  # 80%+ filled cells indicates high-quality table region
    DENSITY_MEDIUM: Final[float] = 0.6  # 60-79% filled cells indicates medium quality
    DENSITY_LOW: Final[float] = 0.3  # Below 30% filled cells indicates sparse/poor region

    # Aspect ratio limits
    ASPECT_RATIO_MIN: Final[float] = 0.1  # Minimum width/height ratio (prevents very tall/thin)
    ASPECT_RATIO_MAX: Final[float] = 10.0  # Maximum width/height ratio (prevents very wide/flat)

    # Confidence scoring
    BASE_CONFIDENCE: Final[float] = 0.5  # Starting confidence for all detected islands
    CONFIDENCE_BOOST_LARGE: Final[float] = 0.2  # Boost for large islands (20+ cells)
    CONFIDENCE_BOOST_MEDIUM: Final[float] = 0.1  # Boost for medium islands (10-19 cells)
    CONFIDENCE_PENALTY_SMALL: Final[float] = 0.2  # Penalty for small islands (<10 cells)
    CONFIDENCE_PENALTY_LOW_DENSITY: Final[float] = 0.2  # Penalty for sparse islands (<30% filled)

    # Structural analysis parameters
    COLUMN_CONSISTENCY_THRESHOLD: Final[float] = (
        0.8  # Min similarity for grouping rows by column usage
    )
    MIN_EMPTY_ROWS_TO_SPLIT: Final[int] = 1  # Number of empty rows needed to split islands
    DEFAULT_MAX_GAP: Final[int] = 1  # Default max gap between cells for connectivity
    TEXT_FILE_MAX_GAP: Final[int] = 0  # Max gap for text files (stricter separation)
    EXCEL_FILE_MAX_GAP: Final[int] = 1  # Max gap for Excel files

    # Relative size thresholds (percentage of total sheet cells)
    RELATIVE_SIZE_LARGE: Final[float] = 0.05  # 5%+ of sheet cells is a large table
    RELATIVE_SIZE_MEDIUM: Final[float] = 0.01  # 1-5% of sheet cells is medium table
    RELATIVE_SIZE_SMALL: Final[float] = 0.005  # 0.5-1% of sheet cells is small table
    RELATIVE_SIZE_TINY: Final[float] = 0.001  # <0.1% is tiny (usually noise)

    # Border analysis parameters
    BORDER_WIDTH: Final[int] = 2  # Width of border to check around tables
    BORDER_CELL_THRESHOLD: Final[float] = 0.3  # 30%+ border cells populated is concerning
    CONFIDENCE_PENALTY_BORDER: Final[float] = 0.2  # Penalty for high border population
    CONFIDENCE_PENALTY_SUBSET: Final[float] = 0.3  # Penalty for being subset of another table


@dataclass(frozen=True)
class FormatAnalysisConstants:
    """Constants for semantic format analysis."""

    # Thresholds
    BLANK_ROW_THRESHOLD: Final[float] = 0.9  # Row is blank if 90%+ cells are empty
    TOTAL_FORMATTING_THRESHOLD: Final[float] = 0.5  # Row is total/summary if 50%+ cells are bold
    CONSISTENT_COLUMN_THRESHOLD: Final[float] = (
        0.8  # Column has consistent type if 80%+ cells match
    )

    # Pattern detection
    MIN_DATA_ROWS_FOR_PATTERN: Final[int] = 4  # Need at least 4 data rows to detect patterns
    MAX_ROWS_TO_SAMPLE: Final[int] = 20  # Sample up to 20 rows for pattern analysis
    FIRST_ROWS_TO_CHECK: Final[int] = 10  # Check first 10 rows for header candidates
    SECTION_BOUNDARY_MIN_ROWS: Final[int] = 2  # Minimum rows between table sections


@dataclass(frozen=True)
class ComplexTableConstants:
    """Constants for complex table detection."""

    # Confidence thresholds
    DEFAULT_SIMPLE_HEADER_CONFIDENCE: Final[float] = (
        0.7  # Default confidence for simple header detection
    )
    MIN_CONFIDENCE_FOR_ISLAND: Final[float] = 0.5  # Minimum confidence to accept island detection
    MIN_CONFIDENCE_FOR_GOOD_ISLAND: Final[float] = (
        0.6  # Confidence threshold for high-quality islands (lowered to accept more good results)
    )

    # Analysis parameters
    SEMANTIC_ROW_SCORE_DIVISOR: Final[int] = 5  # Divisor for semantic row scoring calculation
    PREVIEW_ROW_COUNT: Final[int] = 5  # Number of rows to include in table preview
    DATA_TYPE_SAMPLE_SIZE: Final[int] = 20  # Sample size for data type inference
    BOLD_HEADER_THRESHOLD: Final[float] = 0.5  # Row is header if 50%+ cells are bold

    # Vision estimates
    VISION_FULL_TOKEN_ESTIMATE: Final[int] = 5000  # Estimated tokens for full vision analysis
    VISION_FULL_COST_ESTIMATE: Final[float] = 0.05  # Estimated cost in USD for full vision analysis
    MIN_PROCESSING_TIME_MS: Final[int] = 1  # Minimum processing time to avoid division by zero


@dataclass(frozen=True)
class CostOptimizationConstants:
    """Constants for cost optimization."""

    # Cost limits
    DEFAULT_MAX_COST_PER_SESSION: Final[float] = 1.0  # Maximum USD cost allowed per session
    DEFAULT_MAX_COST_PER_FILE: Final[float] = 0.1  # Maximum USD cost allowed per file
    DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = (
        0.6  # Default confidence threshold for detection (lowered for better recall)
    )

    # Caching
    DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # Cache time-to-live: 1 hour (3600 seconds)

    # Batch processing
    MAX_BATCH_SIZE: Final[int] = 10  # Maximum files to process in a single batch
    MAX_VISION_PER_BATCH_DEFAULT: Final[int] = 3  # Max vision analyses per batch (cost control)
    SHEETS_THRESHOLD_FOR_STOP_ON_COMPLEX: Final[int] = 5  # Stop early if 5+ complex sheets found

    # Cost calculation
    BASE_CELL_COUNT: Final[int] = 1000  # Base size for cost estimation: 100 rows x 10 cols
    AVERAGE_COST_PER_SHEET: Final[float] = 0.02  # Average cost per sheet in USD
    COMPLEXITY_INDICATORS_THRESHOLD: Final[int] = 2  # Sheet is complex if 2+ indicators present

    # Token estimates
    VISION_TOKEN_MULTIPLIER: Final[int] = 100000  # Multiplier for vision token estimates
    MINI_MODEL_COST_PER_TOKEN: Final[float] = (
        0.0000004  # Cost per token for mini models (GPT-4o-mini)
    )
    LARGE_MODEL_COST_PER_TOKEN: Final[float] = 0.00001  # Cost per token for large models (GPT-4o)
    CLAUDE_COST_PER_TOKEN: Final[float] = 0.000008  # Cost per token for Claude models
    DEFAULT_COST_PER_TOKEN: Final[float] = 0.000001  # Default/fallback cost per token


@dataclass(frozen=True)
class ExcelLimits:
    """Excel format limitations."""

    # XLSX limits
    XLSX_MAX_ROWS: Final[int] = 1048576  # Excel 2007+ maximum row count
    XLSX_MAX_COLS: Final[int] = 16384  # Excel 2007+ maximum column count (XFD)

    # XLS limits
    XLS_MAX_ROWS: Final[int] = 65536  # Excel 97-2003 maximum row count
    XLS_MAX_COLS: Final[int] = 256  # Excel 97-2003 maximum column count (IV)


@dataclass(frozen=True)
class ComplexityAssessmentConstants:
    """Constants for sheet complexity assessment."""

    # Thresholds for different assessment factors
    LARGE_SHEET_THRESHOLD: Final[int] = 1000000  # Sheet is large if it has 1m+ cells
    MULTI_TABLE_THRESHOLD: Final[int] = 3  # Sheet is multi-table if 3+ patterns detected
    FORMAT_VARIETY_THRESHOLD: Final[int] = 5  # High format variety if 5+ different formats
    MERGED_CELL_THRESHOLD: Final[float] = 0.1  # High merged cell usage if 10%+ cells are merged
    VISION_THRESHOLD: Final[float] = 0.6  # Use vision if complexity score exceeds 0.6

    # Assessment weights
    SPARSITY_WEIGHT: Final[float] = 0.3  # 30% weight for data sparsity in complexity score
    SIZE_WEIGHT: Final[float] = 0.2  # 20% weight for sheet size in complexity score
    PATTERN_WEIGHT: Final[float] = 0.25  # 25% weight for pattern count in complexity score
    MERGED_CELL_WEIGHT: Final[float] = 0.15  # 15% weight for merged cells in complexity score
    FORMAT_WEIGHT: Final[float] = 0.1  # 10% weight for format variety in complexity score


@dataclass(frozen=True)
class VisionOrchestratorConstants:
    """Constants for vision orchestrator agent."""

    # Cost estimation thresholds (in number of cells)
    SMALL_SHEET_THRESHOLD: Final[int] = 1000  # Sheets with <1k cells are small
    MEDIUM_SHEET_THRESHOLD: Final[int] = 1000000  # Sheets with 1k-1m cells are medium

    # Cost estimates (in USD)
    SMALL_SHEET_COST: Final[float] = 0.01  # Estimated cost for small sheet vision analysis
    MEDIUM_SHEET_COST: Final[float] = 0.03  # Estimated cost for medium sheet vision analysis
    LARGE_SHEET_COST: Final[float] = 0.08  # Estimated cost for large sheet vision analysis

    # Strategy selection parameters
    CONFIDENCE_THRESHOLD_HIGH: Final[float] = 0.9  # High confidence: use premium strategies
    CONFIDENCE_THRESHOLD_MEDIUM: Final[float] = 0.75  # Medium confidence: balanced approach
    CONFIDENCE_THRESHOLD_LOW: Final[float] = 0.6  # Low confidence: cost-conscious approach


@dataclass(frozen=True)
class FormattingDetectionConstants:
    """Constants for formatting-based table detection."""

    # Header detection thresholds
    HEADER_BOLD_THRESHOLD: Final[float] = 0.7  # 70% of header row cells must be bold
    HEADER_BACKGROUND_WEIGHT: Final[float] = 0.8  # Weight for background color in header detection

    # Formatting change thresholds
    BACKGROUND_CHANGE_THRESHOLD: Final[float] = 0.3  # Significant background color change
    FONT_SIZE_CHANGE_THRESHOLD: Final[float] = 2.0  # Font size difference indicating new section
    FONT_COLOR_CHANGE_THRESHOLD: Final[float] = 0.5  # Font color difference threshold

    # Border-based boundary detection
    BORDER_CONSISTENCY_THRESHOLD: Final[float] = 0.8  # Border similarity needed to group cells
    BORDER_CHANGE_THRESHOLD: Final[float] = 0.5  # Significant border pattern change
    OUTER_BORDER_WEIGHT: Final[float] = 0.3  # Weight for outer border in boundary detection
    INNER_BORDER_WEIGHT: Final[float] = 0.7  # Weight for inner border consistency

    # Border signature types
    NO_BORDERS: Final[str] = "none"  # No borders present
    ALL_BORDERS: Final[str] = "all"  # All borders present (top/bottom/left/right)
    OUTER_ONLY: Final[str] = "outer"  # Only outer borders (table boundary)
    HORIZONTAL_ONLY: Final[str] = "horizontal"  # Only top/bottom borders
    VERTICAL_ONLY: Final[str] = "vertical"  # Only left/right borders
    MIXED_BORDERS: Final[str] = "mixed"  # Mixed border pattern

    # Formatting consistency requirements
    MIN_FORMATTING_CONSISTENCY: Final[float] = 0.8  # Formatting consistency within table
    FORMATTING_SIMILARITY_THRESHOLD: Final[float] = 0.7  # How similar formatting must be to group

    # Visual boundary detection
    EMPTY_ROW_FORMATTING_WEIGHT: Final[float] = 0.5  # Weight formatting in empty row analysis
    MERGED_CELL_BOUNDARY_RESPECT: Final[bool] = True  # Respect merged cell boundaries

    # Format-aware merging parameters
    ALLOW_CROSS_FORMAT_MERGE: Final[bool] = (
        False  # Don't merge tables with different header formats
    )
    HEADER_FORMAT_MERGE_THRESHOLD: Final[float] = (
        0.9  # Similarity needed to merge different header formats
    )


@dataclass(frozen=True)
class Keywords:
    """Keywords for pattern detection."""

    # Subtotal keywords
    SUBTOTAL_KEYWORDS: Final[tuple[str, ...]] = (
        "subtotal",
        "sub-total",
    )  # Keywords indicating subtotal rows

    # Grand total keywords
    GRAND_TOTAL_KEYWORDS: Final[tuple[str, ...]] = (
        "grand total",
        "total",
        "sum",
    )

    # Section keywords
    SECTION_KEYWORDS: Final[tuple[str, ...]] = (
        "section",
        "category",
        "group",
    )  # Keywords indicating section headers

    # Hierarchical subtotal keywords (extended set)
    HIERARCHICAL_SUBTOTAL_KEYWORDS: Final[tuple[str, ...]] = (
        "total",
        "subtotal",
        "sum",
        "sub-total",
        "grand total",
        "net",  # Net total/amount
        "gross",  # Gross total/amount
        "overall",  # Overall total
    )  # Comprehensive set for hierarchical table detection


# Method costs for cost optimizer (in USD per operation)
METHOD_COSTS: Final[dict[str, float]] = {
    "simple_case": 0.0,  # Free: uses only basic heuristics
    "island_detection": 0.0,  # Free: uses only algorithmic detection
    "excel_metadata": 0.0,  # Free: uses only Excel's built-in metadata
    "vision_basic": 0.01,  # ~1000 tokens for basic vision analysis
    "vision_full": 0.05,  # ~5000 tokens for full vision with refinement
}

# Method processing times (in seconds)
METHOD_TIMES: Final[dict[str, float]] = {
    "simple_case": 0.01,  # Very fast: basic checks only
    "island_detection": 0.1,  # Fast: algorithmic detection
    "excel_metadata": 0.05,  # Fast: metadata extraction
    "vision_basic": 2.0,  # Moderate: API call + processing
    "vision_full": 5.0,  # Slow: multiple API calls + refinement
}


# Create singleton instances for easy access
ISLAND_DETECTION = IslandDetectionConstants()  # Constants for island detection algorithm
FORMAT_ANALYSIS = FormatAnalysisConstants()  # Constants for semantic format analysis
COMPLEX_TABLE = ComplexTableConstants()  # Constants for complex table detection
COST_OPTIMIZATION = CostOptimizationConstants()  # Constants for cost optimization
EXCEL_LIMITS = ExcelLimits()  # Excel format limitations
COMPLEXITY_ASSESSMENT = ComplexityAssessmentConstants()  # Constants for complexity assessment
VISION_ORCHESTRATOR = VisionOrchestratorConstants()  # Constants for vision orchestrator
FORMATTING_DETECTION = FormattingDetectionConstants()  # Constants for formatting-based detection
KEYWORDS = Keywords()  # Multi-language keywords for pattern detection
