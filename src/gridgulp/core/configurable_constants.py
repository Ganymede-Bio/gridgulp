"""Configurable constants that can be overridden via Config."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .constants import (
    COMPLEX_TABLE,
    COST_OPTIMIZATION,
    FORMAT_ANALYSIS,
    ISLAND_DETECTION,
    ComplexTableConstants,
    CostOptimizationConstants,
    FormatAnalysisConstants,
    IslandDetectionConstants,
)

if TYPE_CHECKING:
    from ..config import Config


@dataclass(frozen=True)
class ConfigurableConstants:
    """Container for all configurable constants."""

    island_detection: IslandDetectionConstants
    format_analysis: FormatAnalysisConstants
    complex_table: ComplexTableConstants
    cost_optimization: CostOptimizationConstants


_default_constants = ConfigurableConstants(
    island_detection=ISLAND_DETECTION,
    format_analysis=FORMAT_ANALYSIS,
    complex_table=COMPLEX_TABLE,
    cost_optimization=COST_OPTIMIZATION,
)

_current_constants = _default_constants


def apply_config_overrides(config: "Config") -> None:
    """Apply configuration overrides to constants.

    Args:
        config: Configuration object with override values
    """
    global _current_constants

    # Create new instances with overridden values
    island_overrides: dict[str, Any] = {}
    if hasattr(config, "island_min_cells"):
        island_overrides["MIN_CELLS_GOOD"] = int(config.island_min_cells)
    if hasattr(config, "island_density_threshold"):
        island_overrides["DENSITY_HIGH"] = float(config.island_density_threshold)

    format_overrides: dict[str, Any] = {}
    if hasattr(config, "format_blank_row_threshold"):
        format_overrides["BLANK_ROW_THRESHOLD"] = float(config.format_blank_row_threshold)
    if hasattr(config, "format_total_formatting_threshold"):
        format_overrides["TOTAL_FORMATTING_THRESHOLD"] = float(
            config.format_total_formatting_threshold
        )

    # Apply overrides if any exist
    if island_overrides:
        new_island = replace(ISLAND_DETECTION, **island_overrides)
        _current_constants = replace(_current_constants, island_detection=new_island)

    if format_overrides:
        new_format = replace(FORMAT_ANALYSIS, **format_overrides)
        _current_constants = replace(_current_constants, format_analysis=new_format)


def get_constants() -> ConfigurableConstants:
    """Get the current constants (with any config overrides applied)."""
    return _current_constants


def reset_to_defaults() -> None:
    """Reset all constants to their default values."""
    global _current_constants
    _current_constants = _default_constants
