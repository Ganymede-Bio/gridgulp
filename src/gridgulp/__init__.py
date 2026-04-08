"""GridGulp - Gemma-first spreadsheet schematization."""

__version__ = "0.5.0"

from gridgulp.config import (
    Config,
    DetectionStrategy,
    GemmaConfig,
    VisionConfig,
    VisionMode,
)
from gridgulp.gridgulp import GridGulp
from gridgulp.models import DetectionResult, TableInfo, WorkbookSchemaResult

__all__ = [
    "Config",
    "DetectionResult",
    "DetectionStrategy",
    "GemmaConfig",
    "GridGulp",
    "TableInfo",
    "WorkbookSchemaResult",
    "VisionConfig",
    "VisionMode",
]
