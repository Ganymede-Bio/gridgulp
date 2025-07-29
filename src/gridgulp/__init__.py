"""GridGulp - Intelligent spreadsheet ingestion framework."""

__version__ = "0.3.2"

from gridgulp.gridgulp import GridGulp
from gridgulp.models import DetectionResult, TableInfo

__all__ = ["GridGulp", "DetectionResult", "TableInfo"]
