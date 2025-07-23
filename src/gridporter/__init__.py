"""GridPorter - Intelligent spreadsheet ingestion framework."""

__version__ = "0.1.0"

from gridporter.gridporter import GridPorter
from gridporter.models import DetectionResult, TableInfo

__all__ = ["GridPorter", "DetectionResult", "TableInfo"]
