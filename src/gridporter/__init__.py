"""GridPorter - Intelligent spreadsheet ingestion framework."""

__version__ = "0.2.1"

from gridporter.gridporter import GridPorter
from gridporter.models import DetectionResult, TableInfo

__all__ = ["GridPorter", "DetectionResult", "TableInfo"]
