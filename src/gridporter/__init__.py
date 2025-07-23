"""GridPorter - Intelligent spreadsheet ingestion framework."""

__version__ = "0.1.0"

from gridporter.models import DetectionResult, TableInfo
from gridporter.gridporter import GridPorter

__all__ = ["GridPorter", "DetectionResult", "TableInfo"]