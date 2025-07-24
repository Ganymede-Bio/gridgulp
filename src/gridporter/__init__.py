"""GridPorter - Intelligent spreadsheet ingestion framework."""

__version__ = "0.1.1"

from .gridporter import GridPorter
from .models import DetectionResult, TableInfo

__all__ = ["GridPorter", "DetectionResult", "TableInfo"]
