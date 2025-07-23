"""Detection strategies for table identification."""

from .base import BaseDetector
from .single_table import SingleTableDetector
from .list_objects import ListObjectsDetector
from .island_detector import IslandDetector
from .csv_detector import CSVDetector
from .heuristics import HeuristicsDetector

__all__ = [
    "BaseDetector",
    "SingleTableDetector",
    "ListObjectsDetector",
    "IslandDetector",
    "CSVDetector",
    "HeuristicsDetector",
]