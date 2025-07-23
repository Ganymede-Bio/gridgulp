"""Detection strategies for table identification."""

from .base import BaseDetector
from .csv_detector import CSVDetector
from .heuristics import HeuristicsDetector
from .island_detector import IslandDetector
from .list_objects import ListObjectsDetector
from .single_table import SingleTableDetector

__all__ = [
    "BaseDetector",
    "SingleTableDetector",
    "ListObjectsDetector",
    "IslandDetector",
    "CSVDetector",
    "HeuristicsDetector",
]
