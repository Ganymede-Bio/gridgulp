"""Utility functions for GridGulp."""

from .file_magic import detect_file_info, detect_file_type
from .visualization import visualize_detection

__all__ = ["detect_file_type", "detect_file_info", "visualize_detection"]
