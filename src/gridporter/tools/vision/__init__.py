"""Vision tools for bitmap generation and prompt building."""

from .bitmap_generator import generate_multi_scale_bitmaps
from .compression_calculator import calculate_optimal_compression
from .prompt_builder import build_vision_prompt
from .region_highlighter import highlight_region_with_context
from .response_parser import parse_vision_response

__all__ = [
    "generate_multi_scale_bitmaps",
    "build_vision_prompt",
    "calculate_optimal_compression",
    "highlight_region_with_context",
    "parse_vision_response",
]
