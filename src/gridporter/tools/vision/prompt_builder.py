"""Tool for building vision model prompts."""

import json
from typing import Any


def build_vision_prompt(template: str, images: list[Any], context: dict[str, Any]) -> str:
    """
    Build a prompt for vision model with template and context.

    This tool constructs prompts that include compression information,
    definitions, and expected output formats.

    Args:
        template: Template name to use
        images: List of images with metadata
        context: Additional context for the prompt

    Returns:
        Complete prompt string
    """
    # Load template
    templates = _get_prompt_templates()
    base_prompt = templates.get(template, templates["default"])

    # Add compression information
    compression_info = _build_compression_info(images)

    # Add definitions if requested
    definitions = context.get("definitions", {})
    definitions_text = _format_definitions(definitions) if definitions else ""

    # Add expected output schema
    output_schema = context.get("expected_output", {})
    schema_text = _format_output_schema(output_schema) if output_schema else ""

    # Build complete prompt
    prompt_parts = [
        base_prompt,
        "\n## IMAGE INFORMATION:",
        compression_info,
        definitions_text,
        schema_text,
    ]

    # Add any proposals
    if "proposals" in context:
        prompt_parts.append("\n## INITIAL PROPOSALS TO VERIFY:")
        prompt_parts.append(json.dumps(context["proposals"], indent=2))

    return "\n".join(filter(None, prompt_parts))


def _get_prompt_templates() -> dict[str, str]:
    """Get prompt templates."""
    return {
        "default": "Analyze the provided spreadsheet images to detect tables.",
        "EXPLICIT_MULTI_SCALE": """You are analyzing spreadsheet visualizations at multiple scales to detect precise table boundaries.

CRITICAL: Each image has different compression levels:
- Overview images show the overall layout but NOT exact boundaries
- Detail images (1:1 scale) show EXACT cell-level boundaries

Use overview for context, then use detail images for precise coordinates.""",
        "PROGRESSIVE_REFINEMENT": """You are analyzing a very large spreadsheet through progressive refinement.

This is phase {phase} of the analysis:
- Phase 1: Identify general regions containing tables
- Phase 2: Refine boundaries of identified regions
- Phase 3: Verify exact cell-level boundaries

Focus on the specific phase objective.""",
        "MULTI_TABLE_DETECTION": """You are analyzing a spreadsheet that likely contains multiple tables.

Key objectives:
1. Identify ALL distinct tables (don't merge separate tables)
2. Look for natural breaks: empty rows/columns, format changes
3. Each table should be cohesive and self-contained
4. Tables may have different structures and purposes""",
    }


def _build_compression_info(images: list[Any]) -> str:
    """Build compression information text."""
    lines = []
    for i, img in enumerate(images):
        if hasattr(img, "compression_level"):
            if img.compression_level == 0:
                lines.append(
                    f"- Image {i+1}: NO COMPRESSION (1 pixel = 1 cell) - Use for EXACT boundaries"
                )
            else:
                blocks = img.block_size
                lines.append(
                    f"- Image {i+1}: {blocks[0]}×{blocks[1]} compression "
                    f"(each pixel = {blocks[0]} rows × {blocks[1]} columns)"
                )
    return "\n".join(lines)


def _format_definitions(definitions: dict[str, str]) -> str:
    """Format definitions for the prompt."""
    if not definitions:
        return ""

    lines = ["\n## DEFINITIONS:"]
    for term, definition in definitions.items():
        lines.append(f"- **{term}**: {definition}")
    return "\n".join(lines)


def _format_output_schema(schema: dict[str, Any]) -> str:
    """Format output schema for the prompt."""
    if not schema:
        return ""

    return f"\n## EXPECTED OUTPUT FORMAT:\n```json\n{json.dumps(schema, indent=2)}\n```"
