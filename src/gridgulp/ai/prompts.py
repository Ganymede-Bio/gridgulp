"""Prompt construction for Gemma-first table reasoning."""

from __future__ import annotations


class GemmaPromptBuilder:
    """Build prompts for Gemma-first table reasoning."""

    SYSTEM_PROMPT = """You are an expert spreadsheet table detector.
Return only valid JSON.

Use this exact response shape:
{
  "analysis_summary": {
    "sheet_complexity": "simple|moderate|complex",
    "reasoning_notes": "short explanation",
    "confidence": 0.0
  },
  "decisions": [
    {
      "action": "keep|merge|split|add|reject",
      "proposal_ids": ["heuristic ids when applicable"],
      "range": "A1:C10",
      "suggested_name": "optional_table_name",
      "confidence": 0.0,
      "reason": "short explanation"
    }
  ]
}

Rules:
- Use proposal_ids from the prompt only when referring to heuristic proposals.
- keep: exactly one proposal id.
- merge: two or more proposal ids into one output range.
- split: one or more proposal ids, one output range per split decision.
- add: no proposal ids; create a brand new range you found from context.
- reject: proposal_ids required, no range needed.
- Use exact Excel ranges.
- Prefer coherent table boundaries over preserving heuristic output.
- Suggested names should be short, stable, and descriptive.
"""

    def build(
        self,
        formatted_context: str,
        image_notes: list[str] | None = None,
    ) -> tuple[str, str]:
        """Return system and user prompts."""
        del image_notes
        user_prompt = (
            "Analyze the spreadsheet context below and decide the final tables.\n\n"
            f"{formatted_context}\n\n"
            "Return only valid JSON."
        )
        return self.SYSTEM_PROMPT, user_prompt


class GemmaMultimodalPromptBuilder(GemmaPromptBuilder):
    """Build prompts for Gemma multimodal fallback reasoning."""

    SYSTEM_PROMPT = """You are an expert spreadsheet table detector working with both structured text and rendered spreadsheet images.
Return only valid JSON.

Use this exact response shape:
{
  "analysis_summary": {
    "sheet_complexity": "simple|moderate|complex",
    "reasoning_notes": "short explanation",
    "confidence": 0.0
  },
  "decisions": [
    {
      "action": "keep|merge|split|add|reject",
      "proposal_ids": ["heuristic ids when applicable"],
      "range": "A1:C10",
      "suggested_name": "optional_table_name",
      "confidence": 0.0,
      "reason": "short explanation"
    }
  ]
}

Rules:
- Use the structured text for exact cell addresses, ranges, and header text.
- Use the images to confirm visual grouping, spacer rows/columns, merged headers, and side-by-side block separation.
- Use proposal_ids from the prompt only when referring to heuristic proposals.
- keep: exactly one proposal id.
- merge: two or more proposal ids into one output range.
- split: one or more proposal ids, one output range per split decision.
- add: no proposal ids; create a brand new range you found from context.
- reject: proposal_ids required, no range needed.
- Use exact Excel ranges.
- Prefer coherent visual table boundaries over preserving heuristic output.
- Suggested names should be short, stable, and descriptive.
"""

    def build(
        self,
        formatted_context: str,
        image_notes: list[str] | None = None,
    ) -> tuple[str, str]:
        """Return system and user prompts for multimodal reasoning."""
        images_section = (
            "\n".join(f"- {note}" for note in image_notes) if image_notes else "- No images"
        )
        user_prompt = (
            "Analyze the spreadsheet context below and decide the final tables.\n\n"
            "Use the attached images only as layout evidence. Ground final ranges in the structured text.\n\n"
            "## Attached Images\n"
            f"{images_section}\n\n"
            f"{formatted_context}\n\n"
            "Return only valid JSON."
        )
        return self.SYSTEM_PROMPT, user_prompt
