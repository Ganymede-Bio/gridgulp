"""Strict parsing for Gemma table-detection responses."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GemmaDecisionAction(StrEnum):
    """Supported decision actions returned by Gemma."""

    KEEP = "keep"
    MERGE = "merge"
    SPLIT = "split"
    ADD = "add"
    REJECT = "reject"


class GemmaDecision(BaseModel):
    """One table-boundary decision returned by Gemma."""

    model_config = ConfigDict(strict=True)

    action: GemmaDecisionAction = Field(..., description="Decision type")
    proposal_ids: list[str] = Field(
        default_factory=list, description="Referenced heuristic proposals"
    )
    range: str | None = Field(None, description="Excel range for non-reject decisions")
    suggested_name: str | None = Field(None, description="Optional semantic table name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    reason: str = Field(..., min_length=1, description="Short rationale")

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, value: GemmaDecisionAction | str) -> GemmaDecisionAction:
        if isinstance(value, GemmaDecisionAction):
            return value
        return GemmaDecisionAction(value)

    @model_validator(mode="after")
    def _validate_action_shape(self) -> GemmaDecision:
        if self.action == GemmaDecisionAction.REJECT:
            if not self.proposal_ids:
                raise ValueError("reject decisions must reference at least one proposal id")
            return self

        if self.range is None:
            raise ValueError("non-reject decisions must include a range")

        if self.action == GemmaDecisionAction.KEEP and len(self.proposal_ids) != 1:
            raise ValueError("keep decisions must reference exactly one proposal id")
        if self.action == GemmaDecisionAction.MERGE and len(self.proposal_ids) < 2:
            raise ValueError("merge decisions must reference at least two proposal ids")
        if self.action == GemmaDecisionAction.SPLIT and not self.proposal_ids:
            raise ValueError("split decisions must reference at least one proposal id")
        if self.action == GemmaDecisionAction.ADD and self.proposal_ids:
            raise ValueError("add decisions must not reference proposal ids")

        return self


class GemmaAnalysisSummary(BaseModel):
    """Top-level summary returned by Gemma."""

    model_config = ConfigDict(strict=True)

    sheet_complexity: str = Field(..., min_length=1)
    reasoning_notes: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class GemmaReasoningResult(BaseModel):
    """Parsed Gemma response."""

    model_config = ConfigDict(strict=True)

    analysis_summary: GemmaAnalysisSummary
    decisions: list[GemmaDecision] = Field(default_factory=list)


class GemmaResponseParser:
    """Parse JSON-only responses from Gemma."""

    def parse(self, raw_response: str) -> GemmaReasoningResult:
        """Parse raw response text into a validated reasoning result."""
        json_payload = self._extract_json_payload(raw_response)
        try:
            payload = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemma response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Gemma response JSON must be an object.")

        return GemmaReasoningResult.model_validate(payload)

    def _extract_json_payload(self, raw_response: str) -> str:
        stripped = raw_response.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Gemma response did not contain a JSON object.")
        return stripped[start : end + 1]
