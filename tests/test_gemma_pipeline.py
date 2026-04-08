"""Tests for Gemma-first reasoning, fallback, and post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gridgulp.ai import GemmaTableReasoner
from gridgulp.config import Config
from gridgulp.models.sheet_data import CellData, SheetData
from gridgulp.models.table import TableInfo, TableRange


@pytest.fixture
def simple_reasoner_sheet() -> SheetData:
    """Create a simple one-table sheet that should stay text-only."""
    sheet = SheetData(name="SimpleSheet")
    values = {
        "A1": "Name",
        "B1": "Score",
        "A2": "Alice",
        "B2": 9,
        "A3": "Bob",
        "B3": 8,
    }
    for address, value in values.items():
        row = int("".join(char for char in address if char.isdigit())) - 1
        letters = "".join(char for char in address if char.isalpha())
        col = 0
        for char in letters:
            col = col * 26 + (ord(char) - ord("A") + 1)
        col -= 1
        sheet.cells[address] = CellData(
            value=value,
            data_type="number" if isinstance(value, int) else "text",
            row=row,
            column=col,
            is_bold=row == 0,
        )
    sheet.max_row = 2
    sheet.max_column = 1
    return sheet


@pytest.fixture
def adjacent_blocks_sheet() -> SheetData:
    """Create side-by-side blocks that should trigger vision fallback."""
    sheet = SheetData(name="AdjacentBlocks")
    values = {
        "A1": "Region",
        "B1": "Sales",
        "A2": "West",
        "B2": 10,
        "D1": "Owner",
        "E1": "Notes",
        "D2": "Ops",
        "E2": "Follow up",
    }
    for address, value in values.items():
        row = int("".join(char for char in address if char.isdigit())) - 1
        letters = "".join(char for char in address if char.isalpha())
        col = 0
        for char in letters:
            col = col * 26 + (ord(char) - ord("A") + 1)
        col -= 1
        sheet.cells[address] = CellData(
            value=value,
            data_type="number" if isinstance(value, int) else "text",
            row=row,
            column=col,
            is_bold=row == 0,
        )
    sheet.max_row = 1
    sheet.max_column = 4
    return sheet


def _reasoner_config(debug_dir: Path | None = None) -> Config:
    payload: dict[str, object] = {
        "detection_strategy": "gemma_first",
        "gemma": {
            "enabled": True,
            "base_url": "http://localhost:11434/v1",
            "model_id": "gemma4:26b",
        },
        "vision": {
            "enabled": True,
            "mode": "fallback_only",
        },
    }
    if debug_dir is not None:
        payload["vision"] = {
            "enabled": True,
            "mode": "fallback_only",
            "debug_image_dir": debug_dir,
        }
    return Config.model_validate(payload)


def test_gemma_reasoner_keeps_simple_sheet_text_only(simple_reasoner_sheet: SheetData) -> None:
    """Simple sheets with confident text reasoning should not invoke multimodal fallback."""
    proposals = [
        TableInfo(
            id="proposal_main",
            range=TableRange.from_excel("A1:B3"),
            confidence=0.95,
            detection_method="simple_case_fast",
            headers=["Name", "Score"],
        )
    ]
    reasoner = GemmaTableReasoner(_reasoner_config())
    reasoner.client.chat_completion = AsyncMock(
        return_value=json.dumps(
            {
                "analysis_summary": {
                    "sheet_complexity": "simple",
                    "reasoning_notes": "single table",
                    "confidence": 0.99,
                },
                "decisions": [
                    {
                        "action": "keep",
                        "proposal_ids": ["proposal_main"],
                        "range": "A1:B3",
                        "suggested_name": "score_table",
                        "confidence": 0.98,
                        "reason": "The sheet contains one contiguous table.",
                    }
                ],
            }
        )
    )
    reasoner.client.multimodal_chat_completion = AsyncMock()

    tables, metadata = asyncio_run(reasoner.reason_about_tables(simple_reasoner_sheet, proposals))

    assert len(tables) == 1
    assert metadata["final_reasoner"] == "text"
    assert metadata["vision_fallback_triggered"] is False
    reasoner.client.multimodal_chat_completion.assert_not_called()


def test_gemma_reasoner_uses_multimodal_fallback_for_adjacent_blocks(
    adjacent_blocks_sheet: SheetData,
) -> None:
    """Adjacent heuristic blocks should trigger multimodal fallback and replace the result."""
    proposals = [
        TableInfo(
            id="proposal_left",
            range=TableRange.from_excel("A1:B2"),
            confidence=0.88,
            detection_method="simple_case",
            headers=["Region", "Sales"],
        ),
        TableInfo(
            id="proposal_right",
            range=TableRange.from_excel("D1:E2"),
            confidence=0.83,
            detection_method="simple_case",
            headers=["Owner", "Notes"],
        ),
    ]
    reasoner = GemmaTableReasoner(_reasoner_config())
    reasoner.client.chat_completion = AsyncMock(
        return_value=json.dumps(
            {
                "analysis_summary": {
                    "sheet_complexity": "moderate",
                    "reasoning_notes": "ambiguous side-by-side blocks",
                    "confidence": 0.94,
                },
                "decisions": [
                    {
                        "action": "keep",
                        "proposal_ids": ["proposal_left"],
                        "range": "A1:B2",
                        "suggested_name": "left_block",
                        "confidence": 0.9,
                        "reason": "Left block is valid.",
                    }
                ],
            }
        )
    )
    reasoner.client.multimodal_chat_completion = AsyncMock(
        return_value=json.dumps(
            {
                "analysis_summary": {
                    "sheet_complexity": "moderate",
                    "reasoning_notes": "two visual tables separated by a spacer column",
                    "confidence": 0.97,
                },
                "decisions": [
                    {
                        "action": "keep",
                        "proposal_ids": ["proposal_left"],
                        "range": "A1:B2",
                        "suggested_name": "sales_table",
                        "confidence": 0.95,
                        "reason": "Left block is a standalone table.",
                    },
                    {
                        "action": "keep",
                        "proposal_ids": ["proposal_right"],
                        "range": "D1:E2",
                        "suggested_name": "notes_table",
                        "confidence": 0.94,
                        "reason": "Right block is visually separate.",
                    },
                ],
            }
        )
    )

    tables, metadata = asyncio_run(reasoner.reason_about_tables(adjacent_blocks_sheet, proposals))

    assert [table.range.excel_range for table in tables] == ["A1:B2", "D1:E2"]
    assert metadata["final_reasoner"] == "multimodal"
    assert metadata["vision_fallback_triggered"] is True
    assert metadata["vision_image_count"] == 3
    assert metadata["vision_trigger_reason"] == "adjacent_blocks"
    reasoner.client.multimodal_chat_completion.assert_awaited_once()


def test_gemma_reasoner_keeps_text_result_when_multimodal_fails(
    adjacent_blocks_sheet: SheetData,
    tmp_path: Path,
) -> None:
    """Multimodal errors should preserve the text result and capture the failure."""
    proposals = [
        TableInfo(
            id="proposal_left",
            range=TableRange.from_excel("A1:B2"),
            confidence=0.88,
            detection_method="simple_case",
            headers=["Region", "Sales"],
        ),
        TableInfo(
            id="proposal_right",
            range=TableRange.from_excel("D1:E2"),
            confidence=0.83,
            detection_method="simple_case",
            headers=["Owner", "Notes"],
        ),
    ]
    reasoner = GemmaTableReasoner(_reasoner_config(debug_dir=tmp_path))
    reasoner.client.chat_completion = AsyncMock(
        return_value=json.dumps(
            {
                "analysis_summary": {
                    "sheet_complexity": "moderate",
                    "reasoning_notes": "text path sees one block",
                    "confidence": 0.93,
                },
                "decisions": [
                    {
                        "action": "keep",
                        "proposal_ids": ["proposal_left"],
                        "range": "A1:B2",
                        "suggested_name": "sales_table",
                        "confidence": 0.91,
                        "reason": "The left block is a valid table.",
                    }
                ],
            }
        )
    )
    reasoner.client.multimodal_chat_completion = AsyncMock(
        side_effect=RuntimeError("vision timeout")
    )

    tables, metadata = asyncio_run(reasoner.reason_about_tables(adjacent_blocks_sheet, proposals))

    assert [table.range.excel_range for table in tables] == ["A1:B2"]
    assert metadata["final_reasoner"] == "text"
    assert metadata["vision_fallback_triggered"] is True
    assert "vision timeout" in metadata["vision_error"]
    assert len(metadata["vision_debug_images"]) == 3
    assert all(Path(path).exists() for path in metadata["vision_debug_images"])


def asyncio_run(awaitable):
    """Run an awaitable in tests without pulling in extra helpers."""
    import asyncio

    return asyncio.run(awaitable)
