"""Tests for the GridGulp CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from gridgulp.cli import main
from gridgulp.config import VisionMode
from gridgulp.models import (
    DetectionResult,
    EntitySchema,
    ExtractionOrientation,
    FileInfo,
    FileType,
    FieldSchema,
    ProvenanceRef,
    RecordGroup,
    RegionRole,
    RegionSchema,
    SheetResult,
    SheetSchemaResult,
    TableInfo,
    TableRange,
    WorkbookSchemaResult,
)


def _sample_detection_result() -> DetectionResult:
    return DetectionResult(
        file_info=FileInfo(path=Path("sample.xlsx"), type=FileType.XLSX, size=10),
        sheets=[
            SheetResult(
                name="Sheet1",
                tables=[
                    TableInfo(
                        id="table_1",
                        range=TableRange.from_excel("A1:B2"),
                        confidence=0.95,
                        detection_method="gemma_keep",
                        suggested_name="sample_table",
                    )
                ],
                processing_time=0.1,
                metadata={"methods_used": ["gemma_first"]},
            )
        ],
        detection_time=0.2,
        methods_used=["file_reading", "gemma_first"],
        metadata={"detection_strategy": "heuristic_only"},
    )


def _sample_schema_result() -> WorkbookSchemaResult:
    return WorkbookSchemaResult(
        file_info=FileInfo(path=Path("sample.xlsx"), type=FileType.XLSX, size=10),
        workbook_metadata={"file_format": "xlsx", "sheet_count": 1},
        sheet_results=[
            SheetSchemaResult(
                sheet_name="Sheet1",
                regions=[
                    RegionSchema(
                        id="Sheet1::table_1",
                        sheet_name="Sheet1",
                        range=TableRange.from_excel("A1:B2"),
                        role=RegionRole.DATA_TABLE,
                        confidence=0.9,
                        entity_id="sample_table",
                    )
                ],
                entity_ids=["sample_table"],
            )
        ],
        entities=[
            EntitySchema(
                id="sample_table",
                suggested_name="Sample Table",
                source_regions=["Sheet1::table_1"],
                sheet_names=["Sheet1"],
                fields=[
                    FieldSchema(
                        id="name",
                        display_name="Name",
                        semantic_type="name",
                        physical_type="string",
                        confidence=0.9,
                    )
                ],
                extraction_orientation=ExtractionOrientation.ROW_RECORDS,
                record_groups=[
                    RecordGroup(
                        id="sample_table::records",
                        entity_id="sample_table",
                        orientation=ExtractionOrientation.ROW_RECORDS,
                        record_count=1,
                        records=[{"Name": "Alice"}],
                        confidence=0.9,
                        provenance=[ProvenanceRef(sheet_name="Sheet1", confidence=0.9)],
                    )
                ],
                record_count_estimate=1,
                confidence=0.9,
            )
        ],
        relationships=[],
        warnings=[],
        provenance_summary={"sheet_count": 1},
        schematization_time=0.2,
        methods_used=["workbook_schematizer"],
    )


def test_cli_detect_json(capsys, tmp_path: Path) -> None:
    """`gridgulp detect --json` should emit JSON for file paths."""
    target = tmp_path / "sample.xlsx"
    target.write_text("placeholder")

    with patch("gridgulp.cli.GridGulp") as mock_gridgulp:
        mock_gridgulp.return_value.detect_tables_sync.return_value = _sample_detection_result()

        exit_code = main(["--strategy", "heuristic_only", "detect", str(target), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["total_tables"] == 1
    assert payload["methods_used"] == ["file_reading", "gemma_first"]


def test_cli_detect_directory(capsys, tmp_path: Path) -> None:
    """Directory detection should call the directory sync API and print summaries."""
    target_dir = tmp_path / "reports"
    target_dir.mkdir()

    with patch("gridgulp.cli.GridGulp") as mock_gridgulp:
        mock_gridgulp.return_value.detect_tables_in_directory_sync.return_value = {
            target_dir / "a.xlsx": _sample_detection_result()
        }

        exit_code = main(["--strategy", "heuristic_only", "detect", str(target_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "a.xlsx: 1 tables" in captured.out


def test_cli_schematize_json(capsys, tmp_path: Path) -> None:
    """`gridgulp schematize --json` should emit schema JSON for file paths."""
    target = tmp_path / "sample.xlsx"
    target.write_text("placeholder")

    with patch("gridgulp.cli.GridGulp") as mock_gridgulp:
        mock_gridgulp.return_value.schematize_sync.return_value = _sample_schema_result()

        exit_code = main(["--strategy", "heuristic_only", "schematize", str(target), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["entities"][0]["id"] == "sample_table"
    assert payload["sheet_results"][0]["entity_ids"] == ["sample_table"]


def test_cli_doctor_success(capsys) -> None:
    """Doctor should validate Gemma config and endpoint reachability."""
    fake_client = Mock()
    fake_client.list_models = AsyncMock(return_value=["gemma-4-local"])
    fake_client.probe_multimodal_support = AsyncMock(return_value='{"ok":true}')

    with patch("gridgulp.cli.GemmaClient", return_value=fake_client):
        exit_code = main(
            [
                "--strategy",
                "gemma_first",
                "--gemma-base-url",
                "http://localhost:8080/v1",
                "--gemma-model-id",
                "gemma-4-local",
                "doctor",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "config: ok" in captured.out
    assert "model: ok" in captured.out
    assert "multimodal: ok" in captured.out


def test_cli_detect_debug_images_override(capsys, tmp_path: Path) -> None:
    """Debug image output should be plumbed into the GridGulp config."""
    target = tmp_path / "sample.xlsx"
    debug_dir = tmp_path / "debug"
    target.write_text("placeholder")

    with patch("gridgulp.cli.GridGulp") as mock_gridgulp:
        mock_gridgulp.return_value.detect_tables_sync.return_value = _sample_detection_result()

        exit_code = main(
            [
                "--strategy",
                "gemma_first",
                "--gemma-base-url",
                "http://localhost:11434/v1",
                "--gemma-model-id",
                "gemma4:26b",
                "--vision-mode",
                "fallback_only",
                "detect",
                str(target),
                "--save-debug-images",
                str(debug_dir),
                "--json",
            ]
        )

    _ = capsys.readouterr()
    config = mock_gridgulp.call_args.kwargs["config"]
    assert exit_code == 0
    assert config.vision.debug_image_dir == debug_dir
    assert config.vision.mode == VisionMode.FALLBACK_ONLY
