"""Tests for extraction result models."""

import json
from datetime import datetime

import pytest

from gridgulp.models.extraction_result import (
    ExtractedTable,
    FileExtractionResult,
    SheetExtractionResult,
)
from gridgulp.models.table import TableRange


def test_extracted_table_creation():
    """Test creating an ExtractedTable instance."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=10, end_col=5)

    table = ExtractedTable(
        range=cell_range,
        detection_confidence=0.95,
        detection_method="island_detection",
        extraction_status="success",
        quality_score=0.88,
        has_headers=True,
        header_rows=1,
        headers=["A", "B", "C", "D", "E"],
        data_rows=10,
        data_columns=5,
        data_density=0.92,
        orientation="vertical",
    )

    assert table.range == cell_range
    assert table.extraction_status == "success"
    assert table.quality_score == 0.88
    assert table.headers == ["A", "B", "C", "D", "E"]


def test_extracted_table_with_dataframe():
    """Test ExtractedTable with DataFrame data."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)

    df_dict = {
        "Name": ["John", "Jane"],
        "Age": [25, 30],
    }

    table = ExtractedTable(
        range=cell_range,
        detection_confidence=0.9,
        detection_method="simple",
        extraction_status="success",
        dataframe_dict=df_dict,
    )

    assert table.dataframe_dict == df_dict

    # Test JSON serialization
    json_data = table.model_dump(mode="json")
    assert isinstance(json_data["dataframe_dict"], dict)


def test_extracted_table_failed_status():
    """Test ExtractedTable with failed extraction."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=5, end_col=5)

    table = ExtractedTable(
        range=cell_range,
        detection_confidence=0.3,
        detection_method="island_detection",
        extraction_status="failed",
        error_message="Could not detect valid headers",
    )

    assert table.extraction_status == "failed"
    assert table.error_message == "Could not detect valid headers"
    assert table.quality_score == 0.0


def test_sheet_extraction_result():
    """Test SheetExtractionResult model."""
    sheet_result = SheetExtractionResult(
        sheet_name="Sheet1",
        total_tables_detected=3,
        tables_extracted=2,
        tables_failed=1,
    )

    assert sheet_result.sheet_name == "Sheet1"
    assert sheet_result.total_tables_detected == 3
    assert sheet_result.tables_extracted == 2
    assert sheet_result.tables_failed == 1
    assert sheet_result.success_rate == 2 / 3

    # Add extracted tables
    table1 = ExtractedTable(
        range=TableRange(start_row=0, start_col=0, end_row=5, end_col=5),
        detection_confidence=0.9,
        detection_method="simple",
        extraction_status="success",
        quality_score=0.85,
    )

    table2 = ExtractedTable(
        range=TableRange(start_row=10, start_col=0, end_row=15, end_col=5),
        detection_confidence=0.8,
        detection_method="island",
        extraction_status="failed",
    )

    sheet_result.extracted_tables = [table1, table2]
    assert len(sheet_result.extracted_tables) == 2


def test_file_extraction_result():
    """Test FileExtractionResult model."""
    result = FileExtractionResult(
        file_path="/path/to/file.xlsx",
        file_type="xlsx",
        timestamp=datetime.now().isoformat(),
        total_sheets=2,
        total_tables_detected=5,
        total_tables_extracted=4,
        total_tables_failed=1,
        detection_time=1.5,
        extraction_time=2.3,
    )

    assert result.file_path == "/path/to/file.xlsx"
    assert result.total_tables_detected == 5
    assert result.total_tables_extracted == 4
    assert result.overall_success_rate == 0.8

    # Test high quality tables property
    sheet1 = SheetExtractionResult(
        sheet_name="Sheet1",
        total_tables_detected=3,
    )

    # Add tables with different quality scores
    sheet1.extracted_tables = [
        ExtractedTable(
            range=TableRange(start_row=0, start_col=0, end_row=5, end_col=5),
            detection_confidence=0.9,
            detection_method="simple",
            extraction_status="success",
            quality_score=0.85,  # High quality
            dataframe_dict={"col1": [1, 2, 3], "col2": [4, 5, 6]},  # Add dataframe for is_valid
        ),
        ExtractedTable(
            range=TableRange(start_row=10, start_col=0, end_row=15, end_col=5),
            detection_confidence=0.8,
            detection_method="island",
            extraction_status="success",
            quality_score=0.65,  # Low quality
            dataframe_dict={"col1": [7, 8, 9], "col2": [10, 11, 12]},  # Add dataframe for is_valid
        ),
    ]

    result.sheets = [sheet1]
    high_quality = result.high_quality_tables
    assert len(high_quality) == 1
    assert high_quality[0].quality_score == 0.85


def test_model_validation():
    """Test model validation."""
    # Test invalid quality score
    with pytest.raises(ValueError):
        ExtractedTable(
            range=TableRange(start_row=0, start_col=0, end_row=5, end_col=5),
            detection_confidence=1.5,  # Invalid: > 1.0
            detection_method="simple",
            extraction_status="success",
        )

    # Test invalid data density
    with pytest.raises(ValueError):
        ExtractedTable(
            range=TableRange(start_row=0, start_col=0, end_row=5, end_col=5),
            detection_confidence=0.9,
            detection_method="simple",
            extraction_status="success",
            data_density=1.5,  # Invalid: > 1.0
        )


def test_json_serialization():
    """Test JSON serialization of models."""
    # Create a complete extraction result
    result = FileExtractionResult(
        file_path="/test/file.xlsx",
        file_type="xlsx",
        timestamp=datetime.now().isoformat(),
        total_sheets=1,
        total_tables_detected=1,
        total_tables_extracted=1,
        total_tables_failed=0,
        detection_time=0.5,
        extraction_time=1.0,
    )

    sheet = SheetExtractionResult(
        sheet_name="Data",
        total_tables_detected=1,
        tables_extracted=1,
        tables_failed=0,
    )

    table = ExtractedTable(
        range=TableRange(start_row=0, start_col=0, end_row=10, end_col=5),
        detection_confidence=0.95,
        detection_method="simple",
        extraction_status="success",
        quality_score=0.9,
        has_headers=True,
        headers=["A", "B", "C", "D", "E"],
        dataframe_dict={"A": [1, 2, 3], "B": [4, 5, 6]},
    )

    sheet.extracted_tables = [table]
    result.sheets = [sheet]

    # Serialize to JSON
    json_data = result.model_dump(mode="json")
    json_str = json.dumps(json_data)

    # Should be able to serialize without errors
    assert isinstance(json_str, str)

    # Check structure
    loaded = json.loads(json_str)
    assert loaded["file_path"] == "/test/file.xlsx"
    assert len(loaded["sheets"]) == 1
    assert len(loaded["sheets"][0]["extracted_tables"]) == 1


def test_properties():
    """Test computed properties."""
    # Test SheetExtractionResult success_rate
    sheet = SheetExtractionResult(
        sheet_name="Test",
        total_tables_detected=10,
        tables_extracted=7,
        tables_failed=3,
    )
    assert sheet.success_rate == 0.7

    # Test with no tables
    empty_sheet = SheetExtractionResult(
        sheet_name="Empty",
        total_tables_detected=0,
    )
    assert empty_sheet.success_rate == 0.0

    # Test FileExtractionResult overall_success_rate
    file_result = FileExtractionResult(
        file_path="/test.xlsx",
        file_type="xlsx",
        timestamp=datetime.now().isoformat(),
        total_sheets=1,
        total_tables_detected=0,
        total_tables_extracted=0,
        total_tables_failed=0,
    )
    assert file_result.overall_success_rate == 0.0
