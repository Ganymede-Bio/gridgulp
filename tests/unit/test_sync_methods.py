"""Tests for synchronous convenience methods in GridGulp."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gridgulp import GridGulp
from gridgulp.models import (
    DetectionResult,
    EntitySchema,
    ExtractionOrientation,
    FileInfo,
    FileType,
    FieldSchema,
    RecordGroup,
    SheetResult,
    SheetSchemaResult,
    TableInfo,
    TableRange,
    WorkbookSchemaResult,
)


@pytest.fixture
def mock_detection_result():
    """Create a mock DetectionResult for testing."""
    return DetectionResult(
        file_info=FileInfo(
            path=Path("test.xlsx"),
            type=FileType.XLSX,
            size=1024,
        ),
        sheets=[
            SheetResult(
                name="Sheet1",
                tables=[
                    TableInfo(
                        id="table_1",
                        range=TableRange(start_row=0, start_col=0, end_row=9, end_col=3),
                        confidence=0.95,
                        detection_method="simple_case",
                    )
                ],
                processing_time=0.1,
            )
        ],
        detection_time=0.5,
        methods_used=["simple_case"],
        metadata={"total_tables": 1},
    )


@pytest.fixture
def mock_schema_result() -> WorkbookSchemaResult:
    """Create a mock WorkbookSchemaResult for testing."""
    return WorkbookSchemaResult(
        file_info=FileInfo(path=Path("test.xlsx"), type=FileType.XLSX, size=1024),
        workbook_metadata={"file_format": "xlsx"},
        sheet_results=[SheetSchemaResult(sheet_name="Sheet1", entity_ids=["table_1"])],
        entities=[
            EntitySchema(
                id="table_1",
                suggested_name="Table 1",
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
                        id="table_1::records",
                        entity_id="table_1",
                        orientation=ExtractionOrientation.ROW_RECORDS,
                        record_count=1,
                        records=[{"Name": "Alice"}],
                        confidence=0.9,
                    )
                ],
                record_count_estimate=1,
                confidence=0.9,
            )
        ],
        relationships=[],
        warnings=[],
        provenance_summary={},
        schematization_time=0.4,
        methods_used=["workbook_schematizer"],
    )


class TestSyncMethods:
    """Test synchronous convenience methods."""

    def test_detect_tables_sync_no_event_loop(self, mock_detection_result):
        """Test detect_tables_sync when no event loop is running."""
        gg = GridGulp()

        # Mock the async detect_tables method
        gg.detect_tables = AsyncMock(return_value=mock_detection_result)

        # Call sync method
        result = gg.detect_tables_sync("test.xlsx")

        # Verify result
        assert result == mock_detection_result
        assert result.total_tables == 1
        gg.detect_tables.assert_called_once_with("test.xlsx")

    def test_detect_tables_sync_with_event_loop(self, mock_detection_result):
        """Test detect_tables_sync when event loop is already running (like in Jupyter)."""
        gg = GridGulp()

        # Mock the async detect_tables method
        gg.detect_tables = AsyncMock(return_value=mock_detection_result)

        # Simulate running event loop by mocking get_running_loop
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.return_value = Mock()  # Simulate existing loop

            # Call sync method
            result = gg.detect_tables_sync("test.xlsx")

            # Verify result
            assert result == mock_detection_result
            assert result.total_tables == 1

    def test_batch_detect_sync_no_event_loop(self, mock_detection_result):
        """Test batch_detect_sync when no event loop is running."""
        gg = GridGulp()

        # Mock the async batch_detect method
        gg.batch_detect = AsyncMock(return_value=[mock_detection_result, mock_detection_result])

        # Call sync method
        results = gg.batch_detect_sync(["test1.xlsx", "test2.xlsx"])

        # Verify results
        assert len(results) == 2
        assert all(r == mock_detection_result for r in results)
        gg.batch_detect.assert_called_once_with(["test1.xlsx", "test2.xlsx"])

    def test_schematize_sync_no_event_loop(self, mock_schema_result):
        """Test schematize_sync when no event loop is running."""
        gg = GridGulp()
        gg.schematize = AsyncMock(return_value=mock_schema_result)

        result = gg.schematize_sync("test.xlsx")

        assert result == mock_schema_result
        gg.schematize.assert_called_once_with("test.xlsx")

    def test_batch_detect_sync_with_event_loop(self, mock_detection_result):
        """Test batch_detect_sync when event loop is already running."""
        gg = GridGulp()

        # Mock the async batch_detect method
        gg.batch_detect = AsyncMock(return_value=[mock_detection_result])

        # Simulate running event loop
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.return_value = Mock()

            # Call sync method
            results = gg.batch_detect_sync(["test.xlsx"])

            # Verify results
            assert len(results) == 1
            assert results[0] == mock_detection_result

    def test_extract_from_file_deprecation(self, mock_detection_result):
        """Test that extract_from_file shows deprecation warning."""
        gg = GridGulp()

        # Mock detect_tables_sync
        gg.detect_tables_sync = Mock(return_value=mock_detection_result)

        # Call deprecated method and check for warning
        with pytest.warns(DeprecationWarning, match="extract_from_file is deprecated"):
            result = gg.extract_from_file("test.xlsx")

        # Verify it still works
        assert result == mock_detection_result
        gg.detect_tables_sync.assert_called_once_with("test.xlsx")

    def test_sync_method_error_handling(self):
        """Test that sync methods properly propagate errors."""
        gg = GridGulp()

        # Mock detect_tables to raise an error
        error_msg = "File not found"
        gg.detect_tables = AsyncMock(side_effect=FileNotFoundError(error_msg))

        # Verify error is propagated
        with pytest.raises(FileNotFoundError, match=error_msg):
            gg.detect_tables_sync("nonexistent.xlsx")

    def test_sync_method_with_path_object(self, mock_detection_result):
        """Test sync methods work with Path objects."""
        gg = GridGulp()

        # Mock the async method
        gg.detect_tables = AsyncMock(return_value=mock_detection_result)

        # Call with Path object
        test_path = Path("test.xlsx")
        result = gg.detect_tables_sync(test_path)

        # Verify
        assert result == mock_detection_result
        gg.detect_tables.assert_called_once_with(test_path)
