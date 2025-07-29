"""Tests for the TableDetectionAgent class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from gridgulp.detection import TableDetectionAgent
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.models.file_info import FileType
from gridgulp.models.table import TableInfo, TableRange


@pytest.fixture
def sheet_data():
    """Create basic sheet data for testing."""
    sheet = SheetData(name="TestSheet")

    # Simple table data
    data = [
        ["Name", "Age", "City"],
        ["Alice", 25, "NYC"],
        ["Bob", 30, "LA"],
        ["Charlie", 35, "Chicago"],
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    return sheet


@pytest.fixture
def multi_table_sheet():
    """Create sheet with multiple tables."""
    sheet = SheetData(name="MultiTableSheet")

    # Table 1: A1:C3
    table1_data = [
        ["Product", "Price", "Stock"],
        ["Apple", 1.5, 100],
        ["Banana", 0.8, 150],
    ]

    # Table 2: E5:G7
    table2_data = [
        ["Employee", "Dept", "Salary"],
        ["John", "Sales", 50000],
        ["Jane", "IT", 60000],
    ]

    # Add tables to sheet
    for row_idx, row in enumerate(table1_data):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    for row_idx, row in enumerate(table2_data):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx + 4, col_idx + 4)] = CellData(
                row=row_idx + 4,
                column=col_idx + 4,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    return sheet


class TestTableDetectionAgent:
    """Test the TableDetectionAgent class."""

    @pytest.mark.asyncio
    async def test_init_default(self):
        """Test agent initialization with defaults."""
        agent = TableDetectionAgent()

        assert agent.confidence_threshold == 0.7
        assert agent.file_type == FileType.UNKNOWN

    @pytest.mark.asyncio
    async def test_init_with_params(self):
        """Test agent initialization with custom parameters."""
        agent = TableDetectionAgent(confidence_threshold=0.9, file_type=FileType.XLSX)

        assert agent.confidence_threshold == 0.9
        assert agent.file_type == FileType.XLSX

    @pytest.mark.asyncio
    async def test_detect_simple_table(self, sheet_data):
        """Test detection of simple single table."""
        agent = TableDetectionAgent()

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = [
                TableInfo(
                    id="simple_1",
                    range=TableRange(start_row=0, start_col=0, end_row=3, end_col=2),
                    confidence=0.95,
                    detection_method="simple_case",
                    headers=["Name", "Age", "City"],
                )
            ]

            result = await agent.detect_tables(sheet_data)

            assert len(result.tables) == 1
            assert result.tables[0].confidence == 0.95
            assert result.tables[0].detection_method == "simple_case"
            assert "simple_case" in result.processing_metadata["methods_used"]

    @pytest.mark.asyncio
    async def test_detect_multiple_tables(self, multi_table_sheet):
        """Test detection of multiple tables using island detection."""
        agent = TableDetectionAgent()

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = []  # No single table

            with patch.object(agent, "_run_island_detection") as mock_island:
                mock_island.return_value = [
                    TableInfo(
                        id="island_1",
                        range=TableRange(start_row=0, start_col=0, end_row=2, end_col=2),
                        confidence=0.85,
                        detection_method="island",
                    ),
                    TableInfo(
                        id="island_2",
                        range=TableRange(start_row=4, start_col=4, end_row=6, end_col=6),
                        confidence=0.85,
                        detection_method="island",
                    ),
                ]

                result = await agent.detect_tables(multi_table_sheet)

                assert len(result.tables) == 2
                assert all(t.detection_method == "island" for t in result.tables)
                assert "island" in result.processing_metadata["methods_used"]

    @pytest.mark.asyncio
    async def test_excel_metadata_detection(self, sheet_data):
        """Test Excel metadata detection for XLSX files."""
        agent = TableDetectionAgent(file_type=FileType.XLSX)

        with patch.object(agent, "_run_excel_metadata_extraction") as mock_excel:
            mock_excel.return_value = [
                TableInfo(
                    id="excel_table_1",
                    range=TableRange(start_row=0, start_col=0, end_row=3, end_col=2),
                    confidence=1.0,
                    detection_method="excel_metadata",
                    suggested_name="DataTable",
                )
            ]

            # Mock sheet to have excel_metadata
            sheet_data.metadata = {"has_excel_tables": True}

            result = await agent.detect_tables(sheet_data)

            assert len(result.tables) == 1
            assert result.tables[0].detection_method == "excel_metadata"
            assert result.tables[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_structured_text_detection(self, sheet_data):
        """Test structured text detection for text files."""
        agent = TableDetectionAgent(file_type=FileType.TXT)

        with patch.object(agent, "_run_structured_text_detection") as mock_text:
            mock_text.return_value = [
                TableInfo(
                    id="text_1",
                    range=TableRange(start_row=0, start_col=0, end_row=3, end_col=2),
                    confidence=0.8,
                    detection_method="structured_text",
                )
            ]

            result = await agent.detect_tables(sheet_data)

            assert len(result.tables) == 1
            assert result.tables[0].detection_method == "structured_text"

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, sheet_data):
        """Test that low confidence tables are filtered out."""
        agent = TableDetectionAgent(confidence_threshold=0.8)

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = [
                TableInfo(
                    id="high_conf",
                    range=TableRange(start_row=0, start_col=0, end_row=1, end_col=1),
                    confidence=0.9,
                    detection_method="simple_case",
                ),
                TableInfo(
                    id="low_conf",
                    range=TableRange(start_row=2, start_col=0, end_row=3, end_col=1),
                    confidence=0.6,  # Below threshold
                    detection_method="simple_case",
                ),
            ]

            result = await agent.detect_tables(sheet_data)

            assert len(result.tables) == 1
            assert result.tables[0].id == "high_conf"
            assert "filtered_count" in result.processing_metadata
            assert result.processing_metadata["filtered_count"] == 1

    @pytest.mark.asyncio
    async def test_method_selection_by_file_type(self):
        """Test that detection methods are chosen based on file type."""
        sheet = SheetData(name="Test")

        # Excel file should try Excel metadata first
        agent_xlsx = TableDetectionAgent(file_type=FileType.XLSX)
        with patch.object(agent_xlsx, "_run_excel_metadata_extraction") as mock_excel:
            with patch.object(agent_xlsx, "_run_simple_case_detection") as mock_simple:
                mock_excel.return_value = []
                mock_simple.return_value = []

                await agent_xlsx.detect_tables(sheet)

                mock_excel.assert_called_once()

        # Text file should use structured text detection
        agent_txt = TableDetectionAgent(file_type=FileType.TXT)
        with patch.object(agent_txt, "_run_structured_text_detection") as mock_text:
            mock_text.return_value = []

            await agent_txt.detect_tables(sheet)

            mock_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_sheet_handling(self):
        """Test handling of empty sheets."""
        empty_sheet = SheetData(name="Empty")
        agent = TableDetectionAgent()

        result = await agent.detect_tables(empty_sheet)

        assert len(result.tables) == 0
        assert result.processing_metadata["methods_used"] == []

    @pytest.mark.asyncio
    async def test_detection_timing(self, sheet_data):
        """Test that processing time is recorded."""
        agent = TableDetectionAgent()

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = [Mock(spec=TableInfo, confidence=0.9)]

            result = await agent.detect_tables(sheet_data)

            assert "total_time" in result.processing_metadata
            assert result.processing_metadata["total_time"] > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, sheet_data):
        """Test graceful error handling in detection methods."""
        agent = TableDetectionAgent()

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.side_effect = Exception("Detection error")

            # Should not raise, but return empty results
            result = await agent.detect_tables(sheet_data)

            assert len(result.tables) == 0
            assert "errors" in result.processing_metadata

    @pytest.mark.asyncio
    async def test_fallback_detection_strategy(self, sheet_data):
        """Test fallback when primary detection methods fail."""
        agent = TableDetectionAgent(file_type=FileType.XLSX)

        with patch.object(agent, "_run_excel_metadata_extraction") as mock_excel:
            with patch.object(agent, "_run_simple_case_detection") as mock_simple:
                with patch.object(agent, "_run_island_detection") as mock_island:
                    # Excel metadata fails
                    mock_excel.return_value = []
                    # Simple case fails
                    mock_simple.return_value = []
                    # Island detection finds tables
                    mock_island.return_value = [Mock(spec=TableInfo, confidence=0.8)]

                    result = await agent.detect_tables(sheet_data)

                    assert len(result.tables) == 1
                    # Should have tried all methods
                    assert len(result.processing_metadata["methods_used"]) >= 2

    @pytest.mark.asyncio
    async def test_header_enhancement(self, sheet_data):
        """Test that headers are enhanced after detection."""
        agent = TableDetectionAgent()

        table = TableInfo(
            id="test",
            range=TableRange(start_row=0, start_col=0, end_row=3, end_col=2),
            confidence=0.9,
            detection_method="simple_case",
            headers=None,  # No headers initially
        )

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = [table]

            with patch.object(agent, "_enhance_table_headers") as mock_enhance:
                await agent.detect_tables(sheet_data)

                # Should attempt to enhance headers
                mock_enhance.assert_called()

    @pytest.mark.asyncio
    async def test_deduplication(self, sheet_data):
        """Test deduplication of overlapping tables."""
        agent = TableDetectionAgent()

        # Create overlapping tables
        table1 = TableInfo(
            id="table1",
            range=TableRange(start_row=0, start_col=0, end_row=3, end_col=2),
            confidence=0.9,
            detection_method="simple_case",
        )

        table2 = TableInfo(
            id="table2",
            range=TableRange(start_row=1, start_col=0, end_row=3, end_col=2),
            confidence=0.8,
            detection_method="island",
        )

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            with patch.object(agent, "_run_island_detection") as mock_island:
                mock_simple.return_value = [table1]
                mock_island.return_value = [table2]

                result = await agent.detect_tables(sheet_data)

                # Should keep higher confidence table
                assert len(result.tables) == 1
                assert result.tables[0].id == "table1"

    @pytest.mark.asyncio
    async def test_custom_detection_params(self):
        """Test passing custom parameters to detection methods."""
        agent = TableDetectionAgent()

        # Test with custom params
        custom_params = {"min_table_size": 10, "max_gap_size": 2}

        with patch.object(agent, "_run_simple_case_detection") as mock_simple:
            mock_simple.return_value = []

            sheet = SheetData(name="Test")
            await agent.detect_tables(sheet, **custom_params)

            # Detection method should receive custom params
            # (Implementation would need to support this)
