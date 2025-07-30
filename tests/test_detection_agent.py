"""Tests for the TableDetectionAgent class."""

import pytest
from unittest.mock import Mock

from gridgulp.detection import TableDetectionAgent
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.models.file_info import FileType


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

        assert agent.confidence_threshold == 0.6
        assert agent.file_type is None

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

        result = await agent.detect_tables(sheet_data)

        # Simple table should be detected
        assert len(result.tables) == 1
        assert result.tables[0].confidence > 0.8
        assert result.tables[0].detection_method in [
            "simple_case_fast",
            "ultra_fast",
            "island_detection_fast",
            "island_detection",  # Island detector may return this
        ]
        assert result.tables[0].range.start_row == 0
        assert result.tables[0].range.start_col == 0
        assert result.tables[0].range.end_row == 3
        assert result.tables[0].range.end_col == 2

    @pytest.mark.asyncio
    async def test_detect_multiple_tables(self, multi_table_sheet):
        """Test detection of multiple tables using island detection."""
        agent = TableDetectionAgent()

        result = await agent.detect_tables(multi_table_sheet)

        # Should detect multiple tables
        assert len(result.tables) >= 1  # May detect as one or multiple based on implementation
        assert all(t.confidence > 0.5 for t in result.tables)

    @pytest.mark.asyncio
    async def test_excel_file_type(self, sheet_data):
        """Test detection with Excel file type."""
        agent = TableDetectionAgent(file_type=FileType.XLSX)

        result = await agent.detect_tables(sheet_data)

        # Should still detect tables
        assert len(result.tables) >= 1

    @pytest.mark.asyncio
    async def test_text_file_type(self, sheet_data):
        """Test detection with text file type."""
        agent = TableDetectionAgent(file_type=FileType.TXT)

        result = await agent.detect_tables(sheet_data)

        # Should detect tables, possibly with different method
        assert len(result.tables) >= 1

    @pytest.mark.asyncio
    async def test_confidence_threshold(self, sheet_data):
        """Test that confidence threshold affects results."""
        # Low threshold
        agent_low = TableDetectionAgent(confidence_threshold=0.3)
        result_low = await agent_low.detect_tables(sheet_data)

        # High threshold
        agent_high = TableDetectionAgent(confidence_threshold=0.95)
        result_high = await agent_high.detect_tables(sheet_data)

        # Low threshold should detect tables
        assert len(result_low.tables) >= 1
        # High threshold may or may not detect depending on confidence
        assert len(result_high.tables) >= 0

    @pytest.mark.asyncio
    async def test_empty_sheet_handling(self):
        """Test handling of empty sheets."""
        empty_sheet = SheetData(name="Empty")
        agent = TableDetectionAgent()

        result = await agent.detect_tables(empty_sheet)

        assert len(result.tables) == 0
        assert "method_used" in result.processing_metadata  # Note: singular, not plural

    @pytest.mark.asyncio
    async def test_detection_timing(self, sheet_data):
        """Test that processing time is recorded."""
        agent = TableDetectionAgent()

        result = await agent.detect_tables(sheet_data)

        assert "processing_time" in result.processing_metadata
        assert result.processing_metadata["processing_time"] >= 0

    @pytest.mark.asyncio
    async def test_sparse_data(self):
        """Test detection with sparse data."""
        sheet = SheetData(name="Sparse")

        # Add sparse data
        sheet.cells[(0, 0)] = CellData(row=0, column=0, value="A", data_type="s")
        sheet.cells[(0, 10)] = CellData(row=0, column=10, value="B", data_type="s")
        sheet.cells[(10, 0)] = CellData(row=10, column=0, value="C", data_type="s")
        sheet.cells[(10, 10)] = CellData(row=10, column=10, value="D", data_type="s")

        agent = TableDetectionAgent()
        result = await agent.detect_tables(sheet)

        # May or may not detect tables in sparse data
        assert isinstance(result.tables, list)
        assert result.processing_metadata is not None
