"""Tests for the ExcelMetadataExtractor class."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from gridgulp.detectors.excel_metadata_extractor import ExcelMetadataExtractor
from gridgulp.models.file_info import FileInfo, FileType
from gridgulp.models.table import TableRange


@pytest.fixture
def extractor():
    """Create an ExcelMetadataExtractor instance."""
    return ExcelMetadataExtractor()


@pytest.fixture
def mock_workbook_with_tables():
    """Create a mock workbook with Excel tables (ListObjects)."""
    workbook = Mock()

    # Mock worksheet with tables
    worksheet = Mock()
    worksheet.title = "Sheet1"

    # Mock table (ListObject)
    table1 = Mock()
    table1.name = "SalesTable"
    table1.ref = "A1:D10"
    table1.displayName = "Sales Data"
    table1.headerRowCount = 1
    table1.totalsRowCount = 0

    table2 = Mock()
    table2.name = "EmployeeTable"
    table2.ref = "F1:I20"
    table2.displayName = "Employee Records"
    table2.headerRowCount = 1
    table2.totalsRowCount = 1

    # Set up worksheet tables
    worksheet.tables.values.return_value = [table1, table2]

    # Mock named ranges
    workbook.defined_names = Mock()

    # Create named range mocks
    named_range1 = Mock()
    named_range1.name = "SalesData"
    named_range1.attr_text = "Sheet1!$A$1:$D$10"
    named_range1.localSheetId = None

    named_range2 = Mock()
    named_range2.name = "TotalSales"
    named_range2.attr_text = "Sheet1!$D$11"
    named_range2.localSheetId = None

    workbook.defined_names.definedName = [named_range1, named_range2]

    # Set up workbook worksheets
    workbook.worksheets = [worksheet]
    workbook.sheetnames = ["Sheet1"]

    return workbook


@pytest.fixture
def mock_workbook_no_metadata():
    """Create a mock workbook without any metadata."""
    workbook = Mock()

    worksheet = Mock()
    worksheet.title = "Sheet1"
    worksheet.tables.values.return_value = []

    workbook.worksheets = [worksheet]
    workbook.sheetnames = ["Sheet1"]
    workbook.defined_names = Mock()
    workbook.defined_names.definedName = []

    return workbook


@pytest.fixture
def file_info():
    """Create FileInfo for Excel file."""
    return FileInfo(path=Path("test.xlsx"), type=FileType.XLSX, size=1024)


class TestExcelMetadataExtractor:
    """Test the ExcelMetadataExtractor class."""

    def test_extract_tables_from_workbook(self, extractor, mock_workbook_with_tables):
        """Test extraction of Excel tables (ListObjects)."""
        result = extractor.extract_from_workbook(mock_workbook_with_tables)

        assert result is not None
        assert len(result.tables) == 2
        assert result.confidence > 0.9  # High confidence for native tables

        # Check first table
        table1 = result.tables[0]
        assert table1.id == "excel_table_SalesTable"
        assert table1.suggested_name == "Sales Data"
        assert table1.detection_method == "excel_metadata"
        assert table1.range.excel_range == "A1:D10"
        assert table1.metadata["table_name"] == "SalesTable"
        assert table1.metadata["has_headers"] is True
        assert table1.metadata["has_totals"] is False

        # Check second table
        table2 = result.tables[1]
        assert table2.id == "excel_table_EmployeeTable"
        assert table2.suggested_name == "Employee Records"
        assert table2.range.excel_range == "F1:I20"
        assert table2.metadata["has_totals"] is True

    def test_extract_named_ranges(self, extractor, mock_workbook_with_tables):
        """Test extraction of named ranges."""
        result = extractor.extract_from_workbook(mock_workbook_with_tables)

        # Named ranges should be included in metadata
        assert "named_ranges" in result.metadata
        named_ranges = result.metadata["named_ranges"]

        assert len(named_ranges) == 2
        assert any(nr["name"] == "SalesData" for nr in named_ranges)
        assert any(nr["name"] == "TotalSales" for nr in named_ranges)

    def test_parse_excel_range(self, extractor):
        """Test parsing of Excel range notation."""
        # Standard range
        range1 = extractor._parse_excel_range("A1:D10")
        assert range1.start_row == 0
        assert range1.start_col == 0
        assert range1.end_row == 9
        assert range1.end_col == 3

        # Range with sheet name
        range2 = extractor._parse_excel_range("Sheet1!B2:E20")
        assert range2.start_row == 1
        assert range2.start_col == 1
        assert range2.end_row == 19
        assert range2.end_col == 4

        # Range with absolute references
        range3 = extractor._parse_excel_range("$A$1:$C$5")
        assert range3.start_row == 0
        assert range3.start_col == 0
        assert range3.end_row == 4
        assert range3.end_col == 2

        # Single cell
        range4 = extractor._parse_excel_range("B3")
        assert range4.start_row == 2
        assert range4.start_col == 1
        assert range4.end_row == 2
        assert range4.end_col == 1

    def test_no_metadata_workbook(self, extractor, mock_workbook_no_metadata):
        """Test extraction when workbook has no metadata."""
        result = extractor.extract_from_workbook(mock_workbook_no_metadata)

        assert result is not None
        assert len(result.tables) == 0
        assert result.confidence == 0  # No tables found
        assert result.metadata["named_ranges"] == []

    def test_extract_from_file_path(self, extractor, file_info, mock_workbook_with_tables):
        """Test extraction from file path."""
        with patch("openpyxl.load_workbook") as mock_load:
            mock_load.return_value = mock_workbook_with_tables

            result = extractor.extract_from_file(file_info.path, file_info)

            assert result is not None
            assert len(result.tables) == 2
            mock_load.assert_called_once_with(file_info.path, read_only=True, data_only=True)

    def test_file_load_error(self, extractor, file_info):
        """Test handling of file load errors."""
        with patch("openpyxl.load_workbook") as mock_load:
            mock_load.side_effect = Exception("Cannot load file")

            result = extractor.extract_from_file(file_info.path, file_info)

            # Should return None on error
            assert result is None

    def test_multiple_worksheets(self, extractor):
        """Test extraction from workbook with multiple worksheets."""
        workbook = Mock()

        # Sheet 1 with one table
        sheet1 = Mock()
        sheet1.title = "Data"
        table1 = Mock()
        table1.name = "DataTable"
        table1.ref = "A1:C10"
        table1.displayName = "Main Data"
        table1.headerRowCount = 1
        table1.totalsRowCount = 0
        sheet1.tables.values.return_value = [table1]

        # Sheet 2 with two tables
        sheet2 = Mock()
        sheet2.title = "Summary"
        table2 = Mock()
        table2.name = "SummaryTable1"
        table2.ref = "A1:B5"
        table2.displayName = "Summary 1"
        table2.headerRowCount = 1
        table2.totalsRowCount = 0

        table3 = Mock()
        table3.name = "SummaryTable2"
        table3.ref = "D1:F8"
        table3.displayName = "Summary 2"
        table3.headerRowCount = 1
        table3.totalsRowCount = 0
        sheet2.tables.values.return_value = [table2, table3]

        workbook.worksheets = [sheet1, sheet2]
        workbook.sheetnames = ["Data", "Summary"]
        workbook.defined_names = Mock()
        workbook.defined_names.definedName = []

        result = extractor.extract_from_workbook(workbook)

        assert result is not None
        assert len(result.tables) == 3

        # Tables should include sheet information
        assert any(t.metadata.get("sheet_name") == "Data" for t in result.tables)
        assert sum(1 for t in result.tables if t.metadata.get("sheet_name") == "Summary") == 2

    def test_table_with_special_characters(self, extractor):
        """Test handling of tables with special characters in names."""
        workbook = Mock()
        worksheet = Mock()
        worksheet.title = "Sheet1"

        table = Mock()
        table.name = "Sales_Data_2024"
        table.ref = "A1:D10"
        table.displayName = "Sales Data (2024)"
        table.headerRowCount = 1
        table.totalsRowCount = 0

        worksheet.tables.values.return_value = [table]
        workbook.worksheets = [worksheet]
        workbook.sheetnames = ["Sheet1"]
        workbook.defined_names = Mock()
        workbook.defined_names.definedName = []

        result = extractor.extract_from_workbook(workbook)

        assert result is not None
        assert len(result.tables) == 1
        assert result.tables[0].suggested_name == "Sales Data (2024)"
        assert result.tables[0].id == "excel_table_Sales_Data_2024"

    def test_invalid_range_handling(self, extractor):
        """Test handling of invalid range references."""
        # Invalid range format
        result = extractor._parse_excel_range("InvalidRange")
        assert result is None

        # Empty range
        result = extractor._parse_excel_range("")
        assert result is None

        # Range with invalid characters
        result = extractor._parse_excel_range("A1:@#$")
        assert result is None
