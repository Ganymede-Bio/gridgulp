"""Tests for the ExcelMetadataExtractor class."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from gridgulp.detectors.excel_metadata_extractor import (
    ExcelMetadataExtractor,
    ExcelMetadata,
    ExcelTableInfo,
    NamedRangeInfo,
    PrintAreaInfo,
)
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
    table1.ref = "A1:D10"
    table1.displayName = "Sales Data"
    table1.headerRowCount = 1
    table1.totalsRowCount = 0
    table_style_info = Mock()
    table_style_info.name = "TableStyleMedium2"
    table1.tableStyleInfo = table_style_info

    table2 = Mock()
    table2.ref = "F1:I20"
    table2.displayName = "Employee Records"
    table2.headerRowCount = 1
    table2.totalsRowCount = 1
    table2.tableStyleInfo = None

    # Set up worksheet tables as dict
    worksheet.tables = {"SalesTable": table1, "EmployeeTable": table2}

    # Mock named ranges
    workbook.defined_names = Mock()

    # Create named range mocks
    named_range1 = Mock()
    named_range1.name = "SalesData"
    named_range1.value = "Sheet1!$A$1:$D$10"
    named_range1.localSheetId = None
    named_range1.hidden = False
    named_range1.comment = None

    named_range2 = Mock()
    named_range2.name = "TotalSales"
    named_range2.value = "Sheet1!$D$11"
    named_range2.localSheetId = None
    named_range2.hidden = False
    named_range2.comment = None

    # Filter out built-in names
    named_range3 = Mock()
    named_range3.name = "_xlnm.Print_Area"
    named_range3.value = "Sheet1!$A$1:$Z$100"

    workbook.defined_names.definedName = [named_range1, named_range2, named_range3]

    # Set up workbook worksheets
    workbook.__getitem__ = Mock(return_value=worksheet)
    workbook.sheetnames = ["Sheet1"]

    # Mock for print areas and pivot tables
    worksheet.print_area = None
    worksheet.print_title_rows = None
    worksheet.print_title_cols = None
    worksheet._pivots = []  # No pivot tables

    return workbook


@pytest.fixture
def mock_workbook_no_metadata():
    """Create a mock workbook without any metadata."""
    workbook = Mock()

    worksheet = Mock()
    worksheet.title = "Sheet1"
    worksheet.tables = {}  # Empty dict for no tables
    worksheet.print_area = None
    worksheet.print_title_rows = None
    worksheet.print_title_cols = None
    worksheet._pivots = []  # No pivot tables

    workbook.__getitem__ = Mock(return_value=worksheet)
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
        metadata = extractor.extract_metadata_openpyxl(mock_workbook_with_tables)

        assert metadata is not None
        assert len(metadata.list_objects) == 2

        # Check first table
        table1 = metadata.list_objects[0]
        assert table1.name == "SalesTable"
        assert table1.display_name == "Sales Data"
        assert table1.range_address == "Sheet1!A1:D10"
        assert table1.has_headers is True
        assert table1.has_totals is False
        assert table1.table_style == "TableStyleMedium2"

        # Check second table
        table2 = metadata.list_objects[1]
        assert table2.name == "EmployeeTable"
        assert table2.display_name == "Employee Records"
        assert table2.range_address == "Sheet1!F1:I20"
        assert table2.has_totals is True

    def test_extract_named_ranges(self, extractor, mock_workbook_with_tables):
        """Test extraction of named ranges."""
        metadata = extractor.extract_metadata_openpyxl(mock_workbook_with_tables)

        # Should have filtered out built-in names
        assert len(metadata.named_ranges) == 2
        assert any(nr.name == "SalesData" for nr in metadata.named_ranges)
        assert any(nr.name == "TotalSales" for nr in metadata.named_ranges)

        # Should not include built-in names
        assert not any(nr.name.startswith("_xlnm.") for nr in metadata.named_ranges)

    def test_convert_to_detection_hints(self, extractor, mock_workbook_with_tables):
        """Test conversion of metadata to detection hints."""
        metadata = extractor.extract_metadata_openpyxl(mock_workbook_with_tables)
        hints = extractor.convert_to_detection_hints(metadata)

        assert len(hints) >= 2  # At least the two Excel tables

        # Check Excel table hints
        table_hints = [h for h in hints if h["source"] == "excel_table"]
        assert len(table_hints) == 2

        # First table hint
        hint1 = next(h for h in table_hints if h["name"] == "Sales Data")
        assert hint1["range"] == "Sheet1!A1:D10"
        assert hint1["confidence"] == 0.95  # High confidence for native tables
        assert hint1["has_headers"] is True

        # Check named range hints
        # Only ranges (not single cells) are included as hints
        named_range_hints = [h for h in hints if h["source"] == "named_range"]
        assert len(named_range_hints) == 1  # Only SalesData (range), not TotalSales (single cell)

    def test_no_metadata_workbook(self, extractor, mock_workbook_no_metadata):
        """Test extraction when workbook has no metadata."""
        metadata = extractor.extract_metadata_openpyxl(mock_workbook_no_metadata)

        assert metadata is not None
        assert len(metadata.list_objects) == 0
        assert len(metadata.named_ranges) == 0
        assert len(metadata.print_areas) == 0
        assert metadata.has_pivot_tables is False

    def test_is_likely_data_range(self, extractor):
        """Test the _is_likely_data_range method."""
        # Should return True for ranges
        assert extractor._is_likely_data_range("A1:D10") is True
        assert extractor._is_likely_data_range("Sheet1!B2:E20") is True
        assert extractor._is_likely_data_range("$A$1:$C$5") is True

        # Should return False for single cells
        assert extractor._is_likely_data_range("A1") is False
        assert extractor._is_likely_data_range("Sheet1!B2") is False

    def test_xlrd_metadata_extraction(self, extractor):
        """Test metadata extraction from xlrd workbook."""
        # Mock xlrd workbook
        workbook = Mock()

        # Mock named ranges (xlrd style)
        name_obj = Mock()
        name_obj.formula_text = "Sheet1!$A$1:$D$10"
        name_obj.hidden = 0

        workbook.name_map = {"DataRange": [name_obj]}

        metadata = extractor.extract_metadata_xlrd(workbook)

        assert metadata is not None
        assert len(metadata.named_ranges) == 1
        assert metadata.named_ranges[0].name == "DataRange"
        assert metadata.named_ranges[0].refers_to == "Sheet1!$A$1:$D$10"
        assert metadata.named_ranges[0].scope == "Workbook"

    def test_multiple_worksheets(self, extractor):
        """Test extraction from workbook with multiple worksheets."""
        workbook = Mock()

        # Sheet 1 with one table
        sheet1 = Mock()
        sheet1.title = "Data"
        table1 = Mock()
        table1.ref = "A1:C10"
        table1.displayName = "Main Data"
        table1.headerRowCount = 1
        table1.totalsRowCount = 0
        table1.tableStyleInfo = None
        sheet1.tables = {"DataTable": table1}
        sheet1.print_area = None
        sheet1.print_title_rows = None
        sheet1.print_title_cols = None

        # Sheet 2 with two tables
        sheet2 = Mock()
        sheet2.title = "Summary"
        table2 = Mock()
        table2.ref = "A1:B5"
        table2.displayName = "Summary 1"
        table2.headerRowCount = 1
        table2.totalsRowCount = 0
        table2.tableStyleInfo = None

        table3 = Mock()
        table3.ref = "D1:F8"
        table3.displayName = "Summary 2"
        table3.headerRowCount = 1
        table3.totalsRowCount = 0
        table3.tableStyleInfo = None
        sheet2.tables = {"SummaryTable1": table2, "SummaryTable2": table3}
        sheet2.print_area = None
        sheet2.print_title_rows = None
        sheet2.print_title_cols = None

        workbook.sheetnames = ["Data", "Summary"]
        workbook.__getitem__ = Mock(side_effect=lambda name: sheet1 if name == "Data" else sheet2)
        workbook.defined_names = Mock()
        workbook.defined_names.definedName = []

        metadata = extractor.extract_metadata_openpyxl(workbook)

        assert metadata is not None
        assert len(metadata.list_objects) == 3

        # Tables should include sheet information in range_address
        assert any("Data!" in t.range_address for t in metadata.list_objects)
        assert sum(1 for t in metadata.list_objects if "Summary!" in t.range_address) == 2

    def test_table_with_special_characters(self, extractor):
        """Test handling of tables with special characters in names."""
        workbook = Mock()
        worksheet = Mock()
        worksheet.title = "Sheet1"

        table = Mock()
        table.ref = "A1:D10"
        table.displayName = "Sales Data (2024)"
        table.headerRowCount = 1
        table.totalsRowCount = 0
        table.tableStyleInfo = None

        worksheet.tables = {"Sales_Data_2024": table}
        worksheet.print_area = None
        worksheet.print_title_rows = None
        worksheet.print_title_cols = None

        workbook.__getitem__ = Mock(return_value=worksheet)
        workbook.sheetnames = ["Sheet1"]
        workbook.defined_names = Mock()
        workbook.defined_names.definedName = []

        metadata = extractor.extract_metadata_openpyxl(workbook)

        assert metadata is not None
        assert len(metadata.list_objects) == 1
        assert metadata.list_objects[0].display_name == "Sales Data (2024)"
        assert metadata.list_objects[0].name == "Sales_Data_2024"

    def test_print_areas_extraction(self, extractor):
        """Test extraction of print areas."""
        workbook = Mock()
        worksheet = Mock()
        worksheet.title = "Sheet1"
        worksheet.tables = {}
        worksheet.print_area = "A1:Z100"
        worksheet.print_title_rows = "1:2"
        worksheet.print_title_cols = "A:B"

        workbook.__getitem__ = Mock(return_value=worksheet)
        workbook.sheetnames = ["Sheet1"]
        workbook.defined_names = Mock()
        workbook.defined_names.definedName = []

        metadata = extractor.extract_metadata_openpyxl(workbook)

        assert len(metadata.print_areas) == 1
        print_area = metadata.print_areas[0]
        assert print_area.sheet_name == "Sheet1"
        assert print_area.print_area == "A1:Z100"
        assert print_area.print_titles_rows == "1:2"
        assert print_area.print_titles_cols == "A:B"
