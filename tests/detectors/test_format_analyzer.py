"""Tests for semantic format analysis."""

import pytest

from gridgulp.models.table import TableRange
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.detectors.format_analyzer import (
    SemanticFormatAnalyzer,
    RowType,
    SemanticRow,
    TableStructure,
    FormatPattern,
)


class TestSemanticFormatAnalyzer:
    """Test cases for SemanticFormatAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a SemanticFormatAnalyzer instance."""
        return SemanticFormatAnalyzer()

    @pytest.fixture
    def financial_sheet_data(self):
        """Create sample financial data with subtotals and sections."""
        sheet = SheetData(name="Financial Report")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Category", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Q1", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Q2", is_bold=True))
        sheet.set_cell(0, 3, CellData(value="Q3", is_bold=True))
        sheet.set_cell(0, 4, CellData(value="Q4", is_bold=True))

        # Section 1: Revenue
        sheet.set_cell(1, 0, CellData(value="Revenue", is_bold=True, background_color="#E0E0E0"))
        sheet.set_cell(2, 0, CellData(value="Product Sales", indentation_level=1))
        sheet.set_cell(2, 1, CellData(value=1000, data_type="number"))
        sheet.set_cell(2, 2, CellData(value=1200, data_type="number"))
        sheet.set_cell(2, 3, CellData(value=1100, data_type="number"))
        sheet.set_cell(2, 4, CellData(value=1300, data_type="number"))

        sheet.set_cell(3, 0, CellData(value="Services", indentation_level=1))
        sheet.set_cell(3, 1, CellData(value=500, data_type="number"))
        sheet.set_cell(3, 2, CellData(value=600, data_type="number"))
        sheet.set_cell(3, 3, CellData(value=550, data_type="number"))
        sheet.set_cell(3, 4, CellData(value=700, data_type="number"))

        # Subtotal
        sheet.set_cell(4, 0, CellData(value="Total Revenue", is_bold=True))
        sheet.set_cell(4, 1, CellData(value=1500, data_type="number", is_bold=True))
        sheet.set_cell(4, 2, CellData(value=1800, data_type="number", is_bold=True))
        sheet.set_cell(4, 3, CellData(value=1650, data_type="number", is_bold=True))
        sheet.set_cell(4, 4, CellData(value=2000, data_type="number", is_bold=True))

        # Blank separator row
        sheet.set_cell(5, 0, CellData(value=None))

        # Section 2: Expenses
        sheet.set_cell(6, 0, CellData(value="Expenses", is_bold=True, background_color="#E0E0E0"))
        sheet.set_cell(7, 0, CellData(value="Salaries", indentation_level=1))
        sheet.set_cell(7, 1, CellData(value=800, data_type="number"))
        sheet.set_cell(7, 2, CellData(value=850, data_type="number"))

        return sheet

    @pytest.fixture
    def simple_table_data(self):
        """Create simple table without special formatting."""
        sheet = SheetData(name="Simple Table")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Score", is_bold=True))

        # Data
        sheet.set_cell(1, 0, CellData(value="Alice", data_type="string"))
        sheet.set_cell(1, 1, CellData(value=85, data_type="number"))
        sheet.set_cell(2, 0, CellData(value="Bob", data_type="string"))
        sheet.set_cell(2, 1, CellData(value=92, data_type="number"))

        return sheet

    def test_analyze_table_structure(self, analyzer, financial_sheet_data):
        """Test analysis of table structure with sections and subtotals."""
        table_range = TableRange(start_row=0, start_col=0, end_row=7, end_col=4)

        structure = analyzer.analyze_table_structure(
            financial_sheet_data, table_range, header_rows=1
        )

        assert isinstance(structure, TableStructure)
        assert len(structure.semantic_rows) == 8  # All rows analyzed
        assert structure.has_subtotals is True
        assert len(structure.sections) > 0
        assert len(structure.preserve_blank_rows) > 0  # Should preserve separator

    def test_row_type_detection(self, analyzer, financial_sheet_data):
        """Test detection of different row types."""
        table_range = TableRange(start_row=0, start_col=0, end_row=7, end_col=4)

        structure = analyzer.analyze_table_structure(
            financial_sheet_data, table_range, header_rows=1
        )

        # Check specific row types
        row_types = {row.row_index: row.row_type for row in structure.semantic_rows}

        assert row_types[0] == RowType.HEADER
        assert row_types[1] == RowType.SECTION_HEADER  # "Revenue"
        assert row_types[2] == RowType.DATA  # "Product Sales"
        assert row_types[4] == RowType.SUBTOTAL  # "Total Revenue"
        assert row_types[5] == RowType.BLANK  # Separator

    def test_blank_row_preservation(self, analyzer, financial_sheet_data):
        """Test identification of semantic blank rows."""
        table_range = TableRange(start_row=0, start_col=0, end_row=7, end_col=4)

        structure = analyzer.analyze_table_structure(
            financial_sheet_data, table_range, header_rows=1
        )

        # Should preserve blank row between sections
        assert 5 in structure.preserve_blank_rows

    def test_section_detection(self, analyzer, financial_sheet_data):
        """Test detection of logical sections."""
        table_range = TableRange(start_row=0, start_col=0, end_row=7, end_col=4)

        structure = analyzer.analyze_table_structure(
            financial_sheet_data, table_range, header_rows=1
        )

        assert len(structure.sections) >= 2  # Revenue and Expenses sections

        # Check section boundaries
        for start, end in structure.sections:
            assert start >= 0
            assert end >= start

    def test_format_pattern_detection(self, analyzer):
        """Test detection of formatting patterns."""
        sheet = SheetData(name="Alternating Colors")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Value", is_bold=True))

        # Alternating row colors
        colors = ["#FFFFFF", "#F0F0F0"]
        for i in range(1, 11):
            color = colors[i % 2]
            sheet.set_cell(i, 0, CellData(value=f"Item {i}", background_color=color))
            sheet.set_cell(i, 1, CellData(value=i * 10, data_type="number", background_color=color))

        table_range = TableRange(start_row=0, start_col=0, end_row=10, end_col=1)

        structure = analyzer.analyze_table_structure(sheet, table_range, header_rows=1)

        # Should detect alternating background pattern
        alternating_patterns = [
            p for p in structure.format_patterns if p.pattern_type == "alternating_background"
        ]
        assert len(alternating_patterns) > 0

    def test_subtotal_detection(self, analyzer):
        """Test detection of subtotal and total rows."""
        sheet = SheetData(name="Sales Report")

        # Data with subtotals
        sheet.set_cell(0, 0, CellData(value="Product", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Sales", is_bold=True))

        sheet.set_cell(1, 0, CellData(value="Product A"))
        sheet.set_cell(1, 1, CellData(value=100, data_type="number"))

        sheet.set_cell(2, 0, CellData(value="Product B"))
        sheet.set_cell(2, 1, CellData(value=150, data_type="number"))

        sheet.set_cell(3, 0, CellData(value="Subtotal", is_bold=True))
        sheet.set_cell(3, 1, CellData(value=250, data_type="number", is_bold=True))

        sheet.set_cell(4, 0, CellData(value="Grand Total", is_bold=True))
        sheet.set_cell(4, 1, CellData(value=250, data_type="number", is_bold=True))

        table_range = TableRange(start_row=0, start_col=0, end_row=4, end_col=1)

        structure = analyzer.analyze_table_structure(sheet, table_range, header_rows=1)

        assert structure.has_subtotals is True
        assert structure.has_grand_total is True

        # Check row types
        row_types = {row.row_index: row.row_type for row in structure.semantic_rows}
        assert row_types[3] == RowType.SUBTOTAL
        assert row_types[4] == RowType.TOTAL

    def test_column_formatting_detection(self, analyzer):
        """Test detection of consistent column formatting."""
        sheet = SheetData(name="Formatted Columns")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Amount", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Status", is_bold=True))

        # Data with consistent column formatting
        for i in range(1, 6):
            sheet.set_cell(i, 0, CellData(value=f"Item {i}", alignment="left"))
            sheet.set_cell(i, 1, CellData(value=i * 100, data_type="number", alignment="right"))
            sheet.set_cell(i, 2, CellData(value="Active", is_bold=True, alignment="center"))

        table_range = TableRange(start_row=0, start_col=0, end_row=5, end_col=2)

        structure = analyzer.analyze_table_structure(sheet, table_range, header_rows=1)

        # Should detect column alignment patterns
        alignment_patterns = [
            p for p in structure.format_patterns if p.pattern_type == "column_alignment"
        ]
        assert len(alignment_patterns) >= 2  # At least for amount and status columns

        # Should detect bold column
        bold_patterns = [p for p in structure.format_patterns if p.pattern_type == "column_bold"]
        assert len(bold_patterns) >= 1  # Status column

    def test_simple_table_analysis(self, analyzer, simple_table_data):
        """Test analysis of simple table without special formatting."""
        table_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=1)

        structure = analyzer.analyze_table_structure(simple_table_data, table_range, header_rows=1)

        assert structure.has_subtotals is False
        assert structure.has_grand_total is False
        assert len(structure.sections) == 0
        assert len(structure.preserve_blank_rows) == 0

        # All non-header rows should be data
        data_rows = [r for r in structure.semantic_rows if r.row_type == RowType.DATA]
        assert len(data_rows) == 2
