"""Tests for multi-row header detection."""

import pytest
from unittest.mock import MagicMock

from gridgulp.models.table import TableRange
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.detectors.multi_header_detector import (
    MultiHeaderDetector,
    MultiRowHeader,
    HeaderCell,
)
from gridgulp.detectors.merged_cell_analyzer import MergedCell


class TestMultiHeaderDetector:
    """Test cases for MultiHeaderDetector."""

    @pytest.fixture
    def detector(self):
        """Create a MultiHeaderDetector instance."""
        return MultiHeaderDetector()

    @pytest.fixture
    def sample_sheet_data(self):
        """Create sample sheet data with multi-row headers."""
        sheet = SheetData(name="Test Sheet")

        # Row 0: Department headers (merged cells)
        sheet.set_cell(
            0,
            0,
            CellData(value="Department", is_bold=True, is_merged=True, merge_range="A1:A2"),
        )
        sheet.set_cell(
            0,
            1,
            CellData(value="Sales", is_bold=True, is_merged=True, merge_range="B1:C1"),
        )
        sheet.set_cell(
            0,
            3,
            CellData(value="Support", is_bold=True, is_merged=True, merge_range="D1:E1"),
        )

        # Row 1: Sub-headers
        sheet.set_cell(1, 1, CellData(value="Q1", is_bold=True))
        sheet.set_cell(1, 2, CellData(value="Q2", is_bold=True))
        sheet.set_cell(1, 3, CellData(value="Q1", is_bold=True))
        sheet.set_cell(1, 4, CellData(value="Q2", is_bold=True))

        # Data rows
        sheet.set_cell(2, 0, CellData(value="John Doe", data_type="string"))
        sheet.set_cell(2, 1, CellData(value=100, data_type="number"))
        sheet.set_cell(2, 2, CellData(value=150, data_type="number"))
        sheet.set_cell(2, 3, CellData(value=50, data_type="number"))
        sheet.set_cell(2, 4, CellData(value=75, data_type="number"))

        sheet.set_cell(3, 0, CellData(value="Jane Smith", data_type="string"))
        sheet.set_cell(3, 1, CellData(value=120, data_type="number"))
        sheet.set_cell(3, 2, CellData(value=130, data_type="number"))
        sheet.set_cell(3, 3, CellData(value=60, data_type="number"))
        sheet.set_cell(3, 4, CellData(value=80, data_type="number"))

        return sheet

    @pytest.fixture
    def simple_sheet_data(self):
        """Create sheet data with single-row headers."""
        sheet = SheetData(name="Simple Sheet")

        # Single header row
        sheet.set_cell(0, 0, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Age", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="City", is_bold=True))

        # Data rows
        sheet.set_cell(1, 0, CellData(value="Alice", data_type="string"))
        sheet.set_cell(1, 1, CellData(value=25, data_type="number"))
        sheet.set_cell(1, 2, CellData(value="New York", data_type="string"))

        return sheet

    def test_detect_multi_row_headers(self, detector, sample_sheet_data):
        """Test detection of multi-row headers."""
        table_range = TableRange(start_row=0, start_col=0, end_row=3, end_col=4)

        result = detector.detect_multi_row_headers(sample_sheet_data, table_range)

        assert result is not None
        assert isinstance(result, MultiRowHeader)
        assert result.start_row == 0
        assert result.end_row == 1  # Two header rows (0 and 1)
        assert len(result.cells) > 0
        assert result.confidence > 0.5

    def test_no_multi_row_headers(self, detector, simple_sheet_data):
        """Test that single-row headers are not detected as multi-row."""
        table_range = TableRange(start_row=0, start_col=0, end_row=1, end_col=2)

        result = detector.detect_multi_row_headers(simple_sheet_data, table_range)

        assert result is None  # No multi-row headers detected

    def test_column_mappings(self, detector, sample_sheet_data):
        """Test that column mappings are built correctly."""
        table_range = TableRange(start_row=0, start_col=0, end_row=3, end_col=4)

        result = detector.detect_multi_row_headers(sample_sheet_data, table_range)

        assert result is not None
        assert result.column_mappings is not None

        # Check specific column mappings
        assert 0 in result.column_mappings  # Department column
        assert 1 in result.column_mappings  # Sales Q1
        assert 2 in result.column_mappings  # Sales Q2

        # Check hierarchy
        assert "Sales" in result.column_mappings[1]
        assert "Q1" in result.column_mappings[1]

    def test_estimate_header_rows(self, detector, sample_sheet_data):
        """Test header row estimation."""
        table_range = TableRange(start_row=0, start_col=0, end_row=3, end_col=4)

        # Create merged cells for testing
        merged_cells = [
            MergedCell(
                start_row=0,
                start_col=1,
                end_row=0,
                end_col=2,
                value="Sales",
                is_header=True,
            ),
            MergedCell(
                start_row=0,
                start_col=3,
                end_row=0,
                end_col=4,
                value="Support",
                is_header=True,
            ),
        ]

        header_rows = detector._estimate_header_rows_from_sheet(
            sample_sheet_data, table_range, merged_cells
        )

        assert header_rows >= 2  # Should detect at least 2 header rows

    def test_header_cell_extraction(self, detector, sample_sheet_data):
        """Test extraction of header cells."""
        table_range = TableRange(start_row=0, start_col=0, end_row=3, end_col=4)

        # Mock merged cells
        merged_cells = [
            MergedCell(
                start_row=0,
                start_col=1,
                end_row=0,
                end_col=2,
                value="Sales",
                is_header=True,
            )
        ]

        header_cells = detector._extract_header_cells_from_sheet(
            sample_sheet_data, 2, table_range, merged_cells
        )

        assert len(header_cells) > 0

        # Check for merged cell
        merged_header_cells = [cell for cell in header_cells if cell.is_merged]
        assert len(merged_header_cells) > 0

        # Check for regular cells
        regular_cells = [cell for cell in header_cells if not cell.is_merged]
        assert len(regular_cells) > 0

    def test_confidence_calculation(self, detector):
        """Test confidence score calculation."""
        header_cells = [
            HeaderCell(row=0, col=0, value="Department", is_merged=True, col_span=1, row_span=2),
            HeaderCell(row=0, col=1, value="Sales", is_merged=True, col_span=2, row_span=1),
            HeaderCell(row=1, col=1, value="Q1"),
            HeaderCell(row=1, col=2, value="Q2"),
        ]

        column_mappings = {
            0: ["Department"],
            1: ["Sales", "Q1"],
            2: ["Sales", "Q2"],
        }

        sheet_data = MagicMock()
        merged_cells = [cell for cell in header_cells if cell.is_merged]

        confidence = detector._calculate_confidence_from_sheet(
            header_cells, column_mappings, sheet_data, merged_cells
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have reasonable confidence

    def test_format_boundary_detection(self, detector):
        """Test detection of formatting boundaries between rows."""
        sheet = SheetData(name="Format Test")

        # Header row with bold formatting
        sheet.set_cell(0, 0, CellData(value="Header", is_bold=True, background_color="#CCCCCC"))
        sheet.set_cell(0, 1, CellData(value="Header2", is_bold=True, background_color="#CCCCCC"))

        # Data row without special formatting
        sheet.set_cell(1, 0, CellData(value="Data", is_bold=False))
        sheet.set_cell(1, 1, CellData(value="Data2", is_bold=False))

        table_range = TableRange(start_row=0, start_col=0, end_row=1, end_col=1)

        has_boundary = detector._has_format_boundary_in_sheet(sheet, table_range, 0, 1)

        assert has_boundary  # Should detect formatting change

    def test_empty_sheet(self, detector):
        """Test behavior with empty sheet."""
        empty_sheet = SheetData(name="Empty")
        table_range = TableRange(start_row=0, start_col=0, end_row=0, end_col=0)

        result = detector.detect_multi_row_headers(empty_sheet, table_range)

        assert result is None  # Should handle empty sheet gracefully
