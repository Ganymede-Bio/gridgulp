"""Tests for the MergedCellAnalyzer class."""

import pytest
from unittest.mock import Mock, MagicMock

from gridgulp.detectors.merged_cell_analyzer import MergedCellAnalyzer
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.models.table import TableRange


@pytest.fixture
def analyzer():
    """Create a MergedCellAnalyzer instance."""
    return MergedCellAnalyzer()


@pytest.fixture
def sheet_with_merged_cells():
    """Create sheet data with merged cells."""
    sheet = SheetData(name="MergedSheet")

    # Regular data
    data = [
        ["Company Report", None, None, None],  # A1:D1 merged
        ["Q1", "Q2", "Q3", "Q4"],
        ["Sales", None, "Marketing", None],  # B3:C3 and D3:E3 merged
        [100, 150, 200, 250],
        [300, 350, 400, 450],
    ]

    # Add regular cells
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if value is not None:
                sheet.cells[(row_idx, col_idx)] = CellData(
                    row=row_idx,
                    column=col_idx,
                    value=value,
                    data_type="s" if isinstance(value, str) else "n",
                )

    # Define merged ranges
    sheet.merged_ranges = [
        TableRange(start_row=0, start_col=0, end_row=0, end_col=3),  # A1:D1
        TableRange(start_row=2, start_col=1, end_row=2, end_col=2),  # B3:C3
        TableRange(start_row=2, start_col=3, end_row=2, end_col=4),  # D3:E3 (extends beyond data)
    ]

    return sheet


@pytest.fixture
def sheet_with_vertical_merges():
    """Create sheet with vertically merged cells."""
    sheet = SheetData(name="VerticalMerged")

    # Data with vertical headers
    data = [
        ["Department", "Jan", "Feb", "Mar"],
        ["Sales", 100, 110, 120],
        [None, 200, 210, 220],  # Department cell merged down
        [None, 300, 310, 320],
        ["Marketing", 150, 160, 170],
        [None, 250, 260, 270],  # Marketing cell merged down
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if value is not None:
                sheet.cells[(row_idx, col_idx)] = CellData(
                    row=row_idx,
                    column=col_idx,
                    value=value,
                    data_type="s" if isinstance(value, str) else "n",
                )

    # Vertical merges
    sheet.merged_ranges = [
        TableRange(start_row=1, start_col=0, end_row=3, end_col=0),  # A2:A4
        TableRange(start_row=4, start_col=0, end_row=5, end_col=0),  # A5:A6
    ]

    return sheet


@pytest.fixture
def sheet_no_merged_cells():
    """Create sheet without merged cells."""
    sheet = SheetData(name="NoMerged")

    data = [
        ["Name", "Age", "City"],
        ["Alice", 25, "NYC"],
        ["Bob", 30, "LA"],
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    sheet.merged_ranges = []
    return sheet


class TestMergedCellAnalyzer:
    """Test the MergedCellAnalyzer class."""

    def test_analyze_with_merged_cells(self, analyzer, sheet_with_merged_cells):
        """Test analysis of sheet with merged cells."""
        result = analyzer.analyze_merged_cells(sheet_with_merged_cells)

        assert result is not None
        assert result.has_merged_cells is True
        assert len(result.merged_ranges) == 3
        assert result.confidence > 0

        # Check metadata
        assert "total_merged_ranges" in result.metadata
        assert result.metadata["total_merged_ranges"] == 3
        assert "merged_cell_impact" in result.metadata

    def test_analyze_no_merged_cells(self, analyzer, sheet_no_merged_cells):
        """Test analysis of sheet without merged cells."""
        result = analyzer.analyze_merged_cells(sheet_no_merged_cells)

        assert result is not None
        assert result.has_merged_cells is False
        assert len(result.merged_ranges) == 0
        assert result.confidence == 1.0  # High confidence when no merges

    def test_get_unmerged_bounds(self, analyzer, sheet_with_merged_cells):
        """Test getting bounds with merged cells unmerged."""
        # Original table range
        table_range = TableRange(start_row=0, start_col=0, end_row=4, end_col=3)

        result = analyzer.analyze_merged_cells(sheet_with_merged_cells)
        unmerged_range = analyzer.get_unmerged_bounds(table_range, result)

        # Should be same as original since merges are within bounds
        assert unmerged_range.start_row == 0
        assert unmerged_range.start_col == 0
        assert unmerged_range.end_row == 4
        assert unmerged_range.end_col == 3

    def test_merged_cells_affect_headers(self, analyzer, sheet_with_merged_cells):
        """Test detection of merged cells in header rows."""
        result = analyzer.analyze_merged_cells(sheet_with_merged_cells)

        # First row has merged cells (title)
        assert any(mr.start_row == 0 for mr in result.merged_ranges)

        # Should detect this affects headers
        if "affects_headers" in result.metadata:
            assert result.metadata["affects_headers"] is True

    def test_vertical_merged_cells(self, analyzer, sheet_with_vertical_merges):
        """Test handling of vertically merged cells."""
        result = analyzer.analyze_merged_cells(sheet_with_vertical_merges)

        assert result is not None
        assert result.has_merged_cells is True
        assert len(result.merged_ranges) == 2

        # Check that vertical merges are detected
        for mr in result.merged_ranges:
            assert mr.start_col == mr.end_col  # Same column
            assert mr.end_row > mr.start_row  # Multiple rows

    def test_is_cell_merged(self, analyzer):
        """Test checking if specific cell is part of merged range."""
        merged_ranges = [
            TableRange(start_row=0, start_col=0, end_row=0, end_col=3),  # A1:D1
            TableRange(start_row=2, start_col=1, end_row=3, end_col=2),  # B3:C4
        ]

        # Test cells in merged ranges
        assert analyzer.is_cell_merged(0, 1, merged_ranges) is True  # B1 is merged
        assert analyzer.is_cell_merged(0, 3, merged_ranges) is True  # D1 is merged
        assert analyzer.is_cell_merged(2, 2, merged_ranges) is True  # C3 is merged
        assert analyzer.is_cell_merged(3, 1, merged_ranges) is True  # B4 is merged

        # Test cells not in merged ranges
        assert analyzer.is_cell_merged(1, 0, merged_ranges) is False  # A2
        assert analyzer.is_cell_merged(4, 4, merged_ranges) is False  # E5

    def test_split_merged_range_horizontally(self, analyzer):
        """Test splitting a merged range into individual cells."""
        merged_range = TableRange(start_row=0, start_col=0, end_row=0, end_col=3)

        cells = analyzer.split_merged_range(merged_range)

        assert len(cells) == 4  # A1, B1, C1, D1
        assert all(c[0] == 0 for c in cells)  # All in row 0
        assert [c[1] for c in cells] == [0, 1, 2, 3]  # Columns 0-3

    def test_split_merged_range_vertically(self, analyzer):
        """Test splitting a vertical merged range."""
        merged_range = TableRange(start_row=1, start_col=0, end_row=3, end_col=0)

        cells = analyzer.split_merged_range(merged_range)

        assert len(cells) == 3  # A2, A3, A4
        assert all(c[1] == 0 for c in cells)  # All in column 0
        assert [c[0] for c in cells] == [1, 2, 3]  # Rows 1-3

    def test_split_merged_range_block(self, analyzer):
        """Test splitting a block merged range."""
        merged_range = TableRange(start_row=1, start_col=1, end_row=2, end_col=2)

        cells = analyzer.split_merged_range(merged_range)

        assert len(cells) == 4  # B2, C2, B3, C3
        expected = [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert sorted(cells) == sorted(expected)

    def test_merged_cells_at_table_boundary(self, analyzer):
        """Test merged cells that extend beyond table boundaries."""
        sheet = SheetData(name="BoundaryMerge")

        # Small table
        for i in range(3):
            for j in range(3):
                sheet.cells[(i, j)] = CellData(row=i, column=j, value=f"Cell{i}{j}", data_type="s")

        # Merged range that extends beyond table
        sheet.merged_ranges = [
            TableRange(start_row=2, start_col=2, end_row=3, end_col=4)  # C3:E4
        ]

        result = analyzer.analyze_merged_cells(sheet)

        # Should detect merge extends beyond data
        table_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)
        unmerged = analyzer.get_unmerged_bounds(table_range, result)

        # Unmerged bounds might be expanded if merge extends table
        assert unmerged is not None

    def test_complex_merged_pattern(self, analyzer):
        """Test complex pattern of merged cells."""
        sheet = SheetData(name="ComplexMerge")

        # Create a report-style layout with various merges
        sheet.merged_ranges = [
            TableRange(start_row=0, start_col=0, end_row=0, end_col=5),  # Title row
            TableRange(start_row=1, start_col=0, end_row=2, end_col=0),  # Row header
            TableRange(start_row=1, start_col=1, end_row=1, end_col=2),  # Column group 1
            TableRange(start_row=1, start_col=3, end_row=1, end_col=4),  # Column group 2
            TableRange(start_row=5, start_col=0, end_row=5, end_col=5),  # Footer row
        ]

        result = analyzer.analyze_merged_cells(sheet)

        assert result.has_merged_cells is True
        assert len(result.merged_ranges) == 5

        # Should identify this as a complex merge pattern
        if "merge_pattern" in result.metadata:
            assert result.metadata["merge_pattern"] == "complex"

    def test_empty_sheet_with_merged_ranges(self, analyzer):
        """Test sheet that has merged ranges but no data."""
        sheet = SheetData(name="EmptyMerged")
        sheet.cells = {}  # No data
        sheet.merged_ranges = [TableRange(start_row=0, start_col=0, end_row=0, end_col=3)]

        result = analyzer.analyze_merged_cells(sheet)

        assert result.has_merged_cells is True
        assert len(result.merged_ranges) == 1
        # Confidence might be lower due to no data
        assert result.confidence <= 1.0
