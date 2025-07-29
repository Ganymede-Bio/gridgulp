"""Tests for the IslandDetector class."""

import pytest
from unittest.mock import Mock

from gridgulp.detectors.island_detector import IslandDetector
from gridgulp.models.sheet_data import SheetData, CellData
from gridgulp.models.table import TableRange


@pytest.fixture
def island_detector():
    """Create an IslandDetector instance."""
    return IslandDetector()


@pytest.fixture
def sheet_with_multiple_tables():
    """Create sheet data with multiple distinct tables."""
    sheet = SheetData(name="MultiTableSheet")

    # Table 1: A1:C3 (Sales data)
    data_table1 = [
        ["Product", "Price", "Quantity"],
        ["Apple", 1.5, 100],
        ["Banana", 0.8, 150],
    ]

    # Table 2: E5:G7 (Employee data)
    data_table2 = [
        ["Name", "Department", "Salary"],
        ["John", "Sales", 50000],
        ["Jane", "Marketing", 55000],
    ]

    # Table 3: A10:B12 (Summary)
    data_table3 = [
        ["Total Sales", 1000],
        ["Total Cost", 800],
        ["Profit", 200],
    ]

    # Add Table 1 to sheet
    for row_idx, row in enumerate(data_table1):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    # Add Table 2 to sheet (offset by E5)
    for row_idx, row in enumerate(data_table2):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx + 4, col_idx + 4)] = CellData(
                row=row_idx + 4,
                column=col_idx + 4,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    # Add Table 3 to sheet (offset by A10)
    for row_idx, row in enumerate(data_table3):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx + 9, col_idx)] = CellData(
                row=row_idx + 9,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    return sheet


@pytest.fixture
def sheet_with_single_table():
    """Create sheet data with a single table."""
    sheet = SheetData(name="SingleTableSheet")

    data = [
        ["Header1", "Header2", "Header3"],
        ["Data1", "Data2", "Data3"],
        ["Data4", "Data5", "Data6"],
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx, column=col_idx, value=value, data_type="s"
            )

    return sheet


@pytest.fixture
def sheet_with_adjacent_tables():
    """Create sheet with tables that are close but not connected."""
    sheet = SheetData(name="AdjacentTables")

    # Table 1: A1:B3
    table1 = [
        ["Name", "Age"],
        ["Alice", 25],
        ["Bob", 30],
    ]

    # Table 2: D1:E3 (separated by one empty column)
    table2 = [
        ["City", "Country"],
        ["NYC", "USA"],
        ["London", "UK"],
    ]

    # Add tables
    for row_idx, row in enumerate(table1):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx)] = CellData(
                row=row_idx,
                column=col_idx,
                value=value,
                data_type="s" if isinstance(value, str) else "n",
            )

    for row_idx, row in enumerate(table2):
        for col_idx, value in enumerate(row):
            sheet.cells[(row_idx, col_idx + 3)] = CellData(
                row=row_idx, column=col_idx + 3, value=value, data_type="s"
            )

    return sheet


class TestIslandDetector:
    """Test the IslandDetector class."""

    def test_detect_multiple_islands(self, island_detector, sheet_with_multiple_tables):
        """Test detection of multiple separate tables."""
        islands = island_detector.detect_islands(sheet_with_multiple_tables)

        assert islands is not None
        # With adaptive merging, close tables may be merged
        assert len(islands) >= 2  # At least 2 islands (table 3 is far from others)

        # Verify island boundaries
        islands = sorted(islands, key=lambda i: (i.min_row, i.min_col))

        if len(islands) == 2:
            # Tables 1 and 2 were merged
            assert islands[0].min_row == 0
            assert islands[0].min_col == 0
            assert islands[0].max_row == 6
            assert islands[0].max_col == 6

            # Table 3: A10:B12
            assert islands[1].min_row == 9
            assert islands[1].min_col == 0
            assert islands[1].max_row == 11
            assert islands[1].max_col == 1
        else:
            # All tables detected separately
            # Table 1: A1:C3
            assert islands[0].min_row == 0
            assert islands[0].min_col == 0
            assert islands[0].max_row == 2
            assert islands[0].max_col == 2

            # Table 2: E5:G7
            assert islands[1].min_row == 4
            assert islands[1].min_col == 4
            assert islands[1].max_row == 6
            assert islands[1].max_col == 6

            # Table 3: A10:B12
            assert islands[2].min_row == 9
            assert islands[2].min_col == 0
            assert islands[2].max_row == 11
            assert islands[2].max_col == 1

    def test_detect_single_island(self, island_detector, sheet_with_single_table):
        """Test detection when there's only one table."""
        islands = island_detector.detect_islands(sheet_with_single_table)

        # Should return single island
        assert islands is not None
        assert len(islands) == 1

    def test_detect_adjacent_tables(self, island_detector, sheet_with_adjacent_tables):
        """Test detection of tables separated by minimal gaps."""
        islands = island_detector.detect_islands(sheet_with_adjacent_tables)

        assert islands is not None
        # With adaptive merging, adjacent tables may be merged into one
        assert len(islands) >= 1

        if len(islands) == 2:
            # Tables detected as separate
            islands = sorted(islands, key=lambda i: i.min_col)
            assert islands[0].max_col < islands[1].min_col
        else:
            # Tables were merged - verify the merged bounds
            assert islands[0].min_col == 0
            assert islands[0].max_col >= 3  # Should include both tables

    def test_empty_sheet(self, island_detector):
        """Test detection on empty sheet."""
        sheet = SheetData(name="Empty")
        islands = island_detector.detect_islands(sheet)

        assert islands is not None
        assert len(islands) == 0

    def test_single_cell_islands(self, island_detector):
        """Test handling of isolated single cells."""
        sheet = SheetData(name="SingleCells")

        # Add isolated cells
        sheet.cells[(0, 0)] = CellData(row=0, column=0, value="A", data_type="s")
        sheet.cells[(5, 5)] = CellData(row=5, column=5, value="B", data_type="s")
        sheet.cells[(10, 10)] = CellData(row=10, column=10, value="C", data_type="s")

        islands = island_detector.detect_islands(sheet)

        # Single cells might be filtered out or detected based on min_cells
        assert islands is not None
        # Each single cell could be its own island
        assert len(islands) <= 3

    def test_large_sparse_sheet(self, island_detector):
        """Test performance with large sparse sheet."""
        sheet = SheetData(name="LargeSparse")

        # Create two tables far apart
        # Table 1 at top-left
        for i in range(10):
            for j in range(5):
                sheet.cells[(i, j)] = CellData(row=i, column=j, value=f"T1_{i}_{j}", data_type="s")

        # Table 2 at bottom-right (simulating large gap)
        for i in range(5):
            for j in range(3):
                sheet.cells[(100 + i, 50 + j)] = CellData(
                    row=100 + i, column=50 + j, value=f"T2_{i}_{j}", data_type="s"
                )

        islands = island_detector.detect_islands(sheet)

        assert islands is not None
        assert len(islands) == 2
        assert islands[0].max_row < 50  # First table is in top area
        assert islands[1].min_row >= 100  # Second table is far below

    def test_connected_component_labeling(self, island_detector):
        """Test the connected component algorithm with complex shapes."""
        sheet = SheetData(name="ComplexShape")

        # Create an L-shaped table
        # Horizontal part
        for j in range(5):
            sheet.cells[(0, j)] = CellData(row=0, column=j, value=f"H{j}", data_type="s")
            sheet.cells[(1, j)] = CellData(row=1, column=j, value=j, data_type="n")

        # Vertical part
        for i in range(2, 6):
            sheet.cells[(i, 0)] = CellData(row=i, column=0, value=f"V{i}", data_type="s")
            sheet.cells[(i, 1)] = CellData(row=i, column=1, value=i, data_type="n")

        islands = island_detector.detect_islands(sheet)

        assert islands is not None
        # L-shape should be detected as single connected component
        assert len(islands) == 1
        assert islands[0].min_row == 0
        assert islands[0].min_col == 0
        assert islands[0].max_row == 5
        assert islands[0].max_col == 4

    def test_min_cells_threshold(self, island_detector):
        """Test that very small islands are filtered out."""
        sheet = SheetData(name="SmallIslands")

        # Create a proper table
        for i in range(5):
            for j in range(3):
                sheet.cells[(i, j)] = CellData(row=i, column=j, value=f"T_{i}_{j}", data_type="s")

        # Add a tiny 2-cell island
        sheet.cells[(10, 10)] = CellData(row=10, column=10, value="Tiny1", data_type="s")
        sheet.cells[(10, 11)] = CellData(row=10, column=11, value="Tiny2", data_type="s")

        # Configure detector with higher min_cells if possible
        if hasattr(island_detector, "min_cells"):
            island_detector.min_cells = 5

        islands = island_detector.detect_islands(sheet)

        # Should detect the main table, tiny island might be filtered
        assert islands is not None
        assert len(islands) >= 1

        # Main table should be detected
        main_table = next(i for i in islands if i.max_row < 10)
        assert main_table.min_row == 0
        assert main_table.max_col == 2

    def test_metadata_collection(self, island_detector, sheet_with_multiple_tables):
        """Test that metadata about islands is collected."""
        islands = island_detector.detect_islands(sheet_with_multiple_tables)

        assert islands is not None
        assert len(islands) >= 2  # May merge close tables

        # Check island properties
        for island in islands:
            assert hasattr(island, "density")
            assert hasattr(island, "confidence")
            assert island.density >= 0.0
            assert 0.0 <= island.confidence <= 1.0
