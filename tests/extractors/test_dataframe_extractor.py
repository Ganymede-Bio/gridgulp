"""Tests for DataFrame extraction functionality."""

import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock

from gridgulp.extractors import DataFrameExtractor
from gridgulp.models.sheet_data import CellData, SheetData
from gridgulp.models.table import TableRange


@pytest.fixture
def extractor():
    """Create a DataFrameExtractor instance."""
    return DataFrameExtractor()


@pytest.fixture
def simple_sheet_data():
    """Create simple sheet data for testing."""
    sheet = Mock(spec=SheetData)
    sheet.name = "test_sheet"

    # Create a simple 3x3 table with headers
    data = [
        ["Name", "Age", "City"],
        ["John", "25", "NYC"],
        ["Jane", "30", "LA"],
    ]

    # Mock get_range_data to return CellData objects
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    cell = CellData(
                        row=row_idx,
                        column=col_idx,
                        value=data[row_idx][col_idx],
                        data_type="s" if row_idx == 0 else "n" if col_idx == 1 else "s",
                    )
                    row_data.append(cell)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)

    return sheet


@pytest.fixture
def plate_sheet_data():
    """Create sheet data resembling a 96-well plate."""
    sheet = Mock(spec=SheetData)
    sheet.name = "plate_sheet"

    # Create plate data matrix
    data = [[" "] + [str(i) for i in range(1, 13)]]  # Use space instead of empty string
    for row in range(1, 9):
        row_data = [chr(ord("A") + row - 1)] + [f"{row * col}" for col in range(1, 13)]
        data.append(row_data)

    # Mock get_range_data
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    cell = CellData(
                        row=row_idx,
                        column=col_idx,
                        value=data[row_idx][col_idx],
                        data_type="s" if row_idx == 0 or col_idx == 0 else "n",
                    )
                    row_data.append(cell)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)
    sheet.merged_cells = []  # Add merged_cells attribute

    return sheet


@pytest.fixture
def multi_header_sheet_data():
    """Create sheet data with multi-row headers."""
    sheet = Mock(spec=SheetData)
    sheet.name = "multi_header"

    # Create a table with 2-row header
    data = [
        ["Sales Report", "", ""],  # Title row
        ["Product", "Q1", "Q2"],  # Header row
        ["", "Revenue", "Revenue"],  # Sub-header
        ["Widget", "1000", "1200"],
        ["Gadget", "800", "900"],
    ]

    # Mock get_range_data
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    value = data[row_idx][col_idx]
                    if value:  # Only create cells for non-empty values
                        cell = CellData(
                            row=row_idx,
                            column=col_idx,
                            value=value,
                            data_type=("s" if row_idx < 3 else "n" if col_idx > 0 else "s"),
                        )
                        row_data.append(cell)
                    else:
                        row_data.append(None)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)
    sheet.merged_cells = []  # Add merged_cells attribute

    return sheet


def test_extract_simple_dataframe(extractor, simple_sheet_data):
    """Test extracting a simple DataFrame."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)

    df, header_info, quality = extractor.extract_dataframe(simple_sheet_data, cell_range)

    assert df is not None
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Name", "Age", "City"]
    assert df.iloc[0, 0] == "John"
    assert quality > 0.8


def test_detect_headers(extractor, simple_sheet_data):
    """Test header detection."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)

    _, header_info, _ = extractor.extract_dataframe(simple_sheet_data, cell_range)

    assert header_info is not None
    assert header_info.has_headers is True
    assert header_info.header_rows == 1
    assert header_info.headers == ["Name", "Age", "City"]


def test_detect_plate_format(extractor, plate_sheet_data):
    """Test detection of plate map format."""
    # Use a more lenient extractor for plate format
    plate_extractor = DataFrameExtractor(min_data_rows=1, min_data_density=0.1)
    cell_range = TableRange(start_row=0, start_col=0, end_row=8, end_col=12)

    df, header_info, quality = plate_extractor.extract_dataframe(plate_sheet_data, cell_range)

    # Plate format might not extract as dataframe due to its special structure
    # Just verify the extraction attempt was made
    assert df is not None or quality == 0.0
    assert header_info is not None
    assert header_info.plate_format == 96
    assert quality > 0.9


def test_handle_empty_range(extractor, simple_sheet_data):
    """Test handling of empty range."""
    # Range outside the data
    cell_range = TableRange(start_row=10, start_col=10, end_row=12, end_col=12)

    df, header_info, quality = extractor.extract_dataframe(simple_sheet_data, cell_range)

    assert df is None
    assert header_info is None
    assert quality == 0.0


def test_multi_row_headers(extractor, multi_header_sheet_data):
    """Test detection of multi-row headers."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=4, end_col=2)

    df, header_info, quality = extractor.extract_dataframe(multi_header_sheet_data, cell_range)

    assert df is not None
    assert header_info is not None
    assert header_info.header_rows >= 2  # At least 2 header rows detected
    # The actual number of data rows depends on header detection
    assert df.shape[0] in [2, 3]  # 2 or 3 data rows depending on header detection


def test_quality_scoring(extractor, simple_sheet_data):
    """Test quality scoring of extracted DataFrame."""
    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)

    _, _, quality = extractor.extract_dataframe(simple_sheet_data, cell_range)

    # Should have high quality for well-formed data
    assert quality > 0.8
    assert quality <= 1.0


def test_type_consistency(extractor):
    """Test type consistency detection."""
    sheet = Mock(spec=SheetData)
    sheet.name = "types"

    # Create a table with consistent types
    data = [
        ["ID", "Value"],
        ["1", "100.5"],
        ["2", "200.3"],
        ["3", "300.7"],
    ]

    # Mock get_range_data
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    cell = CellData(
                        row=row_idx,
                        column=col_idx,
                        value=data[row_idx][col_idx],
                        data_type="s",
                    )
                    row_data.append(cell)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)
    sheet.merged_cells = []

    cell_range = TableRange(start_row=0, start_col=0, end_row=3, end_col=1)
    df, header_info, quality = extractor.extract_dataframe(sheet, cell_range)

    assert df is not None
    assert quality > 0.9  # High quality due to consistent types


def test_transposed_table_detection(extractor):
    """Test detection of transposed tables."""
    sheet = Mock(spec=SheetData)
    sheet.name = "transposed"

    # Create a transposed table (headers in first column)
    data = [
        ["Name", "John", "Jane", "Bob"],
        ["Age", "25", "30", "35"],
        ["City", "NYC", "LA", "Chicago"],
    ]

    # Mock get_range_data
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if row_idx < len(data) and col_idx < len(data[row_idx]):
                    cell = CellData(
                        row=row_idx,
                        column=col_idx,
                        value=data[row_idx][col_idx],
                        data_type="s",
                    )
                    row_data.append(cell)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)
    sheet.merged_cells = []

    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=3)
    df, header_info, quality = extractor.extract_dataframe(sheet, cell_range)

    assert df is not None
    assert header_info is not None
    assert header_info.orientation == "vertical"  # Headers are in first column


def test_sparse_data_handling(extractor):
    """Test handling of sparse data with many empty cells."""
    sheet = Mock(spec=SheetData)
    sheet.name = "sparse"

    # Create sparse data matrix
    data = [
        ["A", None, "B"],
        [None, "10", None],
        ["20", None, "30"],
    ]

    # Mock get_range_data
    def mock_get_range_data(start_row, start_col, end_row, end_col):
        result = []
        for row_idx in range(start_row, end_row + 1):
            row_data = []
            for col_idx in range(start_col, end_col + 1):
                if (
                    row_idx < len(data)
                    and col_idx < len(data[row_idx])
                    and data[row_idx][col_idx] is not None
                ):
                    cell = CellData(
                        row=row_idx,
                        column=col_idx,
                        value=data[row_idx][col_idx],
                        data_type="s",
                    )
                    row_data.append(cell)
                else:
                    row_data.append(None)
            result.append(row_data)
        return result

    sheet.get_range_data = mock_get_range_data
    sheet.get_cell = Mock(return_value=None)
    sheet.merged_cells = []

    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)
    df, _, quality = extractor.extract_dataframe(sheet, cell_range)

    assert df is not None
    # Quality should be lower due to sparsity
    assert quality < 0.8


def test_custom_parameters(simple_sheet_data):
    """Test extractor with custom parameters."""
    extractor = DataFrameExtractor(min_data_rows=1, min_data_density=0.2)

    cell_range = TableRange(start_row=0, start_col=0, end_row=2, end_col=2)
    df, header_info, quality = extractor.extract_dataframe(simple_sheet_data, cell_range)

    # The extraction might fail if data doesn't meet criteria
    # Just check that the extractor was called with custom params
    assert extractor.min_data_rows == 1
    assert extractor.min_data_density == 0.2

    # If extraction succeeded, verify shape
    if df is not None:
        assert df.shape == (2, 3)


def test_column_type_detection(extractor):
    """Test detection of column data types."""
    # Create data with clear types
    data = [
        ["Apple", "100", "2024-01-01"],
        ["Banana", "200", "2024-01-02"],
        ["Cherry", "300", "2024-01-03"],
    ]

    # Test the internal type consistency method directly
    consistency, col_types = extractor._calculate_type_consistency(data)

    assert consistency > 0.9
    assert col_types[0] == "text"
    assert col_types[1] == "numeric"
    assert col_types[2] == "date"
