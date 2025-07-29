"""Tests for StructuredTextDetector."""

import pytest

from gridgulp.detectors.structured_text_detector import StructuredTextDetector
from gridgulp.models.sheet_data import CellData, SheetData


@pytest.fixture
def detector():
    """Create a StructuredTextDetector instance."""
    return StructuredTextDetector()


@pytest.fixture
def plate_data_sheet():
    """Create sheet data resembling plate reader output."""
    sheet = SheetData(name="plate_data", cells={})

    # Metadata section
    metadata_rows = [
        ["Experiment:", "Plate Reading", "", ""],
        ["Date:", "2024-01-01", "", ""],
        ["", "", "", ""],
    ]

    # 96-well plate data
    plate_headers = [""] + [str(i) for i in range(1, 13)]  # Column headers 1-12
    plate_rows = []

    for i in range(8):
        row_label = chr(ord("A") + i)
        row_data = [row_label] + [f"{i * 12 + j}" for j in range(12)]
        plate_rows.append(row_data)

    # Combine all data
    all_data = metadata_rows + [plate_headers] + plate_rows

    # Add to sheet
    for row_idx, row in enumerate(all_data):
        for col_idx, value in enumerate(row):
            if value:  # Only add non-empty cells
                cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
                sheet.cells[(row_idx, col_idx)] = cell

    return sheet


@pytest.fixture
def multi_table_sheet():
    """Create sheet with multiple tables."""
    sheet = SheetData(name="multi_table", cells={})

    # First table
    table1 = [
        ["Sample", "Value"],
        ["A1", "100"],
        ["A2", "200"],
    ]

    # Gap
    gap_rows = 2

    # Second table
    table2 = [
        ["Name", "Concentration", "Unit"],
        ["Standard1", "10", "mg/ml"],
        ["Standard2", "20", "mg/ml"],
        ["Standard3", "30", "mg/ml"],
    ]

    # Add first table
    for row_idx, row in enumerate(table1):
        for col_idx, value in enumerate(row):
            cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(row_idx, col_idx)] = cell

    # Add second table
    start_row = len(table1) + gap_rows
    for row_idx, row in enumerate(table2):
        for col_idx, value in enumerate(row):
            cell = CellData(row=start_row + row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(start_row + row_idx, col_idx)] = cell

    return sheet


@pytest.fixture
def wide_table_sheet():
    """Create sheet with a very wide table (like plate readings)."""
    sheet = SheetData(name="wide_table", cells={})

    # Create a wide table with 100 columns
    headers = ["Well"] + [f"Reading_{i}" for i in range(1, 101)]
    data_rows = [
        ["A1"] + [str(i * 10) for i in range(1, 101)],
        ["A2"] + [str(i * 20) for i in range(1, 101)],
        ["A3"] + [str(i * 30) for i in range(1, 101)],
    ]

    all_data = [headers] + data_rows

    for row_idx, row in enumerate(all_data):
        for col_idx, value in enumerate(row):
            cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(row_idx, col_idx)] = cell

    return sheet


def test_detect_plate_format(detector, plate_data_sheet):
    """Test detection of standard plate formats."""
    tables = detector.detect_tables(plate_data_sheet)

    # Should detect metadata and plate separately
    assert len(tables) >= 1

    # Find the plate table
    plate_table = None
    for table in tables:
        if table.metadata and table.metadata.get("plate_format") == "96-well":
            plate_table = table
            break

    assert plate_table is not None
    assert plate_table.confidence > 0.9
    assert plate_table.detection_method == "plate_format_detection"


def test_detect_multiple_tables(detector, multi_table_sheet):
    """Test detection of multiple tables with gaps."""
    tables = detector.detect_tables(multi_table_sheet)

    assert len(tables) == 2

    # First table
    assert tables[0].range.start_row == 0
    assert tables[0].range.row_count == 3
    assert tables[0].range.col_count == 2

    # Second table
    assert tables[1].range.start_row == 5
    assert tables[1].range.row_count == 4
    assert tables[1].range.col_count == 3


def test_detect_wide_tables(detector, wide_table_sheet):
    """Test detection of very wide tables."""
    tables = detector.detect_tables(wide_table_sheet)

    assert len(tables) == 1

    table = tables[0]
    assert table.range.col_count == 101  # 100 data columns + 1 header column
    assert table.detection_method == "wide_table_detection"
    assert table.metadata["table_type"] == "wide_table"


def test_empty_sheet(detector):
    """Test handling of empty sheet."""
    empty_sheet = SheetData(name="empty", cells={})
    tables = detector.detect_tables(empty_sheet)

    assert tables == []


def test_instrument_output_detection(detector):
    """Test detection of instrument output patterns."""
    sheet = SheetData(name="instrument", cells={})

    # Create instrument-like output
    data = [
        ["Sample Name", "Absorbance", "Concentration", "CV%"],
        ["Control", "0.123", "10.5", "2.3"],
        ["Sample 1", "0.456", "45.2", "1.8"],
        ["Sample 2", "0.789", "78.9", "3.1"],
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(row_idx, col_idx)] = cell

    tables = detector.detect_tables(sheet)

    assert len(tables) == 1
    table = tables[0]
    assert table.has_headers is True
    assert table.headers == data[0]
    assert table.metadata.get("instrument_output") is True


def test_384_well_plate_detection(detector):
    """Test detection of 384-well plate format."""
    sheet = SheetData(name="384_plate", cells={})

    # Create 384-well plate headers (16 rows x 24 columns)
    # Column headers 1-24
    headers = [""] + [str(i) for i in range(1, 25)]

    # Row headers A-P
    plate_data = [headers]
    for i in range(16):
        row_label = chr(ord("A") + i)
        row_data = [row_label] + [f"{i},{j}" for j in range(24)]
        plate_data.append(row_data)

    for row_idx, row in enumerate(plate_data):
        for col_idx, value in enumerate(row):
            if value:
                cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
                sheet.cells[(row_idx, col_idx)] = cell

    tables = detector.detect_tables(sheet)

    # Find 384-well plate
    plate_found = False
    for table in tables:
        if table.metadata and "384-well" in str(table.metadata.get("plate_format", "")):
            plate_found = True
            assert table.confidence > 0.9
            break

    assert plate_found


def test_mixed_format_detection(detector):
    """Test detection with mixed table formats."""
    sheet = SheetData(name="mixed", cells={})

    # Small metadata table
    metadata = [
        ["Parameter", "Value"],
        ["Temperature", "37°C"],
        ["Time", "2h"],
    ]

    # Wide data table (starts at row 5)
    wide_data = [
        ["Sample"] + [f"T{i}" for i in range(20)],
        ["S1"] + [str(i * 10) for i in range(20)],
        ["S2"] + [str(i * 15) for i in range(20)],
    ]

    # Add metadata
    for row_idx, row in enumerate(metadata):
        for col_idx, value in enumerate(row):
            cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(row_idx, col_idx)] = cell

    # Add wide data
    start_row = 5
    for row_idx, row in enumerate(wide_data):
        for col_idx, value in enumerate(row):
            cell = CellData(row=start_row + row_idx, column=col_idx, value=value, data_type="s")
            sheet.cells[(start_row + row_idx, col_idx)] = cell

    tables = detector.detect_tables(sheet)

    assert len(tables) == 2

    # Should have one small and one wide table
    widths = [table.range.col_count for table in tables]
    assert min(widths) == 2  # Metadata table
    assert max(widths) == 21  # Wide data table


def test_structural_analysis(detector):
    """Test structural analysis for column consistency."""
    sheet = SheetData(name="structural", cells={})

    # Create table with consistent column structure
    data = [
        ["ID", "Name", "Score"],
        ["1", "Alice", "95"],
        ["2", "Bob", "87"],
        ["3", "Charlie", "92"],
        ["", "", ""],  # Empty row
        ["Summary", "", ""],
        ["Total", "", "274"],
        ["Average", "", "91.3"],
    ]

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if value:
                cell = CellData(row=row_idx, column=col_idx, value=value, data_type="s")
                sheet.cells[(row_idx, col_idx)] = cell

    tables = detector.detect_tables(sheet)

    # Should detect as separate tables due to empty row
    assert len(tables) >= 2
