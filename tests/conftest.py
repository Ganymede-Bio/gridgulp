"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest

from gridgulp.config import Config
from gridgulp.models.sheet_data import CellData, SheetData


@pytest.fixture
def sample_sheet_data() -> SheetData:
    """Create sample sheet data for testing."""
    sheet = SheetData(name="SampleSheet")
    sheet.cells["A1"] = CellData(value="Name", data_type="text", is_bold=True, row=0, column=0)
    sheet.cells["B1"] = CellData(value="Age", data_type="text", is_bold=True, row=0, column=1)
    sheet.cells["C1"] = CellData(value="City", data_type="text", is_bold=True, row=0, column=2)
    sheet.cells["A2"] = CellData(value="Alice", data_type="text", row=1, column=0)
    sheet.cells["B2"] = CellData(value=25, data_type="number", row=1, column=1)
    sheet.cells["C2"] = CellData(value="New York", data_type="text", row=1, column=2)
    sheet.cells["A3"] = CellData(value="Bob", data_type="text", row=2, column=0)
    sheet.cells["B3"] = CellData(value=30, data_type="number", row=2, column=1)
    sheet.cells["C3"] = CellData(value="London", data_type="text", row=2, column=2)
    sheet.max_row = 2
    sheet.max_column = 2
    return sheet


@pytest.fixture
def large_sheet_data() -> SheetData:
    """Create large sheet data for testing."""
    sheet = SheetData(name="LargeSheet")

    # Create a 100x50 sheet with sparse data
    for row in range(100):
        for col in range(50):
            if (row + col) % 5 == 0:  # Sparse pattern
                col_letter = chr(65 + col % 26)
                if col >= 26:
                    col_letter = chr(65 + col // 26 - 1) + col_letter
                addr = f"{col_letter}{row + 1}"
                sheet.cells[addr] = CellData(
                    value=f"Cell_{row}_{col}", data_type="text", row=row, column=col
                )

    sheet.max_row = 99
    sheet.max_column = 49
    return sheet


@pytest.fixture
def test_config() -> Config:
    """Create test configuration."""
    return Config(
        confidence_threshold=0.7,
        min_table_size=(2, 2),
        enable_simple_case_detection=True,
        enable_island_detection=True,
    )


@pytest.fixture
def complex_sheet_data() -> SheetData:
    """Create complex sheet data with various cell types for comprehensive testing."""
    sheet = SheetData(name="ComplexSheet")

    # Header row
    sheet.cells["A1"] = CellData(value="Product", data_type="text", is_bold=True, row=0, column=0)
    sheet.cells["B1"] = CellData(value="Price", data_type="text", is_bold=True, row=0, column=1)
    sheet.cells["C1"] = CellData(value="Quantity", data_type="text", is_bold=True, row=0, column=2)
    sheet.cells["D1"] = CellData(value="Total", data_type="text", is_bold=True, row=0, column=3)
    sheet.cells["E1"] = CellData(value="Date", data_type="text", is_bold=True, row=0, column=4)

    # Data rows with various types
    sheet.cells["A2"] = CellData(value="Widget A", data_type="text", row=1, column=0)
    sheet.cells["B2"] = CellData(value=19.99, data_type="number", row=1, column=1)
    sheet.cells["C2"] = CellData(value=5, data_type="number", row=1, column=2)
    sheet.cells["D2"] = CellData(
        value="=B2*C2", data_type="formula", has_formula=True, row=1, column=3
    )
    sheet.cells["E2"] = CellData(value="2024-01-15", data_type="date", row=1, column=4)

    sheet.cells["A3"] = CellData(value="Widget B", data_type="text", row=2, column=0)
    sheet.cells["B3"] = CellData(value=29.99, data_type="number", row=2, column=1)
    sheet.cells["C3"] = CellData(value=3, data_type="number", row=2, column=2)
    sheet.cells["D3"] = CellData(
        value="=B3*C3", data_type="formula", has_formula=True, row=2, column=3
    )
    sheet.cells["E3"] = CellData(value="2024-01-16", data_type="date", row=2, column=4)

    # Merged cell
    sheet.cells["A5"] = CellData(
        value="Summary", data_type="text", is_merged=True, is_bold=True, row=4, column=0
    )

    sheet.max_row = 5
    sheet.max_column = 4
    return sheet


@pytest.fixture
def huge_sheet_data() -> SheetData:
    """Create huge sheet data for testing sheets > 1M cells."""
    sheet = SheetData(name="HugeSheet")

    # Set bounds for 2M cells
    sheet.max_row = 1999
    sheet.max_column = 999

    # Add sparse data to avoid memory issues
    for row in range(0, 2000, 100):
        for col in range(0, 1000, 100):
            sheet.cells[f"{chr(65 + col % 26)}{row + 1}"] = CellData(
                value=f"Huge_{row}_{col}", data_type="text", row=row, column=col
            )

    return sheet


@pytest.fixture
def multi_table_sheet_data() -> SheetData:
    """Create sheet with multiple separated tables."""
    sheet = SheetData(name="MultiTableSheet")

    # Table 1: Sales data (0,0)
    headers1 = ["Product", "Q1", "Q2", "Q3", "Q4"]
    for col, header in enumerate(headers1):
        sheet.cells[f"{chr(65 + col)}1"] = CellData(
            value=header, data_type="text", is_bold=True, row=0, column=col
        )

    # Sales data
    for row in range(1, 4):
        sheet.cells[f"A{row + 1}"] = CellData(
            value=f"Product {row}", data_type="text", row=row, column=0
        )
        for col in range(1, 5):
            sheet.cells[f"{chr(65 + col)}{row + 1}"] = CellData(
                value=row * col * 1000, data_type="number", row=row, column=col
            )

    # Table 2: Employee data (10,8)
    start_row, start_col = 10, 8
    headers2 = ["ID", "Name", "Department", "Salary"]
    for col, header in enumerate(headers2):
        addr = f"{chr(65 + start_col + col)}{start_row + 1}"
        sheet.cells[addr] = CellData(
            value=header,
            data_type="text",
            is_bold=True,
            row=start_row,
            column=start_col + col,
        )

    # Employee data
    employees = [
        ["E001", "Alice", "Engineering", 85000],
        ["E002", "Bob", "Sales", 65000],
        ["E003", "Charlie", "Marketing", 70000],
    ]

    for row_idx, emp in enumerate(employees, 1):
        for col_idx, value in enumerate(emp):
            addr = f"{chr(65 + start_col + col_idx)}{start_row + row_idx + 1}"
            data_type = "number" if isinstance(value, int) else "text"
            sheet.cells[addr] = CellData(
                value=value,
                data_type=data_type,
                row=start_row + row_idx,
                column=start_col + col_idx,
            )

    sheet.max_row = 13
    sheet.max_column = 11
    return sheet
