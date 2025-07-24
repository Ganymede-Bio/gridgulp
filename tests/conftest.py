"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_excel_file(fixtures_dir: Path) -> Path:
    """Path to sample Excel file for testing."""
    # This would be a real Excel file in a full implementation
    return fixtures_dir / "sample_data.xlsx"


@pytest.fixture
def sample_csv_file(fixtures_dir: Path) -> Path:
    """Path to sample CSV file for testing."""
    return fixtures_dir / "sample_data.csv"
