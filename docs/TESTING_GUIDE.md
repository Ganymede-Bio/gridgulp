# GridGulp Testing Guide

## Overview

This guide covers testing procedures for GridGulp, including unit tests, integration tests, and performance testing.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for end-to-end workflows
├── detectors/        # Detector-specific tests
├── fixtures/         # Shared test fixtures
├── manual/           # Manual test files organized by complexity
│   ├── level0/       # Basic single-table files
│   ├── level1/       # Medium complexity files
│   └── level2/       # Complex multi-table files
└── outputs/          # Test output captures
```

## Running Tests

### All Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gridgulp

# Run with verbose output
pytest -v
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Detector tests
pytest tests/detectors/
```

### Individual Test Files
```bash
# Test specific detector
pytest tests/detectors/test_format_analyzer.py

# Test file detection
pytest tests/unit/test_file_detection.py
```

## Unit Tests

### SimpleCaseDetector Tests
Tests for the fast single-table detector:
```python
# tests/test_simple_detector.py
- test_single_table_detection
- test_offset_table_detection
- test_empty_sheet_handling
- test_sparse_data_handling
```

### IslandDetector Tests
Tests for multi-table detection:
```python
# tests/detectors/test_island_detector.py
- test_multiple_tables_detection
- test_connected_component_analysis
- test_density_calculation
- test_edge_cases
```

### File Type Detection Tests
```python
# tests/unit/test_file_detection.py
- test_excel_detection
- test_csv_detection
- test_text_file_detection
- test_encoding_detection
- test_malformed_files
```

## Integration Tests

### End-to-End Detection
```python
# tests/integration/test_complex_tables.py
async def test_financial_report():
    """Test complete detection pipeline on financial data."""
    porter = GridGulp()
    result = await porter.detect_tables("tests/manual/level1/complex_table.xlsx")

    assert result.total_tables >= 3
    assert result.detection_time < 5.0
```

### Text File Processing
```python
# tests/integration/test_text_files.py
async def test_scientific_data():
    """Test text file with tab-delimited scientific data."""
    porter = GridGulp()
    result = await porter.detect_tables("examples/proprietary/NOV_PEGDA6000.txt")

    assert len(result.sheets) == 1
    assert result.sheets[0].tables[0].shape[1] > 50  # Wide table
```

## Performance Testing

### Benchmark Script
```python
# scripts/testing/benchmark_detection.py
import time
import asyncio
from gridgulp import GridGulp

async def benchmark_file(file_path):
    porter = GridGulp()
    start = time.time()
    result = await porter.detect_tables(file_path)
    duration = time.time() - start

    print(f"File: {file_path}")
    print(f"Tables: {result.total_tables}")
    print(f"Time: {duration:.2f}s")
    print(f"Cells/sec: {calculate_cells_per_sec(result, duration)}")
```

### Performance Targets
- **Small files (<1MB)**: < 0.5 seconds
- **Medium files (1-10MB)**: < 2 seconds
- **Large files (10-50MB)**: < 10 seconds
- **Cell processing rate**: > 100,000 cells/second

## Test Data

### Level 0: Basic Files
Simple single-table files for baseline testing:
- `test_basic.xlsx` - Basic Excel table
- `test_comma.csv` - Standard CSV
- `test_tab.tsv` - Tab-separated values
- `test_formatting.xlsx` - Formatted Excel table

### Level 1: Medium Complexity
Real-world files with multiple tables:
- `complex_table.xlsx` - Financial report with sections
- `large_table.csv` - Large dataset (>10k rows)
- `simple_table.xlsx` - Clean multi-sheet workbook

### Level 2: Complex Files
Edge cases and challenging layouts:
- `creative_tables.xlsx` - Unusual table layouts
- `weird_tables.xlsx` - Non-standard structures

## Writing New Tests

### Test Structure
```python
import pytest
from gridgulp import GridGulp, Config

@pytest.mark.asyncio
async def test_my_feature():
    """Test description."""
    # Arrange
    config = Config(confidence_threshold=0.8)
    porter = GridGulp(config)

    # Act
    result = await porter.detect_tables("path/to/test/file.xlsx")

    # Assert
    assert result.total_tables == expected_count
    assert all(table.confidence > 0.7 for sheet in result.sheets
               for table in sheet.tables)
```

### Using Fixtures
```python
@pytest.fixture
def sample_sheet_data():
    """Create sample sheet data for testing."""
    sheet = SheetData(name="test_sheet")
    # Add test data
    return sheet

async def test_with_fixture(sample_sheet_data):
    detector = SimpleCaseDetector()
    result = detector.detect_simple_table(sample_sheet_data)
    assert result.is_simple_table
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=gridgulp
```

## Debugging Tests

### Verbose Output
```bash
# See detailed test execution
pytest -vv

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

### Test Isolation
```bash
# Run tests in random order to detect dependencies
pytest --random-order

# Run specific test by name
pytest -k "test_scientific_data"
```

## Coverage Reports

### Generate Coverage
```bash
# Generate coverage report
pytest --cov=gridgulp --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage Goals
- Overall coverage: > 80%
- Core detectors: > 90%
- File readers: > 85%
- Error handling paths: > 70%
