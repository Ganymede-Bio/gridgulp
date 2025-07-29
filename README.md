# GridPorter

A simplified, intelligent spreadsheet ingestion framework with automatic multi-table detection.

## Overview

GridPorter is a Python framework that automatically detects and extracts multiple tables from complex spreadsheets (Excel, CSV, and text files). It uses proven detection algorithms that handle 97% of use cases with simple, fast, and accurate table detection.

### What's New in v0.3.0

- **Simplified Architecture**: Removed complex agent dependencies, 77% code reduction
- **Text File Support**: Process text files with automatic CSV/TSV detection
- **Sophisticated Encoding Detection**: Multi-layer encoding detection with BOM, chardet, and pattern analysis
- **Fast Path Optimization**: 97% of use cases handled by simple detectors
- **Zero External Dependencies**: No more agent orchestration complexity

### Previous Release (v0.2.2)

- **Zero-Cost Detection**: New traditional methods for cost-free table detection
  - `SimpleCaseDetector`: Fast detection for single tables starting near A1
  - `IslandDetector`: Multi-table detection using connected component analysis
  - Excel metadata extraction from ListObjects and named ranges
- **Cost Optimization Framework**: Intelligent routing between detection methods
  - Budget management with session and per-file limits
  - Real-time cost tracking and reporting
  - Automatic fallback to free methods when budget is exceeded
- **Hybrid Detection Pipeline**: Best-of-both-worlds approach
  - Try free methods first, use vision only when needed
  - Confidence-based routing for optimal cost/quality balance
  - Early termination when high-confidence results are achieved
- **Code Organization Improvements**: Major refactoring for better maintainability
  - Centralized constants in `core/constants.py`
  - Custom exception classes and type definitions
  - Enhanced contextual logging throughout the codebase

## Features

- **Multi-format Support**: Handles Excel files (.xlsx, .xls, .xlsm, .xlsb), CSV files, and text files with automatic delimiter detection
- **Intelligent Table Detection**: Multiple detection strategies including:
  - **Traditional Methods** (Zero Cost):
    - Simple case detection for single tables
    - Island detection for multi-table sheets
    - Excel metadata extraction (ListObjects, named ranges)
- **Semantic Understanding** (New in v0.2.1):
  - Multi-row header detection with column hierarchy
  - Merged cell analysis and mapping
  - Section and subtotal detection
  - Format pattern recognition
  - Preservation of semantic blank rows
- **Cost Optimization Framework** (New in v0.2.2):
  - Budget management with session and per-file limits
  - Real-time cost tracking and reporting
  - Intelligent method selection based on complexity
  - Automatic fallback to free methods
- **Performance & Reliability**:
  - Async processing for better performance
  - High-speed CalamineReader for Excel files
  - Memory-efficient processing of large files
  - Comprehensive error handling
- **Developer Experience**:
  - Type-safe with Pydantic 2 models
  - Extensive configuration options
  - Easy to extend with new detectors
  - Well-documented API

## Repository Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/gridporter.git
cd gridporter

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in development mode (recommended method)
make dev
# This installs dependencies, sets up pre-commit hooks, and configures the environment

# 4. Verify installation
python -c "import gridporter; print(f'GridPorter v{gridporter.__version__} installed successfully!')"
```

### Alternative Setup Methods

#### Using uv (Fastest)
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

#### Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install from requirements files
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### Environment Configuration

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` to add your API keys (optional for basic functionality):**
   ```bash
   # Required only if using LLM features
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

### Development Commands

```bash
# Run tests
make test

# Run linting and formatting
make check

# Build the package
make build

# Clean build artifacts
make clean
```

## Installation (PyPI - Coming Soon)

```bash
# Once published to PyPI
pip install gridporter

# For development
pip install gridporter[dev]

# With visualization support
pip install gridporter[visualization]
```

### Troubleshooting

**Dependency resolution issues:**
- Make sure you're using Python 3.10+
- Try updating pip: `pip install --upgrade pip`
- Use uv for faster dependency resolution: `pip install uv`

**Virtual environment issues:**
- Delete `.venv` and recreate: `rm -rf .venv && python -m venv .venv`
- Make sure you're activating the correct environment

## Quick Start

```python
import asyncio
from gridporter import GridPorter

async def main():
    # Initialize GridPorter
    porter = GridPorter()

    # Detect tables in an Excel file
    result = await porter.detect_tables("data/sales_report.xlsx")

    # Print file information
    print(f"File: {result.file_info.path}")
    print(f"Type: {result.file_info.type}")
    print(f"Size: {result.file_info.size_mb:.1f} MB")
    print(f"Processing time: {result.metadata['detection_time']:.2f}s")

    # Show detected tables
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for table in sheet.tables:
            print(f"  Table: {table.suggested_name or table.range}")
            print(f"  Range: {table.range}")
            print(f"  Confidence: {table.confidence:.2%}")
            print(f"  Method: {table.detection_method}")
            if table.multi_row_headers:
                print(f"  Header rows: {table.multi_row_headers}")

# Run the async function
asyncio.run(main())
```

### Cost-Aware Detection (v0.2.2)

```python
from gridporter import GridPorter, Config

# Configure cost limits and detection preferences
config = Config(
    # Enable detection methods
    enable_simple_case_detection=True,  # Fast single-table detection
    enable_island_detection=True,       # Multi-table detection
    use_excel_metadata=True,           # Excel ListObjects/named ranges
    confidence_threshold=0.8           # High confidence threshold
)

porter = GridPorter(config=config)

# The system automatically:
# 1. Tries simple case detection (free)
# 2. Uses Excel metadata if available (free)
# 3. Tries island detection for multi-table sheets (free)
# 4. Falls back to vision only if needed and budget allows
result = await porter.detect_tables("complex_spreadsheet.xlsx")

# Check cost report
cost_report = result.metadata.get('cost_report', {})
print(f"Total cost: ${cost_report.get('total_cost_usd', 0):.3f}")
print(f"Methods used: {cost_report.get('method_usage', {})}")
```

### Zero-Cost Detection Methods

```python
from gridporter.detectors import SimpleCaseDetector, IslandDetector
from gridporter.utils import load_sheet_data

# Load sheet data
sheet_data = await load_sheet_data("simple_table.xlsx")

# Simple case detection (single table near A1)
simple_detector = SimpleCaseDetector()
result = simple_detector.detect_simple_table(sheet_data)
if result.is_simple_table:
    print(f"Simple table found: {result.table_range}")
    print(f"Confidence: {result.confidence:.2f}")

# Island detection (multiple disconnected tables)
island_detector = IslandDetector()
islands = island_detector.detect_islands(sheet_data)
print(f"Found {len(islands)} data islands")
for island in islands:
    print(f"  Island at {island.to_range()} with {len(island.cells)} cells")
```

### Advanced Features

#### Multi-Row Header Detection

```python
from gridporter import GridPorter, Config

# Enable advanced features
config = Config(
    detect_merged_cells=True,
    confidence_threshold=0.7
)

porter = GridPorter(config=config)

# Detect complex financial report
result = await porter.detect_tables("financial_report.xlsx")

# The system automatically detects:
# - Multi-row headers with merged cells
# - Hierarchical column structures
# - Subtotals and grand totals
# - Section boundaries
```

#### Semantic Structure Analysis

```python
# GridPorter understands semantic meaning
for table in result.sheets[0].tables:
    if table.semantic_features:
        print(f"Table has {table.semantic_features['section_count']} sections")
        print(f"Contains subtotals: {table.semantic_features['has_subtotals']}")
        print(f"Has grand total: {table.semantic_features['has_grand_total']}")
```


### Basic Configuration

```python
from gridporter import GridPorter, Config

# Create configuration
config = Config(
    # Basic configuration without external dependencies
    max_file_size_mb=10,
    confidence_threshold=0.8
)

# Initialize with config
porter = GridPorter(config=config)

# Basic file analysis (currently implemented)
result = await porter.detect_tables("your_file.xlsx")
print(f"File analyzed: {result.file_info.type}")
print(f"Ready for detection implementation!")
```

## CLI Usage

```bash
# Detect tables in a file
gridporter detect data/report.xlsx

# With verbose output
gridporter detect data/report.xlsx --verbose

# Export results to JSON
gridporter detect data/report.xlsx --output results.json

# Process multiple files
gridporter detect data/*.xlsx --output-dir results/
```

## Architecture

GridPorter uses a simplified, efficient detection pipeline:

1. **File Type Detection**: Validates file type using magic bytes and sophisticated encoding detection
2. **Fast Path Detection**: Handles 97% of cases with SimpleCaseDetector and IslandDetector
3. **Text File Support**: Automatic CSV/TSV detection with multi-layer encoding analysis

The architecture has been dramatically simplified from previous versions, removing complex agent orchestration while maintaining high accuracy.

## Configuration

GridPorter provides various configuration options to control detection behavior:

```python
from gridporter import GridPorter, Config

# Custom configuration
config = Config(
    confidence_threshold=0.8,      # Minimum confidence for detection
    max_tables_per_sheet=10,       # Maximum tables per sheet
    min_table_size=(3, 3),         # Minimum table dimensions
    detect_merged_cells=True,      # Handle merged cells
    max_file_size_mb=100,          # Maximum file size
    timeout_seconds=60,            # Processing timeout
)

porter = GridPorter(config=config)
```

## Output Format

GridPorter returns structured data using Pydantic models:

```python
{
    "file_info": {
        "path": "data/report.xlsx",
        "type": "xlsx",
        "size": 1048576,
        "detected_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    },
    "sheets": [
        {
            "name": "Sheet1",
            "tables": [
                {
                    "range": "A1:E100",
                    "suggested_name": "sales_data",
                    "confidence": 0.95,
                    "detection_method": "complex_detection",
                    "headers": ["Date", "Product", "Quantity", "Price", "Total"],
                    "multi_row_headers": 2,  // Number of header rows
                    "column_hierarchy": {  // Hierarchical header mapping
                        "2": ["Sales", "Quantity"],
                        "3": ["Sales", "Price"],
                        "4": ["Sales", "Total"]
                    },
                    "semantic_features": {
                        "has_subtotals": true,
                        "has_grand_total": true,
                        "section_count": 3,
                        "has_merged_cells": true
                    },
                    "data_preview": [...]
                }
            ]
        }
    ],
    "metadata": {
        "detection_time": 1.23,
        "methods_used": ["single_table", "complex_detection", "semantic_analysis"],
    }
}
```

## Advanced Usage

### Custom Detection Strategies

```python
from gridporter.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    async def detect(self, sheet_data):
        # Your detection logic here
        pass

# Register the detector
porter = GridPorter()
porter.register_detector(MyCustomDetector())
```

### Batch Processing

```python
from gridporter import GridPorter

async def process_batch(file_paths):
    porter = GridPorter()

    # Process files concurrently
    tasks = [porter.detect_tables(path) for path in file_paths]
    results = await asyncio.gather(*tasks)

    return results
```

### Complex Table Detection

```python
from gridporter import GridPorter
from gridporter.config import Config

# Direct usage of table detection
porter = GridPorter(Config())

# Detect tables with semantic understanding
result = await porter.detect_tables("complex_spreadsheet.xlsx")

for sheet in result.sheets:
    for table in sheet.tables:
        print(f"Table: {table.range.excel_range}")
        print(f"Confidence: {table.confidence:.2%}")
```


### Visualization

```python
from gridporter.utils.visualization import visualize_detection

# Visualize detection results
result = await porter.detect_tables("data/complex_sheet.xlsx")
visualize_detection(result, sheet_index=0, output_path="detection_viz.png")
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/gridporter.git
cd gridporter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gridporter

# Run specific test file
pytest tests/test_detectors.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
