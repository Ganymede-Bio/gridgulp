# GridPorter

An intelligent spreadsheet ingestion framework with automatic multi-table detection using AI agents.

## Overview

GridPorter is a Python framework that automatically detects and extracts multiple tables from complex spreadsheets (Excel and CSV files). It uses a hierarchical detection strategy combined with LLM-powered naming suggestions to provide a robust solution for spreadsheet data extraction.

## Features

- **Multi-format Support**: Handles Excel files (.xlsx, .xls, .xlsm, .xlsb) and CSV files
- **Intelligent Table Detection**: Multiple detection strategies including:
  - Single table fast check
  - Excel ListObjects detection
  - Mask-based island detection
  - Format and header heuristics
- **AI-Powered Naming**: Uses LLMs to suggest meaningful names for detected ranges
- **File Type Detection**: Uses file magic to correctly identify file types
- **Confidence Scoring**: Every detection includes confidence metrics
- **Async Processing**: Built for performance with async I/O operations
- **Extensible Architecture**: Easy to add new detection strategies or file formats

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

> **Note**: This is v0.1.0 (Foundation Release). The examples below show the intended API design. Core detection functionality will be implemented in upcoming releases.

```python
import asyncio
from gridporter import GridPorter

async def main():
    # Initialize GridPorter
    porter = GridPorter()
    
    # Detect tables in an Excel file (placeholder implementation in v0.1.0)
    result = await porter.detect_tables("data/sales_report.xlsx")
    
    # Print basic file information
    print(f"File: {result.file_info.path}")
    print(f"Type: {result.file_info.type}")
    print(f"Size: {result.file_info.size_mb:.1f} MB")
    print(f"Processing time: {result.detection_time:.2f}s")
    
    # In future versions, this will show detected tables:
    # for sheet in result.sheets:
    #     print(f"\nSheet: {sheet.name}")
    #     for table in sheet.tables:
    #         print(f"  Table: {table.suggested_name} ({table.range.excel_range})")
    #         print(f"  Confidence: {table.confidence:.2%}")
    #         print(f"  Method: {table.detection_method}")

# Run the async function
asyncio.run(main())
```

### Current v0.1.0 Capabilities

```python
from gridporter import GridPorter, Config

# Create configuration
config = Config(
    suggest_names=False,  # No LLM calls
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

GridPorter uses a multi-stage detection pipeline:

1. **File Type Detection**: Validates file type using magic bytes
2. **Single Table Check**: Fast check for simple single-table sheets
3. **ListObjects Detection**: Checks for Excel's native table objects
4. **Island Detection**: Uses image processing to find disconnected regions
5. **Heuristic Enhancement**: Applies formatting and header analysis
6. **AI Naming**: Suggests meaningful names using LLMs

## Configuration

Create a `.env` file for API keys:

```env
OPENAI_API_KEY=your_api_key_here
```

Or configure programmatically:

```python
from gridporter import GridPorter

porter = GridPorter(
    llm_provider="openai",
    api_key="your_api_key",
    confidence_threshold=0.8
)
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
                    "detection_method": "island_detection",
                    "headers": ["Date", "Product", "Quantity", "Price", "Total"],
                    "data_preview": [...]
                }
            ]
        }
    ],
    "metadata": {
        "detection_time": 1.23,
        "methods_used": ["single_table", "island_detection", "llm_naming"]
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

## Acknowledgments

- Inspired by [TableSense](https://github.com/microsoft/TableSense) and [table-transformer](https://github.com/microsoft/table-transformer)
- Built with [openai-agents-python](https://github.com/openai/openai-agents-python)
- Uses [openpyxl](https://openpyxl.readthedocs.io/) and [xlrd](https://xlrd.readthedocs.io/) for Excel handling

## Roadmap

- [ ] Support for Google Sheets API
- [ ] ML-based table detection using table-transformer
- [ ] Automatic data type inference
- [ ] Table relationship detection
- [ ] Export to Parquet/Arrow formats
- [ ] Web UI for confirmation workflow
- [ ] Plugin system for custom processors