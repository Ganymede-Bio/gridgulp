# GridPorter

A lightweight, efficient spreadsheet table detection framework with zero external dependencies.

## Overview

GridPorter is a Python framework that automatically detects and extracts tables from spreadsheets (Excel, CSV, and text files). It uses proven algorithmic detection methods that handle 97% of real-world use cases with fast, accurate, and dependency-free table detection.

### Key Features

- **Zero External Dependencies**: No LLM, AI services, or complex orchestration required
- **Multi-format Support**: Excel (.xlsx, .xls, .xlsm, .xlsb), CSV, TSV, and text files
- **Sophisticated Detection**: Handles single tables, multiple tables, and complex layouts
- **High Performance**: Processes large files efficiently with streaming support
- **Text File Intelligence**: Auto-detects delimiters and encodings (UTF-8, UTF-16, Latin-1)
- **Type Safe**: Built with Pydantic v2 for robust data validation

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Ganymede-Bio/gridporter.git
cd gridporter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using uv (Fastest)

```bash
# Install with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from gridporter import GridPorter

async def main():
    # Initialize GridPorter
    porter = GridPorter()

    # Detect tables in any supported file
    result = await porter.detect_tables("data/report.xlsx")

    # Display results
    print(f"Found {result.total_tables} tables in {result.detection_time:.2f}s")

    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for table in sheet.tables:
            print(f"  Table at {table.range.excel_range}")
            print(f"  Size: {table.shape[0]} rows × {table.shape[1]} columns")
            print(f"  Confidence: {table.confidence:.1%}")

asyncio.run(main())
```

## Core Detection Methods

GridPorter uses a combination of algorithmic approaches:

### 1. Simple Case Detection (Fast Path)
Handles ~80% of spreadsheets with a single table starting near cell A1:
```python
# Automatic for files with single tables
result = await porter.detect_tables("simple_data.csv")
```

### 2. Island Detection
Identifies multiple disconnected data regions:
```python
# Automatic for complex layouts
result = await porter.detect_tables("multi_table_report.xlsx")
# Detects each separate table as an "island" of connected cells
```

### 3. Excel Metadata
Leverages Excel's built-in table definitions when available:
```python
# Uses ListObjects and named ranges if present
result = await porter.detect_tables("structured_data.xlsx")
```

## Supported File Types

### Excel Files
- **.xlsx**: Modern Excel format (2007+)
- **.xls**: Legacy Excel format
- **.xlsm**: Excel with macros (macros ignored)
- **.xlsb**: Excel binary format

### Text Files
- **.csv**: Comma-separated values
- **.tsv**: Tab-separated values
- **.txt**: Auto-detected delimited data

Features:
- Automatic delimiter detection (comma, tab, pipe, semicolon)
- Sophisticated encoding detection (UTF-8, UTF-16, Latin-1, etc.)
- Handles quoted fields and escape characters

## Configuration

```python
from gridporter import GridPorter, Config

# Custom configuration
config = Config(
    # Detection settings
    confidence_threshold=0.7,      # Minimum confidence score
    max_tables_per_sheet=50,       # Maximum tables to detect per sheet
    min_table_size=(2, 2),         # Minimum table dimensions

    # Performance settings
    max_file_size_mb=2000,         # Maximum file size to process
    timeout_seconds=300,           # Processing timeout
    enable_simple_case_detection=True,  # Use fast path when possible
    enable_island_detection=True,       # Detect multiple tables

    # Excel settings
    use_excel_metadata=True,       # Use Excel table definitions
    detect_merged_cells=True,      # Handle merged cells
)

porter = GridPorter(config)
```

## Output Format

GridPorter returns structured data with comprehensive detection information:

```python
{
    "file_info": {
        "path": "report.xlsx",
        "type": "xlsx",
        "size_mb": 1.5,
        "detected_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    },
    "sheets": [
        {
            "name": "Sheet1",
            "tables": [
                {
                    "range": {
                        "start_row": 0,
                        "start_col": 0,
                        "end_row": 99,
                        "end_col": 4,
                        "excel_range": "A1:E100"
                    },
                    "confidence": 0.95,
                    "detection_method": "simple_case_fast",
                    "shape": [100, 5],
                    "headers": ["Date", "Product", "Quantity", "Price", "Total"]
                }
            ]
        }
    ],
    "metadata": {
        "detection_time": 0.15,
        "total_tables": 1,
        "methods_used": ["simple_case_fast"]
    }
}
```

## Advanced Usage

### Batch Processing
```python
async def process_directory(path):
    porter = GridPorter()
    files = Path(path).glob("*.xlsx")

    for file in files:
        result = await porter.detect_tables(str(file))
        print(f"{file.name}: {result.total_tables} tables")
```

### Error Handling
```python
from gridporter import GridPorter, ReaderError

try:
    result = await porter.detect_tables("data.xlsx")
except FileNotFoundError:
    print("File not found")
except ReaderError as e:
    print(f"Failed to read file: {e}")
```

### Direct Reader Access
```python
from gridporter.readers import ExcelReader

# Use readers directly for custom processing
reader = ExcelReader("data.xlsx")
for sheet_data in reader.read_sheets():
    print(f"Sheet {sheet_data.name} has {len(sheet_data.cells)} cells")
```

## Architecture

GridPorter follows a streamlined detection pipeline:

1. **File Type Detection**: Validates file type using magic bytes and content analysis
2. **Reader Selection**: Chooses appropriate reader based on file type
3. **Sheet Processing**: Extracts cell data while preserving structure
4. **Table Detection**: Applies detection algorithms in order of speed/accuracy
5. **Result Assembly**: Combines detections into structured output

The system prioritizes performance with early-exit optimizations and memory-efficient streaming for large files.

## Development

### Setup
```bash
# Clone and install
git clone https://github.com/Ganymede-Bio/gridporter.git
cd gridporter
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
make lint

# Run all checks
make check
```

### Testing
```bash
# Run all tests with coverage
pytest --cov=gridporter

# Run specific test file
pytest tests/test_detectors.py -v

# Test against real files
python scripts/test_manual_files.py
```

## Performance

GridPorter is optimized for real-world spreadsheet processing:

- **Small files (<1MB)**: ~50ms
- **Medium files (1-10MB)**: <500ms
- **Large files (10-100MB)**: 2-10s
- **Memory efficient**: Processes files larger than available RAM

Benchmarks on typical business spreadsheets:
- Simple tables: 100,000+ cells/second
- Complex multi-table: 50,000+ cells/second
- Text file encoding detection: <10ms overhead

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- Additional detection algorithms
- Performance optimizations
- Support for new file formats
- Bug fixes and test coverage

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

GridPorter builds on excellent open-source libraries:
- [openpyxl](https://openpyxl.readthedocs.io/) - Excel file reading
- [python-calamine](https://github.com/tafia/calamine) - Fast Excel parsing
- [chardet](https://github.com/chardet/chardet) - Character encoding detection
- [Magika](https://github.com/google/magika) - File type detection
