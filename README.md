# GridGulp

Automatically detect and extract tables from Excel, CSV, and text files.

## What is GridGulp?

GridGulp finds tables in your spreadsheets - even when there are multiple tables on one sheet or when tables don't start at cell A1. It comes with reasonable defaults and is fully configurable.

**Supported formats:** `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.csv`, `.tsv`, `.txt`

## Installation

```bash
pip install gridgulp
```

## Quick Start

```python
from gridgulp import GridGulp

# Detect tables in a file
porter = GridGulp()
result = await porter.detect_tables("sales_report.xlsx")

# Process results
for sheet in result.sheets:
    print(f"{sheet.name}: {len(sheet.tables)} tables found")
    for table in sheet.tables:
        print(f"  - {table.range.excel_range}")
```

### Jupyter Notebook Usage

In Jupyter notebooks, you can use synchronous methods for simplicity:

```python
from gridgulp import GridGulp

# Create GridGulp instance
gg = GridGulp()

# Use the sync method - works in Jupyter without any async complexity
result = gg.detect_tables_sync("sales_report.xlsx")

# Display results
print(f"📄 File: {result.file_info.path.name}")
print(f"📊 Total tables found: {result.total_tables}\n")

for sheet in result.sheets:
    print(f"Sheet: {sheet.name}")
    for table in sheet.tables:
        print(f"  - Table at {table.range.excel_range}")
        print(f"    Size: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"    Confidence: {table.confidence:.1%}")
```

### Extract DataFrames

Extract detected tables as pandas DataFrames with automatic type inference and quality scoring:

```python
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

# Example: Extract tables from a sales report
reader = get_reader("sales_report.xlsx")
file_data = reader.read_sync()

extractor = DataFrameExtractor()
for sheet_result in result.sheets:
    sheet_data = next(s for s in file_data.sheets if s.name == sheet_result.name)

    for table in sheet_result.tables:
        df, metadata, quality = extractor.extract_dataframe(sheet_data, table.range)
        if df is not None:
            print(f"\n📊 Extracted table from {table.range.excel_range}")
            print(f"   Shape: {df.shape} | Quality: {quality:.1%}")
            print(f"   Headers: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            print(f"\nFirst few rows:")
            print(df.head())
```

## Key Features

- **Automatic Detection** - Finds all tables with sensible defaults
- **Fully Configurable** - Customize detection thresholds and behavior
- **Smart Headers** - Detects single and multi-row headers automatically
- **Multiple Tables** - Handles sheets with multiple separate tables
- **Quality Scoring** - Confidence scores for each detected table
- **Fast** - Processes most files in under a second

## Documentation

- [Full Usage Guide](docs/USAGE_GUIDE.md) - Detailed examples and configuration
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Architecture](docs/ARCHITECTURE.md) - How GridGulp works internally

## License

MIT License - see [LICENSE](LICENSE) file.
