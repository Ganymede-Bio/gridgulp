# GridGulp

Automatically detect and extract tables from Excel, CSV, and text files.

## What is GridGulp?

GridGulp finds tables in your spreadsheets - even when there are multiple tables on one sheet or when tables don't start at cell A1. No configuration required.

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

### Extract DataFrames

```python
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

# Extract detected tables as pandas DataFrames
reader = get_reader("sales_report.xlsx")
file_data = reader.read_sync()

extractor = DataFrameExtractor()
for sheet_result in result.sheets:
    sheet_data = next(s for s in file_data.sheets if s.name == sheet_result.name)

    for table in sheet_result.tables:
        df, metadata, quality = extractor.extract_dataframe(sheet_data, table.range)
        if df is not None:
            print(f"Extracted {len(df)} rows with quality score: {quality:.2f}")
```

## Key Features

- **Automatic Detection** - Finds all tables without configuration
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
