# GridGulp

[![PyPI version](https://badge.fury.io/py/gridgulp.svg)](https://pypi.org/project/gridgulp/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gridgulp.svg)](https://pypi.org/project/gridgulp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ganymede-bio.github.io/gridgulp/)

Automatically detect and extract tables from Excel, CSV, and text files.

## What is GridGulp?

GridGulp finds tables in your spreadsheets, even when

- there are multiple tables on one sheet
- tables don't start at cell A1
- file extensions do not reflect its file type
- the file encoding is opaque

**Supported formats:** `.xlsx`, `.xls`, `.xlsm`, `.csv`, `.tsv`, `.txt`

## Installation

```bash
pip install gridgulp
```

## Quick Start

### Trying GridGulp Out

To quickly try GridGulp on some spreadsheets, clone the repo, place example spreadsheets in the examples/ directory, and run

```bash
python scripts/test_example_files.py
```

You will receive output that looks like, representing identified ranges:

📁 tests/manual
----------------------------------------------------------------------------------------------------
✓ sample.xlsx                              | Tables: 1  | Time: 1.099s | Size: 122.6KB | Method: magika
  📄 Sheet: Sheet
     └─ A1:CV203        | 203×100 | Conf: 70%


### Table Ranges vs DataFrames

GridGulp provides two ways to work with detected tables:

1. **Table Ranges** - JSON metadata about where tables are located (e.g., "A1:E100")
   - Fast and memory-efficient
   - Perfect for agent use as tools - mapping table locations or visualizing spreadsheet structure
   - No actual data is loaded into memory

2. **DataFrames** - The actual data extracted from those ranges as pandas DataFrames
   - Contains the full data with proper types
   - Ready for analysis, transformation, or export

Choose based on your needs:
- Use **ranges only** when you need to know where tables are and want to submit to other tasks - for example, a downstream process to infer purpose / intent based on data content
- Use **DataFrames** when you need to analyze or transform the actual data

### Getting Table Ranges Only

```python
from gridgulp import GridGulp

# Detect tables in a file (lightweight - just finds locations)
porter = GridGulp()
result = await porter.detect_tables("sales_report.xlsx")

# Process results - no data loaded yet, just locations
for sheet in result.sheets:
    print(f"{sheet.name}: {len(sheet.tables)} tables found")
    for table in sheet.tables:
        print(f"  - Table at {table.range.excel_range}")
        print(f"    Size: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"    Confidence: {table.confidence:.1%}")
```

### Getting DataFrames

To extract the actual data as pandas DataFrames:

```python
from gridgulp import GridGulp
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

# Step 1: Detect table locations
porter = GridGulp()
result = await porter.detect_tables("sales_report.xlsx")

# Step 2: Read the file data
reader = get_reader("sales_report.xlsx")
file_data = reader.read_sync()

# Step 3: Extract DataFrames from detected ranges
extractor = DataFrameExtractor()
for sheet_result in result.sheets:
    sheet_data = next(s for s in file_data.sheets if s.name == sheet_result.name)

    for table in sheet_result.tables:
        df, metadata, quality = extractor.extract_dataframe(sheet_data, table.range)
        if df is not None:
            print(f"\n📊 Extracted table from {table.range.excel_range}")
            print(f"   Shape: {df.shape} | Quality: {quality:.1%}")
            print(f"   Headers: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            # Now you have a pandas DataFrame to work with
            print(df.head())
```

### Processing Multiple Files

GridGulp makes it easy to process entire directories of spreadsheets:

```python
from gridgulp import GridGulp

# Create GridGulp instance
gg = GridGulp()

# Process all spreadsheets in a directory (recursively by default)
results = await gg.detect_tables_in_directory("~/data")

# Summary statistics
total_tables = sum(r.total_tables for r in results.values())
print(f"Found {total_tables} tables across {len(results)} files")

# Process each file's results
for file_path, result in results.items():
    if result.total_tables > 0:
        print(f"\n{file_path.name}:")
        for sheet in result.sheets:
            for table in sheet.tables:
                print(f"  - {sheet.name}: {table.range.excel_range}")
```

#### Directory Processing Options

```python
# Process only Excel files
results = await gg.detect_tables_in_directory(
    "~/reports",
    patterns=["*.xlsx", "*.xls"]
)

# Non-recursive (single directory only)
results = await gg.detect_tables_in_directory(
    "~/data",
    recursive=False
)

# With progress tracking for large directories
def show_progress(current, total):
    print(f"Processing file {current} of {total}...")

results = await gg.detect_tables_in_directory(
    "~/large_dataset",
    progress_callback=show_progress
)

# Sync version for Jupyter notebooks
results = gg.detect_tables_in_directory_sync("~/data")
```

#### Extracting DataFrames from Multiple Files

```python
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

# Process directory and extract all tables as DataFrames
gg = GridGulp()
extractor = DataFrameExtractor()

results = gg.detect_tables_in_directory_sync("~/sales_data")

all_dataframes = []
for file_path, detection_result in results.items():
    if detection_result.total_tables > 0:
        # Read the file
        reader = get_reader(file_path)
        file_data = reader.read_sync()

        # Extract each table
        for sheet_result in detection_result.sheets:
            sheet_data = next(s for s in file_data.sheets if s.name == sheet_result.name)

            for table in sheet_result.tables:
                df, metadata, quality = extractor.extract_dataframe(sheet_data, table.range)
                if df is not None:
                    # Add source info to the dataframe
                    df['_source_file'] = file_path.name
                    df['_source_sheet'] = sheet_result.name
                    all_dataframes.append(df)

print(f"Extracted {len(all_dataframes)} tables from {len(results)} files")

# Combine all tables if they have the same structure
import pandas as pd
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined dataset: {combined_df.shape}")
```

## Key Features

- **Automatic Detection** - Finds all tables with sensible defaults
- **Fully Configurable** - Customize detection thresholds and behavior
- **Smart Headers** - Detects single and multi-row headers automatically
- **Multiple Tables** - Handles sheets with multiple separate tables
- **Quality Scoring** - Confidence scores for each detected table
- **Fast** - Processes 1M+ cells/second for simple tables, 100K+ cells/second for complex tables

## Documentation

- [Full Usage Guide](docs/USAGE_GUIDE.md) - Detailed examples and configuration
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Architecture](docs/ARCHITECTURE.md) - How GridGulp works internally
- [Testing Guide](docs/TESTING_GUIDE.md) - Test spreadsheets in bulk with the unified test script

## License

MIT License - see [LICENSE](LICENSE) file.
