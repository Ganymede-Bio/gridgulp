# Quick Start

This guide will help you get started with GridGulp in just a few minutes.

## Basic Table Detection

The simplest way to use GridGulp is to detect tables in a spreadsheet:

```python
from gridgulp import GridGulp

# Create a GridGulp instance
gg = GridGulp()

# Detect tables in a file (async)
result = await gg.detect_tables("sales_report.xlsx")

# Process the results
print(f"Found {result.total_tables} tables in {result.file_info.path.name}")

for sheet in result.sheets:
    print(f"\nSheet: {sheet.name}")
    for table in sheet.tables:
        print(f"  Table at {table.range.excel_range}")
        print(f"  Size: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"  Confidence: {table.confidence:.2f}")
```

## Synchronous Usage

If you're not using async/await (e.g., in scripts or Jupyter notebooks), use the synchronous API:

```python
from gridgulp import GridGulp

gg = GridGulp()
result = gg.detect_tables_sync("report.xlsx")

# Same result object, but no await needed
for sheet in result.sheets:
    print(f"{sheet.name}: {len(sheet.tables)} tables")
```

## Extract Data as DataFrames

Once you've detected tables, extract them as pandas DataFrames:

```python
# Using the result from above
for sheet in result.sheets:
    for table in sheet.tables:
        # Extract as pandas DataFrame
        df = await gg.extract_dataframe(result.file_data, table)

        # Or synchronously
        df = gg.extract_dataframe_sync(result.file_data, table)

        print(f"Extracted table with shape: {df.shape}")
        print(df.head())
```

## Working with Different File Types

GridGulp automatically detects file types:

### Excel Files
```python
# Supported Excel formats
result = await gg.detect_tables("data.xlsx")  # Excel 2007+
result = await gg.detect_tables("data.xls")   # Legacy Excel
result = await gg.detect_tables("data.xlsm")  # With macros

# Note: .xlsb (Excel Binary) format is not supported
# If you have .xlsb files, save them as .xlsx in Excel first
```

### CSV/TSV Files
```python
# Automatic delimiter detection
result = await gg.detect_tables("data.csv")
result = await gg.detect_tables("data.tsv")
result = await gg.detect_tables("data.txt")  # If it contains delimited data
```

## Basic Configuration

Customize detection behavior with configuration:

```python
from gridgulp import GridGulp, Config

config = Config(
    confidence_threshold=0.8,      # Higher confidence requirement
    min_table_size=(3, 2),        # At least 3 rows, 2 columns
    max_tables_per_sheet=10,      # Limit tables per sheet
)

gg = GridGulp(config=config)
result = await gg.detect_tables("complex_report.xlsx")
```

## Handling Multiple Tables

When a sheet contains multiple tables:

```python
result = await gg.detect_tables("multi_table_sheet.xlsx")

for sheet in result.sheets:
    if len(sheet.tables) > 1:
        print(f"{sheet.name} has {len(sheet.tables)} tables:")

        for i, table in enumerate(sheet.tables, 1):
            print(f"\nTable {i}:")
            print(f"  Location: {table.range.excel_range}")
            print(f"  Headers: {table.headers}")

            # Extract each table separately
            df = await gg.extract_dataframe(result.file_data, table)
            print(f"  Data preview:")
            print(df.head(3))
```

## Error Handling

Always handle potential errors:

```python
from gridgulp import GridGulp, FileNotFoundError, ReaderError

gg = GridGulp()

try:
    result = await gg.detect_tables("missing_file.xlsx")
except FileNotFoundError:
    print("File not found!")
except ReaderError as e:
    print(f"Could not read file: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Next Steps

- Learn about [Jupyter notebook integration](jupyter.md) for interactive use
- Explore [configuration options](../user-guide/configuration.md) for fine-tuning
- Understand [detection methods](../user-guide/detection-methods.md) for complex scenarios
- See the [API Reference](../reference/index.md) for complete documentation
