# Using GridGulp in Jupyter Notebooks

GridGulp provides a synchronous API specifically designed for Jupyter notebooks and interactive environments. This guide shows you how to use GridGulp effectively in Jupyter.

## Installation

First, install GridGulp in your Jupyter environment:

```python
!pip install gridgulp
```

## Basic Usage

In Jupyter, use the synchronous methods (ending with `_sync`):

```python
from gridgulp import GridGulp

# Create GridGulp instance
gg = GridGulp()

# Detect tables - no async/await needed!
result = gg.detect_tables_sync("sales_data.xlsx")

# Display results
print(f"📄 File: {result.file_info.path.name}")
print(f"📊 Total tables found: {result.total_tables}")
print(f"📑 Sheets: {len(result.sheets)}")
```

## Interactive Table Exploration

Explore detected tables interactively:

```python
# Show all detected tables
for sheet in result.sheets:
    print(f"\n📋 Sheet: {sheet.name}")
    for i, table in enumerate(sheet.tables):
        print(f"  Table {i+1}:")
        print(f"    📍 Location: {table.range.excel_range}")
        print(f"    📏 Size: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"    🎯 Confidence: {table.confidence:.1%}")
        if table.headers:
            print(f"    📝 Headers: {', '.join(table.headers[:5])}")
```

## Extract and Display DataFrames

Convert tables to pandas DataFrames for analysis:

```python
import pandas as pd

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Extract first table from first sheet
if result.sheets and result.sheets[0].tables:
    table = result.sheets[0].tables[0]
    df = gg.extract_dataframe_sync(result.file_data, table)

    print(f"Table from {result.sheets[0].name}:")
    display(df.head(10))  # Jupyter's display() for pretty output

    # Show basic statistics
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
```

## Process Multiple Tables

Work with multiple tables efficiently:

```python
# Extract all tables into a dictionary
all_tables = {}

for sheet in result.sheets:
    sheet_tables = []
    for table in sheet.tables:
        df = gg.extract_dataframe_sync(result.file_data, table)
        sheet_tables.append(df)

    if sheet_tables:
        all_tables[sheet.name] = sheet_tables

# Display summary
for sheet_name, tables in all_tables.items():
    print(f"\n{sheet_name}: {len(tables)} tables")
    for i, df in enumerate(tables):
        print(f"  Table {i+1}: {df.shape}")
```

## Visualize Table Locations

Create a visual representation of where tables are located:

```python
def visualize_sheet_layout(sheet):
    """Show ASCII representation of table locations"""
    if not sheet.tables:
        print("No tables found")
        return

    print(f"Sheet: {sheet.name}")
    print("-" * 50)

    for i, table in enumerate(sheet.tables):
        r = table.range
        print(f"Table {i+1}: {r.excel_range}")
        print(f"  Position: Row {r.start_row+1} to {r.end_row+1}, "
              f"Col {chr(65+r.start_col)} to {chr(65+r.end_col)}")
        print()

# Visualize first sheet
if result.sheets:
    visualize_sheet_layout(result.sheets[0])
```

## Interactive Configuration

Experiment with different configurations:

```python
from gridgulp import Config

# Try different confidence thresholds
for threshold in [0.5, 0.7, 0.9]:
    config = Config(confidence_threshold=threshold)
    gg = GridGulp(config=config)
    result = gg.detect_tables_sync("complex_file.xlsx")
    print(f"Threshold {threshold}: Found {result.total_tables} tables")
```

## Working with Large Files

For large files, monitor progress and performance:

```python
import time

start_time = time.time()

# Process file
gg = GridGulp()
result = gg.detect_tables_sync("large_file.xlsx")

elapsed = time.time() - start_time

print(f"⏱️  Processing time: {elapsed:.2f} seconds")
print(f"📁 File size: {result.file_info.size_mb:.1f} MB")
print(f"📊 Tables found: {result.total_tables}")
print(f"⚡ Tables/second: {result.total_tables/elapsed:.1f}")
```

## Batch Processing

Process multiple files in a notebook:

```python
from pathlib import Path

# Find all Excel files
excel_files = list(Path("data/").glob("*.xlsx"))

results = []
for file_path in excel_files:
    try:
        result = gg.detect_tables_sync(str(file_path))
        results.append({
            'file': file_path.name,
            'tables': result.total_tables,
            'sheets': len(result.sheets)
        })
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

# Display results as DataFrame
summary_df = pd.DataFrame(results)
display(summary_df)
```

## Tips for Jupyter Usage

1. **Use `_sync` methods**: All async methods have synchronous equivalents
2. **Display with `display()`**: Use Jupyter's display() for better formatting
3. **Set pandas options**: Configure pandas display for better table viewing
4. **Use progress indicators**: For long operations, consider using tqdm
5. **Save intermediate results**: Store results in variables for iterative exploration

## Example: Complete Analysis Workflow

```python
# 1. Load and detect tables
gg = GridGulp()
result = gg.detect_tables_sync("quarterly_report.xlsx")

# 2. Find the largest table
largest_table = None
largest_size = 0

for sheet in result.sheets:
    for table in sheet.tables:
        size = table.shape[0] * table.shape[1]
        if size > largest_size:
            largest_size = size
            largest_table = (sheet.name, table)

# 3. Extract and analyze the largest table
if largest_table:
    sheet_name, table = largest_table
    df = gg.extract_dataframe_sync(result.file_data, table)

    print(f"Largest table from sheet '{sheet_name}':")
    print(f"Size: {table.shape}")
    display(df.head())

    # 4. Basic analysis
    print("\nNumeric columns summary:")
    display(df.describe())

    # 5. Save to CSV
    output_file = f"{sheet_name}_largest_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
```

## Next Steps

- Explore [configuration options](../user-guide/configuration.md) for fine-tuning detection
- Learn about [DataFrame extraction](../user-guide/dataframe-extraction.md) features
- See [basic usage](../user-guide/basic-usage.md) for more examples
