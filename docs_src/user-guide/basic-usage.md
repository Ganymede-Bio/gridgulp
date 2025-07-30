# Basic Usage

This guide covers the fundamental concepts and common usage patterns for GridGulp.

## Core Concepts

### Detection Result

When you detect tables in a file, GridGulp returns a `DetectionResult` object containing:

- **file_info**: Information about the processed file
- **sheets**: List of sheets with detected tables
- **file_data**: Raw data for extraction
- **metadata**: Detection statistics and timing

### Table Information

Each detected table includes:

- **range**: Cell boundaries (start/end row/column)
- **confidence**: Detection confidence score (0.0-1.0)
- **shape**: Tuple of (rows, columns)
- **headers**: List of column headers (if detected)
- **detection_method**: Algorithm used for detection

## Basic Workflows

### Single File Processing

```python
from gridgulp import GridGulp

async def process_single_file():
    gg = GridGulp()

    # Detect tables
    result = await gg.detect_tables("report.xlsx")

    # Check if any tables were found
    if result.total_tables == 0:
        print("No tables found!")
        return

    # Process each sheet
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")

        # Process each table
        for table in sheet.tables:
            print(f"  Found table at {table.range.excel_range}")

            # Extract as DataFrame
            df = await gg.extract_dataframe(result.file_data, table)

            # Do something with the data
            process_dataframe(df)
```

### Batch Processing

Process multiple files efficiently:

```python
async def process_multiple_files(file_paths: list[str]):
    gg = GridGulp()

    # Process files in batch
    results = await gg.batch_detect(file_paths)

    # Process results
    for result in results:
        if result.error:
            print(f"Error in {result.file_info.path}: {result.error}")
            continue

        print(f"\n{result.file_info.path.name}:")
        print(f"  Tables: {result.total_tables}")
        print(f"  Sheets: {len(result.sheets)}")
```

## Working with Different Table Types

### Simple Tables

Most spreadsheets contain simple tables that start near cell A1:

```python
# GridGulp automatically handles simple cases
result = await gg.detect_tables("simple_data.xlsx")

# Usually returns one table per sheet with high confidence
for sheet in result.sheets:
    if sheet.tables:
        table = sheet.tables[0]
        if table.confidence > 0.9:
            print(f"High confidence table found: {table.range.excel_range}")
```

### Multiple Tables per Sheet

When sheets contain multiple tables:

```python
result = await gg.detect_tables("multi_table_sheet.xlsx")

for sheet in result.sheets:
    if len(sheet.tables) > 1:
        print(f"{sheet.name} contains {len(sheet.tables)} tables:")

        # Sort tables by position (top to bottom, left to right)
        sorted_tables = sorted(
            sheet.tables,
            key=lambda t: (t.range.start_row, t.range.start_col)
        )

        for i, table in enumerate(sorted_tables, 1):
            print(f"  Table {i}: {table.range.excel_range}")
```

### Tables with Merged Cells

Handle complex headers with merged cells:

```python
from gridgulp import Config

# Enable merged cell detection
config = Config(detect_merged_cells=True)
gg = GridGulp(config=config)

result = await gg.detect_tables("merged_headers.xlsx")

for sheet in result.sheets:
    for table in sheet.tables:
        if table.has_merged_cells:
            print(f"Table with merged cells at {table.range.excel_range}")

            # Extract with special handling
            df = await gg.extract_dataframe(
                result.file_data,
                table,
                handle_merged_cells=True
            )
```

## Data Extraction Options

### Basic Extraction

```python
# Simple extraction
df = await gg.extract_dataframe(result.file_data, table)
```

### Advanced Extraction

```python
# Extract with options
df = await gg.extract_dataframe(
    result.file_data,
    table,
    include_headers=True,      # Include headers in DataFrame
    infer_types=True,         # Infer column data types
    clean_values=True,        # Clean whitespace and formatting
    handle_merged_cells=True, # Unmerge cells properly
    parse_dates=True          # Parse date columns
)
```

### Direct Cell Access

For custom processing, access cells directly:

```python
# Get sheet data
sheet_data = result.file_data.sheets[0]

# Access specific cell
cell = sheet_data.get_cell(row=5, col=2)
if cell:
    print(f"Value: {cell.value}")
    print(f"Format: {cell.data_type}")
    print(f"Bold: {cell.is_bold}")
```

## Error Handling

### Common Errors

```python
from gridgulp import (
    GridGulp,
    FileNotFoundError,
    ReaderError,
    UnsupportedFormatError,
    DetectionError
)

gg = GridGulp()

try:
    result = await gg.detect_tables("data.xlsx")
except FileNotFoundError:
    print("File not found!")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except ReaderError as e:
    print(f"Could not read file: {e}")
except DetectionError as e:
    print(f"Detection failed: {e}")
```

### Partial Results

GridGulp tries to return partial results when possible:

```python
result = await gg.detect_tables("partially_corrupted.xlsx")

# Check for sheet-level errors
for sheet in result.sheets:
    if sheet.error:
        print(f"Error in sheet {sheet.name}: {sheet.error}")
    else:
        print(f"Successfully processed {sheet.name}")
```

## Performance Tips

### 1. Use Appropriate Detection Methods

```python
from gridgulp import Config

# For simple files, disable complex detection
config = Config(
    enable_island_detection=False,
    enable_multi_header_detection=False
)
```

### 2. Set Reasonable Limits

```python
config = Config(
    max_tables_per_sheet=10,  # Don't look for too many tables
    timeout_seconds=30        # Set timeout for large files
)
```

### 3. Configure for Your Use Case

```python
# Use appropriate settings for your files
config = Config()  # Default configuration works well for most files
```

## Logging and Debugging

Enable detailed logging for troubleshooting:

```python
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# GridGulp will now output detailed logs
gg = GridGulp()
result = await gg.detect_tables("problem_file.xlsx")
```

## Next Steps

- Learn about [configuration options](configuration.md) for fine-tuning
- Explore [detection methods](detection-methods.md) in detail
- See [DataFrame extraction](dataframe-extraction.md) for data processing
- Read about [file format support](file-formats.md)
