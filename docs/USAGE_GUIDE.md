# GridGulp Usage Guide

## Installation

```bash
# Install from source
git clone https://github.com/Ganymede-Bio/gridgulp.git
cd gridgulp
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from gridgulp import GridGulp

async def main():
    # Initialize GridGulp
    porter = GridGulp()

    # Detect tables in a file
    result = await porter.detect_tables("data/report.xlsx")

    # Print results
    print(f"Found {result.total_tables} tables")
    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"  {sheet.name}: {table.range.excel_range}")

asyncio.run(main())
```

## Supported File Formats

### Excel Files
- `.xlsx` - Modern Excel format (2007+)
- `.xls` - Legacy Excel format (97-2003)
- `.xlsm` - Excel with macros

**Note:** `.xlsb` (Excel Binary format) files are detected but not supported. If you have XLSB files, please save them as XLSX format in Excel before processing.

### CSV Files
- `.csv` - Comma-separated values
- `.tsv` - Tab-separated values
- Automatic delimiter detection

### Text Files
- `.txt` - Plain text files
- Automatic CSV/TSV detection
- Sophisticated encoding detection (UTF-8, UTF-16, etc.)
- Scientific instrument data support

## Basic Usage

### Simple Detection

```python
from gridgulp import GridGulp

# Synchronous wrapper for simple scripts
def detect_tables_sync(file_path):
    porter = GridGulp()
    result = asyncio.run(porter.detect_tables(file_path))
    return result

# Use it
result = detect_tables_sync("sales_data.xlsx")
print(f"Tables found: {result.total_tables}")
```

### Processing Results

```python
async def process_file(file_path):
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Iterate through sheets and tables
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")

        for table in sheet.tables:
            print(f"  Table at {table.range.excel_range}")
            print(f"  Size: {table.shape[0]} rows × {table.shape[1]} columns")
            print(f"  Confidence: {table.confidence:.1%}")
            print(f"  Method: {table.detection_method}")

            # Access headers if available
            if table.headers:
                print(f"  Headers: {', '.join(table.headers[:5])}")
```

## Configuration Options

### Basic Configuration

```python
from gridgulp import GridGulp, Config

# Create custom configuration
config = Config(
    confidence_threshold=0.8,      # Higher confidence requirement
    max_tables_per_sheet=10,       # Limit tables per sheet
    min_table_size=(3, 2),         # Minimum 3 rows, 2 columns
    max_file_size_mb=100,          # Maximum file size
    timeout_seconds=30,            # Processing timeout
)

porter = GridGulp(config=config)
```

### Detection Settings

```python
config = Config(
    # Detection algorithms
    enable_simple_case_detection=True,  # Fast single-table detection
    enable_island_detection=True,       # Multi-table detection
    use_excel_metadata=True,           # Use Excel table definitions

    # Thresholds
    confidence_threshold=0.7,          # Minimum confidence score
    island_min_cells=20,               # Minimum cells for island
    island_density_threshold=0.8,      # Required cell density
)
```

### Performance Settings

```python
config = Config(

    # Memory management
    max_memory_mb=1000,         # Maximum memory usage
    chunk_size=10000,           # Rows per chunk for streaming

    # File detection
    enable_magika=True,         # AI-powered file type detection
    file_detection_buffer_size=8192,  # Buffer for type detection
)
```

## Advanced Usage

### Batch Processing

```python
async def process_directory(directory_path):
    porter = GridGulp()
    results = []

    # Find all spreadsheet files
    from pathlib import Path
    files = []
    for pattern in ["*.xlsx", "*.xls", "*.csv", "*.txt"]:
        files.extend(Path(directory_path).glob(pattern))

    # Process files concurrently
    for file_path in files:
        try:
            result = await porter.detect_tables(str(file_path))
            results.append({
                "file": file_path.name,
                "tables": result.total_tables,
                "time": result.detection_time
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results
```

### Text File Processing

```python
async def process_scientific_data(file_path):
    """Process scientific instrument output files."""
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Text files are processed as single sheet
    sheet = result.sheets[0]
    table = sheet.tables[0]

    print(f"Detected format: {result.metadata.get('format', 'unknown')}")
    print(f"Encoding: {result.metadata.get('encoding', 'unknown')}")
    print(f"Delimiter: {result.metadata.get('delimiter', 'unknown')}")
    print(f"Table size: {table.shape}")
```

### Error Handling

```python
from gridgulp import GridGulp, ReaderError

async def safe_process(file_path):
    porter = GridGulp()

    try:
        result = await porter.detect_tables(file_path)
        return result

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except ReaderError as e:
        print(f"Failed to read file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
```

### Custom Processing

```python
async def extract_data_with_context(file_path):
    """Extract tables with surrounding context."""
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Get the reader for direct access
    from gridgulp.readers import ReaderFactory
    reader = ReaderFactory.create_reader(
        file_path,
        result.file_info
    )

    # Read full file data
    file_data = await reader.read()

    # Process each detected table
    for sheet_idx, sheet_result in enumerate(result.sheets):
        sheet_data = file_data.sheets[sheet_idx]

        for table in sheet_result.tables:
            # Get table data with context
            start_row = max(0, table.range.start_row - 2)
            end_row = min(sheet_data.max_row, table.range.end_row + 2)

            # Extract cells in extended range
            context_data = sheet_data.get_range_data(
                start_row, table.range.start_col,
                end_row, table.range.end_col
            )

            # Process context data...
```

## Common Patterns

### Financial Reports

```python
async def process_financial_report(file_path):
    config = Config(
        detect_merged_cells=True,      # Handle merged headers
        min_table_size=(5, 3),         # Larger minimum size
    )

    porter = GridGulp(config)
    result = await porter.detect_tables(file_path)

    # Financial reports often have multiple sections
    for sheet in result.sheets:
        sections = [t for t in sheet.tables if t.confidence > 0.8]
        print(f"Found {len(sections)} sections in {sheet.name}")
```

### Scientific Data

```python
async def process_lab_data(file_path):
    # Scientific data often has many columns
    config = Config(
        max_tables_per_sheet=1,  # Usually single table
        timeout_seconds=60,      # May take longer
    )

    porter = GridGulp(config)
    result = await porter.detect_tables(file_path)

    table = result.sheets[0].tables[0]
    print(f"Data dimensions: {table.shape}")
    print(f"Columns: {len(table.headers) if table.headers else 'Unknown'}")
```

### Data Validation

```python
async def validate_extraction(file_path, expected_tables):
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Validate table count
    if result.total_tables != expected_tables:
        print(f"Warning: Expected {expected_tables} tables, "
              f"found {result.total_tables}")

    # Validate detection confidence
    low_confidence = []
    for sheet in result.sheets:
        for table in sheet.tables:
            if table.confidence < 0.7:
                low_confidence.append(
                    f"{sheet.name}:{table.range.excel_range}"
                )

    if low_confidence:
        print(f"Low confidence tables: {low_confidence}")
```

## Performance Tips

### Large Files

```python
# For files > 50MB
config = Config(
    chunk_size=50000,            # Larger chunks
    timeout_seconds=300,         # Longer timeout
)
```

### Many Small Files

```python
# Process in parallel
async def process_many_files(file_paths):
    porter = GridGulp()

    # Limit concurrency to avoid resource exhaustion
    semaphore = asyncio.Semaphore(10)

    async def process_one(path):
        async with semaphore:
            return await porter.detect_tables(path)

    tasks = [process_one(p) for p in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results
```

### Memory Optimization

```python
# For memory-constrained environments
config = Config(
    max_memory_mb=500,           # Limit memory
    chunk_size=5000,             # Smaller chunks
    max_file_size_mb=50,         # Reject huge files
)
```

## Debugging

### Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Now detection will show detailed progress
porter = GridGulp()
result = await porter.detect_tables("problem_file.xlsx")
```

### Inspection Tools

```python
# Save detection visualization
from gridgulp.utils.visualization import visualize_detection

result = await porter.detect_tables("complex_sheet.xlsx")
visualize_detection(result, output_path="detection_debug.png")
```

## FAQ

**Q: Why is detection failing on my file?**
- Check file size limits in configuration
- Enable debug logging to see detailed errors
- Verify file is not corrupted (try opening in Excel)
- Check if file has password protection

**Q: How to handle files with inconsistent formatting?**
- Lower the `confidence_threshold`
- Adjust `min_table_size` for smaller tables
- Enable `detect_merged_cells` for complex headers

**Q: Can I process files from URLs?**
- Download the file first using `requests` or `httpx`
- Save to temporary file
- Process the local file

**Q: How to export detected tables?**
- Use pandas: `pd.read_excel()` with detected ranges
- Or access sheet data directly via the reader
- Build custom export logic based on your needs
