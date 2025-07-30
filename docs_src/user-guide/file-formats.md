# File Format Support

GridGulp supports a wide range of spreadsheet and text file formats, with automatic format detection and specialized handling for each type.

## Supported Formats

### Excel Formats

| Format | Extension | Description | Reader Options |
|--------|-----------|-------------|----------------|
| Excel 2007+ | .xlsx | Modern Excel (Office Open XML) | openpyxl |
| Excel 97-2003 | .xls | Legacy Excel (BIFF) | xlrd |
| Excel Macro | .xlsm | Excel with macros | openpyxl |

### Text Formats

| Format | Extension | Description | Auto-Detection |
|--------|-----------|-------------|----------------|
| CSV | .csv | Comma-separated values | ✓ Delimiter |
| TSV | .tsv | Tab-separated values | ✓ Delimiter |
| Text | .txt | Delimited text files | ✓ Delimiter & encoding |
| PSV | .psv | Pipe-separated values | ✓ Delimiter |

## Format Detection

GridGulp uses multiple methods to accurately detect file formats:

### 1. File Signature Detection

```python
# GridGulp checks file headers/magic bytes
# More reliable than extensions

# Example: Excel files start with specific bytes
xlsx: 50 4B 03 04  # ZIP archive
xls:  D0 CF 11 E0  # OLE2 compound document
```

### 2. AI-Powered Detection (Magika)

```python
config = Config(
    enable_magika=True  # Uses Google's Magika AI model
)

# Provides 99%+ accuracy for format detection
# Handles misnamed files correctly
```

### 3. Content Analysis

For text files, GridGulp analyzes content:
- Delimiter consistency
- Character encoding
- Line endings
- Data patterns

## Excel-Specific Features

### Modern Excel (.xlsx, .xlsm)

**Features supported**:
- Multiple sheets
- Merged cells
- Cell formatting (bold, colors, borders)
- Native Excel tables (ListObjects)
- Named ranges
- Hidden sheets/cells
- Formulas (as values)

**Reader configuration**:
```python
# GridGulp uses openpyxl for Excel files
config = Config()  # Default settings work well
```

### Legacy Excel (.xls)

**Limitations**:
- Max 256 columns × 65,536 rows
- Limited formatting information
- No native table support

**Special handling**:
```python
# Automatically uses xlrd for .xls files
# No additional configuration needed
```

### Excel Binary (.xlsb) - Not Supported

**Important**: GridGulp does not support .xlsb (Excel Binary) format.

If you have .xlsb files, you must:
1. Open the file in Microsoft Excel
2. Save As → Excel Workbook (.xlsx)
3. Use the .xlsx file with GridGulp

**Why not supported**:
- Limited Python library support
- Complexity of binary format
- Most features available in .xlsx format

### Working with Excel Tables

Excel native tables are detected automatically:

```python
result = await gg.detect_tables("excel_with_tables.xlsx")

for sheet in result.sheets:
    for table in sheet.tables:
        if table.metadata.get('excel_table_name'):
            print(f"Native Excel table: {table.metadata['excel_table_name']}")
            print(f"Style: {table.metadata.get('excel_table_style')}")
```

## CSV/Text File Features

### Delimiter Detection

GridGulp automatically detects delimiters:

```python
# Common delimiters tested:
delimiters = [',', '\t', ';', '|', ' ']

# Scores based on:
# - Consistency across rows
# - Column count stability
# - Data patterns
```

### Encoding Detection

Sophisticated encoding detection:

```python
# Detection order:
1. BOM (Byte Order Mark) check
2. Chardet statistical analysis
3. Pattern-based detection
4. Fallback encodings

# Supported encodings:
- UTF-8, UTF-16 (LE/BE), UTF-32
- Latin-1, Windows-1252
- ASCII, ISO-8859-*
- Many more via chardet
```

### Custom CSV Options

```python
config = Config(
    # Delimiter options
    csv_delimiter_candidates=[',', '\t', ';'],
    csv_force_delimiter=None,  # Force specific delimiter

    # Parsing options
    csv_quote_char='"',
    csv_escape_char='\\',
    csv_skip_initial_space=True,

    # Header detection
    csv_has_header='auto',  # 'auto', True, False
    csv_header_row=None     # Specific row number
)
```

## Format-Specific Examples

### Complex Excel with Multiple Tables

```python
# Excel file with multiple tables per sheet
config = Config(
    enable_island_detection=True,
    enable_excel_metadata=True,
    detect_merged_cells=True
)

gg = GridGulp(config=config)
result = await gg.detect_tables("financial_report.xlsx")
```

### Scientific Instrument Output (TSV)

```python
# Tab-delimited instrument data
config = Config(
    csv_delimiter_candidates=['\t'],
    text_min_delimiter_consistency=0.7,
    expected_header_rows=2  # Multi-line headers
)

gg = GridGulp(config=config)
result = await gg.detect_tables("instrument_output.txt")
```

### Messy CSV with Inconsistent Formatting

```python
# CSV with mixed delimiters and encodings
config = Config(
    csv_delimiter_candidates=[',', ';', '\t', '|'],
    fallback_encoding='latin1',
    encoding_errors='replace',
    confidence_threshold=0.6  # Lower threshold for messy data
)
```

## Performance by Format

| Format | Speed | Memory Usage | Notes |
|--------|-------|--------------|-------|
| .xlsx | Medium | Medium | Full feature support |
| .xls | Medium | Medium | Limited by xlrd |
| .csv/.tsv | Fast | Very Low | Streaming capable |
| .txt | Fast | Very Low | Depends on content |

## Large File Handling

### Streaming for Text Files

```python
config = Config(
    enable_streaming=True,
    chunk_size=10000  # Rows per chunk
)

# Process large CSV without loading entire file
async for chunk in gg.stream_tables("large_file.csv"):
    process_chunk(chunk)
```

### Memory Limits

```python
config = Config(
    max_file_size_mb=2000,      # Don't process files over 2GB
    max_cell_cache_size=1000000 # Limit cells in memory
)
```

## Format Conversion

Extract detected tables to different formats:

```python
# Detect tables
result = await gg.detect_tables("input.xlsx")

# Extract as different formats
for table in result.sheets[0].tables:
    # As pandas DataFrame
    df = await gg.extract_dataframe(result.file_data, table)

    # Save as CSV
    df.to_csv("output.csv", index=False)

    # Save as Excel
    df.to_excel("output.xlsx", index=False)

    # Save as JSON
    df.to_json("output.json", orient='records')
```

## Troubleshooting Format Issues

### File Not Recognized

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check detection details
try:
    result = await gg.detect_tables("unknown_file.dat")
except UnsupportedFormatError as e:
    print(f"Format detection failed: {e}")
    print(f"Detected type: {e.detected_type}")
    print(f"Expected types: {e.supported_types}")
```

### Encoding Problems

```python
# Force specific encoding
config = Config(
    force_encoding='utf-8',
    encoding_errors='ignore'  # or 'replace'
)

# Or try multiple encodings
encodings = ['utf-8', 'latin1', 'cp1252']
for encoding in encodings:
    try:
        config = Config(force_encoding=encoding)
        gg = GridGulp(config=config)
        result = await gg.detect_tables("problematic.csv")
        break
    except UnicodeDecodeError:
        continue
```

### Format-Specific Errors

```python
# Handle format-specific issues
from gridgulp import ReaderError

try:
    result = await gg.detect_tables("corrupted.xlsx")
except ReaderError as e:
    if "password protected" in str(e):
        print("File is password protected")
    elif "corrupt" in str(e):
        print("File appears to be corrupted")
    else:
        print(f"Read error: {e}")
```

## Best Practices

1. **Let GridGulp detect formats**: Don't rely on file extensions
2. **Configure appropriate limits**: Prevent memory issues with large files
3. **Handle encoding errors**: Always have a fallback plan
4. **Test with sample files**: Each format source may have quirks

## Next Steps

- See [configuration](configuration.md) for format-specific options
- Learn about [DataFrame extraction](dataframe-extraction.md) from any format
- Explore [detection methods](detection-methods.md) for complex files
