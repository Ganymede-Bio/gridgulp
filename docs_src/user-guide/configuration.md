# Configuration

GridGulp provides extensive configuration options to fine-tune table detection for your specific needs.

## Configuration Object

All configuration is done through the `Config` class:

```python
from gridgulp import GridGulp, Config

config = Config(
    confidence_threshold=0.8,
    max_tables_per_sheet=10
)

gg = GridGulp(config=config)
```

## Detection Configuration

### Confidence Threshold

Control the minimum confidence required to accept a detected table:

```python
config = Config(
    confidence_threshold=0.7  # Default: 0.7 (0.0-1.0)
)
```

- **Higher values (0.8-0.95)**: Fewer false positives, might miss some tables
- **Lower values (0.5-0.7)**: More tables detected, might include false positives

### Table Size Limits

Set minimum and maximum table dimensions:

```python
config = Config(
    min_table_size=(2, 2),      # Min (rows, cols) - Default: (2, 2)
    max_table_size=(1048576, 16384)  # Max size - Default: Excel limits
)
```

### Tables Per Sheet

Limit the number of tables detected per sheet:

```python
config = Config(
    max_tables_per_sheet=50  # Default: 50
)
```

## Detection Methods

Enable or disable specific detection algorithms:

```python
config = Config(
    # Simple case detection (single table starting near A1)
    enable_simple_case_detection=True,  # Default: True

    # Island detection (multiple disconnected regions)
    enable_island_detection=True,       # Default: True

    # Excel metadata (native Excel tables)
    enable_excel_metadata=True,         # Default: True

    # Multi-row header detection
    enable_multi_header_detection=True, # Default: True

    # Semantic analysis
    enable_semantic_analysis=True       # Default: True
)
```

### Method-Specific Settings

#### Island Detection

```python
config = Config(
    # Island detection parameters
    island_min_cells=5,           # Min cells for valid island
    island_max_gap=1,             # Max empty cells between regions
    island_density_threshold=0.3,  # Min filled cells ratio
)
```

#### Multi-Header Detection

```python
config = Config(
    # Header detection settings
    max_header_rows=10,           # Max rows to check for headers
    header_confidence_threshold=0.7,  # Min confidence for headers
    detect_merged_cells=True      # Handle merged header cells
)
```

## File Detection

### File Type Detection

```python
config = Config(
    # File type detection
    enable_magika=True,           # AI-powered file detection
    strict_format_checking=False,  # Strict validation
    file_detection_buffer_size=8192  # Bytes to read for detection
)
```

### Encoding Detection

```python
config = Config(
    # Text file encoding
    encoding_detection_sample_size=65536,  # Bytes for encoding detection
    fallback_encoding='utf-8',    # Fallback if detection fails
    encoding_errors='replace'      # How to handle encoding errors
)
```

## Performance Options

### Reader Selection

```python
config = Config(
    # Streaming options
    enable_streaming=True,        # Stream large files
    chunk_size=1000,             # Rows per chunk for streaming
)
```

### Resource Limits

```python
config = Config(
    # Memory limits
    max_file_size_mb=2000,       # Max file size to process
    max_cell_cache_size=1000000,  # Max cells to keep in memory

    # Time limits
    timeout_seconds=300,          # Overall timeout
    per_sheet_timeout_seconds=60  # Timeout per sheet
)
```

## Format-Specific Configuration

### Excel Options

```python
config = Config(
    # Excel-specific
    read_excel_formulas=False,    # Read formula strings
    read_excel_formatting=True,   # Read cell formatting
    read_excel_comments=False,    # Read cell comments
    excel_date_system='1900',     # Date system: '1900' or '1904'
)
```

### CSV/Text Options

```python
config = Config(
    # CSV detection
    csv_delimiter_candidates=[',', '\t', ';', '|'],  # Delimiters to try
    csv_quote_char='"',           # Quote character
    csv_skip_initial_space=True,  # Skip spaces after delimiter

    # Text file handling
    text_min_delimiter_consistency=0.8,  # Min consistency for delimiter
    text_max_columns=1000         # Max columns in text file
)
```

## Advanced Configuration

### Custom Confidence Calculation

```python
config = Config(
    # Confidence weights
    confidence_weights={
        'size': 0.2,           # Table size impact
        'density': 0.3,        # Data density impact
        'formatting': 0.2,     # Formatting consistency
        'headers': 0.3         # Header quality
    }
)
```

### Detection Hints

Provide hints about expected table structure:

```python
config = Config(
    # Hints for better detection
    expected_tables_per_sheet=1,   # Optimize for single table
    expected_header_rows=1,        # Standard single header row
    expected_column_count=None,    # Any number of columns
    data_starts_row=None          # Auto-detect data start
)
```

## Configuration Profiles

Create reusable configuration profiles:

```python
# Profile for simple CSV files
simple_csv_config = Config(
    enable_island_detection=False,
    enable_multi_header_detection=False,
    max_tables_per_sheet=1,
    confidence_threshold=0.6
)

# Profile for complex Excel reports
complex_excel_config = Config(
    enable_island_detection=True,
    enable_multi_header_detection=True,
    detect_merged_cells=True,
    max_tables_per_sheet=20,
    confidence_threshold=0.75
)

# Profile for instrument output files
instrument_config = Config(
    text_min_delimiter_consistency=0.7,
    enable_semantic_analysis=True,
    expected_header_rows=2,
    csv_delimiter_candidates=['\t', ',', ' ']
)
```

## Environment Variables

Some settings can be controlled via environment variables:

```bash
# Set default confidence threshold
export GRIDGULP_CONFIDENCE_THRESHOLD=0.8

# Enable debug logging
export GRIDGULP_DEBUG=1

# Set file size limit
export GRIDGULP_MAX_FILE_SIZE_MB=500
```

## Validation

Configuration is validated on creation:

```python
try:
    config = Config(
        confidence_threshold=1.5  # Invalid: must be 0.0-1.0
    )
except ValueError as e:
    print(f"Invalid config: {e}")
```

## Best Practices

1. **Start with defaults**: The default configuration works well for most files
2. **Adjust gradually**: Change one setting at a time to understand impact
3. **Profile your data**: Different file types may need different settings
4. **Monitor performance**: Some settings significantly impact speed
5. **Use appropriate limits**: Set reasonable limits to prevent excessive processing

## Examples

### High-Precision Mode

For critical data where accuracy is paramount:

```python
config = Config(
    confidence_threshold=0.9,
    strict_format_checking=True,
    enable_semantic_analysis=True,
    min_table_size=(3, 2)
)
```

### Fast Mode

For quick processing of simple files:

```python
config = Config(
    enable_island_detection=False,
    enable_multi_header_detection=False,
    enable_semantic_analysis=False,
    max_tables_per_sheet=1
)
```

### Flexible Mode

For messy data with various formats:

```python
config = Config(
    confidence_threshold=0.6,
    text_min_delimiter_consistency=0.7,
    csv_delimiter_candidates=[',', '\t', ';', '|', ' '],
    fallback_encoding='latin1'
)
```

## Next Steps

- See [detection methods](detection-methods.md) for algorithm details
- Learn about [file format](file-formats.md) specific options
- Explore [basic usage](basic-usage.md) with configuration examples
