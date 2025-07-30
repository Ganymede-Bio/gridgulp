# GridGulp Project Instructions

## Overview
GridGulp is a lightweight, efficient spreadsheet table detection framework with zero external dependencies. It automatically detects and extracts tables from spreadsheets (Excel, CSV, and text files) using proven algorithmic detection methods that handle most real-world use cases.

## Core Architecture

### Detection Pipeline
The system follows a hierarchical detection strategy:

1. **File Type Detection**: Use file magic and content analysis to determine actual file type
2. **Single Table Check**: Fast check if file/sheet contains only one table (handles ~80% of cases)
3. **Box Table Detection**: For Excel files, detect tables with complete borders (95% confidence)
4. **Excel Metadata**: For Excel files, check native table objects and named ranges
5. **Island Detection**: Algorithm to find disconnected data regions for multi-table sheets
6. **Heuristic Analysis**: Apply header/format analysis for improved accuracy

### Detection Components

#### SimpleCaseDetector
- Handles ~80% of spreadsheets with a single table starting near A1
- Fast path optimization for common cases
- Uses gap detection and data density analysis
- Returns high confidence scores for clear single-table layouts
- Always extracts headers from the first row

#### BoxTableDetector
- Detects tables with complete borders on all four sides
- Assigns 95% confidence to these tables (addresses user feedback)
- Verifies data density to avoid empty bordered regions
- Ideal for formatted Excel tables with clear boundaries
- Extracts headers with formatting-based detection

#### IslandDetector
- Identifies multiple disconnected data regions
- Creates binary mask of non-empty cells
- Uses connected component analysis
- Handles complex multi-table layouts
- Always extracts headers from first row of each island

#### ExcelMetadataExtractor
- Extracts Excel ListObjects (native tables)
- Reads named ranges
- Preserves Excel-defined table structures
- Zero-overhead when metadata is available

### Data Models (Pydantic 2)

All models use Pydantic 2 with strict validation:

```python
from pydantic import BaseModel, Field, ConfigDict

class TableInfo(BaseModel):
    model_config = ConfigDict(strict=True)

    range: CellRange = Field(..., description="Table boundaries")
    confidence: float = Field(..., ge=0.0, le=1.0)
    detection_method: str
    headers: list[str] | None = None  # Always extracted from first row
    has_headers: bool = True  # Header detection confidence
    shape: tuple[int, int] = Field(..., description="(rows, columns)")
```

### File Handling Strategy

#### Excel Files
- Use openpyxl for .xlsx/.xlsm files
- Use xlrd for legacy .xls files
- Preserve formatting metadata for detection
- Handle multiple sheets independently

#### CSV/Text Files
- Auto-detect delimiter using csv.Sniffer
- Sophisticated encoding detection (BOM, chardet, pattern-based)
- Handle various delimiters (comma, tab, pipe, semicolon)
- Support UTF-8, UTF-16 (LE/BE), Latin-1, and more
- Detect header rows using heuristics (bold text, data type differences)
- Background color is no longer a primary header indicator

#### File Type Detection
- Check file signatures before trusting extensions
- Use python-magic and Magika for robust detection
- Provide clear error messages for unsupported formats
- Handle compressed files appropriately

### API Design Principles

1. **Async-First**: All I/O operations are async for performance
2. **Progressive Enhancement**: Start with simple detection, add complexity as needed
3. **Fail Gracefully**: Always return partial results rather than failing completely
4. **Confidence Scores**: Every detection includes confidence metrics
5. **Memory Efficient**: Stream large files without loading entirely into memory

### Output Format

The framework outputs a standardized structure:

```json
{
  "file_info": {
    "path": "path/to/file.xlsx",
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

## Testing Requirements

1. **Unit Tests**: Each detector module has comprehensive tests
2. **Integration Tests**: Test the full pipeline with various file types
3. **Performance Tests**: Ensure reasonable performance on large files
4. **Edge Cases**: Test with malformed files, empty sheets, merged cells
5. **Manual Tests**: Test suite in tests/manual/ for real-world files

## Development Guidelines

1. **Type Hints**: Use Python 3.11+ type hints everywhere
2. **Error Handling**: Never let exceptions bubble up without context
3. **Logging**: Use Python's logging module with appropriate levels
4. **Documentation**: Every public method needs docstrings
5. **Code Style**: Follow PEP 8 with Ruff for linting

## Performance Considerations

1. **Fast Path**: SimpleCaseDetector handles 80% of cases in <50ms
2. **Streaming**: Process files larger than available RAM
3. **Early Exit**: Stop processing when confidence is high enough
4. **Resource Limits**: Configurable file size and timeout limits
5. **Async I/O**: Non-blocking file operations

## Security Considerations

1. **File Validation**: Always validate file types before processing
2. **Size Limits**: Enforce reasonable file size limits (default: 2GB)
3. **No Code Execution**: Never execute Excel macros or formulas
4. **Memory Protection**: Prevent memory exhaustion attacks
5. **Input Validation**: Validate all user inputs

## Extension Points

The framework is designed for easy extension:

1. **Custom Detectors**: Add new detection algorithms
2. **Format Support**: Easy to add new file formats
3. **Output Formats**: Export to different formats
4. **Reader Plugins**: Add support for new file types
5. **Configuration**: Extensive configuration options

## Common Patterns

### Basic Usage
```python
import asyncio
from gridgulp import GridGulp

async def detect_tables(file_path: str):
    porter = GridGulp()
    result = await porter.detect_tables(file_path)
    return result
```

### Error Handling
```python
from gridgulp import GridGulp, ReaderError

try:
    result = await porter.detect_tables(file_path)
except FileNotFoundError:
    # Handle missing file
except ReaderError as e:
    # Handle read errors
except Exception as e:
    # Handle unexpected errors
```

### Custom Configuration
```python
from gridgulp import GridGulp, Config

config = Config(
    confidence_threshold=0.8,
    max_tables_per_sheet=50,
    enable_simple_case_detection=True,
    enable_island_detection=True,
)
porter = GridGulp(config)
```

## Debugging Tips

1. **Logging**: Set logging level to DEBUG for detailed output
2. **Test Scripts**: Use scripts/test_manual_files.py for testing
3. **Performance**: Time detection phases to identify bottlenecks
4. **Memory**: Monitor memory usage for large files
5. **Test Files**: Comprehensive test suite in tests/manual/

## Architecture Benefits

- **Zero Dependencies**: No external services or AI APIs required
- **Fast**: Processes most files in under 1 second
- **Accurate**: Solid success rate on real-world spreadsheets
- **Lightweight**: Small memory footprint
- **Portable**: Pure Python implementation

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
