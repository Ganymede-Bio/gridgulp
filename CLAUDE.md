# GridPorter Project Instructions

## Overview
GridPorter is an multi-agent tool for intelligently ingesting spreadsheet files (Excel and CSV) with automatic multi-table detection. It uses openai-agents-python to orchestrate a multi-stage detection pipeline that can identify and extract multiple tables from complex spreadsheets.

## Core Architecture

### Detection Pipeline
The system follows a hierarchical detection strategy:

1. **File Type Detection**: Use file magic to determine actual file type
2. **Single Table Check**: Fast check if file/sheet contains only one table
3. **ListObjects Detection**: For Excel files, check native table objects
4. **Island Detection**: Mask-based approach to find disconnected data regions
5. **Heuristic Augmentation**: Apply header/format analysis for accuracy
6. **LLM Range Naming**: Use AI to suggest meaningful names for detected ranges

### Agent Design Patterns

#### TableDetectorAgent
- Orchestrates the entire detection pipeline
- Manages fallback strategies
- Aggregates results from different detection methods
- Handles confidence scoring

#### RangeNamerAgent
- Takes detected table ranges and their content
- Analyzes headers, data patterns, and context
- Suggests meaningful names using LLM
- Provides confidence scores for suggestions

#### FormatAnalyzerAgent
- Analyzes cell formatting patterns
- Detects headers based on formatting
- Identifies data types and patterns
- Assists in table boundary detection

### Data Models (Pydantic 2)

All models use Pydantic 2 with strict validation:

```python
from pydantic import BaseModel, Field, ConfigDict

class TableInfo(BaseModel):
    model_config = ConfigDict(strict=True)
    
    range: str = Field(..., description="Excel-style range (e.g., 'A1:D10')")
    suggested_name: str | None = Field(None, description="LLM-suggested name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    detection_method: str
    headers: list[str] | None = None
    data_preview: list[dict] | None = None
```

### File Handling Strategy

#### Excel Files
- Use openpyxl for .xlsx/.xlsm/.xlsb files
- Use xlrd for legacy .xls files
- Preserve formatting metadata for detection
- Handle multiple sheets independently

#### CSV Files
- Auto-detect delimiter using csv.Sniffer
- Use chardet for encoding detection
- Handle various delimiters (comma, tab, pipe, semicolon)
- Detect header rows using heuristics

#### File Type Detection
- Check file signatures before trusting extensions
- Use python-magic for robust detection
- Provide clear error messages for unsupported formats
- Handle compressed files appropriately

### API Design Principles

1. **Async-First**: All I/O operations should be async
2. **Progressive Enhancement**: Start with simple detection, add complexity as needed
3. **Fail Gracefully**: Always return partial results rather than failing completely
4. **Confidence Scores**: Every detection includes confidence metrics
5. **Streaming Support**: Large files should be processed in chunks

### Output Format

The framework outputs a standardized JSON structure:

```json
{
  "file_info": {
    "path": "path/to/file.xlsx",
    "type": "xlsx",
    "size": 1048576,
    "detected_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  },
  "sheets": [
    {
      "name": "Sheet1",
      "tables": [
        {
          "range": "A1:E100",
          "suggested_name": "sales_data",
          "confidence": 0.95,
          "detection_method": "island_detection",
          "headers": ["Date", "Product", "Quantity", "Price", "Total"],
          "data_preview": [...]
        }
      ]
    }
  ],
  "metadata": {
    "detection_time": 1.23,
    "methods_used": ["single_table", "island_detection", "llm_naming"]
  }
}
```

## Testing Requirements

1. **Unit Tests**: Each detector module must have comprehensive tests
2. **Integration Tests**: Test the full pipeline with various file types
3. **Performance Tests**: Ensure reasonable performance on large files
4. **Edge Cases**: Test with malformed files, empty sheets, merged cells
5. **LLM Mocking**: Mock LLM calls in tests for reproducibility

## Development Guidelines

1. **Type Hints**: Use Python 3.10+ type hints everywhere
2. **Error Handling**: Never let exceptions bubble up without context
3. **Logging**: Use structured logging with appropriate levels
4. **Documentation**: Every public method needs docstrings
5. **Code Style**: Follow PEP 8 with Black formatting

## Performance Considerations

1. **Lazy Loading**: Don't load entire files into memory
2. **Caching**: Cache detection results for repeated operations
3. **Parallel Processing**: Process multiple sheets concurrently
4. **Early Exit**: Stop processing when confidence is high enough
5. **Resource Limits**: Set maximum file size and processing time limits

## Security Considerations

1. **File Validation**: Always validate file types before processing
2. **Size Limits**: Enforce reasonable file size limits
3. **Sandboxing**: Process untrusted files in isolated environments
4. **No Macros**: Never execute Excel macros or formulas
5. **Input Sanitization**: Sanitize all data before sending to LLM

## Extension Points

The framework should be designed for easy extension:

1. **Custom Detectors**: Allow plugins for new detection strategies
2. **Format Support**: Easy to add new file formats
3. **LLM Providers**: Support multiple LLM backends
4. **Output Formats**: Pluggable output serializers
5. **UI Integration**: Clear hooks for UI feedback

## Common Patterns

### Handling Multiple Tables
```python
async def detect_tables(file_path: str) -> DetectionResult:
    # 1. Detect file type
    file_type = await detect_file_type(file_path)
    
    # 2. Load appropriate reader
    reader = get_reader(file_type)
    
    # 3. Run detection pipeline
    for sheet in reader.sheets():
        if await is_single_table(sheet):
            tables = [extract_single_table(sheet)]
        else:
            tables = await detect_multiple_tables(sheet)
        
        # 4. Enhance with LLM
        for table in tables:
            table.suggested_name = await suggest_name(table)
    
    return DetectionResult(...)
```

### Error Recovery
```python
try:
    result = await primary_detection(file)
except DetectionError:
    # Fallback to simpler method
    result = await fallback_detection(file)
finally:
    # Always return something useful
    return result or empty_result()
```

## Debugging Tips

1. **Verbose Mode**: Enable detailed logging for troubleshooting
2. **Visualization**: Export detection masks as images
3. **Step Mode**: Run pipeline step-by-step
4. **Profiling**: Built-in performance profiling
5. **Test Fixtures**: Comprehensive test file collection

## Future Enhancements

- Support for Google Sheets API
- ML-based table detection (using table-transformer)
- Automatic data type inference
- Table relationship detection
- Export to various formats (Parquet, Arrow, etc.)