# GridGulp Architecture

## Overview

GridGulp is a streamlined table detection framework that uses proven algorithms to extract tables from spreadsheets and text files. The architecture prioritizes simplicity, performance, and accuracy.

## Core Design Principles

1. **Fast Path First**: most use cases handled by simple algorithms
2. **No External Dependencies**: Pure algorithmic detection without AI/ML services
3. **Format Agnostic**: Unified interface for Excel, CSV, and text files
4. **Memory Efficient**: Streaming processing for large files
5. **Type Safe**: Pydantic models for all data structures

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    GridGulp API                         │
├─────────────────────────────────────────────────────────┤
│                  File Type Detection                    │
│                  (Magika + Magic)                       │
├─────────────────────────────────────────────────────────┤
│                    File Readers                         │
│  ┌─────────────┬──────────────┬────────────────────┐    │
│  │ ExcelReader │  CSVReader   │    TextReader      │    │
│  │ (openpyxl)  │  (csv.reader)│ (encoding detect)  │    │
│  └─────────────┴──────────────┴────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                 Detection Pipeline                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. SimpleCaseDetector (single table near A1)    │    │
│  │ 2. BoxTableDetector (tables with complete borders)│    │
│  │ 3. IslandDetector (multi-table detection)       │    │
│  │ 4. ExcelMetadataExtractor (ListObjects)         │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                   Output Models                         │
│  DetectionResult → SheetResult → TableInfo              │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. File Type Detection

**Purpose**: Accurately identify file types regardless of extension

**Components**:
- `FileFormatDetector`: Main detection class
- `Magika`: Google's AI-powered file type detection
- `python-magic`: Fallback using libmagic
- `EncodingResult`: Sophisticated encoding detection for text files

**Key Features**:
- BOM (Byte Order Mark) detection
- Multi-layer encoding detection with chardet
- Pattern-based detection for scientific data
- Handles misnamed files (e.g., CSV with .xlsx extension)

### 2. File Readers

**Purpose**: Extract cell data from various file formats

#### ExcelReader
- Uses `openpyxl` for .xlsx/.xlsm files
- Uses `xlrd` for legacy .xls files
- Preserves formatting information
- Handles merged cells

#### CSVReader
- Automatic delimiter detection
- Encoding detection with fallbacks
- Type inference for cell values
- Memory-efficient streaming

#### TextReader
- Sophisticated encoding detection (UTF-8, UTF-16, etc.)
- Automatic CSV/TSV detection
- Scientific instrument data support
- Handles wide tables (100+ columns)

### 3. Detection Pipeline

**Purpose**: Identify table boundaries within sheets

#### SimpleCaseDetector
- **Use Case**: Single table starting near cell A1
- **Performance**: < 1ms for most sheets
- **Accuracy**: 100% for standard tables
- **Algorithm**: Find data bounds, check density
- **Headers**: Always extracted from first row

#### BoxTableDetector
- **Use Case**: Tables with complete borders on all four sides
- **Performance**: < 10ms for most sheets
- **Accuracy**: 95% confidence for bordered tables
- **Algorithm**: Detect cells with borders on all sides, verify data density
- **Headers**: Extracted with formatting-based detection

#### IslandDetector
- **Use Case**: Multiple disconnected tables
- **Performance**: < 100ms for complex sheets
- **Accuracy**: 95%+ for well-formatted data
- **Algorithm**: Connected component analysis
- **Headers**: Always extracted, with header detection confidence

#### ExcelMetadataExtractor
- **Use Case**: Excel tables with defined ListObjects
- **Performance**: < 10ms
- **Accuracy**: 100% for defined tables
- **Algorithm**: Direct metadata extraction

### 4. Data Models

All models use Pydantic v2 for validation and serialization:

```python
# Core detection result
class DetectionResult(BaseModel):
    file_info: FileInfo
    sheets: list[SheetResult]
    metadata: dict[str, Any]

# Table information
class TableInfo(BaseModel):
    range: TableRange
    confidence: float
    detection_method: str
    headers: list[str] | None
    shape: tuple[int, int]
```

## Processing Flow

### 1. File Loading
```python
# Detect file type
file_info = detector.detect_file_type(file_path)

# Create appropriate reader
reader = ReaderFactory.create_reader(file_path, file_info)

# Read file data
file_data = await reader.read()
```

### 2. Table Detection
```python
# For each sheet
for sheet in file_data.sheets:
    # Try simple case first (fast path)
    if simple_detector.is_simple_case(sheet):
        tables = [simple_detector.detect_simple_table(sheet)]
    # Try box detection for Excel files with formatting
    elif file_type in [FileType.XLSX, FileType.XLS]:
        box_tables = box_detector.detect_box_tables(sheet)
        if box_tables:
            tables = box_tables
        else:
            # Fall back to island detection
            tables = island_detector.detect_tables(sheet)
    else:
        # Use island detection for other formats
        tables = island_detector.detect_tables(sheet)
```

### 3. Result Assembly
```python
# Create detection result
result = DetectionResult(
    file_info=file_info,
    sheets=[
        SheetResult(
            name=sheet.name,
            tables=tables
        )
        for sheet, tables in detected_tables
    ]
)
```

## Performance Characteristics

### Memory Usage
- **Streaming**: Large files processed in chunks
- **Cell Storage**: Only non-empty cells stored
- **Format Data**: Minimal formatting preserved

### Processing Speed
- **Simple Tables**: 1M+ cells/second
- **Complex Tables**: 100K+ cells/second
- **File Loading**: Limited by I/O speed

### Scalability
- **File Size**: Tested up to 100MB files
- **Row Limit**: 1M rows (configurable)
- **Column Limit**: 16K columns (Excel limit)

## Extension Points

### Adding New File Formats
1. Create reader class extending `BaseReader`
2. Implement `read()` method
3. Register in `ReaderFactory`

### Adding New Detectors
1. Create detector class
2. Implement detection algorithm
3. Add to detection pipeline

### Custom Output Formats
1. Extend base models
2. Add serialization methods
3. Configure in GridGulp

## Error Handling

### Reader Errors
- `FileNotFoundError`: File doesn't exist
- `PermissionError`: No read access
- `ReaderError`: Format-specific issues

### Detection Errors
- `NoTablesFoundError`: No tables detected
- `DetectionTimeoutError`: Processing timeout
- `InvalidSheetError`: Corrupted sheet data

## Configuration

Key configuration options:
- `confidence_threshold`: Minimum confidence (0.0-1.0)
- `max_tables_per_sheet`: Limit table count
- `min_table_size`: Minimum rows/columns
- `timeout_seconds`: Processing timeout
- `enable_magika`: Use AI file detection

## Testing Strategy

### Unit Tests
- Individual detector algorithms
- Reader format handling
- Model validation

### Integration Tests
- End-to-end detection
- Cross-format compatibility
- Performance benchmarks

### Test Data
- Level 0: Basic single tables
- Level 1: Real-world files
- Level 2: Edge cases
