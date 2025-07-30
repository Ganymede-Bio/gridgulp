# API Reference

This section contains the complete API reference for GridGulp, automatically generated from the source code.

## Core Classes

### [GridGulp](gridgulp/gridgulp.md)
The main entry point for table detection and extraction.

### [Config](gridgulp/config.md)
Configuration options for customizing GridGulp behavior.

## Models

### [DetectionResult](gridgulp/models/detection_result.md)
The result of table detection, containing all detected tables and metadata.

### [TableInfo](gridgulp/models/table.md)
Information about a detected table including range, confidence, and headers.

### [FileInfo](gridgulp/models/file_info.md)
File metadata and type information.

### [SheetData](gridgulp/models/sheet_data.md)
Raw sheet data with cell access methods.

## Detectors

### [SimpleCaseDetector](gridgulp/detectors/simple_case_detector.md)
Fast detector for single tables starting near A1.

### [IslandDetector](gridgulp/detectors/island_detector.md)
Detects multiple disconnected data regions.

### [ExcelMetadataExtractor](gridgulp/detectors/excel_metadata_extractor.md)
Extracts native Excel table definitions.

### [MultiHeaderDetector](gridgulp/detectors/multi_header_detector.md)
Detects complex multi-row headers with merged cells.

## Readers

### [ExcelReader](gridgulp/readers/excel_reader.md)
Reader for Excel files using openpyxl.


### [CSVReader](gridgulp/readers/csv_reader.md)
Reader for CSV and delimited text files.

### [TextReader](gridgulp/readers/text_reader.md)
Reader for generic text files with table detection.

## Utilities

### [FileFormatDetector](gridgulp/utils/file_magic.md)
Advanced file type detection using multiple methods.

### [DataFrameExtractor](gridgulp/extractors/dataframe_extractor.md)
Converts detected tables to pandas DataFrames.

## Exceptions

All GridGulp exceptions inherit from `GridGulpError`:

- `FileNotFoundError` - File does not exist
- `ReaderError` - Error reading file
- `UnsupportedFormatError` - File format not supported
- `DetectionError` - Error during table detection
- `ExtractionError` - Error extracting table data
- `ConfigError` - Invalid configuration

## Type Hints

GridGulp uses extensive type hints. Key types include:

```python
from pathlib import Path
from typing import Any, Optional, Union

FilePath = Union[str, Path]
CellValue = Union[str, int, float, bool, None]
TableRange = tuple[int, int, int, int]  # start_row, start_col, end_row, end_col
```
