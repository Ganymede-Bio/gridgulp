# GridPorter API Reference

## Overview

This document provides a comprehensive reference for the GridPorter API, focusing on the new features introduced in v0.2.1 for semantic understanding and complex table detection.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Agents](#agents)
3. [Detectors](#detectors)
4. [Models](#models)
5. [Telemetry](#telemetry)
6. [Configuration](#configuration)
7. [Utilities](#utilities)

## Core Classes

### GridPorter

The main entry point for table detection.

```python
class GridPorter:
    def __init__(
        self,
        config: Config | None = None,
        use_vision: bool = True,
        suggest_names: bool = True,
        confidence_threshold: float = 0.7,
        enable_feature_collection: bool = False,
        feature_db_path: str | None = None,
        **kwargs
    )
```

#### Parameters
- `config`: Configuration object (optional, will use defaults if not provided)
- `use_vision`: Enable vision-based detection (default: True)
- `suggest_names`: Use LLM for table naming suggestions (default: True)
- `confidence_threshold`: Minimum confidence for table detection (default: 0.7)
- `enable_feature_collection`: Enable telemetry collection (default: False)
- `feature_db_path`: Path to feature database (default: ~/.gridporter/features.db)

#### Methods

##### detect_tables
```python
async def detect_tables(
    self,
    file_path: str | Path
) -> DetectionResult
```

Detect all tables in a spreadsheet file.

**Parameters:**
- `file_path`: Path to the Excel or CSV file

**Returns:**
- `DetectionResult`: Contains file info, detected tables, and metadata

**Example:**
```python
gridporter = GridPorter()
result = await gridporter.detect_tables("report.xlsx")
```

## Agents

### ComplexTableAgent

Orchestrates complex table detection with semantic understanding.

```python
class ComplexTableAgent:
    def __init__(self, config: Config)
```

#### Methods

##### detect_complex_tables
```python
async def detect_complex_tables(
    self,
    sheet_data: SheetData
) -> ComplexTableResult
```

Detect complex tables with multi-row headers and semantic structure.

**Parameters:**
- `sheet_data`: Sheet data to analyze

**Returns:**
- `ComplexTableResult`: Contains detected tables with semantic features

**Example:**
```python
agent = ComplexTableAgent(config)
result = await agent.detect_complex_tables(sheet_data)

for table in result.tables:
    print(f"Table: {table.range}")
    print(f"Multi-row headers: {table.multi_row_headers}")
    print(f"Semantic features: {table.semantic_features}")
```

## Detectors

### MultiHeaderDetector

Detects multi-row headers with merged cells.

```python
class MultiHeaderDetector:
    def __init__(
        self,
        min_header_confidence: float = 0.7,
        max_header_rows: int = 10
    )
```

#### Methods

##### detect_multi_row_headers
```python
def detect_multi_row_headers(
    self,
    sheet_data: SheetData,
    table_range: TableRange,
    data: pd.DataFrame | None = None
) -> MultiRowHeader | None
```

**Parameters:**
- `sheet_data`: Sheet containing the table
- `table_range`: Bounds of the table
- `data`: Optional pandas DataFrame

**Returns:**
- `MultiRowHeader`: Header information with column mappings

### MergedCellAnalyzer

Analyzes merged cells in spreadsheets.

```python
class MergedCellAnalyzer:
    def __init__(self)
```

#### Methods

##### analyze_merged_cells
```python
def analyze_merged_cells(
    self,
    sheet_data: SheetData,
    table_range: TableRange | None = None
) -> list[MergedCell]
```

**Parameters:**
- `sheet_data`: Sheet to analyze
- `table_range`: Optional bounds to limit analysis

**Returns:**
- List of `MergedCell` objects

##### get_column_header_mapping
```python
def get_column_header_mapping(
    self,
    merged_cells: list[MergedCell],
    num_cols: int,
    start_col: int = 0
) -> dict[int, list[str]]
```

Build hierarchical column mappings from merged cells.

**Returns:**
- Dictionary mapping column index to header hierarchy

### SemanticFormatAnalyzer

Analyzes semantic structure and formatting patterns.

```python
class SemanticFormatAnalyzer:
    def __init__(
        self,
        section_keywords: list[str] | None = None,
        total_keywords: list[str] | None = None
    )
```

#### Methods

##### analyze_semantic_structure
```python
def analyze_semantic_structure(
    self,
    sheet_data: SheetData,
    table_range: TableRange
) -> SemanticStructure
```

**Returns:**
- `SemanticStructure`: Contains semantic rows, sections, format patterns

## Models

### TableInfo (Enhanced)

Extended with semantic features in v0.2.1.

```python
class TableInfo(BaseModel):
    range: str  # Excel-style range (e.g., "A1:E10")
    suggested_name: str | None = None
    confidence: float
    detection_method: str
    headers: list[str] | None = None

    # New in v0.2.1
    multi_row_headers: int | None = None
    column_hierarchy: dict[int, list[str]] | None = None
    semantic_features: dict[str, Any] | None = None
```

### MultiRowHeader

Information about multi-row headers.

```python
class MultiRowHeader(BaseModel):
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    column_mappings: dict[int, list[str]]
    merged_cells: list[MergedCell]
    confidence: float
```

### SemanticRow

Represents a row with semantic meaning.

```python
class SemanticRow(BaseModel):
    row_index: int
    row_type: RowType
    confidence: float
    metadata: dict[str, Any] = {}
```

### RowType

Enum for semantic row types.

```python
class RowType(Enum):
    HEADER = "header"
    DATA = "data"
    TOTAL = "total"
    SUBTOTAL = "subtotal"
    SECTION_HEADER = "section_header"
    BLANK = "blank"
    SEPARATOR = "separator"
```

## Telemetry

### FeatureCollector

Singleton service for collecting detection features.

```python
from gridporter.telemetry import get_feature_collector

collector = get_feature_collector()
```

#### Methods

##### initialize
```python
def initialize(
    self,
    enabled: bool = False,
    db_path: str | None = None
) -> None
```

##### record_detection
```python
def record_detection(
    self,
    file_path: str,
    file_type: str,
    sheet_name: str | None,
    table_range: str,
    detection_method: str,
    confidence: float,
    success: bool,
    **features
) -> int | None
```

##### get_summary_statistics
```python
def get_summary_statistics(self) -> dict[str, Any]
```

Get statistics about collected features.

##### export_features
```python
def export_features(
    self,
    output_path: str,
    min_confidence: float | None = None,
    detection_method: str | None = None
) -> None
```

Export features to CSV for analysis.

### DetectionFeatures

Comprehensive feature model with 40+ metrics.

```python
class DetectionFeatures(BaseModel):
    # Identification
    file_path: str
    file_type: str
    sheet_name: str | None
    table_range: str
    detection_method: str

    # Geometric features
    rectangularness: float | None
    filledness: float | None
    density: float | None
    contiguity: float | None
    edge_quality: float | None
    aspect_ratio: float | None
    size_ratio: float | None

    # Pattern features
    pattern_type: str | None
    has_multi_headers: bool | None
    orientation: str | None

    # Format features
    header_row_count: int | None
    has_bold_headers: bool | None
    has_totals: bool | None
    has_subtotals: bool | None
    section_count: int | None

    # Results
    confidence: float
    detection_success: bool
    processing_time_ms: int | None
```

## Configuration

### Config

Enhanced configuration options for v0.2.1.

```python
class Config(BaseModel):
    # Core settings
    use_vision: bool = True
    suggest_names: bool = True
    confidence_threshold: float = 0.7

    # Feature collection (new in v0.2.1)
    enable_feature_collection: bool = False
    feature_db_path: str | None = None
    feature_retention_days: int = 30

    # API settings
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    use_local_llm: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_text_model: str = "deepseek-r1:7b"
    ollama_vision_model: str = "qwen2.5vl:7b"

    # Processing limits
    max_file_size_mb: float = 100
    max_processing_time: int = 300
```

### Environment Variables

All configuration options can be set via environment variables:

```bash
export GRIDPORTER_USE_VISION=true
export GRIDPORTER_ENABLE_FEATURE_COLLECTION=true
export GRIDPORTER_FEATURE_DB_PATH=~/.gridporter/features.db
export OPENAI_API_KEY=your_key_here
```

## Utilities

### Excel Range Utilities

```python
from gridporter.utils.excel_utils import (
    excel_range_to_indices,
    indices_to_excel_range,
    expand_range,
    merge_ranges
)

# Convert Excel range to indices
start_row, start_col, end_row, end_col = excel_range_to_indices("A1:C10")

# Convert indices to Excel range
range_str = indices_to_excel_range(0, 0, 9, 2)  # Returns "A1:C10"
```

### Visualization

```python
from gridporter.utils.visualization import visualize_detection

# Visualize detection results
visualize_detection(
    detection_result,
    sheet_index=0,
    output_path="detection_viz.png"
)
```

## Error Handling

GridPorter uses custom exceptions for specific error cases:

```python
from gridporter.exceptions import (
    GridPorterError,
    FileNotFoundError,
    UnsupportedFormatError,
    DetectionError,
    ConfigurationError
)

try:
    result = await gridporter.detect_tables("file.xlsx")
except FileNotFoundError:
    print("File not found")
except UnsupportedFormatError:
    print("File format not supported")
except DetectionError as e:
    print(f"Detection failed: {e}")
```

## Best Practices

1. **Enable Feature Collection** for production deployments to improve detection over time:
   ```python
   gridporter = GridPorter(enable_feature_collection=True)
   ```

2. **Use Confidence Thresholds** appropriate for your use case:
   ```python
   # Higher threshold for critical data
   gridporter = GridPorter(confidence_threshold=0.9)

   # Lower threshold for exploration
   gridporter = GridPorter(confidence_threshold=0.5)
   ```

3. **Handle Complex Tables** explicitly when needed:
   ```python
   if result.metadata.get('has_complex_tables'):
       # Process with special handling
       pass
   ```

4. **Monitor Performance** using telemetry:
   ```python
   collector = get_feature_collector()
   stats = collector.get_summary_statistics()
   print(f"Avg processing time: {stats['avg_processing_time_ms']}ms")
   ```

## Examples

### Complete Example: Financial Report Processing

```python
import asyncio
from gridporter import GridPorter
from gridporter.telemetry import get_feature_collector

async def process_financial_report():
    # Initialize with all features
    gridporter = GridPorter(
        use_vision=True,
        suggest_names=True,
        enable_feature_collection=True,
        confidence_threshold=0.8
    )

    # Process the report
    result = await gridporter.detect_tables("quarterly_report.xlsx")

    # Handle each sheet
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")

        for table in sheet.tables:
            print(f"\nTable: {table.suggested_name or table.range}")
            print(f"Confidence: {table.confidence:.2%}")

            # Check for multi-row headers
            if table.multi_row_headers:
                print(f"Multi-row headers: {table.multi_row_headers} rows")

                # Show column hierarchy
                if table.column_hierarchy:
                    for col, hierarchy in table.column_hierarchy.items():
                        print(f"  Column {col}: {' > '.join(hierarchy)}")

            # Check semantic features
            if table.semantic_features:
                features = table.semantic_features
                if features.get('has_subtotals'):
                    print("  Contains subtotals")
                if features.get('section_count', 0) > 0:
                    print(f"  Has {features['section_count']} sections")

    # Get feature statistics
    collector = get_feature_collector()
    stats = collector.get_summary_statistics()
    print(f"\nProcessed {stats['total_records']} tables")
    print(f"Average confidence: {stats['avg_confidence']:.2%}")

# Run the example
asyncio.run(process_financial_report())
```

## Version History

### v0.2.1 (2025-07-27)
- Added `ComplexTableAgent` for semantic understanding
- Added `MultiHeaderDetector` for multi-row headers
- Added `MergedCellAnalyzer` for merged cell handling
- Added `SemanticFormatAnalyzer` for structure analysis
- Added comprehensive feature collection system
- Enhanced `TableInfo` with semantic features

### v0.2.0 (2025-07-25)
- Vision infrastructure and region verification
- Basic table detection capabilities

### v0.1.0 (2025-07-23)
- Initial release with project foundation
