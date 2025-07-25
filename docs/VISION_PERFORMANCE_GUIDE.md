# Vision Module Performance Guide

This guide explains the performance optimizations in the GridPorter vision module.

## Overview

The vision module has been optimized for efficient table detection:

1. **Fast Excel Reading**: CalamineReader provides 10-100x faster Excel file reading
2. **Telemetry Integration**: Automatic performance tracking using OpenTelemetry
3. **Efficient Algorithms**: Optimized sparse cell access patterns for vision tasks
4. **Smart Compression**: Automatic bitmap compression for large sheets

## Key Components

### 1. BitmapGenerator

Optimized for memory efficiency and speed:

```python
from gridporter.vision import BitmapGenerator

generator = BitmapGenerator()
# Efficiently generates bitmaps from sheet data
image_bytes, metadata = generator.generate(sheet_data)
```

Features:
- Automatic compression for large regions
- Configurable cell dimensions
- Multiple visualization modes (binary, grayscale, color)

### 2. BitmapAnalyzer

Fast pattern analysis using image processing:

```python
from gridporter.vision.bitmap_analyzer import BitmapAnalyzer

analyzer = BitmapAnalyzer()
# Generates binary bitmap for analysis
bitmap, metadata = analyzer.generate_binary_bitmap(sheet_data)
```

### 3. IntegratedVisionPipeline

Complete pipeline for table detection:

```python
from gridporter.vision.integrated_pipeline import IntegratedVisionPipeline

pipeline = IntegratedVisionPipeline()
result = pipeline.process_sheet(sheet_data)
```

## Performance Gains

### Excel Reading Performance

| Reader | Speed | Use Case |
|--------|-------|----------|
| CalamineReader | 10-100x faster | Default, most files |
| ExcelReader (openpyxl) | Baseline | When formatting details needed |

### Vision Operations

The vision module is optimized for sparse cell access patterns:
- Efficient iteration over non-empty cells
- Smart region density calculations
- Minimal memory overhead

## Usage Examples

### Basic Usage

```python
from gridporter.models.sheet_data import SheetData
from gridporter.vision import BitmapGenerator

# Create and populate sheet
sheet = SheetData(name="MySheet")
# ... add cells ...

# Generate bitmap
generator = BitmapGenerator()
image_bytes, metadata = generator.generate(sheet)
```

### With Fast Excel Reading

```python
from gridporter.readers import CalamineReader
from gridporter.models import FileInfo, FileType

# Read Excel file with Calamine (10-100x faster)
file_info = FileInfo(
    path="large_file.xlsx",
    type=FileType.XLSX,
    size=file_size
)

reader = CalamineReader(file_path, file_info)
file_data = await reader.read()

# Process sheets
for sheet in file_data.sheets:
    result = pipeline.process_sheet(sheet)
```

### With Telemetry

```python
from gridporter.telemetry import get_metrics_collector

# Telemetry is automatic when available
metrics = get_metrics_collector()

# Generate bitmap - timing is tracked automatically
generator = BitmapGenerator()
image_bytes, metadata = generator.generate(sheet)

# View metrics
totals = metrics.get_totals()
print(f"Bitmap generation time: {totals.get('bitmap_generation', 0):.3f}s")
```

### Export to DataFrames for Analysis

After detection, you can export to Polars or Pandas for further analysis:

```python
# Detect tables
result = pipeline.process_sheet(sheet)

# Export to Polars DataFrame (requires polars)
from gridporter.readers import CalamineReader
reader = CalamineReader(file_path, file_info)
dataframes = reader.read_to_polars()  # Direct to Polars

# Or convert detected ranges to your preferred format
for table in result.detected_tables:
    # Extract the detected table range
    # Convert to DataFrame for analysis
    pass
```

## Running Benchmarks

### Performance Test Suite

```bash
python tests/manual/vision_performance_test.py
```

This runs comprehensive benchmarks showing:
- Bitmap generation performance
- Pattern detection speed
- Full pipeline timing
- Excel reading comparisons

### Week 3 Performance Demo

```bash
python tests/manual/week3_performance_demo.py
```

Interactive demo showing:
- Real-time performance metrics
- Visual output examples
- Configuration options

## Configuration

### GridPorterConfig Options

```python
config = GridPorterConfig(
    # Performance options
    excel_reader="calamine",   # Fast Excel reader (default)
    enable_telemetry=True,     # Track performance

    # Vision-specific
    vision_cell_width=10,      # Cell dimensions in pixels
    vision_cell_height=10,
    vision_mode="binary",      # Bitmap mode
)
```

### Environment Variables

```bash
# Performance features
export GRIDPORTER_EXCEL_READER=calamine
export GRIDPORTER_ENABLE_TELEMETRY=true

# Vision settings
export GRIDPORTER_VISION_CELL_WIDTH=10
export GRIDPORTER_VISION_CELL_HEIGHT=10
export GRIDPORTER_VISION_MODE=binary
```

## Best Practices

### For Large Files

1. **Use CalamineReader**: 10-100x faster than openpyxl
2. **Enable auto-compression**: BitmapGenerator handles this automatically
3. **Process in chunks**: For extremely large files

### For Complex Sheets

1. **Adjust detection parameters**: Tune min_table_size and density thresholds
2. **Use visualization regions**: The quadtree analyzer optimizes what to send to LLMs
3. **Enable telemetry**: Identify bottlenecks in your specific use case

## Troubleshooting

### Slow Excel Reading

Ensure you're using CalamineReader:
```python
config = GridPorterConfig(excel_reader="calamine")
```

### Memory Issues

1. The bitmap generator automatically compresses large regions
2. For very large sheets, process in sections
3. Consider streaming approaches for massive files

### Missing Dependencies

For full performance features:
```bash
pip install python-calamine  # Fast Excel reading
pip install opentelemetry-instrumentation-openai  # LLM tracking
```

## Architecture Notes

GridPorter is designed for **table detection**, not bulk data processing:
- Optimized for sparse cell access patterns
- Efficient bitmap-based analysis
- Smart region detection algorithms

For post-detection data analysis, export to your preferred DataFrame library (Polars, Pandas, etc.).

## Future Enhancements

- GPU acceleration for bitmap operations
- Parallel processing for multi-sheet files
- Incremental detection for real-time updates
- Advanced ML-based table detection
