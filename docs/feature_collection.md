# Feature Collection for Table Detection

GridPorter includes a comprehensive feature collection system that captures detailed metrics about table detection for analysis and future improvements.

## Overview

The feature collection system records detailed information about each table detection attempt, including:
- Geometric features (rectangularness, density, contiguity)
- Pattern features (pattern type, orientation, headers)
- Format features (bold headers, totals, sections)
- Content features (cell counts, data types)
- Detection metadata (confidence, method, timing)

All features are stored in a local SQLite database for easy querying and analysis.

## Configuration

Feature collection is disabled by default. Enable it using:

### Environment Variable
```bash
export GRIDPORTER_ENABLE_FEATURE_COLLECTION=true
export GRIDPORTER_FEATURE_DB_PATH=~/.gridporter/features.db
export GRIDPORTER_FEATURE_RETENTION_DAYS=30
```

### Python Configuration
```python
from gridporter import GridPorter

gridporter = GridPorter(
    enable_feature_collection=True,
    feature_db_path="~/.gridporter/features.db"
)
```

## Collected Features

### Geometric Features
- **rectangularness**: How well the data fits a rectangular shape (0-1)
- **filledness**: Ratio of filled cells to total cells (0-1)
- **density**: Data density considering sparsity patterns (0-1)
- **contiguity**: How connected the data regions are (0-1)
- **edge_quality**: Quality of table boundaries (0-1)
- **aspect_ratio**: Width/height ratio of the table
- **size_ratio**: Table size relative to sheet size (0-1)

### Pattern Features
- **pattern_type**: Type of pattern detected (header_data, matrix, form, etc.)
- **orientation**: Table orientation (horizontal, vertical, matrix)
- **has_multi_headers**: Whether multiple header rows were detected
- **header_row_count**: Number of header rows
- **fill_ratio**: Overall data fill ratio
- **header_density**: Density of headers

### Format Features
- **header_row_count**: Number of header rows
- **has_bold_headers**: Whether headers use bold formatting
- **has_totals**: Whether total rows were detected
- **has_subtotals**: Whether subtotal rows were detected
- **section_count**: Number of sections in the table
- **separator_row_count**: Number of separator rows

### Content Features
- **total_cells**: Total cells in the table region
- **filled_cells**: Number of non-empty cells
- **numeric_ratio**: Ratio of numeric cells (0-1)
- **date_columns**: Number of date columns
- **text_columns**: Number of text columns
- **empty_cell_ratio**: Ratio of empty cells (0-1)

### Hierarchical Features
- **max_hierarchy_depth**: Maximum indentation depth
- **has_indentation**: Whether indentation was detected
- **subtotal_count**: Number of subtotal rows

## Using the Feature Store

### Query Features
```python
from gridporter.telemetry import get_feature_collector

collector = get_feature_collector()

# Get all features
all_features = collector._feature_store.query_features()

# Query high-confidence detections
high_conf = collector._feature_store.query_features(
    min_confidence=0.9,
    detection_method="complex_detection"
)

# Query by file
file_features = collector._feature_store.query_features(
    file_path="/path/to/file.xlsx"
)
```

### Get Statistics
```python
stats = collector.get_summary_statistics()
print(f"Total detections: {stats['total_records']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
print(f"Success rate: {stats['success_rate']:.1%}")

# By detection method
for method, info in stats['by_method'].items():
    print(f"{method}: {info['count']} detections")
```

### Export for Analysis
```python
# Export to CSV for analysis in pandas/Excel
collector.export_features("features.csv")

# Export with filters
collector.export_features(
    "high_conf_features.csv",
    min_confidence=0.8,
    success_only=True
)
```

## Database Schema

The SQLite database uses a single `detection_features` table with indexes on:
- timestamp
- confidence
- detection_method
- file_path
- detection_success

## Privacy and Performance

- Feature collection is completely local - no data is sent externally
- The SQLite database is stored in your home directory by default
- Feature collection adds minimal overhead (<5ms per detection)
- Old data is automatically cleaned up based on retention settings

## Using Features for Analysis

The collected features can be used for:

1. **Performance Analysis**: Identify which detection methods work best for different table types
2. **Confidence Calibration**: Understand what features correlate with successful detection
3. **Pattern Discovery**: Find common patterns in your spreadsheets
4. **Debugging**: Investigate why certain tables weren't detected properly

### Example: Analyzing Features with Pandas
```python
import pandas as pd
import sqlite3

# Connect to feature database
conn = sqlite3.connect("~/.gridporter/features.db")

# Load features into pandas
df = pd.read_sql_query("SELECT * FROM detection_features", conn)

# Analyze confidence by pattern type
pattern_confidence = df.groupby('pattern_type')['confidence'].agg(['mean', 'std', 'count'])
print(pattern_confidence)

# Find features that correlate with high confidence
high_conf = df[df['confidence'] > 0.9]
low_conf = df[df['confidence'] < 0.5]

# Compare geometric features
print("High confidence tables:")
print(high_conf[['rectangularness', 'filledness', 'density']].mean())
print("\nLow confidence tables:")
print(low_conf[['rectangularness', 'filledness', 'density']].mean())
```

## Future Enhancements

The feature collection system is designed to support future improvements:

1. **LLM-based Confidence**: Features can be passed to an LLM for more intelligent confidence scoring
2. **Adaptive Detection**: Learn from successful detections to improve future performance
3. **Custom Models**: Train lightweight models on collected features for specific use cases
4. **Feedback Loop**: User corrections can be incorporated to improve detection

## Disabling Feature Collection

To disable feature collection:
1. Set `GRIDPORTER_ENABLE_FEATURE_COLLECTION=false`
2. Or pass `enable_feature_collection=False` to GridPorter
3. Delete the database file at `~/.gridporter/features.db` to remove all collected data
