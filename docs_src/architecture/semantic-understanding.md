# Semantic Understanding in GridGulp

## Overview

GridGulp v0.2.1 introduces semantic understanding capabilities that go beyond simple cell detection to understand the meaning and structure of complex spreadsheets. This guide explains how GridGulp interprets spreadsheet semantics and how to leverage these features.

## Table of Contents

1. [What is Semantic Understanding?](#what-is-semantic-understanding)
2. [Multi-Row Headers](#multi-row-headers)
3. [Merged Cell Analysis](#merged-cell-analysis)
4. [Semantic Structure Analysis](#semantic-structure-analysis)
5. [Format Pattern Detection](#format-pattern-detection)
6. [Complex Table Agent](#complex-table-agent)
7. [Real-World Examples](#real-world-examples)
8. [Best Practices](#best-practices)

## What is Semantic Understanding?

Semantic understanding in GridGulp means recognizing not just where data is located, but what it represents and how it's structured. This includes:

- **Hierarchical Headers**: Understanding multi-level column structures
- **Data Sections**: Identifying logical groupings within tables
- **Summary Rows**: Detecting subtotals, totals, and aggregations
- **Format Semantics**: Interpreting formatting as structural indicators
- **Relationship Detection**: Understanding how different parts relate

## Multi-Row Headers

Many enterprise spreadsheets use multiple header rows to create hierarchical column structures. GridGulp's `MultiHeaderDetector` automatically identifies and maps these structures.

### Example: Financial Report Headers

```
|   Department    |        Q1 Sales         |        Q2 Sales         |
|                 |  Units  |  Revenue ($) |  Units  |  Revenue ($) |
|-----------------|---------|--------------|---------|--------------|
| North Region    |   100   |    10,000   |   120   |    12,000   |
```

GridGulp detects:
- 2 header rows
- Column hierarchy:
  - Columns 1-2: "Q1 Sales" → "Units" / "Revenue ($)"
  - Columns 3-4: "Q2 Sales" → "Units" / "Revenue ($)"

### Using Multi-Row Header Detection

```python
from gridgulp import GridGulp

gridgulp = GridGulp()
result = await gridgulp.detect_tables("financial_report.xlsx")

for table in result.sheets[0].tables:
    if table.multi_row_headers:
        print(f"Header rows: {table.multi_row_headers}")

        # Access column hierarchy
        for col_idx, hierarchy in table.column_hierarchy.items():
            print(f"Column {col_idx}: {' > '.join(hierarchy)}")
```

## Merged Cell Analysis

Merged cells often indicate structural relationships in spreadsheets. The `MergedCellAnalyzer` interprets these relationships to understand table structure.

### Types of Merged Cells

1. **Header Spans**: Merged cells in headers indicate grouped columns
2. **Row Groups**: Vertical merges often indicate data categories
3. **Summary Cells**: Merged cells at edges often contain totals

### Example: Pivot Table Structure

```python
# GridGulp automatically detects merged cells
analyzer = MergedCellAnalyzer()
merged_cells = analyzer.analyze_merged_cells(sheet_data)

for cell in merged_cells:
    print(f"Merged cell '{cell.value}' spans {cell.row_span}x{cell.col_span}")
    if cell.is_header:
        print("  Identified as header cell")
```

## Semantic Structure Analysis

The `SemanticFormatAnalyzer` identifies the meaning of different rows based on content and formatting patterns.

### Row Types

GridGulp classifies rows into semantic types:

- **HEADER**: Column headers (often bold, at top)
- **DATA**: Regular data rows
- **SECTION_HEADER**: Section dividers (e.g., "Q1 Results")
- **SUBTOTAL**: Intermediate summaries
- **TOTAL**: Grand totals (usually at bottom)
- **BLANK**: Semantically meaningful blank rows
- **SEPARATOR**: Visual separators

### Example: Financial Statement Analysis

```python
analyzer = SemanticFormatAnalyzer()
structure = analyzer.analyze_semantic_structure(sheet_data, table_range)

# Understand the table structure
for row in structure.semantic_rows:
    print(f"Row {row.row_index}: {row.row_type.value}")

# Identify sections
for start, end in structure.sections:
    print(f"Section from row {start} to {end}")
```

### Detecting Totals and Subtotals

GridGulp identifies summary rows through:
- Keywords ("Total", "Subtotal", "Sum")
- Formatting (bold, borders, background)
- Position (bottom of sections/table)
- Formula patterns

## Format Pattern Detection

Formatting often carries semantic meaning. GridGulp detects and interprets format patterns.

### Common Patterns

1. **Alternating Row Colors**: Data grouping
2. **Bold Headers**: Column identifiers
3. **Indentation**: Hierarchical relationships
4. **Borders**: Section boundaries
5. **Background Colors**: Category indicators

### Example: Detecting Hierarchical Data

```python
# GridGulp detects indentation patterns
financial_data = """
Revenue                    1,000,000
  Product Sales              700,000
    Hardware                 400,000
    Software                 300,000
  Service Revenue            300,000
Total Revenue             1,000,000
"""

# Automatically detected as hierarchical with 3 levels
```

## Complex Table Agent

The `ComplexTableAgent` orchestrates all semantic understanding components to handle sophisticated spreadsheet structures.

### Key Capabilities

1. **Multi-Strategy Detection**: Combines multiple detection methods
2. **Confidence Scoring**: Weighted scoring based on multiple factors
3. **Format Preservation**: Maintains semantic blank rows and formatting
4. **Integration**: Works with vision pipeline for enhanced detection

### Example: Complete Semantic Analysis

```python
from gridgulp.agents import ComplexTableAgent
from gridgulp.config import Config

config = Config(
    use_vision=True,
    confidence_threshold=0.7
)

agent = ComplexTableAgent(config)
result = await agent.detect_complex_tables(sheet_data)

for table in result.tables:
    print(f"\nTable: {table.range}")
    print(f"Confidence: {table.confidence:.2%}")

    # Semantic features
    features = table.semantic_features
    print(f"Has sections: {features.get('section_count', 0) > 0}")
    print(f"Has subtotals: {features.get('has_subtotals', False)}")
    print(f"Has grand total: {features.get('has_grand_total', False)}")
    print(f"Multi-row headers: {features.get('has_multi_headers', False)}")
```

## Real-World Examples

### Example 1: Quarterly Sales Report

```python
# Input structure:
# - Multi-level headers (Region > Quarter > Metric)
# - Subtotals per region
# - Grand total at bottom
# - Format-based sections

result = await gridgulp.detect_tables("quarterly_sales.xlsx")
table = result.sheets[0].tables[0]

# GridGulp identifies:
# - 3 header rows with hierarchy
# - 4 sections (one per region)
# - 4 subtotal rows
# - 1 grand total row
# - Preserved blank rows between sections
```

### Example 2: Financial Statement

```python
# Input structure:
# - Indented account hierarchy
# - Bold section headers
# - Multiple subtotal levels
# - Formatted total rows

# GridGulp automatically:
# - Detects indentation levels (up to 4 deep)
# - Identifies account groupings
# - Preserves calculation structure
# - Maps parent-child relationships
```

### Example 3: Pivot Table

```python
# Input structure:
# - Row and column dimensions
# - Merged cells for grouping
# - Nested totals
# - Cross-tabulation data

# GridGulp handles:
# - Multi-dimensional headers
# - Merged cell interpretation
# - Aggregate identification
# - Dimension mapping
```

## Best Practices

### 1. Configure for Your Use Case

```python
# For financial reports with complex structure
config = Config(
    use_vision=True,
    confidence_threshold=0.8,
    enable_feature_collection=True
)

# For simple data extraction
config = Config(
    use_vision=False,
    confidence_threshold=0.6
)
```

### 2. Leverage Semantic Features

```python
# Check for complex structures
if table.semantic_features.get('section_count', 0) > 1:
    # Process sections separately
    for section in get_sections(table):
        process_section(section)

# Handle hierarchical data
if table.semantic_features.get('max_hierarchy_depth', 0) > 1:
    # Build tree structure
    tree = build_hierarchy(table)
```

### 3. Preserve Semantic Meaning

```python
# Don't remove "empty" rows - they may be semantic
preserved_rows = table.semantic_features.get('preserved_blank_rows', [])

# Maintain format-based groupings
if table.semantic_features.get('has_alternating_rows'):
    # Process in pairs
    process_grouped_data(table)
```

### 4. Use Feature Collection

Enable feature collection to improve detection over time:

```python
gridgulp = GridGulp(
    enable_feature_collection=True,
    feature_db_path="~/.gridgulp/features.db"
)

# Later, analyze patterns
from gridgulp.telemetry import get_feature_collector
collector = get_feature_collector()
stats = collector.get_summary_statistics()
```

## Advanced Topics

### Custom Semantic Rules

You can extend GridGulp with custom semantic rules:

```python
from gridgulp.detectors import SemanticFormatAnalyzer

class CustomAnalyzer(SemanticFormatAnalyzer):
    def __init__(self):
        super().__init__(
            section_keywords=["Division", "Department", "Region"],
            total_keywords=["Grand Total", "Sum", "Overall"]
        )
```

### Confidence Tuning

Semantic detection confidence is based on multiple factors:

- Header complexity (merged cells, multiple rows)
- Format consistency (regular patterns)
- Keyword presence (section/total indicators)
- Structure regularity (consistent indentation)

Adjust thresholds based on your data:

```python
# For well-structured reports
agent = ComplexTableAgent(Config(
    min_header_confidence=0.8,
    min_section_confidence=0.7
))

# For inconsistent data
agent = ComplexTableAgent(Config(
    min_header_confidence=0.5,
    min_section_confidence=0.4
))
```

### Performance Optimization

For large spreadsheets with complex semantics:

```python
# Use sampling for initial detection
config = Config(
    enable_sampling=True,
    sample_size=1000
)

# Disable expensive features if not needed
config = Config(
    detect_indentation=False,  # Skip hierarchy detection
    detect_formats=False       # Skip format analysis
)
```

## Troubleshooting

### Common Issues

1. **Headers Not Detected**
   - Check for consistent formatting (bold, background)
   - Ensure headers are at table start
   - Verify merged cells have values

2. **Sections Missed**
   - Add section keywords to configuration
   - Check for visual separators (blank rows, borders)
   - Ensure consistent section formatting

3. **Wrong Total Detection**
   - Verify total keywords match your data
   - Check position (totals usually at bottom)
   - Ensure consistent formatting

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger("gridgulp.semantic").setLevel(logging.DEBUG)

# This will show:
# - Row classification decisions
# - Confidence calculations
# - Pattern matching details
```

## Summary

GridGulp's semantic understanding transforms spreadsheet ingestion from a mechanical process to an intelligent interpretation. By understanding not just the data but its meaning and structure, GridGulp can handle the complex, real-world spreadsheets that traditional parsers fail on.

Key benefits:
- **Automatic Structure Detection**: No manual configuration needed
- **Preservation of Meaning**: Maintains semantic relationships
- **Flexible Interpretation**: Adapts to different spreadsheet styles
- **Continuous Improvement**: Learns from usage patterns

For more examples and use cases, see the [examples directory](https://github.com/Ganymede-Bio/gridgulp/tree/main/examples/).
