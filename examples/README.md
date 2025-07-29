# GridGulp Example Spreadsheets

This directory contains sample spreadsheet files to test and demonstrate GridGulp's table detection capabilities across different formats and complexities.

## Directory Structure

```
examples/
├── spreadsheets/    # Test spreadsheet files
│   ├── simple/      # Single-table files with clear structure
│   ├── sales/       # Sales and business data examples
│   ├── financial/   # Financial statements and accounting data
│   └── complex/     # Multi-table layouts and complex structures
├── basic_usage.py   # Basic GridGulp usage
├── feature_collection_example.py  # Feature collection demo
├── week5_complex_tables_with_features.py  # Complex table detection (v0.2.1)
└── week5_feature_collection_example.py   # Advanced feature analysis (v0.2.1)
```

## File Categories

### Simple Examples (`simple/`)

**`basic_table.csv`**
- Single table with employee data
- Clear headers, consistent data types
- Good for testing basic detection algorithms

**`product_inventory.csv`**
- Product inventory with mixed data types
- Includes dates, numbers, text
- Tests data type inference capabilities

### Sales Examples (`sales/`)

**`monthly_sales.csv`**
- Sales performance data by region and month
- Multiple dimensions (time, geography, product)
- Suitable for testing aggregation and grouping

### Financial Examples (`financial/`)

**`income_statement.csv`**
- Quarterly P&L statement
- Hierarchical structure with subtotals
- Tests handling of financial formatting and calculations

**`balance_sheet.csv`**
- Standard balance sheet format
- Multiple sections with headers
- Tests detection of grouped financial data

### Complex Examples (`complex/`)

**`multi_table_report.csv`**
- Multiple distinct tables in one file
- Different table structures and purposes
- Tests ability to separate and identify multiple tables

## New in v0.2.1: Semantic Understanding Examples

### Complex Table Detection with Multi-Row Headers

**`week5_complex_tables_with_features.py`**

Demonstrates GridGulp's new semantic understanding capabilities:
- Multi-row header detection with merged cells
- Financial report analysis with sections and subtotals
- Hierarchical data structure recognition
- Feature collection for continuous improvement

```python
# Run the example
python examples/week5_complex_tables_with_features.py
```

This example creates and analyzes:
1. Complex financial report with multi-level headers
2. Sales dashboard with pivot-table structure
3. Hierarchical financial statements
4. Multi-section reports with subtotals

### Feature Collection and Analysis

**`week5_feature_collection_example.py`**

Shows how to use the feature collection system:
- Enable telemetry collection
- Analyze detection patterns
- Export features for analysis
- Improve detection accuracy over time

```python
# Run the example
python examples/week5_feature_collection_example.py
```

Features collected include:
- Geometric properties (rectangularness, density)
- Pattern characteristics (headers, orientation)
- Format features (bold headers, totals)
- Performance metrics

## Usage Examples

### Basic Detection

```python
import asyncio
from gridgulp import GridGulp

async def test_simple_detection():
    porter = GridGulp()

    # Test basic table detection
    result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")
    print(f"Detected {result.total_tables} tables")

    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"Table: {table.suggested_name}")
            print(f"Range: {table.range.excel_range}")
            print(f"Confidence: {table.confidence:.2%}")

asyncio.run(test_simple_detection())
```

### Complex Layout Analysis

```python
async def test_complex_detection():
    porter = GridGulp(
        use_local_llm=True,  # Use Ollama for better table naming
        ollama_vision_model="qwen2.5vl:7b"  # Vision model for layout analysis
    )

    # Test multi-table detection
    result = await porter.detect_tables("examples/spreadsheets/complex/multi_table_report.csv")

    print(f"File: {result.file_info.path}")
    print(f"Total tables detected: {result.total_tables}")
    print(f"Detection methods used: {', '.join(result.methods_used)}")

    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for i, table in enumerate(sheet.tables, 1):
            print(f"  Table {i}: {table.suggested_name}")
            print(f"    Range: {table.range.excel_range}")
            print(f"    Confidence: {table.confidence:.2%}")
            print(f"    Headers: {table.headers[:5]}")  # First 5 headers

asyncio.run(test_complex_detection())
```

### Batch Processing

```python
async def test_batch_processing():
    porter = GridGulp()

    # Process all example files
    files = [
        "examples/spreadsheets/simple/basic_table.csv",
        "examples/spreadsheets/sales/monthly_sales.csv",
        "examples/spreadsheets/financial/balance_sheet.csv",
        "examples/spreadsheets/complex/multi_table_report.csv"
    ]

    results = await porter.batch_detect(files)

    print("Batch Processing Results:")
    print("-" * 50)

    for result in results:
        file_name = result.file_info.path.name
        print(f"{file_name:<25} | {result.total_tables:>2} tables | {result.detection_time:>6.2f}s")

asyncio.run(test_batch_processing())
```

## Model Testing Scenarios

### DeepSeek-R1 Text Model Testing

These examples are designed to test the reasoning capabilities of the DeepSeek-R1 model:

1. **Table Naming**: How well does it generate meaningful names for detected table regions?
2. **Data Type Recognition**: Can it identify patterns in financial vs. sales vs. inventory data?
3. **Hierarchical Understanding**: Does it recognize parent-child relationships in financial statements?

### Qwen2.5-VL Vision Model Testing

These examples test vision model capabilities:

1. **Layout Analysis**: Identifying table boundaries in complex layouts
2. **Structure Recognition**: Understanding table headers, data rows, and summary sections
3. **Multi-table Detection**: Separating distinct tables within the same sheet

## Performance Benchmarks

Expected performance characteristics for different file types:

| File Type | Expected Tables | Complexity | DeepSeek-R1 Use Cases | Qwen2.5-VL Use Cases |
|-----------|----------------|------------|---------------------|-------------------|
| Simple CSV | 1 | Low | Basic naming | Layout verification |
| Sales Data | 1-2 | Medium | Category recognition | Region grouping |
| Financial | 2-3 | Medium-High | Account classification | Section identification |
| Multi-table | 3-5 | High | Semantic naming | Boundary detection |

## Creating Custom Examples

To add your own test files:

1. **Choose appropriate directory** based on complexity
2. **Include variety** in data types, structures, and formats
3. **Document expected behavior** in comments or separate notes
4. **Test with both models** to compare capabilities

### Example File Naming Convention

```
{category}_{description}_{complexity}.{ext}

Examples:
- simple_employee_data_basic.csv
- financial_income_statement_standard.xlsx
- complex_multi_region_sales_advanced.xlsx
```

## Running All Examples

To run all example scripts:

```bash
# Basic usage
python examples/basic_usage.py

# Feature collection
python examples/feature_collection_example.py

# Week 5 complex tables (requires pandas)
python examples/week5_complex_tables_with_features.py

# Week 5 feature analysis
python examples/week5_feature_collection_example.py
```

### Requirements

- Python 3.10+
- GridGulp installed (`pip install -e .`)
- For Week 5 examples: `pip install pandas openpyxl`
- Optional: Ollama with models for enhanced detection

## Troubleshooting

### Common Issues

**No tables detected:**
- Check if file has clear headers
- Verify data is properly structured
- Ensure file is not corrupted

**Multiple false positives:**
- Adjust confidence threshold
- Use vision model for better layout understanding
- Check for spurious data patterns

**Poor table naming:**
- Ensure LLM models are properly loaded
- Try different model sizes (1.5b vs 7b vs 32b)
- Check if sufficient context is provided

### Model-Specific Tips

**DeepSeek-R1:**
- Works best with structured, logical data
- Excels at financial and business terminology
- Provide clear column headers for better naming

**Qwen2.5-VL:**
- Better for complex visual layouts
- Handles merged cells and irregular structures
- Use for files with charts, images, or complex formatting

## Contributing Examples

To contribute new example files:

1. Fork the repository
2. Add files to appropriate subdirectory
3. Update this README with descriptions
4. Test with both Ollama models
5. Submit pull request with test results

Include in your PR:
- File description and expected behavior
- Test results with different model configurations
- Any special setup or requirements
