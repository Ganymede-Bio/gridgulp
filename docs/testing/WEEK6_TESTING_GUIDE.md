# Week 6 Manual Testing Guide

## Overview
This guide provides comprehensive manual testing procedures for Week 6 features: Excel metadata extraction, simple case detection, island detection, and cost optimization in the hybrid detection pipeline.

## Test Environment Setup

### 1. Python Environment
```bash
# Ensure you're in the gridporter directory
cd ~/dev/gridporter

# Install development dependencies
pip install -e .

# Install Excel libraries
pip install openpyxl xlrd
```

### 2. Configuration File
Create a test configuration file `test_config.json`:
```json
{
  "use_excel_metadata": true,
  "enable_simple_case_detection": true,
  "enable_island_detection": true,
  "max_cost_per_file": 0.10,
  "max_cost_per_session": 1.00,
  "confidence_threshold": 0.8,
  "use_vision": false,
  "enable_cache": true,
  "log_level": "DEBUG"
}
```


### 3. Test Data Preparation

#### Create Test Excel Files

**File 1: simple_table.xlsx**
- Single table starting at A1
- Headers in row 1 (bold formatting)
- 10 rows of data
- No gaps or empty cells
- Expected: Simple case detection with high confidence

**File 2: excel_tables.xlsx**
- Sheet1: Create a ListObject (Insert > Table) named "SalesData" (A1:D10)
- Sheet2: Create another ListObject "Products" (B2:E20)
- Add Named Range "TotalRevenue" pointing to Sheet1!D11
- Set Print Area on Sheet1 (A1:D15)
- Expected: Excel metadata extraction with 95% confidence

**File 3: multi_tables.xlsx**
- Table 1: A1:C10 (Sales data)
- Empty columns D-E
- Table 2: F1:H10 (Inventory)
- Empty rows 11-12
- Table 3: A14:D20 (Summary)
- Expected: Island detection finds 3 separate tables

**File 4: complex_layout.xlsx**
- Merged cells in header (A1:C1)
- Subtotal rows with formatting
- Sparse data with gaps
- Multiple small tables
- Expected: Requires vision or returns low confidence

## Test Scenarios

### A. Simple Case Detection Tests

#### Test A1: Perfect Simple Table
```python
from gridporter import GridPorter
from gridporter.config import Config

config = Config.from_env()
config.use_vision = False  # Force traditional methods
gp = GridPorter(config)

result = await gp.detect_tables('simple_table.xlsx')

# Verify:
assert len(result.tables) == 1
assert result.tables[0].detection_method == "simple_case"
assert result.tables[0].confidence >= 0.9
assert result.detection_metadata['cost_report']['total_cost_usd'] == 0
```

#### Test A2: Table Not Starting at A1
Create a table starting at B2 instead of A1.
- Expected: Simple case fails, falls back to island detection

#### Test A3: Table with Empty Rows
Insert empty rows within the data.
- Expected: Simple case fails due to gaps

#### Test A4: Table with Headers Only
Create a file with only header row, no data.
- Expected: Simple case detects but low confidence

#### Test A5: Dense vs Sparse Data
Create tables with varying data density (50%, 75%, 90% filled).
- Expected: Confidence correlates with density

### B. Island Detection Tests

#### Test B1: Two Distinct Tables
```python
# Create sheet with two separate tables
result = await gp.detect_tables('multi_tables.xlsx')

# Verify:
assert len(result.tables) >= 2
assert all(t.detection_method == "island_detection" for t in result.tables)
assert all(t.confidence >= 0.7 for t in result.tables)
```

#### Test B2: Tables with Different Sizes
- Small table (2x2)
- Medium table (10x5)
- Large table (100x20)
- Expected: Larger tables have higher confidence

#### Test B3: Adjacent Tables (1 cell gap)
Tables separated by single empty row/column.
- Expected: Detected as separate islands

#### Test B4: Diagonal Tables
Tables positioned diagonally from each other.
- Expected: Each detected as separate island

#### Test B5: Nested Tables
One table containing another smaller table.
- Expected: May detect as single large island

### C. Excel Metadata Extraction Tests

#### Test C1: ListObjects Detection
```python
# File with Excel Tables (ListObjects)
result = await gp.detect_tables('excel_tables.xlsx')

# Verify:
metadata_tables = [t for t in result.tables if t.detection_method == "excel_metadata"]
assert len(metadata_tables) >= 1
assert any(t.suggested_name == "SalesData" for t in metadata_tables)
assert all(t.confidence >= 0.95 for t in metadata_tables)
```

#### Test C2: Named Ranges as Tables
Create named ranges covering table-like regions.
- Expected: Detected with 70% confidence

#### Test C3: Print Areas
Set print areas covering multiple tables.
- Expected: Used as hints with 50% confidence

#### Test C4: Hidden ListObjects
Create hidden Excel tables.
- Expected: Still detected

#### Test C5: Cross-Sheet References
Named ranges referring to other sheets.
- Expected: Properly parsed with sheet references

#### Test C6: Legacy Excel Format
Test with .xls files.
- Expected: Limited metadata support, falls back to other methods

### D. Cost Optimization Tests

#### Test D1: Cost Tracking
```python
config.max_cost_per_file = 0.05
gp = GridPorter(config)

# Process multiple files
for file in ['file1.xlsx', 'file2.xlsx', 'file3.xlsx']:
    result = await gp.detect_tables(file)
    cost_report = result.detection_metadata['cost_report']

    # Verify:
    assert cost_report['total_cost_usd'] <= 0.05
    print(f"{file}: ${cost_report['total_cost_usd']:.3f}")
```

#### Test D2: Budget Exceeded
Set very low budget (e.g., $0.001).
- Expected: Only free methods used

#### Test D3: Method Priority
Disable specific methods and verify fallback.
```python
config.enable_simple_case_detection = False
# Expected: Skips to island detection
```

#### Test D4: Cache Effectiveness
Process same file twice.
- Expected: Second run uses cache, zero cost

### E. OpenAI Cost Tracking Tests

#### Test E1: Token Usage Tracking
```python
config = Config(
    openai_api_key="YOUR_KEY",
    use_vision=True,
    max_cost_per_file=0.10
)

gp = GridPorter(config)
result = await gp.detect_tables('complex_layout.xlsx')

# Verify actual costs from API response
cost_report = result.detection_metadata['cost_report']
assert cost_report['total_cost_usd'] > 0
assert cost_report['total_tokens'] > 0

# Print detailed breakdown
print(f"Model used: {result.detection_metadata.get('vision_model', 'gpt-4o-mini')}")
print(f"Total tokens: {cost_report['total_tokens']}")
print(f"Actual cost: ${cost_report['total_cost_usd']:.6f}")
```

#### Test E2: Cost Calculation Verification
```python
from gridporter.utils.openai_pricing import get_pricing_instance

pricing = get_pricing_instance()

# Test cost calculation
model = "gpt-4o-mini"
prompt_tokens = 1000
completion_tokens = 500

cost = pricing.calculate_cost(
    model_id=model,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens
)

# Verify against known pricing
expected_cost = (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
assert abs(cost - expected_cost) < 0.000001

print(pricing.format_cost_breakdown(model, prompt_tokens, completion_tokens))
```

#### Test E3: Pricing Age Check
```python
pricing = get_pricing_instance()
age_info = pricing.check_pricing_age()

print(f"Pricing last updated: {age_info['last_updated']}")
print(f"Days old: {age_info['days_old']}")
print(age_info['message'])

# Warn if pricing is outdated
if age_info['needs_update']:
    print("WARNING: Pricing data may be outdated. Check OpenAI's pricing page.")
```

#### Test E4: OpenAI Costs API (Admin Only)
```python
from gridporter.utils.openai_pricing import OpenAICostsAPI
import time

# Only run if admin key is available
admin_key = os.getenv("OPENAI_ADMIN_KEY")
if admin_key:
    costs_api = OpenAICostsAPI(admin_key)

    # Get costs for last 7 days
    start_time = int(time.time()) - (7 * 24 * 60 * 60)

    costs_data = await costs_api.get_costs(
        start_time=start_time,
        bucket_width="1d",
        limit=7
    )

    if costs_data:
        print("Daily costs for last 7 days:")
        for bucket in costs_data.get('data', []):
            print(f"  {bucket['date']}: ${bucket['cost']:.2f}")
else:
    print("Skipping Costs API test - OPENAI_ADMIN_KEY not set")
```

#### Test E5: Cost Optimizer with OpenAI Pricing
```python
from gridporter.utils.cost_optimizer import CostOptimizer, DetectionMethod

optimizer = CostOptimizer(
    max_cost_per_file=0.05,
    max_cost_per_session=0.50
)

# Simulate vision usage
usage = {
    "prompt_tokens": 500,
    "completion_tokens": 200,
}

actual_cost = await optimizer.update_with_actual_usage(
    DetectionMethod.VISION_BASIC,
    "gpt-4o-mini",
    usage
)

print(f"Actual vision cost: ${actual_cost:.6f}")
assert optimizer.tracker.total_cost_usd == actual_cost
```

### F. Hybrid Pipeline Integration Tests

#### Test F1: Full Pipeline Flow
```python
# Enable all methods with detailed logging
config = Config(
    use_excel_metadata=True,
    enable_simple_case_detection=True,
    enable_island_detection=True,
    use_vision=True,
    log_level="DEBUG"
)

result = await gp.detect_tables('complex_layout.xlsx')

# Check detection flow in logs:
# 1. Simple case attempted
# 2. Excel metadata checked
# 3. Island detection run
# 4. Vision used if confidence < threshold
```

#### Test F2: Early Exit on High Confidence
File with ListObject at 95% confidence.
- Expected: No vision processing, immediate return

#### Test F3: Multiple Methods Contributing
File with partial Excel metadata + island detection.
- Expected: Combined results from multiple methods

#### Test F4: Confidence Threshold Testing
Vary confidence_threshold (0.5, 0.7, 0.9).
- Expected: Higher threshold → more vision usage

#### Test F5: Performance Benchmarking
```python
import time

files = ['simple.xlsx', 'complex.xlsx', 'metadata.xlsx']
for file in files:
    start = time.time()
    result = await gp.detect_tables(file)
    duration = time.time() - start

    print(f"{file}:")
    print(f"  Time: {duration:.2f}s")
    print(f"  Method: {result.tables[0].detection_method}")
    print(f"  Cost: ${result.detection_metadata['cost_report']['total_cost_usd']:.3f}")
```

## Verification Checklist

### For Each Test:
- [ ] Correct detection method used
- [ ] Confidence score in expected range
- [ ] Cost tracking accurate
- [ ] Performance acceptable (<1s for simple, <5s for complex)
- [ ] No errors or exceptions
- [ ] Results match expected table count
- [ ] Metadata properly extracted

### OpenAI Cost Tracking Verification:
- [ ] Token usage extracted from API responses
- [ ] Costs calculated using hardcoded pricing table
- [ ] Pricing age check warns when outdated (>30 days)
- [ ] Optional admin API for historical costs works
- [ ] Cost optimizer uses actual token counts
- [ ] Fallback to estimates when model not in pricing table

### Cost Report Verification:
```python
cost_report = result.detection_metadata['cost_report']
print(f"Total cost: ${cost_report['total_cost_usd']}")
print(f"Total tokens: {cost_report['total_tokens']}")
print(f"API calls: {cost_report['total_api_calls']}")
print(f"Methods used: {cost_report['method_usage']}")
print(f"Remaining budget: ${cost_report['remaining_budget']}")
```

### Detection Methods Verification:
```python
methods = result.detection_metadata['methods_used']
print(f"Detection methods: {methods}")

for table in result.tables:
    print(f"Table {table.range}: {table.detection_method} (conf: {table.confidence:.2f})")
```

## Edge Cases to Test

1. **Empty Excel file** - No sheets with data
2. **Corrupted metadata** - Malformed ListObjects
3. **Circular references** - Named ranges with circular refs
4. **Very large files** - 1M+ cells
5. **Unicode in names** - Non-ASCII table names
6. **Protected sheets** - Read-only or password protected
7. **External references** - Links to other files
8. **Pivot tables** - Special Excel objects
9. **Filtered data** - Hidden rows/columns
10. **Conditional formatting** - May affect visual detection

## Performance Targets

| Scenario | Target Time | Max Cost |
|----------|------------|----------|
| Simple table (A1) | <0.1s | $0.00 |
| Excel with ListObjects | <0.2s | $0.00 |
| Multi-table with islands | <0.5s | $0.00 |
| Complex requiring vision | <5s | <$0.05 |
| Batch of 10 files | <10s | <$0.10 |

## Troubleshooting

### Common Issues:

1. **Simple case not detected**
   - Check data starts within 3 cells of A1
   - Verify no empty rows/columns in data
   - Check density > 50%

2. **Island detection missing tables**
   - Verify minimum size (2x2)
   - Check gap settings (default: 1 cell)
   - Review confidence threshold

3. **Excel metadata not found**
   - Ensure file saved after creating ListObjects
   - Check Excel version compatibility
   - Verify openpyxl installed

4. **Cost exceeding budget**
   - Check vision not called unnecessarily
   - Verify cache is enabled
   - Review confidence thresholds

## Reporting Issues

When reporting issues, include:
1. Excel file (or description of structure)
2. Configuration used
3. Expected vs actual results
4. Detection methods used
5. Cost report
6. Log output (DEBUG level)

## Automated Test Script

Save as `test_week6_features.py`:
```python
import asyncio
import os
from gridporter import GridPorter
from gridporter.config import Config
from gridporter.utils.openai_pricing import get_pricing_instance

async def test_traditional_methods():
    """Test Week 6 features without vision."""
    print("=== Testing Traditional Methods ===")
    config = Config(
        use_excel_metadata=True,
        enable_simple_case_detection=True,
        enable_island_detection=True,
        max_cost_per_file=0.05,
        confidence_threshold=0.8,
        use_vision=False,
        log_level="INFO"
    )

    gp = GridPorter(config)

    test_files = [
        "simple_table.xlsx",
        "excel_tables.xlsx",
        "multi_tables.xlsx",
        "complex_layout.xlsx"
    ]

    for file in test_files:
        print(f"\nTesting {file}:")
        try:
            result = await gp.detect_tables(file)
            print(f"  Tables found: {len(result.tables)}")
            print(f"  Methods used: {result.detection_metadata['methods_used']}")
            print(f"  Total cost: ${result.detection_metadata['cost_report']['total_cost_usd']:.3f}")

            for i, table in enumerate(result.tables):
                print(f"  Table {i+1}: {table.detection_method} (conf: {table.confidence:.2f})")

        except Exception as e:
            print(f"  ERROR: {e}")

async def test_openai_cost_tracking():
    """Test OpenAI cost tracking if configured."""
    print("\n=== Testing OpenAI Cost Tracking ===")

    # Check if OpenAI is configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI tests - OPENAI_API_KEY not set")
        return

    # Test pricing module
    print("\nTesting OpenAI pricing module...")
    pricing = get_pricing_instance()

    # Show some model prices
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    print("\n  OpenAI Model Pricing (per million tokens):")
    for model_id in models:
        model_pricing = pricing.get_model_pricing(model_id)
        if model_pricing:
            print(f"    {model_id}:")
            print(f"      Input: ${model_pricing.input_cost_per_million:.2f}")
            print(f"      Output: ${model_pricing.output_cost_per_million:.2f}")
            print(f"      Vision: {'Yes' if model_pricing.supports_vision else 'No'}")

    # Check pricing age
    age_info = pricing.check_pricing_age()
    print(f"\n  Pricing age: {age_info['days_old']} days")
    if age_info['needs_update']:
        print("  WARNING: Pricing may be outdated!")

    # Test with vision detection
    print("\nTesting vision detection with cost tracking...")
    config = Config(
        openai_api_key=api_key,
        use_vision=True,
        max_cost_per_file=0.10,
        enable_simple_case_detection=True,
        enable_island_detection=True,
        log_level="INFO"
    )

    gp = GridPorter(config)

    # Test a complex file that needs vision
    test_file = "complex_layout.xlsx"
    try:
        result = await gp.detect_tables(test_file)
        print(f"\n  Results for {test_file}:")
        print(f"    Tables found: {len(result.tables)}")
        print(f"    Methods used: {result.detection_metadata['methods_used']}")

        cost_report = result.detection_metadata['cost_report']
        print(f"    Total cost: ${cost_report['total_cost_usd']:.6f}")
        print(f"    Total tokens: {cost_report['total_tokens']}")

        # Show method breakdown
        if 'method_usage' in cost_report:
            print(f"    Method usage: {cost_report['method_usage']}")

    except Exception as e:
        print(f"  ERROR: {e}")

async def run_all_tests():
    """Run all Week 6 tests."""
    await test_traditional_methods()
    await test_openai_cost_tracking()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
```
