# Testing Spreadsheets with GridGulp Test Script

The `test_all_spreadsheets.py` script provides a unified way to test GridGulp's table detection capabilities across multiple spreadsheet files. This guide explains how to use the script effectively for testing, validation, and reporting.

## Quick Start

The simplest way to test all spreadsheets in the default directories (examples/ and tests/manual/):

```bash
python scripts/test_all_spreadsheets.py
```

## Command Line Options

### Basic Options

- `--directories DIR [DIR ...]` - Specify custom directories to test (default: examples tests/manual)
- `--pattern PATTERN` - Filter files by pattern (can be specified multiple times)
- `--save` - Save detailed results to file
- `--format {json,csv,markdown}` - Output format for saved results (default: json)
- `--output-dir DIR` - Directory for saving results (default: tests/outputs)
- `--quiet` - Less verbose output (no table details)
- `--no-recursive` - Don't search subdirectories recursively

## Testing Directory Trees

By default, the script **recursively searches all subdirectories** for spreadsheet files. This makes it easy to test an entire project or folder hierarchy with a single command.

### Test All Spreadsheets in a Directory Tree

```bash
# Test all spreadsheets in your entire project (including subdirectories)
python scripts/test_all_spreadsheets.py --directories ~/my_project

# Test all spreadsheets in multiple directory trees
python scripts/test_all_spreadsheets.py --directories ~/data ~/reports ~/archives

# Test only the current directory (no subdirectories)
python scripts/test_all_spreadsheets.py --directories . --no-recursive
```

### Common Directory Testing Patterns

```bash
# Test all Excel files in your Documents folder and subdirectories
python scripts/test_all_spreadsheets.py --directories ~/Documents --pattern "*.xlsx" --pattern "*.xls"

# Test all CSV files in a data directory tree
python scripts/test_all_spreadsheets.py --directories ~/data --pattern "*.csv"

# Test everything in current directory and save results
python scripts/test_all_spreadsheets.py --directories . --save --format markdown
```

### Example: Testing a Data Science Project

```bash
# Typical data science project structure:
# my_project/
#   ├── data/
#   │   ├── raw/
#   │   │   ├── sales_2023.xlsx
#   │   │   └── sales_2024.xlsx
#   │   ├── processed/
#   │   │   └── combined_sales.csv
#   │   └── external/
#   │       └── market_data.xlsx
#   ├── reports/
#   │   ├── monthly/
#   │   │   └── jan_report.xlsx
#   │   └── annual/
#   │       └── 2023_summary.xlsx
#   └── notebooks/
#       └── analysis_results.csv

# Test all spreadsheets in the entire project
python scripts/test_all_spreadsheets.py --directories ~/my_project

# Test only data files
python scripts/test_all_spreadsheets.py --directories ~/my_project/data

# Test only Excel reports
python scripts/test_all_spreadsheets.py --directories ~/my_project/reports --pattern "*.xlsx"
```

### Usage Examples

#### Test a specific directory
```bash
python scripts/test_all_spreadsheets.py --directories ~/my_spreadsheets
```

#### Test only Excel files
```bash
python scripts/test_all_spreadsheets.py --pattern "*.xlsx" --pattern "*.xls"
```

#### Test only CSV files and save results
```bash
python scripts/test_all_spreadsheets.py --pattern "*.csv" --save --format csv
```

#### Test specific file pattern
```bash
python scripts/test_all_spreadsheets.py --pattern "sales_*.xlsx"
```

#### Save results in all formats
```bash
python scripts/test_all_spreadsheets.py --save --format json
python scripts/test_all_spreadsheets.py --save --format csv
python scripts/test_all_spreadsheets.py --save --format markdown
```

#### Quiet mode for CI/CD pipelines
```bash
python scripts/test_all_spreadsheets.py --quiet --save
```

## Output Formats

### Console Output (Default)

The script provides rich console output showing:
- File processing status (✓ success, ✗ error, ⚠️ validation mismatch)
- Number of tables detected
- Processing time
- File size
- Detection method used
- Format mismatches (when file content doesn't match extension)
- Table details (range, headers, confidence)

Example output:
```
📁 examples/proprietary
----------------------------------------------------------------------------------------------------
✓ sales_data.xlsx                         | Tables: 1  | Time: 0.123s | Size: 45.2KB | Method: simple_case
  📄 Sheet: Sheet1
     └─ A1:E100          | 100×5 | Conf: 95% | Headers: Date, Product, Quantity ... +2
```

### JSON Format

Comprehensive machine-readable format with all detection details:

```json
{
  "timestamp": "2025-01-15T10:30:45",
  "config": {
    "confidence_threshold": 0.7,
    "enable_simple_case": true,
    "enable_island": true
  },
  "summary": {
    "total_files": 25,
    "successful": 23,
    "failed": 2,
    "total_tables": 45,
    "total_time": 3.456
  },
  "results": [
    {
      "file": "sales_data.xlsx",
      "path": "/path/to/sales_data.xlsx",
      "status": "success",
      "tables_found": 1,
      "detection_time": 0.123,
      "details": [...]
    }
  ]
}
```

### CSV Format

Simple tabular format for spreadsheet analysis:

```csv
file,path,status,tables_found,sheets_found,detection_time,size_kb,detected_type,format_mismatch,validation,error
sales_data.xlsx,/path/to/sales_data.xlsx,success,1,1,0.123,45.2,xlsx,False,pass,
```

### Markdown Format

Human-readable report format:

```markdown
# GridGulp Test Results

Generated: 2025-01-15 10:30:45

## Summary

- Total files tested: 25
- Successful: 23
- Failed: 2
- Total tables found: 45

## Results

| File | Status | Tables | Time (s) | Size (KB) | Notes |
|------|--------|--------|----------|-----------|-------|
| sales_data.xlsx | ✓ | 1 | 0.123 | 45.2 |  |
```

## Understanding Results

### Status Indicators

- **✓ (Success)**: File processed successfully
- **✗ (Error)**: File processing failed
- **⚠️ (Warning)**: Validation mismatch (expected vs actual tables)

### Validation

The script includes built-in validation for test files. When a file has expected results defined, the script will validate:
- Number of tables detected matches expectations
- Files that should fail actually fail
- Files with multiple tables are detected correctly

### Performance Metrics

The summary includes:
- Total processing time
- Average time per file
- Fastest and slowest processing times
- Memory usage (when available)

## Exit Codes

The script uses exit codes for CI/CD integration:
- `0`: All tests passed successfully
- `1`: One or more tests failed or had validation errors

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Test Spreadsheet Detection
  run: |
    python scripts/test_all_spreadsheets.py --quiet --save --format json
  continue-on-error: false
```

### Jenkins Example

```groovy
stage('Test Spreadsheets') {
    steps {
        sh 'python scripts/test_all_spreadsheets.py --save --format json'
        archiveArtifacts artifacts: 'tests/outputs/*.json'
    }
}
```

## Advanced Usage

### Custom Configuration

Create a custom configuration by modifying the script:

```python
config = Config(
    confidence_threshold=0.8,  # Higher confidence requirement
    enable_simple_case_detection=True,
    enable_island_detection=True,
    max_file_size_mb=100,  # Limit file size
)
```

### Batch Processing

Process multiple directories with different patterns:

```bash
# Test all Excel files in data/
python scripts/test_all_spreadsheets.py --directories data/ --pattern "*.xlsx"

# Then test all CSV files in archives/
python scripts/test_all_spreadsheets.py --directories archives/ --pattern "*.csv"
```

### Filtering Results

Use shell commands to filter results:

```bash
# Find files with no tables detected
python scripts/test_all_spreadsheets.py | grep "Tables: 0"

# Find files that took more than 1 second
python scripts/test_all_spreadsheets.py | grep -E "Time: [1-9]\.[0-9]+s"
```

## Troubleshooting

### Common Issues

1. **"No files found to test"**
   - Check that the specified directories exist
   - Verify file patterns match actual files
   - Ensure you have read permissions

2. **Format mismatch warnings**
   - File extension doesn't match actual content
   - GridGulp will still process the file correctly

3. **Validation failures**
   - Expected results don't match actual detection
   - Review the file manually to verify expectations

### Debug Mode

For detailed debugging information:

```bash
export GRIDGULP_LOG_LEVEL=DEBUG
python scripts/test_all_spreadsheets.py
```

## Best Practices

1. **Regular Testing**: Run the script regularly to catch regressions
2. **Save Results**: Use `--save` to keep historical test results
3. **Review Failures**: Investigate any validation failures or errors
4. **Update Expectations**: Keep expected results up to date with changes
5. **Use in CI**: Integrate into your continuous integration pipeline

## Example Workflows

### Daily Test Run
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
python scripts/test_all_spreadsheets.py \
    --save \
    --format json \
    --output-dir "test_results/$DATE"
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
python scripts/test_all_spreadsheets.py --quiet || {
    echo "Spreadsheet tests failed. Commit aborted."
    exit 1
}
```

### Comparative Testing
```bash
# Test before changes
git checkout main
python scripts/test_all_spreadsheets.py --save --output-dir results_before

# Test after changes
git checkout feature-branch
python scripts/test_all_spreadsheets.py --save --output-dir results_after

# Compare results
diff results_before/*.json results_after/*.json
```
