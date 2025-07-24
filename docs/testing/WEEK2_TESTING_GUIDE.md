# Week 2 Manual Testing Guide - File Reading Infrastructure

This guide provides comprehensive test scenarios for manually testing the Week 2 implementation of GridPorter's file reading infrastructure.

## Prerequisites

1. Ensure GridPorter is installed:
   ```bash
   uv pip install -e .
   ```

2. Create test files as described in each section below.

## 1. Excel Reader Testing

### 1.1 Modern Excel Formats (.xlsx, .xlsm, .xlsb)

#### Test Case 1: Basic Single Sheet
**Setup**: Create `test_basic.xlsx` with:
- Sheet1 containing a simple table (A1:D5)
- Headers: Name, Age, City, Salary
- 4 data rows with mixed types

**Test Command**:
```python
from gridporter.readers import get_reader

reader = get_reader("test_basic.xlsx")
data = reader.read_file("test_basic.xlsx")
print(f"Sheets found: {len(data.sheets)}")
print(f"First sheet name: {data.sheets[0].name}")
print(f"Data shape: {len(data.sheets[0].data)} rows")
```

**Expected**:
- 1 sheet detected
- 5 rows (including header)
- All data types preserved

#### Test Case 2: Multiple Sheets
**Setup**: Create `test_multi_sheet.xlsx` with:
- Sheet1: Sales data (10 rows)
- Sheet2: Employee data (5 rows)
- Sheet3: Empty sheet

**Test Command**:
```python
reader = get_reader("test_multi_sheet.xlsx")
data = reader.read_file("test_multi_sheet.xlsx")
for sheet in data.sheets:
    print(f"Sheet '{sheet.name}': {len(sheet.data)} rows")
```

**Expected**:
- 3 sheets detected
- Correct row counts
- Empty sheet handled gracefully

#### Test Case 3: Cell Formatting
**Setup**: Create `test_formatting.xlsx` with:
- Bold headers
- Colored cells (background and font)
- Different font sizes
- Merged cells (B2:C2)

**Test Command**:
```python
reader = get_reader("test_formatting.xlsx")
data = reader.read_file("test_formatting.xlsx")
sheet = data.sheets[0]
print(f"Cell formatting found: {any(cell.formatting for row in sheet.data for cell in row)}")
print(f"Merged cells: {sheet.merged_cells}")
```

**Expected**:
- Formatting metadata captured
- Merged cells detected with correct ranges

#### Test Case 4: Formulas and Data Types
**Setup**: Create `test_formulas.xlsx` with:
- Column A: Text data
- Column B: Numbers
- Column C: Dates
- Column D: Formulas (=B2*2)
- Column E: Booleans

**Test Command**:
```python
reader = get_reader("test_formulas.xlsx")
data = reader.read_file("test_formulas.xlsx")
sheet = data.sheets[0]
for i, row in enumerate(sheet.data[:2]):
    print(f"Row {i}: {[cell.value_type for cell in row]}")
```

**Expected**:
- Correct type detection for each column
- Formula cells identified

### 1.2 Legacy Excel Format (.xls)

#### Test Case 5: Legacy Format Support
**Setup**: Create `test_legacy.xls` (Excel 97-2003 format) with basic data

**Test Command**:
```python
reader = get_reader("test_legacy.xls")
data = reader.read_file("test_legacy.xls")
print(f"Successfully read legacy format: {len(data.sheets)} sheets")
```

**Expected**:
- File reads successfully using xlrd
- Data extracted correctly

### 1.3 Error Handling

#### Test Case 6: Corrupted File
**Setup**: Create a corrupted Excel file by renaming a text file to `.xlsx`

**Test Command**:
```python
try:
    reader = get_reader("corrupted.xlsx")
    data = reader.read_file("corrupted.xlsx")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
```

**Expected**:
- CorruptedFileError raised
- Clear error message

#### Test Case 7: Password Protected File
**Setup**: Create a password-protected Excel file

**Test Command**:
```python
try:
    reader = get_reader("protected.xlsx")
    data = reader.read_file("protected.xlsx")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
```

**Expected**:
- PasswordProtectedError raised

## 2. CSV Reader Testing

### 2.1 Delimiter Detection

#### Test Case 8: Common Delimiters
**Setup**: Create test files:
- `test_comma.csv`: Standard comma-separated
- `test_tab.tsv`: Tab-separated
- `test_pipe.csv`: Pipe-separated (|)
- `test_semicolon.csv`: Semicolon-separated

**Test Command**:
```python
for filename in ["test_comma.csv", "test_tab.tsv", "test_pipe.csv", "test_semicolon.csv"]:
    reader = get_reader(filename)
    data = reader.read_file(filename)
    print(f"{filename}: {len(data.sheets[0].data[0])} columns detected")
```

**Expected**:
- All delimiters auto-detected correctly
- Correct column count for each file

### 2.2 Encoding Detection

#### Test Case 9: Various Encodings
**Setup**: Create CSV files with different encodings:
- `test_utf8.csv`: UTF-8 with special characters (é, ñ, 中文)
- `test_latin1.csv`: Latin-1 encoding
- `test_utf16.csv`: UTF-16 encoding

**Test Command**:
```python
for filename in ["test_utf8.csv", "test_latin1.csv", "test_utf16.csv"]:
    reader = get_reader(filename)
    data = reader.read_file(filename)
    print(f"{filename}: Successfully read {len(data.sheets[0].data)} rows")
    # Print first row to verify encoding
    print(f"  First row: {[cell.value for cell in data.sheets[0].data[0]]}")
```

**Expected**:
- All encodings detected and handled
- Special characters preserved

### 2.3 Type Inference

#### Test Case 10: Data Type Detection
**Setup**: Create `test_types.csv` with:
```csv
String,Integer,Float,Boolean,Date
Hello,42,3.14,TRUE,2024-01-15
World,100,2.718,FALSE,2024-12-25
Test,0,-1.5,true,2024/06/30
```

**Test Command**:
```python
reader = get_reader("test_types.csv")
data = reader.read_file("test_types.csv")
sheet = data.sheets[0]
for i, row in enumerate(sheet.data[:2]):
    print(f"Row {i}: {[cell.value_type for cell in row]}")
```

**Expected**:
- Correct type inference for each column
- Dates recognized in various formats

### 2.4 Edge Cases

#### Test Case 11: Large File Handling
**Setup**: Create a CSV with 10,000 rows

**Test Command**:
```python
import time
start = time.time()
reader = get_reader("large_file.csv")
data = reader.read_file("large_file.csv")
elapsed = time.time() - start
print(f"Read {len(data.sheets[0].data)} rows in {elapsed:.2f} seconds")
```

**Expected**:
- File loads successfully
- Reasonable performance

#### Test Case 12: Quoted Fields
**Setup**: Create `test_quoted.csv` with:
```csv
"Name","Description","Price"
"Product A","Contains, comma","$19.99"
"Product B","Has ""quotes""","$29.99"
"Product C","Multi
line","$39.99"
```

**Test Command**:
```python
reader = get_reader("test_quoted.csv")
data = reader.read_file("test_quoted.csv")
for row in data.sheets[0].data:
    print([cell.value for cell in row])
```

**Expected**:
- Commas in quotes preserved
- Escaped quotes handled
- Multiline fields supported

## 3. Unified Reader Interface Testing

### 3.1 Factory Pattern

#### Test Case 13: Automatic Reader Selection
**Setup**: Have various file types ready

**Test Command**:
```python
from gridporter.readers import get_reader, is_supported

files = ["test.xlsx", "test.csv", "test.tsv", "test.xls", "test.pdf"]
for file in files:
    print(f"{file}: Supported = {is_supported(file)}")
    if is_supported(file):
        reader = get_reader(file)
        print(f"  Reader type: {type(reader).__name__}")
```

**Expected**:
- Correct reader selected for each type
- PDF marked as unsupported

### 3.2 Async Support

#### Test Case 14: Async Reading
**Setup**: Use any test Excel/CSV file

**Test Command**:
```python
import asyncio
from gridporter.readers import get_async_reader

async def test_async():
    reader = await get_async_reader("test.xlsx")
    data = await reader.read_file("test.xlsx")
    print(f"Async read successful: {len(data.sheets)} sheets")

asyncio.run(test_async())
```

**Expected**:
- Async operations work correctly
- Same results as sync version

### 3.3 Error Handling

#### Test Case 15: Comprehensive Error Testing
**Test Command**:
```python
from gridporter.readers.errors import *

# Test unsupported file
try:
    reader = get_reader("test.pdf")
except UnsupportedFileError as e:
    print(f"Unsupported file handled: {e}")

# Test non-existent file
try:
    reader = get_reader("test.csv")
    data = reader.read_file("non_existent.csv")
except ReaderError as e:
    print(f"Missing file handled: {e}")
```

**Expected**:
- Appropriate errors for each case
- Clear, helpful error messages

## 4. Integration Testing

### Test Case 16: End-to-End GridPorter Usage
**Setup**: Use the example from `examples/basic_usage.py`

**Test Command**:
```bash
python examples/basic_usage.py sample_data.xlsx
```

**Expected**:
- File processed successfully
- Detection results shown
- Proper integration with detection pipeline

## Testing Checklist

- [ ] Excel reader handles all modern formats
- [ ] Excel reader handles legacy .xls format
- [ ] CSV reader detects delimiters correctly
- [ ] CSV reader handles various encodings
- [ ] Type inference works for both readers
- [ ] Error handling is comprehensive
- [ ] Factory pattern selects correct readers
- [ ] Async support functions properly
- [ ] Large files handled efficiently
- [ ] Edge cases (merged cells, formulas, quotes) work
- [ ] Integration with main GridPorter pipeline works

## Performance Benchmarks

Expected performance targets:
- Small files (<1MB): < 0.5 seconds
- Medium files (1-10MB): < 2 seconds
- Large files (10-50MB): < 10 seconds
- Memory usage: < 2x file size

## Common Issues and Solutions

1. **Import errors**: Ensure all dependencies installed (`openpyxl`, `xlrd`, `chardet`)
2. **Encoding errors**: Check file encoding, reader should auto-detect
3. **Memory issues**: For very large files, consider processing in chunks
4. **Type detection issues**: Check data consistency in columns

## Reporting Issues

If any tests fail:
1. Note the exact error message
2. Save the test file that caused the issue
3. Include Python version and dependency versions
4. Report with full traceback
