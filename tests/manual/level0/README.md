# Manual Test Files

This directory contains sample files for manual testing of GridGulp's vision components.

## Creating Test Files

Run the test file creation script to generate sample data:

```bash
python create_test_files.py
```

This will create:

### Excel Files (requires openpyxl)
- `simple_table.xlsx` - Basic table with headers and 5 rows of data
- `large_table.xlsx` - Large table (100 rows × 20 columns) for testing scaling
- `complex_table.xlsx` - Multiple tables with different formatting

### CSV Files (always created)
- `simple_table.csv` - Basic table with headers and 5 rows of data
- `large_table.csv` - Large table (100 rows × 20 columns) for testing scaling

## Using Test Files

These files are referenced in the `docs/testing/WEEK3_TESTING_GUIDE.md`. The testing guide includes fallback code that creates data programmatically if the files don't exist.

### Example Usage

```python
from gridgulp.readers import get_reader
from gridgulp.vision import BitmapGenerator

# Load test file
reader = get_reader("tests/manual/simple_table.xlsx")
sheets = reader.read_all()
sheet = sheets[0]

# Generate bitmap for vision analysis
generator = BitmapGenerator()
image_bytes, metadata = generator.generate(sheet)
```

## File Descriptions

### simple_table.xlsx/csv
- **Purpose**: Basic vision testing
- **Structure**: 4 columns × 6 rows (including header)
- **Content**: Employee data (Name, Age, City, Salary)
- **Use Cases**: Basic bitmap generation, single table detection

### large_table.xlsx/csv
- **Purpose**: Scaling and performance testing
- **Structure**: 20 columns × 101 rows (including header)
- **Content**: Generated cell data (R1C1, R1C2, etc.)
- **Use Cases**: Image scaling, memory usage testing

### complex_table.xlsx
- **Purpose**: Multi-table detection testing
- **Structure**: 3 separate tables on one sheet
- **Content**:
  - Sales data (A1:D6)
  - Employee list (F1:I5)
  - Summary table (A8:C10)
- **Use Cases**: Multiple region detection, complex layouts

## Requirements

- **For Excel files**: `pip install openpyxl`
- **For CSV files**: No additional requirements (uses Python stdlib)

## Integration with Tests

The manual test files complement the automated unit tests by providing:

1. **Real file formats** for testing actual file reading
2. **Realistic data patterns** for vision model testing
3. **Performance benchmarking** with large datasets
4. **Visual verification** through saved bitmap images

See `docs/testing/WEEK3_TESTING_GUIDE.md` for complete testing procedures.
