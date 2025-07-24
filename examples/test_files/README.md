# Test Files for Format Detection

This directory contains test files with intentionally incorrect file extensions to test GridPorter's robust file format detection capabilities.

## Test Files

### `csv_with_xls_extension.xls`
- **Actual Content**: CSV (comma-separated values)
- **File Extension**: `.xls` (suggests Excel binary format)
- **Expected Detection**: Should detect as CSV with format mismatch warning
- **Test Purpose**: Verify CSV content detection regardless of misleading extension

### `tsv_with_csv_extension.csv`
- **Actual Content**: TSV (tab-separated values)
- **File Extension**: `.csv` (suggests comma-separated values)
- **Expected Detection**: Should detect as TSV with format mismatch warning
- **Test Purpose**: Verify delimiter-based content analysis

### `csv_with_xlsx_extension.xlsx`
- **Actual Content**: CSV (comma-separated values)
- **File Extension**: `.xlsx` (suggests Excel XML format)
- **Expected Detection**: Should detect as CSV with format mismatch warning
- **Test Purpose**: Verify content takes precedence over extension

## Usage Example

```python
import asyncio
from pathlib import Path
from gridporter import GridPorter

async def test_format_detection():
    porter = GridPorter()

    test_files = [
        "examples/test_files/csv_with_xls_extension.xls",
        "examples/test_files/tsv_with_csv_extension.csv",
        "examples/test_files/csv_with_xlsx_extension.xlsx"
    ]

    for file_path in test_files:
        print(f"\n=== Testing {Path(file_path).name} ===")

        try:
            result = await porter.detect_tables(file_path)
            file_info = result.file_info

            print(f"File: {file_info.path.name}")
            print(f"Extension suggests: {file_info.extension_format}")
            print(f"Detected format: {file_info.type}")
            print(f"Detection method: {file_info.detection_method}")
            print(f"Confidence: {file_info.detection_confidence:.2%}")
            print(f"Format mismatch: {file_info.format_mismatch}")

            if file_info.format_mismatch:
                print("⚠️  WARNING: File extension doesn't match content!")
            else:
                print("✅ Extension matches detected content")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# Run the test
asyncio.run(test_format_detection())
```

## Expected Results

When running the test above, you should see output similar to:

```
=== Testing csv_with_xls_extension.xls ===
File: csv_with_xls_extension.xls
Extension suggests: XLS
Detected format: CSV
Detection method: content
Confidence: 85%
Format mismatch: True
⚠️  WARNING: File extension doesn't match content!

=== Testing tsv_with_csv_extension.csv ===
File: tsv_with_csv_extension.csv
Extension suggests: CSV
Detected format: TSV
Detection method: content
Confidence: 90%
Format mismatch: True
⚠️  WARNING: File extension doesn't match content!

=== Testing csv_with_xlsx_extension.xlsx ===
File: csv_with_xlsx_extension.xlsx
Extension suggests: XLSX
Detected format: CSV
Detection method: content
Confidence: 88%
Format mismatch: True
⚠️  WARNING: File extension doesn't match content!
```

## Detection Process

GridPorter uses a multi-layer detection strategy:

1. **Magic Bytes**: Checks file headers for binary signatures
2. **MIME Type**: Uses python-magic for MIME type detection
3. **Content Analysis**: Analyzes file structure (delimiters, patterns)
4. **Extension Check**: Compares detected format with file extension
5. **Fallback**: Uses extension if content detection fails

## Benefits

This robust detection approach ensures:
- ✅ **Correct Processing**: Files are processed according to their actual content
- ✅ **User Warnings**: Clear notifications when extensions are misleading
- ✅ **Error Prevention**: Avoids processing errors from format mismatches
- ✅ **Flexibility**: Handles files that have been renamed or have wrong extensions
