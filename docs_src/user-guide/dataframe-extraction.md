# DataFrame Extraction Guide

GridGulp can extract detected tables as pandas DataFrames with intelligent header detection and quality scoring.

## Basic Extraction

```python
from gridgulp import GridGulp
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

async def extract_dataframes(file_path):
    # First detect tables
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Get file data
    reader = get_reader(file_path)
    file_data = reader.read_sync()

    # Extract DataFrames
    extractor = DataFrameExtractor()

    for sheet in result.sheets:
        # Find corresponding sheet data
        sheet_data = next(s for s in file_data.sheets if s.name == sheet.name)

        for table in sheet.tables:
            # Extract DataFrame
            df, header_info, quality = extractor.extract_dataframe(
                sheet_data,
                table.range
            )

            if df is not None:
                print(f"Extracted table from {sheet.name}")
                print(f"  Shape: {df.shape}")
                print(f"  Quality: {quality:.2f}")
                print(f"  Headers: {list(df.columns)}")
                print(df.head())
```

## Advanced Features

### Header Detection

The extractor automatically detects:
- Single-row headers
- Multi-row headers with merged cells
- Title rows to skip
- Column data types (by analyzing up to 100 rows)

```python
extractor = DataFrameExtractor(
    max_header_rows=10,           # Check up to 10 rows for headers
    min_data_rows=2,              # Need at least 2 data rows
    type_consistency_threshold=0.8 # 80% of values must match type
)
```

### Quality Scoring

Each extraction gets a quality score (0-1) based on:
- Data density (non-empty cells)
- Type consistency across columns
- Header detection confidence
- Table structure regularity

```python
# Only process high-quality tables
if quality >= 0.7:
    # Process the DataFrame
    pass
```

### Handling Special Formats

```python
# Plate map detection (96-well, 384-well, etc.)
if header_info.plate_format:
    print(f"Detected {header_info.plate_format} plate")

# Transposed tables
if header_info.orientation == "horizontal":
    df = df.T  # Transpose back
```

## Complete Example

```python
import asyncio
import pandas as pd
from pathlib import Path
from gridgulp import GridGulp
from gridgulp.extractors import DataFrameExtractor
from gridgulp.readers import get_reader

async def extract_all_tables(file_path, output_dir):
    """Extract all tables from a file and save as CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Detect tables
    porter = GridGulp()
    result = await porter.detect_tables(file_path)

    # Get file data
    reader = get_reader(file_path)
    file_data = reader.read_sync()

    # Extract each table
    extractor = DataFrameExtractor()
    extracted_count = 0

    for sheet in result.sheets:
        sheet_data = next(s for s in file_data.sheets if s.name == sheet.name)

        for idx, table in enumerate(sheet.tables):
            df, header_info, quality = extractor.extract_dataframe(
                sheet_data,
                table.range
            )

            if df is not None and quality > 0.5:
                # Generate filename
                filename = f"{Path(file_path).stem}_{sheet.name}_table{idx}_q{quality:.2f}.csv"
                csv_path = output_path / filename

                # Save DataFrame
                df.to_csv(csv_path, index=False)
                extracted_count += 1

                print(f"Saved: {filename}")
                print(f"  Shape: {df.shape}")
                print(f"  Quality: {quality:.2f}")
                print(f"  Headers: {list(df.columns)[:5]}...")
                print()

    print(f"Total tables extracted: {extracted_count}")

# Run the extraction
asyncio.run(extract_all_tables("report.xlsx", "output/"))
```

## Integration with Scripts

See the example scripts in the `scripts/` directory:
- `extract_dataframes.py` - Full extraction pipeline
- `save_extracted_csvs.py` - Export DataFrames to CSV files
- `test_dataframe_extraction.py` - Test extraction on multiple files
