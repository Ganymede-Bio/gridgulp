"""Basic usage example for GridGulp."""

import asyncio
from pathlib import Path

from gridgulp import GridGulp


async def detect_tables_example():
    """Example of basic table detection."""
    # Initialize GridGulp with default settings
    porter = GridGulp()

    # Example file path
    file_path = Path("examples/spreadsheets/simple/product_inventory.csv")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Detecting tables in: {file_path}")
    print("-" * 50)

    # Detect tables
    result = await porter.detect_tables(str(file_path))

    # Print summary
    print(f"File type: {result.file_info.type.value}")
    print(f"Total sheets: {len(result.sheets)}")
    print(f"Total tables found: {result.total_tables}")
    print(f"Detection time: {result.detection_time:.2f}s")

    # Print details for each sheet
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        if sheet.tables:
            for i, table in enumerate(sheet.tables, 1):
                print(f"  Table {i}:")
                print(f"    Range: {table.range.excel_range}")
                print(f"    Shape: {table.shape}")
                print(f"    Suggested name: {table.suggested_name or 'N/A'}")
                print(f"    Confidence: {table.confidence:.2%}")
                print(f"    Method: {table.detection_method}")
                if table.headers:
                    print(f"    Headers: {', '.join(table.headers[:5])}")
                    if len(table.headers) > 5:
                        print(f"             ... and {len(table.headers) - 5} more")
        else:
            print("  No tables detected")


async def custom_config_example():
    """Example of custom configuration."""
    # Configure with custom settings
    porter = GridGulp(
        confidence_threshold=0.8,  # Higher threshold
        max_tables_per_sheet=5,
        min_table_size=(3, 3),
    )

    file_path = Path("examples/spreadsheets/simple/product_inventory.csv")
    if file_path.exists():
        result = await porter.detect_tables(str(file_path))
        print(f"Detected {result.total_tables} tables with custom config")


async def batch_processing_example():
    """Example of processing multiple files."""
    porter = GridGulp()

    # Process all files in the examples directory
    examples_dir = Path("examples")
    all_files = []
    for pattern in ["*.xlsx", "*.xls", "*.csv", "*.txt"]:
        all_files.extend(examples_dir.rglob(pattern))

    if not all_files:
        print("No files found in examples/")
        return

    print(f"Processing {len(all_files)} files...")

    for file_path in all_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            result = await porter.detect_tables(str(file_path))
            summary = result.to_summary()
            print(f"  Tables found: {summary['total_tables']}")
            print(f"  Processing time: {summary['detection_time']}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run examples."""
    print("GridGulp Examples")
    print("=" * 50)

    # Run the basic example
    asyncio.run(detect_tables_example())

    # Uncomment to try other examples:
    # asyncio.run(custom_config_example())
    # asyncio.run(batch_processing_example())


if __name__ == "__main__":
    main()
