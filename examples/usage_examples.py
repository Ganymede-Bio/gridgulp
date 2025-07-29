#!/usr/bin/env python3
"""
GridGulp Usage Examples
Demonstrates various configuration options and usage patterns.
"""

import asyncio
from pathlib import Path

from gridgulp import Config, GridGulp


async def example_basic_detection():
    """Example 1: Basic table detection."""
    print("=== Basic Detection Example ===")

    # Initialize with default settings
    porter = GridGulp()

    result = await porter.detect_tables("examples/spreadsheets/simple/product_inventory.csv")

    print(f"File: {result.file_info.path.name}")
    print(f"Tables detected: {result.total_tables}")
    print(f"Processing time: {result.detection_time:.2f}s")

    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"  Table: {table.range.excel_range}")
            print(f"    Confidence: {table.confidence:.2%}")
            print(f"    Method: {table.detection_method}")
    print()


async def example_custom_config():
    """Example 2: Custom configuration."""
    print("=== Custom Configuration Example ===")

    # Custom configuration
    config = Config(
        confidence_threshold=0.8,  # Higher confidence threshold
        max_tables_per_sheet=10,  # Limit tables per sheet
        min_table_size=(3, 3),  # Minimum 3x3 table
        detect_merged_cells=True,
    )

    porter = GridGulp(config=config)

    result = await porter.detect_tables("examples/spreadsheets/complex/multi_table_report.csv")

    print(f"Total tables: {result.total_tables}")
    print(f"Detection time: {result.detection_time:.2f}s")

    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for i, table in enumerate(sheet.tables, 1):
            print(f"  Table {i}: {table.range.excel_range}")
            print(f"    Shape: {table.shape}")
            print(f"    Method: {table.detection_method}")
    print()


async def example_batch_processing():
    """Example 3: Batch processing multiple files."""
    print("=== Batch Processing Example ===")

    porter = GridGulp()

    # Process multiple files
    files = [
        "examples/spreadsheets/simple/product_inventory.csv",
        "examples/spreadsheets/sales/monthly_sales.csv",
        "examples/spreadsheets/financial/balance_sheet.csv",
    ]

    # Filter to only existing files
    existing_files = [f for f in files if Path(f).exists()]

    if existing_files:
        print(f"Processing {len(existing_files)} files:")
        print("-" * 60)

        total_tables = 0
        total_time = 0

        for file_path in existing_files:
            result = await porter.detect_tables(file_path)
            file_name = result.file_info.path.name
            print(
                f"{file_name:<25} | {result.total_tables:>2} tables | {result.detection_time:>6.2f}s"
            )
            total_tables += result.total_tables
            total_time += result.detection_time

        print("-" * 60)
        print(f"{'TOTAL':<25} | {total_tables:>2} tables | {total_time:>6.2f}s")
    else:
        print("No example files found. Please ensure example files exist.")
    print()


async def example_text_file_detection():
    """Example 4: Text file detection with encoding handling."""
    print("=== Text File Detection Example ===")

    porter = GridGulp()

    # Try to process a text file
    text_file = "examples/proprietary/NOV PEGDA6000 QC BB0310-241 242 247 248.txt"
    if Path(text_file).exists():
        result = await porter.detect_tables(text_file)

        print(f"File: {result.file_info.path.name}")
        print(f"Type: {result.file_info.type.value}")
        print(f"Tables detected: {result.total_tables}")

        for sheet in result.sheets:
            for table in sheet.tables:
                print(f"  Table range: {table.range.excel_range}")
                print(f"  Table shape: {table.shape}")
    else:
        print("Text file example not found")
    print()


async def example_error_handling():
    """Example 5: Error handling."""
    print("=== Error Handling Example ===")

    porter = GridGulp()

    # Try to process a non-existent file
    try:
        result = await porter.detect_tables("non_existent_file.xlsx")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")

    # Try to process with very strict configuration
    strict_config = Config(
        confidence_threshold=0.99,  # Very high threshold
        min_table_size=(10, 10),  # Large minimum size
    )

    porter_strict = GridGulp(config=strict_config)

    if Path("examples/spreadsheets/simple/product_inventory.csv").exists():
        result = await porter_strict.detect_tables(
            "examples/spreadsheets/simple/product_inventory.csv"
        )
        print(f"With strict config - Tables detected: {result.total_tables}")
    print()


async def main():
    """Run all examples."""
    print("GridGulp Usage Examples")
    print("=" * 50)
    print()

    examples = [
        example_basic_detection,
        example_custom_config,
        example_batch_processing,
        example_text_file_detection,
        example_error_handling,
    ]

    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print()

    print("All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
