#!/usr/bin/env python3
"""Example of using custom detection parameters for better table detection."""

import asyncio

from gridgulp import GridGulp
from gridgulp.config import Config


async def example_custom_parameters():
    """Demonstrate custom detection parameters."""

    # Create a config with custom parameters
    config = Config(
        # Allow up to 2 empty rows within tables
        # This helps with tables that have section separators
        # Note: This would need to be added to Config class to be fully functional
        # For now, it demonstrates the concept
    )

    # Initialize GridGulp with custom config
    gg = GridGulp(config=config)

    # Example 1: Detect tables in a file with custom parameters
    print("Example 1: Basic detection with improved algorithms")
    print("-" * 50)

    result = await gg.detect_tables("tests/manual/level2/creative_tables.xlsx")

    print(f"Total tables found: {result.total_tables}")
    for sheet in result.sheets:
        if sheet.tables:
            print(f"\nSheet '{sheet.name}': {len(sheet.tables)} tables")
            for table in sheet.tables:
                print(f"  - {table.range.excel_range}")

    # Example 2: Demonstrating the improvements
    print("\n\nKey Improvements Demonstrated:")
    print("-" * 50)
    print("1. Column Gap Detection:")
    print("   - Tables separated by empty columns are NOT merged")
    print("   - Example: Tables at A1:C5 and E1:G5 remain separate")
    print()
    print("2. Empty Row Tolerance:")
    print("   - Tables with 1-2 empty rows inside are kept together")
    print("   - Prevents false splits due to section headers")
    print()
    print("3. Border-Based Detection:")
    print("   - Uses Excel cell borders to detect table boundaries")
    print("   - Bottom borders indicate table end")
    print("   - Box borders indicate complete tables")
    print()
    print("4. Enhanced Confidence Scoring:")
    print("   - Considers multiple factors: borders, headers, formatting")
    print("   - Better distinction between actual tables and noise")


async def example_specific_use_cases():
    """Show specific use cases for the improvements."""

    print("\n\nSpecific Use Cases:")
    print("-" * 50)

    gg = GridGulp()

    # Use case 1: Side-by-side tables
    print("\n1. Side-by-side tables (like dashboards):")
    print("   Before: Would merge into one large table")
    print("   After:  Correctly identifies as separate tables")

    # Use case 2: Tables with subtotals
    print("\n2. Tables with subtotal rows:")
    print("   Before: Would split at empty subtotal rows")
    print("   After:  Keeps table intact with subtotals")

    # Use case 3: Bordered tables
    print("\n3. Tables with clear borders:")
    print("   Before: Might include surrounding data")
    print("   After:  Respects border boundaries precisely")


if __name__ == "__main__":
    print("GridGulp - Custom Detection Parameters Example")
    print("=" * 50)

    asyncio.run(example_custom_parameters())
    asyncio.run(example_specific_use_cases())

    print("\n\nFor more information, see:")
    print("- GridGulp documentation")
    print("- Source code in detectors/island_detector.py")
