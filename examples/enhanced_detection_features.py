#!/usr/bin/env python3
"""
Example demonstrating all enhanced table detection features in GridGulp v0.3.3+

Features demonstrated:
1. Column gap detection prevents over-merging
2. Empty row tolerance prevents over-splitting
3. Border-based boundary detection
4. Enhanced weighted scoring system
5. Configurable detection parameters
"""

import asyncio

from gridgulp import GridGulp
from gridgulp.config import Config


async def example_column_gap_detection():
    """Demonstrate how column gaps prevent table merging."""
    print("\n" + "=" * 60)
    print("1. Column Gap Detection")
    print("=" * 60)

    # Default config prevents merging across column gaps
    gg = GridGulp()

    print("\nBehavior: Tables separated by empty columns are NOT merged")
    print("Example: Dashboard layouts with side-by-side KPIs")
    print("\nBefore: A1:C10 and E1:G10 → merged into A1:G10 ❌")
    print("After:  A1:C10 and E1:G10 → kept as 2 tables ✓")

    # To allow merging (not recommended), you can configure:
    config = Config(column_gap_prevents_merge=False)
    gg_allow_merge = GridGulp(config=config)

    print("\nConfiguration option: column_gap_prevents_merge (default: True)")


async def example_empty_row_tolerance():
    """Demonstrate empty row tolerance within tables."""
    print("\n" + "=" * 60)
    print("2. Empty Row Tolerance")
    print("=" * 60)

    # Configure tolerance for empty rows
    config = Config(empty_row_tolerance=2)  # Allow up to 2 empty rows
    gg = GridGulp(config=config)

    print("\nBehavior: Tables with small gaps (1-2 rows) stay together")
    print("Example: Financial tables with subtotal sections")
    print("\nBefore: Table splits at every empty row ❌")
    print("After:  Table remains intact with section breaks ✓")
    print("\nConfiguration option: empty_row_tolerance (default: 1, range: 0-5)")


async def example_border_detection():
    """Demonstrate border-based table detection."""
    print("\n" + "=" * 60)
    print("3. Border-Based Boundary Detection")
    print("=" * 60)

    gg = GridGulp()  # Border detection enabled by default

    print("\nBehavior: Cell borders define precise table boundaries")
    print("Features detected:")
    print("  - Bottom borders indicate table end")
    print("  - Box borders (all 4 sides) indicate complete tables")
    print("  - Border pattern changes signal new tables")
    print("\nConfiguration option: use_border_detection (default: True)")


async def example_weighted_scoring():
    """Demonstrate the enhanced scoring system."""
    print("\n" + "=" * 60)
    print("4. Enhanced Weighted Scoring System")
    print("=" * 60)

    gg = GridGulp()

    print("\nScoring components and weights:")
    print("  - Size Score (20%): Relative and absolute table size")
    print("  - Density Score (15%): How filled the region is")
    print("  - Shape Score (10%): Prefer rectangular tables")
    print("  - Header Score (15%): Tables usually have headers")
    print("  - Border Score (15%): Clean borders indicate tables")
    print("  - Formatting Score (15%): Consistency in formatting")
    print("  - Isolation Score (10%): Not subset of another table")

    print("\nResult: More accurate confidence scores (0.0-1.0)")
    print("  - High confidence (>0.9): Well-formed tables with headers")
    print("  - Medium confidence (0.7-0.9): Good tables, minor issues")
    print("  - Low confidence (<0.7): Possible noise or fragments")


async def example_custom_configuration():
    """Demonstrate custom configuration for specific needs."""
    print("\n" + "=" * 60)
    print("5. Custom Configuration Example")
    print("=" * 60)

    # Create custom config for specific use case
    config = Config(
        # Strict mode: no gaps allowed
        empty_row_tolerance=0,
        # Require high column overlap for merging
        min_column_overlap_for_merge=0.8,
        # Use all detection features
        use_border_detection=True,
        column_gap_prevents_merge=True,
        # Higher confidence threshold
        confidence_threshold=0.8,
        # Prefer larger tables
        prefer_large_tables=True,
        min_table_percentage=0.01,  # At least 1% of sheet
    )

    gg = GridGulp(config=config)

    print("\nCustom configuration for financial reports:")
    print("  - No tolerance for empty rows (strict tables)")
    print("  - 80% column overlap required for merging")
    print("  - Higher confidence threshold (0.8)")
    print("  - Minimum table size: 1% of sheet")


async def example_real_world_scenarios():
    """Show real-world scenarios that benefit from enhancements."""
    print("\n" + "=" * 60)
    print("Real-World Scenarios")
    print("=" * 60)

    scenarios = [
        {
            "name": "Financial Dashboard",
            "issue": "Multiple KPI tables merged into one",
            "solution": "Column gap detection keeps them separate",
            "config": "Default settings work perfectly",
        },
        {
            "name": "Income Statement",
            "issue": "Table splits at subtotal rows",
            "solution": "Empty row tolerance preserves structure",
            "config": "empty_row_tolerance=2 for subtotals",
        },
        {
            "name": "Bordered Report",
            "issue": "Detection includes surrounding notes",
            "solution": "Border detection respects table edges",
            "config": "use_border_detection=True (default)",
        },
        {
            "name": "Engineering Data",
            "issue": "Small parameter tables missed",
            "solution": "Enhanced scoring identifies them",
            "config": "Lower min_table_percentage if needed",
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Problem: {scenario['issue']}")
        print(f"  Solution: {scenario['solution']}")
        print(f"  Config: {scenario['config']}")


async def main():
    """Run all examples."""
    print("GridGulp Enhanced Table Detection Features")
    print("==========================================")

    await example_column_gap_detection()
    await example_empty_row_tolerance()
    await example_border_detection()
    await example_weighted_scoring()
    await example_custom_configuration()
    await example_real_world_scenarios()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey improvements in v0.3.3+:")
    print("✓ Prevents over-merging of side-by-side tables")
    print("✓ Prevents over-splitting of tables with gaps")
    print("✓ Respects border-defined table boundaries")
    print("✓ Provides accurate confidence scores")
    print("✓ Fully configurable for specific needs")

    print("\nFor more information:")
    print("- See Config class for all options")
    print("- Check examples/ directory for use cases")
    print("- Read CHANGELOG.md for version history")


if __name__ == "__main__":
    asyncio.run(main())
