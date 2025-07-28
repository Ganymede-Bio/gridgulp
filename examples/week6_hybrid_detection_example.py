"""Example demonstrating Week 6 hybrid detection features.

This example shows:
1. Simple case detection for single-table sheets
2. Island detection for multi-table sheets
3. Excel metadata extraction (ListObjects, named ranges)
4. Cost optimization and tracking
5. Hybrid decision making
"""

import asyncio
import logging
from pathlib import Path

# Configure logging to see detection decisions
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the src directory to Python path for development
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import GridPorter components
# ruff: noqa: E402
from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.detectors.island_detector import IslandDetector
from gridporter.detectors.simple_case_detector import SimpleCaseDetector
from gridporter.utils.cost_optimizer import CostOptimizer


async def demonstrate_simple_case_detection():
    """Demonstrate simple case detection on a single-table sheet."""
    print("\n" + "=" * 60)
    print("SIMPLE CASE DETECTION DEMO")
    print("=" * 60)

    # Create a mock sheet with a simple table
    from gridporter.models.sheet_data import CellData, SheetData

    sheet = SheetData(name="SimpleSheet")

    # Add headers
    headers = ["Product", "Price", "Quantity", "Total"]
    for col, header in enumerate(headers):
        sheet.set_cell(
            0, col, CellData(value=header, data_type="string", is_bold=True, row=0, column=col)
        )

    # Add data rows
    data = [
        ["Apple", 1.50, 10, 15.00],
        ["Banana", 0.75, 20, 15.00],
        ["Orange", 2.00, 5, 10.00],
        ["Total", "", 35, 40.00],
    ]

    for row_idx, row_data in enumerate(data, 1):
        for col_idx, value in enumerate(row_data):
            if value != "":
                sheet.set_cell(
                    row_idx,
                    col_idx,
                    CellData(
                        value=value,
                        data_type="number" if isinstance(value, int | float) else "string",
                        is_bold=(row_idx == len(data)),  # Bold last row
                        row=row_idx,
                        column=col_idx,
                    ),
                )

    # Run simple case detection
    detector = SimpleCaseDetector()
    result = detector.detect_simple_table(sheet)

    print(f"\nSimple table detected: {result.is_simple_table}")
    print(f"Table range: {result.table_range}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Has headers: {result.has_headers}")
    print(f"Reason: {result.reason}")


async def demonstrate_island_detection():
    """Demonstrate island detection on a multi-table sheet."""
    print("\n" + "=" * 60)
    print("ISLAND DETECTION DEMO")
    print("=" * 60)

    from gridporter.models.sheet_data import CellData, SheetData

    sheet = SheetData(name="MultiTableSheet")

    # Table 1: Sales by Region (top-left)
    sheet.set_cell(
        0, 0, CellData(value="Sales by Region", data_type="string", is_bold=True, row=0, column=0)
    )
    sheet.set_cell(
        1, 0, CellData(value="Region", data_type="string", is_bold=True, row=1, column=0)
    )
    sheet.set_cell(1, 1, CellData(value="Q1", data_type="string", is_bold=True, row=1, column=1))
    sheet.set_cell(1, 2, CellData(value="Q2", data_type="string", is_bold=True, row=1, column=2))

    regions = ["North", "South", "East", "West"]
    for i, region in enumerate(regions, 2):
        sheet.set_cell(i, 0, CellData(value=region, data_type="string", row=i, column=0))
        sheet.set_cell(i, 1, CellData(value=1000 + i * 100, data_type="number", row=i, column=1))
        sheet.set_cell(i, 2, CellData(value=1200 + i * 100, data_type="number", row=i, column=2))

    # Table 2: Product Inventory (top-right, with gap)
    start_col = 5  # Leave gap
    sheet.set_cell(
        0,
        start_col,
        CellData(
            value="Product Inventory", data_type="string", is_bold=True, row=0, column=start_col
        ),
    )
    sheet.set_cell(
        1,
        start_col,
        CellData(value="Product", data_type="string", is_bold=True, row=1, column=start_col),
    )
    sheet.set_cell(
        1,
        start_col + 1,
        CellData(value="Stock", data_type="string", is_bold=True, row=1, column=start_col + 1),
    )

    products = ["Widget A", "Widget B", "Widget C"]
    for i, product in enumerate(products, 2):
        sheet.set_cell(
            i, start_col, CellData(value=product, data_type="string", row=i, column=start_col)
        )
        sheet.set_cell(
            i,
            start_col + 1,
            CellData(value=50 + i * 10, data_type="number", row=i, column=start_col + 1),
        )

    # Table 3: Summary (bottom, with gap)
    start_row = 8  # Leave gap
    sheet.set_cell(
        start_row,
        0,
        CellData(value="Summary", data_type="string", is_bold=True, row=start_row, column=0),
    )
    sheet.set_cell(
        start_row + 1,
        0,
        CellData(value="Total Sales", data_type="string", row=start_row + 1, column=0),
    )
    sheet.set_cell(
        start_row + 1, 1, CellData(value=10000, data_type="number", row=start_row + 1, column=1)
    )

    # Run island detection
    detector = IslandDetector()
    islands = detector.detect_islands(sheet)

    print(f"\nDetected {len(islands)} data islands:")
    for i, island in enumerate(islands):
        print(f"\nIsland {i+1}:")
        print(f"  Range: {island.to_range()}")
        print(f"  Cells: {len(island.cells)}")
        print(f"  Density: {island.density:.2f}")
        print(f"  Confidence: {island.confidence:.2f}")
        print(f"  Has headers: {island.has_headers}")


async def demonstrate_cost_optimization():
    """Demonstrate cost optimization and tracking."""
    print("\n" + "=" * 60)
    print("COST OPTIMIZATION DEMO")
    print("=" * 60)

    # Create cost optimizer
    optimizer = CostOptimizer(
        max_cost_per_session=1.0, max_cost_per_file=0.1, confidence_threshold=0.8
    )

    # Simulate detection strategy selection
    sheet_complexity = {
        "has_merged_cells": False,
        "table_count": 1,
        "row_count": 100,
        "col_count": 10,
    }

    print("\nSimple sheet complexity:")
    for key, value in sheet_complexity.items():
        print(f"  {key}: {value}")

    strategies = optimizer.select_detection_strategy(sheet_complexity, available_hints=[])
    print("\nRecommended detection strategies:")
    for strategy in strategies:
        estimate = optimizer.estimate_cost(strategy, sheet_complexity)
        print(
            f"  - {strategy.value}: cost=${estimate.estimated_cost_usd:.3f}, "
            f"time={estimate.estimated_time_seconds:.1f}s, "
            f"confidence={estimate.confidence_range[0]:.1f}-{estimate.confidence_range[1]:.1f}"
        )

    # Simulate complex sheet
    complex_sheet = {
        "has_merged_cells": True,
        "table_count": 3,
        "has_sparse_data": True,
        "row_count": 5000,
        "col_count": 50,
    }

    print("\n\nComplex sheet complexity:")
    for key, value in complex_sheet.items():
        print(f"  {key}: {value}")

    strategies = optimizer.select_detection_strategy(complex_sheet, available_hints=[])
    print("\nRecommended detection strategies:")
    for strategy in strategies:
        estimate = optimizer.estimate_cost(strategy, complex_sheet)
        print(
            f"  - {strategy.value}: cost=${estimate.estimated_cost_usd:.3f}, "
            f"time={estimate.estimated_time_seconds:.1f}s, "
            f"confidence={estimate.confidence_range[0]:.1f}-{estimate.confidence_range[1]:.1f}"
        )

    # Show cost report
    from gridporter.utils.cost_optimizer import DetectionMethod

    optimizer.tracker.add_usage(DetectionMethod.SIMPLE_CASE)
    optimizer.tracker.add_usage(DetectionMethod.ISLAND_DETECTION)
    optimizer.tracker.add_usage(DetectionMethod.VISION_BASIC, tokens=1000, cost=0.01)

    print("\n\nCost Report:")
    report = optimizer.get_cost_report()
    for key, value in report.items():
        print(f"  {key}: {value}")


async def demonstrate_hybrid_detection():
    """Demonstrate the full hybrid detection pipeline."""
    print("\n" + "=" * 60)
    print("HYBRID DETECTION PIPELINE DEMO")
    print("=" * 60)

    # Create configuration with Week 6 features enabled
    config = Config(
        enable_simple_case_detection=True,
        enable_island_detection=True,
        use_excel_metadata=True,
        max_cost_per_file=0.1,
        confidence_threshold=0.8,
        enable_cache=True,
        use_vision=False,  # Disable vision for this demo
    )

    # Create the complex table agent
    agent = ComplexTableAgent(config)

    # Create a test sheet
    from gridporter.models.sheet_data import CellData, SheetData

    sheet = SheetData(name="TestSheet")

    # Add a simple table
    headers = ["ID", "Name", "Value"]
    for col, header in enumerate(headers):
        sheet.set_cell(
            0, col, CellData(value=header, data_type="string", is_bold=True, row=0, column=col)
        )

    for row in range(1, 6):
        sheet.set_cell(row, 0, CellData(value=row, data_type="number", row=row, column=0))
        sheet.set_cell(row, 1, CellData(value=f"Item {row}", data_type="string", row=row, column=1))
        sheet.set_cell(row, 2, CellData(value=row * 10.5, data_type="number", row=row, column=2))

    # Run hybrid detection
    result = await agent.detect_complex_tables(sheet)

    print("\nDetection Results:")
    print(f"  Tables found: {len(result.tables)}")
    print(f"  Overall confidence: {result.confidence:.2f}")
    print(f"  Methods used: {result.detection_metadata['methods_used']}")

    if "cost_report" in result.detection_metadata:
        print("\nCost Report:")
        cost_report = result.detection_metadata["cost_report"]
        print(f"  Total cost: ${cost_report['total_cost_usd']:.3f}")
        print(f"  Methods used: {cost_report['method_usage']}")

    for i, table in enumerate(result.tables):
        print(f"\nTable {i+1}:")
        print(f"  Range: {table.range.excel_range}")
        print(f"  Detection method: {table.detection_method}")
        print(f"  Confidence: {table.confidence:.2f}")
        print(f"  Has headers: {table.has_headers}")


async def main():
    """Run all demonstrations."""
    print("GridPorter Week 6: Hybrid Detection Demo")
    print("========================================")

    # Run each demonstration
    await demonstrate_simple_case_detection()
    await demonstrate_island_detection()
    await demonstrate_cost_optimization()
    await demonstrate_hybrid_detection()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
