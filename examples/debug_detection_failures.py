#!/usr/bin/env python
"""Debug why detection is failing to find expected tables."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.detectors.island_detector import IslandDetector
from gridporter.detectors.simple_case_detector import SimpleCaseDetector
from gridporter.readers.convenience import get_reader


def load_ground_truth():
    """Load ground truth for comparison."""
    gt_path = Path("examples/structured_ground_truth.json")
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return None


async def debug_single_file(file_path: Path, expected_tables: list):
    """Debug detection for a single file."""
    print(f"\n{'='*80}")
    print(f"DEBUGGING: {file_path.name}")
    print(f"Expected tables: {len(expected_tables)}")
    for i, range_str in enumerate(expected_tables):
        print(f"  {i+1}. {range_str}")
    print("=" * 80)

    # Read file
    try:
        reader = get_reader(str(file_path))
        file_data = reader.read_sync()
        sheet_data = file_data.sheets[0]

        print("File successfully read:")
        print(f"  Sheet name: {sheet_data.name}")
        print(f"  Dimensions: {sheet_data.max_row + 1}x{sheet_data.max_column + 1}")
        print(f"  Total cells: {(sheet_data.max_row + 1) * (sheet_data.max_column + 1)}")

        # Show first few rows of data
        print("\nFirst 5 rows of data:")
        for row in range(min(5, sheet_data.max_row + 1)):
            row_data = []
            for col in range(min(8, sheet_data.max_column + 1)):
                cell = sheet_data.get_cell(row, col)
                value = str(cell.value)[:20] if cell and cell.value else ""
                row_data.append(value)
            print(f"  Row {row}: {row_data}")

        # Check if data looks like it has tables
        non_empty_cells = len(sheet_data.get_non_empty_cells())
        print(f"Non-empty cells: {non_empty_cells}")

        # Test individual detectors
        print("\nTesting individual detectors:")

        # 1. Simple case detector
        simple_detector = SimpleCaseDetector()
        simple_result = simple_detector.detect_simple_table(sheet_data)
        print("  Simple case detector:")
        print(f"    Is simple table: {simple_result.is_simple_table if simple_result else False}")
        print(f"    Confidence: {simple_result.confidence if simple_result else 'N/A'}")
        print(f"    Range: {simple_result.table_range if simple_result else 'N/A'}")

        # 2. Island detector
        island_detector = IslandDetector()
        islands = island_detector.detect_islands(sheet_data)
        print("  Island detector:")
        print(f"    Islands found: {len(islands)}")
        for i, island in enumerate(islands):
            print(
                f"      Island {i+1}: {island.min_row},{island.min_col} to {island.max_row},{island.max_col} (density: {island.density:.2f})"
            )

        # Convert islands to table infos
        if islands:
            table_infos = island_detector.convert_to_table_infos(
                islands,
                sheet_data.name,
                min_confidence=0.1,  # Lower threshold for testing
            )
            print(f"    Converted to {len(table_infos)} table infos")
            for table_info in table_infos:
                print(f"      {table_info.range.excel_range} (confidence: {table_info.confidence})")

        # 3. Full orchestrator
        config = Config(
            use_vision=False, confidence_threshold=0.1, log_level="ERROR"
        )  # Lower threshold
        orchestrator = VisionOrchestratorAgent(config)

        print("\nFull orchestrator detection:")
        result = await orchestrator.orchestrate_detection(sheet_data)
        print(f"  Strategy used: {result.orchestrator_decision.detection_strategy}")
        print(f"  Complexity score: {result.complexity_assessment.complexity_score}")
        print(f"  Tables detected: {len(result.tables)}")

        for i, table in enumerate(result.tables):
            print(
                f"    Table {i+1}: {table.range.excel_range} (confidence: {table.confidence}, method: {table.detection_method})"
            )

        # Compare with ground truth
        detected_ranges = {table.range.excel_range.upper() for table in result.tables}
        expected_ranges = {r.upper() for r in expected_tables}

        print("\nGround truth comparison:")
        print(f"  Expected: {expected_ranges}")
        print(f"  Detected: {detected_ranges}")
        print(f"  Matches: {expected_ranges.intersection(detected_ranges)}")
        print(f"  Missing: {expected_ranges - detected_ranges}")
        print(f"  Extra: {detected_ranges - expected_ranges}")

        # Analyze why tables might be missed
        print("\nAnalysis:")
        if not result.tables:
            print("  NO TABLES DETECTED - investigating why:")

            # Check if simple detection threshold is too high
            if simple_result and simple_result.is_simple_table:
                print(
                    f"    - Simple detector found table but confidence too low: {simple_result.confidence}"
                )

            # Check if all strategies are failing
            print("    - Check logs for 'All detection strategies failed' messages")

            # Check confidence thresholds
            print(f"    - Current confidence threshold: {config.confidence_threshold}")
            print("    - Consider lowering threshold or improving detection logic")

    except Exception as e:
        print(f"Error debugging file: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Debug detection failures on key files."""
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("No ground truth found")
        return

    # Test key files that should have tables but are showing 0 detected
    test_cases = [
        ("spreadsheets/simple/product_inventory.csv", ["A1:B7"]),
        ("spreadsheets/sales/monthly_sales.csv", ["A1:G17"]),
        (
            "spreadsheets/financial/balance_sheet.csv",
            ["A1:C7", "A9:C15", "A17:B17", "A20:C26", "A28:C33", "A36:C40"],
        ),
    ]

    for file_rel_path, expected_ranges in test_cases:
        file_path = Path("examples") / file_rel_path
        if file_path.exists():
            await debug_single_file(file_path, expected_ranges)
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
