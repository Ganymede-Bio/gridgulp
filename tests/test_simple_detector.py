#!/usr/bin/env python
"""Test simple case detector directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp.detectors.simple_case_detector import SimpleCaseDetector
from gridgulp.readers.convenience import get_reader


def test_simple_detector():
    """Test simple case detector directly."""
    detector = SimpleCaseDetector()

    # Create test file
    test_file = Path("test_simple.csv")
    with open(test_file, "w") as f:
        f.write("Name,Age,City\n")
        f.write("Alice,25,NYC\n")
        f.write("Bob,30,LA\n")
        f.write("Charlie,35,Chicago\n")

    try:
        # Read file
        reader = get_reader(test_file)
        file_data = reader.read_sync()
        sheet_data = file_data.sheets[0]

        print(f"Sheet: {sheet_data.name}")
        print(f"Size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")
        print(f"Non-empty cells: {len(sheet_data.get_non_empty_cells())}")

        # Test detection
        result = detector.detect_simple_table(sheet_data)
        print(f"\nDetection result: {result}")

        if result and result.is_simple_table:
            print("  Single table detected!")
            print(f"  Range: {result.table_range}")
            print(f"  Confidence: {result.confidence}")

            # Convert to TableInfo
            table_info = detector.convert_to_table_info(result, sheet_data.name)
            if table_info:
                print("\nTableInfo created:")
                print(f"  ID: {table_info.id}")
                print(f"  Range: {table_info.range.excel_range}")
                print(f"  Method: {table_info.detection_method}")
            else:
                print("\nFailed to create TableInfo")
        else:
            print("  No single table detected")

    finally:
        test_file.unlink()


if __name__ == "__main__":
    test_simple_detector()
