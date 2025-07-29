#!/usr/bin/env python
"""Debug bottlenecks in large file processing."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp.detectors.simple_case_detector import SimpleCaseDetector
from gridgulp.readers.convenience import get_reader


def test_file_reading():
    """Test just the file reading performance."""
    file_path = Path("examples/spreadsheets/scientific/Sample T, pH data.csv")

    if not file_path.exists():
        print(f"❌ Large test file not found: {file_path}")
        return False

    print("🔍 FILE READING TEST")
    print("=" * 40)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # Test file reading
    start_time = time.time()
    try:
        reader = get_reader(str(file_path))
        print(f"Reader created: {time.time() - start_time:.3f}s")

        file_data = reader.read_sync()
        read_time = time.time() - start_time

        sheet_data = file_data.sheets[0]
        cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

        print("✅ File read successfully!")
        print(f"Read time: {read_time:.3f}s")
        print(f"Dimensions: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")
        print(f"Total cells: {cells:,}")
        print(f"Read rate: {cells/read_time:,.0f} cells/sec")

        return sheet_data

    except Exception as e:
        print(f"❌ File reading failed: {e}")
        return False


def test_simple_detection(sheet_data):
    """Test simple case detection performance."""
    print("\n🔍 SIMPLE DETECTION TEST")
    print("=" * 40)

    detector = SimpleCaseDetector()

    start_time = time.time()
    result = detector.detect_simple_table(sheet_data)
    detection_time = time.time() - start_time

    cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

    print(f"Detection time: {detection_time:.3f}s")
    print(f"Detection rate: {cells/detection_time:,.0f} cells/sec")
    print(f"Is simple table: {result.is_simple_table}")
    print(f"Confidence: {result.confidence}")
    print(f"Range: {result.table_range}")
    print(f"Reason: {result.reason}")

    return result


def test_sheet_data_methods(sheet_data):
    """Test various sheet data methods for performance."""
    print("\n🔍 SHEET DATA METHODS TEST")
    print("=" * 40)

    # Test get_non_empty_cells
    start_time = time.time()
    non_empty = sheet_data.get_non_empty_cells()
    non_empty_time = time.time() - start_time
    print(f"get_non_empty_cells(): {non_empty_time:.3f}s ({len(non_empty)} cells)")

    # Test has_data
    start_time = time.time()
    has_data = sheet_data.has_data()
    has_data_time = time.time() - start_time
    print(f"has_data(): {has_data_time:.3f}s (result: {has_data})")

    # Test random cell access
    start_time = time.time()
    for i in range(100):  # Sample 100 cells
        cell = sheet_data.get_cell(i, min(i, sheet_data.max_column))
    cell_access_time = time.time() - start_time
    print(f"100 cell accesses: {cell_access_time:.3f}s")


def main():
    """Run debugging tests."""
    print("🐛 LARGE FILE BOTTLENECK DEBUGGING")
    print("=" * 60)

    # Step 1: Test file reading
    sheet_data = test_file_reading()
    if not sheet_data:
        return

    # Step 2: Test sheet data methods
    test_sheet_data_methods(sheet_data)

    # Step 3: Test simple detection
    result = test_simple_detection(sheet_data)

    # Summary
    print("\n📊 SUMMARY")
    print("=" * 40)
    cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
    print(f"Total cells: {cells:,}")
    print(f"Expected ground truth: A1:W9066 ({23 * 9066:,} cells)")

    if result.is_simple_table:
        print("✅ Simple detection working - this should trigger ultra-fast path!")
    else:
        print("❌ Simple detection failed - will use slower methods")
        print(f"   Reason: {result.reason}")


if __name__ == "__main__":
    main()
