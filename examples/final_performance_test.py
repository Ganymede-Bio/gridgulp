#!/usr/bin/env python
"""Final comprehensive performance test across all file sizes."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


def performance_grade(cells_per_sec: float) -> str:
    """Get performance grade with emoji."""
    if cells_per_sec >= 500000:
        return "🚀 S+ (500K+ cells/sec) - SUPERB!"
    elif cells_per_sec >= 200000:
        return "🏆 S (200K+ cells/sec) - EXCELLENT!"
    elif cells_per_sec >= 100000:
        return "🥇 A+ (100K+ cells/sec) - TARGET ACHIEVED!"
    elif cells_per_sec >= 50000:
        return "🥈 A (50K+ cells/sec) - Very Good"
    elif cells_per_sec >= 25000:
        return "🥉 B (25K+ cells/sec) - Good"
    elif cells_per_sec >= 10000:
        return "✅ C (10K+ cells/sec) - Acceptable"
    else:
        return "❌ F (<10K cells/sec) - Needs Work"


async def test_file(file_path: Path, expected_range: str = None):
    """Test a single file."""
    try:
        # Read file
        reader = get_reader(str(file_path))
        file_data = reader.read_sync()
        sheet_data = file_data.sheets[0]

        cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

        # Run detection
        config = Config(use_vision=False, confidence_threshold=0.5, log_level="ERROR")
        agent = ComplexTableAgent(config)

        start_time = time.time()
        result = await agent.detect_complex_tables(sheet_data)
        detection_time = time.time() - start_time

        cells_per_sec = cells / detection_time if detection_time > 0 else 0

        return {
            "file": file_path.name,
            "cells": cells,
            "time": detection_time,
            "rate": cells_per_sec,
            "grade": performance_grade(cells_per_sec),
            "tables": len(result.tables),
            "method": result.tables[0].detection_method if result.tables else "none",
            "range": result.tables[0].range.excel_range if result.tables else "none",
            "expected": expected_range,
            "match": (
                (result.tables[0].range.excel_range.upper() == expected_range.upper())
                if result.tables and expected_range
                else None
            ),
            "success": True,
        }

    except Exception as e:
        return {"file": file_path.name, "error": str(e), "success": False}


async def main():
    """Run comprehensive performance test."""
    print("🚀 FINAL COMPREHENSIVE PERFORMANCE TEST")
    print("=" * 80)
    print("Testing GridPorter detection across all file sizes")
    print()

    # Test files from small to very large
    test_cases = [
        ("spreadsheets/simple/product_inventory.csv", "A1:F7", "Small dense table"),
        ("spreadsheets/sales/monthly_sales.csv", "A1:G17", "Medium dense table"),
        (
            "spreadsheets/financial/income_statement.csv",
            "A1:F35",
            "Medium sparse table",
        ),
        ("spreadsheets/financial/balance_sheet.csv", None, "Multi-table file"),
        (
            "spreadsheets/scientific/Sample T, pH data.csv",
            "A1:W9066",
            "Very large dense table",
        ),
    ]

    results = []
    total_cells = 0
    total_time = 0

    for file_rel_path, expected_range, description in test_cases:
        file_path = Path("examples") / file_rel_path
        if not file_path.exists():
            print(f"❌ Missing: {file_path.name}")
            continue

        print(f"Testing: {file_path.name} ({description})")
        result = await test_file(file_path, expected_range)
        results.append({**result, "description": description})

        if result["success"]:
            total_cells += result["cells"]
            total_time += result["time"]

            print(f"  ✅ {result['cells']:,} cells in {result['time']:.3f}s")
            print(f"     {result['grade']}")
            print(f"     Method: {result['method']}")
            if result.get("match") is not None:
                match_str = "✅ MATCH" if result["match"] else "⚠️ DIFFERENT"
                print(f"     Range: {result['range']} ({match_str})")
        else:
            print(f"  ❌ ERROR: {result['error']}")
        print()

    # Results table
    print("📊 PERFORMANCE RESULTS TABLE")
    print("=" * 120)
    print(f"{'File':<40} {'Cells':<10} {'Time':<8} {'Rate':<15} {'Method':<20} {'Grade'}")
    print("-" * 120)

    for result in results:
        if result["success"]:
            rate_str = f"{result['rate']:,.0f}/s"
            grade_str = result["grade"].split(" ")[1]  # Extract grade letter
            print(
                f"{result['file']:<40} {result['cells']:<10,} {result['time']:<8.3f} {rate_str:<15} {result['method']:<20} {grade_str}"
            )

    # Overall summary
    print("\n" + "=" * 120)
    print("🎯 FINAL SUMMARY")
    print("=" * 120)

    successful_tests = [r for r in results if r["success"]]

    if total_time > 0:
        overall_rate = total_cells / total_time
        overall_grade = performance_grade(overall_rate)

        print(f"Files tested: {len(successful_tests)}/{len(results)}")
        print(f"Total cells processed: {total_cells:,}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Overall rate: {overall_rate:,.0f} cells/sec")
        print(f"Overall grade: {overall_grade}")

        # Method breakdown
        methods = {}
        for result in successful_tests:
            method = result["method"]
            methods[method] = methods.get(method, 0) + 1

        print("\nDetection methods used:")
        for method, count in sorted(methods.items()):
            print(f"  - {method}: {count} files")

        # Performance achievement
        print("\n🎉 ACHIEVEMENT SUMMARY:")
        target_met = sum(1 for r in successful_tests if r["rate"] >= 100000)
        print(f"✅ Files meeting 100K cells/sec target: {target_met}/{len(successful_tests)}")

        if overall_rate >= 100000:
            print(f"🏆 OVERALL TARGET ACHIEVED: {overall_rate:,.0f} cells/sec exceeds 100K target!")
        else:
            print(f"⚠️  Overall target not met: {overall_rate:,.0f} cells/sec < 100K target")

        # Ultra-fast path usage
        ultra_fast_count = sum(1 for r in successful_tests if r.get("method") == "ultra_fast")
        if ultra_fast_count > 0:
            print(f"🚀 Ultra-fast path used on {ultra_fast_count} large files")


if __name__ == "__main__":
    asyncio.run(main())
