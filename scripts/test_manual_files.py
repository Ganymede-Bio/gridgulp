#!/usr/bin/env python3
"""
Test all manual test files to verify detection works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp
from gridgulp.config import Config

# Expected results for each test file
EXPECTED_RESULTS = {
    # Level 0 - Basic files
    "test_basic.xlsx": {"tables": 1, "description": "Basic single table"},
    "test_comma.csv": {"tables": 1, "description": "Standard CSV"},
    "test_formatting.xlsx": {"tables": 1, "description": "Table with formatting"},
    "test_formulas.xlsx": {"tables": 1, "description": "Table with formulas"},
    "test_latin1.csv": {"tables": 1, "description": "Latin-1 encoded CSV"},
    "test_legacy.xls": {"tables": 1, "description": "Legacy Excel format"},
    "test_multi_sheet.xlsx": {
        "tables": "multiple",
        "description": "Multiple sheets with tables",
    },
    "test_pipe.csv": {"tables": 1, "description": "Pipe-delimited file"},
    "test_quoted.csv": {"tables": 1, "description": "CSV with quoted fields"},
    "test_semicolon.csv": {"tables": 1, "description": "Semicolon-delimited file"},
    "test_tab.tsv": {"tables": 1, "description": "Tab-separated values"},
    "test_types.csv": {"tables": 1, "description": "Various data types"},
    "test_utf16.csv": {"tables": 1, "description": "UTF-16 encoded file"},
    "test_utf8.csv": {"tables": 1, "description": "UTF-8 encoded file"},
    "large_file.csv": {"tables": 1, "description": "Large CSV file"},
    # Level 1 - Medium complexity
    "simple_table.xlsx": {"tables": 1, "description": "Simple table (4 cols × 6 rows)"},
    "simple_table.csv": {"tables": 1, "description": "Simple table CSV"},
    "large_table.xlsx": {
        "tables": 1,
        "description": "Large table (20 cols × 101 rows)",
    },
    "large_table.csv": {"tables": 1, "description": "Large table CSV"},
    "complex_table.xlsx": {
        "tables": 3,
        "description": "3 tables: Sales (A1:D6), Employees (F1:I5), Summary (A8:C10)",
    },
    # Level 2 - Complex files
    "creative_tables.xlsx": {
        "tables": "multiple",
        "description": "Creative table layouts",
    },
    "weird_tables.xlsx": {
        "tables": "multiple",
        "description": "Unusual table structures",
    },
    # Other
    "sample.xlsx": {"tables": "unknown", "description": "Sample file"},
    # Expected failures
    "corrupted.xlsx": {
        "tables": "error",
        "description": "Corrupted file (should fail)",
    },
    "protected.xlsx": {
        "tables": "error",
        "description": "Password protected (should fail)",
    },
}


async def test_file(porter: GridGulp, file_path: Path, expected: dict) -> dict:
    """Test a single file and return results."""
    result = {
        "file": file_path.name,
        "path": str(file_path),
        "expected": expected,
        "status": "unknown",
        "tables_found": 0,
        "error": None,
        "details": [],
    }

    try:
        detection_result = await porter.detect_tables(str(file_path))

        result["tables_found"] = detection_result.total_tables
        result["detection_time"] = detection_result.detection_time

        # Collect details about each table
        for sheet in detection_result.sheets:
            for table in sheet.tables:
                detail = {
                    "sheet": sheet.name,
                    "range": table.range.excel_range,
                    "shape": table.shape,
                    "confidence": table.confidence,
                    "method": table.detection_method,
                }
                result["details"].append(detail)

        # Check if result matches expectations
        if expected["tables"] == "error":
            result["status"] = "unexpected_success"
        elif expected["tables"] == "multiple":
            result["status"] = "pass" if result["tables_found"] > 1 else "fail"
        elif expected["tables"] == "unknown":
            result["status"] = "pass"  # Any result is acceptable
        elif isinstance(expected["tables"], int):
            result["status"] = "pass" if result["tables_found"] == expected["tables"] else "fail"
        else:
            result["status"] = "unknown"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error" if expected["tables"] == "error" else "fail"

    return result


async def test_all_files():
    """Test all manual test files."""
    print("Testing Manual Files")
    print("=" * 80)

    # Initialize GridGulp with default config
    config = Config(
        confidence_threshold=0.7,
        enable_simple_case_detection=True,
        enable_island_detection=True,
    )
    porter = GridGulp(config)

    # Find all test files
    base_path = Path(__file__).parent.parent / "tests" / "manual"

    # Organize by level
    levels = {"level0": [], "level1": [], "level2": [], "root": []}

    for file_name, expected in EXPECTED_RESULTS.items():
        # Skip image files
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            continue

        # Find file in different levels
        file_found = False
        for level in ["level0", "level1", "level2"]:
            file_path = base_path / level / file_name
            if file_path.exists():
                levels[level].append((file_path, expected))
                file_found = True
                break

        # Check root level
        if not file_found:
            file_path = base_path / file_name
            if file_path.exists():
                levels["root"].append((file_path, expected))

    # Test files by level
    all_results = []

    for level_name, files in levels.items():
        if not files:
            continue

        print(f"\n{level_name.upper()} Files:")
        print("-" * 80)

        for file_path, expected in sorted(files):
            result = await test_file(porter, file_path, expected)
            all_results.append(result)

            # Print result
            status_symbol = {
                "pass": "✓",
                "fail": "✗",
                "error": "!",
                "unexpected_success": "?",
            }.get(result["status"], "?")

            print(f"{status_symbol} {result['file']:<30} | ", end="")

            if result["status"] == "error":
                print(f"ERROR: {result['error'][:50]}")
            else:
                print(
                    f"Tables: {result['tables_found']:<2} | Time: {result['detection_time']:.3f}s | ",
                    end="",
                )
                if result["details"]:
                    # Show first table info
                    first_table = result["details"][0]
                    print(
                        f"Range: {first_table['range']:<15} | Conf: {first_table['confidence']:.2%}"
                    )
                else:
                    print("No tables detected")

            # Show all tables for multi-table files
            if len(result["details"]) > 1:
                for i, detail in enumerate(result["details"][1:], 2):
                    print(
                        f"  └─ Table {i}: {detail['range']:<15} | Sheet: {detail['sheet']:<15} | Conf: {detail['confidence']:.2%}"
                    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in all_results if r["status"] == "pass")
    failed = sum(1 for r in all_results if r["status"] == "fail")
    errors = sum(1 for r in all_results if r["status"] == "error")
    unexpected = sum(1 for r in all_results if r["status"] == "unexpected_success")

    print(f"Total files tested: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors (expected): {errors}")
    print(f"Unexpected successes: {unexpected}")

    # Show failures
    if failed > 0:
        print("\nFAILURES:")
        for r in all_results:
            if r["status"] == "fail":
                print(
                    f"  - {r['file']}: Expected {r['expected']['tables']} tables, found {r['tables_found']}"
                )

    # Show unexpected successes
    if unexpected > 0:
        print("\nUNEXPECTED SUCCESSES:")
        for r in all_results:
            if r["status"] == "unexpected_success":
                print(f"  - {r['file']}: Expected to fail, but found {r['tables_found']} tables")

    return passed, failed, errors, unexpected


if __name__ == "__main__":
    passed, failed, errors, unexpected = asyncio.run(test_all_files())

    # Exit with error code if there were failures
    if failed > 0 or unexpected > 0:
        sys.exit(1)
