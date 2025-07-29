#!/usr/bin/env python3
"""
Unified test script to run GridGulp detection on all spreadsheet files in the repository.
Combines functionality from test_example_files.py, test_manual_files.py, and capture_outputs.py.

Usage Examples:
    # Test all files in default directories (examples/ and tests/manual/)
    python scripts/test_all_spreadsheets.py

    # Test specific directory
    python scripts/test_all_spreadsheets.py --directories ~/my_spreadsheets

    # Test only Excel files
    python scripts/test_all_spreadsheets.py --pattern "*.xlsx" --pattern "*.xls"

    # Test and save results in JSON format
    python scripts/test_all_spreadsheets.py --save --format json

    # Quiet mode for CI/CD
    python scripts/test_all_spreadsheets.py --quiet --save

    # Test specific file pattern
    python scripts/test_all_spreadsheets.py --pattern "sales_*.xlsx"

Output:
    The script provides detailed console output showing:
    - File processing status (✓ success, ✗ error, ⚠️ validation mismatch)
    - Number of tables detected
    - Processing time and file size
    - Detection method used
    - Table details including range, headers, and confidence

    When --save is used, results are saved in the specified format:
    - JSON: Comprehensive machine-readable format with all details
    - CSV: Simple tabular format for spreadsheet analysis
    - Markdown: Human-readable report format

Exit Codes:
    0: All tests passed successfully
    1: One or more tests failed or had validation errors

For more information, see docs/TESTING_WITH_SCRIPT.md
"""

import argparse
import asyncio
import csv as csv_module
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp
from gridgulp.config import Config

# Expected results for manual test files
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


class SpreadsheetTester:
    """Unified spreadsheet testing framework."""

    def __init__(self, config: Config | None = None):
        """Initialize tester with optional config."""
        self.config = config or Config(
            confidence_threshold=0.7,
            enable_simple_case_detection=True,
            enable_island_detection=True,
        )
        self.porter = GridGulp(self.config)
        self.results: list[dict] = []
        self.errors: list[dict] = []
        self.timestamp = datetime.now()

    async def test_file(self, file_path: Path) -> dict:
        """Test a single file and return results."""
        result = {
            "file": file_path.name,
            "path": str(file_path),
            "relative_path": str(file_path),
            "directory": str(file_path.parent),
            "size_kb": file_path.stat().st_size / 1024,
            "status": "unknown",
            "tables_found": 0,
            "sheets_found": 0,
            "detection_time": 0.0,
            "error": None,
            "details": [],
            "expected": None,
        }

        # Check if we have expected results
        if file_path.name in EXPECTED_RESULTS:
            result["expected"] = EXPECTED_RESULTS[file_path.name]

        try:
            # Run detection
            start_time = datetime.now()
            detection_result = await self.porter.detect_tables(str(file_path))
            end_time = datetime.now()

            # Extract results
            result["detection_time"] = (end_time - start_time).total_seconds()
            result["tables_found"] = detection_result.total_tables
            result["sheets_found"] = len(detection_result.sheets)
            result["status"] = "success"

            # File info
            file_info = detection_result.file_info
            result["file_info"] = {
                "detected_type": str(file_info.type),
                "extension_format": str(file_info.extension_format)
                if file_info.extension_format
                else None,
                "encoding": file_info.encoding,
                "format_mismatch": getattr(file_info, "format_mismatch", False),
                "detection_method": getattr(file_info, "detection_method", None),
                "detection_confidence": getattr(file_info, "detection_confidence", None),
            }

            # Add metadata if available
            if hasattr(detection_result, "metadata") and detection_result.metadata:
                result["metadata"] = detection_result.metadata

            # Collect details about each sheet and table
            for sheet in detection_result.sheets:
                sheet_data = {
                    "name": sheet.name,
                    "tables": [],
                    "processing_time": sheet.processing_time,
                }

                for table in sheet.tables:
                    table_data = {
                        "id": table.id,
                        "range": table.range.excel_range,
                        "shape": table.shape,
                        "confidence": table.confidence,
                        "method": table.detection_method,
                        "suggested_name": table.suggested_name,
                        "headers": table.headers if table.headers else None,
                        "header_count": len(table.headers) if table.headers else 0,
                    }
                    sheet_data["tables"].append(table_data)

                result["details"].append(sheet_data)

            # Validate against expected results
            if result["expected"]:
                expected = result["expected"]
                if expected["tables"] == "error":
                    result["validation"] = "unexpected_success"
                elif expected["tables"] == "multiple":
                    result["validation"] = "pass" if result["tables_found"] > 1 else "fail"
                elif expected["tables"] == "unknown":
                    result["validation"] = "pass"
                elif isinstance(expected["tables"], int):
                    result["validation"] = (
                        "pass" if result["tables_found"] == expected["tables"] else "fail"
                    )
                else:
                    result["validation"] = "unknown"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.errors.append({"file": file_path.name, "error": str(e)})

            # Validate error cases
            if result["expected"] and result["expected"]["tables"] == "error":
                result["validation"] = "pass"
            elif result["expected"]:
                result["validation"] = "fail"

        return result

    async def test_directory(
        self, directory: Path, patterns: list[str] | None = None, recursive: bool = True
    ) -> list[dict]:
        """Test all files in a directory matching the given patterns."""
        if patterns is None:
            patterns = ["*.xlsx", "*.xls", "*.csv", "*.tsv", "*.txt"]

        # Collect all files
        all_files = []
        for pattern in patterns:
            if recursive:
                all_files.extend(directory.rglob(pattern))
            else:
                all_files.extend(directory.glob(pattern))

        # Sort files by path for consistent output
        all_files.sort()

        # Test each file
        results = []
        for file_path in all_files:
            # Skip temporary or hidden files
            if file_path.name.startswith(".") or file_path.name.startswith("~"):
                continue

            result = await self.test_file(file_path)
            results.append(result)
            self.results.append(result)

        return results

    def print_results(self, results: list[dict], title: str = "Test Results", verbose: bool = True):
        """Print formatted results to console."""
        print(f"\n{'=' * 100}")
        print(f"{title}")
        print(f"{'=' * 100}")

        if not results:
            print("No files tested.")
            return

        # Group by directory
        by_dir: dict[str, list[dict]] = {}
        for result in results:
            dir_path = Path(result["path"]).parent
            try:
                dir_key = str(dir_path.relative_to(Path.cwd()))
            except ValueError:
                # If path is not relative to cwd, use the parent directory name
                dir_key = str(dir_path)
            if dir_key not in by_dir:
                by_dir[dir_key] = []
            by_dir[dir_key].append(result)

        # Print by directory
        for dir_path in sorted(by_dir.keys()):
            dir_results = by_dir[dir_path]
            print(f"\n📁 {dir_path}")
            print("-" * 100)

            for result in sorted(dir_results, key=lambda x: x["file"]):
                # Status symbol
                if result["status"] == "error":
                    status_symbol = "✗"
                elif result.get("validation") == "fail":
                    status_symbol = "⚠️"
                else:
                    status_symbol = "✓"

                print(f"{status_symbol} {result['file']:<40} | ", end="")

                if result["status"] == "error":
                    print(f"ERROR: {result['error'][:40]}")
                else:
                    detection_info = ""
                    if result["file_info"].get("detection_method"):
                        detection_info = f" | Method: {result['file_info']['detection_method']}"

                    validation_info = ""
                    if result.get("expected") and result.get("validation") == "fail":
                        validation_info = f" | MISMATCH: expected {result['expected']['tables']}"

                    print(
                        f"Tables: {result['tables_found']:<2} | "
                        f"Time: {result['detection_time']:.3f}s | "
                        f"Size: {result['size_kb']:.1f}KB"
                        f"{detection_info}"
                        f"{validation_info}"
                    )

                    # Show format mismatch warning
                    if result.get("file_info", {}).get("format_mismatch"):
                        detected = result["file_info"]["detected_type"]
                        extension = result["file_info"]["extension_format"]
                        print(
                            f"  ⚠️  Format mismatch: detected as {detected}, extension suggests {extension}"
                        )

                    # Show table details if verbose
                    if verbose and result["tables_found"] >= 1:
                        for sheet_data in result["details"]:
                            if sheet_data["tables"]:
                                print(f"  📄 Sheet: {sheet_data['name']}")
                                for table in sheet_data["tables"]:
                                    headers_str = ""
                                    if table.get("headers"):
                                        headers_preview = ", ".join(
                                            str(h) for h in table["headers"][:3]
                                        )
                                        if table.get("header_count", 0) > 3:
                                            headers_preview += f" ... +{table['header_count'] - 3}"
                                        headers_str = f" | Headers: {headers_preview}"

                                    print(
                                        f"     └─ {table['range']:<15} | "
                                        f"{table['shape'][0]}×{table['shape'][1]} | "
                                        f"Conf: {table['confidence']:.0%}"
                                        f"{headers_str}"
                                    )

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)

        if not self.results:
            print("No files were tested.")
            return

        # Basic counts
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]

        print(f"Total files tested: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        # Validation results
        if any(r.get("validation") for r in self.results):
            validation_pass = [r for r in self.results if r.get("validation") == "pass"]
            validation_fail = [r for r in self.results if r.get("validation") == "fail"]
            print("\nValidation Results:")
            print(f"  Passed: {len(validation_pass)}")
            print(f"  Failed: {len(validation_fail)}")

            if validation_fail:
                print("\n  Validation Failures:")
                for r in validation_fail[:10]:  # Show first 10
                    expected = r["expected"]["tables"]
                    actual = r["tables_found"]
                    print(f"    - {r['file']}: expected {expected}, got {actual}")
                if len(validation_fail) > 10:
                    print(f"    ... and {len(validation_fail) - 10} more")

        if successful:
            # Table statistics
            total_tables = sum(r["tables_found"] for r in successful)
            files_with_tables = [r for r in successful if r["tables_found"] > 0]
            multi_table_files = [r for r in successful if r["tables_found"] > 1]

            print("\nTable Detection:")
            print(f"  Total tables found: {total_tables}")
            print(f"  Files with tables: {len(files_with_tables)}")
            print(f"  Files with multiple tables: {len(multi_table_files)}")
            print(f"  Files with no tables: {len(successful) - len(files_with_tables)}")

            # Performance statistics
            times = [r["detection_time"] for r in successful]
            print("\nPerformance:")
            print(f"  Total processing time: {sum(times):.3f}s")
            print(f"  Average time per file: {sum(times) / len(times):.3f}s")
            print(f"  Fastest: {min(times):.3f}s")
            print(f"  Slowest: {max(times):.3f}s")

            # File type statistics
            file_types: dict[str, int] = {}
            for r in successful:
                ext = Path(r["file"]).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

            print("\nFile Types Tested:")
            for ext, count in sorted(file_types.items()):
                print(f"  {ext}: {count}")

            # Format mismatches
            mismatches = [r for r in successful if r.get("file_info", {}).get("format_mismatch")]
            if mismatches:
                print(f"\nFormat Mismatches: {len(mismatches)}")
                for r in mismatches[:5]:
                    print(f"  - {r['file']}: detected as {r['file_info']['detected_type']}")
                if len(mismatches) > 5:
                    print(f"  ... and {len(mismatches) - 5} more")

        # Errors
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error['file']}: {error['error'][:60]}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")

    def save_results(self, output_path: Path, format: str = "json") -> Path:
        """Save results in specified format."""
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = output_path / f"test_results_{timestamp}.json"
            self._save_json(output_file)
        elif format == "csv":
            output_file = output_path / f"test_results_{timestamp}.csv"
            self._save_csv(output_file)
        elif format == "markdown":
            output_file = output_path / f"test_results_{timestamp}.md"
            self._save_markdown(output_file)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_file

    def _save_json(self, output_file: Path):
        """Save results as JSON."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "enable_simple_case": self.config.enable_simple_case_detection,
                "enable_island": self.config.enable_island_detection,
            },
            "summary": {
                "total_files": len(self.results),
                "successful": len([r for r in self.results if r["status"] == "success"]),
                "failed": len([r for r in self.results if r["status"] == "error"]),
                "total_tables": sum(r["tables_found"] for r in self.results),
                "total_time": sum(r["detection_time"] for r in self.results),
            },
            "results": self.results,
            "errors": self.errors,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, output_file: Path):
        """Save results as CSV."""
        if not self.results:
            return

        fieldnames = [
            "file",
            "path",
            "status",
            "tables_found",
            "sheets_found",
            "detection_time",
            "size_kb",
            "detected_type",
            "format_mismatch",
            "validation",
            "error",
        ]

        with open(output_file, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "file": result["file"],
                    "path": result["path"],
                    "status": result["status"],
                    "tables_found": result["tables_found"],
                    "sheets_found": result["sheets_found"],
                    "detection_time": f"{result['detection_time']:.3f}",
                    "size_kb": f"{result['size_kb']:.1f}",
                    "detected_type": result.get("file_info", {}).get("detected_type", ""),
                    "format_mismatch": result.get("file_info", {}).get("format_mismatch", False),
                    "validation": result.get("validation", ""),
                    "error": result.get("error", ""),
                }
                writer.writerow(row)

    def _save_markdown(self, output_file: Path):
        """Save results as Markdown."""
        with open(output_file, "w") as f:
            f.write("# GridGulp Test Results\n\n")
            f.write(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total files tested: {len(self.results)}\n")
            f.write(f"- Successful: {len([r for r in self.results if r['status'] == 'success'])}\n")
            f.write(f"- Failed: {len([r for r in self.results if r['status'] == 'error'])}\n")
            f.write(f"- Total tables found: {sum(r['tables_found'] for r in self.results)}\n\n")

            # Results table
            f.write("## Results\n\n")
            f.write("| File | Status | Tables | Time (s) | Size (KB) | Notes |\n")
            f.write("|------|--------|--------|----------|-----------|-------|\n")

            for result in sorted(self.results, key=lambda x: x["path"]):
                status = "✓" if result["status"] == "success" else "✗"
                notes = []
                if result.get("validation") == "fail":
                    notes.append(f"Expected {result['expected']['tables']}")
                if result.get("file_info", {}).get("format_mismatch"):
                    notes.append("Format mismatch")
                if result.get("error"):
                    notes.append(result["error"][:30])

                f.write(
                    f"| {result['file']} | {status} | "
                    f"{result['tables_found']} | {result['detection_time']:.3f} | "
                    f"{result['size_kb']:.1f} | {', '.join(notes)} |\n"
                )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test GridGulp on all spreadsheet files in the repository"
    )
    parser.add_argument(
        "--directories",
        type=Path,
        nargs="+",
        default=[Path("examples"), Path("tests/manual")],
        help="Directories to test - recursively searches all subdirectories by default (default: examples tests/manual)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        action="append",
        help="File patterns to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format for saved results",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detailed results to file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/outputs"),
        help="Output directory for results (default: tests/outputs)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively (by default, all subdirectories are searched)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output (no table details)",
    )

    args = parser.parse_args()

    # Create output directory if saving
    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create tester
    tester = SpreadsheetTester()

    print("Testing GridGulp on spreadsheet files")
    print(f"Directories: {', '.join(str(d) for d in args.directories)}")
    if args.pattern:
        print(f"Patterns: {', '.join(args.pattern)}")
    print()

    # Run tests on all directories
    all_results = []
    for directory in args.directories:
        if not directory.exists():
            print(f"Warning: Directory '{directory}' does not exist, skipping.")
            continue

        results = await tester.test_directory(
            directory,
            patterns=args.pattern,
            recursive=not args.no_recursive,
        )
        all_results.extend(results)

    # Print results
    if all_results:
        tester.print_results(
            all_results,
            "Test Results for All Spreadsheets",
            verbose=not args.quiet,
        )
        tester.print_summary()

        # Save results if requested
        if args.save:
            output_path = tester.save_results(args.output_dir, args.format)
            print(f"\nResults saved to: {output_path}")

        # Exit with error code if there were failures or validation errors
        validation_failures = [r for r in all_results if r.get("validation") == "fail"]
        if tester.errors or validation_failures:
            sys.exit(1)
    else:
        print("No files found to test.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
