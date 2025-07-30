#!/usr/bin/env python3
"""
Test script to run GridGulp detection on all files in the examples/ directory.
This script provides comprehensive testing and reporting for example files.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp
from gridgulp.config import Config


class ExampleFileTester:
    """Test runner for example files."""

    def __init__(self, config: Config | None = None):
        """Initialize the tester with optional config."""
        self.config = config or Config(
            confidence_threshold=0.7,
            enable_simple_case_detection=True,
            enable_island_detection=True,
        )
        self.porter = GridGulp(self.config)
        self.results: list[dict] = []
        self.errors: list[dict] = []

    async def test_file(self, file_path: Path) -> dict:
        """Test a single file and return results."""
        result = {
            "file": file_path.name,
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(Path("examples"))),
            "size_kb": file_path.stat().st_size / 1024,
            "status": "unknown",
            "tables_found": 0,
            "sheets_found": 0,
            "detection_time": 0.0,
            "error": None,
            "details": [],
        }

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
                "detected_type": file_info.type,
                "extension_format": file_info.extension_format,
                "encoding": file_info.encoding,
                "format_mismatch": (
                    file_info.format_mismatch if hasattr(file_info, "format_mismatch") else False
                ),
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
                        "headers": table.headers[:5] if table.headers else None,
                        "header_count": len(table.headers) if table.headers else 0,
                    }
                    sheet_data["tables"].append(table_data)

                result["details"].append(sheet_data)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.errors.append({"file": file_path.name, "error": str(e)})

        return result

    async def test_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        recursive: bool = True,
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

    def print_results(self, results: list[dict], title: str = "Test Results"):
        """Print formatted results."""
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}")

        if not results:
            print("No files tested.")
            return

        # Group by subdirectory
        by_dir: dict[str, list[dict]] = {}
        for result in results:
            dir_path = str(Path(result["relative_path"]).parent)
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append(result)

        # Print by directory
        for dir_path in sorted(by_dir.keys()):
            dir_results = by_dir[dir_path]
            print(f"\n📁 {dir_path}/")
            print("-" * 80)

            for result in sorted(dir_results, key=lambda x: x["file"]):
                status_symbol = "✓" if result["status"] == "success" else "✗"
                print(f"{status_symbol} {result['file']:<40} | ", end="")

                if result["status"] == "error":
                    print(f"ERROR: {result['error'][:40]}")
                else:
                    detection_info = ""
                    if result["file_info"].get("detection_method"):
                        detection_info = f" | Detection: {result['file_info']['detection_method']}"

                    print(
                        f"Tables: {result['tables_found']:<2} | "
                        f"Sheets: {result['sheets_found']:<2} | "
                        f"Time: {result['detection_time']:.3f}s | "
                        f"Size: {result['size_kb']:.1f}KB"
                        f"{detection_info}"
                    )

                    # Show format mismatch warning
                    if result.get("file_info", {}).get("format_mismatch"):
                        detected = result["file_info"]["detected_type"]
                        extension = result["file_info"]["extension_format"]
                        print(
                            f"  ⚠️  Format mismatch: detected as {detected}, extension suggests {extension}"
                        )

                    # Show table details for multi-table files or if file has tables
                    if result["tables_found"] >= 1:
                        for sheet_data in result["details"]:
                            if sheet_data["tables"]:
                                print(f"  📄 Sheet: {sheet_data['name']}")
                                for table in sheet_data["tables"]:
                                    headers_str = ""
                                    if table.get("headers"):
                                        headers_preview = ", ".join(table["headers"][:3])
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
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        if not self.results:
            print("No files were tested.")
            return

        # Basic counts
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]

        print(f"Total files tested: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

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
                for r in mismatches:
                    print(f"  - {r['file']}: detected as {r['file_info']['detected_type']}")

        # Errors
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error['file']}: {error['error'][:60]}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")

    def save_results(self, output_path: Path | None = None) -> Path:
        """Save detailed results to JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"example_test_results_{timestamp}.json")

        data = {
            "timestamp": datetime.now().isoformat(),
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

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test GridGulp on example files")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("examples"),
        help="Directory to test (default: examples)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        action="append",
        help="File patterns to test (can be specified multiple times)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (default: auto-generated)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively",
    )

    args = parser.parse_args()

    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)

    # Create tester
    tester = ExampleFileTester()

    print(f"Testing files in: {args.directory}")
    print(f"Recursive: {not args.no_recursive}")
    if args.pattern:
        print(f"Patterns: {', '.join(args.pattern)}")
    print()

    # Run tests
    try:
        results = await tester.test_directory(
            args.directory,
            patterns=args.pattern,
            recursive=not args.no_recursive,
        )

        # Print results
        tester.print_results(results, f"Results for {args.directory}")
        tester.print_summary()

        # Save results if requested
        if args.save_results:
            output_path = tester.save_results(args.output)
            print(f"\nDetailed results saved to: {output_path}")

        # Exit with error code if there were failures
        if tester.errors:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
