#!/usr/bin/env python
"""Benchmark detection performance and validate against ground truth."""

import asyncio
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


class PerformanceBenchmark:
    """Benchmark detection performance with detailed metrics."""

    def __init__(self):
        self.results = []
        self.ground_truth = self.load_ground_truth()

    def load_ground_truth(self):
        """Load the structured ground truth."""
        gt_path = Path("examples/structured_ground_truth.json")
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return None

    def calculate_cell_count(self, range_str: str) -> int:
        """Calculate approximate cell count from Excel range."""
        try:
            if ":" not in range_str:
                return 1

            start, end = range_str.upper().split(":")

            # Parse start cell (e.g., A1)
            start_col = 0
            start_row = 0
            i = 0
            while i < len(start) and start[i].isalpha():
                start_col = start_col * 26 + (ord(start[i]) - ord("A") + 1)
                i += 1
            start_row = int(start[i:]) if i < len(start) else 1
            start_col -= 1  # Convert to 0-based
            start_row -= 1  # Convert to 0-based

            # Parse end cell (e.g., W9066)
            end_col = 0
            end_row = 0
            i = 0
            while i < len(end) and end[i].isalpha():
                end_col = end_col * 26 + (ord(end[i]) - ord("A") + 1)
                i += 1
            end_row = int(end[i:]) if i < len(end) else 1
            end_col -= 1  # Convert to 0-based
            end_row -= 1  # Convert to 0-based

            # Calculate cell count
            rows = end_row - start_row + 1
            cols = end_col - start_col + 1
            return rows * cols

        except Exception as e:
            print(f"Error calculating cell count for {range_str}: {e}")
            return 0

    async def benchmark_file(self, file_path: Path, config: Config) -> dict[str, Any]:
        """Benchmark detection for a single file."""
        print(f"\nBenchmarking: {file_path}")

        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        try:
            # Read the file
            reader = get_reader(str(file_path))
            file_data = reader.read_sync()

            read_time = time.time()

            # Initialize orchestrator
            orchestrator = VisionOrchestratorAgent(config)

            # Process each sheet
            sheet_results = []
            total_cells_processed = 0

            for sheet_data in file_data.sheets:
                sheet_start = time.time()

                # Calculate sheet size
                sheet_cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
                total_cells_processed += sheet_cells

                # Run detection
                result = await orchestrator.orchestrate_detection(sheet_data)

                sheet_end = time.time()
                sheet_time = sheet_end - sheet_start

                # Extract detected tables
                detected_tables = []
                for table in result.tables:
                    detected_tables.append(
                        {
                            "range": table.range.excel_range,
                            "confidence": table.confidence,
                            "method": table.detection_method,
                        }
                    )

                sheet_results.append(
                    {
                        "name": sheet_data.name,
                        "cells": sheet_cells,
                        "processing_time": sheet_time,
                        "cells_per_second": sheet_cells / sheet_time if sheet_time > 0 else 0,
                        "detected_tables": detected_tables,
                        "complexity_score": result.complexity_assessment.complexity_score,
                        "strategy": result.orchestrator_decision.detection_strategy,
                    }
                )

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate overall metrics
            total_time = end_time - start_time
            detection_time = end_time - read_time

            # Validate against ground truth
            validation_results = self.validate_against_ground_truth(file_path, sheet_results)

            return {
                "file": str(file_path),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "total_cells": total_cells_processed,
                "read_time": read_time - start_time,
                "detection_time": detection_time,
                "total_time": total_time,
                "cells_per_second": total_cells_processed / total_time if total_time > 0 else 0,
                "peak_memory_mb": peak / (1024 * 1024),
                "sheets": sheet_results,
                "validation": validation_results,
                "performance_grade": self.calculate_performance_grade(
                    total_cells_processed, total_time
                ),
            }

        except Exception as e:
            print(f"Error benchmarking {file_path}: {e}")
            traceback.print_exc()
            return {"file": str(file_path), "error": str(e), "performance_grade": "F"}

    def validate_against_ground_truth(
        self, file_path: Path, sheet_results: list[dict]
    ) -> dict[str, Any]:
        """Validate detection results against ground truth."""
        if not self.ground_truth:
            return {"error": "No ground truth available"}

        # Find relative path for ground truth lookup
        try:
            rel_path = file_path.relative_to(Path("examples"))
            gt_key = str(rel_path)
        except:
            gt_key = file_path.name

        if gt_key not in self.ground_truth["files"]:
            return {"error": f"File {gt_key} not found in ground truth"}

        gt_file = self.ground_truth["files"][gt_key]

        validation = {
            "total_expected": 0,
            "total_detected": 0,
            "correct_detections": 0,
            "missed_detections": [],
            "false_positives": [],
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        for sheet_result in sheet_results:
            sheet_name = sheet_result["name"]
            detected_ranges = {table["range"].upper() for table in sheet_result["detected_tables"]}

            if sheet_name in gt_file:
                expected_ranges = {r.upper() for r in gt_file[sheet_name]["expected_ranges"]}
                validation["total_expected"] += len(expected_ranges)

                # Find correct detections
                correct = expected_ranges.intersection(detected_ranges)
                validation["correct_detections"] += len(correct)

                # Find missed detections
                missed = expected_ranges - detected_ranges
                validation["missed_detections"].extend([(sheet_name, r) for r in missed])

                # Find false positives
                false_pos = detected_ranges - expected_ranges
                validation["false_positives"].extend([(sheet_name, r) for r in false_pos])

            validation["total_detected"] += len(detected_ranges)

        # Calculate metrics
        if validation["total_expected"] > 0:
            validation["recall"] = validation["correct_detections"] / validation["total_expected"]

        if validation["total_detected"] > 0:
            validation["precision"] = (
                validation["correct_detections"] / validation["total_detected"]
            )

        if validation["total_expected"] > 0 or validation["total_detected"] > 0:
            validation["accuracy"] = validation["correct_detections"] / max(
                validation["total_expected"], validation["total_detected"]
            )

        return validation

    def calculate_performance_grade(self, cells: int, time_seconds: float) -> str:
        """Calculate performance grade based on cells/second."""
        if time_seconds <= 0:
            return "N/A"

        cells_per_second = cells / time_seconds

        # Performance grades based on cells/second
        if cells_per_second >= 100000:  # 100K+ cells/sec
            return "A+"
        elif cells_per_second >= 50000:  # 50K+ cells/sec
            return "A"
        elif cells_per_second >= 25000:  # 25K+ cells/sec
            return "B"
        elif cells_per_second >= 10000:  # 10K+ cells/sec
            return "C"
        elif cells_per_second >= 5000:  # 5K+ cells/sec
            return "D"
        else:
            return "F"

    def print_benchmark_results(self, results: list[dict[str, Any]]):
        """Print detailed benchmark results."""
        print("\n" + "=" * 100)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 100)

        total_files = len([r for r in results if "error" not in r])
        total_cells = sum(r.get("total_cells", 0) for r in results if "error" not in r)
        total_time = sum(r.get("total_time", 0) for r in results if "error" not in r)

        print(f"Files processed: {total_files}")
        print(f"Total cells: {total_cells:,}")
        print(f"Total time: {total_time:.3f}s")
        print(
            f"Overall rate: {total_cells/total_time:,.0f} cells/second" if total_time > 0 else "N/A"
        )

        print(
            f"\n{'File':<40} {'Cells':<10} {'Time':<8} {'Rate':<15} {'Grade':<6} {'Accuracy':<10}"
        )
        print("-" * 100)

        for result in results:
            if "error" in result:
                file_name = Path(result["file"]).name
                print(f"{file_name:<40} {'ERROR':<10} {'N/A':<8} {'N/A':<15} {'F':<6} {'N/A':<10}")
                continue

            file_name = Path(result["file"]).name
            cells = result.get("total_cells", 0)
            time_taken = result.get("total_time", 0)
            rate = f"{result.get('cells_per_second', 0):,.0f}/s"
            grade = result.get("performance_grade", "N/A")
            accuracy = f"{result.get('validation', {}).get('accuracy', 0):.2%}"

            print(
                f"{file_name:<40} {cells:<10,} {time_taken:<8.3f} {rate:<15} {grade:<6} {accuracy:<10}"
            )

        # Performance summary by grade
        grades = {}
        for result in results:
            if "error" not in result:
                grade = result.get("performance_grade", "N/A")
                grades[grade] = grades.get(grade, 0) + 1

        print("\nPerformance Grade Distribution:")
        for grade in ["A+", "A", "B", "C", "D", "F"]:
            if grade in grades:
                print(f"  {grade}: {grades[grade]} files")

        # Validation summary
        total_expected = sum(
            r.get("validation", {}).get("total_expected", 0) for r in results if "error" not in r
        )
        total_detected = sum(
            r.get("validation", {}).get("total_detected", 0) for r in results if "error" not in r
        )
        total_correct = sum(
            r.get("validation", {}).get("correct_detections", 0)
            for r in results
            if "error" not in r
        )

        if total_expected > 0 and total_detected > 0:
            overall_precision = total_correct / total_detected
            overall_recall = total_correct / total_expected
            overall_f1 = (
                2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
                if (overall_precision + overall_recall) > 0
                else 0
            )

            print("\nOverall Validation Metrics:")
            print(f"  Expected tables: {total_expected}")
            print(f"  Detected tables: {total_detected}")
            print(f"  Correct detections: {total_correct}")
            print(f"  Precision: {overall_precision:.2%}")
            print(f"  Recall: {overall_recall:.2%}")
            print(f"  F1-Score: {overall_f1:.2%}")


async def main():
    """Run performance benchmark on example files."""
    # Configuration for benchmarking
    config = Config(
        use_vision=False,  # Disable vision for performance testing
        confidence_threshold=0.5,  # Lower threshold to catch more tables
        log_level="ERROR",  # Minimal logging for clean output
    )

    benchmark = PerformanceBenchmark()

    # Find files to benchmark (skip the very large proprietary file for now)
    examples_dir = Path("examples")
    files_to_benchmark = []

    # Add specific files from ground truth
    test_files = [
        "spreadsheets/simple/product_inventory.csv",
        "spreadsheets/simple/basic_table.xls",
        "spreadsheets/sales/monthly_sales.csv",
        "spreadsheets/financial/income_statement.csv",
        "spreadsheets/financial/balance_sheet.csv",
        "spreadsheets/complex/multi_table_report.csv",
        "spreadsheets/complex/multiple ranges.xlsx",
        # Add the large file for performance testing
        "spreadsheets/scientific/Sample T, pH data.csv",
    ]

    for file_rel_path in test_files:
        file_path = examples_dir / file_rel_path
        if file_path.exists():
            files_to_benchmark.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    print(f"Benchmarking {len(files_to_benchmark)} files...")

    # Run benchmarks
    results = []
    for file_path in files_to_benchmark:
        result = await benchmark.benchmark_file(file_path, config)
        results.append(result)
        benchmark.results.append(result)

    # Print results
    benchmark.print_benchmark_results(results)

    # Save detailed results
    output_file = Path("examples/benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
