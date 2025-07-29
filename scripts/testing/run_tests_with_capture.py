#!/usr/bin/env python
"""Run tests with output capture for quality assessment."""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp
from gridgulp.agents import VisionOrchestratorAgent
from gridgulp.config import Config
from gridgulp.readers import create_reader


class TestRunner:
    """Runs tests and captures outputs for assessment."""

    def __init__(self):
        self.results = []
        self.output_dir = Path(__file__).parent / "outputs" / "captures"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_unit_tests(self):
        """Run unit tests and capture results."""
        print("\n" + "=" * 60)
        print("Running Unit Tests")
        print("=" * 60)

        # Run pytest with JSON output
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/agents/test_vision_orchestrator_agent.py",
                "--json-report",
                "--json-report-file=tests/outputs/reports/unit_tests.json",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        self.results.append(
            {
                "test_type": "unit_tests",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

        print(f"Unit tests completed with exit code: {result.returncode}")

        # Parse results
        if Path("tests/outputs/reports/unit_tests.json").exists():
            with open("tests/outputs/reports/unit_tests.json") as f:
                test_data = json.load(f)
                print(f"Tests run: {test_data['summary']['total']}")
                print(f"Passed: {test_data['summary'].get('passed', 0)}")
                print(f"Failed: {test_data['summary'].get('failed', 0)}")
                print(f"Skipped: {test_data['summary'].get('skipped', 0)}")

    async def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("Running Integration Tests")
        print("=" * 60)

        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/integration/",
                "--json-report",
                "--json-report-file=tests/outputs/reports/integration_tests.json",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        self.results.append(
            {
                "test_type": "integration_tests",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

        print(f"Integration tests completed with exit code: {result.returncode}")

    async def run_manual_test_scenarios(self):
        """Run manual test scenarios and capture outputs."""
        print("\n" + "=" * 60)
        print("Running Manual Test Scenarios")
        print("=" * 60)

        test_files = [
            ("Simple CSV", "tests/manual/level0/test_comma.csv"),
            ("Simple Excel", "tests/manual/level0/test_basic.xlsx"),
            ("Multi-sheet Excel", "tests/manual/level0/test_multi_sheet.xlsx"),
            ("Complex Table", "tests/manual/level1/complex_table.xlsx"),
            ("Large File", "tests/manual/level1/large_table.csv"),
        ]

        # Test with different configurations
        configs = [
            ("No Vision", Config(use_vision=False, confidence_threshold=0.7)),
            (
                "Vision Basic",
                Config(
                    use_vision=True,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    max_cost_per_file=0.05,
                    confidence_threshold=0.8,
                ),
            ),
            (
                "Cost Conscious",
                Config(
                    use_vision=True,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    max_cost_per_file=0.01,
                    max_cost_per_session=0.05,
                    confidence_threshold=0.6,
                ),
            ),
        ]

        for config_name, config in configs:
            print(f"\n--- Testing with {config_name} configuration ---")

            for file_desc, file_path in test_files:
                if not Path(file_path).exists():
                    print(f"  ⚠️  {file_desc}: File not found - {file_path}")
                    continue

                try:
                    print(f"  Testing {file_desc}...")

                    # Use VisionOrchestratorAgent directly
                    orchestrator = VisionOrchestratorAgent(config)
                    reader = create_reader(file_path)
                    sheets = list(reader.read_sheets())

                    if sheets:
                        sheet_data = sheets[0]
                        result = await orchestrator.orchestrate_detection(sheet_data)

                        # Capture output
                        output = {
                            "file": file_path,
                            "config": config_name,
                            "timestamp": datetime.now().isoformat(),
                            "complexity_score": result.complexity_assessment.complexity_score,
                            "requires_vision": result.complexity_assessment.requires_vision,
                            "strategy": result.orchestrator_decision.detection_strategy,
                            "vision_used": result.orchestrator_decision.use_vision,
                            "cost_estimate": result.orchestrator_decision.cost_estimate,
                            "tables_found": len(result.tables),
                            "processing_time": result.processing_metadata.get(
                                "processing_time_seconds", 0
                            ),
                            "tables": [
                                {
                                    "range": table.range.excel_range,
                                    "confidence": table.confidence,
                                    "method": table.detection_method,
                                }
                                for table in result.tables
                            ],
                        }

                        # Save output
                        output_file = (
                            self.output_dir
                            / f"{self.timestamp}_{Path(file_path).stem}_{config_name}.json"
                        )
                        with open(output_file, "w") as f:
                            json.dump(output, f, indent=2)

                        print(
                            f"    ✅ Tables: {len(result.tables)}, "
                            f"Strategy: {result.orchestrator_decision.detection_strategy}, "
                            f"Time: {result.processing_metadata.get('processing_time_seconds', 0):.2f}s"
                        )

                except Exception as e:
                    print(f"    ❌ Error: {e}")

                    # Capture error
                    error_output = {
                        "file": file_path,
                        "config": config_name,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

                    error_file = (
                        self.output_dir
                        / f"{self.timestamp}_{Path(file_path).stem}_{config_name}_error.json"
                    )
                    with open(error_file, "w") as f:
                        json.dump(error_output, f, indent=2)

    async def run_performance_benchmarks(self):
        """Run performance benchmarks."""
        print("\n" + "=" * 60)
        print("Running Performance Benchmarks")
        print("=" * 60)

        import time

        config = Config(use_vision=False)  # Test without vision for speed

        test_files = [
            ("Small", "tests/manual/level0/test_basic.xlsx"),
            ("Medium", "tests/manual/level1/complex_table.xlsx"),
            ("Large", "tests/manual/level1/large_table.csv"),
        ]

        perf_results = []

        for size, file_path in test_files:
            if not Path(file_path).exists():
                continue

            print(f"\n  Benchmarking {size} file...")

            times = []
            for i in range(3):  # Run 3 times
                start = time.time()

                try:
                    gp = GridGulp(config)
                    result = await gp.extract_from_file(file_path)
                    elapsed = time.time() - start
                    times.append(elapsed)

                    print(f"    Run {i+1}: {elapsed:.3f}s")

                except Exception as e:
                    print(f"    Run {i+1}: Failed - {e}")

            if times:
                avg_time = sum(times) / len(times)
                perf_results.append(
                    {
                        "file": file_path,
                        "size": size,
                        "runs": len(times),
                        "times": times,
                        "average": avg_time,
                        "min": min(times),
                        "max": max(times),
                    }
                )

                print(
                    f"  Average: {avg_time:.3f}s (min: {min(times):.3f}s, max: {max(times):.3f}s)"
                )

        # Save performance results
        perf_file = self.output_dir / f"{self.timestamp}_performance_benchmarks.json"
        with open(perf_file, "w") as f:
            json.dump(perf_results, f, indent=2)

    def generate_summary_report(self):
        """Generate summary report of all tests."""
        print("\n" + "=" * 60)
        print("Test Summary Report")
        print("=" * 60)

        summary = {
            "timestamp": self.timestamp,
            "test_runs": self.results,
            "output_files": [str(f) for f in self.output_dir.glob(f"{self.timestamp}_*.json")],
        }

        # Save summary
        summary_file = self.output_dir / f"{self.timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")
        print(f"Total output files: {len(summary['output_files'])}")

        # Create a human-readable report
        report_file = self.output_dir / f"{self.timestamp}_report.txt"
        with open(report_file, "w") as f:
            f.write("GridGulp Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for result in self.results:
                f.write(f"{result['test_type'].upper()}\n")
                f.write(f"Exit Code: {result['exit_code']}\n")
                f.write("-" * 40 + "\n")
                if result["exit_code"] != 0:
                    f.write("STDERR:\n")
                    f.write(
                        result["stderr"][:1000] + "...\n"
                        if len(result["stderr"]) > 1000
                        else result["stderr"]
                    )
                f.write("\n")

        print(f"Report saved to: {report_file}")


async def main():
    """Run all tests with output capture."""
    runner = TestRunner()

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set - vision tests will be skipped")

    # Run test suites
    await runner.run_unit_tests()
    await runner.run_integration_tests()
    await runner.run_manual_test_scenarios()
    await runner.run_performance_benchmarks()

    # Generate report
    runner.generate_summary_report()

    print("\n✅ Test run completed!")


if __name__ == "__main__":
    asyncio.run(main())
