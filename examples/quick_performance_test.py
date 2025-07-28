#!/usr/bin/env python
"""Quick performance test to identify bottlenecks."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_small_files():
    """Test performance on small files first."""
    config = Config(use_vision=False, confidence_threshold=0.5, log_level="ERROR")

    orchestrator = VisionOrchestratorAgent(config)

    # Test files from simple to complex
    test_files = [
        "spreadsheets/simple/product_inventory.csv",  # Small CSV
        "spreadsheets/sales/monthly_sales.csv",  # Medium CSV
        "spreadsheets/financial/income_statement.csv",  # Larger CSV
    ]

    print("=" * 60)
    print("QUICK PERFORMANCE TEST")
    print("=" * 60)

    for file_rel_path in test_files:
        file_path = Path("examples") / file_rel_path
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        print(f"\nTesting: {file_path.name}")

        # Read file
        start_time = time.time()
        reader = get_reader(str(file_path))
        file_data = reader.read_sync()
        read_time = time.time() - start_time

        sheet_data = file_data.sheets[0]
        cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)

        print(f"  File size: {file_path.stat().st_size / 1024:.1f} KB")
        print(
            f"  Sheet size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1} = {cells} cells"
        )
        print(f"  Read time: {read_time:.3f}s ({cells/read_time:,.0f} cells/sec)")

        # Run detection with timing
        detection_start = time.time()
        try:
            result = await orchestrator.orchestrate_detection(sheet_data)
            detection_time = time.time() - detection_start

            print(
                f"  Detection time: {detection_time:.3f}s ({cells/detection_time:,.0f} cells/sec)"
            )
            print(f"  Tables found: {len(result.tables)}")
            print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")

            # Performance grade
            rate = cells / detection_time
            if rate >= 100000:
                grade = "A+ (100K+ cells/sec)"
            elif rate >= 50000:
                grade = "A (50K+ cells/sec)"
            elif rate >= 25000:
                grade = "B (25K+ cells/sec)"
            elif rate >= 10000:
                grade = "C (10K+ cells/sec)"
            else:
                grade = "F (<10K cells/sec)"

            print(f"  Grade: {grade}")

        except Exception as e:
            print(f"  Detection error: {e}")


async def profile_detection_steps():
    """Profile individual detection steps to find bottlenecks."""
    print(f"\n{'='*60}")
    print("PROFILING DETECTION STEPS")
    print("=" * 60)

    config = Config(use_vision=False, confidence_threshold=0.5, log_level="ERROR")

    # Use a medium-sized file
    file_path = Path("examples/spreadsheets/financial/income_statement.csv")
    if not file_path.exists():
        print("Test file not found, skipping profiling")
        return

    # Read file
    reader = get_reader(str(file_path))
    file_data = reader.read_sync()
    sheet_data = file_data.sheets[0]

    print(f"Profiling with: {file_path.name}")
    print(f"Sheet size: {(sheet_data.max_row + 1) * (sheet_data.max_column + 1)} cells")

    # Test individual components
    from gridporter.agents.complex_table_agent import ComplexTableAgent
    from gridporter.detectors.island_detector import IslandDetector
    from gridporter.detectors.simple_case_detector import SimpleCaseDetector

    # Simple case detector
    start = time.time()
    simple_detector = SimpleCaseDetector()
    simple_result = simple_detector.detect_simple_table(sheet_data)
    simple_time = time.time() - start
    print(f"  Simple case detection: {simple_time:.3f}s")

    # Island detector
    start = time.time()
    island_detector = IslandDetector()
    islands = island_detector.detect_islands(sheet_data)
    island_time = time.time() - start
    print(f"  Island detection: {island_time:.3f}s (found {len(islands)} islands)")

    # Complex table agent
    start = time.time()
    complex_agent = ComplexTableAgent(config)
    complex_result = await complex_agent.detect_complex_tables(sheet_data)
    complex_time = time.time() - start
    print(f"  Complex detection: {complex_time:.3f}s (found {len(complex_result.tables)} tables)")

    total_time = simple_time + island_time + complex_time
    print(f"  Total component time: {total_time:.3f}s")


async def main():
    """Run quick performance tests."""
    await test_small_files()
    await profile_detection_steps()

    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Current system is far from 100K cells/sec target.")
    print("Key bottlenecks to address:")
    print("1. Complex detection algorithms are too slow")
    print("2. Multiple detection strategies running sequentially")
    print("3. Heavy use of cell-by-cell processing")
    print("4. Lack of early exit conditions")
    print("5. Memory-intensive operations")
    print("\nOptimization priorities:")
    print("1. Implement fast-path detection for simple cases")
    print("2. Use vectorized operations where possible")
    print("3. Add better caching and memoization")
    print("4. Profile and optimize hot paths")
    print("5. Consider parallel processing for multiple sheets")


if __name__ == "__main__":
    asyncio.run(main())
