#!/usr/bin/env python
"""Test performance on the large scientific dataset (200K+ cells)."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_large_file_performance():
    """Test performance on the large scientific data file."""
    file_path = Path("examples/spreadsheets/scientific/Sample T, pH data.csv")

    if not file_path.exists():
        print(f"❌ Large test file not found: {file_path}")
        return

    print("🧪 LARGE FILE PERFORMANCE TEST")
    print("=" * 60)
    print(f"File: {file_path.name}")

    # Configuration optimized for performance
    config = Config(
        use_vision=False,
        confidence_threshold=0.5,
        log_level="ERROR",  # Minimize logging overhead
    )

    # Test file reading performance
    read_start = time.time()
    reader = get_reader(str(file_path))
    file_data = reader.read_sync()
    sheet_data = file_data.sheets[0]
    read_time = time.time() - read_start

    cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Sheet dimensions: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")
    print(f"Total cells: {cells:,}")
    print(f"Read time: {read_time:.3f}s ({cells/read_time:,.0f} cells/sec)")

    # Test detection performance
    orchestrator = VisionOrchestratorAgent(config)

    print("\nRunning detection...")
    detection_start = time.time()

    try:
        result = await orchestrator.orchestrate_detection(sheet_data)
        detection_time = time.time() - detection_start

        cells_per_sec = cells / detection_time

        print("✅ Detection completed!")
        print(f"Detection time: {detection_time:.3f}s")
        print(f"Performance: {cells_per_sec:,.0f} cells/sec")
        print(f"Strategy: {result.orchestrator_decision.detection_strategy}")
        print(f"Tables detected: {len(result.tables)}")

        if result.tables:
            for i, table in enumerate(result.tables):
                print(
                    f"  Table {i+1}: {table.range.excel_range} (method: {table.detection_method}, confidence: {table.confidence:.3f})"
                )

        # Performance grading
        if cells_per_sec >= 100000:
            grade = "🏆 A+ (100K+ cells/sec) - TARGET ACHIEVED!"
        elif cells_per_sec >= 50000:
            grade = "🥇 A (50K+ cells/sec) - Excellent!"
        elif cells_per_sec >= 25000:
            grade = "🥈 B (25K+ cells/sec) - Good"
        elif cells_per_sec >= 10000:
            grade = "🥉 C (10K+ cells/sec) - Acceptable"
        else:
            grade = "❌ F (<10K cells/sec) - Needs optimization"

        print(f"\nPerformance Grade: {grade}")

        # Ground truth validation
        expected_range = "A1:W9066"  # From ground truth
        if result.tables and result.tables[0].range.excel_range.upper() == expected_range.upper():
            print(f"✅ Ground truth match: {expected_range}")
        else:
            detected_range = result.tables[0].range.excel_range if result.tables else "None"
            print(f"⚠️  Range mismatch - Expected: {expected_range}, Got: {detected_range}")

        return {
            "cells": cells,
            "detection_time": detection_time,
            "cells_per_sec": cells_per_sec,
            "grade": grade,
            "success": True,
        }

    except Exception as e:
        detection_time = time.time() - detection_start
        print(f"❌ Detection failed after {detection_time:.3f}s")
        print(f"Error: {e}")
        return {
            "cells": cells,
            "detection_time": detection_time,
            "cells_per_sec": 0,
            "grade": "❌ FAILED",
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    asyncio.run(test_large_file_performance())
