#!/usr/bin/env python
"""Test just the orchestrator on large file without full pipeline."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_complex_agent_only():
    """Test just the complex table agent on large file."""
    file_path = Path("examples/spreadsheets/scientific/Sample T, pH data.csv")

    if not file_path.exists():
        print(f"❌ Large test file not found: {file_path}")
        return

    print("🧪 COMPLEX AGENT ONLY TEST")
    print("=" * 50)

    # Read file (we know this works)
    reader = get_reader(str(file_path))
    file_data = reader.read_sync()
    sheet_data = file_data.sheets[0]

    cells = (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
    print(f"Testing with {cells:,} cells")

    # Test complex agent
    config = Config(use_vision=False, confidence_threshold=0.5, log_level="ERROR")
    agent = ComplexTableAgent(config)

    print("Running complex table agent...")
    start_time = time.time()

    try:
        result = await agent.detect_complex_tables(sheet_data)
        detection_time = time.time() - start_time

        cells_per_sec = cells / detection_time

        print("✅ Complex agent completed!")
        print(f"Time: {detection_time:.3f}s")
        print(f"Rate: {cells_per_sec:,.0f} cells/sec")
        print(f"Tables: {len(result.tables)}")

        if result.tables:
            table = result.tables[0]
            print(f"Table: {table.range.excel_range} (method: {table.detection_method})")

        # Check if ultra-fast path was used
        if result.tables and result.tables[0].detection_method == "ultra_fast":
            print("🚀 ULTRA-FAST PATH USED!")
        elif result.tables and "fast" in result.tables[0].detection_method:
            print("⚡ Fast path used")
        else:
            print("🐌 Slow path used")

        return True

    except Exception as e:
        detection_time = time.time() - start_time
        print(f"❌ Failed after {detection_time:.3f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_complex_agent_only())
