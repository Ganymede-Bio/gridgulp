#!/usr/bin/env python
"""Test complex table detection directly."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_complex_detection():
    """Test complex table agent directly."""
    config = Config(use_vision=False, confidence_threshold=0.7)

    agent = ComplexTableAgent(config)

    # Create test file
    test_file = Path("test_complex.csv")
    with open(test_file, "w") as f:
        f.write("Name,Age,City\n")
        f.write("Alice,25,NYC\n")
        f.write("Bob,30,LA\n")
        f.write("Charlie,35,Chicago\n")

    try:
        # Read file
        reader = get_reader(test_file)
        file_data = reader.read_sync()
        sheet_data = file_data.sheets[0]

        print(f"Sheet: {sheet_data.name}")
        print(f"Size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")

        # Test detection
        result = await agent.detect_complex_tables(sheet_data)

        print(f"\nDetection result: {result}")
        print(f"Tables found: {len(result.tables)}")

        for i, table in enumerate(result.tables):
            print(f"\n  Table {i+1}:")
            print(f"    ID: {table.id}")
            print(f"    Range: {table.range.excel_range}")
            print(f"    Method: {table.detection_method}")
            print(f"    Confidence: {table.confidence}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    finally:
        test_file.unlink()


if __name__ == "__main__":
    asyncio.run(test_complex_detection())
