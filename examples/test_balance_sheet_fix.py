#!/usr/bin/env python
"""Test the balance sheet detection fix specifically."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_balance_sheet():
    """Test balance sheet detection."""
    file_path = Path("examples/spreadsheets/financial/balance_sheet.csv")

    config = Config(use_vision=False, confidence_threshold=0.1, log_level="INFO")
    orchestrator = VisionOrchestratorAgent(config)

    # Read file
    reader = get_reader(str(file_path))
    file_data = reader.read_sync()
    sheet_data = file_data.sheets[0]

    print(f"Testing: {file_path.name}")
    print(f"Sheet size: {(sheet_data.max_row + 1) * (sheet_data.max_column + 1)} cells")

    # Run detection
    result = await orchestrator.orchestrate_detection(sheet_data)

    print(f"Strategy: {result.orchestrator_decision.detection_strategy}")
    print(f"Tables detected: {len(result.tables)}")

    for i, table in enumerate(result.tables):
        print(
            f"  Table {i+1}: {table.range.excel_range} (confidence: {table.confidence:.3f}, method: {table.detection_method})"
        )

    # Expected ranges from ground truth
    expected = ["A1:C7", "A9:C15", "A17:B17", "A20:C26", "A28:C33", "A36:C40"]
    detected = [table.range.excel_range for table in result.tables]

    print("\nComparison:")
    print(f"Expected: {expected}")
    print(f"Detected: {detected}")

    # Calculate rough matches (some ranges might be slightly different)
    matches = 0
    for exp_range in expected:
        for det_range in detected:
            if exp_range[:4] in det_range:  # Check if start position matches
                matches += 1
                break

    print(f"Approximate matches: {matches}/{len(expected)}")


if __name__ == "__main__":
    asyncio.run(test_balance_sheet())
