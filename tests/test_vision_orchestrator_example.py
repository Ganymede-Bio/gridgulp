#!/usr/bin/env python
"""Simple example to test VisionOrchestratorAgent functionality."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


async def test_simple_file():
    """Test vision orchestrator with a simple file."""
    print("=" * 60)
    print("Testing VisionOrchestratorAgent")
    print("=" * 60)

    # Configuration
    config = Config(
        use_vision=False,  # Start without vision
        confidence_threshold=0.7,
        log_level="INFO",
    )

    # Create orchestrator
    orchestrator = VisionOrchestratorAgent(config)

    # Check status
    status = await orchestrator.get_model_status()
    print("\nModel Status:")
    print(f"  Vision available: {status['vision_model_available']}")
    print(f"  Cost optimizer enabled: {status['cost_optimizer_enabled']}")
    print(f"  Session cost: ${status['session_cost']:.4f}")

    # Test with a simple CSV file
    test_file = Path("tests/manual/level0/test_comma.csv")

    # Create a simple test file if it doesn't exist
    if not test_file.exists():
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w") as f:
            f.write("Product,Category,Price,Stock\n")
            f.write("Widget A,Electronics,29.99,150\n")
            f.write("Widget B,Electronics,39.99,75\n")
            f.write("Gadget X,Home,19.99,200\n")
            f.write("Gadget Y,Home,24.99,100\n")
        print(f"Created test file: {test_file}")

    # Read the file
    reader = get_reader(str(test_file))
    file_data = reader.read_sync()

    if not file_data.sheets:
        print("No sheets found in file!")
        return

    sheet_data = file_data.sheets[0]
    print(f"\nProcessing sheet: {sheet_data.name}")
    print(f"  Size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")
    print(f"  Non-empty cells: {len(sheet_data.get_non_empty_cells())}")

    # Run orchestration
    try:
        result = await orchestrator.orchestrate_detection(sheet_data)

        print("\nOrchestration Results:")
        print(f"  Complexity Score: {result.complexity_assessment.complexity_score:.3f}")
        print(f"  Requires Vision: {result.complexity_assessment.requires_vision}")
        print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")
        print(f"  Vision Used: {result.orchestrator_decision.use_vision}")
        print(f"  Cost Estimate: ${result.orchestrator_decision.cost_estimate:.4f}")
        print(f"  Confidence: {result.orchestrator_decision.confidence:.2f}")
        print(f"  Processing Time: {result.processing_metadata['processing_time_seconds']:.3f}s")

        print("\nComplexity Factors:")
        for factor, value in result.complexity_assessment.assessment_factors.items():
            print(f"  {factor}: {value:.3f}")

        print(f"\nTables Detected: {len(result.tables)}")
        for i, table in enumerate(result.tables):
            print(f"\n  Table {i+1}:")
            print(f"    ID: {table.id}")
            print(f"    Range: {table.range.excel_range}")
            print(f"    Size: {table.range.row_count} x {table.range.col_count}")
            print(f"    Confidence: {table.confidence:.3f}")
            print(f"    Method: {table.detection_method}")
            if table.headers:
                print(f"    Headers: {table.headers[:5]}")

        # Save results
        output_dir = Path("tests/outputs/captures")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_dir / f"orchestrator_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        output_data = {
            "file": str(test_file),
            "timestamp": datetime.now().isoformat(),
            "complexity_assessment": {
                "score": result.complexity_assessment.complexity_score,
                "requires_vision": result.complexity_assessment.requires_vision,
                "factors": result.complexity_assessment.assessment_factors,
                "reasoning": result.complexity_assessment.reasoning,
            },
            "orchestrator_decision": {
                "strategy": result.orchestrator_decision.detection_strategy,
                "use_vision": result.orchestrator_decision.use_vision,
                "cost_estimate": result.orchestrator_decision.cost_estimate,
                "confidence": result.orchestrator_decision.confidence,
            },
            "tables": [
                {
                    "id": table.id,
                    "range": table.range.excel_range,
                    "confidence": table.confidence,
                    "method": table.detection_method,
                }
                for table in result.tables
            ],
            "processing_metadata": result.processing_metadata,
            "cost_report": result.cost_report,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        print(f"\nError during orchestration: {e}")
        import traceback

        traceback.print_exc()


async def test_with_vision():
    """Test with vision enabled (requires API key)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping vision test - OPENAI_API_KEY not set")
        return

    print("\n" + "=" * 60)
    print("Testing with Vision Enabled")
    print("=" * 60)

    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_file=0.05,
        confidence_threshold=0.8,
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Test with a more complex file
    test_file = Path("tests/manual/level1/complex_table.xlsx")

    if test_file.exists():
        reader = get_reader(str(test_file))
        file_data = reader.read_sync()

        if file_data.sheets:
            sheet_data = file_data.sheets[0]
            result = await orchestrator.orchestrate_detection(sheet_data)

            print("\nVision Test Results:")
            print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")
            print(f"  Vision Model: {result.orchestrator_decision.vision_model}")
            print(f"  Actual Cost: ${result.cost_report.get('total_cost_usd', 0):.4f}")
            print(f"  Tables Found: {len(result.tables)}")
    else:
        print(f"Complex test file not found: {test_file}")


async def main():
    """Run all tests."""
    await test_simple_file()
    await test_with_vision()

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
