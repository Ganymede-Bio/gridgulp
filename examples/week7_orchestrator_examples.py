"""Week 7 - Vision Orchestrator Agent Examples

This module demonstrates the capabilities of the VisionOrchestratorAgent,
showing how to use intelligent table detection with cost optimization,
complexity assessment, and multi-model support.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers import create_reader

# Load environment variables from .env file
load_dotenv()


async def basic_orchestrator_usage():
    """Basic usage of VisionOrchestratorAgent with automatic strategy selection."""
    print("=" * 60)
    print("Basic VisionOrchestratorAgent Usage")
    print("=" * 60)

    # Configure with vision and reasonable cost limits
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Set your API key
        max_cost_per_session=1.0,  # $1 per session
        max_cost_per_file=0.1,  # $0.10 per file
        confidence_threshold=0.8,
    )

    # Initialize orchestrator
    orchestrator = VisionOrchestratorAgent(config)

    # Get model status
    status = await orchestrator.get_model_status()
    print(f"Vision Model Available: {status['vision_model_available']}")
    print(f"Vision Model: {status.get('vision_model_name', 'None')}")
    print(f"Session Cost: ${status['session_cost']:.4f}")
    print(f"Budget Per File: ${status['cost_limits']['per_file']:.2f}")
    print()

    # Test with a simple spreadsheet
    test_file = Path("examples/spreadsheets/simple/product_inventory.csv")
    if test_file.exists():
        print(f"Processing: {test_file}")

        # Read the file
        reader = create_reader(str(test_file))
        sheets = list(reader.read_sheets())

        for sheet_data in sheets:
            print(f"\nAnalyzing sheet: {sheet_data.name}")

            # Orchestrate detection
            result = await orchestrator.orchestrate_detection(sheet_data)

            # Display results
            print(f"Detected Tables: {len(result.tables)}")
            print(f"Complexity Score: {result.complexity_assessment.complexity_score:.2f}")
            print(f"Requires Vision: {result.complexity_assessment.requires_vision}")
            print(f"Strategy Used: {result.orchestrator_decision.detection_strategy}")
            print(f"Vision Used: {result.orchestrator_decision.use_vision}")
            print(f"Cost Estimate: ${result.orchestrator_decision.cost_estimate:.4f}")
            print(f"Processing Time: {result.processing_metadata['processing_time_seconds']:.2f}s")
            print(f"Reasoning: {result.complexity_assessment.reasoning}")

            # Show table details
            for i, table in enumerate(result.tables):
                print(f"\nTable {i + 1}:")
                print(f"  Range: {table.range.excel_range}")
                print(f"  Confidence: {table.confidence:.2f}")
                print(f"  Detection Method: {table.detection_method}")
                if table.suggested_name:
                    print(f"  Suggested Name: {table.suggested_name}")
    else:
        print(f"Test file not found: {test_file}")


async def cost_conscious_processing():
    """Demonstrate cost-conscious processing with tight budget constraints."""
    print("\n" + "=" * 60)
    print("Cost-Conscious Processing")
    print("=" * 60)

    # Very conservative cost limits
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_session=0.20,  # Only $0.20 per session
        max_cost_per_file=0.02,  # Only $0.02 per file
        confidence_threshold=0.6,  # Accept lower confidence for cost savings
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Process multiple files to show budget management
    test_files = [
        "examples/spreadsheets/simple/product_inventory.csv",
        "examples/spreadsheets/sales/monthly_sales.csv",
        "examples/spreadsheets/financial/balance_sheet.csv",
    ]

    for file_path in test_files:
        test_file = Path(file_path)
        if not test_file.exists():
            continue

        print(f"\nProcessing: {test_file.name}")

        # Check remaining budget
        status = await orchestrator.get_model_status()
        remaining_budget = status["cost_limits"]["per_session"] - status["session_cost"]
        print(f"Remaining Budget: ${remaining_budget:.4f}")

        if remaining_budget <= 0:
            print("Budget exhausted - would use free methods only")
            continue

        reader = create_reader(str(test_file))
        sheets = list(reader.read_sheets())

        for sheet_data in sheets[:1]:  # Process only first sheet
            result = await orchestrator.orchestrate_detection(sheet_data)

            print(f"  Strategy: {result.orchestrator_decision.detection_strategy}")
            print(f"  Vision Used: {result.orchestrator_decision.use_vision}")
            print(f"  Actual Cost: ${result.cost_report.get('total_cost_usd', 0):.4f}")
            print(f"  Tables Found: {len(result.tables)}")


async def quality_focused_processing():
    """Demonstrate quality-focused processing with higher budget."""
    print("\n" + "=" * 60)
    print("Quality-Focused Processing")
    print("=" * 60)

    # Higher budget for best quality
    config = Config(
        use_vision=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_cost_per_session=5.0,  # Higher session budget
        max_cost_per_file=0.5,  # Higher per-file budget
        confidence_threshold=0.9,  # Require high confidence
        verification_strict_mode=True,  # Strict verification
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Test with complex spreadsheet
    test_file = Path("examples/spreadsheets/complex/multiple ranges.xlsx")
    if test_file.exists():
        print(f"Processing complex file: {test_file}")

        reader = create_reader(str(test_file))
        sheets = list(reader.read_sheets())

        for sheet_data in sheets[:1]:  # Process first sheet
            print(f"\nAnalyzing complex sheet: {sheet_data.name}")

            result = await orchestrator.orchestrate_detection(sheet_data)

            # Detailed analysis
            print("\nComplexity Analysis:")
            print(f"  Overall Score: {result.complexity_assessment.complexity_score:.3f}")
            for factor, value in result.complexity_assessment.assessment_factors.items():
                print(f"  {factor}: {value:.3f}")

            print("\nOrchestrator Decision:")
            print(f"  Primary Strategy: {result.orchestrator_decision.detection_strategy}")
            print(f"  Fallback Strategies: {result.orchestrator_decision.fallback_strategies}")
            print(f"  Cost Estimate: ${result.orchestrator_decision.cost_estimate:.4f}")
            print(f"  Decision Confidence: {result.orchestrator_decision.confidence:.2f}")

            print("\nDetection Results:")
            print(f"  Tables Detected: {len(result.tables)}")
            print(
                f"  Average Confidence: {sum(t.confidence for t in result.tables) / len(result.tables):.2f}"
            )

            # Show individual tables
            for i, table in enumerate(result.tables):
                print(f"\n  Table {i + 1}:")
                print(f"    Range: {table.range.excel_range}")
                print(f"    Confidence: {table.confidence:.3f}")
                print(f"    Method: {table.detection_method}")
                if table.has_headers and table.header_info:
                    print(
                        f"    Headers: {table.header_info.headers[:3] if table.header_info.headers else 'Multi-row'}"
                    )
    else:
        print(f"Complex test file not found: {test_file}")


async def local_llm_processing():
    """Demonstrate processing with local Ollama models."""
    print("\n" + "=" * 60)
    print("Local LLM Processing (Ollama)")
    print("=" * 60)

    # Local LLM configuration
    config = Config(
        use_vision=True,
        use_local_llm=True,
        ollama_url="http://localhost:11434",
        ollama_vision_model="qwen2.5-vl:7b",
        confidence_threshold=0.7,
        # No cost limits for local models
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Check if local model is available
    status = await orchestrator.get_model_status()
    print(f"Vision Model Available: {status['vision_model_available']}")
    print(f"Vision Model: {status.get('vision_model_name', 'None')}")
    print(f"Model Reachable: {status.get('vision_model_reachable', False)}")

    if not status.get("vision_model_reachable", False):
        print(
            "Local vision model not available. Please ensure Ollama is running with qwen2.5-vl:7b"
        )
        return

    # Process with local model
    test_file = Path("examples/spreadsheets/scientific/Sample T, pH data.csv")
    if test_file.exists():
        print(f"\nProcessing with local model: {test_file}")

        reader = create_reader(str(test_file))
        sheets = list(reader.read_sheets())

        for sheet_data in sheets[:1]:
            result = await orchestrator.orchestrate_detection(sheet_data)

            print(f"Strategy: {result.orchestrator_decision.detection_strategy}")
            print(f"Vision Model: {result.orchestrator_decision.vision_model}")
            print(f"Processing Time: {result.processing_metadata['processing_time_seconds']:.2f}s")
            print(f"Tables Found: {len(result.tables)}")
            print("No API costs - local processing!")


async def complexity_analysis_showcase():
    """Showcase complexity analysis across different spreadsheet types."""
    print("\n" + "=" * 60)
    print("Complexity Analysis Showcase")
    print("=" * 60)

    config = Config(
        use_vision=False,  # Just for complexity analysis
        confidence_threshold=0.7,
    )

    orchestrator = VisionOrchestratorAgent(config)

    # Test different file types
    test_files = [
        ("Simple CSV", "examples/spreadsheets/simple/product_inventory.csv"),
        ("Sales Data", "examples/spreadsheets/sales/monthly_sales.csv"),
        ("Financial Statement", "examples/spreadsheets/financial/balance_sheet.csv"),
        ("Scientific Data", "examples/spreadsheets/scientific/Sample T, pH data.csv"),
        ("Complex Multi-table", "examples/spreadsheets/complex/multiple ranges.xlsx"),
    ]

    results = []

    for file_type, file_path in test_files:
        test_file = Path(file_path)
        if not test_file.exists():
            continue

        reader = create_reader(str(test_file))
        sheets = list(reader.read_sheets())

        for sheet_data in sheets[:1]:  # First sheet only
            # Just assess complexity
            complexity = await orchestrator._assess_complexity(sheet_data)

            results.append(
                {
                    "type": file_type,
                    "file": test_file.name,
                    "complexity": complexity.complexity_score,
                    "requires_vision": complexity.requires_vision,
                    "reasoning": complexity.reasoning,
                    "factors": complexity.assessment_factors,
                }
            )

    # Display comparison
    print(f"{'File Type':<20} {'Complexity':<12} {'Vision?':<8} {'Reasoning'}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x["complexity"]):
        print(
            f"{result['type']:<20} {result['complexity']:<12.3f} "
            f"{'Yes' if result['requires_vision'] else 'No':<8} "
            f"{result['reasoning'][:40]}..."
        )

    print("\nDetailed Factor Analysis:")
    for result in results:
        print(f"\n{result['type']} ({result['file']}):")
        for factor, value in result["factors"].items():
            print(f"  {factor}: {value:.3f}")


async def strategy_comparison():
    """Compare different detection strategies on the same file."""
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)

    test_file = Path("examples/spreadsheets/complex/multiple ranges.xlsx")
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return

    reader = create_reader(str(test_file))
    sheets = list(reader.read_sheets())
    sheet_data = sheets[0] if sheets else None

    if not sheet_data:
        print("No sheet data available")
        return

    # Test different strategy configurations
    strategies = [
        ("Conservative (No Vision)", Config(use_vision=False, confidence_threshold=0.6)),
        (
            "Balanced (Hybrid)",
            Config(
                use_vision=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_cost_per_file=0.05,
                confidence_threshold=0.7,
            ),
        ),
        (
            "Premium (Full Vision)",
            Config(
                use_vision=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_cost_per_file=0.20,
                confidence_threshold=0.9,
            ),
        ),
    ]

    print(f"Comparing strategies on: {test_file.name}")
    print("-" * 60)

    for strategy_name, config in strategies:
        orchestrator = VisionOrchestratorAgent(config)

        try:
            result = await orchestrator.orchestrate_detection(sheet_data)

            print(f"\n{strategy_name}:")
            print(f"  Strategy Used: {result.orchestrator_decision.detection_strategy}")
            print(f"  Tables Found: {len(result.tables)}")
            print(
                f"  Avg Confidence: {sum(t.confidence for t in result.tables) / len(result.tables):.2f}"
            )
            print(
                f"  Processing Time: {result.processing_metadata['processing_time_seconds']:.2f}s"
            )
            print(f"  Cost: ${result.cost_report.get('total_cost_usd', 0):.4f}")

        except Exception as e:
            print(f"\n{strategy_name}: Failed - {e}")


async def main():
    """Run all examples."""
    print("GridPorter Week 7 - VisionOrchestratorAgent Examples")
    print("=" * 60)

    try:
        await basic_orchestrator_usage()
        await cost_conscious_processing()
        await quality_focused_processing()
        await local_llm_processing()
        await complexity_analysis_showcase()
        await strategy_comparison()

    except Exception as e:
        print(f"Example failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Ensure examples directory exists
    os.chdir(Path(__file__).parent.parent)

    # Run examples
    asyncio.run(main())
