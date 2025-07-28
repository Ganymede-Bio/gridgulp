"""Example demonstrating the new GridPorter architecture with pandas parameters."""

import asyncio
import json
from pathlib import Path

from gridporter.agents import PipelineOrchestrator
from gridporter.config import Config


async def extract_tables_with_pandas_params(file_path: str):
    """
    Extract tables from a spreadsheet and get pandas parameters.

    This example shows how the new architecture provides ready-to-use
    pandas parameters for each detected table.
    """
    # Configure GridPorter
    config = Config(
        use_vision=True,
        check_named_ranges=True,  # Enable fast path for Excel tables
        min_table_size=(2, 2),
        confidence_threshold=0.8,
        enable_cache=True,
        use_llm_for_table_naming=True,
    )

    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(config)

    # Process file with options
    result = await orchestrator.execute(
        file_path,
        options={
            "enable_vision": True,
            "check_named_ranges": True,
            "context_margin": 10,  # Cells around tables for vision
            "max_tables": 10,
        },
    )

    return result


def demonstrate_pandas_usage(result: dict):
    """Show how to use the pandas parameters from extraction results."""

    print("=== GridPorter Extraction Results ===\n")

    for table in result["tables"]:
        print(f"Table: {table['name']}")
        print(f"Range: {table['range']}")
        print(f"Confidence: {table['confidence']:.2%}")
        print(f"Detection Method: {table.get('detection_method', 'unknown')}")

        # Get pandas parameters
        pandas_params = table.get("pandas_config", {})

        print("\nPandas Parameters:")
        print(json.dumps(pandas_params, indent=2))

        # Show how to use the parameters
        if pandas_params.get("read_function") == "read_excel":
            print("\nTo read this table with pandas:")
            print("```python")
            print("df = pd.read_excel(")
            print(f"    '{result['file_info']['path']}',")
            for key, value in pandas_params.items():
                if key not in [
                    "read_function",
                    "table_metadata",
                    "semantic_metadata",
                    "quality_hints",
                    "post_processing",
                ]:
                    print(f"    {key}={repr(value)},")
            print(")")
            print("```")

        # Show field descriptions
        field_descriptions = table.get("field_descriptions", {})
        if field_descriptions:
            print("\nField Descriptions:")
            for field, description in field_descriptions.items():
                print(f"  - {field}: {description}")

        # Show post-processing suggestions
        post_processing = pandas_params.get("post_processing", {})
        if post_processing:
            print("\nPost-processing Suggestions:")
            if post_processing.get("convert_hierarchy"):
                print("  - Convert indented hierarchy to MultiIndex")
            if "handle_totals" in post_processing:
                print(
                    f"  - Handle total rows at indices: {post_processing['handle_totals']['total_rows']}"
                )

        # Show data quality hints
        quality_hints = pandas_params.get("quality_hints", {})
        if quality_hints.get("recommended_cleaning"):
            print("\nData Quality Recommendations:")
            for rec in quality_hints["recommended_cleaning"]:
                print(f"  - {rec['column']}: {rec['suggestion']} ({rec['issue']})")

        print("\n" + "=" * 50 + "\n")


async def main():
    """Main example execution."""
    # Example files (you would use real files)
    example_files = [
        "examples/data/financial_report.xlsx",
        "examples/data/employee_list.xlsx",
        "examples/data/inventory_with_tables.xlsx",
    ]

    for file_path in example_files:
        if Path(file_path).exists():
            print(f"\nProcessing: {file_path}")
            print("-" * 50)

            try:
                # Extract tables
                result = await extract_tables_with_pandas_params(file_path)

                # Show results and pandas usage
                demonstrate_pandas_usage(result)

                # Show performance metrics
                metrics = result.get("performance_metrics", {})
                print("\nPerformance Metrics:")
                print(f"Total Time: {metrics.get('total_time_seconds', 0):.2f}s")
                print(f"Vision Cost: ${metrics.get('total_cost_usd', 0):.4f}")
                print(
                    f"Methods Used: {', '.join(result.get('detection_summary', {}).get('methods_used', []))}"
                )

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")


def show_architecture_benefits():
    """Demonstrate the benefits of the new architecture."""
    print("\n=== New Architecture Benefits ===\n")

    print("1. Clean Agent-Tool Separation:")
    print("   - 5 strategic agents make decisions")
    print("   - 40+ deterministic tools execute operations")
    print("   - Clear boundaries and responsibilities")

    print("\n2. Fast Paths for Pre-defined Tables:")
    print("   - Excel ListObjects detected instantly")
    print("   - Named ranges checked without cell analysis")
    print("   - Vision only used when necessary")

    print("\n3. Enhanced Vision Analysis:")
    print("   - Multi-scale images with explicit compression info")
    print("   - Context cells shown around boundaries")
    print("   - Clear definitions provided to LLM")

    print("\n4. Rich Pandas Integration:")
    print("   - Ready-to-use read_excel/read_csv parameters")
    print("   - Automatic dtype inference")
    print("   - Date parsing configuration")
    print("   - Hierarchy handling instructions")

    print("\n5. Semantic Understanding:")
    print("   - Field descriptions for each column")
    print("   - Table type classification")
    print("   - Relationship detection between tables")
    print("   - Data quality recommendations")

    print("\n6. Test Output Capture:")
    print("   - Automatic capture of all pipeline stages")
    print("   - JSON outputs for analysis")
    print("   - Diff reports against golden outputs")
    print("   - GitHub Actions integration")


if __name__ == "__main__":
    # Show architecture benefits
    show_architecture_benefits()

    # Run example
    print("\n" + "=" * 60)
    print("Running Table Extraction Example")
    print("=" * 60)

    asyncio.run(main())
