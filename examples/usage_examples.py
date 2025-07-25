#!/usr/bin/env python3
"""
GridPorter Usage Examples
Demonstrates various configuration options for both OpenAI and Ollama models.
"""

import asyncio
import os
from pathlib import Path

from gridporter import Config, GridPorter


async def example_auto_detection():
    """Example 1: Auto-detection of LLM provider."""
    print("=== Auto-Detection Example ===")

    # GridPorter automatically chooses Ollama if no OpenAI key is found
    porter = GridPorter()

    result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")

    print(f"LLM Provider: {'Ollama' if porter.config.use_local_llm else 'OpenAI'}")
    print(f"File: {result.file_info.path.name}")
    print(f"Tables detected: {result.total_tables}")
    print(f"Processing time: {result.detection_time:.2f}s")
    print()


async def example_openai_configuration():
    """Example 2: Explicit OpenAI configuration."""
    print("=== OpenAI Configuration Example ===")

    # Only run if OpenAI key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - No OpenAI API key found")
        print()
        return

    config = Config(
        use_local_llm=False,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model="gpt-4o-mini",
        suggest_names=True,
        confidence_threshold=0.8,
    )

    porter = GridPorter(config=config)

    result = await porter.detect_tables("examples/spreadsheets/financial/income_statement.csv")

    print(f"Model: {config.openai_model}")
    print(f"LLM calls made: {result.llm_calls}")
    print(f"LLM tokens used: {result.llm_tokens}")

    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"  Table: {table.suggested_name} ({table.confidence:.1%})")
    print()


async def example_ollama_basic():
    """Example 3: Basic Ollama configuration."""
    print("=== Ollama Basic Configuration Example ===")

    config = Config(
        use_local_llm=True,
        ollama_text_model="deepseek-r1:7b",
        ollama_vision_model="qwen2.5vl:7b",
        suggest_names=True,
    )

    porter = GridPorter(config=config)

    result = await porter.detect_tables("examples/spreadsheets/sales/monthly_sales.csv")

    print(f"Text Model: {config.ollama_text_model}")
    print(f"Vision Model: {config.ollama_vision_model}")
    print(f"Tables detected: {result.total_tables}")
    print(f"Methods used: {', '.join(result.methods_used)}")

    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"  Table: {table.suggested_name}")
            print(f"    Range: {table.range.excel_range}")
    print()


async def example_ollama_advanced():
    """Example 4: Advanced Ollama configuration with different models."""
    print("=== Ollama Advanced Configuration Example ===")

    # Configuration for high-performance analysis
    config_advanced = Config(
        use_local_llm=True,
        ollama_text_model="deepseek-r1:32b",  # Larger model for better reasoning
        ollama_vision_model="qwen2.5vl:32b",  # Enhanced vision capabilities
        ollama_url="http://localhost:11434",
        llm_temperature=0.1,  # More deterministic responses
        max_tokens_per_table=100,  # Longer descriptions
        confidence_threshold=0.9,  # Higher confidence threshold
    )

    porter = GridPorter(config=config_advanced)

    result = await porter.detect_tables("examples/spreadsheets/complex/multi_table_report.csv")

    print(
        f"Advanced Models: {config_advanced.ollama_text_model} + {config_advanced.ollama_vision_model}"
    )
    print(f"Total tables: {result.total_tables}")
    print(f"Detection time: {result.detection_time:.2f}s")

    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        for i, table in enumerate(sheet.tables, 1):
            print(f"  Table {i}: {table.suggested_name}")
            print(f"    Confidence: {table.confidence:.2%}")
            print(f"    Method: {table.detection_method}")
    print()


async def example_ollama_lightweight():
    """Example 5: Lightweight Ollama configuration for resource-constrained environments."""
    print("=== Ollama Lightweight Configuration Example ===")

    config_light = Config(
        use_local_llm=True,
        ollama_text_model="deepseek-r1:1.5b",  # Smaller model for lower resource usage
        ollama_vision_model="qwen2.5vl:7b",
        suggest_names=True,
        llm_temperature=0.3,
        max_tokens_per_table=50,  # Shorter responses
    )

    porter = GridPorter(config=config_light)

    result = await porter.detect_tables("examples/spreadsheets/financial/balance_sheet.csv")

    print(
        f"Lightweight Models: {config_light.ollama_text_model} + {config_light.ollama_vision_model}"
    )
    print("Memory efficient configuration")
    print(f"Tables detected: {result.total_tables}")

    for sheet in result.sheets:
        for table in sheet.tables:
            print(f"  Table: {table.suggested_name}")
    print()


async def example_batch_processing():
    """Example 6: Batch processing with Ollama."""
    print("=== Batch Processing Example ===")

    porter = GridPorter(use_local_llm=True)

    # Process multiple files
    files = [
        "examples/spreadsheets/simple/basic_table.csv",
        "examples/spreadsheets/simple/product_inventory.csv",
        "examples/spreadsheets/sales/monthly_sales.csv",
        "examples/spreadsheets/financial/balance_sheet.csv",
    ]

    # Filter to only existing files
    existing_files = [f for f in files if Path(f).exists()]

    if existing_files:
        results = await porter.batch_detect(existing_files)

        print(f"Processed {len(results)} files:")
        print("-" * 60)

        total_tables = 0
        total_time = 0

        for result in results:
            file_name = result.file_info.path.name
            print(
                f"{file_name:<25} | {result.total_tables:>2} tables | {result.detection_time:>6.2f}s"
            )
            total_tables += result.total_tables
            total_time += result.detection_time

        print("-" * 60)
        print(f"{'TOTAL':<25} | {total_tables:>2} tables | {total_time:>6.2f}s")
    else:
        print("No example files found. Please ensure example files exist.")
    print()


async def example_environment_variables():
    """Example 7: Configuration via environment variables."""
    print("=== Environment Variables Example ===")

    # Set environment variables (normally done in shell or .env file)
    os.environ["OLLAMA_TEXT_MODEL"] = "deepseek-r1:7b"
    os.environ["OLLAMA_VISION_MODEL"] = "qwen2.5vl:7b"
    os.environ["GRIDPORTER_USE_LOCAL_LLM"] = "true"
    os.environ["GRIDPORTER_SUGGEST_NAMES"] = "true"

    # Create config from environment
    config = Config.from_env()
    porter = GridPorter(config=config)

    print("Configuration loaded from environment:")
    print(f"  Use Local LLM: {config.use_local_llm}")
    print(f"  Text Model: {config.ollama_text_model}")
    print(f"  Vision Model: {config.ollama_vision_model}")
    print(f"  Suggest Names: {config.suggest_names}")

    if Path("examples/spreadsheets/simple/basic_table.csv").exists():
        result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")
        print(f"  Tables detected: {result.total_tables}")
    print()


async def example_no_llm_mode():
    """Example 8: Pure table detection without LLM features."""
    print("=== No LLM Mode Example ===")

    config = Config(
        suggest_names=False,  # Disable LLM naming
        use_local_llm=False,
        confidence_threshold=0.6,
    )

    porter = GridPorter(config=config)

    if Path("examples/spreadsheets/simple/basic_table.csv").exists():
        result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")

        print("Table detection without LLM assistance:")
        print(f"  LLM calls made: {result.llm_calls}")
        print(f"  Tables detected: {result.total_tables}")
        print(f"  Detection time: {result.detection_time:.2f}s")

        for sheet in result.sheets:
            for table in sheet.tables:
                print(f"    Range: {table.range.excel_range} (unnamed)")
    else:
        print("Example file not found")
    print()


async def example_error_handling():
    """Example 9: Error handling and fallback strategies."""
    print("=== Error Handling Example ===")

    # Try Ollama first, fallback to no-LLM mode if unavailable
    try:
        config = Config(use_local_llm=True, ollama_text_model="deepseek-r1:7b", timeout_seconds=30)
        porter = GridPorter(config=config)

        # Test with a small file first
        if Path("examples/spreadsheets/simple/basic_table.csv").exists():
            result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")
            print("Ollama connection successful")
            print(f"Tables detected: {result.total_tables}")

    except Exception as e:
        print(f"Ollama error: {e}")
        print("Falling back to no-LLM mode...")

        # Fallback configuration
        config_fallback = Config(suggest_names=False, use_local_llm=False)
        porter = GridPorter(config=config_fallback)

        if Path("examples/spreadsheets/simple/basic_table.csv").exists():
            result = await porter.detect_tables("examples/spreadsheets/simple/basic_table.csv")
            print(f"Fallback successful - Tables detected: {result.total_tables}")
    print()


async def main():
    """Run all examples."""
    print("GridPorter Usage Examples")
    print("=" * 50)
    print()

    examples = [
        example_auto_detection,
        example_openai_configuration,
        example_ollama_basic,
        example_ollama_advanced,
        example_ollama_lightweight,
        example_batch_processing,
        example_environment_variables,
        example_no_llm_mode,
        example_error_handling,
    ]

    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print()

    print("All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
