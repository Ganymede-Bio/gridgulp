"""Basic usage example for GridPorter."""

import asyncio
import os
from pathlib import Path

from gridporter import GridPorter


async def detect_tables_example():
    """Example of basic table detection."""
    # Initialize GridPorter with default settings
    porter = GridPorter()

    # Example file path (you'll need to provide your own Excel/CSV file)
    file_path = Path("examples/data/sample_spreadsheet.xlsx")

    if not file_path.exists():
        print(f"Please add a sample file at: {file_path}")
        return

    print(f"Detecting tables in: {file_path}")
    print("-" * 50)

    # Detect tables
    result = await porter.detect_tables(str(file_path))

    # Print summary
    print(f"File type: {result.file_info.type.value}")
    print(f"Total sheets: {len(result.sheets)}")
    print(f"Total tables found: {result.total_tables}")
    print(f"Detection time: {result.detection_time:.2f}s")
    print(f"LLM calls made: {result.llm_calls}")

    # Print details for each sheet
    for sheet in result.sheets:
        print(f"\nSheet: {sheet.name}")
        if sheet.tables:
            for i, table in enumerate(sheet.tables, 1):
                print(f"  Table {i}:")
                print(f"    Range: {table.range.excel_range}")
                print(f"    Shape: {table.shape}")
                print(f"    Suggested name: {table.suggested_name or 'N/A'}")
                print(f"    Confidence: {table.confidence:.2%}")
                print(f"    Method: {table.detection_method}")
                if table.headers:
                    print(f"    Headers: {', '.join(table.headers[:5])}")
                    if len(table.headers) > 5:
                        print(f"             ... and {len(table.headers) - 5} more")
        else:
            print("  No tables detected")


async def cost_efficient_example():
    """Example of cost-efficient configuration."""
    # Configure for minimal LLM usage
    porter = GridPorter(
        suggest_names=False,  # Don't use LLM for naming
        use_local_llm=False,  # No LLM at all
        confidence_threshold=0.8,  # Higher threshold
    )

    file_path = Path("examples/data/sample_spreadsheet.xlsx")
    if file_path.exists():
        result = await porter.detect_tables(str(file_path))
        print(f"Detected {result.total_tables} tables with 0 LLM calls")


async def local_llm_example():
    """Example using Ollama for local LLM."""
    # Configure for local LLM (requires Ollama running)
    porter = GridPorter(
        use_local_llm=True,
        local_model="mistral:7b",  # Or any model you have in Ollama
        ollama_url="http://localhost:11434",
        suggest_names=True,
    )

    file_path = Path("examples/data/sample_spreadsheet.xlsx")
    if file_path.exists():
        try:
            result = await porter.detect_tables(str(file_path))
            print(f"Detected {result.total_tables} tables using local LLM")
            for sheet in result.sheets:
                for table in sheet.tables:
                    if table.suggested_name:
                        print(f"  Suggested name: {table.suggested_name}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running with: ollama serve")


async def batch_processing_example():
    """Example of processing multiple files."""
    porter = GridPorter()

    # Process all Excel files in a directory
    data_dir = Path("examples/data")
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))

    if not excel_files:
        print("No Excel files found in examples/data/")
        return

    print(f"Processing {len(excel_files)} files...")

    for file_path in excel_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            result = await porter.detect_tables(str(file_path))
            summary = result.to_summary()
            print(f"  Tables found: {summary['total_tables']}")
            print(f"  Processing time: {summary['detection_time']}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run examples."""
    print("GridPorter Examples")
    print("=" * 50)

    # Make sure we have an API key if using remote LLM
    if not os.getenv("OPENAI_API_KEY"):
        print("\nNote: Set OPENAI_API_KEY environment variable for LLM features")
        print("      Or use cost_efficient_example() for local-only processing")

    # Run the basic example
    asyncio.run(detect_tables_example())

    # Uncomment to try other examples:
    # asyncio.run(cost_efficient_example())
    # asyncio.run(local_llm_example())
    # asyncio.run(batch_processing_example())


if __name__ == "__main__":
    main()
