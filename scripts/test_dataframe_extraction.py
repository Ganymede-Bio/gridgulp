#!/usr/bin/env python3
"""
Test script to run table detection and DataFrame extraction on test files.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp


async def run_detection_on_test_files():
    """Run detection on a selection of test files."""
    # Select test files
    test_files = [
        Path("tests/manual/level2/weird_tables.xlsx"),
        Path("tests/manual/sample.xlsx"),
        Path("tests/manual/level0/test_multi_sheet.xlsx"),
        Path("tests/manual/level0/test_types.csv"),
        Path("tests/manual/level0/test_tab.tsv"),
        Path("examples/parsing_answer_guide.xlsx"),
    ]

    # Filter existing files
    existing_files = [f for f in test_files if f.exists()]

    if not existing_files:
        print("No test files found!")
        return

    print(f"Running detection on {len(existing_files)} test files...")

    # Initialize GridGulp
    gp = GridGulp()

    # Run detection
    all_results = []
    for file_path in existing_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            result = await gp.detect_tables(file_path)

            # Prepare detection result for extraction
            detection_data = {
                "file": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "results": [],
            }

            for sheet in result.sheets:
                sheet_data = {
                    "sheet_name": sheet.name,
                    "tables": [],
                    "processing_time": sheet.processing_time,
                    "method_used": (
                        sheet.metadata.get("method_used", "unknown")
                        if hasattr(sheet, "metadata") and sheet.metadata
                        else "unknown"
                    ),
                }

                for table in sheet.tables:
                    # Convert range to Excel format
                    excel_range = (
                        table.range.excel_range
                        if hasattr(table.range, "excel_range")
                        else str(table.range)
                    )

                    table_data = {
                        "id": table.id,
                        "range": excel_range,
                        "suggested_name": table.suggested_name,
                        "detection_method": table.detection_method,
                        "confidence": table.confidence,
                        "headers": table.headers,
                    }
                    sheet_data["tables"].append(table_data)

                detection_data["results"].append(sheet_data)

            all_results.append(detection_data)

            print(
                f"  - Detected {sum(len(sheet.tables) for sheet in result.sheets)} tables across {len(result.sheets)} sheets"
            )

        except Exception as e:
            print(f"  - Error: {e}")

    # Save detection results
    output_dir = Path("tests/outputs/captures")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_detection_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetection results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Run detection
    detection_file = asyncio.run(run_detection_on_test_files())

    if detection_file:
        print("\nNow run the extraction script:")
        print(f"python scripts/extract_dataframes.py --input {detection_file}")
