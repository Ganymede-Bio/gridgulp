#!/usr/bin/env python
"""Capture detection outputs showing input files, tabs, ranges, and proposed names."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


class DetectionOutputCapture:
    """Captures detection outputs in a structured format."""

    def __init__(self):
        self.outputs = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("tests/outputs/captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_file_detection(self, file_path: str, results: list[dict[str, Any]]):
        """Capture detection results for a file."""
        self.outputs.append(
            {"file": file_path, "timestamp": datetime.now().isoformat(), "results": results}
        )

    def save_outputs(self, format="json"):
        """Save captured outputs."""
        if format == "json":
            output_file = self.output_dir / f"detection_outputs_{self.timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(self.outputs, f, indent=2)
            return output_file
        elif format == "markdown":
            output_file = self.output_dir / f"detection_outputs_{self.timestamp}.md"
            with open(output_file, "w") as f:
                f.write(self._format_as_markdown())
            return output_file

    def _format_as_markdown(self) -> str:
        """Format outputs as markdown table."""
        lines = ["# GridPorter Detection Outputs\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("| Input Spreadsheet | Tab | Identified Range | Range Name | Proposed Title |")
        lines.append("|-------------------|-----|------------------|------------|----------------|")

        for file_output in self.outputs:
            file_name = Path(file_output["file"]).name

            for sheet_result in file_output["results"]:
                sheet_name = sheet_result["sheet_name"]

                if not sheet_result["tables"]:
                    lines.append(f"| {file_name} | {sheet_name} | No tables found | - | - |")
                else:
                    for table in sheet_result["tables"]:
                        lines.append(
                            f"| {file_name} | {sheet_name} | "
                            f"{table['range']} | {table['id']} | "
                            f"{table['suggested_name'] or 'Not suggested'} |"
                        )

        return "\n".join(lines)

    def print_summary(self):
        """Print a summary of captured outputs."""
        print("\n" + "=" * 80)
        print("DETECTION OUTPUT SUMMARY")
        print("=" * 80)

        for file_output in self.outputs:
            file_name = Path(file_output["file"]).name
            print(f"\nFile: {file_name}")

            for sheet_result in file_output["results"]:
                print(f"  Tab: {sheet_result['sheet_name']}")

                if not sheet_result["tables"]:
                    print("    No tables detected")
                else:
                    for table in sheet_result["tables"]:
                        print(f"    Range: {table['range']}")
                        print(f"      ID: {table['id']}")
                        print(f"      Name: {table['suggested_name'] or 'Not suggested'}")
                        print(f"      Method: {table['detection_method']}")
                        print(f"      Confidence: {table['confidence']:.3f}")


async def process_file(file_path: Path, config: Config, capture: DetectionOutputCapture):
    """Process a single file and capture outputs."""
    print(f"\nProcessing: {file_path}")

    try:
        # Read the file
        reader = get_reader(str(file_path))
        file_data = reader.read_sync()

        # Initialize orchestrator
        orchestrator = VisionOrchestratorAgent(config)

        # Process each sheet
        results = []
        for sheet_data in file_data.sheets:
            print(f"  Sheet: {sheet_data.name}")

            # Run detection
            result = await orchestrator.orchestrate_detection(sheet_data)

            # Extract table information
            tables = []
            for table in result.tables:
                tables.append(
                    {
                        "id": table.id,
                        "range": table.range.excel_range,
                        "suggested_name": table.suggested_name,
                        "detection_method": table.detection_method,
                        "confidence": table.confidence,
                        "headers": table.headers if table.headers else None,
                    }
                )

            results.append(
                {
                    "sheet_name": sheet_data.name,
                    "tables": tables,
                    "complexity_score": result.complexity_assessment.complexity_score,
                    "detection_strategy": result.orchestrator_decision.detection_strategy,
                }
            )

            print(f"    Found {len(tables)} tables")

        # Capture results
        capture.capture_file_detection(str(file_path), results)

    except Exception as e:
        print(f"  Error: {e}")
        capture.capture_file_detection(
            str(file_path), [{"sheet_name": "Error", "error": str(e), "tables": []}]
        )


async def main():
    """Run detection on test files and capture outputs."""
    # Configuration
    config = Config(
        use_vision=False,  # Start without vision for testing
        confidence_threshold=0.6,
        log_level="WARNING",  # Reduce noise
    )

    # Initialize capture
    capture = DetectionOutputCapture()

    # Test files to process
    test_files = [
        # Level 0 - Basic files
        "tests/manual/level0/test_comma.csv",
        "tests/manual/level0/test_basic.xlsx",
        "tests/manual/level0/test_multi_sheet.xlsx",
        "tests/manual/level0/test_formatting.xlsx",
        # Level 1 - Medium complexity
        "tests/manual/level1/simple_table.csv",
        "tests/manual/level1/simple_table.xlsx",
        "tests/manual/level1/complex_table.xlsx",
        # Level 2 - Complex files
        "tests/manual/level2/creative_tables.xlsx",
        "tests/manual/level2/weird_tables.xlsx",
    ]

    # Create sample files if they don't exist
    sample_dir = Path("tests/manual/level0")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple CSV if it doesn't exist
    simple_csv = sample_dir / "test_comma.csv"
    if not simple_csv.exists():
        with open(simple_csv, "w") as f:
            f.write("Product,Category,Price,Stock\n")
            f.write("Widget A,Electronics,29.99,150\n")
            f.write("Widget B,Electronics,39.99,75\n")
            f.write("Gadget X,Home,19.99,200\n")
            f.write("Gadget Y,Home,24.99,100\n")
        print(f"Created sample file: {simple_csv}")

    # Process files
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            await process_file(path, config, capture)
        else:
            print(f"\nSkipping (not found): {file_path}")

    # Print summary
    capture.print_summary()

    # Save outputs
    json_file = capture.save_outputs("json")
    md_file = capture.save_outputs("markdown")

    print("\n\nOutputs saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Markdown: {md_file}")

    # Also create a simple CSV output
    csv_file = capture.output_dir / f"detection_outputs_{capture.timestamp}.csv"
    with open(csv_file, "w") as f:
        f.write("Input Spreadsheet,Tab,Identified Range,Range Name,Proposed Title\n")

        for file_output in capture.outputs:
            file_name = Path(file_output["file"]).name

            for sheet_result in file_output["results"]:
                sheet_name = sheet_result["sheet_name"]

                if not sheet_result["tables"]:
                    f.write(f'"{file_name}","{sheet_name}","No tables found","-","-"\n')
                else:
                    for table in sheet_result["tables"]:
                        f.write(
                            f'"{file_name}","{sheet_name}",'
                            f'"{table["range"]}","{table["id"]}",'
                            f'"{table["suggested_name"] or "Not suggested"}"\n'
                        )

    print(f"  CSV: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())
