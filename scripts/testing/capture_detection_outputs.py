#!/usr/bin/env python
"""Capture detection outputs showing input files, tabs, ranges, and proposed names."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp.detection import TableDetectionAgent
from gridgulp.readers.convenience import get_reader


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
            {
                "file": file_path,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }
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
        lines = ["# GridGulp Detection Outputs\n"]
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


async def process_file(
    file_path: Path, agent: TableDetectionAgent, capture: DetectionOutputCapture
):
    """Process a single file and capture outputs."""
    print(f"\nProcessing: {file_path}")

    try:
        # Read the file
        reader = get_reader(str(file_path))
        file_data = reader.read_sync()

        # Process each sheet
        results = []
        for sheet_data in file_data.sheets:
            print(f"  Sheet: {sheet_data.name}")

            # Run detection
            result = await agent.detect_tables(sheet_data)

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
                    "processing_time": result.processing_metadata.get("processing_time", 0),
                    "method_used": result.processing_metadata.get("method_used", "unknown"),
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
    """Run detection on ALL files in examples directory and capture outputs."""
    # Initialize simplified agent
    agent = TableDetectionAgent(confidence_threshold=0.6)

    # Initialize capture
    capture = DetectionOutputCapture()

    # Get ALL files from examples directory
    examples_dir = Path("examples")

    # Find all files in examples directory
    all_files = []

    print("\\n" + "=" * 80)
    print("SCANNING EXAMPLES DIRECTORY FOR ALL FILES")
    print("=" * 80)

    for file_path in examples_dir.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Skip output directories and certain file types
            if any(part in file_path.parts for part in ["outputs", "captures"]):
                continue
            if file_path.suffix.lower() in [".md", ".py", ".json"]:
                continue

            all_files.append(file_path)
            print(f"Found: {file_path}")

    print(f"\\nTotal files to process: {len(all_files)}")
    print("=" * 80)

    # Process all found files
    for file_path in sorted(all_files):  # Sort for consistent ordering
        await process_file(file_path, agent, capture)

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
