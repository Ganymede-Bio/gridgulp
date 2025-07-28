#!/usr/bin/env python
"""Capture detection outputs for all Excel and CSV files in the examples directory."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridporter.agents import VisionOrchestratorAgent
from gridporter.config import Config
from gridporter.readers.convenience import get_reader


class DetectionOutputCapture:
    """Captures detection outputs in a structured format."""

    def __init__(self):
        self.outputs = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("examples/outputs/captures")
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
        lines = ["# GridPorter Detection Outputs - Examples Directory\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("| Input Spreadsheet | Tab | Identified Range | Range Name | Proposed Title |")
        lines.append("|-------------------|-----|------------------|------------|----------------|")

        for file_output in self.outputs:
            # Show relative path from examples directory
            file_path = Path(file_output["file"])
            try:
                rel_path = file_path.relative_to(Path("examples"))
                file_display = str(rel_path)
            except:
                file_display = file_path.name

            for sheet_result in file_output["results"]:
                sheet_name = sheet_result["sheet_name"]

                if not sheet_result["tables"]:
                    lines.append(f"| {file_display} | {sheet_name} | No tables found | - | - |")
                else:
                    for table in sheet_result["tables"]:
                        lines.append(
                            f"| {file_display} | {sheet_name} | "
                            f"{table['range']} | {table['id']} | "
                            f"{table['suggested_name'] or 'Not suggested'} |"
                        )

        return "\n".join(lines)

    def print_summary(self):
        """Print a summary of captured outputs."""
        print("\n" + "=" * 80)
        print("DETECTION OUTPUT SUMMARY")
        print("=" * 80)

        total_files = len(self.outputs)
        total_tables = 0

        for file_output in self.outputs:
            file_path = Path(file_output["file"])
            try:
                rel_path = file_path.relative_to(Path("examples"))
                file_display = str(rel_path)
            except:
                file_display = file_path.name

            print(f"\nFile: {file_display}")

            for sheet_result in file_output["results"]:
                print(f"  Tab: {sheet_result['sheet_name']}")

                if not sheet_result["tables"]:
                    print("    No tables detected")
                else:
                    for table in sheet_result["tables"]:
                        total_tables += 1
                        print(f"    Range: {table['range']}")
                        print(f"      ID: {table['id']}")
                        print(f"      Name: {table['suggested_name'] or 'Not suggested'}")
                        print(f"      Method: {table['detection_method']}")
                        print(f"      Confidence: {table['confidence']:.3f}")

        print(f"\n{'='*80}")
        print(f"Total files processed: {total_files}")
        print(f"Total tables detected: {total_tables}")
        print("=" * 80)


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


def find_spreadsheet_files(base_dir: Path) -> list[Path]:
    """Find all CSV, XLS, and XLSX files in the directory tree."""
    spreadsheet_files = []
    extensions = {".csv", ".xls", ".xlsx"}

    for ext in extensions:
        spreadsheet_files.extend(base_dir.rglob(f"*{ext}"))

    # Sort for consistent output
    spreadsheet_files.sort()

    return spreadsheet_files


async def main():
    """Run detection on all spreadsheet files in the examples directory."""
    # Configuration
    config = Config(
        use_vision=False,  # Start without vision for testing
        confidence_threshold=0.6,
        log_level="WARNING",  # Reduce noise
    )

    # Initialize capture
    capture = DetectionOutputCapture()

    # Find all spreadsheet files
    examples_dir = Path("examples")
    spreadsheet_files = find_spreadsheet_files(examples_dir)

    print(f"Found {len(spreadsheet_files)} spreadsheet files in examples directory")
    print("Files to process:")
    for file in spreadsheet_files:
        rel_path = file.relative_to(examples_dir)
        print(f"  - {rel_path}")

    # Process each file
    for file_path in spreadsheet_files:
        await process_file(file_path, config, capture)

    # Print summary
    capture.print_summary()

    # Save outputs in all formats
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
            file_path = Path(file_output["file"])
            try:
                rel_path = file_path.relative_to(Path("examples"))
                file_display = str(rel_path)
            except:
                file_display = file_path.name

            for sheet_result in file_output["results"]:
                sheet_name = sheet_result["sheet_name"]

                if not sheet_result["tables"]:
                    f.write(f'"{file_display}","{sheet_name}","No tables found","-","-"\n')
                else:
                    for table in sheet_result["tables"]:
                        f.write(
                            f'"{file_display}","{sheet_name}",'
                            f'"{table["range"]}","{table["id"]}",'
                            f'"{table["suggested_name"] or "Not suggested"}"\n'
                        )

    print(f"  CSV: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())
