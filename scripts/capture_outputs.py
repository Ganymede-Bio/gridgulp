#!/usr/bin/env python3
"""
Capture detection outputs for all spreadsheet files in examples and tests directories.
Generates JSON, CSV, and Markdown outputs showing detected tables.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp  # noqa: E402
from gridgulp.config import Config  # noqa: E402


class DetectionOutputCapture:
    """Captures detection outputs in multiple formats."""

    def __init__(self):
        self.outputs = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("tests/outputs/captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_file_detection(self, file_path: str, detection_result: Any):
        """Capture detection results for a file."""
        # Convert detection result to JSON-serializable format
        results = []

        for sheet in detection_result.sheets:
            tables = []
            for table in sheet.tables:
                tables.append(
                    {
                        "id": f"{table.detection_method}_{table.range.start_row}_{table.range.start_col}",
                        "range": table.range.excel_range,
                        "suggested_name": table.suggested_name,
                        "detection_method": table.detection_method,
                        "confidence": table.confidence,
                        "headers": table.headers if table.headers else None,
                    }
                )

            results.append(
                {
                    "sheet_name": sheet.name,
                    "tables": tables,
                    "processing_time": sheet.processing_time,
                    "method_used": tables[0]["detection_method"] if tables else "none",
                }
            )

        self.outputs.append(
            {
                "file": file_path,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }
        )

    def save_outputs(self, format="json"):
        """Save captured outputs in specified format."""
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
        elif format == "csv":
            output_file = self.output_dir / f"detection_outputs_{self.timestamp}.csv"
            with open(output_file, "w") as f:
                f.write(self._format_as_csv())
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

    def _format_as_csv(self) -> str:
        """Format outputs as CSV."""
        lines = ["Input Spreadsheet,Tab,Identified Range,Range Name,Proposed Title"]

        for file_output in self.outputs:
            file_name = Path(file_output["file"]).name

            for sheet_result in file_output["results"]:
                sheet_name = sheet_result["sheet_name"]

                if not sheet_result["tables"]:
                    lines.append(f'"{file_name}","{sheet_name}","No tables found","-","-"')
                else:
                    for table in sheet_result["tables"]:
                        lines.append(
                            f'"{file_name}","{sheet_name}",'
                            f'"{table["range"]}","{table["id"]}",'
                            f'"{table["suggested_name"] or "Not suggested"}"'
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
            file_name = Path(file_output["file"]).name
            print(f"\nFile: {file_name}")

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

        print(f"\nTotal files processed: {total_files}")
        print(f"Total tables detected: {total_tables}")


async def process_file(file_path: Path, porter: GridGulp, capture: DetectionOutputCapture):
    """Process a single file and capture outputs."""
    print(f"\nProcessing: {file_path}")

    try:
        # Detect tables
        result = await porter.detect_tables(str(file_path))

        # Capture results
        capture.capture_file_detection(str(file_path), result)

        print(f"  Found {result.total_tables} tables in {result.detection_time:.3f}s")

    except Exception as e:
        print(f"  Error: {e}")
        # Create empty result for errors
        from gridgulp.models.detection_result import DetectionResult
        from gridgulp.models.file_info import FileInfo, FileType

        error_result = DetectionResult(
            file_info=FileInfo(path=Path(file_path), type=FileType.UNKNOWN, size=0),
            sheets=[],
            detection_time=0,
            total_tables=0,
            methods_used=[],
        )
        capture.capture_file_detection(str(file_path), error_result)


def find_spreadsheet_files(directory: Path) -> list[Path]:
    """Find all spreadsheet files in a directory."""
    extensions = {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".txt"}
    files = []

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Skip output directories and hidden files
            if any(part in file_path.parts for part in ["outputs", "captures", "__pycache__"]):
                continue
            if file_path.name.startswith("."):
                continue
            files.append(file_path)

    return sorted(files)  # Sort for consistent ordering


async def main():
    """Run detection on all spreadsheet files and capture outputs."""
    # Initialize GridGulp with default configuration
    config = Config(
        confidence_threshold=0.7,
        enable_simple_case_detection=True,
        enable_island_detection=True,
    )
    porter = GridGulp(config)

    # Initialize capture
    capture = DetectionOutputCapture()

    # Find all files to process
    all_files = []

    # Add files from examples directory
    examples_dir = Path("examples")
    if examples_dir.exists():
        all_files.extend(find_spreadsheet_files(examples_dir))

    # Add files from tests/manual directory
    tests_dir = Path("tests/manual")
    if tests_dir.exists():
        all_files.extend(find_spreadsheet_files(tests_dir))

    print("\n" + "=" * 80)
    print("SCANNING FOR SPREADSHEET FILES")
    print("=" * 80)

    for file_path in all_files:
        print(f"Found: {file_path}")

    print(f"\nTotal files to process: {len(all_files)}")
    print("=" * 80)

    # Process all files
    for file_path in all_files:
        await process_file(file_path, porter, capture)

    # Print summary
    capture.print_summary()

    # Save outputs in all formats
    json_file = capture.save_outputs("json")
    csv_file = capture.save_outputs("csv")
    md_file = capture.save_outputs("markdown")

    print("\n\nOutputs saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    print(f"  Markdown: {md_file}")


if __name__ == "__main__":
    asyncio.run(main())
