#!/usr/bin/env python3
"""
Extract DataFrames from detected table regions with header detection and validation.
Processes detection outputs and creates clean DataFrames with quality scores.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp import GridGulp  # noqa: E402
from gridgulp.extractors import DataFrameExtractor  # noqa: E402
from gridgulp.models import (  # noqa: E402
    ExtractedTable,
    FileExtractionResult,
    SheetExtractionResult,
)
from gridgulp.models.table import CellRange  # noqa: E402
from gridgulp.readers import get_reader  # noqa: E402


class DataFrameExtractionPipeline:
    """Pipeline for extracting DataFrames from detection results."""

    def __init__(self, min_quality_score: float = 0.5):
        """Initialize the extraction pipeline.

        Args:
            min_quality_score: Minimum quality score to include in results
        """
        self.min_quality_score = min_quality_score
        self.extractor = DataFrameExtractor()
        self.output_dir = Path("tests/outputs/dataframes")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process_file(self, file_path: Path, detection_result: Any) -> FileExtractionResult:
        """Process a single file and extract DataFrames from detected tables.

        Args:
            file_path: Path to the source file
            detection_result: Detection result from GridGulp

        Returns:
            FileExtractionResult with extracted DataFrames
        """
        start_time = time.time()

        # Initialize result
        result = FileExtractionResult(
            file_path=str(file_path),
            file_type=detection_result.file_info.type.value,
            timestamp=datetime.now().isoformat(),
            total_sheets=len(detection_result.sheets),
            total_tables_detected=detection_result.total_tables,
            detection_time=detection_result.detection_time,
        )

        # Read the file to get sheet data
        try:
            reader = get_reader(str(file_path))
            file_data = reader.read_sync()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return result

        # Process each sheet
        for _, sheet_result in enumerate(detection_result.sheets):
            sheet_extraction = SheetExtractionResult(
                sheet_name=sheet_result.name,
                total_tables_detected=len(sheet_result.tables),
            )

            # Get the corresponding sheet data
            sheet_data = None
            for s in file_data.sheets:
                if s.name == sheet_result.name:
                    sheet_data = s
                    break

            if sheet_data is None:
                print(f"Warning: Could not find sheet data for {sheet_result.name}")
                continue

            # Extract each table
            for table in sheet_result.tables:
                extracted = self._extract_table(sheet_data, table)
                sheet_extraction.extracted_tables.append(extracted)

                if extracted.extraction_status == "success":
                    sheet_extraction.tables_extracted += 1
                else:
                    sheet_extraction.tables_failed += 1

            result.sheets.append(sheet_extraction)

        # Update totals
        result.total_tables_extracted = sum(s.tables_extracted for s in result.sheets)
        result.total_tables_failed = sum(s.tables_failed for s in result.sheets)
        result.extraction_time = time.time() - start_time

        return result

    def _extract_table(self, sheet_data: Any, table_info: Any) -> ExtractedTable:
        """Extract a single table from sheet data."""
        # Parse the table range
        cell_range = self._parse_range(table_info.range.excel_range)

        # Initialize extracted table
        extracted = ExtractedTable(
            range=cell_range,
            detection_confidence=table_info.confidence,
            detection_method=table_info.detection_method,
            extraction_status="pending",
        )

        try:
            # Extract DataFrame
            df, header_info, quality_score = self.extractor.extract_dataframe(
                sheet_data, cell_range
            )

            if df is None:
                extracted.extraction_status = "failed"
                extracted.error_message = "Could not extract valid DataFrame"
            else:
                extracted.extraction_status = "success"
                extracted.quality_score = quality_score

                # Set header information
                if header_info:
                    extracted.has_headers = header_info.has_headers
                    extracted.header_rows = header_info.header_rows
                    extracted.headers = header_info.headers
                    extracted.orientation = header_info.orientation

                # Set table metrics
                extracted.data_rows = len(df)
                extracted.data_columns = len(df.columns)
                extracted.data_density = df.notna().sum().sum() / (len(df) * len(df.columns))

                # Convert DataFrame to dict for JSON serialization
                extracted.dataframe_dict = df.to_dict(orient="list")

        except Exception as e:
            extracted.extraction_status = "failed"
            extracted.error_message = str(e)

        return extracted

    def _parse_range(self, excel_range: str) -> CellRange:
        """Parse Excel-style range (e.g., 'A1:D10') to CellRange."""
        parts = excel_range.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {excel_range}")

        start_cell = parts[0]
        end_cell = parts[1]

        # Parse start cell
        start_col, start_row = self._parse_cell(start_cell)
        end_col, end_row = self._parse_cell(end_cell)

        return CellRange(
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
        )

    def _parse_cell(self, cell_ref: str) -> tuple[int, int]:
        """Parse cell reference (e.g., 'A1') to (col, row) indices."""
        col_str = ""
        row_str = ""

        for char in cell_ref:
            if char.isalpha():
                col_str += char
            else:
                row_str += char

        # Convert column letters to index
        col = 0
        for char in col_str:
            col = col * 26 + (ord(char.upper()) - ord("A") + 1)
        col -= 1  # 0-based

        # Convert row to 0-based index
        row = int(row_str) - 1

        return col, row

    def save_results(self, results: list[FileExtractionResult], format: str = "json") -> Path:
        """Save extraction results to file.

        Args:
            results: List of extraction results
            format: Output format (json or summary)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_file = self.output_dir / f"extracted_dataframes_{timestamp}.json"

            # Convert to JSON-serializable format
            data = []
            for result in results:
                data.append(result.model_dump(mode="json"))

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "summary":
            output_file = self.output_dir / f"extraction_summary_{timestamp}.md"

            with open(output_file, "w") as f:
                f.write("# DataFrame Extraction Summary\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Overall statistics
                total_files = len(results)
                total_detected = sum(r.total_tables_detected for r in results)
                total_extracted = sum(r.total_tables_extracted for r in results)
                total_high_quality = sum(len(r.high_quality_tables) for r in results)

                f.write("## Overall Statistics\n\n")
                f.write(f"- Files processed: {total_files}\n")
                f.write(f"- Tables detected: {total_detected}\n")
                f.write(f"- Tables extracted: {total_extracted}\n")
                f.write(f"- High quality tables (score > 0.7): {total_high_quality}\n")
                f.write(f"- Overall success rate: {total_extracted/total_detected:.1%}\n\n")

                # File details
                f.write("## File Details\n\n")
                for result in results:
                    f.write(f"### {Path(result.file_path).name}\n\n")
                    f.write(f"- Type: {result.file_type}\n")
                    f.write(f"- Sheets: {result.total_sheets}\n")
                    f.write(f"- Tables: {result.total_tables_detected} detected, ")
                    f.write(f"{result.total_tables_extracted} extracted\n")
                    f.write(f"- Success rate: {result.overall_success_rate:.1%}\n\n")

                    # Show details by sheet
                    for sheet in result.sheets:
                        if sheet.total_tables_detected > 0:
                            f.write(f"**Sheet: {sheet.sheet_name}**\n\n")
                            f.write(f"- Tables detected: {sheet.total_tables_detected}\n")
                            f.write(f"- Tables extracted: {sheet.tables_extracted}\n")
                            f.write(
                                f"- Success rate: {sheet.tables_extracted/sheet.total_tables_detected:.1%}\n\n"
                            )

                            # High quality tables for this sheet
                            high_quality_in_sheet = [
                                table
                                for table in sheet.extracted_tables
                                if table.extraction_status == "success"
                                and table.quality_score > 0.7
                            ]

                            if high_quality_in_sheet:
                                f.write("High Quality Tables:\n\n")
                                for table in high_quality_in_sheet:
                                    f.write(f"- Range: {table.range.excel_range}, ")
                                    f.write(f"Quality: {table.quality_score:.2f}, ")
                                    f.write(f"Shape: ({table.data_rows}, {table.data_columns}), ")
                                    f.write(f"Headers: {', '.join(table.headers[:5])}")
                                    if len(table.headers) > 5:
                                        f.write("...")
                                    f.write("\n")
                                f.write("\n")

                    if result.total_sheets == 0 or all(
                        sheet.total_tables_detected == 0 for sheet in result.sheets
                    ):
                        f.write("\n")

        return output_file


async def main():
    """Run DataFrame extraction on detection results."""
    # Look for most recent detection output
    captures_dir = Path("tests/outputs/captures")
    json_files = sorted(captures_dir.glob("detection_outputs_*.json"), reverse=True)
    test_files = sorted(captures_dir.glob("test_detection_*.json"), reverse=True)

    # Combine and sort all files by modification time
    all_files = json_files + test_files
    if all_files:
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not all_files:
        print("No detection output files found. Run capture_outputs.py first.")
        return

    # Use most recent file
    detection_file = all_files[0]
    print(f"Using detection results from: {detection_file}")

    # Load detection results
    with open(detection_file) as f:
        detection_data = json.load(f)

    # Initialize pipeline
    pipeline = DataFrameExtractionPipeline()

    # Process each file
    all_results = []

    print("\n" + "=" * 80)
    print("EXTRACTING DATAFRAMES FROM DETECTED TABLES")
    print("=" * 80)

    for file_detection in detection_data:
        file_path = Path(file_detection["file"])
        print(f"\nProcessing: {file_path}")

        # Run GridGulp to get full detection result (not just the summary)
        porter = GridGulp()
        try:
            detection_result = await porter.detect_tables(str(file_path))

            # Extract DataFrames
            extraction_result = await pipeline.process_file(file_path, detection_result)
            all_results.append(extraction_result)

            # Print summary
            print(
                f"  Extracted: {extraction_result.total_tables_extracted}/{extraction_result.total_tables_detected}"
            )
            print(f"  High quality: {len(extraction_result.high_quality_tables)}")
            print(f"  Time: {extraction_result.extraction_time:.2f}s")

        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    json_output = pipeline.save_results(all_results, "json")
    summary_output = pipeline.save_results(all_results, "summary")

    print("\nResults saved to:")
    print(f"  JSON: {json_output}")
    print(f"  Summary: {summary_output}")

    # Print final statistics
    total_files = len(all_results)
    total_detected = sum(r.total_tables_detected for r in all_results)
    total_extracted = sum(r.total_tables_extracted for r in all_results)
    total_high_quality = sum(len(r.high_quality_tables) for r in all_results)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Files processed: {total_files}")
    print(f"Tables detected: {total_detected}")
    print(f"Tables extracted: {total_extracted}")
    print(f"High quality tables: {total_high_quality}")
    if total_detected > 0:
        print(f"Overall success rate: {total_extracted/total_detected:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
