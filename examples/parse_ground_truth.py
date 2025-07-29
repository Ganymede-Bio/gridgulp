#!/usr/bin/env python
"""Parse ground truth from parsing_answer_guide.xlsx into a structured format."""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp.readers.convenience import get_reader


def parse_ground_truth():
    """Parse the ground truth answer guide into a structured format."""
    answer_guide_path = Path("examples/parsing_answer_guide.xlsx")

    if not answer_guide_path.exists():
        print(f"Error: Answer guide not found at {answer_guide_path}")
        return None

    print(f"Parsing ground truth from: {answer_guide_path}")

    try:
        # Read the Excel file
        reader = get_reader(str(answer_guide_path))
        file_data = reader.read_sync()

        sheet_data = file_data.sheets[0]  # Assuming first sheet
        print(f"Sheet: {sheet_data.name}")
        print(f"Size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")

        # Parse the structured data
        ground_truth = defaultdict(lambda: defaultdict(list))

        # Skip header row (row 0)
        for row in range(1, sheet_data.max_row + 1):
            # Get values from the three columns
            file_cell = sheet_data.get_cell(row, 0)  # Input Spreadsheet
            tab_cell = sheet_data.get_cell(row, 1)  # Tab
            range_cell = sheet_data.get_cell(row, 2)  # Identified Range

            if file_cell and file_cell.value and range_cell and range_cell.value:
                file_path = str(file_cell.value).strip()
                tab_name = str(tab_cell.value).strip() if tab_cell and tab_cell.value else "Sheet1"
                range_str = str(range_cell.value).strip()

                ground_truth[file_path][tab_name].append(range_str)

        return dict(ground_truth)

    except Exception as e:
        print(f"Error parsing ground truth: {e}")
        import traceback

        traceback.print_exc()
        return None


def display_ground_truth(ground_truth):
    """Display the parsed ground truth in a readable format."""
    print("\n" + "=" * 80)
    print("GROUND TRUTH STRUCTURE")
    print("=" * 80)

    total_files = len(ground_truth)
    total_tables = sum(len(ranges) for tabs in ground_truth.values() for ranges in tabs.values())

    for file_path, tabs in ground_truth.items():
        print(f"\nFile: {file_path}")
        for tab_name, ranges in tabs.items():
            print(f"  Tab: {tab_name}")
            print(f"  Expected tables: {len(ranges)}")
            for i, range_str in enumerate(ranges):
                print(f"    {i+1}. {range_str}")

    print(f"\n{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Total expected tables: {total_tables}")
    print("=" * 80)


def save_structured_ground_truth(ground_truth):
    """Save the structured ground truth to JSON."""
    output_file = Path("examples/structured_ground_truth.json")

    # Convert to a more structured format
    structured = {
        "files": {},
        "summary": {
            "total_files": len(ground_truth),
            "total_tables": sum(
                len(ranges) for tabs in ground_truth.values() for ranges in tabs.values()
            ),
        },
    }

    for file_path, tabs in ground_truth.items():
        structured["files"][file_path] = {}
        for tab_name, ranges in tabs.items():
            structured["files"][file_path][tab_name] = {
                "expected_ranges": ranges,
                "table_count": len(ranges),
            }

    with open(output_file, "w") as f:
        json.dump(structured, f, indent=2)

    print(f"\nStructured ground truth saved to: {output_file}")
    return output_file


def main():
    """Main function to parse and display ground truth."""
    ground_truth = parse_ground_truth()

    if ground_truth:
        display_ground_truth(ground_truth)
        save_structured_ground_truth(ground_truth)

        # Show key insights
        print("\nKEY INSIGHTS:")

        # Files with multiple tables
        multi_table_files = {}
        for file_path, tabs in ground_truth.items():
            total_tables = sum(len(ranges) for ranges in tabs.values())
            if total_tables > 1:
                multi_table_files[file_path] = total_tables

        print(f"Files with multiple tables: {len(multi_table_files)}")
        for file_path, count in sorted(multi_table_files.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {Path(file_path).name}: {count} tables")

        # Large files that need performance optimization
        large_ranges = []
        for file_path, tabs in ground_truth.items():
            for tab_name, ranges in tabs.items():
                for range_str in ranges:
                    if ":" in range_str:
                        try:
                            # Estimate cell count from range
                            start, end = range_str.upper().split(":")
                            # Simple estimation - this could be more accurate
                            if "W9066" in range_str:  # The scientific data file
                                large_ranges.append((file_path, range_str, "~200K cells"))
                        except:
                            pass

        if large_ranges:
            print("\nLarge files requiring performance optimization:")
            for file_path, range_str, estimate in large_ranges:
                print(f"  - {Path(file_path).name}: {range_str} ({estimate})")
    else:
        print("Failed to parse ground truth")


if __name__ == "__main__":
    main()
