#!/usr/bin/env python
"""Extract ground truth from parsing_answer_guide.xlsx"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridgulp.readers.convenience import get_reader


def extract_ground_truth():
    """Extract ground truth information from the answer guide."""
    answer_guide_path = Path("examples/parsing_answer_guide.xlsx")

    if not answer_guide_path.exists():
        print(f"Error: Answer guide not found at {answer_guide_path}")
        return None

    print(f"Reading answer guide: {answer_guide_path}")

    try:
        # Read the Excel file
        reader = get_reader(str(answer_guide_path))
        file_data = reader.read_sync()

        print(f"Found {len(file_data.sheets)} sheets in answer guide")

        ground_truth = {}

        for sheet_data in file_data.sheets:
            print(f"\nAnalyzing sheet: {sheet_data.name}")
            print(f"Sheet size: {sheet_data.max_row + 1} x {sheet_data.max_column + 1}")

            # Print first few rows to understand structure
            print("First 10 rows:")
            for row in range(min(10, sheet_data.max_row + 1)):
                row_data = []
                for col in range(min(10, sheet_data.max_column + 1)):
                    cell = sheet_data.get_cell(row, col)
                    value = cell.value if cell else None
                    row_data.append(str(value) if value is not None else "")
                print(f"  Row {row}: {row_data}")

            # Look for patterns in the data to understand ground truth format
            # This will need to be refined based on the actual structure
            sheet_ground_truth = parse_sheet_ground_truth(sheet_data)
            if sheet_ground_truth:
                ground_truth[sheet_data.name] = sheet_ground_truth

        return ground_truth

    except Exception as e:
        print(f"Error reading answer guide: {e}")
        return None


def parse_sheet_ground_truth(sheet_data):
    """Parse a sheet to extract ground truth information."""
    # This function will need to be customized based on the actual format
    # of the parsing_answer_guide.xlsx file

    ground_truth = {"files": [], "expected_tables": {}}

    # Look for file names and expected ranges
    # This is a placeholder - will need to be adjusted based on actual format
    for row in range(sheet_data.max_row + 1):
        for col in range(sheet_data.max_column + 1):
            cell = sheet_data.get_cell(row, col)
            if cell and cell.value:
                value = str(cell.value)
                # Look for file extensions to identify file names
                if any(ext in value.lower() for ext in [".csv", ".xlsx", ".xls"]):
                    print(f"  Found potential file: {value} at {row},{col}")
                    ground_truth["files"].append({"name": value, "row": row, "col": col})

                # Look for Excel-style ranges
                if (
                    ":" in value
                    and any(c.isalpha() for c in value)
                    and any(c.isdigit() for c in value)
                ):
                    print(f"  Found potential range: {value} at {row},{col}")

    return ground_truth


def main():
    """Main function to extract and display ground truth."""
    ground_truth = extract_ground_truth()

    if ground_truth:
        # Save to JSON for easy access
        output_file = Path("examples/ground_truth.json")
        with open(output_file, "w") as f:
            json.dump(ground_truth, f, indent=2)

        print(f"\nGround truth saved to: {output_file}")

        # Display summary
        print("\n" + "=" * 80)
        print("GROUND TRUTH SUMMARY")
        print("=" * 80)

        for sheet_name, data in ground_truth.items():
            print(f"\nSheet: {sheet_name}")
            print(f"  Files found: {len(data.get('files', []))}")
            for file_info in data.get("files", []):
                print(f"    - {file_info['name']}")
    else:
        print("Failed to extract ground truth")


if __name__ == "__main__":
    main()
