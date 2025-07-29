#!/usr/bin/env python3
"""
Save extracted DataFrames as individual CSV files.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def save_dataframes_as_csv():
    """Load extracted DataFrames and save each as a CSV file."""
    # Find most recent extraction JSON
    dataframes_dir = Path("tests/outputs/dataframes")
    json_files = sorted(dataframes_dir.glob("extracted_dataframes_*.json"), reverse=True)

    if not json_files:
        print("No extracted dataframes found. Run extract_dataframes.py first.")
        return

    # Use most recent file
    json_file = json_files[0]
    print(f"Using extraction results from: {json_file}")

    # Load extraction results
    with open(json_file) as f:
        extraction_data = json.load(f)

    # Create CSV output directory
    csv_dir = dataframes_dir / "csv_exports"
    csv_dir.mkdir(exist_ok=True)

    # Process each file's results
    total_saved = 0

    for file_result in extraction_data:
        file_name = Path(file_result["file_path"]).stem

        # Process each sheet
        for sheet in file_result["sheets"]:
            sheet_name = sheet["sheet_name"]

            # Process each extracted table
            for idx, table in enumerate(sheet["extracted_tables"]):
                if table["extraction_status"] != "success":
                    continue

                # Create DataFrame from dict
                df_dict = table.get("dataframe_dict")
                if not df_dict:
                    continue

                try:
                    # Convert to DataFrame
                    df = pd.DataFrame(df_dict)

                    # Generate filename from range
                    range_data = table["range"]
                    start_row = range_data["start_row"]
                    start_col = range_data["start_col"]
                    end_row = range_data["end_row"]
                    end_col = range_data["end_col"]

                    # Convert to Excel-style range
                    start_col_letter = chr(ord("A") + start_col)
                    end_col_letter = chr(ord("A") + end_col)
                    range_str = f"{start_col_letter}{start_row+1}_{end_col_letter}{end_row+1}"

                    quality_score = table.get("quality_score", 0)
                    csv_filename = f"{file_name}_{sheet_name}_{range_str}_q{quality_score:.2f}.csv"
                    csv_path = csv_dir / csv_filename

                    # Save to CSV
                    df.to_csv(csv_path, index=False)
                    total_saved += 1

                    print(f"Saved: {csv_filename}")
                    print(f"  Shape: {len(df)} rows × {len(df.columns)} columns")
                    print(f"  Quality: {quality_score:.2f}")

                    # Show preview if small enough
                    if len(df) <= 5:
                        print("  Preview:")
                        print(df.to_string(index=False).replace("\n", "\n    "))
                    else:
                        print("  First 3 rows:")
                        print(df.head(3).to_string(index=False).replace("\n", "\n    "))
                    print()

                except Exception as e:
                    print(f"Error saving {file_name}/{sheet_name} table {idx}: {e}")

    print(f"\nTotal CSV files saved: {total_saved}")
    print(f"Location: {csv_dir}")


if __name__ == "__main__":
    save_dataframes_as_csv()
