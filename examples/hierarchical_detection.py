"""Example of hierarchical data detection in GridPorter."""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridporter.models.sheet_data import CellData, SheetData  # noqa: E402
from gridporter.vision.hierarchical_detector import HierarchicalPatternDetector  # noqa: E402
from gridporter.vision.pattern_detector import SparsePatternDetector  # noqa: E402


def create_financial_statement():
    """Create a sample financial statement with hierarchy."""
    sheet = SheetData(name="Income Statement 2024")

    # Financial data with indentation
    financial_data = [
        # (Account Name, Indentation Level, Is Bold, Value)
        ("INCOME STATEMENT", 0, True, None),
        ("For the Year Ended December 31, 2024", 0, False, None),
        ("", 0, False, None),
        ("Revenue:", 0, True, None),
        ("  Product Revenue", 1, False, 850000),
        ("    Hardware Sales", 2, False, 500000),
        ("    Software Sales", 2, False, 350000),
        ("  Service Revenue", 1, False, 450000),
        ("    Consulting", 2, False, 300000),
        ("    Support", 2, False, 150000),
        ("Total Revenue", 0, True, 1300000),
        ("", 0, False, None),
        ("Operating Expenses:", 0, True, None),
        ("  Cost of Goods Sold", 1, False, 400000),
        ("    Hardware Costs", 2, False, 250000),
        ("    Software Licenses", 2, False, 150000),
        ("  Personnel Expenses", 1, False, 500000),
        ("    Salaries", 2, False, 400000),
        ("    Benefits", 2, False, 100000),
        ("  Administrative Expenses", 1, False, 150000),
        ("    Office Rent", 2, False, 60000),
        ("    Utilities", 2, False, 30000),
        ("    Other Admin", 2, False, 60000),
        ("Total Operating Expenses", 0, True, 1050000),
        ("", 0, False, None),
        ("Operating Income", 0, True, 250000),
        ("", 0, False, None),
        ("Other Income/Expenses:", 0, True, None),
        ("  Interest Income", 1, False, 15000),
        ("  Interest Expense", 1, False, -25000),
        ("Net Other Income", 0, True, -10000),
        ("", 0, False, None),
        ("Income Before Tax", 0, True, 240000),
        ("Income Tax", 0, False, 72000),
        ("Net Income", 0, True, 168000),
    ]

    # Populate the sheet
    for row, (account_name, indent_level, is_bold, value) in enumerate(financial_data):
        if account_name:  # Skip empty rows for account names
            sheet.set_cell(
                row,
                0,
                CellData(
                    value=account_name,
                    data_type="text",
                    is_bold=is_bold,
                    indentation_level=indent_level,
                    alignment="left",
                    row=row,
                    column=0,
                ),
            )

        if value is not None:
            sheet.set_cell(
                row,
                1,
                CellData(
                    value=value,
                    data_type="currency",
                    is_bold=is_bold,
                    alignment="right",
                    formatted_value=f"${value:,.0f}",
                    row=row,
                    column=1,
                ),
            )

    sheet.max_row = len(financial_data) - 1
    sheet.max_column = 1

    return sheet


def main():
    """Demonstrate hierarchical pattern detection."""
    print("GridPorter Hierarchical Data Detection Example")
    print("=" * 50)

    # Create sample financial statement
    sheet = create_financial_statement()
    print(f"\nCreated financial statement with {sheet.max_row + 1} rows")

    # Detect patterns using standard detector
    print("\n1. Using Standard Pattern Detector:")
    print("-" * 30)

    detector = SparsePatternDetector()
    patterns = detector.detect_patterns(sheet)

    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i + 1}:")
        print(f"  Type: {pattern.pattern_type.value}")
        print(
            f"  Bounds: Row {pattern.bounds.start_row}-{pattern.bounds.end_row}, "
            f"Col {pattern.bounds.start_col}-{pattern.bounds.end_col}"
        )
        print(f"  Confidence: {pattern.confidence:.2f}")

        if pattern.pattern_type.value == "hierarchical":
            hier_info = pattern.characteristics.get("hierarchical_structure", {})
            print(f"  Max Indentation Depth: {hier_info.get('max_depth', 0)}")
            print(f"  Subtotal Rows: {len(hier_info.get('subtotal_rows', []))}")
            print(f"  Root Level Items: {len(hier_info.get('root_rows', []))}")

    # Direct hierarchical detection
    print("\n\n2. Using Hierarchical Detector Directly:")
    print("-" * 30)

    hier_detector = HierarchicalPatternDetector()
    hier_patterns = hier_detector.detect_hierarchical_patterns(sheet)

    if hier_patterns:
        pattern = hier_patterns[0]
        structure = pattern.characteristics["hierarchical_structure"]

        print("\nHierarchical Structure Detected:")
        print(f"  Maximum indentation depth: {structure['max_depth']}")
        print(f"  Total hierarchical rows: {pattern.characteristics['total_hierarchical_rows']}")

        # Show rows by level
        rows_by_level = structure["rows_by_level"]
        for level, rows in sorted(rows_by_level.items()):
            print(f"\n  Level {level} items ({len(rows)} rows):")
            for row in rows[:5]:  # Show first 5
                cell = sheet.get_cell(row, 0)
                if cell:
                    print(f"    Row {row}: {cell.value}")
            if len(rows) > 5:
                print(f"    ... and {len(rows) - 5} more")

        # Show subtotals
        if structure["subtotal_rows"]:
            print("\n  Subtotal/Total rows:")
            for row in structure["subtotal_rows"]:
                cell = sheet.get_cell(row, 0)
                if cell:
                    print(f"    Row {row}: {cell.value}")

        # Show sample parent-child relationships
        parent_child = structure["parent_child_map"]
        if parent_child:
            print("\n  Sample parent-child relationships:")
            for parent, children in list(parent_child.items())[:3]:
                parent_cell = sheet.get_cell(parent, 0)
                if parent_cell:
                    print(f"    '{parent_cell.value}' has {len(children)} children")
    else:
        print("No hierarchical pattern detected")

    # Show how the data would be useful for pandas
    print("\n\n3. Usage for Data Processing:")
    print("-" * 30)
    print("\nThe hierarchical structure information can be used to:")
    print("- Preserve indentation when converting to DataFrame")
    print("- Create multi-level column headers")
    print("- Identify summary rows to exclude from calculations")
    print("- Build tree structures for financial analysis")
    print("- Generate proper Excel formulas that respect hierarchy")


if __name__ == "__main__":
    main()
