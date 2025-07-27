"""Week 5 Example: Complex Table Detection with Feature Collection.

This example demonstrates:
1. Multi-row header detection
2. Semantic structure analysis
3. Format preservation
4. Feature collection and analysis
"""

import asyncio
import tempfile
from pathlib import Path

import pandas as pd
from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.telemetry import get_feature_collector
from gridporter.telemetry.feature_store import FeatureStore


async def create_complex_financial_report():
    """Create a complex financial report with multi-level structure."""
    sheet = SheetData(name="Q4_Financial_Report")
    sheet.file_path = "financial_report.xlsx"
    sheet.file_type = "xlsx"

    # Multi-row headers with merged cells
    # Level 1: Year
    sheet.set_cell(0, 0, CellData(value="", is_bold=True))  # Empty corner
    sheet.set_cell(0, 1, CellData(value="2023", is_bold=True, is_merged=True, merge_range="B1:E1"))
    sheet.set_cell(0, 5, CellData(value="2024", is_bold=True, is_merged=True, merge_range="F1:I1"))

    # Level 2: Quarters
    sheet.set_cell(1, 0, CellData(value="Account", is_bold=True))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    col = 1
    for _year in range(2):
        for q in quarters:
            sheet.set_cell(1, col, CellData(value=q, is_bold=True))
            col += 1

    # Revenue Section
    row = 2
    sheet.set_cell(row, 0, CellData(value="REVENUE", is_bold=True, background_color="#E8F4FD"))
    row += 1

    revenue_items = [
        ("Product Sales", [100, 120, 140, 160, 110, 130, 150, 170]),
        ("Service Revenue", [50, 55, 60, 65, 55, 60, 65, 70]),
        ("Licensing", [20, 22, 24, 26, 22, 24, 26, 28]),
        ("Other Income", [10, 11, 12, 13, 11, 12, 13, 14]),
    ]

    for item, values in revenue_items:
        sheet.set_cell(row, 0, CellData(value=f"  {item}", indentation_level=1))
        for col, val in enumerate(values, 1):
            sheet.set_cell(row, col, CellData(value=val * 1000, data_type="number"))
        row += 1

    # Revenue Subtotal
    sheet.set_cell(row, 0, CellData(value="Total Revenue", is_bold=True))
    for col in range(1, 9):
        total = sum(item[1][col - 1] * 1000 for item in revenue_items)
        sheet.set_cell(row, col, CellData(value=total, data_type="number", is_bold=True))
    row += 1

    # Blank separator
    row += 1

    # Expenses Section
    sheet.set_cell(row, 0, CellData(value="EXPENSES", is_bold=True, background_color="#FFEBEE"))
    row += 1

    expense_items = [
        ("Cost of Goods Sold", [60, 72, 84, 96, 66, 78, 90, 102]),
        ("Operating Expenses", [30, 33, 36, 39, 33, 36, 39, 42]),
        ("Marketing", [15, 16, 17, 18, 16, 17, 18, 19]),
        ("R&D", [25, 27, 29, 31, 27, 29, 31, 33]),
    ]

    for item, values in expense_items:
        sheet.set_cell(row, 0, CellData(value=f"  {item}", indentation_level=1))
        for col, val in enumerate(values, 1):
            sheet.set_cell(row, col, CellData(value=val * 1000, data_type="number"))
        row += 1

    # Expense Subtotal
    sheet.set_cell(row, 0, CellData(value="Total Expenses", is_bold=True))
    for col in range(1, 9):
        total = sum(item[1][col - 1] * 1000 for item in expense_items)
        sheet.set_cell(row, col, CellData(value=total, data_type="number", is_bold=True))
    row += 1

    # Blank separator
    row += 1

    # Net Income (Grand Total)
    sheet.set_cell(row, 0, CellData(value="NET INCOME", is_bold=True, background_color="#F5F5F5"))
    for col in range(1, 9):
        revenue = sum(item[1][col - 1] * 1000 for item in revenue_items)
        expenses = sum(item[1][col - 1] * 1000 for item in expense_items)
        sheet.set_cell(
            row,
            col,
            CellData(
                value=revenue - expenses,
                data_type="number",
                is_bold=True,
                font_color="#006400" if revenue > expenses else "#8B0000",
            ),
        )

    return sheet


async def main():
    """Run complex table detection with feature collection."""
    print("=== Week 5: Complex Table Detection with Feature Collection ===\n")

    # Create temporary directory for feature database
    with tempfile.TemporaryDirectory() as tmpdir:
        feature_db = Path(tmpdir) / "week5_features.db"

        # Configure GridPorter with feature collection
        config = Config(
            use_vision=False,
            suggest_names=False,
            confidence_threshold=0.5,
            enable_feature_collection=True,
            feature_db_path=str(feature_db),
        )

        # Initialize feature collector
        feature_collector = get_feature_collector()
        feature_collector.initialize(enabled=True, db_path=str(feature_db))

        # Create complex table agent
        agent = ComplexTableAgent(config)

        # Create and process financial report
        print("Creating complex financial report...")
        financial_sheet = await create_complex_financial_report()

        print("Detecting tables...")
        result = await agent.detect_complex_tables(financial_sheet)

        print("\nDetection Results:")
        print(f"  Tables found: {len(result.tables)}")
        print(f"  Overall confidence: {result.confidence:.2f}")
        print(
            f"  Detection method: {result.detection_metadata.get('primary_method', 'complex_detection')}"
        )

        # Show table details
        for i, table in enumerate(result.tables):
            print(f"\nTable {i+1}:")
            print(f"  Range: {table.range.excel_range}")
            print(f"  Confidence: {table.confidence:.2f}")

            if table.header_info:
                print("  Header Information:")
                print(f"    - Row count: {table.header_info.row_count}")
                print(f"    - Multi-row: {table.header_info.is_multi_row}")
                if table.header_info.multi_row_headers:
                    print(
                        f"    - Column mappings: {len(table.header_info.multi_row_headers)} columns"
                    )

            if table.semantic_structure:
                print("  Semantic Structure:")
                print(f"    - Sections: {len(table.semantic_structure.get('sections', []))}")
                print(
                    f"    - Has subtotals: {table.semantic_structure.get('has_subtotals', False)}"
                )
                print(
                    f"    - Has grand total: {table.semantic_structure.get('has_grand_total', False)}"
                )
                print(
                    f"    - Preserve blank rows: {table.semantic_structure.get('preserve_blank_rows', [])}"
                )

        # Analyze collected features
        print("\n" + "=" * 50)
        print("Feature Collection Analysis")
        print("=" * 50)

        # Query features
        store = FeatureStore(str(feature_db))
        features = store.query_features()

        if features:
            print(f"\nTotal features collected: {len(features)}")

            # Show latest feature details
            latest = features[-1]
            print("\nLatest Detection Features:")
            print(f"  File: {latest.file_path}")
            print(f"  Sheet: {latest.sheet_name}")
            print(f"  Table range: {latest.table_range}")
            print(f"  Detection method: {latest.detection_method}")
            print(f"  Confidence: {latest.confidence:.2f}")
            print(f"  Success: {latest.detection_success}")

            print("\nHeader Features:")
            print(f"  Header rows: {latest.header_row_count}")
            print(f"  Multi-row headers: {latest.has_multi_headers}")
            print(f"  Bold headers: {latest.has_bold_headers}")

            print("\nSemantic Features:")
            print(f"  Has subtotals: {latest.has_subtotals}")
            print(f"  Has totals: {latest.has_totals}")
            print(f"  Section count: {latest.section_count}")

            print("\nContent Features:")
            print(f"  Total cells: {latest.total_cells}")
            print(f"  Filled cells: {latest.filled_cells}")
            if latest.total_cells:
                print(f"  Fill ratio: {latest.filled_cells / latest.total_cells:.2%}")
            print(
                f"  Numeric ratio: {latest.numeric_ratio:.2%}"
                if latest.numeric_ratio
                else "  Numeric ratio: N/A"
            )

            print("\nProcessing:")
            print(
                f"  Time: {latest.processing_time_ms}ms"
                if latest.processing_time_ms
                else "  Time: N/A"
            )

        # Get summary statistics
        stats = feature_collector.get_summary_statistics()
        if stats:
            print("\n" + "=" * 50)
            print("Summary Statistics")
            print("=" * 50)
            print(f"  Total records: {stats['total_records']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Average confidence: {stats['avg_confidence']:.2f}")

            print("\nBy Detection Method:")
            for method, method_stats in stats["by_method"].items():
                print(f"  {method}:")
                print(f"    - Count: {method_stats['count']}")
                print(f"    - Avg confidence: {method_stats['avg_confidence']:.2f}")

        # Export features for further analysis
        export_path = Path(tmpdir) / "week5_features.csv"
        feature_collector.export_features(str(export_path))
        print(f"\nFeatures exported to: {export_path}")

        # Show how to analyze with pandas
        print("\n" + "=" * 50)
        print("Feature Analysis with Pandas")
        print("=" * 50)

        try:
            df = pd.read_csv(export_path)
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nColumns: {', '.join(df.columns[:10])}...")  # Show first 10 columns

            # Basic statistics
            if "confidence" in df.columns:
                print("\nConfidence Statistics:")
                print(f"  Mean: {df['confidence'].mean():.3f}")
                print(f"  Std: {df['confidence'].std():.3f}")
                print(f"  Min: {df['confidence'].min():.3f}")
                print(f"  Max: {df['confidence'].max():.3f}")

            # Feature correlations
            numeric_cols = [
                "confidence",
                "header_row_count",
                "section_count",
                "filled_cells",
                "total_cells",
                "processing_time_ms",
            ]
            available_cols = [col for col in numeric_cols if col in df.columns]
            if len(available_cols) > 1:
                print("\nFeature Correlations with Confidence:")
                correlations = df[available_cols].corr()["confidence"].sort_values(ascending=False)
                for feature, corr in correlations.items():
                    if feature != "confidence":
                        print(f"  {feature}: {corr:.3f}")
        except Exception as e:
            print(f"Pandas analysis skipped: {e}")

        # Cleanup
        store.close()
        feature_collector.close()

        print("\n" + "=" * 50)
        print("Week 5 Example Complete!")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
