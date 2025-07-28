"""Test script to verify Week 5 testing guide alignment with implementation."""

import asyncio
import tempfile
from pathlib import Path

from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.config import Config
from gridporter.models.sheet_data import CellData, SheetData
from gridporter.telemetry import get_feature_collector
from gridporter.telemetry.feature_store import FeatureStore


async def test_basic_complex_table():
    """Test basic complex table detection as shown in testing guide."""
    # Configure agent
    config = Config(
        use_vision=False,  # Local analysis only for this test
        suggest_names=False,
        confidence_threshold=0.5,
    )

    agent = ComplexTableAgent(config)

    # Create a simple multi-header table
    sheet = SheetData(name="MultiHeader")

    # Headers spanning 3 rows with hierarchy
    sheet.set_cell(0, 0, CellData(value="Company", is_bold=True))
    sheet.set_cell(0, 1, CellData(value="Financials", is_bold=True))
    sheet.set_cell(0, 2, CellData(value="Financials", is_bold=True))
    sheet.set_cell(0, 3, CellData(value="Financials", is_bold=True))

    sheet.set_cell(1, 1, CellData(value="Revenue", is_bold=True))
    sheet.set_cell(1, 2, CellData(value="Costs", is_bold=True))
    sheet.set_cell(1, 3, CellData(value="Profit", is_bold=True))

    sheet.set_cell(2, 1, CellData(value="2023", is_bold=True))
    sheet.set_cell(2, 2, CellData(value="2023", is_bold=True))
    sheet.set_cell(2, 3, CellData(value="2023", is_bold=True))

    # Data rows
    sheet.set_cell(3, 0, CellData(value="Acme Corp"))
    sheet.set_cell(3, 1, CellData(value=1000000, data_type="number"))
    sheet.set_cell(3, 2, CellData(value=700000, data_type="number"))
    sheet.set_cell(3, 3, CellData(value=300000, data_type="number"))

    sheet.set_cell(4, 0, CellData(value="Beta Inc"))
    sheet.set_cell(4, 1, CellData(value=500000, data_type="number"))
    sheet.set_cell(4, 2, CellData(value=400000, data_type="number"))
    sheet.set_cell(4, 3, CellData(value=100000, data_type="number"))

    # Run detection
    result = await agent.detect_complex_tables(sheet)

    print(f"Detected {len(result.tables)} tables")
    print(f"Confidence: {result.confidence}")
    print(f"Detection metadata: {result.detection_metadata}")

    for i, table in enumerate(result.tables):
        print(f"\nTable {i+1}:")
        print(f"  Range: {table.range.excel_range}")
        print(f"  Has headers: {table.has_headers}")
        if table.header_info:
            print(f"  Header rows: {table.header_info.row_count}")
            print(f"  Is multi-row: {table.header_info.is_multi_row}")
            if table.header_info.multi_row_headers:
                print(f"  Column mappings: {table.header_info.multi_row_headers}")


async def test_with_feature_collection():
    """Test complex table detection with feature collection enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        feature_db = Path(tmpdir) / "test_features.db"

        # Configure with feature collection
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

        agent = ComplexTableAgent(config)

        # Create financial sheet with sections and subtotals
        sheet = SheetData(name="FinancialReport")
        sheet.file_path = "test_financial.xlsx"  # Set for feature collection
        sheet.file_type = "xlsx"

        # Headers
        sheet.set_cell(0, 0, CellData(value="Account", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Q1", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Q2", is_bold=True))
        sheet.set_cell(0, 3, CellData(value="Total", is_bold=True))

        # Revenue section
        sheet.set_cell(1, 0, CellData(value="Revenue", is_bold=True, background_color="#E0E0E0"))
        sheet.set_cell(2, 0, CellData(value="  Product Sales", indentation_level=1))
        sheet.set_cell(2, 1, CellData(value=1000, data_type="number"))
        sheet.set_cell(2, 2, CellData(value=1200, data_type="number"))
        sheet.set_cell(2, 3, CellData(value=2200, data_type="number"))

        sheet.set_cell(3, 0, CellData(value="  Services", indentation_level=1))
        sheet.set_cell(3, 1, CellData(value=500, data_type="number"))
        sheet.set_cell(3, 2, CellData(value=600, data_type="number"))
        sheet.set_cell(3, 3, CellData(value=1100, data_type="number"))

        # Subtotal
        sheet.set_cell(4, 0, CellData(value="Total Revenue", is_bold=True))
        sheet.set_cell(4, 1, CellData(value=1500, data_type="number", is_bold=True))
        sheet.set_cell(4, 2, CellData(value=1800, data_type="number", is_bold=True))
        sheet.set_cell(4, 3, CellData(value=3300, data_type="number", is_bold=True))

        # Run detection
        result = await agent.detect_complex_tables(sheet)

        print("\nWith Feature Collection:")
        print(f"Detected {len(result.tables)} tables")
        print(f"Confidence: {result.confidence}")

        # Query collected features
        store = FeatureStore(str(feature_db))
        features = store.query_features()

        if features:
            latest = features[-1]
            print("\nCollected Features:")
            print(f"  Detection method: {latest.detection_method}")
            print(f"  Header rows: {latest.header_row_count}")
            print(f"  Has subtotals: {latest.has_subtotals}")
            print(f"  Section count: {latest.section_count}")
            print(f"  Processing time: {latest.processing_time_ms}ms")

        # Get summary statistics
        stats = feature_collector.get_summary_statistics()
        if stats:
            print("\nFeature Collection Stats:")
            print(f"  Total detections: {stats['total_records']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Average confidence: {stats['avg_confidence']:.2f}")

            # Show by method
            print("  By method:")
            for method, method_stats in stats.get("by_method", {}).items():
                print(f"    {method}: {method_stats['count']} detections")

        store.close()
        feature_collector.close()


async def main():
    """Run all tests."""
    print("=== Test 1: Basic Complex Table Detection ===")
    await test_basic_complex_table()

    print("\n=== Test 2: Complex Table with Feature Collection ===")
    await test_with_feature_collection()


if __name__ == "__main__":
    asyncio.run(main())
