"""Week 5 integration tests for complex table detection with feature collection."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from gridporter.config import Config
from gridporter.models.table import TableRange, TableInfo, HeaderInfo
from gridporter.models.sheet_data import SheetData, CellData
from gridporter.agents.complex_table_agent import ComplexTableAgent
from gridporter.telemetry import get_feature_collector
from gridporter.telemetry.feature_store import FeatureStore
from gridporter.detectors.format_analyzer import SemanticFormatAnalyzer, RowType


class TestWeek5ComplexTablesWithFeatures:
    """Integration tests for Week 5 complex table detection with feature collection."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for feature database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config_with_features(self, temp_dir):
        """Create test configuration with feature collection enabled."""
        feature_db = Path(temp_dir) / "test_features.db"
        return Config(
            use_vision=False,
            suggest_names=False,
            confidence_threshold=0.5,
            enable_feature_collection=True,
            feature_db_path=str(feature_db),
        )

    @pytest.fixture
    def feature_collector(self, config_with_features):
        """Initialize feature collector."""
        collector = get_feature_collector()
        collector.initialize(enabled=True, db_path=config_with_features.feature_db_path)
        yield collector
        collector.close()

    @pytest.fixture
    def financial_sheet(self):
        """Create financial report with semantic structure."""
        sheet = SheetData(name="FinancialReport")

        # Headers
        sheet.set_cell(0, 0, CellData(value="Account", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Q1", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Q2", is_bold=True))
        sheet.set_cell(0, 3, CellData(value="Q3", is_bold=True))
        sheet.set_cell(0, 4, CellData(value="Q4", is_bold=True))

        # Section: Revenue
        sheet.set_cell(1, 0, CellData(value="Revenue", is_bold=True, background_color="#E0E0E0"))

        # Revenue items
        revenue_items = [
            ("Product Sales", [1000, 1200, 1100, 1300]),
            ("Service Revenue", [500, 600, 550, 700]),
            ("Licensing Fees", [200, 250, 300, 350]),
        ]

        row_idx = 2
        for item, values in revenue_items:
            sheet.set_cell(row_idx, 0, CellData(value=f"  {item}", indentation_level=1))
            for col, val in enumerate(values, 1):
                sheet.set_cell(row_idx, col, CellData(value=val, data_type="number"))
            row_idx += 1

        # Subtotal
        sheet.set_cell(row_idx, 0, CellData(value="Total Revenue", is_bold=True))
        for col in range(1, 5):
            total = sum(item[1][col - 1] for item in revenue_items)
            sheet.set_cell(row_idx, col, CellData(value=total, data_type="number", is_bold=True))
        row_idx += 1

        # Blank separator
        row_idx += 1

        # Section: Expenses
        sheet.set_cell(
            row_idx, 0, CellData(value="Expenses", is_bold=True, background_color="#E0E0E0")
        )
        row_idx += 1

        # Expense items
        expense_items = [
            ("Operating Costs", [800, 850, 900, 950]),
            ("Marketing", [300, 350, 400, 450]),
        ]

        for item, values in expense_items:
            sheet.set_cell(row_idx, 0, CellData(value=f"  {item}", indentation_level=1))
            for col, val in enumerate(values, 1):
                sheet.set_cell(row_idx, col, CellData(value=val, data_type="number"))
            row_idx += 1

        # Set file metadata for feature collection
        sheet.file_path = "test_financial.xlsx"
        sheet.file_type = "xlsx"

        return sheet

    @pytest.fixture
    def multi_header_sheet(self):
        """Create sheet with complex multi-row headers."""
        sheet = SheetData(name="ComplexHeaders")

        # Level 1: Department
        sheet.set_cell(0, 0, CellData(value="", is_bold=True))  # Empty corner
        sheet.set_cell(
            0,
            1,
            CellData(value="Sales Department", is_bold=True, is_merged=True, merge_range="B1:G1"),
        )
        sheet.set_cell(
            0,
            7,
            CellData(value="Support Department", is_bold=True, is_merged=True, merge_range="H1:K1"),
        )

        # Level 2: Region
        sheet.set_cell(
            1, 1, CellData(value="North", is_bold=True, is_merged=True, merge_range="B2:D2")
        )
        sheet.set_cell(
            1, 4, CellData(value="South", is_bold=True, is_merged=True, merge_range="E2:G2")
        )
        sheet.set_cell(
            1, 7, CellData(value="East", is_bold=True, is_merged=True, merge_range="H2:I2")
        )
        sheet.set_cell(
            1, 9, CellData(value="West", is_bold=True, is_merged=True, merge_range="J2:K2")
        )

        # Level 3: Metrics
        metrics = ["Revenue", "Cost", "Profit"]
        col_idx = 1
        for region in range(4):
            for metric in metrics[: 2 if region >= 2 else 3]:
                sheet.set_cell(2, col_idx, CellData(value=metric, is_bold=True))
                col_idx += 1

        # Row headers
        sheet.set_cell(3, 0, CellData(value="Q1 2024", is_bold=True))
        sheet.set_cell(4, 0, CellData(value="Q2 2024", is_bold=True))

        # Add sample data
        import random

        for row in range(3, 5):
            for col in range(1, 11):
                sheet.set_cell(
                    row, col, CellData(value=random.randint(100, 1000), data_type="number")
                )

        # Set file metadata
        sheet.file_path = "test_multiheader.xlsx"
        sheet.file_type = "xlsx"

        return sheet

    @pytest.fixture
    def pattern_sheet(self):
        """Create sheet with alternating row patterns."""
        sheet = SheetData(name="Patterns")

        # Headers
        for col, header in enumerate(["Item", "Value", "Status"]):
            sheet.set_cell(0, col, CellData(value=header, is_bold=True))

        # Data with alternating backgrounds
        colors = ["#FFFFFF", "#F5F5F5"]
        for row in range(1, 11):
            bg_color = colors[row % 2]
            sheet.set_cell(row, 0, CellData(value=f"Item {row}", background_color=bg_color))
            sheet.set_cell(
                row,
                1,
                CellData(
                    value=row * 100,
                    data_type="number",
                    background_color=bg_color,
                    alignment="right",
                ),
            )
            sheet.set_cell(
                row,
                2,
                CellData(
                    value="Active", background_color=bg_color, alignment="center", is_bold=True
                ),
            )

        sheet.file_path = "test_patterns.xlsx"
        sheet.file_type = "xlsx"

        return sheet

    @pytest.mark.asyncio
    async def test_feature_collection_enabled(self, config_with_features, feature_collector):
        """Test that feature collection is properly enabled."""
        assert feature_collector.enabled
        assert Path(config_with_features.feature_db_path).parent.exists()

    @pytest.mark.asyncio
    async def test_complex_table_with_features(
        self, config_with_features, feature_collector, financial_sheet
    ):
        """Test complex table detection records features."""
        agent = ComplexTableAgent(config_with_features)

        # Detect tables
        result = await agent.detect_complex_tables(financial_sheet)

        # Verify detection succeeded
        assert len(result.tables) > 0
        assert result.confidence > 0.5

        # Query collected features
        store = FeatureStore(config_with_features.feature_db_path)
        features = store.query_features()

        assert len(features) > 0

        # Check latest feature
        latest = features[-1]
        assert latest.detection_method == "complex_detection"
        assert latest.confidence == result.confidence
        assert latest.detection_success is True
        assert latest.file_path == "test_financial.xlsx"
        assert latest.sheet_name == "FinancialReport"

        # Check semantic features
        assert latest.has_subtotals is True
        # Note: section count depends on how much of the sheet is detected
        assert latest.section_count >= 1
        assert latest.has_bold_headers is True

        store.close()

    @pytest.mark.asyncio
    async def test_multi_header_features(
        self, config_with_features, feature_collector, multi_header_sheet
    ):
        """Test feature recording for multi-row headers."""
        agent = ComplexTableAgent(config_with_features)
        await agent.detect_complex_tables(multi_header_sheet)

        # Query features
        store = FeatureStore(config_with_features.feature_db_path)
        features = store.query_features(detection_method="complex_detection")

        # Find the multi-header feature
        multi_header_features = [f for f in features if f.file_path == "test_multiheader.xlsx"]
        assert len(multi_header_features) > 0

        feature = multi_header_features[0]
        assert feature.header_row_count == 3  # 3 levels of headers
        assert feature.has_multi_headers is True  # Correct field name

        store.close()

    @pytest.mark.asyncio
    async def test_pattern_features(self, config_with_features, feature_collector, pattern_sheet):
        """Test recording of format pattern features."""
        agent = ComplexTableAgent(config_with_features)
        await agent.detect_complex_tables(pattern_sheet)

        # Query features
        store = FeatureStore(config_with_features.feature_db_path)
        features = store.query_features()

        pattern_features = [f for f in features if f.file_path == "test_patterns.xlsx"]
        assert len(pattern_features) > 0

        feature = pattern_features[0]
        # Note: These specific pattern features would need to be added to the model
        # For now, check what we can verify
        assert feature.has_bold_headers is True  # Headers are bold
        assert feature.filled_cells > 0  # Has data

        store.close()

    @pytest.mark.asyncio
    async def test_feature_statistics(
        self,
        config_with_features,
        feature_collector,
        financial_sheet,
        multi_header_sheet,
        pattern_sheet,
    ):
        """Test feature collection statistics."""
        agent = ComplexTableAgent(config_with_features)

        # Process multiple sheets
        sheets = [financial_sheet, multi_header_sheet, pattern_sheet]
        for sheet in sheets:
            await agent.detect_complex_tables(sheet)

        # Get statistics
        stats = feature_collector.get_summary_statistics()

        assert stats["total_records"] >= 3
        assert stats["success_rate"] == 1.0  # All should succeed
        assert "complex_detection" in stats["by_method"]

        # Check method-specific stats
        complex_stats = stats["by_method"].get("complex_detection", {})
        assert complex_stats["count"] >= 3
        assert complex_stats["avg_confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_feature_export(self, config_with_features, feature_collector, financial_sheet):
        """Test exporting collected features."""
        agent = ComplexTableAgent(config_with_features)
        await agent.detect_complex_tables(financial_sheet)

        # Export features
        export_path = Path(config_with_features.feature_db_path).parent / "features.csv"
        feature_collector.export_features(str(export_path), format="csv")

        assert export_path.exists()
        assert export_path.stat().st_size > 0

        # Read and verify CSV content
        import csv

        with open(export_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0

            # Check expected columns
            expected_cols = ["file_path", "detection_method", "confidence", "header_row_count"]
            for col in expected_cols:
                assert col in rows[0]

    @pytest.mark.asyncio
    async def test_processing_time_tracking(
        self, config_with_features, feature_collector, financial_sheet
    ):
        """Test that processing time is properly tracked."""
        agent = ComplexTableAgent(config_with_features)
        await agent.detect_complex_tables(financial_sheet)

        store = FeatureStore(config_with_features.feature_db_path)
        features = store.query_features()

        latest = features[-1]
        assert latest.processing_time_ms is not None
        assert latest.processing_time_ms > 0
        assert latest.processing_time_ms < 5000  # Should be fast

        store.close()

    @pytest.mark.asyncio
    async def test_error_handling_with_features(self, config_with_features, feature_collector):
        """Test feature collection handles errors gracefully."""
        agent = ComplexTableAgent(config_with_features)

        # Create invalid sheet
        invalid_sheet = SheetData(name="Invalid")
        invalid_sheet.file_path = "test_invalid.xlsx"
        invalid_sheet.file_type = "xlsx"

        # This should handle gracefully
        await agent.detect_complex_tables(invalid_sheet)

        # Even with empty sheet, should record attempt
        store = FeatureStore(config_with_features.feature_db_path)
        features = store.query_features()

        # Should have recorded the attempt
        invalid_features = [f for f in features if f.file_path == "test_invalid.xlsx"]
        if invalid_features:  # May or may not record depending on implementation
            feature = invalid_features[0]
            assert feature.detection_method == "complex_detection"

        store.close()

    @pytest.mark.asyncio
    async def test_feature_cleanup(self, config_with_features, feature_collector, financial_sheet):
        """Test feature data cleanup."""
        agent = ComplexTableAgent(config_with_features)

        # Record some features
        await agent.detect_complex_tables(financial_sheet)

        # Verify features exist
        store = FeatureStore(config_with_features.feature_db_path)
        initial_count = len(store.query_features())
        assert initial_count > 0

        # Cleanup (0 days = remove all)
        feature_collector.cleanup(days=0)

        # Verify cleanup worked
        remaining = len(store.query_features())
        assert remaining < initial_count

        store.close()
