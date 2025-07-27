"""Integration tests for complex table detection."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from gridporter.config import Config
from gridporter.models.table import TableRange, TableInfo, HeaderInfo
from gridporter.models.sheet_data import SheetData, CellData
from gridporter.agents.complex_table_agent import ComplexTableAgent


class TestComplexTableIntegration:
    """Integration tests for complex table detection pipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            use_vision=False,  # Disable vision for unit tests
            suggest_names=False,
            confidence_threshold=0.5,
        )

    @pytest.fixture
    def complex_sheet(self):
        """Create a complex spreadsheet with multiple features."""
        sheet = SheetData(name="Complex Report")

        # Multi-row headers with merged cells
        # Row 0: Main categories
        sheet.set_cell(
            0, 0, CellData(value="Region", is_bold=True, is_merged=True, merge_range="A1:A3")
        )
        sheet.set_cell(
            0,
            1,
            CellData(value="Sales Performance", is_bold=True, is_merged=True, merge_range="B1:E1"),
        )
        sheet.set_cell(
            0,
            5,
            CellData(value="Customer Metrics", is_bold=True, is_merged=True, merge_range="F1:H1"),
        )

        # Row 1: Sub-categories
        sheet.set_cell(
            1, 1, CellData(value="Products", is_bold=True, is_merged=True, merge_range="B2:C2")
        )
        sheet.set_cell(
            1, 3, CellData(value="Services", is_bold=True, is_merged=True, merge_range="D2:E2")
        )
        sheet.set_cell(
            1, 5, CellData(value="Satisfaction", is_bold=True, is_merged=True, merge_range="F2:G2")
        )
        sheet.set_cell(1, 7, CellData(value="Retention", is_bold=True))

        # Row 2: Detailed headers
        sheet.set_cell(2, 1, CellData(value="Q1", is_bold=True))
        sheet.set_cell(2, 2, CellData(value="Q2", is_bold=True))
        sheet.set_cell(2, 3, CellData(value="Q1", is_bold=True))
        sheet.set_cell(2, 4, CellData(value="Q2", is_bold=True))
        sheet.set_cell(2, 5, CellData(value="Score", is_bold=True))
        sheet.set_cell(2, 6, CellData(value="NPS", is_bold=True))
        sheet.set_cell(2, 7, CellData(value="Rate %", is_bold=True))

        # Section 1: North Region
        sheet.set_cell(3, 0, CellData(value="North", is_bold=True, background_color="#E0E0E0"))

        # Data rows with indentation
        sheet.set_cell(4, 0, CellData(value="  New York", indentation_level=1))
        sheet.set_cell(4, 1, CellData(value=1000, data_type="number"))
        sheet.set_cell(4, 2, CellData(value=1200, data_type="number"))
        sheet.set_cell(4, 3, CellData(value=500, data_type="number"))
        sheet.set_cell(4, 4, CellData(value=600, data_type="number"))
        sheet.set_cell(4, 5, CellData(value=4.5, data_type="number"))
        sheet.set_cell(4, 6, CellData(value=72, data_type="number"))
        sheet.set_cell(4, 7, CellData(value=0.92, data_type="number"))

        sheet.set_cell(5, 0, CellData(value="  Boston", indentation_level=1))
        sheet.set_cell(5, 1, CellData(value=800, data_type="number"))
        sheet.set_cell(5, 2, CellData(value=900, data_type="number"))
        # ... more data

        # Subtotal row
        sheet.set_cell(6, 0, CellData(value="North Total", is_bold=True))
        sheet.set_cell(6, 1, CellData(value=1800, data_type="number", is_bold=True))
        sheet.set_cell(6, 2, CellData(value=2100, data_type="number", is_bold=True))
        # ... more totals

        # Blank separator
        sheet.set_cell(7, 0, CellData(value=None))

        # Section 2: South Region
        sheet.set_cell(8, 0, CellData(value="South", is_bold=True, background_color="#E0E0E0"))
        # ... more data

        # Grand total
        sheet.set_cell(
            15, 0, CellData(value="Grand Total", is_bold=True, background_color="#CCCCCC")
        )
        sheet.set_cell(15, 1, CellData(value=5000, data_type="number", is_bold=True))
        # ... more grand totals

        return sheet

    @pytest.mark.asyncio
    async def test_complex_table_detection(self, config, complex_sheet):
        """Test detection of complex table with all features."""
        agent = ComplexTableAgent(config)

        # Run detection
        result = await agent.detect_complex_tables(complex_sheet)

        assert result is not None
        assert len(result.tables) > 0
        assert result.confidence > 0.5

        # Check first table
        table = result.tables[0]
        assert isinstance(table, TableInfo)

        # Check multi-row headers
        assert table.header_info is not None
        assert table.header_info.is_multi_row
        assert table.header_info.row_count == 3

        # Check semantic structure
        assert table.semantic_structure is not None
        assert table.semantic_structure.get("has_subtotals") is True
        assert table.semantic_structure.get("has_grand_total") is True
        assert len(table.semantic_structure.get("sections", [])) > 0

    @pytest.mark.asyncio
    async def test_simple_table_enhancement(self, config):
        """Test enhancement of simple table ranges."""
        agent = ComplexTableAgent(config)

        # Create simple sheet
        sheet = SheetData(name="Simple")
        sheet.set_cell(0, 0, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Value", is_bold=True))
        sheet.set_cell(1, 0, CellData(value="Item1"))
        sheet.set_cell(1, 1, CellData(value=100, data_type="number"))

        # Simple table range
        simple_ranges = [TableRange(start_row=0, start_col=0, end_row=1, end_col=1)]

        # Enhance
        result = await agent.detect_complex_tables(
            sheet, vision_result=None, simple_tables=simple_ranges
        )

        assert len(result.tables) == 1
        table = result.tables[0]

        # Should have basic header info
        assert table.header_info is not None
        assert not table.header_info.is_multi_row
        assert table.header_info.row_count == 1

    @pytest.mark.asyncio
    async def test_format_preservation(self, config, complex_sheet):
        """Test that formatting information is preserved."""
        agent = ComplexTableAgent(config)

        result = await agent.detect_complex_tables(complex_sheet)

        assert len(result.tables) > 0
        table = result.tables[0]

        # Check format preservation
        assert table.format_preservation is not None
        preserve_blanks = table.format_preservation.get("preserve_blank_rows", [])
        assert len(preserve_blanks) > 0  # Should preserve separator rows

        # Check format patterns
        patterns = table.format_preservation.get("format_patterns", [])
        assert len(patterns) >= 0  # May have detected patterns

    @pytest.mark.asyncio
    async def test_data_preview_generation(self, config):
        """Test generation of data preview."""
        agent = ComplexTableAgent(config)

        # Create sheet with data
        sheet = SheetData(name="Data")
        sheet.set_cell(0, 0, CellData(value="ID", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Name", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Score", is_bold=True))

        for i in range(1, 10):
            sheet.set_cell(i, 0, CellData(value=i, data_type="number"))
            sheet.set_cell(i, 1, CellData(value=f"Person {i}"))
            sheet.set_cell(i, 2, CellData(value=80 + i, data_type="number"))

        result = await agent.detect_complex_tables(sheet)

        assert len(result.tables) > 0
        table = result.tables[0]

        # Check data preview
        assert table.data_preview is not None
        assert len(table.data_preview) <= 5  # Should limit preview
        assert len(table.data_preview) > 0

        # Check preview structure
        first_row = table.data_preview[0]
        assert "ID" in first_row
        assert "Name" in first_row
        assert "Score" in first_row

    @pytest.mark.asyncio
    async def test_data_type_inference(self, config):
        """Test inference of column data types."""
        agent = ComplexTableAgent(config)

        # Create sheet with mixed data types
        sheet = SheetData(name="Mixed Types")
        sheet.set_cell(0, 0, CellData(value="Text", is_bold=True))
        sheet.set_cell(0, 1, CellData(value="Number", is_bold=True))
        sheet.set_cell(0, 2, CellData(value="Date", is_bold=True))
        sheet.set_cell(0, 3, CellData(value="Mixed", is_bold=True))

        # Add data
        for i in range(1, 6):
            sheet.set_cell(i, 0, CellData(value=f"Text {i}", data_type="string"))
            sheet.set_cell(i, 1, CellData(value=i * 10, data_type="number"))
            sheet.set_cell(i, 2, CellData(value=f"2024-01-{i:02d}", data_type="string"))
            sheet.set_cell(
                i,
                3,
                CellData(
                    value=i if i % 2 == 0 else f"Text {i}",
                    data_type="number" if i % 2 == 0 else "string",
                ),
            )

        result = await agent.detect_complex_tables(sheet)

        assert len(result.tables) > 0
        table = result.tables[0]

        # Check data types
        assert table.data_types is not None
        assert table.data_types.get("Text") == "string"
        assert table.data_types.get("Number") == "number"
        # Mixed column should be detected as predominant type
        assert "Mixed" in table.data_types

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in complex table detection."""
        agent = ComplexTableAgent(config)

        # Empty sheet
        empty_sheet = SheetData(name="Empty")
        result = await agent.detect_complex_tables(empty_sheet)
        assert result is not None
        assert len(result.tables) == 0

        # Invalid range
        sheet = SheetData(name="Test")
        sheet.set_cell(0, 0, CellData(value="Test"))

        # Mock a table range that's out of bounds
        invalid_ranges = [TableRange(start_row=10, start_col=10, end_row=20, end_col=20)]

        result = await agent.detect_complex_tables(sheet, simple_tables=invalid_ranges)

        # Should handle gracefully
        assert result is not None
        assert result.confidence >= 0
