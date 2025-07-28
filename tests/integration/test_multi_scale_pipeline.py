"""Integration tests for multi-scale bitmap generation pipeline."""

import pytest
from unittest.mock import MagicMock, patch

from gridporter.models.sheet_data import CellData, SheetData
from gridporter.models.multi_scale import CompressionLevel
from gridporter.vision.vision_request_builder import VisionRequestBuilder
from gridporter.agents.vision_orchestrator_agent import VisionOrchestratorAgent
from gridporter.config import Config


class TestMultiScalePipeline:
    """Integration tests for the complete multi-scale pipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            use_vision=True,
            min_table_size=(2, 2),
            confidence_threshold=0.7,
            enable_region_verification=True,
            verification_strict_mode=False,
            max_cost_per_session=10.0,
            max_cost_per_file=1.0,
        )

    @pytest.fixture
    def small_sheet_single_table(self):
        """Create small sheet with single table."""
        sheet = SheetData(name="SmallSingle")

        # Headers
        headers = ["ID", "Name", "Age", "Department", "Salary"]
        for col, header in enumerate(headers):
            sheet.set_cell(
                0, col, CellData(row=0, column=col, value=header, data_type="text", is_bold=True)
            )

        # Data rows
        data = [
            [1, "Alice", 28, "Engineering", 75000],
            [2, "Bob", 32, "Marketing", 65000],
            [3, "Charlie", 25, "Sales", 55000],
            [4, "Diana", 35, "Engineering", 85000],
            [5, "Eve", 29, "HR", 60000],
        ]

        for row_idx, row_data in enumerate(data, 1):
            for col_idx, value in enumerate(row_data):
                data_type = "number" if isinstance(value, (int, float)) else "text"
                sheet.set_cell(
                    row_idx,
                    col_idx,
                    CellData(row=row_idx, column=col_idx, value=value, data_type=data_type),
                )

        return sheet

    @pytest.fixture
    def medium_sheet_multi_table(self):
        """Create medium sheet with multiple tables."""
        sheet = SheetData(name="MediumMulti")

        # Table 1: Sales data (top-left)
        # Headers
        for col, header in enumerate(["Product", "Q1", "Q2", "Q3", "Q4"]):
            sheet.set_cell(
                0, col, CellData(row=0, column=col, value=header, data_type="text", is_bold=True)
            )

        # Sales data
        products = ["Widget A", "Widget B", "Widget C"]
        for row, product in enumerate(products, 1):
            sheet.set_cell(row, 0, CellData(row=row, column=0, value=product, data_type="text"))
            for quarter in range(1, 5):
                sheet.set_cell(
                    row,
                    quarter,
                    CellData(
                        row=row,
                        column=quarter,
                        value=(row + 1) * quarter * 1000,
                        data_type="number",
                    ),
                )

        # Table 2: Employee data (bottom-right with gap)
        start_row, start_col = 10, 8
        for col, header in enumerate(["EmpID", "Name", "Title", "Start Date"]):
            sheet.set_cell(
                start_row,
                start_col + col,
                CellData(
                    row=start_row,
                    column=start_col + col,
                    value=header,
                    data_type="text",
                    is_bold=True,
                ),
            )

        employees = [
            ["E001", "John Doe", "Manager", "2020-01-15"],
            ["E002", "Jane Smith", "Developer", "2021-03-20"],
            ["E003", "Mike Johnson", "Designer", "2022-07-10"],
        ]

        for row_idx, emp_data in enumerate(employees, 1):
            for col_idx, value in enumerate(emp_data):
                sheet.set_cell(
                    start_row + row_idx,
                    start_col + col_idx,
                    CellData(
                        row=start_row + row_idx,
                        column=start_col + col_idx,
                        value=value,
                        data_type="text",
                    ),
                )

        # Table 3: Summary stats (middle)
        start_row, start_col = 6, 0
        sheet.set_cell(
            start_row,
            start_col,
            CellData(
                row=start_row,
                column=start_col,
                value="Summary Statistics",
                data_type="text",
                is_bold=True,
                is_merged=True,
            ),
        )

        return sheet

    @pytest.fixture
    def large_sparse_sheet(self):
        """Create large sheet with sparse data patterns."""
        sheet = SheetData(name="LargeSparse")

        # Set large bounds
        sheet.max_row = 4999
        sheet.max_column = 499

        # Create sparse tables at different locations
        # Table 1: Top-left corner
        for row in range(10):
            for col in range(5):
                sheet.set_cell(
                    row,
                    col,
                    CellData(
                        row=row,
                        column=col,
                        value=f"TL_{row}_{col}",
                        data_type="text",
                        is_bold=(row == 0),
                    ),
                )

        # Table 2: Center
        center_row, center_col = 2500, 250
        for row in range(20):
            for col in range(10):
                sheet.set_cell(
                    center_row + row,
                    center_col + col,
                    CellData(
                        row=center_row + row,
                        column=center_col + col,
                        value=f"C_{row}_{col}",
                        data_type="number",
                    ),
                )

        # Table 3: Bottom-right
        br_row, br_col = 4900, 450
        for row in range(15):
            for col in range(8):
                sheet.set_cell(
                    br_row + row,
                    br_col + col,
                    CellData(
                        row=br_row + row,
                        column=br_col + col,
                        value=f"BR_{row}_{col}",
                        data_type="text",
                    ),
                )

        return sheet

    @pytest.fixture
    def excel_features_sheet(self):
        """Create sheet with Excel-specific features."""
        sheet = SheetData(name="ExcelFeatures")

        # Merged cells header
        sheet.set_cell(
            0,
            0,
            CellData(
                row=0,
                column=0,
                value="Financial Report 2024",
                data_type="text",
                is_bold=True,
                is_merged=True,
                background_color="#4472C4",
            ),
        )

        # Table with formulas
        headers = ["Item", "Quantity", "Unit Price", "Total"]
        for col, header in enumerate(headers):
            sheet.set_cell(
                2,
                col,
                CellData(
                    row=2,
                    column=col,
                    value=header,
                    data_type="text",
                    is_bold=True,
                    background_color="#D9E2F3",
                ),
            )

        # Data with formulas
        items = [
            ["Laptop", 5, 1200],
            ["Mouse", 20, 25],
            ["Keyboard", 15, 75],
        ]

        for row_idx, (item, qty, price) in enumerate(items, 3):
            sheet.set_cell(
                row_idx, 0, CellData(row=row_idx, column=0, value=item, data_type="text")
            )
            sheet.set_cell(
                row_idx, 1, CellData(row=row_idx, column=1, value=qty, data_type="number")
            )
            sheet.set_cell(
                row_idx, 2, CellData(row=row_idx, column=2, value=price, data_type="number")
            )
            sheet.set_cell(
                row_idx,
                3,
                CellData(
                    row=row_idx,
                    column=3,
                    value=f"=B{row_idx+1}*C{row_idx+1}",
                    data_type="formula",
                    has_formula=True,
                ),
            )

        # Summary row with formatting
        sheet.set_cell(
            7, 2, CellData(row=7, column=2, value="Total:", data_type="text", is_bold=True)
        )
        sheet.set_cell(
            7,
            3,
            CellData(
                row=7,
                column=3,
                value="=SUM(D4:D6)",
                data_type="formula",
                has_formula=True,
                is_bold=True,
                background_color="#FFD966",
            ),
        )

        return sheet

    def test_small_sheet_single_image_e2e(self, small_sheet_single_table):
        """Test end-to-end processing of small sheet."""
        builder = VisionRequestBuilder()
        request = builder.build_request(small_sheet_single_table, "SmallSingle")

        # Should use single image strategy
        assert request.prompt_template == "SINGLE_IMAGE"
        assert len(request.images) == 1

        # Image should have no compression
        image = request.images[0]
        assert image.compression_level == 0
        assert image.covers_cells == "A1:E6"

        # Verify prompt
        prompt = builder.create_explicit_prompt(request)
        assert "No compression applied" in image.description
        assert "Analyze this spreadsheet image" in prompt

    def test_medium_sheet_multi_scale_e2e(self, medium_sheet_multi_table):
        """Test end-to-end processing of medium sheet with multiple tables."""
        builder = VisionRequestBuilder()
        request = builder.build_request(medium_sheet_multi_table, "MediumMulti")

        # Should use multi-scale strategy
        assert request.prompt_template == "EXPLICIT_MULTI_SCALE"
        assert len(request.images) >= 2  # Overview + details

        # Check overview image
        overview = next((img for img in request.images if img.image_id == "overview"), None)
        assert overview is not None
        assert overview.compression_level > 0

        # Check detail images exist
        details = [img for img in request.images if img.image_id.startswith("detail_")]
        assert len(details) >= 1

        # Verify multi-scale prompt
        prompt = builder.create_explicit_prompt(request)
        assert "multiple scales" in prompt
        assert "Overview image" in prompt

    def test_large_sheet_progressive_e2e(self, large_sparse_sheet):
        """Test end-to-end processing of large sheet with progressive refinement."""
        builder = VisionRequestBuilder()
        request = builder.build_request(large_sparse_sheet, "LargeSparse")

        # Should use progressive strategy
        assert request.prompt_template == "PROGRESSIVE"

        # Should have phase-based images
        phase_images = [img for img in request.images if "phase" in img.image_id]
        assert len(phase_images) >= 1

        # First phase should use high compression
        phase1 = next((img for img in request.images if "phase1" in img.image_id), None)
        if phase1:
            assert phase1.compression_level >= CompressionLevel.LARGE.value

    def test_excel_features_handling(self, excel_features_sheet):
        """Test handling of Excel-specific features."""
        builder = VisionRequestBuilder()
        request = builder.build_request(excel_features_sheet, "ExcelFeatures")

        # Process the sheet
        assert request.total_images >= 1

        # Data regions should detect the formatted areas
        regions = builder.preprocessor.detect_data_regions(excel_features_sheet)
        assert len(regions) >= 1

        # Should identify headers and formatting
        main_region = regions[0]
        assert main_region.characteristics.get("likely_headers", False)
        assert main_region.characteristics.get("has_formatting", False)

    @pytest.mark.asyncio
    async def test_vision_orchestrator_integration(self, config, medium_sheet_multi_table):
        """Test integration with VisionOrchestratorAgent."""
        # Mock vision model since we're not actually calling APIs
        with patch("gridporter.vision.vision_models.create_vision_model") as mock_create:
            mock_model = MagicMock()
            mock_model.name = "mock-vision"
            mock_create.return_value = mock_model

            agent = VisionOrchestratorAgent(config)

            # Check that vision request builder was initialized
            assert hasattr(agent, "vision_request_builder")

            # Test cost estimation with new approach
            cost = agent._estimate_vision_cost(medium_sheet_multi_table)
            assert cost > 0  # Should estimate based on actual image sizes

    def test_compression_effectiveness(self):
        """Test that compression effectively reduces data size."""
        sheet = SheetData(name="CompressionTest")

        # Create large uniform pattern
        for row in range(1000):
            for col in range(100):
                if row % 10 == 0:  # Every 10th row
                    sheet.set_cell(
                        row, col, CellData(row=row, column=col, value="X", data_type="text")
                    )

        builder = VisionRequestBuilder()

        # Force no compression
        builder.bitmap_gen.auto_compress = False
        uncompressed_request = builder.build_request(sheet, "Uncompressed")

        # Force compression
        builder.bitmap_gen.auto_compress = True
        compressed_request = builder.build_request(sheet, "Compressed")

        # Compressed should be significantly smaller
        if compressed_request.total_images > 0 and uncompressed_request.total_images > 0:
            # If both generated images, compressed should be smaller
            assert compressed_request.total_size_mb <= uncompressed_request.total_size_mb

    def test_edge_case_single_cell(self):
        """Test edge case of single cell sheet."""
        sheet = SheetData(name="SingleCell")
        sheet.set_cell(0, 0, CellData(row=0, column=0, value="Only", data_type="text"))

        builder = VisionRequestBuilder()
        request = builder.build_request(sheet, "SingleCell")

        assert request.prompt_template == "SINGLE_IMAGE"
        assert request.total_images == 1
        assert request.images[0].compression_level == 0

    def test_data_pattern_detection(self):
        """Test that different data patterns are handled appropriately."""
        patterns = {
            "dense": lambda r, c: True,  # All cells filled
            "sparse": lambda r, c: (r + c) % 10 == 0,  # 10% filled
            "striped": lambda r, c: r % 5 == 0,  # Horizontal stripes
            "checkered": lambda r, c: (r + c) % 2 == 0,  # Checkerboard
        }

        builder = VisionRequestBuilder()

        for pattern_name, pattern_func in patterns.items():
            sheet = SheetData(name=f"Pattern_{pattern_name}")

            # Fill according to pattern
            for row in range(100):
                for col in range(50):
                    if pattern_func(row, col):
                        sheet.set_cell(
                            row,
                            col,
                            CellData(
                                row=row, column=col, value=pattern_name[0].upper(), data_type="text"
                            ),
                        )

            request = builder.build_request(sheet, f"Pattern_{pattern_name}")
            assert request.total_images >= 1

            # Different patterns might trigger different strategies
            # but all should produce valid requests
            assert request.validate_size_limit(20.0)
