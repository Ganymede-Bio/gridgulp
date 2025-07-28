"""Unit tests for DataRegionPreprocessor."""

import numpy as np
import pytest

from gridporter.models.sheet_data import CellData, SheetData
from gridporter.vision.data_region_preprocessor import DataRegionPreprocessor


class TestDataRegionPreprocessor:
    """Test DataRegionPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataRegionPreprocessor(gap_threshold=3, min_region_size=4)

    @pytest.fixture
    def empty_sheet(self):
        """Create empty sheet."""
        return SheetData(name="EmptySheet")

    @pytest.fixture
    def single_region_sheet(self):
        """Create sheet with single contiguous region."""
        sheet = SheetData(name="SingleRegion")

        # Create a 5x5 table with headers
        for col in range(5):
            sheet.set_cell(
                0,
                col,
                CellData(row=0, column=col, value=f"Header{col}", data_type="text", is_bold=True),
            )

        # Add data rows
        for row in range(1, 5):
            for col in range(5):
                sheet.set_cell(
                    row,
                    col,
                    CellData(row=row, column=col, value=f"Data_{row}_{col}", data_type="text"),
                )

        return sheet

    @pytest.fixture
    def multi_region_sheet(self):
        """Create sheet with multiple separated regions."""
        sheet = SheetData(name="MultiRegion")

        # Region 1: Top-left (0,0) to (3,3)
        for row in range(4):
            for col in range(4):
                sheet.set_cell(
                    row,
                    col,
                    CellData(
                        row=row,
                        column=col,
                        value=f"R1_{row}_{col}",
                        data_type="text",
                        is_bold=(row == 0),
                    ),
                )

        # Region 2: Bottom-right (10,10) to (13,12)
        for row in range(10, 14):
            for col in range(10, 13):
                sheet.set_cell(
                    row,
                    col,
                    CellData(
                        row=row,
                        column=col,
                        value=f"R2_{row}_{col}",
                        data_type="number" if row > 10 else "text",
                        is_bold=(row == 10),
                    ),
                )

        # Region 3: Middle (6,0) to (7,15) - wide but short
        for row in range(6, 8):
            for col in range(16):
                if col % 3 == 0:  # Sparse data
                    sheet.set_cell(
                        row,
                        col,
                        CellData(row=row, column=col, value=f"R3_{row}_{col}", data_type="text"),
                    )

        return sheet

    @pytest.fixture
    def sparse_sheet(self):
        """Create very sparse sheet with scattered data."""
        sheet = SheetData(name="SparseSheet")

        # Scattered individual cells
        positions = [(0, 0), (5, 5), (10, 10), (15, 15), (20, 20)]
        for row, col in positions:
            sheet.set_cell(
                row,
                col,
                CellData(row=row, column=col, value=f"Sparse_{row}_{col}", data_type="text"),
            )

        return sheet

    def test_empty_sheet_detection(self, preprocessor, empty_sheet):
        """Test detection on empty sheet."""
        regions = preprocessor.detect_data_regions(empty_sheet)
        assert len(regions) == 0

    def test_single_region_detection(self, preprocessor, single_region_sheet):
        """Test detection of single contiguous region."""
        regions = preprocessor.detect_data_regions(single_region_sheet)

        assert len(regions) == 1
        region = regions[0]

        # Check bounds
        assert region.bounds["top"] == 0
        assert region.bounds["left"] == 0
        assert region.bounds["bottom"] == 4
        assert region.bounds["right"] == 4

        # Check metrics
        assert region.cell_count == 25
        assert region.density == 1.0  # All cells filled
        assert region.total_cells == 25

        # Check characteristics
        assert region.characteristics["likely_headers"] is True
        assert region.characteristics["has_formatting"] is True
        assert region.skip is False

    def test_multi_region_detection(self, preprocessor, multi_region_sheet):
        """Test detection of multiple separated regions."""
        regions = preprocessor.detect_data_regions(multi_region_sheet)

        # Should detect at least 2 main regions (R1 and R2)
        assert len(regions) >= 2

        # Sort by top-left position
        regions.sort(key=lambda r: (r.bounds["top"], r.bounds["left"]))

        # Check first region (R1)
        r1 = regions[0]
        assert r1.bounds["top"] == 0
        assert r1.bounds["left"] == 0
        assert r1.bounds["bottom"] == 3
        assert r1.bounds["right"] == 3
        assert r1.characteristics["likely_headers"] is True

        # Check second region (R2)
        r2 = regions[1]
        assert r2.bounds["top"] == 10
        assert r2.bounds["left"] == 10
        assert r2.bounds["bottom"] == 13
        assert r2.bounds["right"] == 12
        assert r2.characteristics["likely_headers"] is True
        assert r2.characteristics["mostly_numbers"] is False  # Has mixed types

    def test_gap_threshold(self):
        """Test different gap thresholds affect region merging."""
        sheet = SheetData(name="GapTest")

        # Create two regions with 4-cell gap
        # Region 1
        for row in range(3):
            for col in range(3):
                sheet.set_cell(row, col, CellData(row=row, column=col, value="A", data_type="text"))

        # Region 2 (5 cells away)
        for row in range(8, 11):
            for col in range(3):
                sheet.set_cell(row, col, CellData(row=row, column=col, value="B", data_type="text"))

        # With gap_threshold=3, should be separate
        preprocessor1 = DataRegionPreprocessor(gap_threshold=3)
        regions1 = preprocessor1.detect_data_regions(sheet)
        assert len(regions1) == 2

        # With gap_threshold=5, should merge
        preprocessor2 = DataRegionPreprocessor(gap_threshold=5)
        regions2 = preprocessor2.detect_data_regions(sheet)
        assert len(regions2) == 1

    def test_min_region_size(self):
        """Test minimum region size filtering."""
        sheet = SheetData(name="MinSizeTest")

        # Small region (2x2)
        for row in range(2):
            for col in range(2):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="Small", data_type="text")
                )

        # Large region (5x5)
        for row in range(10, 15):
            for col in range(5):
                sheet.set_cell(
                    row, col, CellData(row=row, column=col, value="Large", data_type="text")
                )

        # With min_region_size=4, only large region detected
        preprocessor1 = DataRegionPreprocessor(min_region_size=5)
        regions1 = preprocessor1.detect_data_regions(sheet)
        assert len(regions1) == 1
        assert regions1[0].bounds["top"] == 10

        # With min_region_size=2, both detected
        preprocessor2 = DataRegionPreprocessor(min_region_size=2)
        regions2 = preprocessor2.detect_data_regions(sheet)
        assert len(regions2) == 2

    def test_sparse_sheet_handling(self, preprocessor, sparse_sheet):
        """Test handling of very sparse sheets."""
        # Default threshold might filter out scattered cells
        regions = preprocessor.detect_data_regions(sparse_sheet)

        # With scattered cells and min_region_size=4, might get 0 or few regions
        assert len(regions) <= 2

    def test_characteristic_detection(self, preprocessor):
        """Test detection of region characteristics."""
        sheet = SheetData(name="CharTest")

        # Region with headers and mixed data
        headers = ["Name", "Age", "Salary", "Date"]
        for col, header in enumerate(headers):
            sheet.set_cell(
                0, col, CellData(row=0, column=col, value=header, data_type="text", is_bold=True)
            )

        # Data rows
        sheet.set_cell(1, 0, CellData(row=1, column=0, value="Alice", data_type="text"))
        sheet.set_cell(1, 1, CellData(row=1, column=1, value=25, data_type="number"))
        sheet.set_cell(1, 2, CellData(row=1, column=2, value=50000, data_type="number"))
        sheet.set_cell(1, 3, CellData(row=1, column=3, value="2024-01-01", data_type="date"))

        regions = preprocessor.detect_data_regions(sheet)
        assert len(regions) == 1

        char = regions[0].characteristics
        assert char["likely_headers"] is True
        assert char["mixed_types"] is True
        assert char["has_formatting"] is True
        assert char["likely_data"] is True

    def test_region_density_calculation(self, preprocessor):
        """Test density calculation for regions."""
        sheet = SheetData(name="DensityTest")

        # Create 10x10 region with 50% density
        for row in range(10):
            for col in range(10):
                if (row + col) % 2 == 0:  # Checkerboard pattern
                    sheet.set_cell(
                        row, col, CellData(row=row, column=col, value="X", data_type="text")
                    )

        regions = preprocessor.detect_data_regions(sheet)
        assert len(regions) == 1

        # Due to dilation, bounds might be exact
        region = regions[0]
        assert region.cell_count == 50
        assert region.density == pytest.approx(0.5, rel=0.1)

    def test_never_skip_data_regions(self, preprocessor):
        """Test that regions with data are never skipped."""
        sheet = SheetData(name="NoSkipTest")

        # Create several small regions
        positions = [
            (0, 0, 1, 1),  # 2x2
            (5, 5, 6, 6),  # 2x2
            (10, 10, 11, 11),  # 2x2
        ]

        for top, left, bottom, right in positions:
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    sheet.set_cell(
                        row, col, CellData(row=row, column=col, value="Data", data_type="text")
                    )

        # Even with higher min_region_size, should not skip
        preprocessor_strict = DataRegionPreprocessor(min_region_size=10)
        regions = preprocessor_strict.detect_data_regions(sheet)

        # All regions should be detected, none skipped
        for region in regions:
            assert region.skip is False
            assert region.cell_count > 0

    def test_performance_large_sheet(self, preprocessor):
        """Test performance with large sheet."""
        sheet = SheetData(name="LargeSheet")

        # Create 1000x100 sheet with sparse data
        for row in range(0, 1000, 10):
            for col in range(0, 100, 5):
                sheet.set_cell(
                    row,
                    col,
                    CellData(row=row, column=col, value=f"L_{row}_{col}", data_type="text"),
                )

        # Should complete quickly
        import time

        start = time.time()
        regions = preprocessor.detect_data_regions(sheet)
        duration = time.time() - start

        assert duration < 5.0  # Should complete within 5 seconds
        assert len(regions) > 0

    def test_create_data_mask(self, preprocessor, single_region_sheet):
        """Test internal data mask creation."""
        mask = preprocessor._create_data_mask(single_region_sheet)

        assert mask.shape == (5, 5)  # Based on max_row/col
        assert mask.dtype == bool
        assert np.sum(mask) == 25  # All cells filled

        # Check specific positions
        assert mask[0, 0] is True  # Header
        assert mask[4, 4] is True  # Last data cell

    def test_region_merging(self, preprocessor):
        """Test _merge_nearby_regions functionality."""
        regions = [
            {"top": 0, "left": 0, "bottom": 5, "right": 5},
            {"top": 0, "left": 6, "bottom": 5, "right": 10},  # Adjacent horizontally
            {"top": 10, "left": 0, "bottom": 15, "right": 5},  # Far vertically
        ]

        merged = preprocessor._merge_nearby_regions(regions)

        # First two should merge (adjacent), third stays separate
        assert len(merged) == 2

        # Check merged region encompasses both originals
        assert merged[0]["left"] == 0
        assert merged[0]["right"] == 10
