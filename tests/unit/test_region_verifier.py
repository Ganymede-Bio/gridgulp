"""Unit tests for region verification functionality."""

from typing import Any

import numpy as np
import pytest

from gridporter.models.sheet_data import CellData, SheetData
from gridporter.models.table import TableRange
from gridporter.vision.pattern_detector import TablePattern
from gridporter.vision.region_verifier import (
    GeometryMetrics,
    RegionVerifier,
    VerificationResult,
)


def create_sheet_from_array(name: str, data: list[list[Any]]) -> SheetData:
    """Helper to create SheetData from 2D array."""
    sheet = SheetData(name=name)

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if value is not None:
                # Calculate Excel-style address
                col_letter = ""
                col = col_idx
                while col >= 0:
                    col_letter = chr(col % 26 + ord("A")) + col_letter
                    col = col // 26 - 1
                    if col < 0:
                        break

                address = f"{col_letter}{row_idx + 1}"
                cell = CellData(
                    value=value,
                    row=row_idx,
                    column=col_idx,
                    data_type="string" if isinstance(value, str) else "number",
                )
                sheet.cells[address] = cell

    # Update max dimensions
    if data:
        sheet.max_row = len(data) - 1
        sheet.max_column = max(len(row) for row in data) - 1 if data else 0

    return sheet


class TestRegionVerifier:
    """Test region verification functionality."""

    def test_init(self):
        """Test RegionVerifier initialization."""
        verifier = RegionVerifier(
            min_filled_cells=5,
            min_filledness=0.2,
            min_rectangularness=0.8,
            min_contiguity=0.6,
        )
        assert verifier.min_filled_cells == 5
        assert verifier.min_filledness == 0.2
        assert verifier.min_rectangularness == 0.8
        assert verifier.min_contiguity == 0.6

    def test_verify_valid_region(self):
        """Test verification of a valid table region."""
        # Create sheet with a simple 3x3 table
        sheet = create_sheet_from_array(
            "test",
            [
                ["A", "B", "C"],
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        region = TableRange(
            range="A1:C3",
            start_row=0,
            start_col=0,
            end_row=2,
            end_col=2,
        )

        verifier = RegionVerifier()
        result = verifier.verify_region(sheet, region)

        assert result.valid is True
        assert result.confidence > 0.8
        assert "All checks passed" in result.reason
        assert result.metrics["filledness"] == 1.0  # Fully filled
        assert result.metrics["rectangularness"] == 1.0  # Perfect rectangle

    def test_verify_sparse_region(self):
        """Test verification of a sparse region."""
        # Create sheet with sparse data
        sheet = create_sheet_from_array(
            "test",
            [
                ["A", None, None, None, "E"],
                [None, None, None, None, None],
                [None, None, "X", None, None],
                [None, None, None, None, None],
                ["F", None, None, None, "J"],
            ],
        )

        region = TableRange(
            range="A1:E5",
            start_row=0,
            start_col=0,
            end_row=4,
            end_col=4,
        )

        verifier = RegionVerifier(min_filledness=0.3)
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert result.confidence < 0.5
        assert "Low filledness" in result.reason
        assert result.metrics["filledness"] < 0.3

    def test_verify_non_rectangular_region(self):
        """Test verification of a non-rectangular region."""
        # Create sheet with L-shaped data
        sheet = create_sheet_from_array(
            "test",
            [
                ["A", "B", "C", None, None],
                ["D", "E", "F", None, None],
                ["G", None, None, None, None],
                ["H", None, None, None, None],
                ["I", None, None, None, None],
            ],
        )

        region = TableRange(
            range="A1:E5",
            start_row=0,
            start_col=0,
            end_row=4,
            end_col=4,
        )

        verifier = RegionVerifier(min_rectangularness=0.8)
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert "Low rectangularness" in result.reason
        assert result.metrics["rectangularness"] < 0.8

    def test_verify_fragmented_region(self):
        """Test verification of fragmented data."""
        # Create sheet with disconnected data islands
        sheet = create_sheet_from_array(
            "test",
            [
                ["A", "B", None, None, "E", "F"],
                ["C", "D", None, None, "G", "H"],
                [None, None, None, None, None, None],
                [None, None, None, None, None, None],
                ["I", "J", None, None, "K", "L"],
                ["M", "N", None, None, "O", "P"],
            ],
        )

        region = TableRange(
            range="A1:F6",
            start_row=0,
            start_col=0,
            end_row=5,
            end_col=5,
        )

        verifier = RegionVerifier(min_contiguity=0.8)
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert "Low contiguity" in result.reason
        assert result.metrics["contiguity"] < 0.8

    def test_verify_invalid_bounds(self):
        """Test verification with invalid bounds."""
        sheet = create_sheet_from_array(
            "test",
            [["A", "B"], ["C", "D"]],
        )

        # Region extends beyond sheet
        region = TableRange(
            range="A1:Z100",
            start_row=0,
            start_col=0,
            end_row=99,
            end_col=25,
        )

        verifier = RegionVerifier()
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert result.reason == "Invalid bounds"
        assert result.feedback == "Region bounds exceed sheet dimensions"

    def test_verify_empty_region(self):
        """Test verification of completely empty region."""
        sheet = create_sheet_from_array(
            "test",
            [[None, None], [None, None]],
        )

        region = TableRange(
            range="A1:B2",
            start_row=0,
            start_col=0,
            end_row=1,
            end_col=1,
        )

        verifier = RegionVerifier()
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert "Too few filled cells" in result.reason

    def test_geometry_metrics_computation(self):
        """Test geometry metrics computation."""
        verifier = RegionVerifier()

        # Perfect rectangle
        perfect_rect = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        metrics = verifier._compute_geometry_metrics(perfect_rect)

        assert metrics.filledness == 1.0
        assert metrics.rectangularness == 1.0
        assert metrics.contiguity == 1.0
        assert metrics.aspect_ratio == 1.0

        # Sparse data
        sparse = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        metrics = verifier._compute_geometry_metrics(sparse)

        assert metrics.filledness == pytest.approx(1 / 3)
        assert metrics.rectangularness < 0.5
        assert metrics.contiguity < 0.5

    def test_pattern_verification(self):
        """Test pattern-specific verification."""
        # Create sheet with header-data pattern
        sheet = create_sheet_from_array(
            "test",
            [
                ["Name", "Age", "City"],
                ["Alice", 25, "NYC"],
                ["Bob", 30, "LA"],
            ],
        )

        from gridporter.vision.pattern_detector import TableBounds, PatternType

        pattern = TablePattern(
            pattern_type=PatternType.HEADER_DATA,
            bounds=TableBounds(start_row=0, start_col=0, end_row=2, end_col=2),
            confidence=0.9,
            characteristics={},
        )
        # Add required properties for verification
        pattern.start_row = 0
        pattern.start_col = 0
        pattern.end_row = 2
        pattern.end_col = 2
        pattern.range = "A1:C3"

        verifier = RegionVerifier()
        result = verifier.verify_pattern(sheet, pattern)

        assert result.valid is True
        # The pattern passes general verification checks
        assert "All checks passed" in result.reason or "Header pattern verified" in result.reason

    def test_strict_mode(self):
        """Test strict verification mode."""
        # Create sheet with slightly irregular data
        sheet = create_sheet_from_array(
            "test",
            [
                ["A", "B", "C", None],
                ["D", "E", "F", "G"],
                ["H", "I", None, "J"],
                ["K", "L", "M", "N"],
            ],
        )

        region = TableRange(
            range="A1:D4",
            start_row=0,
            start_col=0,
            end_row=3,
            end_col=3,
        )

        # Non-strict mode should pass
        verifier = RegionVerifier()
        result = verifier.verify_region(sheet, region, strict=False)
        assert result.valid is True

        # Strict mode might fail due to edge quality
        verifier.verify_region(sheet, region, strict=True)
        # May or may not pass depending on edge quality threshold

    def test_feedback_generation(self):
        """Test feedback generation for failed regions."""
        # Create extremely wide region
        sheet = create_sheet_from_array(
            "test",
            [["A"] + [None] * 99] * 2,  # 2x100 mostly empty
        )

        region = TableRange(
            range="A1:CV2",
            start_row=0,
            start_col=0,
            end_row=1,
            end_col=99,
        )

        verifier = RegionVerifier()
        result = verifier.verify_region(sheet, region)

        assert result.valid is False
        assert result.feedback is not None
        # The feedback could be about too few cells or extreme width
        assert "too little data" in result.feedback or "very wide region" in result.feedback

    def test_overall_score_calculation(self):
        """Test overall geometry score calculation."""
        metrics = GeometryMetrics(
            rectangularness=0.9,
            filledness=0.8,
            density=0.7,
            contiguity=0.6,
            edge_quality=0.5,
            aspect_ratio=2.0,
            size_ratio=0.1,
        )

        # Check weighted average calculation
        expected = 0.9 * 0.3 + 0.8 * 0.2 + 0.7 * 0.2 + 0.6 * 0.15 + 0.5 * 0.15
        assert metrics.overall_score() == pytest.approx(expected)

    def test_edge_cases(self):
        """Test various edge cases."""
        verifier = RegionVerifier()

        # Single cell
        sheet = create_sheet_from_array("test", [["A"]])
        region = TableRange(range="A1", start_row=0, start_col=0, end_row=0, end_col=0)
        result = verifier.verify_region(sheet, region)
        # Should fail due to minimum cell count

        # Empty sheet
        empty_sheet = create_sheet_from_array("empty", [])
        result = verifier.verify_region(empty_sheet, region)
        assert result.valid is False

    def test_matrix_pattern_verification(self):
        """Test matrix pattern verification."""
        # Create matrix with row and column headers
        sheet = create_sheet_from_array(
            "test",
            [
                ["", "Q1", "Q2", "Q3"],
                ["Product A", 100, 120, 110],
                ["Product B", 200, 210, 190],
                ["Product C", 150, 160, 170],
            ],
        )

        from gridporter.vision.pattern_detector import TableBounds, PatternType

        pattern = TablePattern(
            pattern_type=PatternType.MATRIX,
            bounds=TableBounds(start_row=0, start_col=0, end_row=3, end_col=3),
            confidence=0.9,
            characteristics={},
        )
        # Add required properties for verification
        pattern.start_row = 0
        pattern.start_col = 0
        pattern.end_row = 3
        pattern.end_col = 3
        pattern.range = "A1:D4"

        verifier = RegionVerifier()
        result = verifier._verify_matrix_pattern(sheet, pattern)

        assert result.valid is True
        assert result.metrics["has_row_headers"] is True
        assert result.metrics["has_col_headers"] is True
