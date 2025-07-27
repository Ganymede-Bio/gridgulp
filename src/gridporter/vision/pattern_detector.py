"""Detect common table patterns in sparse spreadsheets."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..models.sheet_data import SheetData
from .bitmap_analyzer import TableOrientation

# Optional telemetry import
try:
    from ..telemetry.metrics import get_metrics_collector

    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

# Optional feature collection import
try:
    from ..telemetry import get_feature_collector

    HAS_FEATURE_COLLECTION = True
except ImportError:
    HAS_FEATURE_COLLECTION = False

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of table patterns that can be detected."""

    HEADER_DATA = "header_data"  # Continuous headers with sparse data below
    MATRIX = "matrix"  # Cross-reference table (headers on both axes)
    FORM = "form"  # Checklist/form pattern (full first column, sparse others)
    TIME_SERIES = "time_series"  # Time-based data with gaps
    SIMPLE_TABLE = "simple_table"  # Basic rectangular table
    HIERARCHICAL = "hierarchical"  # Indented/hierarchical data (e.g., financial statements)


@dataclass
class TableBounds:
    """Bounds of a detected table."""

    start_row: int
    start_col: int
    end_row: int
    end_col: int

    @property
    def total_cells(self) -> int:
        """Total number of cells in the bounded region."""
        return (self.end_row - self.start_row + 1) * (self.end_col - self.start_col + 1)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the bounded region (rows, cols)."""
        return (self.end_row - self.start_row + 1, self.end_col - self.start_col + 1)

    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is within these bounds."""
        return self.start_row <= row <= self.end_row and self.start_col <= col <= self.end_col

    def overlaps(self, other: "TableBounds") -> bool:
        """Check if this bounds overlaps with another."""
        return not (
            self.end_row < other.start_row
            or self.start_row > other.end_row
            or self.end_col < other.start_col
            or self.start_col > other.end_col
        )


@dataclass
class TablePattern:
    """Detected table pattern with metadata."""

    pattern_type: PatternType
    bounds: TableBounds
    confidence: float
    header_rows: list[int] = None
    header_cols: list[int] = None
    characteristics: dict[str, Any] = None
    orientation: TableOrientation = TableOrientation.UNKNOWN
    multi_row_headers: bool = False
    header_end_row: int = None

    def __post_init__(self):
        if self.header_rows is None:
            self.header_rows = []
        if self.header_cols is None:
            self.header_cols = []
        if self.characteristics is None:
            self.characteristics = {}
        # Auto-detect multi-row headers
        if len(self.header_rows) > 1:
            self.multi_row_headers = True
            self.header_end_row = max(self.header_rows) if self.header_rows else None


class SparsePatternDetector:
    """Detect common table patterns in sparse spreadsheets."""

    def __init__(
        self,
        min_filled_ratio: float = 0.1,
        min_table_size: tuple[int, int] = (2, 2),
        header_density_threshold: float = 0.5,
    ):
        """Initialize pattern detector.

        Args:
            min_filled_ratio: Minimum ratio of filled cells to consider a region
            min_table_size: Minimum (rows, cols) to consider as a table
            header_density_threshold: Minimum fill density to consider as header
        """
        self.min_filled_ratio = min_filled_ratio
        self.min_table_size = min_table_size
        self.header_density_threshold = header_density_threshold

    def detect_patterns(self, sheet_data: SheetData) -> list[TablePattern]:
        """Detect all table patterns in the sheet.

        Args:
            sheet_data: Sheet data to analyze (SheetData or PolarsSheetData)

        Returns:
            List of detected table patterns
        """
        # Track timing if telemetry available
        if HAS_TELEMETRY:
            metrics = get_metrics_collector()
            start_time = time.time()
        patterns = []

        # Try different pattern detection strategies
        patterns.extend(self._detect_header_data_patterns(sheet_data))
        patterns.extend(self._detect_matrix_patterns(sheet_data))
        patterns.extend(self._detect_form_patterns(sheet_data))
        patterns.extend(self._detect_time_series_patterns(sheet_data))
        patterns.extend(self._detect_hierarchical_patterns(sheet_data))

        # Merge overlapping patterns
        patterns = self._merge_overlapping_patterns(patterns)

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        logger.info(f"Detected {len(patterns)} table patterns")

        if HAS_TELEMETRY:
            duration = time.time() - start_time
            metrics.record_duration("pattern_detection", duration)
            metrics.record_value("patterns_detected", len(patterns))

        # Record features if collection is enabled
        if HAS_FEATURE_COLLECTION:
            try:
                feature_collector = get_feature_collector()
                if feature_collector.enabled:
                    for pattern in patterns:
                        # Build pattern features
                        pattern_features = {
                            "pattern_type": pattern.pattern_type.value,
                            "orientation": pattern.orientation.value
                            if pattern.orientation
                            else None,
                            "has_multi_headers": len(pattern.header_rows) > 1
                            if pattern.header_rows
                            else False,
                            "header_row_count": len(pattern.header_rows)
                            if pattern.header_rows
                            else 0,
                            "fill_ratio": pattern.characteristics.get("fill_ratio")
                            if pattern.characteristics
                            else None,
                            "header_density": pattern.characteristics.get("header_density")
                            if pattern.characteristics
                            else None,
                        }

                        # Build content features
                        content_features = {
                            "total_cells": pattern.bounds.total_cells,
                            "aspect_ratio": pattern.characteristics.get("aspect_ratio")
                            if pattern.characteristics
                            else None,
                        }

                        # Record the detection
                        feature_collector.record_detection(
                            file_path=getattr(sheet_data, "file_path", "unknown"),
                            file_type=getattr(sheet_data, "file_type", "unknown"),
                            sheet_name=getattr(sheet_data, "name", None),
                            table_range=pattern.range,
                            detection_method="pattern_detection",
                            confidence=pattern.confidence,
                            success=True,
                            pattern_features=pattern_features,
                            content_features=content_features,
                            processing_time_ms=int(duration * 1000) if HAS_TELEMETRY else None,
                        )
            except Exception as e:
                logger.debug(f"Failed to record pattern features: {e}")

        return patterns

    def detect_patterns_in_region(
        self,
        sheet_data: SheetData,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
    ) -> list[TablePattern]:
        """Detect patterns within a specific region of the sheet.

        Args:
            sheet_data: Sheet data to analyze (SheetData or PolarsSheetData)
            start_row: Starting row index
            end_row: Ending row index
            start_col: Starting column index
            end_col: Ending column index

        Returns:
            List of detected patterns within the region
        """
        # Create a view of the sheet data for the region
        # Note: This is a conceptual implementation - actual implementation
        # would need to handle cell access within bounds

        patterns = []

        # Adjust detection to work within region bounds
        # Find dense rows within region
        for row in range(start_row, min(end_row + 1, sheet_data.max_row + 1)):
            filled_count = 0
            total_count = 0

            for col in range(start_col, min(end_col + 1, sheet_data.max_column + 1)):
                if sheet_data.get_cell(row, col) and not sheet_data.get_cell(row, col).is_empty:
                    filled_count += 1
                total_count += 1

            # If this row is dense, check if it could be a header
            if total_count > 0 and filled_count / total_count >= self.header_density_threshold:
                # Try to detect a pattern starting from this row
                data_end_row = self._find_data_end_row(sheet_data, row + 1, start_col, end_col)

                if data_end_row and data_end_row - row >= self.min_table_size[0]:
                    bounds = TableBounds(
                        start_row=row,
                        start_col=start_col,
                        end_row=min(data_end_row, end_row),
                        end_col=end_col,
                    )

                    fill_ratio = self._calculate_fill_ratio(sheet_data, bounds)
                    if fill_ratio >= self.min_filled_ratio:
                        aspect_ratio = bounds.shape[1] / bounds.shape[0]
                        orientation = (
                            TableOrientation.HORIZONTAL
                            if aspect_ratio > 1.0
                            else TableOrientation.VERTICAL
                        )

                        pattern = TablePattern(
                            pattern_type=PatternType.HEADER_DATA,
                            bounds=bounds,
                            confidence=self._calculate_pattern_confidence(sheet_data, bounds, row),
                            header_rows=[row],
                            characteristics={
                                "fill_ratio": fill_ratio,
                                "has_headers": True,
                                "aspect_ratio": aspect_ratio,
                            },
                            orientation=orientation,
                        )
                        patterns.append(pattern)

        # Also check for hierarchical patterns in this region
        from .hierarchical_detector import HierarchicalPatternDetector

        hierarchical_detector = HierarchicalPatternDetector()

        region_bounds = TableBounds(
            start_row=start_row, end_row=end_row, start_col=start_col, end_col=end_col
        )
        hierarchical_patterns = hierarchical_detector.detect_hierarchical_patterns(
            sheet_data, bounds=region_bounds
        )
        patterns.extend(hierarchical_patterns)

        return patterns

    def _detect_header_data_patterns(self, sheet_data: SheetData) -> list[TablePattern]:
        """Detect tables with continuous headers but sparse data.

        Pattern:
        Name  | Age | City    <- Dense header row
        Alice | 25  |         <- Sparse data
        Bob   |     | London  <- Sparse data

        Also supports multi-row headers:
        Department | Sales         | Support       <- Row 1
                   | Q1   | Q2     | Q1   | Q2     <- Row 2
        """
        patterns = []

        # Find rows with high fill density (potential headers)
        dense_rows = self._find_dense_rows(sheet_data)

        for header_row in dense_rows:
            # Get column range of the header
            header_start_col, header_end_col = self._get_row_bounds(sheet_data, header_row)

            if header_start_col is None:
                continue

            # Check for multi-row headers
            header_rows = [header_row]
            header_end_row = header_row

            # Look for additional header rows
            if self._has_merged_cells_in_row(
                sheet_data, header_row, header_start_col, header_end_col
            ):
                # Likely multi-row header, check subsequent rows
                for next_row in range(header_row + 1, min(header_row + 5, sheet_data.max_row + 1)):
                    if self._is_likely_header_row(
                        sheet_data, next_row, header_start_col, header_end_col
                    ):
                        header_rows.append(next_row)
                        header_end_row = next_row
                    else:
                        break

            # Look for data below the header(s)
            data_end_row = self._find_data_end_row(
                sheet_data, header_end_row + 1, header_start_col, header_end_col
            )

            if data_end_row is None or data_end_row - header_end_row < self.min_table_size[0]:
                continue

            # Check if there's enough data
            bounds = TableBounds(
                start_row=header_row,
                start_col=header_start_col,
                end_row=data_end_row,
                end_col=header_end_col,
            )

            fill_ratio = self._calculate_fill_ratio(sheet_data, bounds)
            if fill_ratio >= self.min_filled_ratio:
                # Determine orientation based on aspect ratio
                aspect_ratio = (bounds.end_col - bounds.start_col + 1) / (
                    bounds.end_row - bounds.start_row + 1
                )
                orientation = (
                    TableOrientation.HORIZONTAL if aspect_ratio > 1.0 else TableOrientation.VERTICAL
                )

                pattern = TablePattern(
                    pattern_type=PatternType.HEADER_DATA,
                    bounds=bounds,
                    confidence=self._calculate_pattern_confidence(sheet_data, bounds, header_row),
                    header_rows=header_rows,
                    characteristics={
                        "fill_ratio": fill_ratio,
                        "has_headers": True,
                        "aspect_ratio": aspect_ratio,
                        "multi_row_headers": len(header_rows) > 1,
                    },
                    orientation=orientation,
                )
                patterns.append(pattern)
                logger.debug(
                    f"Found header-data pattern at {bounds} with {len(header_rows)} header rows"
                )

        return patterns

    def _detect_matrix_patterns(self, sheet_data: SheetData) -> list[TablePattern]:
        """Detect cross-reference/matrix patterns.

        Pattern:
              | Col1 | Col2 | Col3
        ------|------|------|------
        Row1  |  X   |      |  X
        Row2  |      |  X   |
        """
        patterns = []

        # Find potential header rows and columns
        dense_rows = self._find_dense_rows(sheet_data)
        dense_cols = self._find_dense_columns(sheet_data)

        for header_row in dense_rows[:3]:  # Check top few rows
            for header_col in dense_cols[:3]:  # Check leftmost columns
                # Check if this could be a matrix intersection
                if not self._is_matrix_corner(sheet_data, header_row, header_col):
                    continue

                # Find the extent of the matrix
                row_end = self._find_matrix_row_end(sheet_data, header_row, header_col)
                col_end = self._find_matrix_col_end(sheet_data, header_row, header_col)

                if row_end is None or col_end is None:
                    continue

                bounds = TableBounds(
                    start_row=header_row,
                    start_col=header_col,
                    end_row=row_end,
                    end_col=col_end,
                )

                if (
                    bounds.shape[0] >= self.min_table_size[0]
                    and bounds.shape[1] >= self.min_table_size[1]
                ):
                    pattern = TablePattern(
                        pattern_type=PatternType.MATRIX,
                        bounds=bounds,
                        confidence=self._calculate_matrix_confidence(sheet_data, bounds),
                        header_rows=[header_row],
                        header_cols=[header_col],
                        characteristics={"is_sparse_matrix": True},
                        orientation=TableOrientation.MATRIX,
                    )
                    patterns.append(pattern)
                    logger.debug(f"Found matrix pattern at {bounds}")

        return patterns

    def _detect_form_patterns(self, sheet_data: SheetData) -> list[TablePattern]:
        """Detect form/checklist patterns.

        Pattern:
        Task          | Status | Date
        --------------|--------|------
        Setup env     |   ✓    | 1/1
        Install deps  |        |
        Run tests     |   ✓    | 1/3
        """
        patterns = []

        # Look for columns with high fill density (typically first column)
        dense_cols = self._find_dense_columns(sheet_data)

        for label_col in dense_cols[:2]:  # Check first few columns
            # Find the vertical extent
            col_start_row, col_end_row = self._get_column_bounds(sheet_data, label_col)

            if col_start_row is None:
                continue

            # Look for sparse data to the right
            data_end_col = self._find_form_data_end_col(
                sheet_data, col_start_row, col_end_row, label_col + 1
            )

            if data_end_col is None or data_end_col - label_col < self.min_table_size[1]:
                continue

            bounds = TableBounds(
                start_row=col_start_row,
                start_col=label_col,
                end_row=col_end_row,
                end_col=data_end_col,
            )

            # Forms are typically vertically oriented
            pattern = TablePattern(
                pattern_type=PatternType.FORM,
                bounds=bounds,
                confidence=self._calculate_form_confidence(sheet_data, bounds, label_col),
                header_cols=[label_col],
                characteristics={"label_column": label_col, "is_checklist": True},
                orientation=TableOrientation.VERTICAL,
            )
            patterns.append(pattern)
            logger.debug(f"Found form pattern at {bounds}")

        return patterns

    def _detect_time_series_patterns(self, _sheet_data: SheetData) -> list[TablePattern]:
        """Detect time series patterns with gaps.

        Pattern:
        Date     | Value1 | Value2
        ---------|--------|--------
        2024-01  |  100   |
        2024-02  |        |
        2024-03  |  150   |  25
        """
        # For now, use header-data detection as time series often have similar structure
        # Could be enhanced with date detection in first column
        return []

    def _detect_hierarchical_patterns(self, sheet_data: SheetData) -> list[TablePattern]:
        """Detect hierarchical patterns using indentation.

        Pattern:
        Revenue              |  1000
          Product Sales      |   800
            Hardware       |   500
            Software       |   300
          Services           |   200
        Total Revenue        |  1000
        """
        # Use the dedicated hierarchical detector
        from .hierarchical_detector import HierarchicalPatternDetector

        hierarchical_detector = HierarchicalPatternDetector()
        return hierarchical_detector.detect_hierarchical_patterns(sheet_data)

    def _find_dense_rows(self, sheet_data: SheetData) -> list[int]:
        """Find rows with high cell fill density."""
        dense_rows = []

        for row in range(sheet_data.max_row + 1):
            filled_count = 0
            total_count = 0

            for col in range(min(sheet_data.max_column + 1, 100)):  # Check first 100 cols
                if sheet_data.get_cell(row, col) and not sheet_data.get_cell(row, col).is_empty:
                    filled_count += 1
                total_count += 1

            if total_count > 0 and filled_count / total_count >= self.header_density_threshold:
                dense_rows.append(row)

        return dense_rows

    def _find_dense_columns(self, sheet_data: SheetData) -> list[int]:
        """Find columns with high cell fill density."""
        dense_cols = []

        for col in range(min(sheet_data.max_column + 1, 50)):  # Check first 50 cols
            filled_count = 0
            total_count = 0

            for row in range(min(sheet_data.max_row + 1, 100)):  # Check first 100 rows
                if sheet_data.get_cell(row, col) and not sheet_data.get_cell(row, col).is_empty:
                    filled_count += 1
                total_count += 1

            if total_count > 0 and filled_count / total_count >= self.header_density_threshold:
                dense_cols.append(col)

        return dense_cols

    def _get_row_bounds(self, sheet_data: SheetData, row: int) -> tuple[int | None, int | None]:
        """Get the start and end column indices for non-empty cells in a row."""
        start_col = None
        end_col = None

        for col in range(sheet_data.max_column + 1):
            cell = sheet_data.get_cell(row, col)
            if cell and not cell.is_empty:
                if start_col is None:
                    start_col = col
                end_col = col

        return start_col, end_col

    def _get_column_bounds(self, sheet_data: SheetData, col: int) -> tuple[int | None, int | None]:
        """Get the start and end row indices for non-empty cells in a column."""
        start_row = None
        end_row = None

        for row in range(sheet_data.max_row + 1):
            cell = sheet_data.get_cell(row, col)
            if cell and not cell.is_empty:
                if start_row is None:
                    start_row = row
                end_row = row

        return start_row, end_row

    def _find_data_end_row(
        self, sheet_data: SheetData, start_row: int, start_col: int, end_col: int
    ) -> int | None:
        """Find the last row with data in the given column range."""
        last_data_row = None
        consecutive_empty = 0
        max_consecutive_empty = 5  # Allow up to 5 empty rows

        for row in range(start_row, min(sheet_data.max_row + 1, start_row + 1000)):
            has_data = False

            for col in range(start_col, end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    has_data = True
                    break

            if has_data:
                last_data_row = row
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty > max_consecutive_empty:
                    break

        return last_data_row

    def _find_form_data_end_col(
        self, sheet_data: SheetData, start_row: int, end_row: int, start_col: int
    ) -> int | None:
        """Find the last column with data in a form pattern."""
        last_data_col = None

        for col in range(start_col, min(sheet_data.max_column + 1, start_col + 50)):
            has_data = False

            for row in range(start_row, end_row + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    has_data = True
                    break

            if has_data:
                last_data_col = col

        return last_data_col

    def _find_matrix_row_end(
        self, sheet_data: SheetData, header_row: int, header_col: int
    ) -> int | None:
        """Find the end row of a matrix pattern."""
        # Look down the header column for the extent
        last_row = header_row

        for row in range(header_row + 1, min(sheet_data.max_row + 1, header_row + 100)):
            cell = sheet_data.get_cell(row, header_col)
            if cell and not cell.is_empty:
                last_row = row

        return last_row if last_row > header_row else None

    def _find_matrix_col_end(
        self, sheet_data: SheetData, header_row: int, header_col: int
    ) -> int | None:
        """Find the end column of a matrix pattern."""
        # Look across the header row for the extent
        last_col = header_col

        for col in range(header_col + 1, min(sheet_data.max_column + 1, header_col + 100)):
            cell = sheet_data.get_cell(header_row, col)
            if cell and not cell.is_empty:
                last_col = col

        return last_col if last_col > header_col else None

    def _is_matrix_corner(self, sheet_data: SheetData, row: int, col: int) -> bool:
        """Check if this position could be the corner of a matrix."""
        # Corner cell is often empty or contains a label
        sheet_data.get_cell(row, col)

        # Check if there's data to the right and below
        has_row_headers = False
        has_col_headers = False

        # Check for headers to the right
        for c in range(col + 1, min(col + 5, sheet_data.max_column + 1)):
            if sheet_data.get_cell(row, c) and not sheet_data.get_cell(row, c).is_empty:
                has_col_headers = True
                break

        # Check for headers below
        for r in range(row + 1, min(row + 5, sheet_data.max_row + 1)):
            if sheet_data.get_cell(r, col) and not sheet_data.get_cell(r, col).is_empty:
                has_row_headers = True
                break

        return has_row_headers and has_col_headers

    def _calculate_fill_ratio(self, sheet_data: SheetData, bounds: TableBounds) -> float:
        """Calculate the ratio of filled cells in the bounded region."""
        filled_count = 0
        total_count = 0

        for row in range(bounds.start_row, bounds.end_row + 1):
            for col in range(bounds.start_col, bounds.end_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    filled_count += 1
                total_count += 1

        return filled_count / total_count if total_count > 0 else 0.0

    def _calculate_pattern_confidence(
        self, sheet_data: SheetData, bounds: TableBounds, header_row: int
    ) -> float:
        """Calculate confidence score for a header-data pattern."""
        confidence = 0.5  # Base confidence

        # Check header density
        header_fill = 0
        for col in range(bounds.start_col, bounds.end_col + 1):
            if (
                sheet_data.get_cell(header_row, col)
                and not sheet_data.get_cell(header_row, col).is_empty
            ):
                header_fill += 1

        header_density = header_fill / (bounds.end_col - bounds.start_col + 1)
        confidence += header_density * 0.3

        # Check for bold headers (indicates intentional headers)
        bold_count = 0
        for col in range(bounds.start_col, bounds.end_col + 1):
            cell = sheet_data.get_cell(header_row, col)
            if cell and cell.is_bold:
                bold_count += 1

        if bold_count > 0:
            confidence += 0.2

        return min(confidence, 1.0)

    def _calculate_matrix_confidence(self, _sheet_data: SheetData, _bounds: TableBounds) -> float:
        """Calculate confidence score for a matrix pattern."""
        # Simple confidence based on having both row and column headers
        return 0.7

    def _calculate_form_confidence(
        self, sheet_data: SheetData, bounds: TableBounds, label_col: int
    ) -> float:
        """Calculate confidence score for a form pattern."""
        confidence = 0.6  # Base confidence

        # Check label column density
        label_fill = 0
        for row in range(bounds.start_row, bounds.end_row + 1):
            if (
                sheet_data.get_cell(row, label_col)
                and not sheet_data.get_cell(row, label_col).is_empty
            ):
                label_fill += 1

        label_density = label_fill / (bounds.end_row - bounds.start_row + 1)
        confidence += label_density * 0.3

        return min(confidence, 1.0)

    def _has_merged_cells_in_row(
        self, sheet_data: SheetData, row: int, start_col: int, end_col: int
    ) -> bool:
        """Check if a row contains merged cells."""
        for col in range(start_col, end_col + 1):
            cell = sheet_data.get_cell(row, col)
            if cell and hasattr(cell, "is_merged") and cell.is_merged:
                return True
        return False

    def _is_likely_header_row(
        self, sheet_data: SheetData, row: int, start_col: int, end_col: int
    ) -> bool:
        """Check if a row is likely to be a header row."""
        # Count filled cells
        filled = 0
        total = 0
        has_formatting = False

        for col in range(start_col, end_col + 1):
            cell = sheet_data.get_cell(row, col)
            total += 1
            if cell and not cell.is_empty:
                filled += 1
                # Check for header-like formatting
                if cell.is_bold or cell.background_color:
                    has_formatting = True

        # High density or formatting suggests header
        fill_ratio = filled / total if total > 0 else 0
        return fill_ratio >= self.header_density_threshold or has_formatting

    def _merge_overlapping_patterns(self, patterns: list[TablePattern]) -> list[TablePattern]:
        """Merge overlapping patterns, keeping the highest confidence one."""
        if not patterns:
            return []

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        merged = []
        for pattern in patterns:
            # Check if this pattern overlaps with any already merged
            overlaps = False
            for merged_pattern in merged:
                if pattern.bounds.overlaps(merged_pattern.bounds):
                    overlaps = True
                    break

            if not overlaps:
                merged.append(pattern)

        return merged
