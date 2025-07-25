"""Detect hierarchical patterns in spreadsheets (e.g., financial statements with indentation)."""

import logging
from dataclasses import dataclass
from typing import Any

from ..models.sheet_data import CellData, SheetData
from .bitmap_analyzer import TableOrientation
from .pattern_detector import PatternType, TableBounds, TablePattern

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalLevel:
    """Represents a level in the hierarchy."""

    row: int
    indentation_level: int
    is_header: bool = False
    is_subtotal: bool = False
    children: list[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class HierarchicalStructure:
    """Complete hierarchical structure of a table."""

    levels: dict[int, HierarchicalLevel]  # row -> level info
    root_rows: list[int]  # Top-level rows (indentation = 0)
    subtotal_rows: list[int]  # Rows containing subtotals
    max_depth: int  # Maximum indentation depth


class HierarchicalPatternDetector:
    """Detect hierarchical patterns in spreadsheet data."""

    def __init__(
        self,
        min_indent_consistency: float = 0.7,
        min_hierarchical_rows: int = 3,
        subtotal_keywords: list[str] | None = None,
    ):
        """Initialize hierarchical detector.

        Args:
            min_indent_consistency: Minimum ratio of rows following indent pattern
            min_hierarchical_rows: Minimum rows with indentation to consider hierarchical
            subtotal_keywords: Keywords that indicate subtotal rows
        """
        self.min_indent_consistency = min_indent_consistency
        self.min_hierarchical_rows = min_hierarchical_rows
        self.subtotal_keywords = subtotal_keywords or [
            "total",
            "subtotal",
            "sum",
            "sub-total",
            "grand total",
            "net",
            "gross",
            "overall",
        ]

    def detect_hierarchical_patterns(
        self, sheet_data: SheetData, bounds: TableBounds | None = None
    ) -> list[TablePattern]:
        """Detect hierarchical patterns in the sheet or region.

        Args:
            sheet_data: Sheet data to analyze
            bounds: Optional bounds to limit analysis

        Returns:
            List of detected hierarchical patterns
        """
        patterns = []

        # Define analysis bounds
        if bounds:
            start_row, end_row = bounds.start_row, bounds.end_row
            start_col, end_col = bounds.start_col, bounds.end_col
        else:
            start_row, end_row = 0, sheet_data.max_row
            start_col, end_col = 0, sheet_data.max_column

        # Find columns that might contain hierarchical data
        hierarchical_cols = self._find_hierarchical_columns(
            sheet_data, start_row, end_row, start_col, end_col
        )

        for col in hierarchical_cols:
            # Analyze the hierarchical structure in this column
            structure = self._analyze_column_hierarchy(sheet_data, col, start_row, end_row)

            if structure and len(structure.levels) >= self.min_hierarchical_rows:
                # Find the bounds of this hierarchical table
                table_bounds = self._find_hierarchical_table_bounds(
                    sheet_data, col, structure, start_col, end_col
                )

                # Calculate confidence
                confidence = self._calculate_hierarchical_confidence(
                    sheet_data, table_bounds, structure
                )

                # Create pattern
                pattern = TablePattern(
                    pattern_type=PatternType.HIERARCHICAL,
                    bounds=table_bounds,
                    confidence=confidence,
                    header_rows=self._find_header_rows(sheet_data, table_bounds),
                    characteristics={
                        "hierarchical_structure": self._structure_to_dict(structure),
                        "primary_column": col,
                        "max_indentation_depth": structure.max_depth,
                        "subtotal_rows": structure.subtotal_rows,
                        "total_hierarchical_rows": len(structure.levels),
                    },
                    orientation=TableOrientation.VERTICAL,  # Hierarchical data is typically vertical
                )

                patterns.append(pattern)
                logger.debug(
                    f"Found hierarchical pattern in column {col} with {len(structure.levels)} levels"
                )

        return patterns

    def _find_hierarchical_columns(
        self,
        sheet_data: SheetData,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
    ) -> list[int]:
        """Find columns that might contain hierarchical data."""
        hierarchical_cols = []

        for col in range(start_col, min(end_col + 1, sheet_data.max_column + 1)):
            indent_counts = {}  # indentation_level -> count
            total_filled = 0

            for row in range(start_row, min(end_row + 1, sheet_data.max_row + 1)):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    total_filled += 1
                    indent_level = self._get_cell_indentation(cell)
                    indent_counts[indent_level] = indent_counts.get(indent_level, 0) + 1

            # Check if this column has multiple indentation levels
            if len(indent_counts) >= 2 and total_filled >= self.min_hierarchical_rows:
                # Check if indentation pattern is consistent enough
                indented_rows = sum(count for level, count in indent_counts.items() if level > 0)
                if indented_rows >= self.min_hierarchical_rows:
                    hierarchical_cols.append(col)

        return hierarchical_cols

    def _get_cell_indentation(self, cell: CellData) -> int:
        """Get the indentation level of a cell."""
        # First check the explicit indentation_level field
        if cell.indentation_level > 0:
            return cell.indentation_level

        # Fall back to analyzing leading spaces in text
        if isinstance(cell.value, str):
            # Count leading spaces (each 2-4 spaces = 1 indent level)
            leading_spaces = len(cell.value) - len(cell.value.lstrip())
            return leading_spaces // 3  # Assume 3 spaces per indent level

        return 0

    def _analyze_column_hierarchy(
        self, sheet_data: SheetData, col: int, start_row: int, end_row: int
    ) -> HierarchicalStructure | None:
        """Analyze the hierarchical structure in a column."""
        levels = {}
        root_rows = []
        subtotal_rows = []
        max_depth = 0

        for row in range(start_row, min(end_row + 1, sheet_data.max_row + 1)):
            cell = sheet_data.get_cell(row, col)
            if cell and not cell.is_empty:
                indent_level = self._get_cell_indentation(cell)
                is_subtotal = self._is_subtotal_row(cell)

                level = HierarchicalLevel(
                    row=row,
                    indentation_level=indent_level,
                    is_header=cell.is_bold and row == start_row,
                    is_subtotal=is_subtotal,
                )

                levels[row] = level

                if indent_level == 0:
                    root_rows.append(row)
                max_depth = max(max_depth, indent_level)

                if is_subtotal:
                    subtotal_rows.append(row)

        # Build parent-child relationships
        self._build_hierarchy_relationships(levels)

        # Validate the hierarchy
        if not self._is_valid_hierarchy(levels, root_rows):
            return None

        return HierarchicalStructure(
            levels=levels,
            root_rows=root_rows,
            subtotal_rows=subtotal_rows,
            max_depth=max_depth,
        )

    def _is_subtotal_row(self, cell: CellData) -> bool:
        """Check if a cell represents a subtotal/total row."""
        if not isinstance(cell.value, str):
            return False

        value_lower = cell.value.lower().strip()

        # Check for subtotal keywords
        for keyword in self.subtotal_keywords:
            if keyword in value_lower:
                return True

        # Check for bold formatting (often used for totals)
        return bool(cell.is_bold and ":" in cell.value)

    def _build_hierarchy_relationships(self, levels: dict[int, HierarchicalLevel]):
        """Build parent-child relationships between hierarchical levels."""
        sorted_rows = sorted(levels.keys())

        for i, row in enumerate(sorted_rows):
            level = levels[row]

            # Find children (rows with higher indentation that come after this row)
            for j in range(i + 1, len(sorted_rows)):
                next_row = sorted_rows[j]
                next_level = levels[next_row]

                # If we hit a row with same or lower indentation, stop
                if next_level.indentation_level <= level.indentation_level:
                    break

                # If it's exactly one level deeper, it's a direct child
                if next_level.indentation_level == level.indentation_level + 1:
                    level.children.append(next_row)

    def _is_valid_hierarchy(
        self, levels: dict[int, HierarchicalLevel], root_rows: list[int]
    ) -> bool:
        """Validate that the detected hierarchy makes sense."""
        if not root_rows:
            return False

        # Check that we have a reasonable number of levels
        if len(levels) < self.min_hierarchical_rows:
            return False

        # Check that indentation is somewhat consistent
        indent_levels = [level.indentation_level for level in levels.values()]
        unique_indents = len(set(indent_levels))

        # Should have at least 2 different indentation levels
        return not unique_indents < 2

    def _find_hierarchical_table_bounds(
        self,
        sheet_data: SheetData,
        primary_col: int,
        structure: HierarchicalStructure,
        start_col: int,
        end_col: int,
    ) -> TableBounds:
        """Find the bounds of the hierarchical table."""
        # Row bounds from the structure
        all_rows = list(structure.levels.keys())
        min_row = min(all_rows)
        max_row = max(all_rows)

        # Find column bounds by looking for related data
        min_col = primary_col
        max_col = primary_col

        # Look for numeric data to the right (common in financial statements)
        for test_col in range(primary_col + 1, min(end_col + 1, sheet_data.max_column + 1)):
            has_data = False

            for row in all_rows[:10]:  # Check first 10 hierarchical rows
                cell = sheet_data.get_cell(row, test_col)
                if cell and not cell.is_empty:
                    has_data = True
                    # Numeric columns are likely part of the table
                    if cell.data_type in ["number", "currency", "percentage"]:
                        max_col = test_col
                    break

            if not has_data:
                break

        # Look for headers to the left
        for test_col in range(primary_col - 1, start_col - 1, -1):
            cell = sheet_data.get_cell(min_row, test_col)
            if cell and not cell.is_empty:
                min_col = test_col
            else:
                break

        return TableBounds(start_row=min_row, end_row=max_row, start_col=min_col, end_col=max_col)

    def _find_header_rows(self, sheet_data: SheetData, bounds: TableBounds) -> list[int]:
        """Find header rows in a hierarchical table."""
        header_rows = []

        # First row is often a header
        first_row_filled = 0
        for col in range(bounds.start_col, bounds.end_col + 1):
            cell = sheet_data.get_cell(bounds.start_row, col)
            if cell and not cell.is_empty and cell.is_bold:
                first_row_filled += 1

        if first_row_filled > 0:
            header_rows.append(bounds.start_row)

        return header_rows

    def _calculate_hierarchical_confidence(
        self,
        sheet_data: SheetData,
        bounds: TableBounds,
        structure: HierarchicalStructure,
    ) -> float:
        """Calculate confidence score for hierarchical pattern."""
        confidence = 0.5  # Base confidence

        # More levels indicate stronger hierarchy
        if structure.max_depth >= 3:
            confidence += 0.2
        elif structure.max_depth >= 2:
            confidence += 0.1

        # Subtotals increase confidence
        if structure.subtotal_rows:
            confidence += 0.15

        # Check for numeric data columns (common in hierarchical financial data)
        numeric_cols = 0
        for col in range(bounds.start_col + 1, bounds.end_col + 1):
            numeric_count = 0
            total_count = 0

            for row in list(structure.levels.keys())[:10]:
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    total_count += 1
                    if cell.data_type in ["number", "currency", "percentage"]:
                        numeric_count += 1

            if total_count > 0 and numeric_count / total_count > 0.5:
                numeric_cols += 1

        if numeric_cols > 0:
            confidence += 0.15

        return min(confidence, 1.0)

    def _structure_to_dict(self, structure: HierarchicalStructure) -> dict[str, Any]:
        """Convert hierarchical structure to a serializable dictionary."""
        return {
            "rows_by_level": self._group_rows_by_level(structure),
            "parent_child_map": self._build_parent_child_map(structure),
            "root_rows": structure.root_rows,
            "subtotal_rows": structure.subtotal_rows,
            "max_depth": structure.max_depth,
        }

    def _group_rows_by_level(self, structure: HierarchicalStructure) -> dict[int, list[int]]:
        """Group rows by their indentation level."""
        by_level = {}
        for row, level in structure.levels.items():
            indent = level.indentation_level
            if indent not in by_level:
                by_level[indent] = []
            by_level[indent].append(row)
        return by_level

    def _build_parent_child_map(self, structure: HierarchicalStructure) -> dict[int, list[int]]:
        """Build a map of parent rows to their children."""
        parent_child = {}
        for row, level in structure.levels.items():
            if level.children:
                parent_child[row] = level.children
        return parent_child
