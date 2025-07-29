"""Semantic format analysis for complex tables."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.constants import FORMAT_ANALYSIS, KEYWORDS
from ..models.sheet_data import CellData, SheetData
from ..models.table import TableRange

logger = logging.getLogger(__name__)


class RowType(str, Enum):
    """Types of rows in a table."""

    HEADER = "header"
    DATA = "data"
    SEPARATOR = "separator"
    SUBTOTAL = "subtotal"
    TOTAL = "total"
    SECTION_HEADER = "section_header"
    BLANK = "blank"
    ANNOTATION = "annotation"


class FormatPattern(BaseModel):
    """Represents a formatting pattern in the table."""

    model_config = ConfigDict(strict=True)

    pattern_type: str = Field(..., description="Type of pattern (e.g., 'border', 'color', 'font')")
    rows: list[int] = Field(..., description="Rows where this pattern appears")
    cols: list[int] = Field(..., description="Columns where this pattern appears")
    value: dict[str, Any] = Field(..., description="Pattern details")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in pattern detection")


class SemanticRow(BaseModel):
    """Represents a row with semantic meaning."""

    model_config = ConfigDict(strict=True)

    row_index: int = Field(..., ge=0, description="Row index in table")
    row_type: RowType = Field(..., description="Semantic type of the row")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    related_rows: list[int] = Field(default_factory=list, description="Related row indices")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TableStructure(BaseModel):
    """Represents the semantic structure of a table."""

    model_config = ConfigDict(strict=True)

    semantic_rows: list[SemanticRow] = Field(..., description="All rows with semantic meaning")
    sections: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Table sections as (start_row, end_row) tuples",
    )
    format_patterns: list[FormatPattern] = Field(
        default_factory=list, description="Detected formatting patterns"
    )
    has_subtotals: bool = Field(False, description="Whether table contains subtotals")
    has_grand_total: bool = Field(False, description="Whether table has a grand total")
    preserve_blank_rows: list[int] = Field(
        default_factory=list, description="Blank rows that should be preserved"
    )


@dataclass
class FormatSignature:
    """Formatting signature for a cell or row."""

    is_bold: bool = False
    is_italic: bool = False
    has_background: bool = False
    background_color: str | None = None
    font_size: float | None = None
    alignment: str | None = None
    has_top_border: bool = False
    has_bottom_border: bool = False
    indentation: int = 0

    def similarity(self, other: "FormatSignature") -> float:
        """Calculate similarity score between two signatures."""
        matches = 0.0
        total = 0.0

        # Boolean attributes
        for attr in [
            "is_bold",
            "is_italic",
            "has_background",
            "has_top_border",
            "has_bottom_border",
        ]:
            total += 1
            if getattr(self, attr) == getattr(other, attr):
                matches += 1

        # Color matching (if both have backgrounds)
        if self.has_background and other.has_background:
            total += 1
            if self.background_color == other.background_color:
                matches += 1

        # Indentation
        if self.indentation == other.indentation:
            matches += 0.5
        total += 0.5

        return matches / total if total > 0 else 0.0


class SemanticFormatAnalyzer:
    """Analyzes formatting to understand table semantics."""

    def __init__(self) -> None:
        self.blank_row_threshold = FORMAT_ANALYSIS.BLANK_ROW_THRESHOLD
        self.subtotal_keywords = list(KEYWORDS.SUBTOTAL_KEYWORDS)
        self.grand_total_keywords = list(KEYWORDS.GRAND_TOTAL_KEYWORDS)
        self.section_keywords = list(KEYWORDS.SECTION_KEYWORDS)

    def analyze_table_structure(
        self, sheet_data: SheetData, table_range: TableRange, header_rows: int = 1
    ) -> TableStructure:
        """
        Analyze the semantic structure of a table.

        Args:
            sheet_data: Sheet data containing the table
            table_range: Range of the table
            header_rows: Number of header rows

        Returns:
            TableStructure with semantic analysis
        """
        if sheet_data is None:
            raise ValueError("sheet_data cannot be None")
        if table_range is None:
            raise ValueError("table_range cannot be None")
        if header_rows < 0:
            raise ValueError(f"header_rows must be non-negative, got {header_rows}")
        if header_rows > table_range.row_count:
            raise ValueError(
                f"header_rows ({header_rows}) cannot exceed table row count ({table_range.row_count}). "
                f"Table range: {table_range.excel_range}. Consider reducing header_rows or "
                f"checking if the table range is correct."
            )

        logger.info(f"Analyzing table structure for range {table_range.excel_range}")

        # Analyze each row
        semantic_rows = []
        for row_offset in range(table_range.row_count):
            row_idx = table_range.start_row + row_offset
            semantic_row = self._analyze_row(
                sheet_data, row_idx, table_range, row_offset < header_rows
            )
            semantic_rows.append(semantic_row)

        # Detect sections
        sections = self._detect_sections(semantic_rows)

        # Detect format patterns
        format_patterns = self._detect_format_patterns(sheet_data, table_range, semantic_rows)

        # Determine which blank rows to preserve
        preserve_blank_rows = self._identify_semantic_blanks(semantic_rows, sections)

        # Check for totals
        has_subtotals = any(r.row_type == RowType.SUBTOTAL for r in semantic_rows)
        has_grand_total = any(r.row_type == RowType.TOTAL for r in semantic_rows)

        return TableStructure(
            semantic_rows=semantic_rows,
            sections=sections,
            format_patterns=format_patterns,
            has_subtotals=has_subtotals,
            has_grand_total=has_grand_total,
            preserve_blank_rows=preserve_blank_rows,
        )

    def _analyze_row(
        self,
        sheet_data: SheetData,
        row_idx: int,
        table_range: TableRange,
        is_header: bool,
    ) -> SemanticRow:
        """Analyze a single row for semantic meaning."""
        row_offset = row_idx - table_range.start_row

        # Get row data
        row_cells = []
        for col_offset in range(table_range.col_count):
            col_idx = table_range.start_col + col_offset
            cell = sheet_data.get_cell(row_idx, col_idx)
            row_cells.append(cell)

        # Check if blank
        empty_count = sum(1 for cell in row_cells if not cell or cell.value is None)
        if empty_count / len(row_cells) >= self.blank_row_threshold:
            return SemanticRow(row_index=row_offset, row_type=RowType.BLANK, confidence=1.0)

        # Header rows
        if is_header:
            return SemanticRow(row_index=row_offset, row_type=RowType.HEADER, confidence=1.0)

        # Check for totals/subtotals
        row_text_parts = []
        for cell in row_cells:
            if cell and cell.value:
                row_text_parts.append(str(cell.value).lower())
        row_text = " ".join(row_text_parts)

        # Check for subtotal keywords first
        for keyword in self.subtotal_keywords:
            if keyword in row_text:
                return SemanticRow(
                    row_index=row_offset,
                    row_type=RowType.SUBTOTAL,
                    confidence=0.9,
                )

        # Then check for grand total keywords
        for keyword in self.grand_total_keywords:
            if keyword in row_text:
                # Check if grand total (last data row with total formatting)
                # A total is only a grand total if it's at the end or clearly final
                is_last_data_row = (
                    row_offset >= table_range.row_count - 2
                )  # Allow for one trailing blank
                has_strong_total_formatting = self._has_total_formatting(
                    [c for c in row_cells if c]
                )

                # If it's in the middle of the table, it's likely a subtotal
                is_grand_total = is_last_data_row and has_strong_total_formatting

                return SemanticRow(
                    row_index=row_offset,
                    row_type=RowType.TOTAL if is_grand_total else RowType.SUBTOTAL,
                    confidence=0.9,
                )

        # Check for section headers
        if self._is_section_header([c for c in row_cells if c], row_text):
            return SemanticRow(
                row_index=row_offset, row_type=RowType.SECTION_HEADER, confidence=0.8
            )

        # Check for separators (formatting-only rows)
        if self._is_separator_row([c for c in row_cells if c]):
            return SemanticRow(row_index=row_offset, row_type=RowType.SEPARATOR, confidence=0.85)

        # Default to data row
        return SemanticRow(row_index=row_offset, row_type=RowType.DATA, confidence=0.7)

    def _has_total_formatting(self, row_cells: list[CellData]) -> bool:
        """Check if row has formatting typical of totals."""
        bold_count = sum(bool(cell and cell.is_bold) for cell in row_cells)
        has_top_border = any(
            cell
            for cell in row_cells
            if cell and hasattr(cell, "has_top_border") and cell.has_top_border
        )

        # Totals often have bold text or top borders
        return (
            bold_count > len(row_cells) * FORMAT_ANALYSIS.TOTAL_FORMATTING_THRESHOLD
            or has_top_border
        )

    def _is_section_header(self, row_cells: list[CellData], row_text: str) -> bool:
        """Check if row is a section header."""
        # Section headers often:
        # 1. Have text in first column only
        # 2. Are bold or have background color
        # 3. Contain section keywords

        non_empty_cells = [cell for cell in row_cells if cell and cell.value]
        if len(non_empty_cells) == 1 and row_cells[0] and row_cells[0].value:
            # Text only in first column
            first_cell = row_cells[0]
            if first_cell.is_bold or first_cell.background_color:
                return True

        # Check for section keywords
        return any(keyword in row_text for keyword in self.section_keywords)

    def _is_separator_row(self, row_cells: list[CellData]) -> bool:
        """Check if row is a separator (formatting only)."""
        # Separator rows have formatting but little/no content
        non_empty = sum(bool(cell and cell.value) for cell in row_cells)
        has_formatting = any(
            cell
            for cell in row_cells
            if cell
            and (
                cell.background_color
                or hasattr(cell, "has_bottom_border")
                and cell.has_bottom_border
            )
        )

        return non_empty <= 1 and has_formatting

    def _detect_sections(self, semantic_rows: list[SemanticRow]) -> list[tuple[int, int]]:
        """Detect logical sections in the table."""
        sections: list[tuple[int, int]] = []
        current_section_start: int | None = None

        for i, row in enumerate(semantic_rows):
            if row.row_type == RowType.SECTION_HEADER:
                # End previous section if exists
                if current_section_start is not None:
                    sections.append((current_section_start, i - 1))
                current_section_start = i
            elif row.row_type in [RowType.BLANK, RowType.SEPARATOR]:
                # Might indicate section boundary
                if (
                    current_section_start is not None
                    and i > current_section_start + FORMAT_ANALYSIS.SECTION_BOUNDARY_MIN_ROWS - 1
                ):  # Not immediately after header
                    sections.append((current_section_start, i - 1))
                    current_section_start = None

        # Close final section
        if current_section_start is not None:
            sections.append((current_section_start, len(semantic_rows) - 1))

        return sections

    def _detect_format_patterns(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        semantic_rows: list[SemanticRow],
    ) -> list[FormatPattern]:
        """Detect recurring format patterns."""
        patterns = []

        # Detect alternating row colors
        bg_pattern = self._detect_alternating_backgrounds(sheet_data, table_range, semantic_rows)
        if bg_pattern:
            patterns.append(bg_pattern)

        # Detect consistent column formatting
        col_patterns = self._detect_column_formatting(sheet_data, table_range, semantic_rows)
        patterns.extend(col_patterns)

        return patterns

    def _detect_alternating_backgrounds(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        semantic_rows: list[SemanticRow],
    ) -> FormatPattern | None:
        """Detect alternating row background colors."""
        # Check data rows only
        data_rows = [r for r in semantic_rows if r.row_type == RowType.DATA]
        if len(data_rows) < FORMAT_ANALYSIS.MIN_DATA_ROWS_FOR_PATTERN:
            return None

        # Get background colors for data rows
        colors = []
        for row in data_rows[: FORMAT_ANALYSIS.FIRST_ROWS_TO_CHECK]:
            row_idx = table_range.start_row + row.row_index
            # Get color from first cell
            cell = sheet_data.get_cell(row_idx, table_range.start_col)
            color = cell.background_color if cell else None
            colors.append(color)

        # Check for alternating pattern
        if len(set(colors)) == 2 and None not in colors:
            # Verify alternation
            alternates = all(colors[i] != colors[i + 1] for i in range(len(colors) - 1))
            if alternates:
                return FormatPattern(
                    pattern_type="alternating_background",
                    rows=[r.row_index for r in data_rows],
                    cols=list(range(table_range.col_count)),
                    value={"colors": list(set(colors))},
                    confidence=0.95,
                )

        return None

    def _detect_column_formatting(
        self,
        sheet_data: SheetData,
        table_range: TableRange,
        semantic_rows: list[SemanticRow],
    ) -> list[FormatPattern]:
        """Detect consistent formatting in columns."""
        patterns: list[FormatPattern] = []
        data_rows = [r for r in semantic_rows if r.row_type == RowType.DATA]

        if not data_rows:
            return patterns

        # Check each column
        for col_offset in range(table_range.col_count):
            col_idx = table_range.start_col + col_offset

            # Collect formatting for this column
            alignments = []
            bold_count = 0

            for row in data_rows[: FORMAT_ANALYSIS.MAX_ROWS_TO_SAMPLE]:
                row_idx = table_range.start_row + row.row_index
                cell = sheet_data.get_cell(row_idx, col_idx)
                if cell:
                    if cell.alignment:
                        alignments.append(cell.alignment)
                    if cell.is_bold:
                        bold_count += 1

            # Check for consistent alignment
            if alignments and len(set(alignments)) == 1:
                patterns.append(
                    FormatPattern(
                        pattern_type="column_alignment",
                        rows=[r.row_index for r in data_rows],
                        cols=[col_offset],
                        value={"alignment": alignments[0]},
                        confidence=0.9,
                    )
                )

            # Check for consistently bold column
            if bold_count > len(data_rows) * FORMAT_ANALYSIS.CONSISTENT_COLUMN_THRESHOLD:
                patterns.append(
                    FormatPattern(
                        pattern_type="column_bold",
                        rows=[r.row_index for r in data_rows],
                        cols=[col_offset],
                        value={"is_bold": True},
                        confidence=0.85,
                    )
                )

        return patterns

    def _identify_semantic_blanks(
        self, semantic_rows: list[SemanticRow], sections: list[tuple[int, int]]
    ) -> list[int]:
        """Identify blank rows that have semantic meaning."""
        preserve = []

        for i, row in enumerate(semantic_rows):
            if row.row_type == RowType.BLANK:
                # Preserve if between sections
                for start, end in sections:
                    if i == end + 1 or i == start - 1:
                        preserve.append(row.row_index)
                        break

                # Preserve if before/after subtotal or total
                if (
                    i > 0
                    and semantic_rows[i - 1].row_type
                    in [
                        RowType.SUBTOTAL,
                        RowType.TOTAL,
                    ]
                    or i < len(semantic_rows) - 1
                    and semantic_rows[i + 1].row_type
                    in [
                        RowType.SUBTOTAL,
                        RowType.TOTAL,
                    ]
                ):
                    preserve.append(row.row_index)

        return preserve

    def get_format_signature(self, cell: CellData) -> FormatSignature:
        """Extract format signature from a cell."""
        return FormatSignature(
            is_bold=cell.is_bold if cell else False,
            is_italic=cell.is_italic if cell else False,
            has_background=bool(cell.background_color) if cell else False,
            background_color=cell.background_color if cell else None,
            font_size=cell.font_size if cell else None,
            alignment=cell.alignment if cell else None,
            indentation=cell.indentation_level if cell else 0,
        )
