"""Island detection algorithm for finding disconnected table regions.

This module implements connected component analysis to identify separate
"islands" of data that likely represent individual tables.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ..core.constants import FORMATTING_DETECTION, ISLAND_DETECTION
from ..models.table import TableInfo, TableRange
from ..utils.excel_utils import get_column_letter

if TYPE_CHECKING:
    from ..models.sheet_data import SheetData

logger = logging.getLogger(__name__)


@dataclass
class DataIsland:
    """Represents a connected region of data (an island)."""

    cells: set[tuple[int, int]] = field(default_factory=set)
    min_row: int | None = None
    max_row: int | None = None
    min_col: int | None = None
    max_col: int | None = None
    density: float = 0.0
    has_headers: bool = False
    confidence: float = 0.0
    total_sheet_cells: int = 0  # Total cells in the sheet for relative size calculation
    border_cell_ratio: float = 0.0  # Ratio of populated cells in border area
    is_subset_of: Optional["DataIsland"] = (
        None  # Reference to containing island if this is a subset
    )

    def add_cell(self, row: int, col: int) -> None:
        """Add a cell to the island and update bounds."""
        self.cells.add((row, col))

        if self.min_row is None or row < self.min_row:
            self.min_row = row
        if self.max_row is None or row > self.max_row:
            self.max_row = row
        if self.min_col is None or col < self.min_col:
            self.min_col = col
        if self.max_col is None or col > self.max_col:
            self.max_col = col

    def calculate_metrics(self, sheet_data: "SheetData") -> None:
        """Calculate island metrics like density and header detection."""
        if (
            not self.cells
            or self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return

        # Store sheet data for formatting analysis
        self._sheet_data = sheet_data

        # Calculate density
        total_cells = (self.max_row - self.min_row + 1) * (self.max_col - self.min_col + 1)
        self.density = len(self.cells) / total_cells if total_cells > 0 else 0

        # Detect headers with enhanced analysis
        first_row_cells = [
            sheet_data.get_cell(self.min_row, col) for col in range(self.min_col, self.max_col + 1)
        ]

        # Check multiple header indicators
        all_text = all(
            cell and cell.value is not None and cell.data_type == "string"
            for cell in first_row_cells
        )
        any_bold = any(cell and cell.is_bold for cell in first_row_cells)

        # Headers likely if all text AND bold (background color less important)
        # or if all text and different data types in next row
        self.has_headers = all_text and any_bold

        # If not bold, check if next row has different data types
        if not self.has_headers and all_text and self.max_row > self.min_row:
            second_row_cells = [
                sheet_data.get_cell(self.min_row + 1, col)
                for col in range(self.min_col, self.max_col + 1)
            ]
            has_numeric = any(
                cell and cell.data_type in ["number", "datetime"] for cell in second_row_cells
            )
            if has_numeric:
                self.has_headers = True

        # Analyze border cells
        self.border_cell_ratio = self._analyze_border_cells(sheet_data)

        # Calculate confidence based on various factors
        self.confidence = self._calculate_confidence()

    def _calculate_confidence(self) -> float:
        """Calculate confidence score for this island being a table.

        Enhanced with weighted scoring system using multiple cell attributes.
        """
        # Start with weighted scoring components
        score_components = {}

        # 1. SIZE SCORE (20% weight) - Relative and absolute size
        cell_count = len(self.cells)
        relative_size = cell_count / self.total_sheet_cells if self.total_sheet_cells > 0 else 0

        size_score = 0.5  # Base
        if relative_size >= ISLAND_DETECTION.RELATIVE_SIZE_LARGE:
            size_score = 1.0
        elif relative_size >= ISLAND_DETECTION.RELATIVE_SIZE_MEDIUM:
            size_score = 0.8
        elif relative_size >= ISLAND_DETECTION.RELATIVE_SIZE_SMALL:
            size_score = 0.6
        elif relative_size < ISLAND_DETECTION.RELATIVE_SIZE_TINY:
            size_score = 0.2

        # Boost for absolute size
        if cell_count >= ISLAND_DETECTION.MIN_CELLS_GOOD:
            size_score = min(1.0, size_score + 0.1)
        elif cell_count < ISLAND_DETECTION.MIN_CELLS_SMALL:
            size_score = max(0.0, size_score - 0.2)

        score_components["size"] = (size_score, 0.20)

        # 2. DENSITY SCORE (15% weight) - How filled the region is
        density_score = 0.5  # Base
        if self.density > ISLAND_DETECTION.DENSITY_HIGH:
            density_score = 1.0
        elif self.density > ISLAND_DETECTION.DENSITY_MEDIUM:
            density_score = 0.7
        elif self.density < ISLAND_DETECTION.DENSITY_LOW:
            density_score = 0.3

        score_components["density"] = (density_score, 0.15)

        # 3. SHAPE SCORE (10% weight) - Prefer rectangular tables
        shape_score = 0.5  # Base
        if (
            self.min_row is not None
            and self.max_row is not None
            and self.min_col is not None
            and self.max_col is not None
        ):
            height = self.max_row - self.min_row + 1
            width = self.max_col - self.min_col + 1
            if height > 0:
                aspect_ratio = width / height
                if (
                    ISLAND_DETECTION.ASPECT_RATIO_MIN
                    <= aspect_ratio
                    <= ISLAND_DETECTION.ASPECT_RATIO_MAX
                ):
                    shape_score = 0.9
                else:
                    # Penalize extreme aspect ratios
                    if aspect_ratio < 0.05 or aspect_ratio > 20:
                        shape_score = 0.2

        score_components["shape"] = (shape_score, 0.10)

        # 4. HEADER SCORE (15% weight) - Tables usually have headers
        header_score = 0.8 if self.has_headers else 0.4
        score_components["headers"] = (header_score, 0.15)

        # 5. BORDER SCORE (15% weight) - Clean borders indicate tables
        border_score = 1.0  # Start high
        if self.border_cell_ratio > ISLAND_DETECTION.BORDER_CELL_THRESHOLD:
            border_score = 0.5
            if self.border_cell_ratio > 0.5:
                border_score = 0.2

        score_components["borders"] = (border_score, 0.15)

        # 6. FORMATTING SCORE (15% weight) - Will be calculated if we have cell data
        formatting_score = (
            self._calculate_formatting_score() if hasattr(self, "_sheet_data") else 0.5
        )
        score_components["formatting"] = (formatting_score, 0.15)

        # 7. ISOLATION SCORE (10% weight) - Not being subset of another table
        isolation_score = 0.2 if self.is_subset_of is not None else 1.0
        score_components["isolation"] = (isolation_score, 0.10)

        # Calculate weighted total
        total_score = 0.0
        total_weight = 0.0

        for _component_name, (score, weight) in score_components.items():
            total_score += score * weight
            total_weight += weight

        # Normalize if weights don't sum to 1.0 (safety check)
        if total_weight > 0:
            confidence = total_score / total_weight
        else:
            confidence = ISLAND_DETECTION.BASE_CONFIDENCE

        # Apply bounds
        return min(max(confidence, 0.0), 1.0)

    def _calculate_formatting_score(self) -> float:
        """Calculate formatting consistency score for the island.

        Analyzes:
        - Border consistency within the table
        - Alignment patterns (numbers right, text left)
        - Font/color consistency
        - Data type patterns in columns

        Returns:
            Score between 0.0 and 1.0
        """
        if not hasattr(self, "_sheet_data") or not self._sheet_data:
            return 0.5  # Default neutral score

        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return 0.5

        score_factors = []

        # 1. Border consistency
        border_consistency = self._analyze_border_consistency()
        score_factors.append(border_consistency)

        # 2. Column data type consistency
        column_consistency = self._analyze_column_type_consistency()
        score_factors.append(column_consistency)

        # 3. Alignment patterns
        alignment_score = self._analyze_alignment_patterns()
        score_factors.append(alignment_score)

        # 4. Formatting uniformity (fonts, colors)
        format_uniformity = self._analyze_format_uniformity()
        score_factors.append(format_uniformity)

        # Return average of all factors
        return sum(score_factors) / len(score_factors) if score_factors else 0.5

    def _analyze_border_consistency(self) -> float:
        """Analyze how consistent borders are within the table."""
        if not hasattr(self, "_sheet_data"):
            return 0.5

        border_patterns: dict[tuple[bool, bool, bool, bool], int] = {}
        total_cells = 0

        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return 0.5

        for row in range(self.min_row, self.max_row + 1):
            for col in range(self.min_col, self.max_col + 1):
                cell = self._sheet_data.get_cell(row, col)
                if cell:
                    total_cells += 1
                    # Create border signature
                    sig = (
                        bool(cell.border_top),
                        bool(cell.border_bottom),
                        bool(cell.border_left),
                        bool(cell.border_right),
                    )
                    border_patterns[sig] = border_patterns.get(sig, 0) + 1

        if not border_patterns:
            return 0.5

        # Calculate consistency as ratio of most common pattern
        most_common_count = max(border_patterns.values())
        consistency = most_common_count / total_cells if total_cells > 0 else 0

        # Bonus for having any borders at all
        has_borders = any(any(sig) for sig in border_patterns)
        if has_borders:
            consistency = min(1.0, consistency + 0.1)

        return consistency

    def _analyze_column_type_consistency(self) -> float:
        """Analyze how consistent data types are within columns."""
        if not hasattr(self, "_sheet_data"):
            return 0.5

        if (
            self.min_col is None
            or self.max_col is None
            or self.min_row is None
            or self.max_row is None
        ):
            return 0.5

        column_scores = []

        for col in range(self.min_col, self.max_col + 1):
            type_counts: dict[str, int] = {}
            total_cells = 0

            # Skip header row if detected
            start_row = self.min_row + 1 if self.has_headers else self.min_row

            for row in range(start_row, self.max_row + 1):
                cell = self._sheet_data.get_cell(row, col)
                if cell and cell.value is not None:
                    total_cells += 1
                    data_type = cell.data_type
                    type_counts[data_type] = type_counts.get(data_type, 0) + 1

            if total_cells > 0 and type_counts:
                # Consistency is ratio of most common type
                most_common = max(type_counts.values())
                consistency = most_common / total_cells
                column_scores.append(consistency)

        return sum(column_scores) / len(column_scores) if column_scores else 0.5

    def _analyze_alignment_patterns(self) -> float:
        """Analyze if alignment follows expected patterns (numbers right, text left)."""
        if not hasattr(self, "_sheet_data"):
            return 0.5

        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return 0.5

        correct_alignments = 0
        total_aligned_cells = 0

        for row in range(self.min_row, self.max_row + 1):
            for col in range(self.min_col, self.max_col + 1):
                cell = self._sheet_data.get_cell(row, col)
                if cell and cell.alignment:
                    total_aligned_cells += 1

                    # Expected patterns
                    if (
                        cell.data_type == "number"
                        and cell.alignment == "right"
                        or cell.data_type == "string"
                        and cell.alignment in ["left", "center"]
                        or cell.data_type == "datetime"
                        and cell.alignment in ["left", "center"]
                    ):
                        correct_alignments += 1

        # If no explicit alignment, assume default (which is usually correct)
        if total_aligned_cells == 0:
            return 0.7  # Neutral-positive score

        return correct_alignments / total_aligned_cells

    def _analyze_format_uniformity(self) -> float:
        """Analyze formatting uniformity within the table."""
        if not hasattr(self, "_sheet_data"):
            return 0.5

        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return 0.5

        # Track formatting variations
        font_sizes = set()
        font_colors = set()
        background_colors = set()
        bold_count = 0
        total_cells = 0

        # Skip header row for uniformity check
        start_row = self.min_row + 1 if self.has_headers else self.min_row

        for row in range(start_row, self.max_row + 1):
            for col in range(self.min_col, self.max_col + 1):
                cell = self._sheet_data.get_cell(row, col)
                if cell:
                    total_cells += 1
                    if cell.font_size:
                        font_sizes.add(cell.font_size)
                    if cell.font_color:
                        font_colors.add(cell.font_color)
                    if cell.background_color:
                        background_colors.add(cell.background_color)
                    if cell.is_bold:
                        bold_count += 1

        if total_cells == 0:
            return 0.5

        # Calculate uniformity scores
        scores = []

        # Font size uniformity (fewer is better)
        if len(font_sizes) <= 1:
            scores.append(1.0)
        elif len(font_sizes) == 2:
            scores.append(0.8)
        else:
            scores.append(0.5)

        # Color uniformity
        if len(font_colors) <= 1:
            scores.append(1.0)
        elif len(font_colors) <= 2:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Background uniformity
        if len(background_colors) == 0:
            scores.append(0.9)  # No backgrounds is uniform
        elif len(background_colors) == 1:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Bold uniformity (all or none is good)
        bold_ratio = bold_count / total_cells
        if bold_ratio == 0 or bold_ratio == 1:
            scores.append(1.0)
        elif bold_ratio < 0.1 or bold_ratio > 0.9:
            scores.append(0.8)
        else:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.5

    def _analyze_border_cells(
        self, sheet_data: "SheetData", border_width: int | None = None
    ) -> float:
        """Analyze cells around the table border to detect if table boundaries might be incorrect.

        Args:
            sheet_data: The sheet data to analyze
            border_width: Width of border to check (default from constants)

        Returns:
            Ratio of populated border cells (0.0 to 1.0)
        """
        if border_width is None:
            border_width = ISLAND_DETECTION.BORDER_WIDTH

        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return 0.0

        border_cells = set()
        populated_border = 0

        # Define border bounds (excluding the table itself)
        border_min_row = max(0, self.min_row - border_width)
        border_max_row = min(sheet_data.max_row, self.max_row + border_width)
        border_min_col = max(0, self.min_col - border_width)
        border_max_col = min(sheet_data.max_column, self.max_col + border_width)

        # Check top border
        for row in range(border_min_row, self.min_row):
            for col in range(border_min_col, border_max_col + 1):
                pos = (row, col)
                if pos not in border_cells:
                    border_cells.add(pos)
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        populated_border += 1

        # Check bottom border
        for row in range(self.max_row + 1, border_max_row + 1):
            for col in range(border_min_col, border_max_col + 1):
                pos = (row, col)
                if pos not in border_cells:
                    border_cells.add(pos)
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        populated_border += 1

        # Check left border (excluding corners already checked)
        for row in range(self.min_row, self.max_row + 1):
            for col in range(border_min_col, self.min_col):
                pos = (row, col)
                if pos not in border_cells:
                    border_cells.add(pos)
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        populated_border += 1

        # Check right border (excluding corners already checked)
        for row in range(self.min_row, self.max_row + 1):
            for col in range(self.max_col + 1, border_max_col + 1):
                pos = (row, col)
                if pos not in border_cells:
                    border_cells.add(pos)
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        populated_border += 1

        # Calculate ratio
        return populated_border / len(border_cells) if border_cells else 0.0

    def to_range(self) -> str:
        """Convert island bounds to Excel range notation."""
        if (
            self.min_row is None
            or self.max_row is None
            or self.min_col is None
            or self.max_col is None
        ):
            return ""
        start = f"{get_column_letter(self.min_col)}{self.min_row + 1}"
        end = f"{get_column_letter(self.max_col)}{self.max_row + 1}"
        return f"{start}:{end}"


class IslandDetector:
    """Detects disconnected table regions using connected component analysis."""

    def __init__(
        self,
        max_gap: int | None = None,
        min_island_size: int = 4,
        include_diagonal: bool = True,
        column_consistency_threshold: float | None = None,
        min_empty_rows_to_split: int | None = None,
        use_structural_analysis: bool = False,
        adaptive_thresholds: bool = True,
        use_formatting_boundaries: bool = True,
        empty_row_tolerance: int = 1,
    ):
        """Initialize the island detector.

        Args:
            max_gap: Maximum gap between cells to consider them connected (None uses default)
            min_island_size: Minimum number of cells to consider as an island
            include_diagonal: Whether diagonal cells are considered connected
            column_consistency_threshold: How similar column usage must be to group rows (None uses default)
            min_empty_rows_to_split: Number of empty rows needed to split islands (None uses default)
            use_structural_analysis: Enable structural analysis for text files
            adaptive_thresholds: Enable adaptive sizing based on sheet size
            use_formatting_boundaries: Enable formatting-based boundary detection
            empty_row_tolerance: Number of empty rows to tolerate within a table (default: 1)
        """
        # Use constants if not provided
        if max_gap is None:
            max_gap = ISLAND_DETECTION.DEFAULT_MAX_GAP
        if column_consistency_threshold is None:
            column_consistency_threshold = ISLAND_DETECTION.COLUMN_CONSISTENCY_THRESHOLD
        if min_empty_rows_to_split is None:
            min_empty_rows_to_split = ISLAND_DETECTION.MIN_EMPTY_ROWS_TO_SPLIT

        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}")
        if min_island_size < 1:
            raise ValueError(f"min_island_size must be positive, got {min_island_size}")

        self.logger = logger
        self.max_gap = max_gap
        self.min_island_size = min_island_size
        self.include_diagonal = include_diagonal
        self.column_consistency_threshold = column_consistency_threshold
        self.min_empty_rows_to_split = min_empty_rows_to_split
        self.use_structural_analysis = use_structural_analysis
        self.adaptive_thresholds = adaptive_thresholds
        self.use_formatting_boundaries = use_formatting_boundaries
        self.empty_row_tolerance = empty_row_tolerance

    def detect_islands(self, sheet_data: "SheetData") -> list[DataIsland]:
        """Detect all data islands in the sheet.

        Uses flood-fill algorithm to find connected components of data.
        If structural analysis is enabled, uses column consistency to split regions.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of DataIsland objects representing disconnected regions
        """
        if sheet_data is None:
            raise ValueError("sheet_data cannot be None")

        if not sheet_data.has_data():
            return []

        # Calculate adaptive thresholds based on sheet size
        if self.adaptive_thresholds:
            total_sheet_cells = len(sheet_data.cells)
            # Adjust min_island_size based on sheet size
            if total_sheet_cells > 10000:
                # Large sheets: require at least 0.1% of cells
                adaptive_min_size = max(20, int(total_sheet_cells * 0.001))
            elif total_sheet_cells > 1000:
                # Medium sheets: require at least 0.5% of cells
                adaptive_min_size = max(10, int(total_sheet_cells * 0.005))
            else:
                # Small sheets: use default minimum
                adaptive_min_size = self.min_island_size

            # Use the larger of configured minimum or adaptive minimum
            effective_min_size = max(self.min_island_size, adaptive_min_size)
        else:
            effective_min_size = self.min_island_size
            total_sheet_cells = len(sheet_data.cells)

        # Use structural analysis for better separation if enabled
        if self.use_structural_analysis:
            return self._detect_islands_structural(sheet_data)

        # Get all cells with data (ignore empty strings)
        data_cells = {
            (cell.row, cell.column)
            for address, cell in sheet_data.cells.items()
            if not cell.is_empty
        }

        # Track visited cells
        visited: set[tuple[int, int]] = set()
        islands = []

        # Find all islands using flood-fill
        for cell_pos in data_cells:
            if cell_pos not in visited:
                island = self._flood_fill(cell_pos, data_cells, visited)
                if len(island.cells) >= effective_min_size:
                    # Pass total sheet cells for relative size calculation
                    island.total_sheet_cells = total_sheet_cells
                    island.calculate_metrics(sheet_data)
                    islands.append(island)

        # Sort islands by size (largest first) and position
        islands.sort(key=lambda i: (-len(i.cells), i.min_row, i.min_col))

        # Apply formatting-based splitting if enabled
        formatting_splits_applied = False
        if self.use_formatting_boundaries and islands:
            original_count = len(islands)
            islands = self._apply_formatting_splits(islands, sheet_data)
            formatting_splits_applied = len(islands) > original_count
            self.logger.info(
                f"After formatting splits: {len(islands)} data islands (was {original_count}, splits applied: {formatting_splits_applied})"
            )

        # Reconnect islands separated by tolerable empty rows
        if self.empty_row_tolerance > 0 and len(islands) > 1:
            original_count = len(islands)
            islands = self._reconnect_gap_separated_islands(islands, sheet_data)
            if len(islands) < original_count:
                self.logger.info(
                    f"After reconnecting gap-separated islands: {len(islands)} data islands (was {original_count})"
                )

        # Apply merging to reduce fragmentation
        if len(islands) > 1:
            # Check if islands are well-separated by empty rows
            well_separated = self._are_islands_well_separated(islands, sheet_data)

            # Adaptive merge distance based on sheet density, formatting splits, and separation
            sheet_density = len(sheet_data.cells) / (
                (sheet_data.max_row + 1) * (sheet_data.max_column + 1)
            )

            if well_separated:
                # Islands are separated by empty rows - be very conservative with merging
                merge_distance = 0  # No merging for well-separated tables
            elif formatting_splits_applied:
                # If formatting splits were applied, be more conservative with merging
                # to preserve the formatting-based table boundaries
                merge_distance = 1
            elif sheet_density < 0.3:
                # Sparse sheet: larger merge distance
                merge_distance = 5
            elif sheet_density < 0.6:
                # Medium density: moderate merge distance
                merge_distance = 3
            else:
                # Dense sheet: smaller merge distance
                merge_distance = 2

            if merge_distance > 0:
                islands = self.merge_nearby_islands(
                    islands, merge_distance=merge_distance, sheet_data=sheet_data
                )
                # Recalculate metrics for merged islands
                for island in islands:
                    # Pass total sheet cells for relative size calculation
                    island.total_sheet_cells = total_sheet_cells
                    island.calculate_metrics(sheet_data)
                self.logger.info(
                    f"After merging with distance {merge_distance}: {len(islands)} data islands"
                )
            else:
                self.logger.info(
                    f"Skipping merge - islands are well-separated: {len(islands)} data islands"
                )

        # Check for subset relationships
        if len(islands) > 1:
            self._check_subset_relationships(islands)

        # Calculate metrics for each island to detect headers and calculate confidence
        for island in islands:
            island.calculate_metrics(sheet_data)

        self.logger.info(f"Detected {len(islands)} data islands")
        return islands

    def _flood_fill(
        self,
        start: tuple[int, int],
        data_cells: set[tuple[int, int]],
        visited: set[tuple[int, int]],
    ) -> DataIsland:
        """Perform flood-fill to find connected component.

        Args
        ----
        start : tuple[int, int]
            Starting cell position as (row, col) tuple. Must be a cell
            that contains data and hasn't been visited yet.
        data_cells : set[tuple[int, int]]
            Set of all cell positions that contain data. This is the
            universe of cells to explore.
        visited : set[tuple[int, int]]
            Set of already visited cell positions. This set is modified
            in-place to track progress across multiple flood-fill calls.

        Returns
        -------
        DataIsland
            Island containing all cells connected to the starting position.
            Connection is determined by the max_gap parameter.

        Notes
        -----
        This implements a breadth-first search (BFS) flood-fill algorithm
        to find all cells that form a connected component. Cells are considered
        connected based on the max_gap setting:

        - max_gap=0: Only directly adjacent cells
        - max_gap=1: Cells within 1 position (allows single empty cell gaps)
        - max_gap=2: Cells within 2 positions (allows small formatting gaps)

        The include_diagonal setting determines whether diagonal connections
        are allowed. The algorithm is efficient with O(n) time complexity
        where n is the number of cells in the island.
        """
        island = DataIsland()
        queue = deque([start])
        visited.add(start)

        while queue:
            row, col = queue.popleft()
            island.add_cell(row, col)

            # Check all neighbors
            neighbors = self._get_neighbors(row, col)
            for neighbor in neighbors:
                if neighbor in data_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return island

    def _get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Get neighboring cell positions based on max_gap setting.

        Args
        ----
        row : int
            Current row index (0-based).
        col : int
            Current column index (0-based).

        Returns
        -------
        list[tuple[int, int]]
            List of neighboring cell positions as (row, col) tuples.
            The number of neighbors depends on max_gap and include_diagonal settings.

        Examples
        --------
        With max_gap=1 and include_diagonal=True::

            >>> detector._get_neighbors(5, 5)
            [(4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)]

        With max_gap=1 and include_diagonal=False::

            >>> detector._get_neighbors(5, 5)
            [(4, 5), (5, 4), (5, 6), (6, 5)]

        Notes
        -----
        The neighborhood pattern is controlled by two parameters:

        - max_gap: Determines the maximum distance for connections (Manhattan distance)
        - include_diagonal: Whether to include diagonal neighbors

        This method doesn't check bounds or validate positions; it returns all
        theoretical neighbors which are then filtered by the calling method.
        """
        neighbors = []

        # Define the range based on max_gap
        for dr in range(-self.max_gap, self.max_gap + 1):
            for dc in range(-self.max_gap, self.max_gap + 1):
                # Skip the center cell
                if dr == 0 and dc == 0:
                    continue

                # Skip diagonals if not included
                if not self.include_diagonal and dr != 0 and dc != 0:
                    continue

                neighbors.append((row + dr, col + dc))

        return neighbors

    def merge_nearby_islands(
        self,
        islands: list[DataIsland],
        merge_distance: int = 2,
        sheet_data: Optional["SheetData"] = None,
    ) -> list[DataIsland]:
        """Merge islands that are very close to each other.

        This helps handle cases where formatting creates small gaps
        that shouldn't separate tables.

        Args:
            islands: List of detected islands
            merge_distance: Maximum distance to consider for merging
            sheet_data: Optional sheet data for checking gaps between islands

        Returns:
            List of islands after merging
        """
        if len(islands) <= 1:
            return islands

        merged = []
        used = set()

        for i, island1 in enumerate(islands):
            if i in used:
                continue

            # Start a new merged island
            merged_island = DataIsland()
            merged_island.cells = island1.cells.copy()
            merged_island.min_row = island1.min_row
            merged_island.max_row = island1.max_row
            merged_island.min_col = island1.min_col
            merged_island.max_col = island1.max_col
            merged_island.total_sheet_cells = island1.total_sheet_cells

            # Check for mergeable islands
            for j, island2 in enumerate(islands[i + 1 :], i + 1):
                if j in used:
                    continue

                # Enhanced merge check with gap analysis
                should_merge = self._should_merge(island1, island2, merge_distance)

                # Additional check: if there's data in the gap, don't merge
                if should_merge and sheet_data:
                    should_merge = self._check_gap_is_empty(merged_island, island2, sheet_data)

                if should_merge:
                    merged_island.cells.update(island2.cells)
                    if island2.min_row is not None and merged_island.min_row is not None:
                        merged_island.min_row = min(merged_island.min_row, island2.min_row)
                    if island2.max_row is not None and merged_island.max_row is not None:
                        merged_island.max_row = max(merged_island.max_row, island2.max_row)
                    if island2.min_col is not None and merged_island.min_col is not None:
                        merged_island.min_col = min(merged_island.min_col, island2.min_col)
                    if island2.max_col is not None and merged_island.max_col is not None:
                        merged_island.max_col = max(merged_island.max_col, island2.max_col)
                    used.add(j)

            # Recalculate metrics for merged island if sheet_data available
            if sheet_data:
                merged_island.calculate_metrics(sheet_data)

            merged.append(merged_island)
            used.add(i)

        return merged

    def _are_islands_well_separated(
        self, islands: list[DataIsland], sheet_data: "SheetData"
    ) -> bool:
        """Check if islands are well-separated by empty rows/columns.

        Args:
            islands: List of islands to check
            sheet_data: Sheet data

        Returns:
            True if islands are separated by at least one completely empty row/column
        """
        if len(islands) <= 1:
            return False

        # Sort islands by position for easier comparison
        sorted_islands = sorted(islands, key=lambda i: (i.min_row or 0, i.min_col or 0))

        for i in range(len(sorted_islands) - 1):
            island1 = sorted_islands[i]
            island2 = sorted_islands[i + 1]

            if (
                island1.max_row is None
                or island2.min_row is None
                or island1.max_col is None
                or island2.min_col is None
            ):
                continue

            # Check if there's at least one completely empty row between islands
            if island2.min_row > island1.max_row + 1:
                # Check if the gap rows are completely empty
                gap_start = island1.max_row + 1
                gap_end = island2.min_row - 1

                # Check columns in the range that both islands span
                check_col_start = min(island1.min_col or 0, island2.min_col or 0)
                check_col_end = max(island1.max_col or 0, island2.max_col or 0)

                gap_has_data = False
                for row in range(gap_start, gap_end + 1):
                    for col in range(check_col_start, check_col_end + 1):
                        cell = sheet_data.get_cell(row, col)
                        if cell and not cell.is_empty:
                            gap_has_data = True
                            break
                    if gap_has_data:
                        break

                if not gap_has_data:
                    # At least one completely empty row exists
                    continue  # This pair is well-separated, check next pair
                else:
                    return False  # Found a pair that's not well-separated

            else:
                return False  # Islands are adjacent or overlapping

        # All consecutive pairs are well-separated
        return True

    def _should_merge(self, island1: DataIsland, island2: DataIsland, max_distance: int) -> bool:
        """Check if two islands should be merged based on proximity.

        Args
        ----
        island1 : DataIsland
            First island to check for merging.
        island2 : DataIsland
            Second island to check for merging.
        max_distance : int
            Maximum distance (in cells) for considering islands as mergeable.

        Returns
        -------
        bool
            True if islands should be merged, False otherwise.

        Notes
        -----
        This method implements sophisticated heuristics to prevent incorrect
        merging of separate tables:

        1. **Column Gap Rule**: Islands separated by 2+ empty columns are never
           merged, as this strongly indicates separate tables.

        2. **Single Column Gap**: For islands separated by exactly 1 column,
           they're only merged if they have >50% row overlap, indicating they
           might be parts of the same table with a formatting gap.

        3. **Proximity Rules**: Islands are merged if they're:
           - Adjacent or overlapping in one dimension and within max_distance
             in the other dimension
           - Within max_distance in both dimensions (diagonal proximity)

        The method prevents common errors like merging side-by-side tables
        or vertically stacked tables with clear separation.
        """
        # Check if islands have valid bounds
        if (
            island1.min_row is None
            or island1.max_row is None
            or island1.min_col is None
            or island1.max_col is None
            or island2.min_row is None
            or island2.max_row is None
            or island2.min_col is None
            or island2.max_col is None
        ):
            return False

        # Check vertical distance
        v_distance = max(
            0,
            island2.min_row - island1.max_row - 1,
            island1.min_row - island2.max_row - 1,
        )

        # Check horizontal distance
        h_distance = max(
            0,
            island2.min_col - island1.max_col - 1,
            island1.min_col - island2.max_col - 1,
        )

        # IMPROVEMENT: Don't merge if there's a significant column gap
        # This prevents merging tables that are side-by-side with empty columns between
        if h_distance > 0:
            # If islands are horizontally separated, check if they should be considered separate tables
            # Tables separated by even 1 empty column are likely distinct
            if h_distance >= 2:  # 2+ empty columns is definitely separate tables
                self.logger.debug(
                    f"Not merging islands due to column gap of {h_distance}: "
                    f"{island1.to_range()} and {island2.to_range()}"
                )
                return False

            # For single column gap, check vertical alignment
            # Only merge if they have significant row overlap (>50%)
            row_overlap_start = max(island1.min_row, island2.min_row)
            row_overlap_end = min(island1.max_row, island2.max_row)

            if row_overlap_start <= row_overlap_end:
                overlap_rows = row_overlap_end - row_overlap_start + 1
                island1_rows = island1.max_row - island1.min_row + 1
                island2_rows = island2.max_row - island2.min_row + 1
                min_rows = min(island1_rows, island2_rows)

                overlap_ratio = overlap_rows / min_rows if min_rows > 0 else 0

                if overlap_ratio < 0.5:  # Less than 50% row overlap
                    self.logger.debug(
                        f"Not merging islands with low row overlap ({overlap_ratio:.1%}): "
                        f"{island1.to_range()} and {island2.to_range()}"
                    )
                    return False

        # Check if overlapping in one dimension and close in the other
        if v_distance == 0 and h_distance <= max_distance:
            return True
        if h_distance == 0 and v_distance <= max_distance:
            return True

        # Check diagonal distance for small gaps
        return v_distance <= max_distance and h_distance <= max_distance

    def _check_gap_is_empty(
        self, island1: DataIsland, island2: DataIsland, sheet_data: "SheetData"
    ) -> bool:
        """Check if the gap between two islands is empty.

        Args
        ----
        island1 : DataIsland
            First island in the pair to check.
        island2 : DataIsland
            Second island in the pair to check.
        sheet_data : SheetData
            Sheet data to check for cells in the gap region.

        Returns
        -------
        bool
            True if gap is empty (safe to merge), False if data exists in gap.

        Notes
        -----
        This method prevents incorrect merging of tables that have scattered
        data between them. It carefully calculates the exact gap region based
        on island positions:

        - For horizontal gaps: Checks columns between islands within their
          overlapping row range
        - For vertical gaps: Checks rows between islands within their
          overlapping column range

        Even a single non-empty cell in the gap prevents merging, as it likely
        indicates the islands are separate tables with unrelated data between them.

        The method returns True (safe to merge) if islands have invalid bounds,
        following a conservative approach that avoids breaking valid merges due
        to data issues.
        """
        if (
            island1.min_row is None
            or island1.max_row is None
            or island1.min_col is None
            or island1.max_col is None
            or island2.min_row is None
            or island2.max_row is None
            or island2.min_col is None
            or island2.max_col is None
        ):
            return True  # Can't check, assume safe

        # Determine the gap region between islands
        # For horizontal gaps
        if island1.max_col < island2.min_col:
            # Island1 is to the left of island2
            gap_col_start = island1.max_col + 1
            gap_col_end = island2.min_col - 1
            gap_row_start = max(island1.min_row, island2.min_row)
            gap_row_end = min(island1.max_row, island2.max_row)
        elif island2.max_col < island1.min_col:
            # Island2 is to the left of island1
            gap_col_start = island2.max_col + 1
            gap_col_end = island1.min_col - 1
            gap_row_start = max(island1.min_row, island2.min_row)
            gap_row_end = min(island1.max_row, island2.max_row)
        else:
            gap_col_start = gap_col_end = -1

        # For vertical gaps
        if island1.max_row < island2.min_row:
            # Island1 is above island2
            gap_row_start = island1.max_row + 1
            gap_row_end = island2.min_row - 1
            gap_col_start = max(island1.min_col, island2.min_col)
            gap_col_end = min(island1.max_col, island2.max_col)
        elif island2.max_row < island1.min_row:
            # Island2 is above island1
            gap_row_start = island2.max_row + 1
            gap_row_end = island1.min_row - 1
            gap_col_start = max(island1.min_col, island2.min_col)
            gap_col_end = min(island1.max_col, island2.max_col)

        # Check if any cells exist in the gap
        if gap_col_start >= 0 and gap_col_end >= gap_col_start:
            for row in range(gap_row_start, gap_row_end + 1):
                for col in range(gap_col_start, gap_col_end + 1):
                    cell = sheet_data.get_cell(row, col)
                    if cell and not cell.is_empty:
                        self.logger.debug(
                            f"Found data at ({row},{col}) in gap between "
                            f"{island1.to_range()} and {island2.to_range()}"
                        )
                        return False  # Data exists in gap, don't merge

        return True  # Gap is empty, safe to merge

    def _has_table_end_border_pattern(
        self, prev_row: int, current_row: int, sheet_data: "SheetData"
    ) -> bool:
        """Check if there's a table-ending border pattern between two rows.

        A table end is indicated by:
        1. Previous row has consistent bottom borders
        2. Current row has no or different border pattern
        3. Optional: Empty row(s) between them

        Args:
            prev_row: Previous row index
            current_row: Current row index
            sheet_data: Sheet data

        Returns:
            True if table-ending border pattern is detected
        """
        # Get cells from previous row
        prev_cells = []
        current_cells = []

        # Find column range to check
        col_start = col_end = None
        for col in range(sheet_data.max_column + 1):
            cell = sheet_data.get_cell(prev_row, col)
            if cell and not cell.is_empty:
                if col_start is None:
                    col_start = col
                col_end = col
                prev_cells.append(cell)

        if not prev_cells or col_start is None or col_end is None:
            return False

        # Check if previous row has consistent bottom borders
        bottom_border_count = 0
        for cell in prev_cells:
            if cell.border_bottom and cell.border_bottom != "none":
                bottom_border_count += 1

        # Need at least 70% of cells to have bottom borders
        if bottom_border_count < len(prev_cells) * 0.7:
            return False

        # Check current row in same column range
        for col in range(col_start, col_end + 1):
            cell = sheet_data.get_cell(current_row, col)
            if cell and not cell.is_empty:
                current_cells.append(cell)

        if not current_cells:
            # Current row is empty in this range - strong table end indicator
            return True

        # Check if current row has different border pattern
        # Count top borders in current row (would indicate continuation)
        top_border_count = 0
        for cell in current_cells:
            if cell.border_top and cell.border_top != "none":
                top_border_count += 1

        # If current row has few top borders, it's likely a new table
        return top_border_count < len(current_cells) * 0.3

    def convert_to_table_infos(
        self,
        islands: list[DataIsland],
        sheet_name: str,
        min_confidence: float = 0.3,
        sheet_data: Optional["SheetData"] = None,
    ) -> list[TableInfo]:
        """Convert data islands to TableInfo objects.

        Args:
            islands: List of detected islands
            sheet_name: Name of the sheet
            min_confidence: Minimum confidence threshold

        Returns:
            List of TableInfo objects for qualifying islands
        """
        table_infos = []

        for i, island in enumerate(islands):
            if island.confidence < min_confidence:
                self.logger.debug(
                    f"Skipping island {i} with low confidence {island.confidence:.2f}"
                )
                continue

            # Skip islands with invalid bounds
            if (
                island.min_row is None
                or island.max_row is None
                or island.min_col is None
                or island.max_col is None
            ):
                self.logger.warning(f"Skipping island {i} with invalid bounds")
                continue

            table_range = TableRange(
                start_row=island.min_row,
                start_col=island.min_col,
                end_row=island.max_row,
                end_col=island.max_col,
            )

            # Always extract headers from first row if sheet_data is available
            headers = None
            if sheet_data:
                headers = self._extract_headers(sheet_data, table_range)
                self.logger.debug(f"Extracted headers for island {i}: {headers}")

            table_info = TableInfo(
                id=f"island_{island.min_row}_{island.min_col}",
                range=table_range,
                suggested_name=f"{sheet_name}_table_{i + 1}",
                confidence=island.confidence,
                detection_method="island_detection",
                has_headers=island.has_headers,
                headers=headers,
                data_preview=None,  # Would need to extract if needed
            )
            table_infos.append(table_info)

        return table_infos

    def _extract_headers(self, sheet_data: "SheetData", table_range: TableRange) -> list[str]:
        """Extract header values from the first row of a table.

        Args
        ----
        sheet_data : SheetData
            Sheet data containing the cells to extract headers from.
        table_range : TableRange
            Range defining the table boundaries. Headers are extracted from
            the first row (start_row) of this range.

        Returns
        -------
        list[str]
            List of header strings, one for each column in the table range.
            Empty cells get their column letter as a fallback (e.g., "A", "B").

        Examples
        --------
        >>> headers = detector._extract_headers(sheet_data, table_range)
        >>> print(headers)
        ['Date', 'Product', 'Quantity', 'Price', 'Total']

        Notes
        -----
        This method always extracts from the first row of the table range,
        regardless of whether headers were detected. The assumption is that
        most tables have headers in the first row, and it's better to extract
        potential headers than to miss them.

        All values are converted to strings and stripped of whitespace. Empty
        cells receive their Excel column letter as a placeholder to ensure
        all columns have identifiers.
        """
        headers = []

        # Extract values from first row
        for col in range(table_range.start_col, table_range.end_col + 1):
            cell = sheet_data.get_cell(table_range.start_row, col)
            if cell and cell.value is not None:
                # Convert value to string for header
                header_val = str(cell.value).strip()
                headers.append(header_val)
            else:
                # Use column letter as fallback for empty headers
                from ..utils.excel_utils import get_column_letter

                headers.append(get_column_letter(col))

        return headers

    def _detect_islands_structural(self, sheet_data: "SheetData") -> list[DataIsland]:
        """Detect islands using structural analysis based on column consistency.

        This method groups rows based on their column usage patterns and splits
        regions when column patterns change significantly or empty rows are found.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of DataIsland objects with better separation
        """
        # First, analyze row patterns
        row_patterns = self._analyze_row_patterns(sheet_data)

        # Group consecutive rows with similar patterns
        row_groups = self._group_rows_by_pattern(row_patterns)

        # Calculate adaptive threshold for structural analysis
        total_sheet_cells = len(sheet_data.cells)
        if self.adaptive_thresholds and total_sheet_cells > 1000:
            effective_min_size = max(self.min_island_size, int(total_sheet_cells * 0.005))
        else:
            effective_min_size = self.min_island_size

        # Convert row groups to islands
        islands = []
        for group in row_groups:
            if len(group) >= effective_min_size:
                island = self._create_island_from_row_group(sheet_data, group)
                if island and len(island.cells) >= effective_min_size:
                    island.total_sheet_cells = total_sheet_cells
                    island.calculate_metrics(sheet_data)
                    islands.append(island)

        # Sort islands by position
        islands.sort(key=lambda i: (i.min_row, i.min_col))

        # Calculate metrics for each island (already done in loop above)

        self.logger.info(f"Detected {len(islands)} islands using structural analysis")
        return islands

    def _analyze_row_patterns(
        self, sheet_data: "SheetData"
    ) -> dict[int, tuple[int, int, set[int]]]:
        """Analyze column usage patterns for each row.

        Returns:
            Dict mapping row index to (min_col, max_col, col_set)
        """
        patterns = {}

        # Get all cells organized by row
        rows_data: dict[int, list[int]] = {}
        for _address, cell in sheet_data.cells.items():
            if not cell.is_empty:
                row = cell.row
                if row not in rows_data:
                    rows_data[row] = []
                rows_data[row].append(cell.column)

        # Analyze each row
        for row, cols in rows_data.items():
            if cols:
                min_col = min(cols)
                max_col = max(cols)
                col_set = set(cols)
                patterns[row] = (min_col, max_col, col_set)

        return patterns

    def _group_rows_by_pattern(
        self, row_patterns: dict[int, tuple[int, int, set[int]]]
    ) -> list[list[int]]:
        """Group consecutive rows with similar column patterns.

        Args:
            row_patterns: Dict mapping row to (min_col, max_col, col_set)

        Returns:
            List of row groups (each group is a list of row indices)
        """
        if not row_patterns:
            return []

        # Sort rows
        sorted_rows = sorted(row_patterns.keys())

        groups = []
        current_group = [sorted_rows[0]]
        prev_row = sorted_rows[0]
        prev_pattern = row_patterns[prev_row]

        for row in sorted_rows[1:]:
            pattern = row_patterns[row]

            # Check if there's an empty row gap
            empty_row_gap = row - prev_row - 1

            # Use empty_row_tolerance to decide if we should split
            if empty_row_gap > self.empty_row_tolerance:
                # Only split if gap exceeds tolerance
                # Check if columns are still aligned despite the gap
                similarity = self._calculate_column_similarity(prev_pattern, pattern)

                if (
                    similarity < self.column_consistency_threshold
                    or empty_row_gap > self.min_empty_rows_to_split
                ):
                    # Start new group due to large gap or different columns
                    groups.append(current_group)
                    current_group = [row]
                    self.logger.debug(
                        f"Starting new group at row {row} due to gap of {empty_row_gap} rows"
                    )
                else:
                    # Tolerate the gap - columns are still aligned
                    current_group.append(row)
                    self.logger.debug(
                        f"Tolerating gap of {empty_row_gap} rows at row {row} due to column alignment"
                    )
            else:
                # Small or no gap - check column similarity as usual
                similarity = self._calculate_column_similarity(prev_pattern, pattern)

                if similarity >= self.column_consistency_threshold:
                    # Add to current group
                    current_group.append(row)
                else:
                    # Start new group due to different column pattern
                    groups.append(current_group)
                    current_group = [row]

            prev_row = row
            prev_pattern = pattern

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _calculate_column_similarity(
        self, pattern1: tuple[int, int, set[int]], pattern2: tuple[int, int, set[int]]
    ) -> float:
        """Calculate similarity between two column patterns.

        Args:
            pattern1: (min_col, max_col, col_set) for first row
            pattern2: (min_col, max_col, col_set) for second row

        Returns:
            Similarity score between 0 and 1
        """
        min1, max1, cols1 = pattern1
        min2, max2, cols2 = pattern2

        # Check range overlap
        range_overlap = min(max1, max2) - max(min1, min2) + 1
        range_union = max(max1, max2) - min(min1, min2) + 1

        if range_union <= 0:
            return 0.0

        range_similarity = range_overlap / range_union

        # Check column set similarity (Jaccard index)
        if not cols1 and not cols2:
            cols_similarity = 1.0
        elif not cols1 or not cols2:
            cols_similarity = 0.0
        else:
            intersection = len(cols1 & cols2)
            union = len(cols1 | cols2)
            cols_similarity = intersection / union if union > 0 else 0.0

        # Weighted average
        return 0.5 * range_similarity + 0.5 * cols_similarity

    def _create_island_from_row_group(
        self, sheet_data: "SheetData", rows: list[int]
    ) -> DataIsland | None:
        """Create an island from a group of rows.

        Args:
            sheet_data: Sheet data
            rows: List of row indices in the group

        Returns:
            DataIsland or None if no data found
        """
        island = DataIsland()

        for row in rows:
            # Add all cells in this row
            for _address, cell in sheet_data.cells.items():
                if cell.row == row and not cell.is_empty:
                    island.add_cell(cell.row, cell.column)

        return island if island.cells else None

    def _check_subset_relationships(self, islands: list[DataIsland]) -> None:
        """Check if any island is a subset of another island.

        Updates the is_subset_of property for islands that are completely
        contained within other islands.

        Args:
            islands: List of islands to check (should be sorted by size, largest first)
        """
        # Compare each island with larger islands
        for i in range(len(islands)):
            smaller = islands[i]
            if smaller.is_subset_of is not None:
                continue  # Already marked as subset

            # Only check against larger islands (which come before in the sorted list)
            for j in range(i):
                larger = islands[j]

                # Check if smaller is contained in larger
                if self._is_island_subset(smaller, larger):
                    smaller.is_subset_of = larger
                    self.logger.debug(
                        f"Island at {smaller.to_range()} is subset of {larger.to_range()}"
                    )
                    break

    def _is_island_subset(self, island1: DataIsland, island2: DataIsland) -> bool:
        """Check if island1 is completely contained within island2.

        Args:
            island1: Potentially smaller island
            island2: Potentially larger island

        Returns:
            True if island1 is a subset of island2
        """
        # Quick bounds check
        if (
            island1.min_row is None
            or island1.max_row is None
            or island1.min_col is None
            or island1.max_col is None
            or island2.min_row is None
            or island2.max_row is None
            or island2.min_col is None
            or island2.max_col is None
        ):
            return False

        # Check if island1 bounds are within island2 bounds
        if not (
            island2.min_row <= island1.min_row
            and island1.max_row <= island2.max_row
            and island2.min_col <= island1.min_col
            and island1.max_col <= island2.max_col
        ):
            return False

        # Check if all cells of island1 are in island2
        # (This handles cases where islands have gaps)
        return island1.cells.issubset(island2.cells)

    def _analyze_row_formatting(self, sheet_data: "SheetData", row: int) -> dict[str, Any]:
        """Analyze formatting characteristics of a row.

        Args:
            sheet_data: Sheet data to analyze
            row: Row index to analyze

        Returns:
            Dictionary with row formatting characteristics
        """
        row_cells = []
        for col in range(sheet_data.max_column + 1):
            cell = sheet_data.get_cell(row, col)
            if cell and not cell.is_empty:
                row_cells.append(cell)

        if not row_cells:
            return {
                "has_data": False,
                "bold_ratio": 0.0,
                "background_colors": set(),
                "font_colors": set(),
                "font_sizes": set(),
                "is_likely_header": False,
                "formatting_consistency": 0.0,
                "border_signature": FORMATTING_DETECTION.NO_BORDERS,
            }

        # Calculate formatting metrics
        bold_count = sum(1 for cell in row_cells if cell.is_bold)
        bold_ratio = bold_count / len(row_cells)

        background_colors = {cell.background_color for cell in row_cells if cell.background_color}
        font_colors = {cell.font_color for cell in row_cells if cell.font_color}
        font_sizes = {cell.font_size for cell in row_cells if cell.font_size}

        # Determine if this looks like a header row
        is_likely_header = bold_ratio >= FORMATTING_DETECTION.HEADER_BOLD_THRESHOLD or (
            len(background_colors) == 1 and len(row_cells) > 1
        )  # Consistent background

        # Calculate formatting consistency (how similar cells are in this row)
        consistency_factors = []
        if bold_count == 0 or bold_count == len(row_cells):
            consistency_factors.append(1.0)  # All same bold state
        else:
            consistency_factors.append(
                max(bold_count, len(row_cells) - bold_count) / len(row_cells)
            )

        if len(background_colors) <= 1:
            consistency_factors.append(1.0)  # Same or no background
        else:
            consistency_factors.append(0.5)  # Mixed backgrounds

        if len(font_colors) <= 1:
            consistency_factors.append(1.0)  # Same or no font color
        else:
            consistency_factors.append(0.5)  # Mixed font colors

        formatting_consistency = sum(consistency_factors) / len(consistency_factors)

        return {
            "has_data": True,
            "bold_ratio": bold_ratio,
            "background_colors": background_colors,
            "font_colors": font_colors,
            "font_sizes": font_sizes,
            "is_likely_header": is_likely_header,
            "formatting_consistency": formatting_consistency,
            "cell_count": len(row_cells),
            "border_signature": self._get_border_signature(row_cells),
        }

    def _detect_formatting_boundaries(
        self, sheet_data: "SheetData", row_start: int, row_end: int
    ) -> list[int]:
        """Detect potential table boundaries based on formatting changes.

        This method identifies logical table boundaries by looking for transitions from
        data rows to header rows, ensuring headers stay with their data sections.

        Args:
            sheet_data: Sheet data to analyze
            row_start: Starting row index
            row_end: Ending row index

        Returns:
            List of row indices where new tables begin (before header rows)
        """
        # First pass: analyze all rows and identify header/data patterns
        row_analysis = {}
        for row in range(row_start, row_end + 1):
            row_analysis[row] = self._analyze_row_formatting(sheet_data, row)

        # Second pass: identify logical table boundaries
        boundaries: list[int] = []
        prev_row = None
        current_table_start = None

        for row in range(row_start, row_end + 1):
            current = row_analysis[row]

            if not current["has_data"]:
                continue

            if prev_row is not None:
                assert prev_row is not None  # Help mypy understand
                prev = row_analysis[prev_row]

                # Check if this looks like a new table start
                is_new_table_start = False

                # Case 1: Transition from data to header (new table starting)
                if not prev["is_likely_header"] and current["is_likely_header"]:
                    is_new_table_start = True

                # Case 2: Significant border pattern change indicating new section
                border_similarity = self._calculate_border_similarity(
                    current["border_signature"], prev["border_signature"]
                )
                if border_similarity < FORMATTING_DETECTION.BORDER_CONSISTENCY_THRESHOLD:
                    is_new_table_start = True

                # Case 2b: Check for "table end" border pattern
                # If previous row had bottom borders and current doesn't, likely table boundary
                if self._has_table_end_border_pattern(prev_row, row, sheet_data):
                    is_new_table_start = True
                    self.logger.debug(
                        f"Table end border pattern detected at row {row}: "
                        f"Previous row has bottom borders, current row starts new table"
                    )

                # Case 3: Major formatting shift (combined factors)
                formatting_change_score = 0.0

                # Bold ratio change
                bold_diff = abs(current["bold_ratio"] - prev["bold_ratio"])
                if bold_diff > 0.5:
                    formatting_change_score += 0.3

                # Consistency change
                consistency_diff = abs(
                    current["formatting_consistency"] - prev["formatting_consistency"]
                )
                if consistency_diff > 0.4:
                    formatting_change_score += 0.2

                # Font/background changes
                if current["background_colors"] != prev["background_colors"]:
                    formatting_change_score += 0.2

                if current["font_colors"] != prev["font_colors"]:
                    formatting_change_score += 0.1

                # If major formatting shift AND looks like header, it's a new table
                if (
                    formatting_change_score >= FORMATTING_DETECTION.BACKGROUND_CHANGE_THRESHOLD
                    and current["is_likely_header"]
                    and not prev["is_likely_header"]
                ):
                    is_new_table_start = True

                # Mark boundary at start of new table (include the header)
                if is_new_table_start and current_table_start is not None:
                    boundaries.append(row)  # Split before this header row
                    logger.debug(
                        f"Table boundary detected before header at row {row + 1}: "
                        f"prev_header={prev['is_likely_header']}, curr_header={current['is_likely_header']}"
                    )

            # Track table start
            if current_table_start is None and current["has_data"]:
                current_table_start = row

            prev_row = row

        return boundaries

    def _calculate_formatting_similarity(
        self, format1: dict[str, Any], format2: dict[str, Any]
    ) -> float:
        """Calculate similarity between two formatting profiles.

        Args:
            format1: First formatting profile
            format2: Second formatting profile

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not format1["has_data"] or not format2["has_data"]:
            return 0.0

        similarity_factors = []

        # Bold ratio similarity
        bold_similarity = 1.0 - abs(format1["bold_ratio"] - format2["bold_ratio"])
        similarity_factors.append(bold_similarity)

        # Background color similarity
        bg1, bg2 = format1["background_colors"], format2["background_colors"]
        if bg1 == bg2:
            similarity_factors.append(1.0)
        elif not bg1 and not bg2:
            similarity_factors.append(1.0)  # Both have no background
        else:
            # Some overlap or difference
            if bg1 and bg2:
                overlap = len(bg1.intersection(bg2))
                union = len(bg1.union(bg2))
                similarity_factors.append(overlap / union if union > 0 else 0.0)
            else:
                similarity_factors.append(0.5)  # One has background, other doesn't

        # Header pattern similarity
        if format1["is_likely_header"] == format2["is_likely_header"]:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.3)  # Different header patterns

        # Consistency similarity
        consistency_similarity = 1.0 - abs(
            format1["formatting_consistency"] - format2["formatting_consistency"]
        )
        similarity_factors.append(consistency_similarity)

        # Border pattern similarity
        border_similarity = self._calculate_border_similarity(
            format1.get("border_signature", FORMATTING_DETECTION.NO_BORDERS),
            format2.get("border_signature", FORMATTING_DETECTION.NO_BORDERS),
        )
        similarity_factors.append(border_similarity)

        return float(sum(similarity_factors) / len(similarity_factors))

    def _get_border_signature(self, row_cells: list) -> str:
        """Get border signature for a row of cells.

        Args:
            row_cells: List of CellData objects in the row

        Returns:
            Border signature string indicating border pattern
        """
        if not row_cells:
            return FORMATTING_DETECTION.NO_BORDERS

        # Count border patterns with more detail
        border_patterns = {
            "all": 0,
            "none": 0,
            "horizontal": 0,
            "vertical": 0,
            "mixed": 0,
            "outer": 0,  # Special pattern for table boundaries
        }

        # Track if this might be a table boundary row
        is_first_cell = True
        is_last_cell = False

        for i, cell in enumerate(row_cells):
            is_last_cell = i == len(row_cells) - 1

            has_top = cell.border_top is not None and cell.border_top != "none"
            has_bottom = cell.border_bottom is not None and cell.border_bottom != "none"
            has_left = cell.border_left is not None and cell.border_left != "none"
            has_right = cell.border_right is not None and cell.border_right != "none"

            border_count = sum([has_top, has_bottom, has_left, has_right])

            if border_count == 0:
                border_patterns["none"] += 1
            elif border_count == 4:
                border_patterns["all"] += 1
            elif is_first_cell and has_left and (has_top or has_bottom):
                # First cell with left border - might be table start
                border_patterns["outer"] += 1
            elif is_last_cell and has_right and (has_top or has_bottom):
                # Last cell with right border - might be table end
                border_patterns["outer"] += 1
            elif has_top or has_bottom:
                if not has_left and not has_right:
                    border_patterns["horizontal"] += 1
                else:
                    border_patterns["mixed"] += 1
            elif has_left or has_right:
                if not has_top and not has_bottom:
                    border_patterns["vertical"] += 1
                else:
                    border_patterns["mixed"] += 1
            else:
                border_patterns["mixed"] += 1

            is_first_cell = False

        # Find most common pattern
        max_count = max(border_patterns.values())
        if max_count == 0:
            return FORMATTING_DETECTION.NO_BORDERS

        # Return the most common pattern
        pattern_mapping = {
            "all": FORMATTING_DETECTION.ALL_BORDERS,
            "none": FORMATTING_DETECTION.NO_BORDERS,
            "horizontal": FORMATTING_DETECTION.HORIZONTAL_ONLY,
            "vertical": FORMATTING_DETECTION.VERTICAL_ONLY,
            "mixed": FORMATTING_DETECTION.MIXED_BORDERS,
            "outer": FORMATTING_DETECTION.OUTER_ONLY,
        }

        for pattern, count in border_patterns.items():
            if count == max_count:
                return pattern_mapping.get(pattern, FORMATTING_DETECTION.MIXED_BORDERS)

        return FORMATTING_DETECTION.MIXED_BORDERS

    def _calculate_border_similarity(self, signature1: str, signature2: str) -> float:
        """Calculate similarity between two border signatures.

        Args:
            signature1: First border signature
            signature2: Second border signature

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if signature1 == signature2:
            return 1.0

        # Define similarity matrix for border patterns
        similarity_matrix = {
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.NO_BORDERS): 1.0,
            (FORMATTING_DETECTION.ALL_BORDERS, FORMATTING_DETECTION.ALL_BORDERS): 1.0,
            (FORMATTING_DETECTION.HORIZONTAL_ONLY, FORMATTING_DETECTION.HORIZONTAL_ONLY): 1.0,
            (FORMATTING_DETECTION.VERTICAL_ONLY, FORMATTING_DETECTION.VERTICAL_ONLY): 1.0,
            (FORMATTING_DETECTION.MIXED_BORDERS, FORMATTING_DETECTION.MIXED_BORDERS): 1.0,
            (FORMATTING_DETECTION.OUTER_ONLY, FORMATTING_DETECTION.OUTER_ONLY): 1.0,
            # Similar patterns have moderate similarity
            (FORMATTING_DETECTION.HORIZONTAL_ONLY, FORMATTING_DETECTION.ALL_BORDERS): 0.7,
            (FORMATTING_DETECTION.VERTICAL_ONLY, FORMATTING_DETECTION.ALL_BORDERS): 0.7,
            (FORMATTING_DETECTION.HORIZONTAL_ONLY, FORMATTING_DETECTION.MIXED_BORDERS): 0.6,
            (FORMATTING_DETECTION.VERTICAL_ONLY, FORMATTING_DETECTION.MIXED_BORDERS): 0.6,
            (FORMATTING_DETECTION.ALL_BORDERS, FORMATTING_DETECTION.MIXED_BORDERS): 0.8,
            (FORMATTING_DETECTION.OUTER_ONLY, FORMATTING_DETECTION.ALL_BORDERS): 0.8,
            (FORMATTING_DETECTION.OUTER_ONLY, FORMATTING_DETECTION.MIXED_BORDERS): 0.7,
            # No borders vs any borders has low similarity
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.ALL_BORDERS): 0.2,
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.HORIZONTAL_ONLY): 0.3,
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.VERTICAL_ONLY): 0.3,
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.MIXED_BORDERS): 0.2,
            (FORMATTING_DETECTION.NO_BORDERS, FORMATTING_DETECTION.OUTER_ONLY): 0.2,
            # Different directional borders have moderate similarity
            (FORMATTING_DETECTION.HORIZONTAL_ONLY, FORMATTING_DETECTION.VERTICAL_ONLY): 0.5,
            (FORMATTING_DETECTION.HORIZONTAL_ONLY, FORMATTING_DETECTION.OUTER_ONLY): 0.6,
            (FORMATTING_DETECTION.VERTICAL_ONLY, FORMATTING_DETECTION.OUTER_ONLY): 0.6,
        }

        # Check both directions
        key1 = (signature1, signature2)
        key2 = (signature2, signature1)

        return similarity_matrix.get(key1, similarity_matrix.get(key2, 0.3))

    def _apply_formatting_splits(
        self, islands: list[DataIsland], sheet_data: "SheetData"
    ) -> list[DataIsland]:
        """Apply formatting-based splits to large islands that span multiple visual tables.

        Args:
            islands: List of detected islands
            sheet_data: Sheet data with formatting information

        Returns:
            List of islands after formatting-based splitting
        """
        split_islands = []

        for island in islands:
            island_size = (
                island.max_row - island.min_row
                if (island.min_row is not None and island.max_row is not None)
                else 0
            )
            self.logger.debug(f"Checking island {island.to_range()}: size={island_size}")

            if (
                island.min_row is not None and island.max_row is not None and island_size > 3
            ):  # Only split larger islands
                # Detect formatting boundaries within this island
                boundaries = self._detect_formatting_boundaries(
                    sheet_data, island.min_row, island.max_row
                )
                self.logger.debug(f"Island {island.to_range()} boundaries: {boundaries}")

                if boundaries:
                    # Split the island at formatting boundaries
                    split_parts = self._split_island_at_boundaries(island, boundaries, sheet_data)
                    split_islands.extend(split_parts)
                    self.logger.debug(
                        f"Split island {island.to_range()} into {len(split_parts)} parts "
                        f"at boundaries: {boundaries}"
                    )
                else:
                    self.logger.debug(f"No boundaries found for island {island.to_range()}")
                    split_islands.append(island)
            else:
                self.logger.debug(
                    f"Island {island.to_range()} too small for splitting (size={island_size})"
                )
                split_islands.append(island)

        return split_islands

    def _split_island_at_boundaries(
        self, island: DataIsland, boundaries: list[int], sheet_data: "SheetData"
    ) -> list[DataIsland]:
        """Split an island at the specified row boundaries.

        Args:
            island: Island to split
            boundaries: List of row indices to split at
            sheet_data: Sheet data

        Returns:
            List of new islands after splitting
        """
        if not boundaries or island.min_row is None or island.max_row is None:
            return [island]

        # Create row ranges between boundaries
        row_ranges = []
        start_row = island.min_row

        for boundary in sorted(boundaries):
            if start_row < boundary:
                row_ranges.append((start_row, boundary - 1))
            start_row = boundary

        # Add final range
        if start_row <= island.max_row:
            row_ranges.append((start_row, island.max_row))

        # Create new islands for each row range
        new_islands = []
        total_sheet_cells = island.total_sheet_cells

        for start_row, end_row in row_ranges:
            new_island = DataIsland()
            new_island.total_sheet_cells = total_sheet_cells

            # Add cells from original island that fall in this row range
            for cell_pos in island.cells:
                row, col = cell_pos
                if start_row <= row <= end_row:
                    new_island.add_cell(row, col)

            # Only keep islands with sufficient cells
            if len(new_island.cells) >= self.min_island_size:
                new_island.calculate_metrics(sheet_data)
                new_islands.append(new_island)

        # If no valid splits were created, return original island
        return new_islands if new_islands else [island]

    def _reconnect_gap_separated_islands(
        self, islands: list[DataIsland], sheet_data: "SheetData"
    ) -> list[DataIsland]:
        """Reconnect islands that were separated by tolerable empty rows.

        This handles cases where tables have empty rows in the middle but
        maintain column alignment and should be treated as a single table.

        Args:
            islands: List of detected islands
            sheet_data: Sheet data

        Returns:
            List of islands after reconnection
        """
        if len(islands) <= 1:
            return islands

        # Sort islands by position for easier comparison
        sorted_islands = sorted(islands, key=lambda i: (i.min_row or 0, i.min_col or 0))
        reconnected = []
        skip_indices = set()

        for i, island1 in enumerate(sorted_islands):
            if i in skip_indices:
                continue

            # Start with current island
            merged_island = DataIsland()
            merged_island.cells = island1.cells.copy()
            merged_island.min_row = island1.min_row
            merged_island.max_row = island1.max_row
            merged_island.min_col = island1.min_col
            merged_island.max_col = island1.max_col
            merged_island.total_sheet_cells = island1.total_sheet_cells

            # Look for islands to reconnect
            for j in range(i + 1, len(sorted_islands)):
                if j in skip_indices:
                    continue

                island2 = sorted_islands[j]

                # Check if islands can be reconnected
                if self._should_reconnect_islands(merged_island, island2, sheet_data):
                    # Merge the islands
                    merged_island.cells.update(island2.cells)
                    if island2.min_row is not None and merged_island.min_row is not None:
                        merged_island.min_row = min(merged_island.min_row, island2.min_row)
                    if island2.max_row is not None and merged_island.max_row is not None:
                        merged_island.max_row = max(merged_island.max_row, island2.max_row)
                    if island2.min_col is not None and merged_island.min_col is not None:
                        merged_island.min_col = min(merged_island.min_col, island2.min_col)
                    if island2.max_col is not None and merged_island.max_col is not None:
                        merged_island.max_col = max(merged_island.max_col, island2.max_col)
                    skip_indices.add(j)

                    self.logger.debug(
                        f"Reconnected islands {island1.to_range()} and {island2.to_range()} "
                        f"separated by empty rows"
                    )

            # Recalculate metrics for reconnected island
            merged_island.calculate_metrics(sheet_data)
            reconnected.append(merged_island)

        return reconnected

    def _should_reconnect_islands(
        self, island1: DataIsland, island2: DataIsland, sheet_data: "SheetData"
    ) -> bool:
        """Check if two islands should be reconnected across empty rows.

        Islands are reconnected if:
        1. They are vertically separated by <= empty_row_tolerance rows
        2. They have significant column overlap (>= 50%)
        3. The gap between them is mostly empty
        4. They have similar column patterns

        Args:
            island1: First island (typically above)
            island2: Second island (typically below)
            sheet_data: Sheet data

        Returns:
            True if islands should be reconnected
        """
        if (
            island1.min_row is None
            or island1.max_row is None
            or island1.min_col is None
            or island1.max_col is None
            or island2.min_row is None
            or island2.max_row is None
            or island2.min_col is None
            or island2.max_col is None
        ):
            return False

        # Check if islands are vertically aligned (one above the other)
        if island2.min_row <= island1.max_row:
            return False  # Islands overlap or wrong order

        # Check vertical gap
        v_gap = island2.min_row - island1.max_row - 1
        if v_gap > self.empty_row_tolerance or v_gap < 1:
            return False  # Gap too large or no gap

        # Check column overlap
        col_start = max(island1.min_col, island2.min_col)
        col_end = min(island1.max_col, island2.max_col)

        if col_start > col_end:
            return False  # No column overlap

        overlap_cols = col_end - col_start + 1
        island1_cols = island1.max_col - island1.min_col + 1
        island2_cols = island2.max_col - island2.min_col + 1
        min_cols = min(island1_cols, island2_cols)

        overlap_ratio = overlap_cols / min_cols if min_cols > 0 else 0
        if overlap_ratio < 0.5:
            return False  # Insufficient column overlap

        # Check if gap is mostly empty
        gap_cells = 0
        for row in range(island1.max_row + 1, island2.min_row):
            for col in range(col_start, col_end + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    gap_cells += 1

        # Allow some cells in gap (e.g., section headers) but not too many
        max_gap_cells = overlap_cols * 0.2  # Max 20% of columns can have data in gap
        return gap_cells <= max_gap_cells
