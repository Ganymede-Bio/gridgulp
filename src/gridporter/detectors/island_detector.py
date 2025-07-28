"""Island detection algorithm for finding disconnected table regions.

This module implements connected component analysis to identify separate
"islands" of data that likely represent individual tables.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.constants import ISLAND_DETECTION
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

        # Calculate density
        total_cells = (self.max_row - self.min_row + 1) * (self.max_col - self.min_col + 1)
        self.density = len(self.cells) / total_cells if total_cells > 0 else 0

        # Detect headers (simple check - all text in first row)
        first_row_cells = [
            sheet_data.get_cell(self.min_row, col) for col in range(self.min_col, self.max_col + 1)
        ]
        self.has_headers = all(
            cell and cell.value is not None and cell.data_type == "string"
            for cell in first_row_cells
        )

        # Calculate confidence based on various factors
        self.confidence = self._calculate_confidence()

    def _calculate_confidence(self) -> float:
        """Calculate confidence score for this island being a table."""
        confidence = ISLAND_DETECTION.BASE_CONFIDENCE

        # Size factors
        cell_count = len(self.cells)
        if cell_count >= ISLAND_DETECTION.MIN_CELLS_GOOD:
            confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_LARGE
        elif cell_count >= ISLAND_DETECTION.MIN_CELLS_MEDIUM:
            confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_MEDIUM
        elif cell_count < ISLAND_DETECTION.MIN_CELLS_SMALL:
            confidence -= ISLAND_DETECTION.CONFIDENCE_PENALTY_SMALL

        # Density factor
        if self.density > ISLAND_DETECTION.DENSITY_HIGH:
            confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_LARGE
        elif self.density > ISLAND_DETECTION.DENSITY_MEDIUM:
            confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_MEDIUM
        elif self.density < ISLAND_DETECTION.DENSITY_LOW:
            confidence -= ISLAND_DETECTION.CONFIDENCE_PENALTY_LOW_DENSITY

        # Shape factor (prefer rectangular)
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
                    confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_MEDIUM

        # Header detection
        if self.has_headers:
            confidence += ISLAND_DETECTION.CONFIDENCE_BOOST_MEDIUM

        return min(max(confidence, 0.0), 1.0)

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
        max_gap: int = 1,
        min_island_size: int = 4,
        include_diagonal: bool = True,
    ):
        """Initialize the island detector.

        Args:
            max_gap: Maximum gap between cells to consider them connected
            min_island_size: Minimum number of cells to consider as an island
            include_diagonal: Whether diagonal cells are considered connected
        """
        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}")
        if min_island_size < 1:
            raise ValueError(f"min_island_size must be positive, got {min_island_size}")

        self.logger = logger
        self.max_gap = max_gap
        self.min_island_size = min_island_size
        self.include_diagonal = include_diagonal

    def detect_islands(self, sheet_data: "SheetData") -> list[DataIsland]:
        """Detect all data islands in the sheet.

        Uses flood-fill algorithm to find connected components of data.

        Args:
            sheet_data: Sheet data to analyze

        Returns:
            List of DataIsland objects representing disconnected regions
        """
        if sheet_data is None:
            raise ValueError("sheet_data cannot be None")

        if not sheet_data.has_data():
            return []

        # Get all cells with data
        data_cells = {
            (cell.row, cell.column)
            for address, cell in sheet_data.cells.items()
            if cell.value is not None
        }

        # Track visited cells
        visited = set()
        islands = []

        # Find all islands using flood-fill
        for cell_pos in data_cells:
            if cell_pos not in visited:
                island = self._flood_fill(cell_pos, data_cells, visited)
                if len(island.cells) >= self.min_island_size:
                    island.calculate_metrics(sheet_data)
                    islands.append(island)

        # Sort islands by size (largest first) and position
        islands.sort(key=lambda i: (-len(i.cells), i.min_row, i.min_col))

        self.logger.info(f"Detected {len(islands)} data islands")
        return islands

    def _flood_fill(
        self,
        start: tuple[int, int],
        data_cells: set[tuple[int, int]],
        visited: set[tuple[int, int]],
    ) -> DataIsland:
        """Perform flood-fill to find connected component.

        Args:
            start: Starting cell position (row, col)
            data_cells: Set of all cells with data
            visited: Set of already visited cells

        Returns:
            DataIsland containing all connected cells
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

        Args:
            row: Current row
            col: Current column

        Returns:
            List of neighboring cell positions
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
        self, islands: list[DataIsland], merge_distance: int = 2
    ) -> list[DataIsland]:
        """Merge islands that are very close to each other.

        This helps handle cases where formatting creates small gaps
        that shouldn't separate tables.

        Args:
            islands: List of detected islands
            merge_distance: Maximum distance to consider for merging

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

            # Check for mergeable islands
            for j, island2 in enumerate(islands[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check if islands are close enough to merge
                if self._should_merge(island1, island2, merge_distance):
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

            merged.append(merged_island)
            used.add(i)

        return merged

    def _should_merge(self, island1: DataIsland, island2: DataIsland, max_distance: int) -> bool:
        """Check if two islands should be merged based on proximity.

        Args:
            island1: First island
            island2: Second island
            max_distance: Maximum distance for merging

        Returns:
            True if islands should be merged
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

        # Check if overlapping in one dimension and close in the other
        if v_distance == 0 and h_distance <= max_distance:
            return True
        if h_distance == 0 and v_distance <= max_distance:
            return True

        # Check diagonal distance for small gaps
        return v_distance <= max_distance and h_distance <= max_distance

    def convert_to_table_infos(
        self, islands: list[DataIsland], sheet_name: str, min_confidence: float = 0.3
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

            table_info = TableInfo(
                id=f"island_{island.min_row}_{island.min_col}",
                range=table_range,
                suggested_name=f"{sheet_name}_table_{i + 1}",
                confidence=island.confidence,
                detection_method="island_detection",
                headers=None,  # Would need to extract if needed
                data_preview=None,  # Would need to extract if needed
            )
            table_infos.append(table_info)

        return table_infos
