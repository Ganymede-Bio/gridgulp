"""Quadtree implementation for efficient spatial analysis of large spreadsheets."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..models.sheet_data import SheetData
from .pattern_detector import TableBounds, TablePattern

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Type of quadtree node."""

    EMPTY = "empty"  # No data in this region
    SPARSE = "sparse"  # Few cells with data
    DENSE = "dense"  # Many cells with data
    MIXED = "mixed"  # Has child nodes (internal node)


@dataclass
class QuadTreeBounds:
    """Bounds of a quadtree node."""

    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def width(self) -> int:
        """Width of the bounds in cells."""
        return self.max_col - self.min_col + 1

    @property
    def height(self) -> int:
        """Height of the bounds in cells."""
        return self.max_row - self.min_row + 1

    @property
    def area(self) -> int:
        """Total area in cells."""
        return self.width * self.height

    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is within these bounds."""
        return self.min_row <= row <= self.max_row and self.min_col <= col <= self.max_col

    def intersects(self, other: "QuadTreeBounds") -> bool:
        """Check if these bounds intersect with another."""
        return not (
            self.max_row < other.min_row
            or self.min_row > other.max_row
            or self.max_col < other.min_col
            or self.min_col > other.max_col
        )

    def split(
        self,
    ) -> tuple["QuadTreeBounds", "QuadTreeBounds", "QuadTreeBounds", "QuadTreeBounds"]:
        """Split bounds into four quadrants."""
        mid_row = (self.min_row + self.max_row) // 2
        mid_col = (self.min_col + self.max_col) // 2

        # Top-left
        tl = QuadTreeBounds(self.min_row, self.min_col, mid_row, mid_col)
        # Top-right
        tr = QuadTreeBounds(self.min_row, mid_col + 1, mid_row, self.max_col)
        # Bottom-left
        bl = QuadTreeBounds(mid_row + 1, self.min_col, self.max_row, mid_col)
        # Bottom-right
        br = QuadTreeBounds(mid_row + 1, mid_col + 1, self.max_row, self.max_col)

        return tl, tr, bl, br


@dataclass
class QuadTreeNode:
    """Node in a quadtree structure."""

    bounds: QuadTreeBounds
    node_type: NodeType = NodeType.EMPTY
    depth: int = 0
    filled_cells: int = 0
    total_cells: int = 0
    density: float = 0.0

    # Child nodes (only for MIXED type)
    top_left: Optional["QuadTreeNode"] = None
    top_right: Optional["QuadTreeNode"] = None
    bottom_left: Optional["QuadTreeNode"] = None
    bottom_right: Optional["QuadTreeNode"] = None

    # Pattern information
    patterns: list[TablePattern] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type != NodeType.MIXED

    @property
    def children(self) -> list["QuadTreeNode"]:
        """Get all non-None children."""
        children = []
        if self.top_left:
            children.append(self.top_left)
        if self.top_right:
            children.append(self.top_right)
        if self.bottom_left:
            children.append(self.bottom_left)
        if self.bottom_right:
            children.append(self.bottom_right)
        return children

    def get_leaf_nodes(self) -> list["QuadTreeNode"]:
        """Get all leaf nodes under this node."""
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves

    def get_nodes_at_depth(self, target_depth: int) -> list["QuadTreeNode"]:
        """Get all nodes at a specific depth."""
        if self.depth == target_depth:
            return [self]

        if self.depth > target_depth or self.is_leaf:
            return []

        nodes = []
        for child in self.children:
            nodes.extend(child.get_nodes_at_depth(target_depth))
        return nodes


@dataclass
class VisualizationRegion:
    """Region selected for visualization."""

    bounds: QuadTreeBounds
    priority: float
    patterns: list[TablePattern]
    estimated_size_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QuadtreeAnalyzer:
    """Build structure-aware quadtree for efficient visualization."""

    def __init__(
        self,
        max_depth: int = 8,
        min_node_size: int = 100,
        density_threshold: float = 0.1,
        pattern_aware: bool = True,
    ):
        """Initialize quadtree analyzer.

        Args:
            max_depth: Maximum tree depth
            min_node_size: Minimum cells in a node before stopping split
            density_threshold: Minimum density to consider region non-empty
            pattern_aware: Whether to respect pattern boundaries when splitting
        """
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.density_threshold = density_threshold
        self.pattern_aware = pattern_aware

    def analyze(
        self, sheet_data: SheetData, patterns: list[TablePattern] | None = None
    ) -> QuadTreeNode:
        """Create quadtree that respects table boundaries.

        Args:
            sheet_data: Sheet data to analyze
            patterns: Detected table patterns to preserve

        Returns:
            Root node of the quadtree
        """
        # Create root node with sheet bounds
        root_bounds = QuadTreeBounds(
            min_row=0,
            min_col=0,
            max_row=sheet_data.max_row,
            max_col=sheet_data.max_column,
        )

        root = QuadTreeNode(bounds=root_bounds, depth=0)

        # Build tree
        self._build_tree(root, sheet_data, patterns or [])

        logger.info(f"Built quadtree with max depth {self._get_max_depth(root)}")
        return root

    def plan_visualization(
        self,
        quadtree: QuadTreeNode,
        max_regions: int = 10,
        max_total_size_mb: float = 20.0,
    ) -> list[VisualizationRegion]:
        """Plan optimal regions for bitmap generation.

        Args:
            quadtree: Root of the quadtree
            max_regions: Maximum number of regions to generate
            max_total_size_mb: Maximum total size in MB

        Returns:
            List of regions to visualize
        """
        # Get candidate nodes
        candidates = self._get_visualization_candidates(quadtree)

        # Sort by priority (density * area * pattern_importance)
        candidates.sort(key=lambda n: self._calculate_priority(n), reverse=True)

        # Select regions within constraints
        selected_regions = []
        total_size = 0.0

        for node in candidates:
            if len(selected_regions) >= max_regions:
                break

            region_size = self._estimate_region_size_mb(node)
            if total_size + region_size > max_total_size_mb:
                continue

            # Check for overlap with already selected regions
            if not self._overlaps_selected(node, selected_regions):
                region = VisualizationRegion(
                    bounds=node.bounds,
                    priority=self._calculate_priority(node),
                    patterns=node.patterns,
                    estimated_size_mb=region_size,
                    metadata={
                        "density": node.density,
                        "filled_cells": node.filled_cells,
                        "depth": node.depth,
                    },
                )
                selected_regions.append(region)
                total_size += region_size

        logger.info(
            f"Selected {len(selected_regions)} regions for visualization "
            f"(total size: {total_size:.1f}MB)"
        )

        return selected_regions

    def _build_tree(
        self,
        node: QuadTreeNode,
        sheet_data: SheetData,
        patterns: list[TablePattern],
        depth: int = 0,
    ):
        """Recursively build the quadtree."""
        node.depth = depth

        # Calculate node statistics
        self._calculate_node_stats(node, sheet_data)

        # Find patterns that intersect this node
        node.patterns = self._find_intersecting_patterns(node.bounds, patterns)

        # Determine node type
        if node.filled_cells == 0:
            node.node_type = NodeType.EMPTY
            return
        elif node.density < self.density_threshold:
            node.node_type = NodeType.SPARSE
            return
        elif node.density > 0.8:
            node.node_type = NodeType.DENSE
            return

        # Check if we should split
        if not self._should_split(node, patterns):
            # Classify as leaf
            if node.density < 0.3:
                node.node_type = NodeType.SPARSE
            else:
                node.node_type = NodeType.DENSE
            return

        # Split into quadrants
        node.node_type = NodeType.MIXED
        tl, tr, bl, br = node.bounds.split()

        # Create child nodes
        node.top_left = QuadTreeNode(bounds=tl)
        node.top_right = QuadTreeNode(bounds=tr)
        node.bottom_left = QuadTreeNode(bounds=bl)
        node.bottom_right = QuadTreeNode(bounds=br)

        # Recursively build children
        for child in node.children:
            self._build_tree(child, sheet_data, patterns, depth + 1)

    def _calculate_node_stats(self, node: QuadTreeNode, sheet_data: SheetData):
        """Calculate statistics for a node."""
        filled_count = 0
        total_count = 0

        for row in range(node.bounds.min_row, node.bounds.max_row + 1):
            for col in range(node.bounds.min_col, node.bounds.max_col + 1):
                cell = sheet_data.get_cell(row, col)
                if cell and not cell.is_empty:
                    filled_count += 1
                total_count += 1

        node.filled_cells = filled_count
        node.total_cells = total_count
        node.density = filled_count / total_count if total_count > 0 else 0.0

    def _should_split(self, node: QuadTreeNode, _patterns: list[TablePattern]) -> bool:
        """Determine if a node should be split."""
        # Don't split if too deep
        if node.depth >= self.max_depth:
            return False

        # Don't split if too small
        if node.bounds.area < self.min_node_size:
            return False

        # Don't split if it would break a pattern (when pattern-aware)
        if self.pattern_aware and node.patterns:
            # Check if splitting would cut through any pattern
            for pattern in node.patterns:
                if self._would_split_pattern(node.bounds, pattern.bounds):
                    return False

        # Split if density is mixed (not too sparse or too dense)
        return 0.2 < node.density < 0.8

    def _would_split_pattern(
        self, node_bounds: QuadTreeBounds, pattern_bounds: TableBounds
    ) -> bool:
        """Check if splitting a node would cut through a pattern."""
        # Convert pattern bounds to quadtree bounds
        p_bounds = QuadTreeBounds(
            min_row=pattern_bounds.start_row,
            min_col=pattern_bounds.start_col,
            max_row=pattern_bounds.end_row,
            max_col=pattern_bounds.end_col,
        )

        # If pattern is entirely within node, check if split would cut it
        if (
            node_bounds.min_row <= p_bounds.min_row
            and node_bounds.max_row >= p_bounds.max_row
            and node_bounds.min_col <= p_bounds.min_col
            and node_bounds.max_col >= p_bounds.max_col
        ):
            # Check if split lines would cut through pattern
            mid_row = (node_bounds.min_row + node_bounds.max_row) // 2
            mid_col = (node_bounds.min_col + node_bounds.max_col) // 2

            # Would horizontal split cut pattern?
            if p_bounds.min_row < mid_row < p_bounds.max_row:
                return True

            # Would vertical split cut pattern?
            if p_bounds.min_col < mid_col < p_bounds.max_col:
                return True

        return False

    def _find_intersecting_patterns(
        self, bounds: QuadTreeBounds, patterns: list[TablePattern]
    ) -> list[TablePattern]:
        """Find patterns that intersect with given bounds."""
        intersecting = []

        for pattern in patterns:
            p_bounds = QuadTreeBounds(
                min_row=pattern.bounds.start_row,
                min_col=pattern.bounds.start_col,
                max_row=pattern.bounds.end_row,
                max_col=pattern.bounds.end_col,
            )

            if bounds.intersects(p_bounds):
                intersecting.append(pattern)

        return intersecting

    def _get_visualization_candidates(self, node: QuadTreeNode) -> list[QuadTreeNode]:
        """Get nodes that are good candidates for visualization."""
        candidates = []

        # For leaf nodes, add if they have data
        if node.is_leaf:
            if node.filled_cells > 0:
                candidates.append(node)
        else:
            # For internal nodes, recurse to children
            for child in node.children:
                candidates.extend(self._get_visualization_candidates(child))

        return candidates

    def _calculate_priority(self, node: QuadTreeNode) -> float:
        """Calculate visualization priority for a node."""
        # Base priority on density and size
        base_priority = node.density * np.log1p(node.filled_cells)

        # Boost priority if node contains patterns
        pattern_boost = 1.0
        if node.patterns:
            # Higher boost for more confident patterns
            max_confidence = max(p.confidence for p in node.patterns)
            pattern_boost = 1.0 + max_confidence

        # Penalize very deep nodes (they're small)
        depth_penalty = 1.0 / (1.0 + node.depth * 0.1)

        return base_priority * pattern_boost * depth_penalty

    def _estimate_region_size_mb(self, node: QuadTreeNode) -> float:
        """Estimate the size of a region's bitmap in MB."""
        # Assume 2-bit representation and PNG compression
        bits_per_cell = 2
        compression_ratio = 0.5  # Conservative estimate

        # Calculate uncompressed size
        pixels = node.bounds.area
        bytes_uncompressed = (pixels * bits_per_cell) / 8

        # Apply compression estimate
        bytes_compressed = bytes_uncompressed * compression_ratio

        # Add overhead for PNG headers, metadata
        overhead = 1024  # 1KB overhead

        total_bytes = bytes_compressed + overhead
        return total_bytes / (1024 * 1024)  # Convert to MB

    def _overlaps_selected(self, node: QuadTreeNode, selected: list[VisualizationRegion]) -> bool:
        """Check if a node overlaps with already selected regions."""
        return any(node.bounds.intersects(region.bounds) for region in selected)

    def _get_max_depth(self, node: QuadTreeNode) -> int:
        """Get the maximum depth of the tree."""
        if node.is_leaf:
            return node.depth

        max_child_depth = 0
        for child in node.children:
            max_child_depth = max(max_child_depth, self._get_max_depth(child))

        return max_child_depth

    def get_coverage_stats(self, quadtree: QuadTreeNode) -> dict[str, Any]:
        """Get statistics about quadtree coverage."""
        leaf_nodes = quadtree.get_leaf_nodes()

        total_cells = sum(node.total_cells for node in leaf_nodes)
        filled_cells = sum(node.filled_cells for node in leaf_nodes)

        empty_nodes = sum(1 for node in leaf_nodes if node.node_type == NodeType.EMPTY)
        sparse_nodes = sum(1 for node in leaf_nodes if node.node_type == NodeType.SPARSE)
        dense_nodes = sum(1 for node in leaf_nodes if node.node_type == NodeType.DENSE)

        return {
            "total_nodes": len(leaf_nodes),
            "empty_nodes": empty_nodes,
            "sparse_nodes": sparse_nodes,
            "dense_nodes": dense_nodes,
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "overall_density": filled_cells / total_cells if total_cells > 0 else 0.0,
            "max_depth": self._get_max_depth(quadtree),
        }
