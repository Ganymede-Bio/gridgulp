"""Region verification for AI-proposed table regions."""

import logging
from dataclasses import dataclass

import numpy as np

from ..models.sheet_data import SheetData
from ..models.table import TableRange
from ..vision.pattern_detector import TablePattern

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of region verification."""

    valid: bool
    confidence: float
    reason: str
    metrics: dict[str, float]
    feedback: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "reason": self.reason,
            "metrics": self.metrics,
            "feedback": self.feedback,
        }


@dataclass
class GeometryMetrics:
    """Geometric analysis metrics for a region."""

    rectangularness: float  # How rectangular is the data distribution
    filledness: float  # Ratio of filled to total cells
    density: float  # Data density considering sparsity patterns
    contiguity: float  # How contiguous the data is
    edge_quality: float  # Quality of region boundaries
    aspect_ratio: float  # Width/height ratio
    size_ratio: float  # Region size relative to sheet

    def overall_score(self) -> float:
        """Compute overall geometry score."""
        return (
            self.rectangularness * 0.3
            + self.filledness * 0.2
            + self.density * 0.2
            + self.contiguity * 0.15
            + self.edge_quality * 0.15
        )


class RegionVerifier:
    """Verify AI-proposed table regions using local analysis."""

    def __init__(
        self,
        min_filled_cells: int = 3,
        min_filledness: float = 0.1,
        min_rectangularness: float = 0.7,
        min_contiguity: float = 0.5,
    ):
        """Initialize region verifier.

        Args:
            min_filled_cells: Minimum number of filled cells in a valid region
            min_filledness: Minimum filledness ratio (0-1)
            min_rectangularness: Minimum rectangularness score (0-1)
            min_contiguity: Minimum data contiguity score (0-1)
        """
        self.min_filled_cells = min_filled_cells
        self.min_filledness = min_filledness
        self.min_rectangularness = min_rectangularness
        self.min_contiguity = min_contiguity

    def verify_region(
        self, sheet: SheetData, region: TableRange | TablePattern, strict: bool = False
    ) -> VerificationResult:
        """Verify a proposed table region.

        Args:
            sheet: Sheet data to verify against
            region: Proposed region to verify
            strict: Whether to apply strict verification rules

        Returns:
            VerificationResult with validity, confidence, and metrics
        """
        try:
            # Extract bounds
            if isinstance(region, TablePattern):
                start_row = region.bounds.start_row
                start_col = region.bounds.start_col
                end_row = region.bounds.end_row
                end_col = region.bounds.end_col
            else:
                start_row = region.start_row
                start_col = region.start_col
                end_row = region.end_row
                end_col = region.end_col

            # Validate bounds
            if not self._validate_bounds(sheet, start_row, start_col, end_row, end_col):
                return VerificationResult(
                    valid=False,
                    confidence=0.0,
                    reason="Invalid bounds",
                    metrics={},
                    feedback="Region bounds exceed sheet dimensions",
                )

            # Extract region data
            region_data = self._extract_region_data(sheet, start_row, start_col, end_row, end_col)

            # Check if region has enough data
            filled_count = np.sum(region_data)
            if filled_count < self.min_filled_cells:
                return VerificationResult(
                    valid=False,
                    confidence=0.1,
                    reason=f"Too few filled cells ({filled_count} < {self.min_filled_cells})",
                    metrics={"filled_cells": float(filled_count)},
                    feedback="Region contains too little data to be a meaningful table",
                )

            # Compute geometry metrics
            metrics = self._compute_geometry_metrics(region_data)

            # Apply verification rules
            verification_result = self._apply_verification_rules(metrics, strict)

            # Generate feedback if invalid
            if not verification_result.valid:
                verification_result.feedback = self._generate_feedback(metrics, region_data)

            return verification_result

        except Exception as e:
            logger.error(f"Error verifying region: {e}")
            return VerificationResult(
                valid=False,
                confidence=0.0,
                reason=f"Verification error: {str(e)}",
                metrics={},
                feedback="Unable to verify region due to an error",
            )

    def verify_pattern(self, sheet: SheetData, pattern: TablePattern) -> VerificationResult:
        """Verify a detected table pattern.

        Args:
            sheet: Sheet data to verify against
            pattern: Pattern to verify

        Returns:
            VerificationResult with pattern-specific validation
        """
        # First do basic region verification
        result = self.verify_region(sheet, pattern)

        # If basic verification passes, do pattern-specific checks
        if result.valid:
            pattern_result = self._verify_pattern_consistency(sheet, pattern)
            if not pattern_result.valid:
                return pattern_result

        return result

    def _validate_bounds(
        self, sheet: SheetData, start_row: int, start_col: int, end_row: int, end_col: int
    ) -> bool:
        """Validate region bounds are within sheet dimensions."""
        return (
            0 <= start_row <= sheet.max_row
            and 0 <= start_col <= sheet.max_column
            and 0 <= end_row <= sheet.max_row
            and 0 <= end_col <= sheet.max_column
            and start_row <= end_row
            and start_col <= end_col
        )

    def _extract_region_data(
        self, sheet: SheetData, start_row: int, start_col: int, end_row: int, end_col: int
    ) -> np.ndarray:
        """Extract binary data array for region (1=filled, 0=empty)."""
        height = end_row - start_row + 1
        width = end_col - start_col + 1
        data = np.zeros((height, width), dtype=bool)

        for row_idx in range(height):
            row = start_row + row_idx
            for col_idx in range(width):
                col = start_col + col_idx
                cell = sheet.get_cell(row, col)
                if cell is not None and cell.value is not None:
                    data[row_idx, col_idx] = True

        return data

    def _compute_geometry_metrics(self, region_data: np.ndarray) -> GeometryMetrics:
        """Compute geometric analysis metrics for region."""
        height, width = region_data.shape
        filled_cells = np.sum(region_data)
        total_cells = height * width

        # Filledness: ratio of filled cells
        filledness = filled_cells / total_cells if total_cells > 0 else 0.0

        # Rectangularness: how well data fits a rectangle
        rectangularness = self._compute_rectangularness(region_data)

        # Density: considering sparse patterns
        density = self._compute_density(region_data)

        # Contiguity: how connected the data is
        contiguity = self._compute_contiguity(region_data)

        # Edge quality: how clean the boundaries are
        edge_quality = self._compute_edge_quality(region_data)

        # Aspect ratio
        aspect_ratio = width / height if height > 0 else 1.0

        # Size ratio (assuming max reasonable table is 1000x100)
        size_ratio = min(1.0, (height * width) / (1000 * 100))

        return GeometryMetrics(
            rectangularness=rectangularness,
            filledness=filledness,
            density=density,
            contiguity=contiguity,
            edge_quality=edge_quality,
            aspect_ratio=aspect_ratio,
            size_ratio=size_ratio,
        )

    def _compute_rectangularness(self, data: np.ndarray) -> float:
        """Compute how rectangular the data distribution is."""
        if not data.any():
            return 0.0

        # Find bounding box of actual data
        rows_with_data = np.any(data, axis=1)
        cols_with_data = np.any(data, axis=0)

        if not rows_with_data.any() or not cols_with_data.any():
            return 0.0

        # Get bounds of actual data
        first_row = np.argmax(rows_with_data)
        last_row = len(rows_with_data) - np.argmax(rows_with_data[::-1]) - 1
        first_col = np.argmax(cols_with_data)
        last_col = len(cols_with_data) - np.argmax(cols_with_data[::-1]) - 1

        # Extract minimal bounding box
        bbox_data = data[first_row : last_row + 1, first_col : last_col + 1]

        # Compute coverage within bounding box
        bbox_filled = np.sum(bbox_data)
        bbox_total = bbox_data.size

        # Rectangularness is coverage within minimal bounding box
        return bbox_filled / bbox_total if bbox_total > 0 else 0.0

    def _compute_density(self, data: np.ndarray) -> float:
        """Compute data density considering sparse patterns."""
        if not data.any():
            return 0.0

        # Row and column densities
        row_density = np.mean([np.sum(row) / len(row) for row in data if row.any()])
        col_density = np.mean(
            [np.sum(data[:, i]) / len(data) for i in range(data.shape[1]) if data[:, i].any()]
        )

        # Combined density (weighted average)
        return 0.5 * row_density + 0.5 * col_density

    def _compute_contiguity(self, data: np.ndarray) -> float:
        """Compute how contiguous/connected the data is."""
        if not data.any():
            return 0.0

        # Count connected components using simple flood fill
        visited = np.zeros_like(data, dtype=bool)
        components = 0

        def flood_fill(i: int, j: int) -> int:
            """Flood fill to count connected cells."""
            if i < 0 or i >= data.shape[0] or j < 0 or j >= data.shape[1]:
                return 0
            if visited[i, j] or not data[i, j]:
                return 0

            visited[i, j] = True
            count = 1

            # Check 4-connected neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                count += flood_fill(i + di, j + dj)

            return count

        # Find all connected components
        component_sizes = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] and not visited[i, j]:
                    size = flood_fill(i, j)
                    component_sizes.append(size)
                    components += 1

        if components == 0:
            return 0.0

        # Contiguity is high when there's one large component
        total_filled = np.sum(data)
        largest_component = max(component_sizes) if component_sizes else 0

        # Penalize multiple components
        contiguity = (largest_component / total_filled) * (1.0 / components)

        return min(1.0, contiguity)

    def _compute_edge_quality(self, data: np.ndarray) -> float:
        """Compute quality of region boundaries."""
        if not data.any():
            return 0.0

        # Check how well-defined the edges are
        # Good edges have clear transitions from data to no-data

        # Top and bottom edges
        top_edge_quality = self._edge_quality_score(data[0, :])
        bottom_edge_quality = self._edge_quality_score(data[-1, :])

        # Left and right edges
        left_edge_quality = self._edge_quality_score(data[:, 0])
        right_edge_quality = self._edge_quality_score(data[:, -1])

        # Average edge quality
        return (top_edge_quality + bottom_edge_quality + left_edge_quality + right_edge_quality) / 4

    def _edge_quality_score(self, edge: np.ndarray) -> float:
        """Score quality of a single edge."""
        # An edge is high quality if it's mostly empty (good boundary)
        # or mostly full (data extends to edge)
        filled_ratio = np.sum(edge) / len(edge) if len(edge) > 0 else 0.0

        # Best scores for very empty or very full edges
        if filled_ratio <= 0.2 or filled_ratio >= 0.8:
            return 1.0
        else:
            # Penalize edges that are partially filled
            return 1.0 - 2 * abs(0.5 - filled_ratio)

    def _apply_verification_rules(
        self, metrics: GeometryMetrics, strict: bool
    ) -> VerificationResult:
        """Apply verification rules based on metrics."""
        reasons = []
        confidence = metrics.overall_score()

        # Check filledness
        if metrics.filledness < self.min_filledness:
            reasons.append(f"Low filledness ({metrics.filledness:.2f} < {self.min_filledness})")
            confidence *= 0.5

        # Check rectangularness
        if metrics.rectangularness < self.min_rectangularness:
            reasons.append(
                f"Low rectangularness ({metrics.rectangularness:.2f} < {self.min_rectangularness})"
            )
            confidence *= 0.7

        # Check contiguity
        if metrics.contiguity < self.min_contiguity:
            reasons.append(f"Low contiguity ({metrics.contiguity:.2f} < {self.min_contiguity})")
            confidence *= 0.8

        # In strict mode, apply additional checks
        if strict:
            if metrics.edge_quality < 0.5:
                reasons.append(f"Poor edge quality ({metrics.edge_quality:.2f})")
                confidence *= 0.6

            if metrics.aspect_ratio > 50 or metrics.aspect_ratio < 0.02:
                reasons.append(f"Extreme aspect ratio ({metrics.aspect_ratio:.2f})")
                confidence *= 0.5

        # Determine validity
        valid = len(reasons) == 0 or (not strict and confidence > 0.5)

        return VerificationResult(
            valid=bool(valid),  # Ensure Python bool, not numpy bool
            confidence=float(confidence),  # Ensure Python float
            reason="; ".join(reasons) if reasons else "All checks passed",
            metrics={
                "rectangularness": float(metrics.rectangularness),
                "filledness": float(metrics.filledness),
                "density": float(metrics.density),
                "contiguity": float(metrics.contiguity),
                "edge_quality": float(metrics.edge_quality),
                "aspect_ratio": float(metrics.aspect_ratio),
                "overall_score": float(metrics.overall_score()),
            },
        )

    def _verify_pattern_consistency(
        self, sheet: SheetData, pattern: TablePattern
    ) -> VerificationResult:
        """Verify pattern-specific consistency."""
        # Pattern-specific validation based on type
        pattern_type = getattr(pattern, "pattern_type", "unknown")

        if pattern_type == "header_data":
            # Verify there's actually a header row
            return self._verify_header_pattern(sheet, pattern)
        elif pattern_type == "matrix":
            # Verify matrix structure
            return self._verify_matrix_pattern(sheet, pattern)
        elif pattern_type == "hierarchical":
            # Verify indentation patterns
            return self._verify_hierarchical_pattern(sheet, pattern)

        # Default: pattern is consistent
        return VerificationResult(
            valid=True,
            confidence=0.9,
            reason="Pattern consistency verified",
            metrics={},
        )

    def _verify_header_pattern(self, sheet: SheetData, pattern: TablePattern) -> VerificationResult:
        """Verify header-data pattern has actual headers."""
        # Check first row has more filled cells than average
        first_row_data = []
        for col in range(pattern.bounds.start_col, pattern.bounds.end_col + 1):
            cell = sheet.get_cell(pattern.bounds.start_row, col)
            if cell is not None and cell.value is not None:
                first_row_data.append(cell.value)

        if len(first_row_data) < 2:
            return VerificationResult(
                valid=False,
                confidence=0.3,
                reason="No valid header row found",
                metrics={"header_cells": len(first_row_data)},
                feedback="Expected header row with multiple columns",
            )

        return VerificationResult(
            valid=True,
            confidence=0.9,
            reason="Header pattern verified",
            metrics={"header_cells": len(first_row_data)},
        )

    def _verify_matrix_pattern(self, sheet: SheetData, pattern: TablePattern) -> VerificationResult:
        """Verify matrix pattern has row and column headers."""
        # Check for headers in first row and column
        has_row_headers = False
        has_col_headers = False

        # Check first row
        filled = 0
        for col in range(pattern.bounds.start_col, pattern.bounds.end_col + 1):
            cell = sheet.get_cell(pattern.bounds.start_row, col)
            if cell is not None and cell.value is not None:
                filled += 1
        has_col_headers = filled >= 2

        # Check first column
        filled_in_first_col = 0
        for row_idx in range(pattern.bounds.start_row, pattern.bounds.end_row + 1):
            cell = sheet.get_cell(row_idx, pattern.bounds.start_col)
            if cell is not None and cell.value is not None:
                filled_in_first_col += 1
        has_row_headers = filled_in_first_col >= 2

        if not (has_row_headers and has_col_headers):
            return VerificationResult(
                valid=False,
                confidence=0.4,
                reason="Matrix pattern missing headers",
                metrics={"has_row_headers": has_row_headers, "has_col_headers": has_col_headers},
                feedback="Matrix tables should have both row and column headers",
            )

        return VerificationResult(
            valid=True,
            confidence=0.95,
            reason="Matrix pattern verified",
            metrics={"has_row_headers": has_row_headers, "has_col_headers": has_col_headers},
        )

    def _verify_hierarchical_pattern(
        self, sheet: SheetData, pattern: TablePattern
    ) -> VerificationResult:
        """Verify hierarchical pattern has indentation."""
        # Look for indentation patterns in first column
        indentation_found = False

        for row_idx in range(pattern.bounds.start_row, pattern.bounds.end_row + 1):
            cell = sheet.get_cell(row_idx, pattern.bounds.start_col)
            if cell is not None and cell.value is not None:
                value = cell.value
                if isinstance(value, str) and value.startswith(("  ", "\t")):
                    indentation_found = True
                    break

        if not indentation_found:
            return VerificationResult(
                valid=False,
                confidence=0.5,
                reason="No indentation found in hierarchical pattern",
                metrics={"indentation_found": False},
                feedback="Hierarchical patterns should have indented rows",
            )

        return VerificationResult(
            valid=True,
            confidence=0.9,
            reason="Hierarchical pattern verified",
            metrics={"indentation_found": True},
        )

    def _generate_feedback(self, metrics: GeometryMetrics, region_data: np.ndarray) -> str:
        """Generate helpful feedback for invalid regions."""
        feedback_parts = []

        if metrics.filledness < self.min_filledness:
            feedback_parts.append(f"Region is too sparse ({metrics.filledness:.1%} filled)")

        if metrics.rectangularness < self.min_rectangularness:
            feedback_parts.append("Data distribution is not rectangular enough")

        if metrics.contiguity < self.min_contiguity:
            feedback_parts.append("Data is too fragmented or disconnected")

        if metrics.edge_quality < 0.5:
            feedback_parts.append("Region boundaries are not well-defined")

        # Add suggestions
        height, width = region_data.shape
        if height > width * 10:
            feedback_parts.append("Consider splitting into multiple tables (very tall region)")
        elif width > height * 10:
            feedback_parts.append("Consider splitting into multiple tables (very wide region)")

        return "; ".join(feedback_parts) if feedback_parts else "Region verification failed"
