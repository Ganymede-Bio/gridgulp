"""Detection Agent - Responsible for detecting table boundaries using various strategies."""

from typing import Any, Optional

from gridporter.agents.base_agent import BaseAgent
from gridporter.config import Config
from gridporter.models.file_data import FileData
from gridporter.models.table import TableInfo, TableRange
from gridporter.tools.detection import (
    detect_named_ranges,
    extract_list_objects,
    find_connected_components,
    preprocess_data_regions,
)


class DetectionAgent(BaseAgent):
    """
    Agent responsible for detecting tables in spreadsheets.

    This agent chooses the appropriate detection strategy based on file
    characteristics and manages progressive refinement for large sheets.
    """

    def __init__(self, config: Config):
        """Initialize the detection agent."""
        super().__init__(config)
        self.strategies = ["named_range", "list_object", "vision", "heuristic"]

    async def execute(
        self, file_data: FileData, options: Optional[dict[str, Any]] = None
    ) -> list[TableInfo]:
        """
        Detect all tables in the file using appropriate strategies.

        Args:
            file_data: Loaded file data
            options: Detection options

        Returns:
            List of detected table proposals
        """
        options = options or {}
        all_proposals = []

        # Process each sheet
        for sheet_name in file_data.sheet_names:
            sheet_data = file_data.get_sheet(sheet_name)

            # Try fast paths first
            proposals = await self._detect_in_sheet(sheet_data, options)

            # Add sheet info to proposals
            for proposal in proposals:
                proposal.sheet_name = sheet_name

            all_proposals.extend(proposals)

        self.logger.info(f"Detected {len(all_proposals)} tables across all sheets")
        return all_proposals

    async def _detect_in_sheet(self, sheet_data: Any, options: dict[str, Any]) -> list[TableInfo]:
        """Detect tables in a single sheet."""

        # Fast path 1: Excel ListObjects (formal tables)
        if hasattr(sheet_data, "list_objects") and sheet_data.list_objects:
            self.logger.info("Checking Excel ListObjects")
            tables = self._detect_list_objects(sheet_data)
            if tables:
                self.update_state("used_list_objects", True)
                return tables

        # Fast path 2: Named ranges
        if options.get("check_named_ranges", True):
            self.logger.info("Checking named ranges")
            tables = self._detect_named_ranges(sheet_data)
            if tables:
                self.update_state("used_named_ranges", True)
                return tables

        # Slow path: Content-based detection
        self.logger.info("Using content-based detection")

        # Pre-process to find data regions
        regions = preprocess_data_regions(
            sheet_data, min_cells=self.config.min_table_size[0] * self.config.min_table_size[1]
        )

        # Skip if sheet is mostly empty
        if regions.sheet_utilization < 0.001:
            self.logger.info("Sheet is mostly empty, skipping")
            return []

        # Decide detection strategy based on sheet characteristics
        if self._should_use_vision(regions, options):
            # Vision detection will be handled by VisionAgent
            self.update_state("detection_strategy", "vision")
            return self._create_region_proposals(regions)
        else:
            # Use heuristic detection
            self.update_state("used_heuristics", True)
            return self._heuristic_detection(sheet_data, regions)

    def _detect_list_objects(self, sheet_data: Any) -> list[TableInfo]:
        """Detect Excel ListObjects (formal tables)."""
        try:
            list_objects = extract_list_objects(sheet_data.worksheet)

            tables = []
            for obj in list_objects:
                table = TableInfo(
                    id=f"table_{obj.name}",
                    range=TableRange.from_excel(obj.range),
                    suggested_name=obj.name,
                    confidence=1.0,  # 100% confidence for defined tables
                    detection_method="list_object",
                    has_headers=obj.has_headers,
                    metadata={"table_style": obj.style, "source": "excel_list_object"},
                )
                tables.append(table)

            return tables

        except Exception as e:
            self.logger.warning(f"Failed to extract ListObjects: {e}")
            return []

    def _detect_named_ranges(self, sheet_data: Any) -> list[TableInfo]:
        """Detect tables from named ranges."""
        try:
            if not hasattr(sheet_data, "workbook"):
                return []

            named_ranges = detect_named_ranges(sheet_data.workbook)

            # Filter to likely table ranges
            table_ranges = [
                nr for nr in named_ranges if self._is_likely_table_range(nr, sheet_data)
            ]

            tables = []
            for nr in table_ranges:
                table = TableInfo(
                    id=f"table_{nr.name}",
                    range=TableRange.from_excel(nr.range),
                    suggested_name=nr.name,
                    confidence=0.9,  # High confidence for named ranges
                    detection_method="named_range",
                    metadata={"source": "named_range", "scope": nr.scope},
                )
                tables.append(table)

            return tables

        except Exception as e:
            self.logger.warning(f"Failed to detect named ranges: {e}")
            return []

    def _is_likely_table_range(self, named_range: Any, sheet_data: Any) -> bool:
        """Check if a named range likely represents a table."""
        # Skip single cells
        if ":" not in named_range.range:
            return False

        # Skip ranges with "print", "criteria", etc. in name
        skip_words = ["print", "criteria", "filter", "_", "chart"]
        if any(word in named_range.name.lower() for word in skip_words):
            return False

        # Check minimum size
        try:
            range_obj = TableRange.from_excel(named_range.range)
            rows = range_obj.end_row - range_obj.start_row + 1
            cols = range_obj.end_col - range_obj.start_col + 1

            if rows < self.config.min_table_size[0] or cols < self.config.min_table_size[1]:
                return False

        except Exception:
            return False

        return True

    def _should_use_vision(self, regions: Any, options: dict[str, Any]) -> bool:
        """Decide whether to use vision detection."""
        # Vision disabled
        if not options.get("enable_vision", True):
            return False

        # Use vision for complex layouts
        if regions.disconnected_regions > 5:
            return True

        # Use vision for very sparse sheets
        if regions.data_density < 0.05 and regions.total_cells > 1000:
            return True

        # Use vision for sheets with mixed patterns
        if regions.has_mixed_patterns:
            return True

        return False

    def _create_region_proposals(self, regions: Any) -> list[TableInfo]:
        """Create initial proposals from detected regions for vision analysis."""
        proposals = []

        for i, region in enumerate(regions.data_regions):
            proposal = TableInfo(
                id=f"region_{i}",
                range=TableRange(
                    start_row=region.bounds["top"],
                    start_col=region.bounds["left"],
                    end_row=region.bounds["bottom"],
                    end_col=region.bounds["right"],
                ),
                confidence=0.5,  # Low confidence, needs vision verification
                detection_method="region_detection",
                metadata={
                    "cell_count": region.cell_count,
                    "density": region.density,
                    "characteristics": region.characteristics,
                },
            )
            proposals.append(proposal)

        return proposals

    def _heuristic_detection(self, sheet_data: Any, regions: Any) -> list[TableInfo]:
        """Detect tables using heuristic rules."""
        tables = []

        # Use connected components with heuristics
        components = find_connected_components(
            sheet_data,
            gap_threshold=3,  # Allow 3 empty rows/cols
        )

        for i, component in enumerate(components):
            # Apply heuristics to determine if component is a table
            if self._is_valid_table(component, sheet_data):
                table = TableInfo(
                    id=f"heuristic_{i}",
                    range=TableRange(
                        start_row=component.bounds["top"],
                        start_col=component.bounds["left"],
                        end_row=component.bounds["bottom"],
                        end_col=component.bounds["right"],
                    ),
                    confidence=self._calculate_confidence(component),
                    detection_method="heuristic",
                    has_headers=component.characteristics.get("likely_headers", False),
                    metadata={"detection_details": component.characteristics},
                )
                tables.append(table)

        return tables

    def _is_valid_table(self, component: Any, sheet_data: Any) -> bool:
        """Check if a component is a valid table."""
        # Minimum size check
        rows = component.bounds["bottom"] - component.bounds["top"] + 1
        cols = component.bounds["right"] - component.bounds["left"] + 1

        if rows < self.config.min_table_size[0] or cols < self.config.min_table_size[1]:
            return False

        # Density check
        if component.density < 0.3:  # Less than 30% filled
            return False

        # Rectangularness check
        if component.characteristics.get("rectangularness", 0) < 0.7:
            return False

        return True

    def _calculate_confidence(self, component: Any) -> float:
        """Calculate confidence score for heuristic detection."""
        confidence = 0.5  # Base confidence

        # Boost for high density
        if component.density > 0.8:
            confidence += 0.2

        # Boost for likely headers
        if component.characteristics.get("likely_headers", False):
            confidence += 0.15

        # Boost for high rectangularness
        rectangularness = component.characteristics.get("rectangularness", 0)
        confidence += rectangularness * 0.15

        return min(confidence, 0.95)  # Cap at 95% for heuristic detection
