"""Parse vision model responses to extract table region proposals."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ..vision.bitmap_generator import BitmapMetadata

logger = logging.getLogger(__name__)


@dataclass
class RegionProposal:
    """A proposed table region from vision analysis."""

    # Pixel coordinates
    x1: int
    y1: int
    x2: int
    y2: int

    # Cell coordinates
    start_row: int
    start_col: int
    end_row: int
    end_col: int

    # Excel-style range
    excel_range: str

    # Confidence and characteristics
    confidence: float
    characteristics: dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "pixel_bounds": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
            "cell_bounds": {
                "start_row": self.start_row,
                "start_col": self.start_col,
                "end_row": self.end_row,
                "end_col": self.end_col,
            },
            "excel_range": self.excel_range,
            "confidence": self.confidence,
            "characteristics": self.characteristics,
        }


class RegionProposer:
    """Extract and parse region proposals from vision model responses."""

    def __init__(self):
        """Initialize region proposer."""
        # Regex patterns for extracting information
        self.pixel_pattern = re.compile(
            r"(?:pixel\s*)?(?:coordinates?|bounds?|box)?\s*:?\s*"
            r"\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?",
            re.IGNORECASE,
        )
        self.range_pattern = re.compile(r"([A-Z]+\d+):([A-Z]+\d+)", re.IGNORECASE)
        self.confidence_pattern = re.compile(r"confidence\s*:?\s*([\d.]+)", re.IGNORECASE)

    def parse_response(
        self, response: str, bitmap_metadata: BitmapMetadata
    ) -> list[RegionProposal]:
        """Parse vision model response to extract region proposals.

        Args:
            response: Text response from vision model
            bitmap_metadata: Metadata about the bitmap

        Returns:
            List of region proposals
        """
        proposals = []

        # First try to parse as JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "tables" in data:
                # Structured response with tables array
                for table in data.get("tables", []):
                    proposal = self._parse_structured_table(table, bitmap_metadata)
                    if proposal:
                        proposals.append(proposal)
            elif isinstance(data, list):
                # Direct array of tables
                for table in data:
                    proposal = self._parse_structured_table(table, bitmap_metadata)
                    if proposal:
                        proposals.append(proposal)
            return proposals
        except json.JSONDecodeError:
            # Not JSON, parse as text
            pass

        # Parse unstructured text response
        # Split by common delimiters that might separate table descriptions
        sections = re.split(
            r"\n\n|\n(?=Table\s*\d+:?)|(?=Region\s*\d+:?)", response, flags=re.IGNORECASE
        )

        for section in sections:
            if not section.strip():
                continue

            proposal = self._parse_text_section(section, bitmap_metadata)
            if proposal:
                proposals.append(proposal)

        logger.info(f"Parsed {len(proposals)} region proposals from vision model response")
        return proposals

    def _parse_structured_table(
        self, table_data: dict, bitmap_metadata: BitmapMetadata
    ) -> RegionProposal | None:
        """Parse a structured table object.

        Args:
            table_data: Dictionary with table information
            bitmap_metadata: Bitmap metadata

        Returns:
            Region proposal or None if parsing fails
        """
        try:
            # Extract pixel bounds
            if "bounds" in table_data:
                bounds = table_data["bounds"]
                if isinstance(bounds, dict):
                    x1 = bounds.get("x1", bounds.get("x", 0))
                    y1 = bounds.get("y1", bounds.get("y", 0))
                    x2 = bounds.get("x2", bounds.get("x", 0) + bounds.get("width", 0))
                    y2 = bounds.get("y2", bounds.get("y", 0) + bounds.get("height", 0))
                elif isinstance(bounds, list | tuple) and len(bounds) >= 4:
                    x1, y1, x2, y2 = bounds[:4]
                else:
                    return None
            elif all(k in table_data for k in ["x1", "y1", "x2", "y2"]):
                x1 = table_data["x1"]
                y1 = table_data["y1"]
                x2 = table_data["x2"]
                y2 = table_data["y2"]
            else:
                return None

            # Convert to cell coordinates
            start_row = y1 // bitmap_metadata.cell_height
            start_col = x1 // bitmap_metadata.cell_width
            end_row = (y2 - 1) // bitmap_metadata.cell_height
            end_col = (x2 - 1) // bitmap_metadata.cell_width

            # Get or generate Excel range
            excel_range = table_data.get("range", table_data.get("excel_range"))
            if not excel_range:
                excel_range = f"{self._col_to_letter(start_col)}{start_row + 1}:{self._col_to_letter(end_col)}{end_row + 1}"

            # Extract confidence
            confidence = float(table_data.get("confidence", 0.5))

            # Extract characteristics
            characteristics = {}
            for key in ["has_headers", "header_rows", "merged_cells", "description", "type"]:
                if key in table_data:
                    characteristics[key] = table_data[key]

            return RegionProposal(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                start_row=start_row,
                start_col=start_col,
                end_row=end_row,
                end_col=end_col,
                excel_range=excel_range,
                confidence=confidence,
                characteristics=characteristics,
            )

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse structured table: {e}")
            return None

    def _parse_text_section(
        self, text: str, bitmap_metadata: BitmapMetadata
    ) -> RegionProposal | None:
        """Parse a text section describing a table.

        Args:
            text: Text description of a table
            bitmap_metadata: Bitmap metadata

        Returns:
            Region proposal or None if parsing fails
        """
        # Look for pixel coordinates
        pixel_match = self.pixel_pattern.search(text)
        if not pixel_match:
            return None

        x1, y1, x2, y2 = map(int, pixel_match.groups())

        # Convert to cell coordinates
        start_row = y1 // bitmap_metadata.cell_height
        start_col = x1 // bitmap_metadata.cell_width
        end_row = (y2 - 1) // bitmap_metadata.cell_height
        end_col = (x2 - 1) // bitmap_metadata.cell_width

        # Look for Excel range
        range_match = self.range_pattern.search(text)
        if range_match:
            excel_range = range_match.group(0)
        else:
            # Generate from cell coordinates
            excel_range = f"{self._col_to_letter(start_col)}{start_row + 1}:{self._col_to_letter(end_col)}{end_row + 1}"

        # Look for confidence
        confidence_match = self.confidence_pattern.search(text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        # Normalize confidence to 0-1 range if needed
        if confidence > 1.0:
            confidence = confidence / 100.0

        # Extract characteristics from text
        characteristics = {}
        if "header" in text.lower():
            characteristics["has_headers"] = True
            # Try to extract number of header rows
            header_rows_match = re.search(r"(\d+)\s*header\s*rows?", text, re.IGNORECASE)
            if header_rows_match:
                characteristics["header_rows"] = int(header_rows_match.group(1))

        if "merged" in text.lower():
            characteristics["has_merged_cells"] = True

        if "formula" in text.lower():
            characteristics["has_formulas"] = True

        # Extract any description
        desc_match = re.search(
            r"(?:description|note|characteristic)[:\s]+([^.]+)", text, re.IGNORECASE
        )
        if desc_match:
            characteristics["description"] = desc_match.group(1).strip()

        return RegionProposal(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
            excel_range=excel_range,
            confidence=confidence,
            characteristics=characteristics,
        )

    def _col_to_letter(self, col: int) -> str:
        """Convert column index to Excel letter.

        Args:
            col: 0-based column index

        Returns:
            Excel column letter (A, B, ..., AA, AB, ...)
        """
        result = ""
        while col >= 0:
            result = chr(col % 26 + ord("A")) + result
            col = col // 26 - 1
        return result

    def filter_proposals(
        self,
        proposals: list[RegionProposal],
        min_confidence: float = 0.5,
        min_size: tuple[int, int] = (2, 2),
    ) -> list[RegionProposal]:
        """Filter proposals based on confidence and size.

        Args:
            proposals: List of proposals to filter
            min_confidence: Minimum confidence threshold
            min_size: Minimum (rows, cols) size

        Returns:
            Filtered list of proposals
        """
        filtered = []
        min_rows, min_cols = min_size

        for proposal in proposals:
            # Check confidence
            if proposal.confidence < min_confidence:
                logger.debug(f"Filtering out proposal with low confidence: {proposal.confidence}")
                continue

            # Check size
            rows = proposal.end_row - proposal.start_row + 1
            cols = proposal.end_col - proposal.start_col + 1

            if rows < min_rows or cols < min_cols:
                logger.debug(f"Filtering out small proposal: {rows}x{cols}")
                continue

            filtered.append(proposal)

        logger.info(f"Filtered {len(proposals)} proposals to {len(filtered)}")
        return filtered
