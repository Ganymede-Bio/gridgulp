"""Extract DataFrames from detected table regions with header detection."""

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gridgulp.models.sheet_data import CellData, SheetData
from gridgulp.models.table import CellRange

logger = logging.getLogger(__name__)


class HeaderDetectionResult(BaseModel):
    """Result of header detection analysis."""

    model_config = ConfigDict(strict=True)

    has_headers: bool = Field(..., description="Whether headers were detected")
    header_rows: int = Field(0, ge=0, description="Number of header rows")
    header_columns: int = Field(0, ge=0, description="Number of header columns (for transposed)")
    orientation: str = Field("vertical", description="Table orientation: vertical or horizontal")
    headers: list[str] = Field(default_factory=list, description="Detected header values")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Header detection confidence")
    title_rows: list[int] = Field(default_factory=list, description="Rows identified as titles")
    data_sample_size: int = Field(0, ge=0, description="Number of rows analyzed for type detection")
    column_types: dict[int, str] = Field(
        default_factory=dict, description="Detected type for each column"
    )
    table_type: str = Field(
        "standard", description="Type of table: standard, plate_map, time_series"
    )
    plate_format: int | None = Field(
        None, description="Plate format if detected (e.g., 96 for 96-well)"
    )


class DataFrameExtractor:
    """Extract pandas DataFrames from sheet data with intelligent header detection."""

    def __init__(self, min_data_rows: int = 2, min_data_density: float = 0.3):
        """Initialize extractor.

        Args:
            min_data_rows: Minimum number of data rows (excluding headers) for valid table
            min_data_density: Minimum ratio of non-empty cells for valid table
        """
        self.min_data_rows = min_data_rows
        self.min_data_density = min_data_density

    def extract_dataframe(
        self, sheet_data: SheetData, cell_range: CellRange, detect_headers: bool = True
    ) -> tuple[pd.DataFrame | None, HeaderDetectionResult | None, float]:
        """Extract a DataFrame from the specified range.

        Args:
            sheet_data: Sheet containing the data
            cell_range: Range to extract
            detect_headers: Whether to detect headers automatically

        Returns:
            Tuple of (dataframe, header_info, quality_score)
            Returns (None, None, 0.0) if extraction fails
        """
        # Get raw cell data
        range_data = sheet_data.get_range_data(
            cell_range.start_row,
            cell_range.start_col,
            cell_range.end_row,
            cell_range.end_col,
        )

        if not range_data:
            return None, None, 0.0

        # Convert to values matrix
        values_matrix = self._extract_values_matrix(range_data)

        if values_matrix is None or len(values_matrix) == 0:
            return None, None, 0.0

        # Check data density
        density = self._calculate_density(values_matrix)
        logger.debug(f"Data density: {density:.2f}, min required: {self.min_data_density}")
        if density < self.min_data_density:
            logger.debug(f"Low data density: {density:.2f} < {self.min_data_density}")
            return None, None, 0.0

        # Detect headers if requested
        header_info = None
        logger.debug(f"Detect headers: {detect_headers}")
        if detect_headers:
            header_info = self._detect_headers(values_matrix, sheet_data, cell_range)
            logger.debug(f"Header info result: {header_info}")

        # Extract DataFrame based on header detection
        df = self._create_dataframe(values_matrix, header_info)

        if df is None:
            # Still return header info even if we can't create a valid dataframe
            return None, header_info, 0.0

        if len(df) < self.min_data_rows:
            # For plate formats, we might have valid data even with few rows
            if header_info and header_info.table_type == "plate_map":
                # Plate maps are valid even with minimal data
                quality_score = 0.95
                return df, header_info, quality_score
            else:
                return None, header_info, 0.0

        # Calculate quality score
        quality_score = self._calculate_quality_score(df, header_info, density)

        return df, header_info, quality_score

    def _extract_values_matrix(
        self, range_data: list[list[CellData | None]]
    ) -> list[list[Any]] | None:
        """Convert CellData matrix to values matrix.

        Args
        ----
        range_data : list[list[CellData | None]]
            2D matrix of CellData objects or None values representing
            a rectangular region of cells.

        Returns
        -------
        list[list[Any]] | None
            2D matrix of cell values extracted from the CellData objects.
            Returns None if range_data is empty. Empty cells are represented
            as None in the output matrix.

        Notes
        -----
        This method extracts raw values from CellData objects, stripping away
        formatting and metadata. The output matrix has the same dimensions as
        the input and is suitable for DataFrame construction or analysis.
        """
        if not range_data:
            return None

        values_matrix = []
        for row in range_data:
            row_values: list[Any] = []
            for cell in row:
                if cell is None or cell.is_empty:
                    row_values.append(None)
                else:
                    row_values.append(cell.value)
            values_matrix.append(row_values)

        return values_matrix

    def _calculate_density(self, values_matrix: list[list[Any]]) -> float:
        """Calculate the ratio of non-empty cells.

        Args
        ----
        values_matrix : list[list[Any]]
            2D matrix of cell values to analyze.

        Returns
        -------
        float
            Density ratio between 0.0 and 1.0, where 1.0 means all cells
            have values and 0.0 means all cells are empty (None).

        Notes
        -----
        Density is a key metric for table quality. Very low density (< 0.2)
        often indicates scattered data rather than a proper table. This metric
        is used to filter out regions that don't contain meaningful tabular data.
        """
        total_cells = len(values_matrix) * len(values_matrix[0]) if values_matrix else 0
        if total_cells == 0:
            return 0.0

        non_empty = sum(1 for row in values_matrix for val in row if val is not None)

        return non_empty / total_cells

    def _detect_headers(
        self,
        values_matrix: list[list[Any]],
        sheet_data: SheetData,
        cell_range: CellRange,
    ) -> HeaderDetectionResult:
        """Detect headers in the data matrix with advanced analysis.

        Args
        ----
        values_matrix : list[list[Any]]
            2D matrix of cell values to analyze for headers.
        sheet_data : SheetData
            Original sheet data for accessing cell formatting information.
        cell_range : CellRange
            Range boundaries for mapping matrix positions to sheet cells.

        Returns
        -------
        HeaderDetectionResult
            Detailed information about detected headers including:
            - Whether headers exist
            - Number of header rows/columns
            - Header orientation (vertical/horizontal)
            - Extracted header values
            - Confidence score

        Notes
        -----
        This method uses multiple detection strategies:

        1. **Plate Map Detection**: Checks for standard plate formats (96-well, etc.)
        2. **Horizontal Headers**: Detects transposed tables with headers in columns
        3. **Vertical Headers**: Most common case with headers in top rows

        The detection considers formatting (bold text), data type consistency,
        and structural patterns to identify headers with high accuracy.
        """
        # First check if this might be a plate map
        plate_info = self._detect_plate_format(values_matrix)
        if plate_info:
            return plate_info

        # Check for vertical headers (normal case)
        vertical_result = self._detect_vertical_headers_enhanced(
            values_matrix, sheet_data, cell_range
        )

        # Check for horizontal headers (transposed case)
        horizontal_result = self._detect_horizontal_headers_enhanced(values_matrix)

        # Choose the better orientation
        if vertical_result.confidence > horizontal_result.confidence:
            return vertical_result
        else:
            return horizontal_result

    def _detect_vertical_headers(
        self, values_matrix: list[list[Any]]
    ) -> tuple[float, list[str], int]:
        """Detect headers in rows (normal orientation)."""
        if len(values_matrix) < 2:
            return 0.0, [], 0

        # Check first few rows as potential headers
        max_header_rows = min(3, len(values_matrix) - self.min_data_rows)
        best_score = 0.0
        best_headers = []
        best_header_rows = 0

        for header_rows in range(1, max_header_rows + 1):
            headers = self._extract_vertical_headers(values_matrix, header_rows)
            score = self._score_headers(headers, values_matrix[header_rows:])

            if score > best_score:
                best_score = score
                best_headers = headers
                best_header_rows = header_rows

        return best_score, best_headers, best_header_rows

    def _detect_horizontal_headers(
        self, values_matrix: list[list[Any]]
    ) -> tuple[float, list[str], int]:
        """Detect headers in columns (transposed orientation)."""
        if not values_matrix or len(values_matrix[0]) < 2:
            return 0.0, [], 0

        # Transpose the matrix
        transposed = list(zip(*values_matrix, strict=False))

        # Check first few columns as potential headers
        max_header_cols = min(3, len(transposed) - self.min_data_rows)
        best_score = 0.0
        best_headers = []
        best_header_cols = 0

        for header_cols in range(1, max_header_cols + 1):
            headers = [
                str(row[0]) if row[0] is not None else f"Col_{i}"
                for i, row in enumerate(values_matrix)
            ]
            # Score based on the remaining columns
            data_cols = [list(row[header_cols:]) for row in values_matrix]
            score = self._score_headers(headers, data_cols)

            if score > best_score:
                best_score = score
                best_headers = headers
                best_header_cols = header_cols

        return best_score, best_headers, best_header_cols

    def _extract_vertical_headers(
        self, values_matrix: list[list[Any]], header_rows: int
    ) -> list[str]:
        """Extract headers from specified number of rows."""
        if header_rows == 1:
            # Single header row
            header_row = values_matrix[0]
            return [str(val) if val is not None else f"Col_{i}" for i, val in enumerate(header_row)]
        else:
            # Multi-row headers - concatenate
            headers = []
            for col_idx in range(len(values_matrix[0])):
                header_parts = []
                for row_idx in range(header_rows):
                    val = values_matrix[row_idx][col_idx]
                    if val is not None:
                        header_parts.append(str(val))

                if header_parts:
                    headers.append(" ".join(header_parts))
                else:
                    headers.append(f"Col_{col_idx}")

            return headers

    def _calculate_coverage(self, row: list[Any]) -> float:
        """Calculate the coverage score (ratio of filled cells)."""
        if not row:
            return 0.0

        filled = sum(1 for cell in row if cell is not None and str(cell).strip())
        return filled / len(row)

    def _score_headers(self, headers: list[str], data_rows: list[list[Any]]) -> float:
        """Score the quality of detected headers.

        Args
        ----
        headers : list[str]
            List of potential header values to evaluate.
        data_rows : list[list[Any]]
            Data rows below the headers for type consistency analysis.

        Returns
        -------
        float
            Quality score between 0.0 and 1.0, where higher scores indicate
            better header quality.

        Notes
        -----
        The scoring algorithm evaluates multiple factors:

        1. **Uniqueness** (30%): Headers should be unique identifiers
        2. **Text Content** (30%): Headers are typically text, not numbers
        3. **Type Consistency** (40%): Data columns should have consistent types

        This multi-factor approach helps distinguish true headers from data rows
        that happen to be at the top of a table.
        """
        if not headers or not data_rows:
            return 0.0

        score = 0.0

        # Check header uniqueness
        unique_ratio = len(set(headers)) / len(headers)
        score += unique_ratio * 0.3

        # Check if headers are mostly text
        text_headers = sum(
            1 for h in headers if h and not h.replace(".", "").replace("-", "").isdigit()
        )
        text_ratio = text_headers / len(headers)
        score += text_ratio * 0.3

        # Check data type consistency in columns
        if data_rows:
            type_consistency = self._check_column_type_consistency(data_rows)
            score += type_consistency * 0.4

        return score

    def _check_column_type_consistency(self, data_rows: list[list[Any]]) -> float:
        """Check if columns have consistent data types.

        Args
        ----
        data_rows : list[list[Any]]
            Data rows to analyze for type consistency. Each inner list
            represents a row of data values.

        Returns
        -------
        float
            Consistency score between 0.0 and 1.0, where 1.0 means all columns
            have perfectly consistent data types (all numeric, all dates, etc.).

        Notes
        -----
        This method categorizes values into broad types:

        - **numeric**: int, float, or numeric strings
        - **date**: datetime objects or date-like strings
        - **text**: all other non-empty values

        A column is considered consistent if at least 80% of its non-empty values
        share the same type. This threshold allows for some data quality issues
        while still identifying well-structured tables.
        """
        if not data_rows or not data_rows[0]:
            return 0.0

        num_cols = len(data_rows[0])
        consistent_cols = 0

        for col_idx in range(num_cols):
            col_values = [
                row[col_idx] for row in data_rows if col_idx < len(row) and row[col_idx] is not None
            ]

            if not col_values:
                continue

            # Check type consistency
            types = set()
            for val in col_values:
                if isinstance(val, int | float):
                    types.add("numeric")
                elif isinstance(val, str):
                    # Try to parse as number
                    try:
                        float(val)
                        types.add("numeric")
                    except (ValueError, TypeError):
                        types.add("text")
                else:
                    types.add(type(val).__name__)

            if len(types) == 1:
                consistent_cols += 1

        return consistent_cols / num_cols if num_cols > 0 else 0.0

    def _create_dataframe(
        self, values_matrix: list[list[Any]], header_info: HeaderDetectionResult | None
    ) -> pd.DataFrame | None:
        """Create DataFrame from values matrix and header info."""
        if not values_matrix:
            return None

        try:
            if header_info is None or not header_info.has_headers:
                # No headers detected
                df = pd.DataFrame(values_matrix)
            elif header_info.orientation == "vertical":
                # Normal orientation with headers in rows
                # Skip title rows if any
                start_row = 0
                if header_info.title_rows:
                    # Find the first row after title rows
                    for i in range(len(values_matrix)):
                        if i not in header_info.title_rows:
                            start_row = i
                            break

                # Skip header rows from the adjusted start
                data_start = start_row + header_info.header_rows
                data_rows = values_matrix[data_start:]
                df = pd.DataFrame(data_rows, columns=header_info.headers)
            else:
                # Transposed orientation with headers in columns
                # Extract data excluding header columns
                data_matrix = []
                for row in values_matrix:
                    data_matrix.append(row[header_info.header_columns :])

                df = pd.DataFrame(data_matrix, columns=header_info.headers)

            return df

        except Exception as e:
            logger.debug(f"Failed to create DataFrame: {e}")
            return None

    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        header_info: HeaderDetectionResult | None,
        density: float,
    ) -> float:
        """Calculate overall quality score for the extracted table."""
        score = 0.0

        # Data density component (30%)
        score += density * 0.3

        # Header quality component (30%)
        if header_info and header_info.has_headers:
            score += header_info.confidence * 0.3
        else:
            score += 0.1  # Small penalty for no headers

        # Shape quality component (20%)
        # Prefer tables with reasonable aspect ratios
        rows, cols = df.shape
        if rows >= self.min_data_rows and cols > 0:
            aspect_ratio = min(rows / cols, cols / rows)
            shape_score = min(1.0, aspect_ratio / 0.1)  # Best at 10:1 or better
            score += shape_score * 0.2

        # Data quality component (20%)
        # Check for variety in data
        non_null_ratio = df.notna().sum().sum() / (rows * cols) if rows * cols > 0 else 0
        score += non_null_ratio * 0.2

        return min(1.0, score)

    def _detect_plate_format(self, values_matrix: list[list[Any]]) -> HeaderDetectionResult | None:
        """Detect if this is a standard plate map format."""
        logger.debug(
            f"Checking plate format. Matrix size: {len(values_matrix) if values_matrix else 0}"
        )
        if not values_matrix or len(values_matrix) < 3:
            return None

        # Standard plate formats: wells -> (rows, cols)
        PLATE_FORMATS = {
            6: [(2, 3), (3, 2)],
            24: [(4, 6), (6, 4)],
            96: [(8, 12), (12, 8)],
            384: [(16, 24), (24, 16)],
            1536: [(32, 48), (48, 32)],
        }

        # Check each possible plate format, preferring exact matches
        best_match = None
        best_score = 0

        for wells, dimensions in PLATE_FORMATS.items():
            for rows, cols in dimensions:
                # Check if matrix dimensions are compatible (accounting for headers)
                logger.debug(
                    f"Checking {wells}-well: need {rows + 1}x{cols + 1}, have {len(values_matrix)}x{len(values_matrix[0]) if values_matrix else 0}"
                )
                if len(values_matrix) >= rows + 1 and len(values_matrix[0]) >= cols + 1:
                    # Check if first column contains row labels (A, B, C, etc.)
                    row_labels_valid = self._check_plate_row_labels(values_matrix, rows)
                    # Check if first row contains column numbers (1, 2, 3, etc.)
                    col_labels_valid = self._check_plate_col_labels(values_matrix[0], cols)

                    logger.debug(
                        f"  Row labels valid: {row_labels_valid}, Col labels valid: {col_labels_valid}"
                    )
                    if row_labels_valid and col_labels_valid:
                        # Calculate match score (prefer exact dimensions)
                        row_diff = abs(len(values_matrix) - (rows + 1))
                        col_diff = abs(len(values_matrix[0]) - (cols + 1))
                        score = 1000 - (row_diff + col_diff)  # Higher score for exact match

                        if score > best_score:
                            best_score = score
                            best_match = (wells, cols)

        if best_match:
            wells, cols = best_match
            # Include row label column in headers
            headers = ["Well"]  # First column for row labels
            for i in range(1, cols + 1):
                headers.append(str(i))

            return HeaderDetectionResult(
                has_headers=True,
                header_rows=1,
                header_columns=0,
                orientation="vertical",
                headers=headers,
                confidence=0.95,
                title_rows=[],
                data_sample_size=0,
                column_types={},
                table_type="plate_map",
                plate_format=wells,
            )

        return None

    def _check_plate_row_labels(self, values_matrix: list[list[Any]], expected_rows: int) -> bool:
        """Check if first column contains valid plate row labels (A, B, C, etc.)."""
        if len(values_matrix) < expected_rows + 1:
            return False

        expected_labels = [chr(ord("A") + i) for i in range(expected_rows)]
        actual_labels = []

        for i in range(1, min(expected_rows + 1, len(values_matrix))):
            if values_matrix[i] and values_matrix[i][0] is not None:
                actual_labels.append(str(values_matrix[i][0]).strip().upper())

        return actual_labels == expected_labels

    def _check_plate_col_labels(self, header_row: list[Any], expected_cols: int) -> bool:
        """Check if header row contains valid column numbers."""
        if not header_row or len(header_row) < expected_cols + 1:
            return False

        for i in range(1, min(expected_cols + 1, len(header_row))):
            if header_row[i] is None:
                return False
            try:
                col_num = int(str(header_row[i]))
                if col_num != i:
                    return False
            except (ValueError, TypeError):
                return False

        return True

    def _is_title_row(self, row: list[Any]) -> bool:
        """Check if a row is likely a title (few filled cells)."""
        if not row:
            return False

        filled_cells = sum(1 for cell in row if cell is not None and str(cell).strip())
        total_cells = len(row)

        # Title rows typically have 1-2 filled cells out of many
        return filled_cells <= 2 and total_cells > 3

    def _detect_vertical_headers_enhanced(
        self,
        values_matrix: list[list[Any]],
        sheet_data: SheetData,
        cell_range: CellRange,
    ) -> HeaderDetectionResult:
        """Enhanced vertical header detection with deep type analysis."""
        if len(values_matrix) < 2:
            return HeaderDetectionResult(
                has_headers=False,
                header_rows=0,
                header_columns=0,
                orientation="vertical",
                headers=[],
                confidence=0.0,
            )

        best_score = 0.0
        best_header_start = 0
        best_header_end = 0
        title_rows = []

        # Check first 10 rows as potential headers
        max_test_rows = min(10, len(values_matrix) - 2)

        for start_row in range(max_test_rows):
            # Skip if this looks like a title row
            if self._is_title_row(values_matrix[start_row]):
                title_rows.append(start_row)
                continue

            # Test 1-3 rows as potential multi-row header
            for header_rows in range(1, min(4, len(values_matrix) - start_row - 1)):
                # Get data sample for type analysis (up to 100 rows)
                data_start = start_row + header_rows
                data_end = min(len(values_matrix), data_start + 100)
                data_sample = values_matrix[data_start:data_end]

                if not data_sample:
                    continue

                # Calculate comprehensive score
                type_score, column_types = self._calculate_type_consistency(data_sample)
                header_score = self._score_header_quality_multi(
                    values_matrix[start_row : start_row + header_rows]
                )
                coverage_score = self._calculate_coverage(values_matrix[start_row])

                # Weight the scores
                total_score = type_score * 0.5 + header_score * 0.3 + coverage_score * 0.2

                if total_score > best_score:
                    best_score = total_score
                    best_header_start = start_row
                    best_header_end = start_row + header_rows

        # Extract the best headers
        if best_score > 0.5:
            # Use the enhanced header extraction with merged cell support
            headers = self._extract_vertical_headers_with_merged(
                values_matrix[best_header_start:best_header_end],
                best_header_end - best_header_start,
                sheet_data,
                cell_range.start_row + best_header_start,
            )

            # Get column types from final analysis
            data_start = best_header_end
            data_end = min(len(values_matrix), data_start + 100)
            data_sample = values_matrix[data_start:data_end]
            _, column_types = self._calculate_type_consistency(data_sample)

            return HeaderDetectionResult(
                has_headers=True,
                header_rows=best_header_end - best_header_start,
                header_columns=0,
                orientation="vertical",
                headers=headers,
                confidence=best_score,
                title_rows=title_rows,
                data_sample_size=len(data_sample),
                column_types=column_types,
                table_type="standard",
            )
        else:
            return HeaderDetectionResult(
                has_headers=False,
                header_rows=0,
                header_columns=0,
                orientation="vertical",
                headers=[],
                confidence=0.0,
                title_rows=title_rows,
            )

    def _detect_horizontal_headers_enhanced(
        self, values_matrix: list[list[Any]]
    ) -> HeaderDetectionResult:
        """Enhanced horizontal header detection (for transposed tables)."""
        # For now, keep simple implementation - can enhance later
        if not values_matrix or len(values_matrix[0]) < 2:
            return HeaderDetectionResult(
                has_headers=False,
                header_rows=0,
                header_columns=0,
                orientation="horizontal",
                headers=[],
                confidence=0.0,
            )

        # Use existing logic for now
        score, headers, cols = self._detect_horizontal_headers(values_matrix)

        return HeaderDetectionResult(
            has_headers=score > 0.5,
            header_rows=0,
            header_columns=cols,
            orientation="horizontal",
            headers=headers,
            confidence=score,
        )

    def _calculate_type_consistency(
        self, data_sample: list[list[Any]]
    ) -> tuple[float, dict[int, str]]:
        """Calculate type consistency for each column in the data sample."""
        if not data_sample or not data_sample[0]:
            return 0.0, {}

        column_types = {}
        consistency_scores = []

        for col_idx in range(len(data_sample[0])):
            types_count = {"numeric": 0, "text": 0, "date": 0, "boolean": 0, "empty": 0}

            for row in data_sample:
                if col_idx < len(row):
                    cell_type = self._detect_cell_type(row[col_idx])
                    types_count[cell_type] += 1

            # Find dominant type
            total_non_empty = sum(v for k, v in types_count.items() if k != "empty")
            if total_non_empty > 0:
                dominant_type = max(
                    (k for k in types_count if k != "empty"),
                    key=lambda k: types_count[k],
                )
                dominant_count = types_count[dominant_type]
                consistency = dominant_count / total_non_empty
                consistency_scores.append(consistency)
                column_types[col_idx] = dominant_type
            else:
                column_types[col_idx] = "empty"

        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        )
        return avg_consistency, column_types

    def _detect_cell_type(self, value: Any) -> str:
        """Detect the type of a cell value."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return "empty"

        # Check boolean
        if isinstance(value, bool):
            return "boolean"

        # Check numeric
        if isinstance(value, int | float):
            return "numeric"

        # Check string that might be numeric
        if isinstance(value, str):
            value_stripped = value.strip()

            # Try to parse as number
            try:
                float(value_stripped.replace(",", ""))
                return "numeric"
            except (ValueError, TypeError):
                pass

            # Check for date patterns
            if any(sep in value_stripped for sep in ["-", "/", ":"]) and len(value_stripped) >= 6:
                # Simple date detection - could be enhanced
                return "date"

            # Check for boolean strings
            if value_stripped.lower() in ["true", "false", "yes", "no", "y", "n"]:
                return "boolean"

        return "text"

    def _score_header_quality_multi(self, header_rows: list[list[Any]]) -> float:
        """Score the quality of multi-row headers."""
        if not header_rows:
            return 0.0

        scores = []

        for row in header_rows:
            # Check text ratio
            text_count = 0
            filled_count = 0

            for cell in row:
                if cell is not None and str(cell).strip():
                    filled_count += 1
                    if self._detect_cell_type(cell) == "text":
                        text_count += 1

            if filled_count > 0:
                text_ratio = text_count / filled_count
                scores.append(text_ratio)

        # Also check uniqueness in the last row (most specific headers)
        if header_rows:
            last_row = header_rows[-1]
            filled_values = [
                str(cell) for cell in last_row if cell is not None and str(cell).strip()
            ]
            if filled_values:
                unique_ratio = len(set(filled_values)) / len(filled_values)
                scores.append(unique_ratio)

        return sum(scores) / len(scores) if scores else 0.0

    def _extract_vertical_headers_with_merged(
        self,
        values_matrix: list[list[Any]],
        header_rows: int,
        sheet_data: SheetData,
        start_row: int,
    ) -> list[str]:
        """Extract headers from rows, handling merged cells."""
        if not values_matrix or header_rows == 0:
            return []

        if header_rows == 1:
            # Single header row
            return [
                str(val) if val is not None else f"Col_{i}"
                for i, val in enumerate(values_matrix[0])
            ]

        # Multi-row headers - need to handle merged cells
        num_cols = len(values_matrix[0]) if values_matrix[0] else 0
        headers = []

        # Build a mapping of merged cell ranges to their values
        merged_cell_values = {}
        for row_idx in range(header_rows):
            for col_idx in range(num_cols):
                cell = sheet_data.get_cell(start_row + row_idx, col_idx)
                if cell and cell.is_merged and cell.merge_range and cell.value is not None:
                    # Store the value for this merge range
                    merged_cell_values[cell.merge_range] = str(cell.value).strip()

        # Extract headers for each column
        for col_idx in range(num_cols):
            header_parts = []

            for row_idx in range(header_rows):
                cell = sheet_data.get_cell(start_row + row_idx, col_idx)

                if cell and cell.value is not None:
                    value = str(cell.value).strip()
                    if value:
                        header_parts.append(value)
                elif cell and cell.is_merged and cell.merge_range:
                    # This is part of a merged cell - get the value from our mapping
                    if cell.merge_range in merged_cell_values:
                        value = merged_cell_values[cell.merge_range]
                        if value and value not in header_parts:
                            header_parts.append(value)

            if header_parts:
                # Join with space, avoiding duplicates
                unique_parts: list[str] = []
                for part in header_parts:
                    if not unique_parts or part != unique_parts[-1]:
                        unique_parts.append(part)
                headers.append(" ".join(unique_parts))
            else:
                headers.append(f"Col_{col_idx}")

        return headers
