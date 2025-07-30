"""Text file reader with CSV/TSV auto-detection."""

import csv
import io
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..models.file_info import FileInfo
from ..models.sheet_data import SheetData
from ..utils.file_magic import FileFormatDetector
from .base_reader import ReaderError, SyncBaseReader

logger = logging.getLogger(__name__)


class TextReader(SyncBaseReader):
    """Reader for text files with CSV/TSV auto-detection."""

    def __init__(self, file_path: Path, file_info: FileInfo):
        super().__init__(file_path, file_info)
        self._detected_format: str | None = None
        self._detected_encoding: str | None = None
        self._detector = FileFormatDetector()  # For sophisticated encoding detection

    def can_read(self) -> bool:
        """Check if the text file can be read as tabular data.

        Returns
        -------
        bool
            True if the file contains tabular data (delimited format);
            False if it's plain text without structure.

        Notes
        -----
        This method performs preliminary analysis to determine if the text file
        contains structured tabular data. It detects encoding and checks for
        consistent delimiter patterns. Files without clear tabular structure
        are rejected to avoid processing non-data text files.
        """
        try:
            # Try to detect encoding and format
            self._detect_encoding_internal()
            self._detect_tabular_format()
            return self._detected_format is not None
        except Exception as e:
            logger.debug(f"Cannot read text file {self.file_path}: {e}")
            return False

    def get_supported_formats(self) -> list[str]:
        """Get supported formats.

        Returns
        -------
        list[str]
            List containing ["txt"] as the only supported extension.

        Notes
        -----
        TextReader specifically handles .txt files that contain tabular data.
        CSV and TSV files with proper extensions are handled by CSVReader instead.
        """
        return ["txt"]

    def read_sheets(self) -> Iterator[SheetData]:
        """Read text file as a single sheet.

        Yields
        ------
        SheetData
            A single sheet containing the parsed tabular data from the text file.

        Raises
        ------
        ReaderError
            If the file cannot be read, contains no tabular data, or parsing fails.

        Notes
        -----
        Text files are always treated as containing a single "sheet" of data.
        The method automatically detects the delimiter (comma, tab, pipe, etc.)
        and parses the content accordingly. Empty rows are skipped during parsing.
        """
        if not self.can_read():
            raise ReaderError(f"Cannot read text file: {self.file_path}")

        try:
            # Read file content with detected encoding
            with open(self.file_path, encoding=self._detected_encoding) as f:
                content = f.read()

            # Parse based on detected format
            if self._detected_format == "csv":
                delimiter = ","
            elif self._detected_format == "tsv":
                delimiter = "\t"
            else:
                # Try to detect delimiter
                delimiter = self._detect_delimiter(content)

            # Parse CSV content
            rows = []
            csv_reader = csv.reader(io.StringIO(content), delimiter=delimiter)

            for _row_idx, row in enumerate(csv_reader):
                if row:  # Skip empty rows
                    rows.append(row)

            if not rows:
                raise ReaderError("No tabular data found in text file")

            # Convert to SheetData format
            sheet_data = self._convert_to_sheet_data(rows)
            yield sheet_data

        except Exception as e:
            raise ReaderError(f"Failed to read text file {self.file_path}: {e}") from e

    def read_sync(self) -> Any:
        """Synchronously read the text file."""
        from ..models.sheet_data import FileData

        sheets = list(self.read_sheets())
        return FileData(
            sheets=sheets,
            metadata={
                "encoding": self._detected_encoding,
                "format": self._detected_format,
            },
            file_format=self.file_info.type.value,
        )

    def _detect_encoding_internal(self) -> None:
        """Internal method to detect file encoding using sophisticated detection."""
        if self._detected_encoding:
            return

        try:
            # Read first chunk for encoding detection
            with open(self.file_path, "rb") as f:
                raw_data = f.read(16384)  # Read first 16KB for better detection

            # Use sophisticated encoding detection
            encoding_result = self._detector._detect_encoding_sophisticated(
                raw_data, self.file_path
            )

            self._detected_encoding = encoding_result.encoding
            logger.debug(
                f"Sophisticated encoding detection: {encoding_result.encoding} "
                f"(method: {encoding_result.method}, confidence: {encoding_result.confidence:.2f}, "
                f"validated: {encoding_result.validated}, BOM: {encoding_result.bom_detected})"
            )

        except Exception as e:
            logger.warning(f"Failed to detect encoding, using utf-8: {e}")
            self._detected_encoding = "utf-8"

    def _detect_tabular_format(self) -> None:
        """Detect if text file contains tabular data."""
        if self._detected_format:
            return

        try:
            # Read a sample of the file with better handling for long lines
            with open(self.file_path, encoding=self._detected_encoding, errors="replace") as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 15:  # Read more lines for better detection
                        break
                    # Handle very long lines by truncating for analysis
                    line_sample = line[:2000] if len(line) > 2000 else line
                    sample_lines.append(line_sample.strip())

            if not sample_lines:
                return

            # Enhanced delimiter detection for scientific data
            delimiters = [
                "\t",
                ",",
                "|",
                ";",
                " ",
            ]  # Tab first (common in scientific data)
            delimiter_scores = {}

            for delimiter in delimiters:
                score = self._score_delimiter_enhanced(sample_lines, delimiter)
                delimiter_scores[delimiter] = score

            # Find best delimiter
            best_delimiter = max(delimiter_scores, key=lambda x: delimiter_scores[x])
            best_score = delimiter_scores[best_delimiter]

            logger.debug(f"Delimiter analysis: best='{best_delimiter}', score={best_score}")

            # Lower threshold for scientific data
            if best_score >= 1.5:  # More lenient for complex scientific files
                if best_delimiter == "\t":
                    self._detected_format = "tsv"
                elif best_delimiter == ",":
                    self._detected_format = "csv"
                else:
                    self._detected_format = "csv"  # Default to CSV for other delimiters

                logger.debug(
                    f"Detected tabular format: {self._detected_format} with delimiter '{best_delimiter}'"
                )
            else:
                logger.debug(f"No consistent tabular format detected (best score: {best_score})")

        except Exception as e:
            logger.warning(f"Failed to detect tabular format: {e}")

    def _score_delimiter(self, lines: list[str], delimiter: str) -> int:
        """Score how well a delimiter works for the given lines."""
        if not lines:
            return 0

        column_counts = []
        for line in lines:
            if line:  # Skip empty lines
                parts = line.split(delimiter)
                column_counts.append(len(parts))

        if not column_counts:
            return 0

        # Score based on consistency and minimum column count
        most_common_count = max(set(column_counts), key=column_counts.count)
        consistency = column_counts.count(most_common_count) / len(column_counts)

        # Require at least 2 columns and good consistency
        if most_common_count >= 2 and consistency >= 0.7:
            return most_common_count

        return 0

    def _score_delimiter_enhanced(self, lines: list[str], delimiter: str) -> float:
        """Enhanced delimiter scoring for scientific/instrument data."""
        if not lines:
            return 0.0

        column_counts = []
        valid_lines = 0

        for line in lines:
            if line and len(line.strip()) > 0:  # Skip empty lines
                if delimiter == " ":
                    # For space delimiter, handle multiple spaces and formatting
                    parts = [p for p in line.split() if p.strip()]
                    count = len(parts)
                else:
                    count = line.count(delimiter) + 1  # +1 because N delimiters = N+1 columns

                if count > 1:  # Must have at least 2 columns
                    column_counts.append(count)
                    valid_lines += 1

        if not column_counts or valid_lines < 2:
            return 0.0

        # Enhanced scoring that considers consistency and data quality for scientific data
        unique_counts = list(set(column_counts))

        if len(unique_counts) == 1:  # Perfect consistency
            base_score = unique_counts[0] * valid_lines * 0.1
            return base_score
        elif (
            len(unique_counts) <= 5
        ):  # More lenient for scientific data with different table sections
            avg_count = sum(column_counts) / len(column_counts)

            # For scientific data, even different column counts can be valid if tabs are present
            # Give credit for having delimiter-separated content
            if avg_count >= 5:  # At least 5 columns on average suggests structured data
                base_score = (
                    avg_count * valid_lines * 0.08
                )  # Slightly lower multiplier for varied data

                # Bonus if most lines have many columns (common in scientific data)
                high_column_lines = sum(1 for count in column_counts if count >= 10)
                if high_column_lines >= valid_lines * 0.5:  # 50% of lines have >=10 columns
                    base_score *= 1.2

                return base_score
            else:
                # Fewer columns but still structured
                consistency_ratio = column_counts.count(
                    max(set(column_counts), key=column_counts.count)
                ) / len(column_counts)
                consistency_bonus = 1.0 if consistency_ratio >= 0.6 else consistency_ratio
                return (avg_count * valid_lines * 0.1) * consistency_bonus
        else:
            # Too many different column counts, but still give some credit if there are many tabs
            if valid_lines >= 2 and sum(column_counts) / len(column_counts) >= 10:
                return 1.0  # Minimum score for clearly tab-separated data
            return 0.0

    def _detect_delimiter(self, content: str) -> str:
        """Detect the best delimiter for CSV parsing."""
        # Try csv.Sniffer first
        try:
            sniffer = csv.Sniffer()
            sample = content[:1024]  # First 1KB
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except Exception:
            pass

        # Fallback to manual detection
        lines = content.split("\n")[:10]  # First 10 lines
        for delimiter in [",", "\t", "|", ";"]:
            score = self._score_delimiter(lines, delimiter)
            if score >= 2:
                return delimiter

        # Default to comma
        return ","

    def _convert_to_sheet_data(self, rows: list[list[str]]) -> SheetData:
        """Convert parsed rows to SheetData."""
        if not rows:
            raise ReaderError("No data to convert")

        # Import the proper models
        from ..models.sheet_data import CellData

        # Create SheetData object following the pattern from CSV reader
        sheet_data = SheetData(name=self.file_path.stem)

        for row_idx, row in enumerate(rows):
            for col_idx, cell_value in enumerate(row):
                if cell_value.strip():  # Only store non-empty cells
                    # Create CellData object for each cell
                    cell_data = CellData(
                        value=cell_value.strip(),
                        formatted_value=cell_value.strip(),
                        data_type="string",  # Text files are string by default
                        row=row_idx,
                        column=col_idx,
                    )
                    sheet_data.set_cell(row_idx, col_idx, cell_data)

        return sheet_data
