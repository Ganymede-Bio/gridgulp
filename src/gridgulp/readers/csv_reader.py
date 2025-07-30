"""CSV file reader with intelligent parsing and encoding detection."""

import csv
import logging
from pathlib import Path
from typing import Any

from ..models.file_info import FileInfo, FileType
from ..models.sheet_data import CellData, FileData, SheetData
from .base_reader import ReaderError, SyncBaseReader

logger = logging.getLogger(__name__)


class CSVReader(SyncBaseReader):
    """Reader for CSV and TSV files with intelligent parsing."""

    def __init__(self, file_path: Path, file_info: FileInfo):
        """Initialize CSV reader.

        Args:
            file_path: Path to CSV file
            file_info: File information
        """
        super().__init__(file_path, file_info)
        self._encoding: str | None = None
        self._dialect: csv.Dialect | None = None

    def can_read(self) -> bool:
        """Check if can read CSV files.

        Returns
        -------
        bool
            True if the file type is CSV or TSV; False otherwise.

        Notes
        -----
        This method validates that the CSVReader can handle the given file type.
        Text files (.txt) are handled by TextReader instead, even if they
        contain delimited data.
        """
        return self.file_info.type in {FileType.CSV, FileType.TSV}

    def get_supported_formats(self) -> list[str]:
        """Get supported CSV formats.

        Returns
        -------
        list[str]
            List of supported file extensions: ["csv", "tsv", "txt"].

        Notes
        -----
        While this method returns "txt" as supported, the actual can_read()
        method only accepts CSV and TSV file types. Text files are included
        here for completeness but are typically handled by TextReader.
        """
        return ["csv", "tsv", "txt"]

    def read_sync(self) -> FileData:
        """Read CSV file synchronously.

        Returns:
            FileData with single sheet containing CSV data

        Raises:
            ReaderError: If file cannot be read
        """
        self._log_read_start()
        self.validate_file()

        try:
            # Detect encoding and dialect
            self._detect_format()

            # Read CSV data
            sheet_data = self._read_csv_data()

            # Create file data with single sheet
            file_data = FileData(
                sheets=[sheet_data],
                metadata={
                    "encoding": self._encoding,
                    "dialect": (
                        {
                            "delimiter": self._dialect.delimiter,
                            "quotechar": self._dialect.quotechar,
                            "quoting": self._dialect.quoting,
                            "skipinitialspace": self._dialect.skipinitialspace,
                            "lineterminator": repr(self._dialect.lineterminator),
                        }
                        if self._dialect
                        else None
                    ),
                },
                file_format=self.file_info.type.value,
            )

            self._log_read_complete(file_data)
            return file_data

        except Exception as e:
            raise ReaderError(f"Failed to read CSV file: {e}") from e

    def _detect_format(self) -> None:
        """Detect encoding and CSV dialect.

        Notes
        -----
        This method performs two critical detection steps:

        1. **Encoding Detection**: Uses multiple strategies including BOM detection,
           chardet analysis, and pattern matching to identify the file encoding.
        2. **Dialect Detection**: Uses csv.Sniffer and manual analysis to determine
           the delimiter, quote character, and other CSV formatting rules.

        The detected encoding and dialect are stored as instance attributes for
        use during the actual file reading phase. Detection uses only the first
        8KB of the file for efficiency.
        """
        # Read sample for detection
        with open(self.file_path, "rb") as f:
            sample_bytes = f.read(8192)  # Read up to 8KB for detection

        # Detect encoding
        self._encoding = self._detect_encoding(sample_bytes)
        self.logger.debug(f"Detected encoding: {self._encoding}")

        # Decode sample for dialect detection
        try:
            sample_text = sample_bytes.decode(self._encoding, errors="replace")
        except Exception as e:
            self.logger.warning(f"Failed to decode with {self._encoding}: {e}")
            self._encoding = "utf-8"
            sample_text = sample_bytes.decode(self._encoding, errors="replace")

        # Detect CSV dialect
        self._dialect = self._detect_dialect(sample_text)
        self.logger.debug(f"Detected delimiter: '{self._dialect.delimiter}'")

    def _detect_dialect(self, sample: str) -> csv.Dialect:
        """Detect CSV dialect from sample text.

        Args:
            sample: Sample text from file

        Returns:
            Detected CSV dialect
        """
        try:
            # Use csv.Sniffer to detect dialect
            sniffer = csv.Sniffer()

            # Try to detect delimiter
            try:
                dialect = sniffer.sniff(sample, delimiters=",;	|")
                self.logger.debug(f"Sniffer detected delimiter: '{dialect.delimiter}'")
                return dialect  # type: ignore[return-value]
            except csv.Error:
                # Sniffer failed, try manual detection
                pass

            # Manual delimiter detection
            delimiter = self._detect_delimiter_manual(sample)

            # Create custom dialect
            class CustomDialect(csv.excel):
                delimiter = delimiter
                quoting = csv.QUOTE_MINIMAL

            return CustomDialect()

        except Exception as e:
            self.logger.warning(f"Dialect detection failed: {e}, using defaults")

            # Fallback based on file type
            if self.file_info.type == FileType.TSV:

                class TSVDialect(csv.excel):
                    delimiter = "	"

                return TSVDialect()
            else:
                return csv.excel()  # Default CSV dialect instance

    def _detect_delimiter_manual(self, sample: str) -> str:
        """Manually detect delimiter by counting occurrences.

        Args:
            sample: Sample text from file

        Returns:
            Most likely delimiter character
        """
        # Common delimiters to test
        delimiters = [",", ";", "	", "|", ":"]

        # Count occurrences of each delimiter
        delimiter_counts = {}
        lines = sample.split("\n")[:10]  # Check first 10 lines

        for delimiter in delimiters:
            consistent_count = 0
            consistent_lines = 0
            first_count = True

            for line in lines:
                if line.strip():  # Skip empty lines
                    count = line.count(delimiter)
                    if count > 0:
                        if first_count:
                            consistent_count = count
                            consistent_lines = 1
                            first_count = False
                        elif consistent_count == count:
                            consistent_lines += 1

            # Score based on consistency and frequency
            if consistent_count and consistent_count > 0:
                delimiter_counts[delimiter] = consistent_lines * consistent_count

        if delimiter_counts:
            best_delimiter = max(delimiter_counts, key=lambda x: delimiter_counts[x])
            self.logger.debug(
                f"Manual detection chose: '{best_delimiter}' (score: {delimiter_counts[best_delimiter]})"
            )
            return best_delimiter

        # Default fallback
        return "," if self.file_info.type == FileType.CSV else "	"

    def _read_csv_data(self) -> SheetData:
        """Read CSV data into SheetData structure.

        Returns:
            SheetData containing all CSV data
        """
        sheet_name = self.file_path.stem  # Use filename as sheet name
        sheet_data = SheetData(name=sheet_name)

        try:
            with open(self.file_path, encoding=self._encoding, newline="") as f:
                reader = csv.reader(f, dialect=self._dialect) if self._dialect else csv.reader(f)

                for row_idx, row in enumerate(reader):
                    for col_idx, cell_value in enumerate(row):
                        if cell_value:  # Only store non-empty cells
                            cell_data = self._create_cell_data(cell_value, row_idx, col_idx)
                            sheet_data.set_cell(row_idx, col_idx, cell_data)

                    # Safety limit to prevent memory issues
                    if row_idx > 1000000:  # 1M rows max
                        self.logger.warning(f"Reached row limit, truncating at {row_idx}")
                        break

        except UnicodeDecodeError as e:
            # Try with different encoding
            self.logger.warning(f"Encoding {self._encoding} failed: {e}")
            return self._read_csv_with_fallback_encoding(sheet_name)

        return sheet_data

    def _read_csv_with_fallback_encoding(self, sheet_name: str) -> SheetData:
        """Retry reading CSV with fallback encodings.

        Args:
            sheet_name: Name for the sheet

        Returns:
            SheetData or raises ReaderError
        """
        fallback_encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

        for encoding in fallback_encodings:
            if encoding == self._encoding:
                continue

            try:
                self.logger.debug(f"Trying fallback encoding: {encoding}")
                sheet_data = SheetData(name=sheet_name)

                with open(self.file_path, encoding=encoding, newline="") as f:
                    reader = (
                        csv.reader(f, dialect=self._dialect) if self._dialect else csv.reader(f)
                    )

                    for row_idx, row in enumerate(reader):
                        for col_idx, cell_value in enumerate(row):
                            if cell_value:
                                cell_data = self._create_cell_data(cell_value, row_idx, col_idx)
                                sheet_data.set_cell(row_idx, col_idx, cell_data)

                        if row_idx > 1000000:
                            break

                self.logger.info(f"Successfully read with fallback encoding: {encoding}")
                self._encoding = encoding
                return sheet_data

            except Exception as e:
                self.logger.debug(f"Fallback encoding {encoding} failed: {e}")
                continue

        raise ReaderError("Could not read CSV with any supported encoding")

    def _create_cell_data(self, value: str, row: int, col: int) -> CellData:
        """Create CellData from string value.

        Args:
            value: String value from CSV
            row: Row index
            col: Column index

        Returns:
            CellData with inferred type
        """
        # Try to infer data type and convert value
        converted_value, data_type = self._infer_type(value)

        return CellData(
            value=converted_value,
            formatted_value=value,  # Keep original string
            data_type=data_type,
            row=row,
            column=col,
        )

    def _infer_type(self, value: str) -> tuple[Any, str]:
        """Infer data type from string value.

        Args:
            value: String value to analyze

        Returns:
            Tuple of (converted_value, data_type)
        """
        stripped = value.strip()

        if not stripped:
            return None, "empty"

        # Try boolean
        if stripped.lower() in {"true", "false", "yes", "no", "1", "0"}:
            bool_value = stripped.lower() in {"true", "yes", "1"}
            return bool_value, "boolean"

        # Try integer
        try:
            if "." not in stripped and "e" not in stripped.lower():
                int_value = int(stripped.replace(",", ""))  # Handle thousand separators
                return int_value, "number"
        except ValueError:
            pass

        # Try float
        try:
            float_value = float(stripped.replace(",", ""))
            return float_value, "number"
        except ValueError:
            pass

        # Try to detect dates (basic patterns)
        if self._looks_like_date(stripped):
            # Keep as string for now, could add datetime parsing later
            return stripped, "date"

        # Default to string
        return stripped, "string"

    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date.

        Args:
            value: String to check

        Returns:
            True if looks like a date
        """
        # Simple heuristic - contains date-like separators and numbers
        date_indicators = ["-", "/", ".", " "]
        has_separator = any(sep in value for sep in date_indicators)
        has_digits = any(c.isdigit() for c in value)

        # Common date patterns (very basic)
        date_patterns = [
            lambda s: len(s) == 10 and s.count("-") == 2,  # YYYY-MM-DD
            lambda s: len(s) == 10 and s.count("/") == 2,  # MM/DD/YYYY
            lambda s: len(s) >= 8
            and any(
                word in s.lower()
                for word in [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
            ),
        ]

        return has_separator and has_digits and any(pattern(value) for pattern in date_patterns)
