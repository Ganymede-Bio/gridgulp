"""Enhanced file type detection using magic bytes, MIME types, and content analysis."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.file_info import FileType, UnsupportedFormatError

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Result of sophisticated encoding detection."""

    encoding: str
    confidence: float
    method: str  # 'bom', 'chardet', 'pattern', 'fallback'
    bom_detected: bool = False
    validated: bool = False
    chardet_raw: dict | None = None  # Raw chardet result for debugging


@dataclass
class DetectionResult:
    """Result of file format detection."""

    detected_type: FileType
    confidence: float
    method: str
    mime_type: str | None = None
    encoding: str | None = None
    magic_bytes: str | None = None
    format_mismatch: bool = False
    extension_type: FileType | None = None
    magika_label: str | None = None
    magika_score: float | None = None
    is_supported: bool = True
    unsupported_reason: str | None = None


class FileFormatDetector:
    """Enhanced file format detector with multi-layer strategy."""

    # Mapping from Magika labels to our FileType enum
    MAGIKA_TO_FILETYPE = {
        # Supported spreadsheet formats
        "csv": FileType.CSV,
        "tsv": FileType.TSV,
        "xlsx": FileType.XLSX,  # Note: Magika can't distinguish XLSM, they appear as xlsx
        "xls": FileType.XLS,
        "xlsb": FileType.XLSB,
        # Note: XLSM detection requires ZIP content analysis, not available via Magika
        # Text formats that might be delimited
        "txt": FileType.TXT,  # Will use content analysis for tabular detection
        # Sometimes Magika misidentifies UTF-16 text files
        "autohotkey": FileType.TXT,  # Often UTF-16 text files get misidentified as this
        # Unsupported but commonly encountered formats
        "pdf": FileType.UNKNOWN,
        "docx": FileType.UNKNOWN,
        "doc": FileType.UNKNOWN,
        "pptx": FileType.UNKNOWN,
        "ppt": FileType.UNKNOWN,
        "zip": None,  # Could be XLSX, needs ZIP analysis
        "json": FileType.UNKNOWN,
        "xml": FileType.UNKNOWN,
        "html": FileType.UNKNOWN,
        "rtf": FileType.UNKNOWN,
        # Programming/markup languages
        "python": FileType.UNKNOWN,
        "javascript": FileType.UNKNOWN,
        "css": FileType.UNKNOWN,
        "sql": FileType.UNKNOWN,
        "yaml": FileType.UNKNOWN,
        "markdown": FileType.UNKNOWN,
        # Media formats
        "png": FileType.UNKNOWN,
        "jpg": FileType.UNKNOWN,
        "gif": FileType.UNKNOWN,
        "mp4": FileType.UNKNOWN,
        "mp3": FileType.UNKNOWN,
        "wav": FileType.UNKNOWN,
        # Archive formats
        "tar": FileType.UNKNOWN,
        "gzip": FileType.UNKNOWN,
        "rar": FileType.UNKNOWN,
        "7zip": FileType.UNKNOWN,
        # Other common formats
        "binary": FileType.UNKNOWN,
        "unknown": FileType.UNKNOWN,
    }

    # Formats that are definitely unsupported for spreadsheet processing
    UNSUPPORTED_FORMATS = {
        "pdf": "PDF documents cannot be processed as spreadsheets",
        "docx": "Word documents are not spreadsheet files",
        "doc": "Word documents are not spreadsheet files",
        "pptx": "PowerPoint presentations are not spreadsheet files",
        "ppt": "PowerPoint presentations are not spreadsheet files",
        "png": "Image files cannot be processed as spreadsheets",
        "jpg": "Image files cannot be processed as spreadsheets",
        "gif": "Image files cannot be processed as spreadsheets",
        "mp4": "Video files cannot be processed as spreadsheets",
        "mp3": "Audio files cannot be processed as spreadsheets",
        "wav": "Audio files cannot be processed as spreadsheets",
        "zip": "Archive files cannot be processed as spreadsheets (unless they contain Excel files)",
        "tar": "Archive files cannot be processed as spreadsheets",
        "gzip": "Archive files cannot be processed as spreadsheets",
        "rar": "Archive files cannot be processed as spreadsheets",
        "7zip": "Archive files cannot be processed as spreadsheets",
        "python": "Source code files cannot be processed as spreadsheets",
        "javascript": "Source code files cannot be processed as spreadsheets",
        "css": "Stylesheet files cannot be processed as spreadsheets",
        "html": "HTML files cannot be processed as spreadsheets",
        "xml": "XML files cannot be processed as spreadsheets",
        "yaml": "YAML files cannot be processed as spreadsheets",
        "markdown": "Markdown files cannot be processed as spreadsheets",
        "sql": "SQL files cannot be processed as spreadsheets",
        "rtf": "Rich text files cannot be processed as spreadsheets",
        "binary": "Binary files cannot be processed as spreadsheets",
    }

    # Magic byte signatures for different formats
    MAGIC_SIGNATURES = {
        # Excel formats
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": FileType.XLS,  # OLE2 compound document
        b"PK\x03\x04": FileType.XLSX,  # ZIP format (potential XLSX/XLSM/XLSB)
        # Other formats that might be confused with Excel
        b"%PDF": FileType.UNKNOWN,  # PDF files
        b"\x89PNG": FileType.UNKNOWN,  # PNG images
    }

    # MIME type mappings
    MIME_TYPE_MAP = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.XLSX,
        "application/vnd.ms-excel.sheet.macroEnabled.12": FileType.XLSM,
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12": FileType.XLSB,
        "application/vnd.ms-excel": FileType.XLS,
        "text/csv": FileType.CSV,
        "text/tab-separated-values": FileType.TSV,
        "text/plain": None,  # Could be CSV/TSV, needs content analysis
        "application/zip": None,  # Could be XLSX, needs ZIP content inspection
    }

    # File extension mappings
    EXTENSION_MAP = {
        ".xlsx": FileType.XLSX,
        ".xls": FileType.XLS,
        ".xlsm": FileType.XLSM,
        ".xlsb": FileType.XLSB,
        ".csv": FileType.CSV,
        ".tsv": FileType.TSV,
        ".txt": FileType.TXT,  # Text files need content analysis
    }

    # Excel-specific ZIP content indicators
    EXCEL_ZIP_INDICATORS = [
        "xl/workbook.xml",
        "xl/sharedStrings.xml",
        "xl/styles.xml",
        "[Content_Types].xml",
        "_rels/.rels",
    ]

    def __init__(self, enable_magika: bool = True):
        """Initialize the detector with available libraries.

        Args:
            enable_magika: Whether to enable Magika detection (default: True)
        """
        self.magic_available = self._check_magic_availability()
        self.filetype_available = self._check_filetype_availability()
        self.magika_available = self._check_magika_availability() if enable_magika else False
        self.enable_magika = enable_magika

    def _check_magic_availability(self) -> bool:
        """Check if python-magic is available.

        Returns
        -------
        bool
            True if python-magic is installed and functional; False otherwise.

        Notes
        -----
        This method not only checks if the module can be imported but also
        verifies it can actually detect MIME types. Some installations may
        have import issues or missing dependencies (like libmagic).
        """
        try:
            import magic

            # Test if it actually works
            magic.Magic()
            return True
        except (ImportError, Exception) as e:
            logger.warning(f"python-magic not available: {e}")
            return False

    def _check_filetype_availability(self) -> bool:
        """Check if filetype library is available.

        Returns
        -------
        bool
            True if the filetype library is installed; False otherwise.

        Notes
        -----
        The filetype library provides pure-Python file type detection based on
        file signatures. It's used as a fallback when python-magic is unavailable
        or for additional validation.
        """
        try:
            import filetype  # noqa: F401

            return True
        except ImportError as e:
            logger.warning(f"filetype library not available: {e}")
            return False

    def _check_magika_availability(self) -> bool:
        """Check if Magika library is available.

        Returns
        -------
        bool
            True if Google's Magika AI-based file detection is available; False otherwise.

        Notes
        -----
        Magika is Google's AI-powered file type detection library that provides
        high accuracy for file format identification. It's optional but recommended
        for improved detection accuracy, especially for files with misleading
        extensions or corrupted headers.
        """
        try:
            from magika import Magika  # noqa: F401

            return True
        except ImportError as e:
            logger.warning(f"Magika library not available: {e}")
            return False

    def detect(self, file_path: Path, buffer_size: int = 8192) -> DetectionResult:
        """
        Detect file format using multi-layer strategy.

        Args:
            file_path: Path to the file to analyze
            buffer_size: Number of bytes to read for analysis

        Returns:
            DetectionResult with detected format and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file header for analysis
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(buffer_size)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return self._fallback_detection(file_path)

        magic_hex = header_bytes[:16].hex() if header_bytes else None

        # Get extension-based type for comparison
        extension_type = self._detect_by_extension(file_path)

        # Try different detection methods in order of reliability
        # Magika is first as it's AI-powered and most accurate
        methods = [
            ("magika", self._detect_by_magika),
            ("magic_mime", self._detect_by_magic_mime),
            ("magic_bytes", self._detect_by_magic_bytes),
            ("filetype", self._detect_by_filetype),
            ("content", self._detect_by_content),
            (
                "extension",
                lambda _fp, _hb: (
                    extension_type,
                    0.3 if extension_type != FileType.UNKNOWN else 0.1,
                ),
            ),
        ]

        best_result = None
        best_confidence = 0.0

        for method_name, method_func in methods:
            try:
                detected_type, confidence = method_func(file_path, header_bytes)

                if not detected_type or confidence <= best_confidence:
                    continue

                mime_type = self._get_mime_type(file_path) if method_name == "magic_mime" else None
                encoding = (
                    self._detect_encoding(header_bytes)
                    if detected_type in [FileType.CSV, FileType.TSV]
                    else None
                )

                # Get Magika-specific information
                magika_info = self._get_magika_info(file_path, method_name)

                best_result = DetectionResult(
                    detected_type=detected_type,
                    confidence=confidence,
                    method=method_name,
                    mime_type=mime_type,
                    encoding=encoding,
                    magic_bytes=magic_hex,
                    extension_type=extension_type,
                    format_mismatch=(detected_type != extension_type if extension_type else False),
                    magika_label=magika_info["label"],
                    magika_score=magika_info["score"],
                    is_supported=magika_info["is_supported"],
                    unsupported_reason=magika_info["unsupported_reason"],
                )
                best_confidence = confidence

                # If we have high confidence, stop here
                if confidence >= 0.9:
                    break

            except Exception as e:
                logger.debug(f"Detection method {method_name} failed: {e}")
                continue

        if best_result:
            return best_result
        else:
            return self._fallback_detection(file_path, magic_hex)

    def _detect_by_magic_mime(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Detect using python-magic MIME type detection."""
        if not self.magic_available:
            return None, 0.0

        try:
            import magic

            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))

            if mime_type in self.MIME_TYPE_MAP:
                detected_type = self.MIME_TYPE_MAP[mime_type]
                if detected_type:
                    return detected_type, 0.95
                else:
                    # Need additional analysis for ZIP/text files
                    if mime_type == "application/zip":
                        return self._analyze_zip_content(file_path)
                    elif mime_type == "text/plain":
                        text_result = self._analyze_text_content(file_path, header_bytes)
                        if text_result[0]:  # If text analysis found a type
                            return text_result
                        # If text analysis failed, fall through to continue detection

            return None, 0.0

        except Exception as e:
            logger.debug(f"Magic MIME detection failed: {e}")
            return None, 0.0

    def _detect_by_magic_bytes(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Detect using magic byte signatures."""
        if not header_bytes:
            return None, 0.0

        for signature, file_type in self.MAGIC_SIGNATURES.items():
            if header_bytes.startswith(signature):
                if signature == b"PK\x03\x04":  # ZIP format, need more analysis
                    return self._analyze_zip_content(file_path)
                else:
                    return file_type, 0.9

        return None, 0.0

    def _detect_by_filetype(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Detect using filetype library."""
        if not self.filetype_available:
            return None, 0.0

        try:
            import filetype

            # Try from bytes first (more efficient)
            if header_bytes:
                kind = filetype.guess(header_bytes)
                if kind:
                    mime_type = kind.mime
                    if mime_type in self.MIME_TYPE_MAP:
                        detected_type = self.MIME_TYPE_MAP[mime_type]
                        if detected_type:
                            return detected_type, 0.8
                        elif mime_type == "application/zip":
                            return self._analyze_zip_content(file_path)

            return None, 0.0

        except Exception as e:
            logger.debug(f"Filetype detection failed: {e}")
            return None, 0.0

    def _detect_by_content(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Detect by analyzing file content structure."""
        # Check if it's likely a text file (for CSV/TSV detection)
        if self._is_likely_text(header_bytes):
            return self._analyze_text_content(file_path, header_bytes)

        return None, 0.0

    def _detect_by_extension(self, file_path: Path) -> FileType:
        """Detect file type based on extension."""
        extension = file_path.suffix.lower()
        return self.EXTENSION_MAP.get(extension, FileType.UNKNOWN)

    def _analyze_zip_content(self, file_path: Path) -> tuple[FileType | None, float]:
        """Analyze ZIP file content to determine if it's Excel format."""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_file:
                file_list = zip_file.namelist()

                # Check for Excel-specific files
                excel_indicators = sum(
                    1 for indicator in self.EXCEL_ZIP_INDICATORS if indicator in file_list
                )

                if excel_indicators >= 3:  # Need at least 3 indicators for confidence
                    # Determine specific Excel format
                    if any("vbaProject" in f for f in file_list):
                        return FileType.XLSM, 0.95  # Macro-enabled
                    elif self._is_xlsb_format(file_list):
                        return FileType.XLSB, 0.95  # Binary format
                    else:
                        return FileType.XLSX, 0.95  # Standard Excel

                elif excel_indicators >= 1:
                    return FileType.XLSX, 0.7  # Likely Excel but not certain

        except (zipfile.BadZipFile, Exception) as e:
            logger.debug(f"ZIP analysis failed: {e}")

        return None, 0.0

    def _is_xlsb_format(self, file_list: list[str]) -> bool:
        """More accurate XLSB detection based on file structure.

        XLSB files have specific characteristics:
        - xl/workbook.bin (not workbook.xml)
        - xl/worksheets/sheet*.bin (not sheet*.xml)
        - No xml worksheet files
        """
        # Check for binary workbook file
        has_workbook_bin = any("xl/workbook.bin" in f for f in file_list)

        # Check for binary worksheet files
        has_sheet_bin = any(
            f.startswith("xl/worksheets/sheet") and f.endswith(".bin") for f in file_list
        )

        # Check for absence of XML worksheet files (XLSB won't have these)
        has_sheet_xml = any(
            f.startswith("xl/worksheets/sheet") and f.endswith(".xml") for f in file_list
        )

        # XLSB should have binary workbook AND sheets, but NO XML sheets
        return has_workbook_bin and has_sheet_bin and not has_sheet_xml

    def _analyze_text_content(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Analyze text content to determine if it's CSV, TSV, or TXT with sophisticated encoding detection."""
        try:
            # Use sophisticated encoding detection
            encoding_result = self._detect_encoding_sophisticated(header_bytes, file_path)
            logger.debug(
                f"Text analysis using encoding: {encoding_result.encoding} (method: {encoding_result.method}, confidence: {encoding_result.confidence})"
            )

            with open(file_path, encoding=encoding_result.encoding, errors="replace") as f:
                # Read first few lines for analysis (handle very long lines)
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 15:  # Analyze more lines for better detection
                        break

            if not lines:
                return None, 0.0

            # Enhanced delimiter analysis for scientific/instrument data
            delimiter_scores: dict[str, float] = {}
            delimiters = [
                "\t",
                ",",
                ";",
                "|",
                " ",
            ]  # Tab first (common in scientific data)

            for delimiter in delimiters:
                column_counts = []
                valid_lines = 0

                for line in lines:
                    if line and len(line.strip()) > 0:  # Skip empty lines
                        if delimiter == " ":
                            # For space delimiter, split on multiple spaces to handle formatting
                            parts = [p for p in line.split() if p]
                            count = len(parts) - 1 if len(parts) > 1 else 0
                        else:
                            count = line.count(delimiter)

                        if count > 0:
                            column_counts.append(count)
                            valid_lines += 1

                if column_counts and valid_lines >= 2:
                    # Enhanced scoring that considers consistency and data quality
                    unique_counts = list(set(column_counts))

                    if len(unique_counts) == 1:  # Perfect consistency
                        base_score = unique_counts[0] * valid_lines
                        delimiter_scores[delimiter] = base_score
                    elif len(unique_counts) <= 3:  # Good consistency (allow some variation)
                        avg_count = sum(column_counts) / len(column_counts)
                        consistency_penalty = len(unique_counts) * 0.1
                        delimiter_scores[delimiter] = (avg_count * valid_lines) * (
                            1 - consistency_penalty
                        )

            # Determine best delimiter and file type
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=lambda x: delimiter_scores[x])
                score = delimiter_scores[best_delimiter]

                logger.debug(f"Best delimiter '{best_delimiter}' with score {score}")

                # Lower threshold for scientific data (often has mixed formatting)
                if score >= 2:
                    if best_delimiter == "\t":
                        confidence = min(0.95, score / 15)
                        return FileType.TSV, confidence
                    elif best_delimiter == ",":
                        confidence = min(0.9, score / 20)
                        return FileType.CSV, confidence
                    elif best_delimiter in [";", "|"]:
                        confidence = min(0.85, score / 20)
                        return FileType.CSV, confidence  # Treat as CSV variant
                    elif best_delimiter == " ":
                        # Space-delimited is often TSV-like
                        confidence = min(0.8, score / 25)
                        return FileType.TSV, confidence

            # If no clear delimited pattern but looks like readable text, return as TXT
            # This allows TextReader to handle it with its own delimiter detection
            total_text_lines = len([line for line in lines if line.strip()])
            if total_text_lines >= 2:
                # Check if it contains reasonable text content
                text_chars = sum(len(line) for line in lines if line.strip())
                if text_chars > 50:  # Minimum content threshold
                    logger.debug(
                        f"Detected as text file with {total_text_lines} lines and {text_chars} characters"
                    )
                    return FileType.TXT, 0.7

        except Exception as e:
            logger.debug(f"Text content analysis failed: {e}")

        return None, 0.0

    def _detect_by_magika(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Detect using Google's Magika AI-powered detection."""
        if not self.magika_available:
            return None, 0.0

        try:
            from magika import Magika

            # Initialize Magika (it loads its model)
            magika = Magika()

            # Detect file type
            result = magika.identify_path(file_path)

            if not result:
                return None, 0.0

            # Extract label and confidence
            magika_label = result.output.label  # Use .label instead of deprecated .ct_label
            confidence_score = result.score  # This is the confidence from Magika

            # Map Magika label to our FileType
            if magika_label in self.MAGIKA_TO_FILETYPE:
                detected_type = self.MAGIKA_TO_FILETYPE[magika_label]

                if detected_type is None:
                    # Special handling for formats that need additional analysis
                    if magika_label == "txt":
                        # Text file - try content analysis
                        text_result = self._analyze_text_content(file_path, header_bytes)
                        if text_result[0]:
                            return text_result[0], min(confidence_score, text_result[1])
                    elif magika_label == "zip":
                        # ZIP file - check if it's Excel
                        zip_result = self._analyze_zip_content(file_path)
                        if zip_result[0]:
                            return zip_result[0], min(confidence_score, zip_result[1])

                    return None, 0.0

                elif detected_type == FileType.XLSX:
                    # Magika detected XLSX, but it might be XLSM or XLSB
                    # Do additional ZIP analysis to distinguish
                    zip_result = self._analyze_zip_content(file_path)
                    if zip_result[0] and zip_result[0] in [
                        FileType.XLSM,
                        FileType.XLSB,
                    ]:
                        # More specific Excel format detected
                        return zip_result[0], min(confidence_score, zip_result[1])
                    else:
                        # Standard XLSX
                        return detected_type, confidence_score

                elif detected_type == FileType.UNKNOWN:
                    # Format is recognized but unsupported
                    if magika_label in self.UNSUPPORTED_FORMATS:
                        # This will be handled by the caller to raise UnsupportedFormatError
                        pass
                    return detected_type, confidence_score

                else:
                    # Other supported formats
                    return detected_type, confidence_score

            # Unknown format from Magika - check if it's in unsupported list
            if magika_label in self.UNSUPPORTED_FORMATS:
                # Return UNKNOWN but the caller can check the unsupported reason
                return FileType.UNKNOWN, confidence_score

            # Truly unknown format from Magika
            return FileType.UNKNOWN, confidence_score if confidence_score > 0.5 else 0.1

        except Exception as e:
            logger.debug(f"Magika detection failed: {e}")
            return None, 0.0

    def _get_magika_info(self, file_path: Path, method_name: str) -> dict[str, Any]:
        """Get Magika-specific information if applicable."""
        default_info = {
            "label": None,
            "score": None,
            "is_supported": True,
            "unsupported_reason": None,
        }

        if method_name != "magika":
            return default_info

        try:
            from magika import Magika

            magika = Magika()
            result = magika.identify_path(file_path)
            if not result:
                return default_info

            magika_label = result.output.label  # Use .label instead of deprecated .ct_label
            magika_score = result.score

            # Check if format is supported
            is_supported = magika_label not in self.UNSUPPORTED_FORMATS
            unsupported_reason = (
                self.UNSUPPORTED_FORMATS.get(magika_label) if not is_supported else None
            )

            return {
                "label": magika_label,
                "score": magika_score,
                "is_supported": is_supported,
                "unsupported_reason": unsupported_reason,
            }
        except Exception:
            return default_info  # Continue without Magika info if extraction fails

    def _is_likely_text(self, header_bytes: bytes) -> bool:
        """Check if the file appears to be text-based."""
        if not header_bytes:
            return False

        # Count non-printable characters
        printable_count = sum(
            1 for byte in header_bytes if 32 <= byte <= 126 or byte in [9, 10, 13]
        )  # Include tab, LF, CR

        if len(header_bytes) == 0:
            return False

        text_ratio = printable_count / len(header_bytes)

        # If more than 80% printable characters, likely text
        return text_ratio > 0.8

    def _detect_encoding_sophisticated(
        self, header_bytes: bytes, file_path: Path | None = None
    ) -> EncodingResult:
        """Sophisticated multi-layer encoding detection."""

        # Phase 1: BOM (Byte Order Mark) Detection
        bom_result = self._detect_bom(header_bytes)
        if bom_result.confidence > 0.95:
            logger.debug(f"BOM detected: {bom_result.encoding}")
            return bom_result

        # Phase 2: Chardet Analysis with Smart Confidence Thresholds
        chardet_result = self._detect_with_chardet(header_bytes, file_path)
        if chardet_result and chardet_result.confidence > 0.8:
            logger.debug(
                f"Chardet detection: {chardet_result.encoding} (confidence: {chardet_result.confidence})"
            )

            # Validate the encoding by attempting to decode
            validated_result = self._validate_encoding(header_bytes, chardet_result, file_path)
            if validated_result.validated:
                return validated_result

        # Phase 3: Pattern-based detection for common cases
        pattern_result = self._detect_by_patterns(header_bytes)
        if pattern_result.confidence > 0.7:
            return pattern_result

        # Phase 4: Fallback chain with validation
        fallback_result = self._fallback_encoding_detection(header_bytes)
        return fallback_result

    def _detect_bom(self, header_bytes: bytes) -> EncodingResult:
        """Detect encoding from Byte Order Mark."""
        if len(header_bytes) < 2:
            return EncodingResult("utf-8", 0.0, "bom")

        # Check for BOMs in order of specificity
        if header_bytes.startswith(b"\xff\xfe\x00\x00"):
            return EncodingResult("utf-32-le", 1.0, "bom", bom_detected=True)
        elif header_bytes.startswith(b"\x00\x00\xfe\xff"):
            return EncodingResult("utf-32-be", 1.0, "bom", bom_detected=True)
        elif header_bytes.startswith(b"\xff\xfe"):
            return EncodingResult("utf-16-le", 1.0, "bom", bom_detected=True)
        elif header_bytes.startswith(b"\xfe\xff"):
            return EncodingResult("utf-16-be", 1.0, "bom", bom_detected=True)
        elif header_bytes.startswith(b"\xef\xbb\xbf"):
            return EncodingResult("utf-8", 1.0, "bom", bom_detected=True)

        return EncodingResult("utf-8", 0.0, "bom")

    def _detect_with_chardet(
        self, header_bytes: bytes, file_path: Path | None = None
    ) -> EncodingResult | None:
        """Use chardet with multiple buffer sizes for better accuracy."""
        try:
            import chardet
        except ImportError:
            return None

        # Try different buffer sizes for better detection
        buffer_sizes = [len(header_bytes)]

        # If we have a file path, try larger buffers
        if file_path and file_path.exists():
            try:
                file_size = file_path.stat().st_size
                if file_size > len(header_bytes):
                    # Add larger buffer sizes up to 64KB
                    buffer_sizes.extend(
                        [
                            min(4096, file_size),
                            min(16384, file_size),
                            min(65536, file_size),
                        ]
                    )
            except Exception:
                pass

        best_result = None
        best_confidence = 0.0

        for buffer_size in buffer_sizes:
            try:
                if buffer_size > len(header_bytes) and file_path:
                    with open(file_path, "rb") as f:
                        data = f.read(buffer_size)
                else:
                    data = header_bytes[:buffer_size]

                result = chardet.detect(data)
                encoding = result.get("encoding")
                confidence = result.get("confidence", 0.0)

                if encoding and confidence > best_confidence:
                    # Apply encoding-specific confidence thresholds
                    min_confidence = self._get_min_confidence_for_encoding(encoding)
                    if confidence >= min_confidence:
                        best_result = EncodingResult(
                            encoding=encoding,
                            confidence=confidence,
                            method="chardet",
                            chardet_raw=dict(result),  # Convert to dict
                        )
                        best_confidence = confidence

            except Exception as e:
                logger.debug(f"Chardet failed with buffer size {buffer_size}: {e}")
                continue

        return best_result

    def _get_min_confidence_for_encoding(self, encoding: str) -> float:
        """Get minimum confidence threshold based on encoding type."""
        encoding_lower = encoding.lower()

        if "utf-16" in encoding_lower or "utf-32" in encoding_lower:
            return 0.8  # UTF-16/32 detection is usually very reliable
        elif "utf-8" in encoding_lower:
            return 0.9  # UTF-8 is common, so be more strict
        elif any(enc in encoding_lower for enc in ["latin", "cp125", "iso-8859"]):
            return 0.85  # Medium confidence for these encodings
        else:
            return 0.9  # Be conservative for unknown encodings

    def _validate_encoding(
        self,
        header_bytes: bytes,
        encoding_result: EncodingResult,
        file_path: Path | None = None,  # noqa: ARG002
    ) -> EncodingResult:
        """Validate encoding by attempting to decode and analyze content."""
        try:
            # Try to decode the header bytes
            decoded_text = header_bytes.decode(encoding_result.encoding, errors="strict")

            # Check for reasonable text characteristics
            printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c.isspace()) / len(
                decoded_text
            )

            if printable_ratio > 0.8:  # At least 80% printable characters
                encoding_result.validated = True
                encoding_result.confidence = min(
                    1.0, encoding_result.confidence + 0.1
                )  # Boost confidence
                logger.debug(
                    f"Encoding {encoding_result.encoding} validated (printable ratio: {printable_ratio:.2f})"
                )

            return encoding_result

        except UnicodeDecodeError as e:
            logger.debug(f"Encoding {encoding_result.encoding} validation failed: {e}")
            encoding_result.confidence *= 0.5  # Reduce confidence
            return encoding_result

    def _detect_by_patterns(self, header_bytes: bytes) -> EncodingResult:
        """Detect encoding based on common byte patterns."""

        # Look for common UTF-16 patterns (alternating null bytes)
        if len(header_bytes) > 10:
            null_pattern_score = 0
            for i in range(1, min(100, len(header_bytes)), 2):
                if header_bytes[i] == 0:  # Every other byte is null
                    null_pattern_score += 1

            if null_pattern_score > 10:  # Strong pattern suggests UTF-16 LE
                return EncodingResult("utf-16-le", 0.8, "pattern")

        # Look for high-bit characters that suggest specific encodings
        high_bit_count = sum(1 for b in header_bytes if b > 127)
        if high_bit_count > len(header_bytes) * 0.1:  # More than 10% high-bit chars
            # Could be Latin-1 or similar
            return EncodingResult("latin-1", 0.6, "pattern")

        # All ASCII suggests UTF-8
        if all(b < 128 for b in header_bytes):
            return EncodingResult("utf-8", 0.7, "pattern")

        return EncodingResult("utf-8", 0.0, "pattern")

    def _fallback_encoding_detection(self, header_bytes: bytes) -> EncodingResult:
        """Final fallback encoding detection with validation."""

        # Comprehensive list of encodings to try
        encodings = [
            "utf-8",
            "utf-16-le",
            "utf-16-be",
            "utf-16",
            "latin-1",
            "cp1252",
            "iso-8859-1",
            "ascii",
            "cp437",
            "cp850",
            "utf-32-le",
            "utf-32-be",
        ]

        for encoding in encodings:
            try:
                decoded = header_bytes.decode(encoding, errors="strict")

                # Simple validation: check for reasonable text
                if len(decoded) > 0:
                    printable_ratio = sum(
                        1 for c in decoded if c.isprintable() or c.isspace()
                    ) / len(decoded)
                    if printable_ratio > 0.7:
                        confidence = 0.6 if encoding == "utf-8" else 0.5
                        return EncodingResult(
                            encoding=encoding,
                            confidence=confidence,
                            method="fallback",
                            validated=True,
                        )

            except (UnicodeDecodeError, LookupError):
                continue

        # Ultimate fallback
        return EncodingResult("utf-8", 0.1, "fallback")

    def _detect_encoding(self, header_bytes: bytes) -> str:
        """Legacy method for backward compatibility."""
        result = self._detect_encoding_sophisticated(header_bytes)
        return result.encoding

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Get MIME type using python-magic."""
        if not self.magic_available:
            return None

        try:
            import magic

            mime = magic.Magic(mime=True)
            return str(mime.from_file(str(file_path)))
        except Exception:
            return None

    def _fallback_detection(self, file_path: Path, magic_hex: str | None = None) -> DetectionResult:
        """Fallback detection using only file extension."""
        extension_type = self._detect_by_extension(file_path)

        return DetectionResult(
            detected_type=extension_type,
            confidence=0.3 if extension_type != FileType.UNKNOWN else 0.1,
            method="extension_fallback",
            magic_bytes=magic_hex,
            extension_type=extension_type,
            format_mismatch=False,  # Can't detect mismatch without content analysis
        )


# Global detector instances cache
_detectors = {}


def get_detector(enable_magika: bool = True) -> FileFormatDetector:
    """Get or create a file format detector with specified configuration.

    Args:
        enable_magika: Whether to enable Magika detection

    Returns:
        FileFormatDetector instance
    """
    global _detectors
    key = f"magika_{enable_magika}"
    if key not in _detectors:
        _detectors[key] = FileFormatDetector(enable_magika=enable_magika)
    return _detectors[key]


def detect_file_type(
    file_path: Path, buffer_size: int = 8192, enable_magika: bool = True
) -> FileType:
    """
    Detect file type using enhanced multi-layer detection.

    Args:
        file_path: Path to the file
        buffer_size: Number of bytes to read for analysis
        enable_magika: Whether to enable Magika AI detection

    Returns:
        Detected FileType
    """
    detector = get_detector(enable_magika=enable_magika)
    result = detector.detect(file_path, buffer_size)
    return result.detected_type


def detect_file_info(
    file_path: Path, buffer_size: int = 8192, enable_magika: bool = True
) -> DetectionResult:
    """
    Detect file format with detailed information.

    Args:
        file_path: Path to the file
        buffer_size: Number of bytes to read for analysis
        enable_magika: Whether to enable Magika AI detection

    Returns:
        DetectionResult with comprehensive detection information
    """
    detector = get_detector(enable_magika=enable_magika)
    return detector.detect(file_path, buffer_size)


def detect_file_info_safe(
    file_path: Path, buffer_size: int = 8192, enable_magika: bool = True
) -> DetectionResult:
    """
    Detect file format and raise UnsupportedFormatError if format is unsupported.

    Args:
        file_path: Path to the file
        buffer_size: Number of bytes to read for analysis
        enable_magika: Whether to enable Magika AI detection

    Returns:
        DetectionResult with comprehensive detection information

    Raises:
        UnsupportedFormatError: If the detected format is not supported for spreadsheet processing
    """
    result = detect_file_info(file_path, buffer_size, enable_magika=enable_magika)

    if not result.is_supported and result.unsupported_reason:
        raise UnsupportedFormatError(
            detected_format=result.magika_label or result.detected_type.value,
            file_path=file_path,
            reason=result.unsupported_reason,
        )

    return result


def detect_file_info_with_config(file_path: Path, config: Any = None) -> DetectionResult:
    """
    Detect file format using configuration settings.

    Args:
        file_path: Path to the file
        config: GridGulp configuration object (if None, uses defaults)

    Returns:
        DetectionResult with comprehensive detection information

    Raises:
        UnsupportedFormatError: If strict_format_checking is True and format is unsupported
    """
    if config is None:
        # Use defaults if no config provided
        buffer_size = 8192
        enable_magika = True
        strict_checking = False
    else:
        buffer_size = config.file_detection_buffer_size
        enable_magika = config.enable_magika
        strict_checking = config.strict_format_checking

    # Use the appropriate detection function based on strict checking
    if strict_checking:
        return detect_file_info_safe(file_path, buffer_size, enable_magika)
    else:
        return detect_file_info(file_path, buffer_size, enable_magika)
