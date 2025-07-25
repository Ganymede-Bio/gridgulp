"""Enhanced file type detection using magic bytes, MIME types, and content analysis."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..models.file_info import FileType, UnsupportedFormatError

logger = logging.getLogger(__name__)


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
        "xlsx": FileType.XLSX,  # Note: Magika can't distinguish XLSM/XLSB, they appear as xlsx
        "xls": FileType.XLS,
        # Note: XLSM and XLSB detection requires ZIP content analysis, not available via Magika
        # Text formats that might be delimited
        "txt": None,  # Needs content analysis
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
        "json": "JSON files cannot be processed as spreadsheets",
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
        ".txt": FileType.TSV,  # Assume tab-separated for .txt
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
        """Check if python-magic is available."""
        try:
            import magic

            # Test if it actually works
            magic.Magic()
            return True
        except (ImportError, Exception) as e:
            logger.warning(f"python-magic not available: {e}")
            return False

    def _check_filetype_availability(self) -> bool:
        """Check if filetype library is available."""
        try:
            import filetype  # noqa: F401

            return True
        except ImportError as e:
            logger.warning(f"filetype library not available: {e}")
            return False

    def _check_magika_availability(self) -> bool:
        """Check if Magika library is available."""
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

                if detected_type and confidence > best_confidence:
                    mime_type = (
                        self._get_mime_type(file_path) if method_name == "magic_mime" else None
                    )
                    encoding = (
                        self._detect_encoding(header_bytes)
                        if detected_type in [FileType.CSV, FileType.TSV]
                        else None
                    )

                    # Check for Magika-specific information
                    magika_label = None
                    magika_score = None
                    is_supported = True
                    unsupported_reason = None

                    if method_name == "magika":
                        # Extract Magika information if this was a Magika detection
                        try:
                            from magika import Magika

                            magika = Magika()
                            result = magika.identify_path(file_path)
                            if result:
                                magika_label = (
                                    result.output.label
                                )  # Use .label instead of deprecated .ct_label
                                magika_score = result.score

                                # Check if format is supported
                                if magika_label in self.UNSUPPORTED_FORMATS:
                                    is_supported = False
                                    unsupported_reason = self.UNSUPPORTED_FORMATS[magika_label]
                        except Exception:
                            pass  # Continue without Magika info if extraction fails

                    best_result = DetectionResult(
                        detected_type=detected_type,
                        confidence=confidence,
                        method=method_name,
                        mime_type=mime_type,
                        encoding=encoding,
                        magic_bytes=magic_hex,
                        extension_type=extension_type,
                        format_mismatch=detected_type != extension_type
                        if extension_type
                        else False,
                        magika_label=magika_label,
                        magika_score=magika_score,
                        is_supported=is_supported,
                        unsupported_reason=unsupported_reason,
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
                    elif any(f.endswith(".bin") for f in file_list):
                        return FileType.XLSB, 0.95  # Binary format
                    else:
                        return FileType.XLSX, 0.95  # Standard Excel

                elif excel_indicators >= 1:
                    return FileType.XLSX, 0.7  # Likely Excel but not certain

        except (zipfile.BadZipFile, Exception) as e:
            logger.debug(f"ZIP analysis failed: {e}")

        return None, 0.0

    def _analyze_text_content(
        self, file_path: Path, header_bytes: bytes
    ) -> tuple[FileType | None, float]:
        """Analyze text content to determine if it's CSV or TSV."""
        try:
            # Try to decode the header
            encoding = self._detect_encoding(header_bytes)

            with open(file_path, encoding=encoding, errors="ignore") as f:
                # Read first few lines for analysis
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 10:  # Analyze first 10 lines
                        break

            if not lines:
                return None, 0.0

            # Analyze delimiter patterns
            delimiter_scores = {}
            delimiters = [",", "\t", ";", "|"]

            for delimiter in delimiters:
                consistent_counts = []
                for line in lines:
                    if line:  # Skip empty lines
                        count = line.count(delimiter)
                        if count > 0:
                            consistent_counts.append(count)

                if consistent_counts:
                    # Check for consistency
                    if len(set(consistent_counts)) == 1:  # All counts are the same
                        delimiter_scores[delimiter] = consistent_counts[0] * len(consistent_counts)
                    elif len(set(consistent_counts)) <= 2:  # Mostly consistent
                        delimiter_scores[delimiter] = sum(consistent_counts) * 0.8

            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                score = delimiter_scores[best_delimiter]

                # Minimum threshold for considering it a delimited file
                if score >= 3:
                    if best_delimiter == ",":
                        return FileType.CSV, min(0.9, score / 20)
                    elif best_delimiter == "\t":
                        return FileType.TSV, min(0.9, score / 20)
                    else:
                        # Other delimiters could be CSV variants
                        return FileType.CSV, min(0.8, score / 20)

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

    def _is_likely_text(self, header_bytes: bytes) -> bool:
        """Check if the file appears to be text-based."""
        if not header_bytes:
            return False

        # Check for binary indicators
        # binary_indicators = [
        #     b'\x00',  # Null bytes
        #     b'\xff\xfe',  # UTF-16 BOM
        #     b'\xfe\xff',  # UTF-16 BE BOM
        #     b'\xef\xbb\xbf',  # UTF-8 BOM (but still text)
        # ]

        # Count non-printable characters
        printable_count = sum(
            1 for byte in header_bytes if 32 <= byte <= 126 or byte in [9, 10, 13]
        )  # Include tab, LF, CR

        if len(header_bytes) == 0:
            return False

        text_ratio = printable_count / len(header_bytes)

        # If more than 80% printable characters, likely text
        return text_ratio > 0.8

    def _detect_encoding(self, header_bytes: bytes) -> str:
        """Detect text encoding from header bytes."""
        try:
            import chardet

            result = chardet.detect(header_bytes)
            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0.0)

            if confidence > 0.7:
                return encoding
        except ImportError:
            pass

        # Fallback encoding detection
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                header_bytes.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        return "utf-8"  # Final fallback

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Get MIME type using python-magic."""
        if not self.magic_available:
            return None

        try:
            import magic

            mime = magic.Magic(mime=True)
            return mime.from_file(str(file_path))
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


def detect_file_info_with_config(file_path: Path, config=None) -> DetectionResult:
    """
    Detect file format using configuration settings.

    Args:
        file_path: Path to the file
        config: GridPorter configuration object (if None, uses defaults)

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
