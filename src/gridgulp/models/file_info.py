"""File information models."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class UnsupportedFormatError(Exception):
    """Raised when file format is detected but not supported for spreadsheet processing."""

    def __init__(self, detected_format: str, file_path: Path, reason: str | None = None):
        self.detected_format = detected_format
        self.file_path = file_path
        self.reason = (
            reason or f"Format '{detected_format}' is not supported for spreadsheet processing"
        )
        super().__init__(self.reason)


class FileType(str, Enum):
    """Supported file types."""

    XLSX = "xlsx"
    XLS = "xls"
    XLSM = "xlsm"
    XLSB = "xlsb"  # Detected but not supported
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"
    UNKNOWN = "unknown"


class FileInfo(BaseModel):
    """Information about a file."""

    model_config = ConfigDict(strict=True)

    path: Path = Field(..., description="File path")
    type: FileType = Field(..., description="Detected file type based on content analysis")
    size: int = Field(..., ge=0, description="File size in bytes")

    # Detection details
    detected_mime: str | None = Field(None, description="MIME type detected by python-magic")
    extension_format: FileType | None = Field(
        None, description="File type based on extension alone"
    )
    detection_confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Confidence score for format detection"
    )
    format_mismatch: bool = Field(
        False, description="True if extension doesn't match detected content"
    )
    detection_method: str = Field(
        "extension",
        description="Method used for detection (magic, content, extension, hybrid)",
    )

    # Content details
    encoding: str | None = Field(None, description="File encoding (for text files)")
    magic_bytes: str | None = Field(None, description="First few bytes of file (hex)")

    # Magika AI detection details
    magika_label: str | None = Field(None, description="Raw Magika detection label")
    magika_score: float | None = Field(None, description="Magika confidence score (0.0-1.0)")

    # Format support
    is_supported: bool = Field(True, description="Whether this format is supported by GridGulp")
    unsupported_reason: str | None = Field(None, description="Reason why format is unsupported")

    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.size / (1024 * 1024)
