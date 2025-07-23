"""File type detection using magic bytes."""

import logging
from pathlib import Path

from ..models.file_info import FileType

logger = logging.getLogger(__name__)


def detect_file_type(file_path: Path) -> FileType:
    """Detect file type using magic bytes and extension.

    Args:
        file_path: Path to the file

    Returns:
        Detected FileType

    Note:
        This is a simplified implementation. In the full version,
        this would use python-magic for robust detection.
    """
    try:
        # Try to import magic for proper detection
        import magic

        mime_type = magic.from_file(str(file_path), mime=True)
        logger.debug(f"Detected MIME type: {mime_type} for {file_path}")

        # Map MIME types to our FileType enum
        mime_to_type = {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.XLSX,
            "application/vnd.ms-excel": FileType.XLS,
            "application/vnd.ms-excel.sheet.macroEnabled.12": FileType.XLSM,
            "application/vnd.ms-excel.sheet.binary.macroEnabled.12": FileType.XLSB,
            "text/csv": FileType.CSV,
            "text/tab-separated-values": FileType.TSV,
        }

        if mime_type in mime_to_type:
            return mime_to_type[mime_type]

    except ImportError:
        logger.warning(
            "python-magic not available, falling back to extension detection"
        )
    except Exception as e:
        logger.warning(f"Magic detection failed: {e}, falling back to extension")

    # Fallback to extension-based detection
    extension = file_path.suffix.lower()
    extension_to_type = {
        ".xlsx": FileType.XLSX,
        ".xls": FileType.XLS,
        ".xlsm": FileType.XLSM,
        ".xlsb": FileType.XLSB,
        ".csv": FileType.CSV,
        ".tsv": FileType.TSV,
        ".txt": FileType.TSV,  # Assume tab-separated for .txt
    }

    return extension_to_type.get(extension, FileType.UNKNOWN)
