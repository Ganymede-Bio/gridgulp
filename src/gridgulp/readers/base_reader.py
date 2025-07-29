"""Abstract base reader for file parsing."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..models.file_info import FileInfo
from ..models.sheet_data import FileData, SheetData

logger = logging.getLogger(__name__)


class ReaderError(Exception):
    """Base exception for reader errors."""

    pass


class UnsupportedFileError(ReaderError):
    """Raised when file format is not supported."""

    pass


class CorruptedFileError(ReaderError):
    """Raised when file is corrupted or unreadable."""

    pass


class PasswordProtectedError(ReaderError):
    """Raised when file is password protected."""

    pass


class BaseReader(ABC):
    """Abstract base class for all file readers."""

    def __init__(self, file_path: Path, file_info: FileInfo):
        """Initialize reader with file information.

        Args:
            file_path: Path to the file to read
            file_info: File information from detection
        """
        self.file_path = file_path
        self.file_info = file_info
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def read(self) -> FileData:
        """Read the file and return structured data.

        Returns:
            FileData containing all sheets and metadata

        Raises:
            ReaderError: If file cannot be read
        """
        pass

    @abstractmethod
    def can_read(self) -> bool:
        """Check if this reader can handle the file.

        Returns:
            True if reader can handle the file
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns:
            List of supported format extensions
        """
        pass

    def validate_file(self) -> None:
        """Validate that file exists and is readable.

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        try:
            with open(self.file_path, "rb") as f:
                # Try to read first few bytes
                f.read(1024)
        except PermissionError as pe:
            raise PermissionError(f"Cannot read file: {self.file_path}") from pe
        except Exception as e:
            raise ReaderError(f"File validation failed: {e}") from e

    def _detect_encoding(self, sample_data: bytes) -> str:
        """Detect text encoding from sample data.

        Args:
            sample_data: Sample bytes from file

        Returns:
            Detected encoding name
        """
        try:
            import chardet

            result = chardet.detect(sample_data)
            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0.0)

            self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

            # Use utf-8 if confidence is very low
            if confidence < 0.3:
                self.logger.warning("Low confidence encoding detection, using utf-8")
                return "utf-8"

            return encoding or "utf-8"

        except ImportError:
            self.logger.warning("chardet not available, using utf-8 encoding")
            return "utf-8"
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return "utf-8"

    def _create_empty_sheet(self, name: str) -> SheetData:
        """Create an empty sheet with given name.

        Args:
            name: Sheet name

        Returns:
            Empty SheetData object
        """
        return SheetData(name=name, cells={}, max_row=0, max_column=0)

    def _log_read_start(self) -> None:
        """Log the start of file reading."""
        self.logger.info(f"Starting to read {self.file_info.type.value} file: {self.file_path}")
        self.logger.debug(f"File size: {self.file_info.size_mb:.2f} MB")

    def _log_read_complete(self, data: FileData) -> None:
        """Log completion of file reading.

        Args:
            data: The read file data
        """
        self.logger.info(f"Successfully read file with {data.sheet_count} sheets")
        for sheet in data.sheets:
            rows, cols = sheet.get_dimensions()
            non_empty = len(sheet.get_non_empty_cells())
            self.logger.debug(f"Sheet '{sheet.name}': {rows}x{cols}, {non_empty} non-empty cells")


class AsyncBaseReader(BaseReader):
    """Base class for readers that support async operations."""

    @abstractmethod
    async def read_async(self) -> FileData:
        """Async version of read method.

        Returns:
            FileData containing all sheets and metadata
        """
        pass

    async def read(self) -> FileData:
        """Default implementation delegates to async version."""
        return await self.read_async()

    async def read_all(self) -> list[SheetData]:
        """Read all sheets from the file asynchronously.

        This is a convenience method that returns sheets directly
        instead of wrapped in FileData.

        Returns:
            List of SheetData objects
        """
        file_data = await self.read()
        return file_data.sheets


class SyncBaseReader(BaseReader):
    """Base class for readers that only support sync operations."""

    @abstractmethod
    def read_sync(self) -> FileData:
        """Synchronous read method.

        Returns:
            FileData containing all sheets and metadata
        """
        pass

    async def read(self) -> FileData:
        """Async wrapper around sync read."""
        import asyncio

        # Run sync method in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_sync)

    def read_all(self) -> list[SheetData]:
        """Read all sheets from the file synchronously.

        This is a convenience method that returns sheets directly
        instead of wrapped in FileData.

        Returns:
            List of SheetData objects
        """
        file_data = self.read_sync()
        return file_data.sheets
