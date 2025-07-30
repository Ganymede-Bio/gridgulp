"""Reader factory for automatic reader selection."""

import logging
from pathlib import Path

from ..models.file_info import FileInfo, FileType
from .base_reader import BaseReader, UnsupportedFileError
from .csv_reader import CSVReader
from .excel_reader import ExcelReader
from .text_reader import TextReader

logger = logging.getLogger(__name__)


class ReaderFactory:
    """Factory for creating appropriate file readers."""

    def __init__(self) -> None:
        """Initialize factory with default readers."""
        self._readers: dict[FileType, type[BaseReader]] = {}
        self._register_default_readers()

    def _register_default_readers(self) -> None:
        """Register the default built-in readers."""
        # Excel readers
        self.register_reader(FileType.XLSX, ExcelReader)
        self.register_reader(FileType.XLS, ExcelReader)
        self.register_reader(FileType.XLSM, ExcelReader)

        # CSV readers
        self.register_reader(FileType.CSV, CSVReader)
        self.register_reader(FileType.TSV, CSVReader)

        # Text readers
        self.register_reader(FileType.TXT, TextReader)

    def register_reader(self, file_type: FileType, reader_class: type[BaseReader]) -> None:
        """Register a reader for a specific file type.

        Args:
            file_type: File type to handle
            reader_class: Reader class to use
        """
        self._readers[file_type] = reader_class
        logger.debug(f"Registered {reader_class.__name__} for {file_type.value} files")

    def unregister_reader(self, file_type: FileType) -> None:
        """Unregister a reader for a file type.

        Args:
            file_type: File type to remove
        """
        if file_type in self._readers:
            del self._readers[file_type]
            logger.debug(f"Unregistered reader for {file_type.value} files")

    def get_reader(self, file_path: Path, file_info: FileInfo) -> BaseReader:
        """Get appropriate reader for a file.

        Args:
            file_path: Path to the file
            file_info: File information from detection

        Returns:
            Reader instance for the file

        Raises:
            UnsupportedFileError: If no reader available for file type
        """
        file_type = file_info.type

        if file_type not in self._readers:
            raise UnsupportedFileError(
                f"No reader available for {file_type.value} files. "
                f"Supported types: {self.get_supported_types()}"
            )

        reader_class = self._readers[file_type]
        reader = reader_class(file_path, file_info)

        # Verify reader can handle the file
        if not reader.can_read():
            raise UnsupportedFileError(
                f"{reader_class.__name__} cannot read {file_type.value} file: {file_path}"
            )

        logger.debug(f"Created {reader_class.__name__} for {file_path}")
        return reader

    def get_supported_types(self) -> list[str]:
        """Get list of supported file types.

        Returns:
            List of supported file type strings
        """
        return [file_type.value for file_type in self._readers]

    def can_read(self, file_info: FileInfo) -> bool:
        """Check if any reader can handle a file type.

        Args:
            file_info: File information

        Returns:
            True if file type is supported
        """
        return file_info.type in self._readers

    def get_reader_info(self) -> dict[str, list[str]]:
        """Get information about registered readers.

        Returns:
            Dictionary mapping reader names to supported formats
        """
        reader_info = {}

        for file_type, reader_class in self._readers.items():
            reader_name = reader_class.__name__
            if reader_name not in reader_info:
                # Create temporary instance to get supported formats
                try:
                    temp_reader = reader_class(
                        Path("dummy"),
                        FileInfo(path=Path("dummy"), type=file_type, size=0),
                    )
                    reader_info[reader_name] = temp_reader.get_supported_formats()
                except Exception:
                    reader_info[reader_name] = [file_type.value]

        return reader_info

    def list_readers(self) -> list[str]:
        """Get list of registered reader names.

        Returns:
            List of reader class names
        """
        return list({reader_class.__name__ for reader_class in self._readers.values()})


# Global factory instance
_factory_instance: ReaderFactory | None = None


def get_factory() -> ReaderFactory:
    """Get the global reader factory instance.

    Returns:
        ReaderFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = ReaderFactory()
    return _factory_instance


def reset_factory() -> None:
    """Reset the global factory instance."""
    global _factory_instance
    _factory_instance = None


def create_reader(file_path: Path, file_info: FileInfo) -> BaseReader:
    """Convenience function to create a reader.

    Args:
        file_path: Path to the file
        file_info: File information

    Returns:
        Appropriate reader instance

    Raises:
        UnsupportedFileError: If no reader available
    """
    factory = get_factory()
    return factory.get_reader(file_path, file_info)


def register_custom_reader(file_type: FileType, reader_class: type[BaseReader]) -> None:
    """Register a custom reader globally.

    Args:
        file_type: File type to handle
        reader_class: Custom reader class
    """
    factory = get_factory()
    factory.register_reader(file_type, reader_class)
