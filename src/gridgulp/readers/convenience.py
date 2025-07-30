"""Convenience functions for easy reader usage."""

import asyncio
from pathlib import Path
from typing import Any

from ..models.file_info import FileInfo
from ..utils.file_magic import detect_file_type
from .base_reader import AsyncBaseReader, SyncBaseReader
from .factory import create_reader, get_factory


def get_reader(file_path: str | Path) -> SyncBaseReader:
    """Get appropriate reader for a file by auto-detecting its type.

    Args:
        file_path: Path to the file (can be string or Path)

    Returns:
        Sync reader instance for the file

    Raises:
        UnsupportedFileError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file type
    detected_type = detect_file_type(file_path)

    # Create FileInfo
    file_info = FileInfo(path=file_path, type=detected_type, size=file_path.stat().st_size)

    # Get reader from factory
    reader = create_reader(file_path, file_info)

    # Wrap all readers to provide read_file() method
    if isinstance(reader, AsyncBaseReader):

        class SyncReaderAdapter(SyncBaseReader):
            def __init__(self, async_reader: AsyncBaseReader) -> None:
                self._async_reader = async_reader
                super().__init__(async_reader.file_path, async_reader.file_info)

            def read_file(self, file_path: str | Path) -> Any:  # noqa: ARG002
                return asyncio.run(self._async_reader.read_async())

            def read_sync(self) -> Any:
                return asyncio.run(self._async_reader.read_async())

            def can_read(self) -> bool:
                return self._async_reader.can_read()

            def get_supported_formats(self) -> list[str]:
                return self._async_reader.get_supported_formats()

        return SyncReaderAdapter(reader)
    else:
        # Wrap sync reader to add read_file() method
        class SyncReaderWrapper(SyncBaseReader):
            def __init__(self, sync_reader: SyncBaseReader) -> None:
                self._sync_reader = sync_reader
                super().__init__(sync_reader.file_path, sync_reader.file_info)

            def read_file(self, file_path: str | Path) -> Any:  # noqa: ARG002
                # Note: file_path parameter is ignored, reader already knows its file
                return self._sync_reader.read_sync()

            def read_sync(self) -> Any:
                return self._sync_reader.read_sync()

            def can_read(self) -> bool:
                return self._sync_reader.can_read()

            def get_supported_formats(self) -> list[str]:
                return self._sync_reader.get_supported_formats()

        return SyncReaderWrapper(reader)  # type: ignore[arg-type]


async def get_async_reader(file_path: str | Path) -> AsyncBaseReader:
    """Get appropriate async reader for a file by auto-detecting its type.

    Args:
        file_path: Path to the file (can be string or Path)

    Returns:
        Async reader instance for the file

    Raises:
        UnsupportedFileError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file type
    detected_type = detect_file_type(file_path)

    # Create FileInfo
    file_info = FileInfo(path=file_path, type=detected_type, size=file_path.stat().st_size)

    # Get reader from factory
    reader = create_reader(file_path, file_info)

    # Wrap all readers to provide consistent interface
    if isinstance(reader, SyncBaseReader):

        class AsyncReaderAdapter(AsyncBaseReader):
            def __init__(self, sync_reader: SyncBaseReader) -> None:
                self._sync_reader = sync_reader
                super().__init__(sync_reader.file_path, sync_reader.file_info)

            async def read_file(self, file_path: str | Path) -> Any:  # noqa: ARG002
                # Note: file_path parameter is ignored, reader already knows its file
                return self._sync_reader.read_sync()

            async def read_async(self) -> Any:
                return self._sync_reader.read_sync()

            def can_read(self) -> bool:
                return self._sync_reader.can_read()

            def get_supported_formats(self) -> list[str]:
                return self._sync_reader.get_supported_formats()

        return AsyncReaderAdapter(reader)
    else:
        # Wrap async reader to add read_file() method
        class AsyncReaderWrapper(AsyncBaseReader):
            def __init__(self, async_reader: AsyncBaseReader) -> None:
                self._async_reader = async_reader
                super().__init__(async_reader.file_path, async_reader.file_info)

            async def read_file(self, file_path: str | Path) -> Any:  # noqa: ARG002
                # Note: file_path parameter is ignored, reader already knows its file
                return await self._async_reader.read_async()

            async def read_async(self) -> Any:
                return await self._async_reader.read_async()

            def can_read(self) -> bool:
                return self._async_reader.can_read()

            def get_supported_formats(self) -> list[str]:
                return self._async_reader.get_supported_formats()

        return AsyncReaderWrapper(reader)  # type: ignore[arg-type]


def is_supported(file_path: str | Path) -> bool:
    """Check if a file type is supported for reading.

    Args:
        file_path: Path to the file (can be string or Path)

    Returns:
        True if file type is supported, False otherwise
    """
    file_path = Path(file_path)

    # Check by extension first (fast)
    extension = file_path.suffix.lower()
    supported_extensions = {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".txt"}
    if extension in supported_extensions:
        return True

    # If extension is unknown, check actual file type
    if file_path.exists():
        try:
            detected_type = detect_file_type(file_path)
            factory = get_factory()
            file_info = FileInfo(path=file_path, type=detected_type, size=file_path.stat().st_size)
            return factory.can_read(file_info)
        except Exception:
            return False

    return False
