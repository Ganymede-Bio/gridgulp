"""Adapter to provide backward compatibility and reader selection."""

import logging
from pathlib import Path

from ..config import GridGulpConfig
from ..models.file_info import FileInfo
from ..models.sheet_data import FileData
from .base_reader import BaseReader
from .factory import ReaderFactory

logger = logging.getLogger(__name__)


class ReaderAdapter:
    """Adapter that selects appropriate reader based on configuration.

    This class provides:
    - Automatic reader selection based on config
    - Backward compatibility
    """

    def __init__(self, config: GridGulpConfig | None = None):
        """Initialize reader adapter.

        Args:
            config: GridGulp configuration
        """
        self.config = config or GridGulpConfig()
        self.factory = ReaderFactory()

    def get_reader(self, file_path: Path, file_info: FileInfo) -> BaseReader:
        """Get appropriate reader based on file type and configuration.

        Args:
            file_path: Path to file
            file_info: File information

        Returns:
            Appropriate reader instance
        """
        # Use factory to get appropriate reader
        return self.factory.get_reader(file_path, file_info)

    async def read_file(self, file_path: Path, file_info: FileInfo) -> FileData:
        """Read file with appropriate reader and convert to configured format.

        Args:
            file_path: Path to file
            file_info: File information

        Returns:
            FileData with sheets in configured format
        """
        # Get reader
        reader = self.get_reader(file_path, file_info)

        # Read file
        file_data = await reader.read()

        return file_data
