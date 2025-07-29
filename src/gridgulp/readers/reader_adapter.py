"""Adapter to provide backward compatibility and reader selection."""

import logging
from pathlib import Path

from ..config import GridGulpConfig
from ..models.file_info import FileInfo, FileType
from ..models.sheet_data import FileData
from .base_reader import BaseReader
from .calamine_reader import CalamineReader
from .excel_reader import ExcelReader
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
        # For Excel files, check configuration
        if file_info.type in {
            FileType.XLSX,
            FileType.XLS,
            FileType.XLSM,
            FileType.XLSB,
        }:
            if self.config.excel_reader == "calamine":
                # Use fast Calamine reader
                logger.info("Using Calamine reader for Excel file")
                return CalamineReader(file_path, file_info)
            elif self.config.excel_reader == "openpyxl":
                # Use feature-rich openpyxl reader
                logger.info("Using openpyxl reader for Excel file")
                return ExcelReader(file_path, file_info)
            else:  # auto
                # Auto-select based on file characteristics
                return self._auto_select_excel_reader(file_path, file_info)

        # For other file types, use factory default
        return self.factory.get_reader(file_path, file_info)

    def _auto_select_excel_reader(self, file_path: Path, file_info: FileInfo) -> BaseReader:
        """Auto-select Excel reader based on file characteristics.

        Args:
            file_path: Path to file
            file_info: File information

        Returns:
            Selected reader
        """
        # Use Calamine by default for performance
        # Could add logic here to detect when openpyxl is needed
        # (e.g., password protected, needs formatting info, etc.)

        # Check if file requires formatting information
        # For now, default to Calamine for speed
        logger.info("Auto-selecting Calamine reader for performance")
        return CalamineReader(file_path, file_info)

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
