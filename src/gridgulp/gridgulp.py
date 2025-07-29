"""Main GridGulp class - simplified for Option 3 (Hybrid Approach)."""

import logging
import time
from pathlib import Path
from typing import Any

from gridgulp.models import DetectionResult, FileInfo, FileType
from gridgulp.readers import ReaderError, create_reader
from gridgulp.readers.base_reader import SyncBaseReader
from gridgulp.utils.file_magic import detect_file_info

from .config import Config
from .detection import TableDetectionAgent
from .utils.logging_context import (
    FileContext,
    OperationContext,
    SheetContext,
    get_contextual_logger,
)

logger = get_contextual_logger(__name__)


class GridGulp:
    """Main class for intelligent spreadsheet table detection - simplified architecture."""

    def __init__(
        self,
        config: Config | None = None,
        confidence_threshold: float | None = None,
        **kwargs: Any,
    ):
        """Initialize GridGulp with simplified architecture.

        Args:
            config: Configuration object. If None, uses minimal config.
            confidence_threshold: Override for confidence threshold
            **kwargs: Additional config overrides
        """
        # Load base config
        if config is None:
            config = Config.from_env()

        # Apply overrides
        if confidence_threshold is not None:
            config.confidence_threshold = confidence_threshold

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self._setup_logging()

        logger.info("GridGulp initialized with simplified architecture")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=self.config.log_file,
        )

    async def detect_tables(self, file_path: str | Path) -> DetectionResult:
        """Detect tables in a spreadsheet file using simplified architecture.

        Args:
            file_path: Path to the spreadsheet file

        Returns:
            DetectionResult containing all detected tables

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If file format is not supported
            FileSizeError: If file is too large
        """
        start_time = time.time()
        file_path = Path(file_path)

        with FileContext(str(file_path)):
            logger.info("Starting table detection")

            # Validate file
            with OperationContext("file_validation"):
                self._validate_file(file_path)

            # Detect file type
            with OperationContext("file_type_detection"):
                file_info = await self._analyze_file(file_path)
                logger.info(f"Detected file type: {file_info.type}")

            # Read file data using appropriate reader
            with OperationContext("file_reading"):
                try:
                    reader = create_reader(file_path, file_info)
                    if isinstance(reader, SyncBaseReader):
                        file_data = reader.read_sync()
                    else:
                        raise ReaderError("Expected sync reader but got async reader")
                    sheet_count = len(list(file_data.sheets))
                    logger.info(f"Successfully read {sheet_count} sheets")
                except ReaderError as e:
                    logger.error(f"Failed to read file: {e}")
                    raise ValueError(
                        f"Could not read file '{file_path}': {type(e).__name__}: {e}. "
                        f"Please check if the file exists and is in a supported format."
                    ) from e

            # Run simplified table detection
            sheets = []
            total_tables = 0

            for sheet_data in file_data.sheets:
                from gridgulp.models import SheetResult

                with SheetContext(sheet_data.name):
                    sheet_start_time = time.time()
                    sheet_errors: list[str] = []

                    try:
                        # Run simplified detection
                        with OperationContext("table_detection"):
                            # Create a new detection agent with the file type
                            detection_agent = TableDetectionAgent(
                                confidence_threshold=self.config.confidence_threshold,
                                file_type=file_info.type,
                            )
                            detection_result = await detection_agent.detect_tables(sheet_data)

                        total_tables += len(detection_result.tables)

                        sheet_result = SheetResult(
                            name=sheet_data.name,
                            tables=detection_result.tables,
                            processing_time=time.time() - sheet_start_time,
                            errors=sheet_errors,
                            metadata=detection_result.processing_metadata,
                        )
                    except Exception as e:
                        logger.error(f"Error processing sheet: {e}")
                        sheet_errors.append(str(e))
                        sheet_result = SheetResult(
                            name=sheet_data.name,
                            tables=[],
                            processing_time=time.time() - sheet_start_time,
                            errors=sheet_errors,
                        )

                    sheets.append(sheet_result)

            detection_time = time.time() - start_time

            # Simplified methods used
            methods_used = ["file_reading", "simplified_detection"]

            # Calculate total cells (simplified)
            total_cells = 0
            try:
                for sheet in file_data.sheets:
                    total_cells += len(sheet.cells)
            except Exception:
                total_cells = 0

            result = DetectionResult(
                file_info=file_info,
                sheets=sheets,
                detection_time=detection_time,
                methods_used=methods_used,
                metadata={
                    "file_data_available": True,
                    "total_cells": total_cells,
                    "total_tables": total_tables,
                    "vision_enabled": False,  # Removed in simplified architecture
                    "simplified_architecture": True,
                },
            )

            logger.info(
                f"Detection completed in {detection_time:.2f}s. "
                f"Read {len(sheets)} sheets with {total_tables} tables detected."
            )

            return result

    def detect_tables_sync(self, file_path: str | Path) -> DetectionResult:
        """Synchronous version of detect_tables for use in Jupyter notebooks and sync code.

        Args:
            file_path: Path to the spreadsheet file

        Returns:
            DetectionResult containing all detected tables

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If file format is not supported
            FileSizeError: If file is too large

        Example:
            >>> gg = GridGulp()
            >>> result = gg.detect_tables_sync("sales_report.xlsx")
            >>> print(f"Found {result.total_tables} tables")
        """
        import asyncio

        try:
            # Check if there's a running event loop (e.g., in Jupyter)
            asyncio.get_running_loop()
            # If we're here, we're in an async context (like Jupyter)
            # Run in a separate thread to avoid event loop conflicts
            import concurrent.futures

            def _run_async() -> DetectionResult:
                return asyncio.run(self.detect_tables(file_path))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async)
                return future.result()
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            return asyncio.run(self.detect_tables(file_path))

    # Backwards compatibility alias
    def extract_from_file(self, file_path: str | Path) -> DetectionResult:
        """Deprecated: Use detect_tables_sync instead."""
        import warnings

        warnings.warn(
            "extract_from_file is deprecated, use detect_tables_sync instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detect_tables_sync(file_path)

    def _validate_file(self, file_path: Path) -> None:
        """Validate that file exists and is within size limits."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB " f"(max: {self.config.max_file_size_mb}MB)"
            )

    async def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze file and create FileInfo object with comprehensive detection."""
        # Use enhanced file detection
        detection_result = detect_file_info(file_path)

        # Log format mismatch warnings
        if detection_result.format_mismatch:
            logger.warning(
                f"File format mismatch detected: {file_path.name} "
                f"has extension suggesting {detection_result.extension_type} "
                f"but content appears to be {detection_result.detected_type}"
            )

        return FileInfo(
            path=file_path,
            type=detection_result.detected_type,
            size=file_path.stat().st_size,
            detected_mime=detection_result.mime_type,
            extension_format=detection_result.extension_type,
            detection_confidence=detection_result.confidence,
            format_mismatch=detection_result.format_mismatch,
            detection_method=detection_result.method,
            encoding=detection_result.encoding,
            magic_bytes=detection_result.magic_bytes,
        )

    async def batch_detect(self, file_paths: list[str | Path]) -> list[DetectionResult]:
        """Detect tables in multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            List of DetectionResult objects
        """
        import asyncio

        async def process_file(file_path: str | Path) -> DetectionResult | None:
            try:
                return await self.detect_tables(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None

        # Process files concurrently
        tasks = [process_file(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks)

        # Filter out None results from errors
        return [result for result in results if result is not None]

    def batch_detect_sync(self, file_paths: list[str | Path]) -> list[DetectionResult]:
        """Synchronous version of batch_detect for use in Jupyter notebooks and sync code.

        Args:
            file_paths: List of file paths

        Returns:
            List of DetectionResult objects (None entries for failed files)

        Example:
            >>> gg = GridGulp()
            >>> files = ["report1.xlsx", "report2.csv", "data.txt"]
            >>> results = gg.batch_detect_sync(files)
            >>> for result in results:
            ...     if result:
            ...         print(f"{result.file_info.path.name}: {result.total_tables} tables")
        """
        import asyncio

        try:
            # Check if there's a running event loop (e.g., in Jupyter)
            asyncio.get_running_loop()
            # Run in a separate thread
            import concurrent.futures

            def _run_async() -> list[DetectionResult]:
                return asyncio.run(self.batch_detect(file_paths))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async)
                return future.result()
        except RuntimeError:
            # No running event loop
            return asyncio.run(self.batch_detect(file_paths))

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return [ft.value for ft in FileType if ft != FileType.UNKNOWN]
