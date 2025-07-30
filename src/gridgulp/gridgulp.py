"""Main GridGulp class - simplified for Option 3 (Hybrid Approach)."""

import logging
import time
from pathlib import Path
from typing import Any

from gridgulp.models import DetectionResult, FileInfo, FileType
from gridgulp.models.file_info import UnsupportedFormatError
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
    """Main class for intelligent spreadsheet table detection - simplified architecture.

    GridGulp automatically detects and extracts tables from spreadsheets with zero
    external dependencies. It handles Excel (xlsx, xls, xlsm), CSV, TSV, and text files,
    detecting tables even when they don't start at A1 or when multiple tables exist
    on a single sheet.

    Examples
    --------
    Basic usage::

        >>> from gridgulp import GridGulp
        >>> gg = GridGulp()
        >>> result = await gg.detect_tables("sales_report.xlsx")
        >>> print(f"Found {result.total_tables} tables")

    With custom configuration::

        >>> config = Config(confidence_threshold=0.8)
        >>> gg = GridGulp(config=config)

    Notes
    -----
    The simplified architecture focuses on algorithmic detection methods that handle
    ~97% of real-world spreadsheets without requiring external services or AI APIs.
    """

    def __init__(
        self,
        config: Config | None = None,
        confidence_threshold: float | None = None,
        **kwargs: Any,
    ):
        """Initialize GridGulp with simplified architecture.

        Args
        ----
        config : Config, optional
            Configuration object. If None, loads from environment or uses defaults.
        confidence_threshold : float, optional
            Override for confidence threshold (0.0-1.0). Tables with confidence
            scores below this threshold are filtered out.
        **kwargs : Any
            Additional config overrides. Any attribute of the Config class can
            be overridden by passing it as a keyword argument.

        Examples
        --------
        Default initialization::

            >>> gg = GridGulp()

        With custom confidence threshold::

            >>> gg = GridGulp(confidence_threshold=0.9)

        With multiple overrides::

            >>> gg = GridGulp(
            ...     confidence_threshold=0.8,
            ...     max_tables_per_sheet=100,
            ...     timeout_seconds=600
            ... )

        Notes
        -----
        Configuration can also be set via environment variables. See Config.from_env()
        for details on supported environment variables.
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
        """Setup logging configuration based on config settings.

        Notes
        -----
        This method configures Python's logging module with the level and format
        specified in the configuration. If a log file is specified, logs will be
        written to that file instead of stderr.
        """
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

                # Check for unsupported XLSB format
                if file_info.type == FileType.XLSB:
                    # If file has .xlsx extension, it might be misdetected
                    if file_path.suffix.lower() == ".xlsx":
                        logger.warning(
                            "File detected as XLSB but has .xlsx extension. "
                            "Will attempt to read as XLSX if initial read fails."
                        )
                        # We'll handle this in the reader section below
                    else:
                        # File has .xlsb extension and detected as XLSB
                        raise UnsupportedFormatError(
                            "XLSB",
                            file_path,
                            "XLSB (Excel Binary) format is not supported. "
                            "Please save the file as XLSX format in Excel.",
                        )

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
                    # If XLSB was detected but reading failed, and file has .xlsx extension, try as XLSX
                    if (
                        file_info.type == FileType.XLSB
                        and file_path.suffix.lower() == ".xlsx"
                        and "XLSB" not in str(e)
                    ):  # Avoid infinite loop if XLSB error was explicit
                        logger.warning(f"Failed to read as XLSB, retrying as XLSX: {e}")
                        # Override the detected type and try again
                        file_info.type = FileType.XLSX
                        try:
                            reader = create_reader(file_path, file_info)
                            if isinstance(reader, SyncBaseReader):
                                file_data = reader.read_sync()
                            else:
                                raise ReaderError("Expected sync reader but got async reader")
                            sheet_count = len(list(file_data.sheets))
                            logger.info(f"Successfully read {sheet_count} sheets as XLSX")
                        except Exception as e2:
                            logger.error(f"Failed to read file as XLSX too: {e2}")
                            # Re-raise the original error
                            raise ValueError(
                                f"Could not read file '{file_path}': {type(e).__name__}: {e}. "
                                f"File was detected as XLSB but could not be read. "
                                f"Please check if the file is corrupted or in an unsupported format."
                            ) from e
                    else:
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
                                config=self.config,
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

    async def detect_tables_in_directory(
        self,
        directory: str | Path,
        patterns: list[str] | None = None,
        recursive: bool = True,
        progress_callback: Any = None,
    ) -> dict[Path, DetectionResult]:
        """Detect tables in all spreadsheet files within a directory.

        Args:
            directory: Path to the directory to process
            patterns: File patterns to match (e.g., ["*.xlsx", "*.csv"]).
                     If None, processes all supported formats
            recursive: Whether to search subdirectories recursively (default: True)
            progress_callback: Optional callback function called with (current_file, total_files)

        Returns:
            Dictionary mapping file paths to their DetectionResults

        Raises:
            ValueError: If directory doesn't exist

        Example:
            >>> gg = GridGulp()
            >>> # Process all spreadsheets in a directory
            >>> results = await gg.detect_tables_in_directory("~/data")
            >>> for file_path, result in results.items():
            ...     print(f"{file_path}: {result.total_tables} tables")

            >>> # Process only Excel files
            >>> results = await gg.detect_tables_in_directory(
            ...     "~/reports",
            ...     patterns=["*.xlsx", "*.xls"]
            ... )

            >>> # With progress tracking
            >>> def show_progress(current, total):
            ...     print(f"Processing {current}/{total} files...")
            >>> results = await gg.detect_tables_in_directory(
            ...     "~/data",
            ...     progress_callback=show_progress
            ... )
        """
        directory = Path(directory).expanduser().resolve()

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Default patterns for all supported formats
        if patterns is None:
            patterns = ["*.xlsx", "*.xls", "*.xlsm", "*.csv", "*.tsv", "*.txt"]

        # Collect all matching files
        all_files: list[Path] = []
        for pattern in patterns:
            if recursive:
                all_files.extend(directory.rglob(pattern))
            else:
                all_files.extend(directory.glob(pattern))

        # Remove duplicates and sort
        all_files = sorted(set(all_files))

        # Skip hidden and temporary files
        all_files = [
            f for f in all_files if not f.name.startswith(".") and not f.name.startswith("~")
        ]

        results = {}
        total_files = len(all_files)

        for i, file_path in enumerate(all_files, 1):
            if progress_callback:
                progress_callback(i, total_files)

            try:
                result = await self.detect_tables(file_path)
                results[file_path] = result
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                # Create a minimal error result
                results[file_path] = DetectionResult(
                    file_info=FileInfo(
                        path=file_path,
                        type=FileType.UNKNOWN,
                        size=0,
                    ),
                    sheets=[],
                    detection_time=0,
                    total_tables=0,
                    methods_used=[],
                    metadata={"error": str(e)},
                )

        return results

    def detect_tables_in_directory_sync(
        self,
        directory: str | Path,
        patterns: list[str] | None = None,
        recursive: bool = True,
        progress_callback: Any = None,
    ) -> dict[Path, DetectionResult]:
        """Synchronous version of detect_tables_in_directory for use in Jupyter notebooks.

        See detect_tables_in_directory for full documentation.

        Example:
            >>> gg = GridGulp()
            >>> results = gg.detect_tables_in_directory_sync("~/data")
            >>> # Summary statistics
            >>> total_files = len(results)
            >>> total_tables = sum(r.total_tables for r in results.values())
            >>> print(f"Found {total_tables} tables across {total_files} files")
        """
        import asyncio

        try:
            # Check if there's a running event loop (e.g., in Jupyter)
            asyncio.get_running_loop()
            # If we're here, we're in an async context (like Jupyter)
            # Run in a separate thread to avoid event loop conflicts
            import concurrent.futures

            def _run_async() -> dict[Path, DetectionResult]:
                return asyncio.run(
                    self.detect_tables_in_directory(
                        directory, patterns, recursive, progress_callback
                    )
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_async)
                return future.result()
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            return asyncio.run(
                self.detect_tables_in_directory(directory, patterns, recursive, progress_callback)
            )

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
        """Validate that file exists and is within size limits.

        Args
        ----
        file_path : Path
            Path to the file to validate.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the specified path.
        ValueError
            If the file size exceeds the configured maximum (max_file_size_mb).

        Notes
        -----
        This method is called before processing any file to ensure it meets
        basic requirements. The size limit prevents memory exhaustion when
        processing extremely large files.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
            )

    async def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze file and create FileInfo object with comprehensive detection.

        Args
        ----
        file_path : Path
            Path to the file to analyze.

        Returns
        -------
        FileInfo
            Detailed file information including detected type, encoding, and metadata.

        Notes
        -----
        This method uses multiple detection strategies including:
        - File extension analysis
        - Magic byte signatures
        - Content sampling and analysis
        - Optional AI-based detection (Magika) if enabled

        Format mismatches (where file extension doesn't match content) are logged
        as warnings but processing continues with the detected content type.
        """
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
        """Detect tables in multiple files concurrently.

        Args
        ----
        file_paths : list[str | Path]
            List of file paths to process. Can be strings or Path objects.

        Returns
        -------
        list[DetectionResult]
            List of DetectionResult objects for successfully processed files.
            Failed files are logged but excluded from the results.

        Examples
        --------
        Process multiple files::

            >>> files = ["report1.xlsx", "report2.csv", "data.txt"]
            >>> results = await gg.batch_detect(files)
            >>> for result in results:
            ...     print(f"{result.file_info.path.name}: {result.total_tables} tables")

        Notes
        -----
        Files are processed concurrently for better performance. Errors in individual
        files don't stop processing of other files. Check logs for details about
        failed files.
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
        """Get list of supported file formats.

        Returns
        -------
        list[str]
            List of supported file extensions without dots (e.g., ["xlsx", "csv"]).

        Examples
        --------
        Check supported formats::

            >>> gg = GridGulp()
            >>> formats = gg.get_supported_formats()
            >>> print(formats)
            ['xlsx', 'xls', 'xlsm', 'csv', 'tsv', 'txt']

        Notes
        -----
        XLSB format is detected but not supported for reading. Files with XLSB
        format will raise an UnsupportedFormatError.
        """
        return [ft.value for ft in FileType if ft != FileType.UNKNOWN]
