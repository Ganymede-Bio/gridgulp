"""Main GridPorter class."""

import logging
import time
from pathlib import Path

from gridporter.models import DetectionResult, FileInfo, FileType
from gridporter.readers import ReaderError, create_reader
from gridporter.utils.file_magic import detect_file_info

from .agents.complex_table_agent import ComplexTableAgent
from .config import Config
from .vision.integrated_pipeline import IntegratedVisionPipeline

logger = logging.getLogger(__name__)


class GridPorter:
    """Main class for intelligent spreadsheet table detection."""

    def __init__(
        self,
        config: Config | None = None,
        suggest_names: bool | None = None,
        use_local_llm: bool | None = None,
        confidence_threshold: float | None = None,
        **kwargs,
    ):
        """Initialize GridPorter.

        Args:
            config: Configuration object. If None, loads from environment.
            suggest_names: Override for LLM name suggestions
            use_local_llm: Override for local LLM usage
            confidence_threshold: Override for confidence threshold
            **kwargs: Additional config overrides
        """
        # Load base config
        if config is None:
            config = Config.from_env()

        # Apply overrides
        if suggest_names is not None:
            config.suggest_names = suggest_names
        if use_local_llm is not None:
            config.use_local_llm = use_local_llm
        if confidence_threshold is not None:
            config.confidence_threshold = confidence_threshold

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self._setup_logging()
        self._setup_telemetry()

        # Initialize components
        self._complex_table_agent = ComplexTableAgent(config)
        self._vision_pipeline = (
            IntegratedVisionPipeline.from_config(config) if config.use_vision else None
        )
        self._readers = {}

        logger.info(f"GridPorter initialized with config: {config}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=self.config.log_file,
        )

    def _setup_telemetry(self) -> None:
        """Setup telemetry and feature collection."""
        # Initialize feature collection if enabled
        if self.config.enable_feature_collection:
            try:
                from .telemetry import get_feature_collector

                feature_collector = get_feature_collector()
                feature_collector.initialize(enabled=True, db_path=self.config.feature_db_path)
                logger.info("Feature collection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize feature collection: {e}")
                # Don't fail the entire initialization

    async def detect_tables(self, file_path: str | Path) -> DetectionResult:
        """Detect tables in a spreadsheet file.

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

        logger.info(f"Starting table detection for: {file_path}")

        # Validate file
        self._validate_file(file_path)

        # Detect file type
        file_info = await self._analyze_file(file_path)
        logger.info(f"Detected file type: {file_info.type}")

        # Read file data using appropriate reader
        try:
            reader = create_reader(file_path, file_info)
            file_data = await reader.read()
            logger.info(f"Successfully read {file_data.sheet_count} sheets")
        except ReaderError as e:
            logger.error(f"Failed to read file: {e}")
            raise ValueError(f"Could not read file: {e}") from e

        # Run table detection using complex table agent
        sheets = []
        total_llm_calls = 0
        total_llm_tokens = 0

        for sheet_data in file_data.sheets:
            from gridporter.models import SheetResult

            sheet_start_time = time.time()
            sheet_errors = []

            # Set file path and type for feature collection
            sheet_data.file_path = str(file_path)
            sheet_data.file_type = file_info.type.value

            try:
                # Run vision pipeline if enabled
                vision_result = None
                if self.config.use_vision and self._vision_pipeline:
                    pipeline_result = self._vision_pipeline.process_sheet(sheet_data)
                    # Convert pipeline result to vision result format
                    # (This is simplified - real implementation would do proper conversion)
                    vision_result = pipeline_result

                # Detect complex tables
                detection_result = await self._complex_table_agent.detect_complex_tables(
                    sheet_data,
                    vision_result=vision_result,
                    simple_tables=None,  # Could pass simple tables from initial detection
                )

                # Track LLM usage if applicable
                if hasattr(detection_result, "llm_calls"):
                    total_llm_calls += detection_result.llm_calls
                if hasattr(detection_result, "llm_tokens"):
                    total_llm_tokens += detection_result.llm_tokens

                sheet_result = SheetResult(
                    name=sheet_data.name,
                    tables=detection_result.tables,
                    processing_time=time.time() - sheet_start_time,
                    errors=sheet_errors,
                    metadata=detection_result.detection_metadata,
                )
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_data.name}: {e}")
                sheet_errors.append(str(e))
                sheet_result = SheetResult(
                    name=sheet_data.name,
                    tables=[],
                    processing_time=time.time() - sheet_start_time,
                    errors=sheet_errors,
                )

            sheets.append(sheet_result)

        detection_time = time.time() - start_time

        # Collect methods used
        methods_used = ["file_reading", "complex_detection"]
        if self.config.use_vision:
            methods_used.extend(["vision_analysis", "bitmap_detection"])

        result = DetectionResult(
            file_info=file_info,
            sheets=sheets,
            detection_time=detection_time,
            methods_used=methods_used,
            metadata={
                "file_data_available": True,
                "total_cells": sum(len(sheet.get_non_empty_cells()) for sheet in file_data.sheets),
                "reader_metadata": file_data.metadata,
                "total_tables": sum(len(sheet.tables) for sheet in sheets),
                "vision_enabled": self.config.use_vision,
                "multi_row_headers_detected": sum(
                    1
                    for sheet in sheets
                    for table in sheet.tables
                    if table.header_info and table.header_info.is_multi_row
                ),
                "llm_calls": total_llm_calls,
                "llm_tokens": total_llm_tokens,
            },
        )

        logger.info(
            f"Detection completed in {detection_time:.2f}s. "
            f"Read {len(file_data.sheets)} sheets with {result.metadata.get('total_cells', 0)} non-empty cells."
        )

        return result

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
        results = []
        for file_path in file_paths:
            try:
                result = await self.detect_tables(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Could create an error result instead of skipping
                continue
        return results

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return [ft.value for ft in FileType if ft != FileType.UNKNOWN]
