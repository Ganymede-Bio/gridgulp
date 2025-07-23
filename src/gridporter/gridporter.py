"""Main GridPorter class."""

import logging
import time
from pathlib import Path
from typing import Optional, Union

from .config import Config
from .models import DetectionResult, FileInfo, FileType
from .utils import detect_file_type


logger = logging.getLogger(__name__)


class GridPorter:
    """Main class for intelligent spreadsheet table detection."""

    def __init__(
        self,
        config: Optional[Config] = None,
        suggest_names: Optional[bool] = None,
        use_local_llm: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
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

        # Initialize components (will be implemented in later phases)
        self._detector_agent = None
        self._readers = {}

        logger.info(f"GridPorter initialized with config: {config}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=self.config.log_file,
        )

    async def detect_tables(self, file_path: Union[str, Path]) -> DetectionResult:
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

        # Create placeholder result for now
        # In a full implementation, this would:
        # 1. Load appropriate reader based on file type
        # 2. Initialize detection agent
        # 3. Run detection pipeline
        # 4. Apply LLM naming if configured

        detection_time = time.time() - start_time

        result = DetectionResult(
            file_info=file_info,
            sheets=[],  # Will be populated by actual detection
            detection_time=detection_time,
            methods_used=["placeholder"],
            llm_calls=0,
            llm_tokens=0,
        )

        logger.info(
            f"Detection completed in {detection_time:.2f}s. "
            f"Found {result.total_tables} tables."
        )

        return result

    def _validate_file(self, file_path: Path) -> None:
        """Validate that file exists and is within size limits."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB "
                f"(max: {self.config.max_file_size_mb}MB)"
            )

    async def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze file and create FileInfo object."""
        # This would use the file magic detection utility
        file_type = detect_file_type(file_path)
        
        return FileInfo(
            path=file_path,
            type=file_type,
            size=file_path.stat().st_size,
            detected_mime=None,  # Would be populated by magic detection
            encoding=None,       # Would be detected for text files
        )

    async def batch_detect(self, file_paths: list[Union[str, Path]]) -> list[DetectionResult]:
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