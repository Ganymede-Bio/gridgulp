"""Configuration model for GridGulp."""

from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for GridGulp."""

    # Detection Configuration
    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence for table detection"
    )
    max_tables_per_sheet: int = Field(50, ge=1, description="Maximum tables to detect per sheet")
    min_table_size: tuple[int, int] = Field((2, 2), description="Minimum table size (rows, cols)")
    detect_merged_cells: bool = Field(True, description="Whether to handle merged cells")

    # File Format Detection Configuration
    enable_magika: bool = Field(True, description="Enable Magika AI-powered file type detection")
    strict_format_checking: bool = Field(
        False, description="Raise errors for unsupported file formats"
    )
    file_detection_buffer_size: int = Field(
        8192, ge=512, description="Buffer size for file detection (bytes)"
    )

    # Processing Limits
    max_file_size_mb: float = Field(2000.0, ge=0.1, description="Maximum file size in MB")
    timeout_seconds: int = Field(300, ge=10, description="Processing timeout in seconds")
    max_sheets: int = Field(10, ge=1, description="Maximum sheets to process")

    # Performance Configuration
    excel_reader: str = Field(
        "calamine",
        description="Excel reader to use: 'calamine' (fast), 'openpyxl' (full features), 'auto'",
    )
    max_memory_mb: int = Field(1000, ge=100, description="Maximum memory usage in MB")
    chunk_size: int = Field(10000, ge=100, description="Rows per chunk for streaming")

    enable_simple_case_detection: bool = Field(
        True, description="Enable simple case detection to avoid vision costs"
    )
    enable_island_detection: bool = Field(
        True, description="Enable traditional island detection as fallback"
    )
    use_excel_metadata: bool = Field(
        True,
        description="Use Excel metadata (ListObjects, named ranges) for detection hints",
    )

    # Detection Thresholds (Configurable Constants)
    island_min_cells: int = Field(
        20, ge=1, description="Minimum cells for good confidence in island detection"
    )
    island_density_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="High density threshold for island detection"
    )
    format_blank_row_threshold: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Percentage of cells that must be empty to consider row blank",
    )
    format_total_formatting_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Percentage of cells that must be bold for total formatting",
    )

    # Caching
    enable_cache: bool = Field(True, description="Enable result caching")
    cache_dir: Path | None = Field(None, description="Cache directory path")
    cache_ttl_hours: int = Field(24, ge=1, description="Cache time-to-live in hours")

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_file: Path | None = Field(None, description="Log file path")
    enable_debug: bool = Field(False, description="Enable debug mode")

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        import os

        return cls(
            # File Detection Configuration
            enable_magika=os.getenv("GRIDGULP_ENABLE_MAGIKA", "true").lower() == "true",
            strict_format_checking=os.getenv("GRIDGULP_STRICT_FORMAT_CHECKING", "false").lower()
            == "true",
            file_detection_buffer_size=int(
                os.getenv("GRIDGULP_FILE_DETECTION_BUFFER_SIZE", "8192")
            ),
            # Performance Configuration
            excel_reader=os.getenv("GRIDGULP_EXCEL_READER", "calamine"),
            max_memory_mb=int(os.getenv("GRIDGULP_MAX_MEMORY_MB", "1000")),
            chunk_size=int(os.getenv("GRIDGULP_CHUNK_SIZE", "10000")),
            enable_simple_case_detection=os.getenv(
                "GRIDGULP_ENABLE_SIMPLE_CASE_DETECTION", "true"
            ).lower()
            == "true",
            enable_island_detection=os.getenv("GRIDGULP_ENABLE_ISLAND_DETECTION", "true").lower()
            == "true",
            use_excel_metadata=os.getenv("GRIDGULP_USE_EXCEL_METADATA", "true").lower() == "true",
            # Other Configuration
            max_file_size_mb=float(os.getenv("GRIDGULP_MAX_FILE_SIZE_MB", "2000")),
            timeout_seconds=int(os.getenv("GRIDGULP_TIMEOUT_SECONDS", "300")),
            log_level=os.getenv("GRIDGULP_LOG_LEVEL", "INFO"),
        )


# Type alias for backwards compatibility
GridGulpConfig = Config
