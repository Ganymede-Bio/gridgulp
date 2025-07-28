"""Configuration model for GridPorter."""

from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for GridPorter."""

    # LLM Configuration
    suggest_names: bool = Field(True, description="Whether to use LLM for suggesting table names")
    use_local_llm: bool = Field(
        False,
        description="Use local LLM instead of remote API (auto-detected if no OpenAI key)",
    )

    # OpenAI Configuration
    openai_api_key: str | None = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model to use")

    # Ollama Configuration
    ollama_url: str = Field("http://localhost:11434", description="Ollama server URL")
    ollama_text_model: str = Field(
        "deepseek-r1:7b", description="Ollama text model for reasoning and tool use"
    )
    ollama_vision_model: str = Field(
        "qwen2.5vl:7b", description="Ollama vision model for spreadsheet analysis"
    )

    # Legacy field for backwards compatibility
    local_model: str = Field("deepseek-r1:7b", description="Legacy: use ollama_text_model instead")

    # LLM Parameters
    max_tokens_per_table: int = Field(
        50, ge=1, le=1000, description="Max tokens for naming each table"
    )
    llm_temperature: float = Field(
        0.3, ge=0.0, le=2.0, description="LLM temperature for creativity"
    )

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

    # Vision Configuration
    use_vision: bool = Field(False, description="Enable vision-based table detection")
    vision_cell_width: int = Field(
        10, ge=3, le=50, description="Cell width in pixels for bitmap generation"
    )
    vision_cell_height: int = Field(
        10, ge=3, le=50, description="Cell height in pixels for bitmap generation"
    )
    vision_mode: str = Field("binary", description="Bitmap mode: binary, grayscale, or color")

    # Region Verification Configuration
    enable_region_verification: bool = Field(
        True, description="Enable AI region proposal verification"
    )
    verification_strict_mode: bool = Field(
        False, description="Use strict verification rules (may reject more regions)"
    )
    min_region_filledness: float = Field(
        0.1, ge=0.0, le=1.0, description="Minimum filledness ratio for valid regions"
    )
    min_region_rectangularness: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum rectangularness score for valid regions"
    )
    min_region_contiguity: float = Field(
        0.5, ge=0.0, le=1.0, description="Minimum contiguity score for valid regions"
    )
    enable_verification_feedback: bool = Field(
        True, description="Enable feedback loop for failed region verifications"
    )
    max_verification_iterations: int = Field(
        2, ge=1, le=5, description="Maximum iterations for verification refinement"
    )

    # Performance Configuration
    excel_reader: str = Field(
        "calamine",
        description="Excel reader to use: 'calamine' (fast), 'openpyxl' (full features), 'auto'",
    )
    max_memory_mb: int = Field(1000, ge=100, description="Maximum memory usage in MB")
    chunk_size: int = Field(10000, ge=100, description="Rows per chunk for streaming")

    # Telemetry Configuration
    enable_telemetry: bool = Field(True, description="Enable OpenTelemetry tracking")
    telemetry_endpoint: str | None = Field(None, description="Custom telemetry endpoint")

    # Feature Collection Configuration
    enable_feature_collection: bool = Field(
        False, description="Enable collection of detection features to SQLite database"
    )
    feature_db_path: str = Field(
        "~/.gridporter/features.db", description="Path to feature collection database"
    )
    feature_retention_days: int = Field(
        30, ge=1, description="Days to retain feature data before cleanup"
    )

    # Cost Optimization Configuration (Week 6)
    max_cost_per_session: float = Field(
        1.0, ge=0.0, description="Maximum cost allowed per session in USD"
    )
    max_cost_per_file: float = Field(
        0.1, ge=0.0, description="Maximum cost allowed per file in USD"
    )
    enable_simple_case_detection: bool = Field(
        True, description="Enable simple case detection to avoid vision costs"
    )
    enable_island_detection: bool = Field(
        True, description="Enable traditional island detection as fallback"
    )
    use_excel_metadata: bool = Field(
        True, description="Use Excel metadata (ListObjects, named ranges) for detection hints"
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

    # OpenAI Admin Configuration
    openai_admin_key: str | None = Field(
        None, description="OpenAI admin API key for accessing costs endpoint"
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
        """Create config from environment variables with auto-detection of LLM provider.

        This method will automatically load from a .env file if present, then read
        configuration from environment variables.
        """
        import os

        from dotenv import load_dotenv

        # Load .env file if it exists (will not override existing env vars)
        load_dotenv()

        # Get OpenAI configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Auto-detect if we should use local LLM (Ollama) when no OpenAI key is provided
        use_local_llm_env = os.getenv("GRIDPORTER_USE_LOCAL_LLM", "").lower()
        if use_local_llm_env in ("true", "false"):
            use_local_llm = use_local_llm_env == "true"
        else:
            # Auto-detect: use Ollama if no OpenAI key is configured
            use_local_llm = openai_api_key is None or openai_api_key.strip() == ""

        return cls(
            # OpenAI Configuration
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            # LLM Provider Selection
            suggest_names=os.getenv("GRIDPORTER_SUGGEST_NAMES", "true").lower() == "true",
            use_local_llm=use_local_llm,
            # Ollama Configuration
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_text_model=os.getenv("OLLAMA_TEXT_MODEL", "deepseek-r1:7b"),
            ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b"),
            # Legacy support
            local_model=os.getenv("GRIDPORTER_LOCAL_MODEL", "deepseek-r1:7b"),
            # File Detection Configuration
            enable_magika=os.getenv("GRIDPORTER_ENABLE_MAGIKA", "true").lower() == "true",
            strict_format_checking=os.getenv("GRIDPORTER_STRICT_FORMAT_CHECKING", "false").lower()
            == "true",
            file_detection_buffer_size=int(
                os.getenv("GRIDPORTER_FILE_DETECTION_BUFFER_SIZE", "8192")
            ),
            # Vision Configuration
            use_vision=os.getenv("GRIDPORTER_USE_VISION", "false").lower() == "true",
            vision_cell_width=int(os.getenv("GRIDPORTER_VISION_CELL_WIDTH", "10")),
            vision_cell_height=int(os.getenv("GRIDPORTER_VISION_CELL_HEIGHT", "10")),
            vision_mode=os.getenv("GRIDPORTER_VISION_MODE", "binary"),
            # Performance Configuration
            excel_reader=os.getenv("GRIDPORTER_EXCEL_READER", "calamine"),
            max_memory_mb=int(os.getenv("GRIDPORTER_MAX_MEMORY_MB", "1000")),
            chunk_size=int(os.getenv("GRIDPORTER_CHUNK_SIZE", "10000")),
            # Telemetry Configuration
            enable_telemetry=os.getenv("GRIDPORTER_ENABLE_TELEMETRY", "true").lower() == "true",
            telemetry_endpoint=os.getenv("GRIDPORTER_TELEMETRY_ENDPOINT"),
            # Feature Collection Configuration
            enable_feature_collection=os.getenv(
                "GRIDPORTER_ENABLE_FEATURE_COLLECTION", "false"
            ).lower()
            == "true",
            feature_db_path=os.getenv("GRIDPORTER_FEATURE_DB_PATH", "~/.gridporter/features.db"),
            feature_retention_days=int(os.getenv("GRIDPORTER_FEATURE_RETENTION_DAYS", "30")),
            # Cost Optimization Configuration (Week 6)
            max_cost_per_session=float(os.getenv("GRIDPORTER_MAX_COST_PER_SESSION", "1.0")),
            max_cost_per_file=float(os.getenv("GRIDPORTER_MAX_COST_PER_FILE", "0.1")),
            enable_simple_case_detection=os.getenv(
                "GRIDPORTER_ENABLE_SIMPLE_CASE_DETECTION", "true"
            ).lower()
            == "true",
            enable_island_detection=os.getenv("GRIDPORTER_ENABLE_ISLAND_DETECTION", "true").lower()
            == "true",
            use_excel_metadata=os.getenv("GRIDPORTER_USE_EXCEL_METADATA", "true").lower() == "true",
            # OpenAI Admin Configuration
            openai_admin_key=os.getenv("OPENAI_ADMIN_KEY"),
            # Other Configuration
            max_file_size_mb=float(os.getenv("GRIDPORTER_MAX_FILE_SIZE_MB", "2000")),
            timeout_seconds=int(os.getenv("GRIDPORTER_TIMEOUT_SECONDS", "300")),
            log_level=os.getenv("GRIDPORTER_LOG_LEVEL", "INFO"),
        )


# Type alias for backwards compatibility
GridPorterConfig = Config
