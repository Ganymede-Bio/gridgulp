"""Configuration model for GridPorter."""

from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for GridPorter."""

    # LLM Configuration
    suggest_names: bool = Field(
        True, description="Whether to use LLM for suggesting table names"
    )
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
    local_model: str = Field(
        "deepseek-r1:7b", description="Legacy: use ollama_text_model instead"
    )

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
    max_tables_per_sheet: int = Field(
        50, ge=1, description="Maximum tables to detect per sheet"
    )
    min_table_size: tuple[int, int] = Field(
        (2, 2), description="Minimum table size (rows, cols)"
    )
    detect_merged_cells: bool = Field(
        True, description="Whether to handle merged cells"
    )

    # File Format Detection Configuration
    enable_magika: bool = Field(
        True, description="Enable Magika AI-powered file type detection"
    )
    strict_format_checking: bool = Field(
        False, description="Raise errors for unsupported file formats"
    )
    file_detection_buffer_size: int = Field(
        8192, ge=512, description="Buffer size for file detection (bytes)"
    )

    # Processing Limits
    max_file_size_mb: float = Field(
        2000.0, ge=0.1, description="Maximum file size in MB"
    )
    timeout_seconds: int = Field(
        300, ge=10, description="Processing timeout in seconds"
    )
    max_sheets: int = Field(10, ge=1, description="Maximum sheets to process")

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
        """Create config from environment variables with auto-detection of LLM provider."""
        import os

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
            strict_format_checking=os.getenv("GRIDPORTER_STRICT_FORMAT_CHECKING", "false").lower() == "true",
            file_detection_buffer_size=int(os.getenv("GRIDPORTER_FILE_DETECTION_BUFFER_SIZE", "8192")),

            # Other Configuration
            max_file_size_mb=float(os.getenv("GRIDPORTER_MAX_FILE_SIZE_MB", "2000")),
            timeout_seconds=int(os.getenv("GRIDPORTER_TIMEOUT_SECONDS", "300")),
            log_level=os.getenv("GRIDPORTER_LOG_LEVEL", "INFO"),
        )
