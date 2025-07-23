"""Configuration model for GridPorter."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for GridPorter."""

    # LLM Configuration
    suggest_names: bool = Field(
        True, description="Whether to use LLM for suggesting table names"
    )
    use_local_llm: bool = Field(
        False, description="Use local LLM instead of remote API"
    )
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model to use")
    local_model: str = Field("mistral:7b", description="Local LLM model name")
    ollama_url: str = Field(
        "http://localhost:11434", description="Ollama server URL"
    )
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

    # Processing Limits
    max_file_size_mb: float = Field(
        50.0, ge=0.1, description="Maximum file size in MB"
    )
    timeout_seconds: int = Field(
        300, ge=10, description="Processing timeout in seconds"
    )
    max_sheets: int = Field(10, ge=1, description="Maximum sheets to process")

    # Caching
    enable_cache: bool = Field(True, description="Enable result caching")
    cache_dir: Optional[Path] = Field(
        None, description="Cache directory path"
    )
    cache_ttl_hours: int = Field(
        24, ge=1, description="Cache time-to-live in hours"
    )

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[Path] = Field(None, description="Log file path")
    enable_debug: bool = Field(False, description="Enable debug mode")

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        import os

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            suggest_names=os.getenv("GRIDPORTER_SUGGEST_NAMES", "true").lower() == "true",
            use_local_llm=os.getenv("GRIDPORTER_USE_LOCAL_LLM", "false").lower() == "true",
            local_model=os.getenv("GRIDPORTER_LOCAL_MODEL", "mistral:7b"),
            max_file_size_mb=float(os.getenv("GRIDPORTER_MAX_FILE_SIZE_MB", "50")),
            timeout_seconds=int(os.getenv("GRIDPORTER_TIMEOUT_SECONDS", "300")),
            log_level=os.getenv("GRIDPORTER_LOG_LEVEL", "INFO"),
        )