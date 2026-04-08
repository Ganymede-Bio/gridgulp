"""Configuration models and loading helpers for GridGulp."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class DetectionStrategy(str, Enum):
    """Supported detection strategies."""

    GEMMA_FIRST = "gemma_first"
    HEURISTIC_ONLY = "heuristic_only"


class VisionMode(str, Enum):
    """Supported multimodal execution modes."""

    OFF = "off"
    FALLBACK_ONLY = "fallback_only"


class GemmaConfig(BaseModel):
    """Runtime configuration for the Gemma inference endpoint."""

    model_config = ConfigDict(strict=True)

    enabled: bool = Field(True, description="Enable Gemma-backed reasoning")
    base_url: str | None = Field(
        None,
        description="OpenAI-compatible API base URL (for example http://localhost:8080/v1)",
    )
    api_key: SecretStr | None = Field(None, description="Optional API key for the endpoint")
    model_id: str | None = Field(None, description="Exact Gemma model id to invoke")
    timeout_seconds: int = Field(60, ge=1, description="HTTP request timeout in seconds")
    max_retries: int = Field(2, ge=0, le=10, description="Maximum retry attempts per request")
    max_tokens: int = Field(1200, ge=1, le=32768, description="Max completion tokens")

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        return normalized or None

    @field_validator("model_id")
    @classmethod
    def _normalize_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def uses_gemma_family(self) -> bool:
        """Return True if the configured model id belongs to the Gemma family."""
        return bool(self.model_id and "gemma" in self.model_id.lower())

    def validate_policy(self) -> None:
        """Enforce the Gemma-only model policy."""
        if self.model_id and not self.uses_gemma_family():
            raise ValueError(
                "Configured model_id must point to a Gemma model because GridGulp only supports "
                "Gemma-backed LLM execution."
            )

    def ensure_ready(self) -> None:
        """Validate that Gemma execution can be used."""
        self.validate_policy()

        if not self.enabled:
            raise ValueError("Gemma execution is disabled.")
        if not self.base_url:
            raise ValueError("Gemma base_url is required for gemma_first detection.")
        if not self.model_id:
            raise ValueError("Gemma model_id is required for gemma_first detection.")


class VisionConfig(BaseModel):
    """Runtime configuration for Gemma-backed vision fallback."""

    model_config = ConfigDict(strict=True)

    enabled: bool = Field(True, description="Enable multimodal fallback reasoning")
    mode: VisionMode = Field(
        VisionMode.FALLBACK_ONLY,
        description="Vision execution mode",
    )
    max_proposal_crops: int = Field(4, ge=0, le=12, description="Maximum proposal crops")
    crop_margin_cells: int = Field(1, ge=0, le=5, description="Padding cells around crops")
    debug_image_dir: Path | None = Field(
        None,
        description="Optional directory for writing rendered debug images",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, value: VisionMode | str) -> VisionMode:
        if isinstance(value, VisionMode):
            return value
        return VisionMode(value)

    @field_validator("debug_image_dir", mode="before")
    @classmethod
    def _coerce_debug_image_dir(cls, value: Path | str | None) -> Path | None:
        if value is None or isinstance(value, Path):
            return value
        return Path(value).expanduser()

    def is_active(self) -> bool:
        """Return True when multimodal fallback should be considered."""
        return self.enabled and self.mode != VisionMode.OFF

    def validate_policy(self) -> None:
        """Enforce the v1 vision policy."""
        if self.mode not in {VisionMode.OFF, VisionMode.FALLBACK_ONLY}:
            raise ValueError("GridGulp only supports vision mode 'fallback_only' or 'off' in v1.")


class Config(BaseModel):
    """Configuration for GridGulp."""

    model_config = ConfigDict(strict=True)

    detection_strategy: DetectionStrategy = Field(
        DetectionStrategy.GEMMA_FIRST,
        description="Primary table detection strategy",
    )
    gemma: GemmaConfig = Field(default_factory=GemmaConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)

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
    max_memory_mb: int = Field(1000, ge=100, description="Maximum memory usage in MB")
    chunk_size: int = Field(10000, ge=100, description="Rows per chunk for streaming")

    enable_simple_case_detection: bool = Field(
        True, description="Enable simple case detection to avoid over-reasoning"
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

    # Adaptive threshold configuration
    adaptive_thresholds: bool = Field(
        True, description="Enable adaptive table size thresholds based on sheet size"
    )
    min_table_percentage: float = Field(
        0.005, ge=0.0, le=1.0, description="Minimum percentage of sheet cells for a valid table"
    )
    prefer_large_tables: bool = Field(
        True, description="Favor detection of larger tables over smaller fragments"
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

    # Enhanced detection parameters
    empty_row_tolerance: int = Field(
        1,
        ge=0,
        le=5,
        description="Number of empty rows to tolerate within tables",
    )
    column_gap_prevents_merge: bool = Field(
        True,
        description="Prevent merging tables separated by empty columns",
    )
    use_border_detection: bool = Field(
        True,
        description="Use cell borders for precise table boundary detection",
    )
    min_column_overlap_for_merge: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum column overlap ratio required to merge tables",
    )

    # Caching
    enable_cache: bool = Field(True, description="Enable result caching")
    cache_dir: Path | None = Field(None, description="Cache directory path")
    cache_ttl_hours: int = Field(24, ge=1, description="Cache time-to-live in hours")

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_file: Path | None = Field(None, description="Log file path")
    enable_debug: bool = Field(False, description="Enable debug mode")

    @field_validator("detection_strategy", mode="before")
    @classmethod
    def _coerce_detection_strategy(cls, value: DetectionStrategy | str) -> DetectionStrategy:
        if isinstance(value, DetectionStrategy):
            return value
        return DetectionStrategy(value)

    @classmethod
    def default_config_path(cls, cwd: str | Path | None = None) -> Path | None:
        """Return the default config file path if it exists."""
        base_dir = Path(cwd) if cwd is not None else Path.cwd()
        candidate = base_dir / "gridgulp.toml"
        return candidate if candidate.exists() else None

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load config from a TOML file."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls.model_validate(data)

    @classmethod
    def from_sources(
        cls,
        config_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Load config from TOML, environment variables, then explicit overrides."""
        raw: dict[str, Any] = {}

        resolved_path: Path | None
        if config_path is None:
            resolved_path = cls.default_config_path()
        else:
            resolved_path = Path(config_path).expanduser().resolve()

        if resolved_path and resolved_path.exists():
            with resolved_path.open("rb") as handle:
                raw = tomllib.load(handle)

        env_data = cls._env_to_dict(os.environ if env is None else env)
        raw = cls._merge_dicts(raw, env_data)
        raw = cls._merge_dicts(raw, cls._normalize_overrides(overrides or {}))

        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> Config:
        """Create config from the default TOML file and environment variables."""
        return cls.from_sources()

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a new config with explicit overrides applied."""
        merged = self.model_dump(mode="python")
        merged = self._merge_dicts(merged, self._normalize_overrides(overrides))
        return type(self).model_validate(merged)

    def validate_runtime(self) -> None:
        """Validate runtime policy and required Gemma configuration."""
        self.gemma.validate_policy()
        self.vision.validate_policy()

        if self.detection_strategy == DetectionStrategy.GEMMA_FIRST:
            self.gemma.ensure_ready()

    @staticmethod
    def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = Config._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _env_to_dict(cls, env: Mapping[str, str]) -> dict[str, Any]:
        mapping: dict[str, tuple[str, ...]] = {
            "GRIDGULP_DETECTION_STRATEGY": ("detection_strategy",),
            "GRIDGULP_GEMMA_ENABLED": ("gemma", "enabled"),
            "GRIDGULP_GEMMA_BASE_URL": ("gemma", "base_url"),
            "GRIDGULP_GEMMA_API_KEY": ("gemma", "api_key"),
            "GRIDGULP_GEMMA_MODEL_ID": ("gemma", "model_id"),
            "GRIDGULP_GEMMA_TIMEOUT_SECONDS": ("gemma", "timeout_seconds"),
            "GRIDGULP_GEMMA_MAX_RETRIES": ("gemma", "max_retries"),
            "GRIDGULP_GEMMA_MAX_TOKENS": ("gemma", "max_tokens"),
            "GRIDGULP_VISION_ENABLED": ("vision", "enabled"),
            "GRIDGULP_VISION_MODE": ("vision", "mode"),
            "GRIDGULP_VISION_MAX_PROPOSAL_CROPS": ("vision", "max_proposal_crops"),
            "GRIDGULP_VISION_CROP_MARGIN_CELLS": ("vision", "crop_margin_cells"),
            "GRIDGULP_VISION_DEBUG_IMAGE_DIR": ("vision", "debug_image_dir"),
            "GRIDGULP_CONFIDENCE_THRESHOLD": ("confidence_threshold",),
            "GRIDGULP_MAX_TABLES_PER_SHEET": ("max_tables_per_sheet",),
            "GRIDGULP_MIN_TABLE_ROWS": ("min_table_size", "rows"),
            "GRIDGULP_MIN_TABLE_COLS": ("min_table_size", "cols"),
            "GRIDGULP_DETECT_MERGED_CELLS": ("detect_merged_cells",),
            "GRIDGULP_ENABLE_MAGIKA": ("enable_magika",),
            "GRIDGULP_STRICT_FORMAT_CHECKING": ("strict_format_checking",),
            "GRIDGULP_FILE_DETECTION_BUFFER_SIZE": ("file_detection_buffer_size",),
            "GRIDGULP_MAX_FILE_SIZE_MB": ("max_file_size_mb",),
            "GRIDGULP_TIMEOUT_SECONDS": ("timeout_seconds",),
            "GRIDGULP_MAX_SHEETS": ("max_sheets",),
            "GRIDGULP_MAX_MEMORY_MB": ("max_memory_mb",),
            "GRIDGULP_CHUNK_SIZE": ("chunk_size",),
            "GRIDGULP_ENABLE_SIMPLE_CASE_DETECTION": ("enable_simple_case_detection",),
            "GRIDGULP_ENABLE_ISLAND_DETECTION": ("enable_island_detection",),
            "GRIDGULP_USE_EXCEL_METADATA": ("use_excel_metadata",),
            "GRIDGULP_ADAPTIVE_THRESHOLDS": ("adaptive_thresholds",),
            "GRIDGULP_MIN_TABLE_PERCENTAGE": ("min_table_percentage",),
            "GRIDGULP_PREFER_LARGE_TABLES": ("prefer_large_tables",),
            "GRIDGULP_EMPTY_ROW_TOLERANCE": ("empty_row_tolerance",),
            "GRIDGULP_USE_BORDER_DETECTION": ("use_border_detection",),
            "GRIDGULP_LOG_LEVEL": ("log_level",),
            "GRIDGULP_ENABLE_DEBUG": ("enable_debug",),
        }

        raw: dict[str, Any] = {}
        for env_name, path in mapping.items():
            if env_name not in env:
                continue

            raw_value = env[env_name]
            value: Any
            if env_name in {
                "GRIDGULP_GEMMA_ENABLED",
                "GRIDGULP_VISION_ENABLED",
                "GRIDGULP_DETECT_MERGED_CELLS",
                "GRIDGULP_ENABLE_MAGIKA",
                "GRIDGULP_STRICT_FORMAT_CHECKING",
                "GRIDGULP_ENABLE_SIMPLE_CASE_DETECTION",
                "GRIDGULP_ENABLE_ISLAND_DETECTION",
                "GRIDGULP_USE_EXCEL_METADATA",
                "GRIDGULP_ADAPTIVE_THRESHOLDS",
                "GRIDGULP_PREFER_LARGE_TABLES",
                "GRIDGULP_USE_BORDER_DETECTION",
                "GRIDGULP_ENABLE_DEBUG",
            }:
                value = cls._parse_bool(raw_value)
            elif env_name in {
                "GRIDGULP_GEMMA_TIMEOUT_SECONDS",
                "GRIDGULP_GEMMA_MAX_RETRIES",
                "GRIDGULP_GEMMA_MAX_TOKENS",
                "GRIDGULP_VISION_MAX_PROPOSAL_CROPS",
                "GRIDGULP_VISION_CROP_MARGIN_CELLS",
                "GRIDGULP_MAX_TABLES_PER_SHEET",
                "GRIDGULP_FILE_DETECTION_BUFFER_SIZE",
                "GRIDGULP_TIMEOUT_SECONDS",
                "GRIDGULP_MAX_SHEETS",
                "GRIDGULP_MAX_MEMORY_MB",
                "GRIDGULP_CHUNK_SIZE",
                "GRIDGULP_EMPTY_ROW_TOLERANCE",
                "GRIDGULP_MIN_TABLE_ROWS",
                "GRIDGULP_MIN_TABLE_COLS",
            }:
                value = int(raw_value)
            elif env_name in {
                "GRIDGULP_CONFIDENCE_THRESHOLD",
                "GRIDGULP_MAX_FILE_SIZE_MB",
                "GRIDGULP_MIN_TABLE_PERCENTAGE",
            }:
                value = float(raw_value)
            else:
                value = raw_value

            cls._set_nested(raw, path, value)

        min_rows = raw.get("min_table_size", {}).get("rows")
        min_cols = raw.get("min_table_size", {}).get("cols")
        if min_rows is not None or min_cols is not None:
            raw["min_table_size"] = (min_rows or 2, min_cols or 2)

        return raw

    @classmethod
    def _normalize_overrides(cls, overrides: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}

        for key, value in overrides.items():
            if value is None:
                continue

            if key == "gemma":
                normalized["gemma"] = value
                continue
            if key == "vision":
                normalized["vision"] = value
                continue

            if key.startswith("gemma_"):
                cls._set_nested(normalized, ("gemma", key.removeprefix("gemma_")), value)
                continue
            if key.startswith("vision_"):
                cls._set_nested(normalized, ("vision", key.removeprefix("vision_")), value)
                continue

            cls._set_nested(normalized, (key,), value)

        return normalized

    @staticmethod
    def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        current = data
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = value


# Type alias for backwards compatibility
GridGulpConfig = Config
