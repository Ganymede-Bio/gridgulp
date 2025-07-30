"""Tests for the Config class and configuration management."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from gridgulp.config import Config


class TestConfig:
    """Test the Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        # Check default values
        assert config.confidence_threshold == 0.7
        assert config.max_file_size_mb == 2000.0
        assert config.enable_simple_case_detection is True
        assert config.enable_island_detection is True
        assert config.log_level == "INFO"
        assert config.log_file is None
        assert config.min_table_size == (2, 2)
        assert config.max_tables_per_sheet == 50

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = Config(
            confidence_threshold=0.9,
            max_file_size_mb=100,
            enable_simple_case_detection=False,
            log_level="DEBUG",
        )

        assert config.confidence_threshold == 0.9
        assert config.max_file_size_mb == 100
        assert config.enable_simple_case_detection is False
        assert config.log_level == "DEBUG"
        # Other values should remain default
        assert config.enable_island_detection is True

    def test_from_env(self):
        """Test loading config from environment variables."""
        env_vars = {
            "GRIDGULP_MAX_FILE_SIZE_MB": "1000",
            "GRIDGULP_LOG_LEVEL": "DEBUG",
            "GRIDGULP_ENABLE_SIMPLE_CASE_DETECTION": "false",
            "GRIDGULP_ENABLE_ISLAND_DETECTION": "true",
            "OTHER_ENV_VAR": "ignored",
        }

        with patch.dict(os.environ, env_vars):
            config = Config.from_env()

        # Check that env vars were applied
        assert config.max_file_size_mb == 1000
        assert config.log_level == "DEBUG"
        assert config.enable_simple_case_detection is False
        assert config.enable_island_detection is True

    def test_from_env_type_conversion(self):
        """Test environment variable type conversion."""
        env_vars = {
            "GRIDGULP_MAX_FILE_SIZE_MB": "3000",  # Float
            "GRIDGULP_ENABLE_ISLAND_DETECTION": "True",  # Bool (capital)
            "GRIDGULP_ENABLE_SIMPLE_CASE_DETECTION": "true",  # Bool (lowercase)
            "GRIDGULP_LOG_LEVEL": "ERROR",  # String
        }

        with patch.dict(os.environ, env_vars):
            config = Config.from_env()

        assert config.max_file_size_mb == 3000
        assert isinstance(config.max_file_size_mb, float)
        assert config.enable_island_detection is True
        assert config.enable_simple_case_detection is True
        assert config.log_level == "ERROR"

    def test_validation_confidence_threshold(self):
        """Test validation of confidence threshold."""
        # Valid values
        config = Config(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0

        config = Config(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

        # Invalid values should raise ValidationError from Pydantic
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            Config(confidence_threshold=1.1)

    def test_validation_file_size(self):
        """Test validation of max file size."""
        from pydantic import ValidationError

        # Valid values
        config = Config(max_file_size_mb=1)
        assert config.max_file_size_mb == 1

        config = Config(max_file_size_mb=0.1)  # Minimum allowed
        assert config.max_file_size_mb == 0.1

        # Invalid values
        with pytest.raises(ValidationError):
            Config(max_file_size_mb=0.05)  # Below minimum

        with pytest.raises(ValidationError):
            Config(max_file_size_mb=-10)

    def test_validation_table_limits(self):
        """Test validation of table size limits."""
        from pydantic import ValidationError

        # Valid values
        config = Config(min_table_size=(1, 1))
        assert config.min_table_size == (1, 1)

        config = Config(max_tables_per_sheet=1)
        assert config.max_tables_per_sheet == 1

        # Invalid values
        with pytest.raises(ValidationError):
            Config(max_tables_per_sheet=0)

    def test_config_fields(self):
        """Test that all expected fields exist."""
        config = Config()

        # Check important fields exist
        assert hasattr(config, "confidence_threshold")
        assert hasattr(config, "max_file_size_mb")
        assert hasattr(config, "enable_simple_case_detection")
        assert hasattr(config, "enable_island_detection")
        assert hasattr(config, "log_level")
        assert hasattr(config, "enable_magika")
        assert hasattr(config, "timeout_seconds")
