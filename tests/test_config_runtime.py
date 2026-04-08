"""Tests for Gemma-first configuration loading and policy enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridgulp.config import Config, DetectionStrategy


def test_config_precedence_toml_then_env_then_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config should merge TOML, then env, then explicit overrides."""
    config_path = tmp_path / "gridgulp.toml"
    config_path.write_text(
        "\n".join(
            [
                'detection_strategy = "heuristic_only"',
                "confidence_threshold = 0.61",
                "[gemma]",
                'base_url = "http://toml.example/v1"',
                'model_id = "gemma-4-toml"',
                "[vision]",
                'mode = "off"',
            ]
        )
    )

    monkeypatch.setenv("GRIDGULP_CONFIDENCE_THRESHOLD", "0.72")
    monkeypatch.setenv("GRIDGULP_GEMMA_MODEL_ID", "gemma-4-env")
    monkeypatch.setenv("GRIDGULP_VISION_MAX_PROPOSAL_CROPS", "3")

    config = Config.from_sources(
        config_path=config_path,
        overrides={
            "detection_strategy": "gemma_first",
            "gemma_base_url": "http://override.example/v1",
            "vision_mode": "fallback_only",
        },
    )

    assert config.detection_strategy == DetectionStrategy.GEMMA_FIRST
    assert config.confidence_threshold == 0.72
    assert config.gemma.base_url == "http://override.example/v1"
    assert config.gemma.model_id == "gemma-4-env"
    assert config.vision.mode.value == "fallback_only"
    assert config.vision.max_proposal_crops == 3


def test_validate_runtime_rejects_non_gemma_model() -> None:
    """Runtime validation should reject non-Gemma model ids."""
    config = Config.model_validate(
        {
            "detection_strategy": "heuristic_only",
            "gemma": {"model_id": "qwen-3", "base_url": "http://localhost:8080/v1"},
        }
    )

    with pytest.raises(ValueError, match="must point to a Gemma model"):
        config.validate_runtime()


def test_validate_runtime_requires_gemma_when_gemma_first() -> None:
    """Gemma-first mode should fail closed when required config is missing."""
    config = Config(detection_strategy=DetectionStrategy.GEMMA_FIRST)

    with pytest.raises(ValueError, match="Gemma base_url is required"):
        config.validate_runtime()


def test_config_supports_debug_image_override(tmp_path: Path) -> None:
    """Vision debug image output should round-trip through explicit overrides."""
    config = Config.from_sources(
        overrides={
            "detection_strategy": "gemma_first",
            "gemma_base_url": "http://localhost:11434/v1",
            "gemma_model_id": "gemma4:26b",
            "vision_debug_image_dir": tmp_path,
        }
    )

    assert config.vision.debug_image_dir == tmp_path
