from pathlib import Path

import pytest

from app.configs.settings import AppSettings, ConfigurationError, InferenceSettings


def test_default_settings_validate() -> None:
    settings = AppSettings()
    settings.validate()
    assert settings.inference.device == "auto"


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="batch_size"):
        InferenceSettings(batch_size=0).validate()


def test_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = AppSettings()
    original.to_json(path)
    loaded = AppSettings.from_json(path)
    assert loaded == original
