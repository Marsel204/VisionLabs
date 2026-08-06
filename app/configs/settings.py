"""Typed application configuration with safe filesystem defaults."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.services.active_learning.active_learning_models import ActiveLearningConfig
from app.services.fusion.fusion_models import FusionConfig


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Filesystem locations used by the application."""

    dataset_root: Path = Path.home() / "TrafficAnnotator" / "datasets"
    cache_root: Path = Path.home() / ".cache" / "traffic-annotator"
    log_root: Path = Path.home() / ".local" / "state" / "traffic-annotator" / "logs"


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Runtime settings shared by model services."""

    device: str = "auto"
    batch_size: int = 4
    confidence_threshold: float = 0.25

    def validate(self) -> None:
        """Validate values that would otherwise fail during inference."""
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be greater than zero")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError("confidence_threshold must be between 0 and 1")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigurationError(f"unsupported inference device: {self.device}")


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Complete immutable application settings."""

    paths: PathSettings = field(default_factory=PathSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    log_level: str = "INFO"

    def validate(self) -> None:
        """Validate all nested settings."""
        self.inference.validate()
        self.fusion.validate()
        self.active_learning.validate()
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(f"unsupported log level: {self.log_level}")

    def ensure_directories(self) -> None:
        """Create application-owned directories when the application starts."""
        for path in (self.paths.cache_root, self.paths.log_root):
            path.expanduser().mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(cls, path: Path) -> AppSettings:
        """Load settings from JSON, retaining defaults for omitted values."""
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"could not read configuration {path}: {error}") from error

        paths_payload = payload.get("paths", {})
        inference_payload = payload.get("inference", {})
        fusion_payload = payload.get("fusion", {})
        active_learning_payload = payload.get("active_learning", {})
        settings = cls(
            paths=PathSettings(**{key: Path(value) for key, value in paths_payload.items()}),
            inference=InferenceSettings(**inference_payload),
            fusion=FusionConfig(**fusion_payload),
            active_learning=ActiveLearningConfig(
                **{
                    **active_learning_payload,
                    "cache_path": Path(active_learning_payload["cache_path"])
                    if "cache_path" in active_learning_payload
                    else ActiveLearningConfig().cache_path,
                }
            ),
            log_level=str(payload.get("log_level", "INFO")).upper(),
        )
        settings.validate()
        return settings

    @classmethod
    def from_active_learning_yaml(
        cls, path: Path, base: AppSettings | None = None
    ) -> AppSettings:
        """Load active-learning values from YAML while retaining other settings."""
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(
                f"could not read active-learning configuration {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ConfigurationError("active-learning YAML root must be a mapping")
        settings = base or cls()
        values = dict(payload)
        if "cache_path" in values:
            values["cache_path"] = Path(values["cache_path"])
        updated = cls(
            paths=settings.paths,
            inference=settings.inference,
            fusion=settings.fusion,
            active_learning=ActiveLearningConfig(**values),
            log_level=settings.log_level,
        )
        updated.validate()
        return updated

    @classmethod
    def from_fusion_yaml(cls, path: Path, base: AppSettings | None = None) -> AppSettings:
        """Load fusion values from YAML while retaining other application settings."""
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(
                f"could not read fusion configuration {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ConfigurationError("fusion YAML root must be a mapping")
        settings = base or cls()
        updated = cls(
            paths=settings.paths,
            inference=settings.inference,
            fusion=FusionConfig(**payload),
            log_level=settings.log_level,
        )
        updated.validate()
        return updated

    def to_json(self, path: Path) -> None:
        """Persist settings as readable JSON."""
        self.validate()
        payload = asdict(self)
        payload["paths"] = {key: str(value) for key, value in payload["paths"].items()}
        payload["active_learning"]["cache_path"] = str(payload["active_learning"]["cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load explicit configuration or construct validated defaults."""
    path_value = config_path or (
        Path(os.environ["TRAFFIC_ANNOTATOR_CONFIG"])
        if os.getenv("TRAFFIC_ANNOTATOR_CONFIG")
        else None
    )
    settings = AppSettings.from_json(path_value) if path_value else AppSettings()
    settings.validate()
    return settings
