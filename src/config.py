"""Project settings, loaded from ``config/settings.yaml`` (overridable by env).

Nothing here is a hyperparameter to tune -- these are the knobs that keep every
result reproducible: which solver is the default, the random seed, and the
tolerances the verification tests assert against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class Settings(BaseSettings):
    """Reproducibility knobs, read from YAML and the ``OPT101_`` env prefix."""

    model_config = SettingsConfigDict(env_prefix="OPT101_", extra="ignore")

    default_backend: str = "pulp"
    seed: int = 42
    float_tol: float = 1e-6
    shadow_price_eps: float = 1e-4


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """Load settings from YAML if present, falling back to the defaults above."""
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    return Settings(**data)


settings = load_settings()
