"""User settings (which AI guides you) and stored credentials.

``settings.yaml`` holds preferences; API keys go to ``credentials.yaml``,
which is created readable only by the current user. Environment variables
always win over stored keys.
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from brewery_ai.paths import config_dir, write_private_file


class BackendSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    preset: str = "anthropic"
    provider: str = "anthropic"  # anthropic | openai (OpenAI-compatible)
    model: str = "claude-opus-5-5"
    base_url: str | None = None
    api_key_env: str | None = "ANTHROPIC_API_KEY"
    capability: str = "auto"  # auto | high | medium | low
    effort: str | None = "medium"  # Anthropic effort level
    temperature: float | None = None
    max_tokens: int | None = None
    tool_mode: str = "native"  # native | text
    context_budget_tokens: int | None = None


class UISettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    emoji: bool = True
    color: bool = True


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: BackendSettings | None = None
    ui: UISettings = Field(default_factory=UISettings)
    default_level: str | None = None

    @property
    def configured(self) -> bool:
        return self.backend is not None


def settings_path():
    return config_dir() / "settings.yaml"


def credentials_path():
    return config_dir() / "credentials.yaml"


def load_settings() -> Settings:
    path = settings_path()
    if not path.exists():
        return Settings()
    return Settings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def save_settings(settings: Settings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json", exclude_none=True), sort_keys=False), encoding="utf-8")


def _load_credentials() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_api_key(backend: BackendSettings) -> str | None:
    if backend.api_key_env and os.environ.get(backend.api_key_env):
        return os.environ[backend.api_key_env]
    return _load_credentials().get("api_keys", {}).get(backend.preset)


def store_api_key(preset: str, key: str) -> None:
    creds = _load_credentials()
    creds.setdefault("api_keys", {})[preset] = key
    write_private_file(credentials_path(), yaml.safe_dump(creds, sort_keys=False))


def forget_api_key(preset: str) -> None:
    creds = _load_credentials()
    if creds.get("api_keys", {}).pop(preset, None) is not None:
        write_private_file(credentials_path(), yaml.safe_dump(creds, sort_keys=False))
