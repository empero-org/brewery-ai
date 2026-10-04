"""Filesystem locations used by the control plane."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_FILE = "homebrew.yaml"
SESSION_DIR = ".homebrew"


def config_dir() -> Path:
    """Per-user config directory (settings, credentials)."""
    override = os.environ.get("HOMEBREW_AI_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "homebrew-ai"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "homebrew-ai"


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for a ``homebrew.yaml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_FILE).is_file():
            return candidate
    return None


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return path


def write_private_file(path: Path, text: str) -> None:
    """Write a file only the current user can read (API keys live here)."""
    ensure_private_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
