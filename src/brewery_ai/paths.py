"""Filesystem locations used by the control plane."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_FILE = "brewery.yaml"
SESSION_DIR = ".brewery"
# Brewery was called Homebrew until 0.2.0; projects and settings made then are picked up and renamed.
LEGACY_PROJECT_FILE = "homebrew.yaml"
LEGACY_SESSION_DIR = ".homebrew"


def config_dir() -> Path:
    """Per-user config directory (settings, credentials)."""
    override = os.environ.get("BREWERY_AI_HOME") or os.environ.get("HOMEBREW_AI_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path, legacy = base / "brewery-ai", base / "homebrew-ai"
    if not path.exists() and legacy.is_dir():
        try:
            legacy.rename(path)  # keep the guiding-AI setup and keys from the Homebrew days
        except OSError:
            return legacy
    return path


def migrate_project(root: Path) -> None:
    """Rename a Homebrew-era project (homebrew.yaml, .homebrew/) to Brewery's names."""
    old, new = root / LEGACY_PROJECT_FILE, root / PROJECT_FILE
    if old.is_file() and not new.exists():
        old.rename(new)
    old_dir, new_dir = root / LEGACY_SESSION_DIR, root / SESSION_DIR
    if old_dir.is_dir() and not new_dir.exists():
        old_dir.rename(new_dir)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for a ``brewery.yaml`` (or a Homebrew-era ``homebrew.yaml``)."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_FILE).is_file() or (candidate / LEGACY_PROJECT_FILE).is_file():
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
