"""First-run setup: which AI guides you, your level, and the project folder."""

from __future__ import annotations

from pathlib import Path

from homebrew_ai.agent.levels import LEVELS
from homebrew_ai.backends.base import BackendError
from homebrew_ai.backends.presets import PRESETS, capability_for, make_backend
from homebrew_ai.project import Project, slugify
from homebrew_ai.settings import BackendSettings, Settings, get_api_key, save_settings, store_api_key
from homebrew_ai.ui.console import ConsoleUI


def setup_backend(ui: ConsoleUI, settings: Settings) -> Settings:
    ui.info(
        "Homebrew is guided by an AI model of your choice (the 'brewmaster'). It never sees your API keys or tokens: "
        "those are stored on this computer only.",
        title="Choose your guide",
    )
    options = [{"label": p.label, "description": p.notes} for p in PRESETS.values()]
    label = ui.choose("Which AI should guide you?", options, allow_other=False)
    preset = next(p for p in PRESETS.values() if p.label == label)
    cfg = BackendSettings(preset=preset.key, provider=preset.provider, base_url=preset.base_url, api_key_env=preset.key_env, model=preset.default_model or "")

    if preset.key == "custom":
        cfg.base_url = ui.text("Server URL (OpenAI-compatible, ends in /v1):", default="http://localhost:8000/v1")

    key = get_api_key(cfg) if preset.key_env else None
    if preset.needs_key and not key:
        if preset.key_url:
            ui.console.print(f"Create a key here: [key]{preset.key_url}[/]")
        key = ui.secret(f"Paste your {preset.label} API key (hidden)").strip()
        if not key:
            raise SystemExit("No key entered; run `homebrew setup` again when you have one.")
        store_api_key(preset.key, key)
    elif preset.key == "custom" and ui.confirm("Does this server need an API key?", default=False):
        store_api_key(preset.key, ui.secret("API key (hidden)").strip())

    if preset.suggested_models:
        options = [{"label": m, "description": d} for m, d in preset.suggested_models]
        cfg.model = ui.choose("Which model?", options, allow_other=True)
    else:
        models: list[str] = []
        try:
            from homebrew_ai.backends.openai_backend import OpenAICompatBackend

            models = OpenAICompatBackend("probe", get_api_key(cfg), cfg.base_url).list_models()
        except Exception:
            models = []
        if models and len(models) <= 60:
            cfg.model = ui.choose("Which model?", [{"label": m} for m in models], allow_other=True)
        else:
            cfg.model = ui.text("Model name (as the provider spells it):")
    cfg.capability = "auto"
    tier = capability_for(cfg.model)
    if tier == "low":
        ui.warn("Small models can get lost in long multi-step tasks. Homebrew will guide this one step by step; a larger model (14B+) works better.")

    with ui.activity("Testing the connection"):
        try:
            reply = make_backend(cfg).check()
        except BackendError as exc:
            ui.error(f"The connection test failed: {exc}")
            if not ui.confirm("Save these settings anyway?", default=False):
                raise SystemExit(1)
            reply = ""
    if reply:
        ui.console.print(f"[ok]✓ connected to {cfg.model}[/]")
    settings.backend = cfg
    save_settings(settings)
    return settings


def choose_level(ui: ConsoleUI, current: str | None = None) -> str:
    options = [{"label": lv.title, "description": lv.blurb} for lv in LEVELS.values()]
    label = ui.choose("How familiar are you with training AI models? (Homebrew adjusts how it talks to you)", options, allow_other=False)
    return next(lv.key for lv in LEVELS.values() if lv.title == label)


def create_project(ui: ConsoleUI, base_dir: Path, name: str | None = None) -> Project:
    if not name:
        name = ui.text("What should we call your project? (a short name, e.g. pirate-bot)", default="my-first-brew")
    folder = base_dir / slugify(name)
    if folder.exists() and any(folder.iterdir()) and not (folder / "homebrew.yaml").exists():
        raise SystemExit(f"{folder} already exists and is not a Homebrew project; pick another name.")
    project = Project.create(folder, name)
    ui.console.print(f"[ok]✓ project folder:[/] {folder}")
    return project
