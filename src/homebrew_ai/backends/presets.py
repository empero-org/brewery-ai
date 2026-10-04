"""Backend presets ("which AI guides you") and per-model capability profiles.

Capability tunes how much structure the agent gets: strong models see every
tool and a principles-first prompt; small local models get phase-scoped tools,
step-by-step instructions and a smaller context budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from homebrew_ai.backends.anthropic_backend import AnthropicBackend
from homebrew_ai.backends.base import Backend, BackendError
from homebrew_ai.backends.openai_backend import OpenAICompatBackend
from homebrew_ai.settings import BackendSettings, get_api_key


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    provider: str
    base_url: str | None
    key_env: str | None
    key_url: str | None
    default_model: str | None
    suggested_models: tuple[tuple[str, str], ...] = ()  # (model id, description)
    needs_key: bool = True
    notes: str = ""


PRESETS: dict[str, Preset] = {
    "anthropic": Preset(
        "anthropic", "Claude (Anthropic)", "anthropic", None, "ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys",
        "claude-opus-5-5",
        (
            ("claude-opus-5-5", "Claude Opus 5.5 — best guidance ($4 / $20 per million tokens)"),
            ("claude-sonnet-5-5", "Claude Sonnet 5.5 — fast and capable ($2 / $10)"),
            ("claude-haiku-4-5", "Claude Haiku 4.5 — cheapest, simpler guidance ($1 / $5)"),
        ),
        notes="A typical brewing session uses well under a million tokens.",
    ),
    "openai": Preset("openai", "OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "https://platform.openai.com/api-keys", None),
    "openrouter": Preset(
        "openrouter", "OpenRouter (many models, one key)", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "https://openrouter.ai/keys",
        "xiaomi/mimo-v2.6-pro",
        (
            ("xiaomi/mimo-v2.6-pro", "Xiaomi MiMo-V2.6-Pro — strong and cheap ($0.44 / $0.87 per million tokens)"),
            ("xiaomi/mimo-v2.6-flash", "Xiaomi MiMo-V2.6-Flash — fastest and cheapest ($0.14 / $0.28)"),
            ("anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5 via OpenRouter ($2 / $10)"),
            ("anthropic/claude-opus-5.5", "Claude Opus 5.5 via OpenRouter ($4 / $20)"),
        ),
        notes="Many models behind one key. Pick one with tool-calling support.",
    ),
    "ollama": Preset(
        "ollama", "Ollama (local, free)", "openai", "http://localhost:11434/v1", None, None, None, needs_key=False,
        notes="Runs on your own computer. Use a model with tool calling and 14B+ parameters for good guidance.",
    ),
    "lmstudio": Preset("lmstudio", "LM Studio (local, free)", "openai", "http://localhost:1234/v1", None, None, None, needs_key=False),
    "custom": Preset("custom", "Other OpenAI-compatible server (vLLM, llama.cpp, ...)", "openai", None, "HOMEBREW_AI_API_KEY", None, None, needs_key=False),
}

_HIGH = re.compile(
    r"(claude-(opus|sonnet|fable|mythos))|(gpt-5(?![\w.-]*(mini|nano)))|(\bo[34]\b)|(gemini-[\d.]+-pro)|(deepseek-(v3|r1|v4))"
    r"|(qwen3[.\d]*-(235b|397b|480b|2\.4t|max))|(kimi-k2)|(glm-[45])|(grok-4)|(mimo-v2[.\d]*-pro)",
    re.I,
)
_SMALL = re.compile(r"(?<![\d.])(0\.\d+|[1-9](\.\d+)?)b\b|nano|tiny", re.I)


def capability_for(model: str, requested: str = "auto") -> str:
    if requested in ("high", "medium", "low"):
        return requested
    name = model.lower()
    if _HIGH.search(name):
        return "high"
    if _SMALL.search(name):
        return "low"
    return "medium"


@dataclass
class CapabilityProfile:
    name: str
    scoped_tools: bool
    history_budget_chars: int
    tool_result_chars: int
    max_steps: int
    extra_guidance: list[str] = field(default_factory=list)


CAPABILITIES = {
    "high": CapabilityProfile("high", False, 600_000, 12_000, 40),
    "medium": CapabilityProfile("medium", False, 200_000, 6_000, 30, ["Work through the workflow one phase at a time and check the project state before acting."]),
    "low": CapabilityProfile(
        "low", True, 48_000, 2_500, 16,
        [
            "Do exactly one step at a time.",
            "Always use a tool to look things up; never guess model names, dataset names or numbers.",
            "Keep replies under 120 words.",
        ],
    ),
}


def make_backend(cfg: BackendSettings) -> Backend:
    preset = PRESETS.get(cfg.preset)
    key = get_api_key(cfg)
    capability = capability_for(cfg.model, cfg.capability)
    provider = cfg.provider or (preset.provider if preset else "openai")
    if provider == "anthropic":
        if not key:
            raise BackendError("no Anthropic API key found; run `homebrew setup` or set ANTHROPIC_API_KEY")
        return AnthropicBackend(cfg.model, key, effort=cfg.effort, max_tokens=cfg.max_tokens, base_url=cfg.base_url, capability=capability)
    base_url = cfg.base_url or (preset.base_url if preset else None)
    if preset and preset.needs_key and not key:
        raise BackendError(f"no API key for {preset.label}; run `homebrew setup` or set {preset.key_env}")
    headers = {"HTTP-Referer": "https://empero.org", "X-Title": "Homebrew by Empero"} if cfg.preset == "openrouter" else None
    return OpenAICompatBackend(
        cfg.model, key, base_url, capability=capability, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        tool_mode=cfg.tool_mode, extra_headers=headers,
    )


def describe(cfg: BackendSettings) -> dict[str, Any]:
    preset = PRESETS.get(cfg.preset)
    return {"preset": preset.label if preset else cfg.preset, "model": cfg.model, "capability": capability_for(cfg.model, cfg.capability)}
