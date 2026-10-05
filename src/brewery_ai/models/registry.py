"""Loads the shipped model profiles and answers questions about them."""

from __future__ import annotations

import functools
import os
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import yaml

from brewery_ai.models.schema import Guidelines, ModelProfile, ResolvedModel, Variant

PROFILE_PACKAGE = "brewery_ai.models.profiles"


def _read_yaml(name: str) -> dict[str, Any]:
    text = resources.files(PROFILE_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


@functools.lru_cache(maxsize=1)
def _defaults() -> dict[str, Any]:
    return _read_yaml("defaults.yaml")


def default_guidelines(modality: str) -> Guidelines:
    data = dict(_defaults().get(modality, {}))
    objectives = data.pop("objectives", None)
    return Guidelines.model_validate(data).merged({"objectives": objectives} if objectives else None)


def user_profile_dirs() -> list[Path]:
    """Extra profile folders: ``$BREWERY_AI_PROFILES`` and ``<config dir>/profiles``."""
    from brewery_ai.paths import config_dir

    dirs = []
    env = os.environ.get("BREWERY_AI_PROFILES")
    if env:
        dirs.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p)
    dirs.append(config_dir() / "profiles")
    return [d for d in dirs if d.is_dir()]


@functools.lru_cache(maxsize=1)
def load_profiles() -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}
    for entry in sorted(resources.files(PROFILE_PACKAGE).iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".yaml") or entry.name == "defaults.yaml":
            continue
        profile = ModelProfile.model_validate(_read_yaml(entry.name))
        if profile.family in profiles:
            raise ValueError(f"duplicate model family {profile.family}")
        profiles[profile.family] = profile
    for folder in user_profile_dirs():
        for path in sorted(folder.glob("*.yaml")):
            profile = ModelProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            profiles[profile.family] = profile  # user profiles may replace shipped ones
    return profiles


def snapshot(model: ResolvedModel) -> dict[str, Any]:
    """Everything the worker needs to know about a model, embedded in job.yaml."""
    profile = model.profile.model_dump(mode="json")
    profile["variants"] = [model.variant.model_dump(mode="json")]
    return profile


def from_snapshot(data: dict[str, Any]) -> ResolvedModel:
    profile = ModelProfile.model_validate(data)
    return _resolve(profile, profile.variants[0])


def families() -> list[ModelProfile]:
    return list(load_profiles().values())


def _resolve(profile: ModelProfile, variant: Variant) -> ResolvedModel:
    guidelines = default_guidelines(profile.modality).merged(profile.guidelines).merged(variant.guidelines)
    return ResolvedModel(profile=profile, variant=variant, guidelines=guidelines)


def all_models() -> list[ResolvedModel]:
    return [_resolve(p, v) for p in load_profiles().values() for v in p.variants]


def find_model(model_id: str) -> ResolvedModel | None:
    """Look up a variant by exact Hub id (case-insensitive) or label."""
    needle = model_id.strip().lower()
    for model in all_models():
        if model.id.lower() == needle or model.variant.label.lower() == needle:
            return model
    return None


def get_model(model_id: str) -> ResolvedModel:
    model = find_model(model_id)
    if model is None:
        raise KeyError(
            f"{model_id!r} is not a supported base model in this Brewery version. "
            f"Supported: {', '.join(m.id for m in all_models())}"
        )
    return model


def search(
    *,
    modality: str | None = None,
    family: str | None = None,
    max_params_b: float | None = None,
    kind: str | None = None,
    include_base: bool = True,
) -> list[ResolvedModel]:
    out = []
    for model in all_models():
        v = model.variant
        if modality and model.modality != modality:
            continue
        if family and model.family != family:
            continue
        if max_params_b is not None and v.params_b > max_params_b:
            continue
        if kind and v.kind != kind:
            continue
        if not include_base and v.kind == "base":
            continue
        out.append(model)
    return out


def summarize(models: Iterable[ResolvedModel]) -> list[dict[str, Any]]:
    """Compact rows for the agent / CLI tables."""
    rows = []
    for m in models:
        v = m.variant
        rows.append(
            {
                "id": v.id,
                "label": v.label,
                "family": m.profile.display_name,
                "modality": m.modality,
                "kind": v.kind,
                "params_b": v.params_b,
                "active_params_b": v.active_params_b,
                "context": v.context,
                "license": m.license.id,
                "gated": v.gated,
                "recommended": v.recommended,
                "publisher": v.publisher or m.profile.vendor,
                "methods": [k for k in ("lora", "qlora", "full") if m.guidelines.for_method(k).allowed],
                "objectives": [o for o in ("sft", "cpt", "dpo") if m.guidelines.objective_allowed(o)],
            }
        )
    return rows
