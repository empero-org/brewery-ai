"""Schema for the shipped model profiles (``profiles/*.yaml``).

A profile describes one model family: how to load it, how its chat template
works, which LoRA targets make sense, and — most importantly — the
hyperparameter guidelines the agent must stay inside. Guidelines are merged
from ``defaults.yaml`` → family → variant, so a family only spells out what is
different.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Method = Literal["lora", "qlora", "full"]
Objective = Literal["sft", "cpt", "dpo"]
Modality = Literal["text", "image"]
OBJECTIVES = ("sft", "cpt", "dpo")


class Range(BaseModel):
    """A numeric knob: hard bounds, a default and an optional recommended band."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    default: float | int | None = None
    recommended: tuple[float, float] | None = None
    choices: list[float | int] | None = None

    def check(self, value: float, name: str) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        if self.choices is not None and value not in self.choices:
            errors.append(f"{name}={value} is not one of {self.choices}")
        if self.min is not None and value < self.min:
            errors.append(f"{name}={value} is below the allowed minimum {self.min}")
        if self.max is not None and value > self.max:
            errors.append(f"{name}={value} is above the allowed maximum {self.max}")
        if not errors and self.recommended is not None:
            lo, hi = self.recommended
            if value < lo or value > hi:
                warnings.append(f"{name}={value} is outside the recommended range {lo}–{hi}")
        return errors, warnings


class MethodGuidelines(BaseModel):
    """Hyperparameter guardrails for one training method."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = True
    learning_rate: Range | None = None
    rank: Range | None = None
    alpha_ratio: Range | None = None
    dropout: Range | None = None
    epochs: Range | None = None
    max_steps: Range | None = None
    warmup_ratio: Range | None = None
    weight_decay: Range | None = None
    max_grad_norm: Range | None = None
    effective_batch: Range | None = None
    max_seq_len: Range | None = None
    resolution: Range | None = None
    caption_dropout: Range | None = None
    beta: Range | None = None
    optimizers: list[str] | None = None
    default_optimizer: str | None = None
    schedulers: list[str] | None = None
    default_scheduler: str | None = None
    notes: list[str] = Field(default_factory=list)

    def merged(self, override: "MethodGuidelines | dict[str, Any] | None") -> "MethodGuidelines":
        if override is None:
            return self
        data = override if isinstance(override, dict) else override.model_dump(exclude_unset=True)
        base = self.model_dump()
        for key, value in data.items():
            if key == "notes":
                base["notes"] = [*base.get("notes", []), *value]
            elif isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            else:
                base[key] = value
        return MethodGuidelines.model_validate(base)


class Guidelines(BaseModel):
    """Per-method guardrails, with optional per-objective overrides (``objectives.dpo.lora`` ...)."""

    model_config = ConfigDict(extra="forbid")

    lora: MethodGuidelines = Field(default_factory=MethodGuidelines)
    qlora: MethodGuidelines = Field(default_factory=MethodGuidelines)
    full: MethodGuidelines = Field(default_factory=MethodGuidelines)
    objectives: dict[str, dict[str, MethodGuidelines]] = Field(default_factory=dict)

    def for_method(self, method: str, objective: str = "sft") -> MethodGuidelines:
        base: MethodGuidelines = getattr(self, method)
        override = self.objectives.get(objective, {}).get(method)
        return base.merged(override) if override is not None else base

    def objective_allowed(self, objective: str) -> bool:
        if objective == "sft":
            return True
        return objective in self.objectives and any(g.allowed for g in self.objectives[objective].values()) if objective in self.objectives else False

    def merged(self, override: dict[str, Any] | None) -> "Guidelines":
        if not override:
            return self
        objectives = {obj: dict(methods) for obj, methods in self.objectives.items()}
        for obj, methods in (override.get("objectives") or {}).items():
            target = objectives.setdefault(obj, {})
            for method, data in (methods or {}).items():
                patch = MethodGuidelines.model_validate(data)
                target[method] = target[method].merged(patch) if method in target else patch
        return Guidelines(
            lora=self.lora.merged(override.get("lora")),
            qlora=self.qlora.merged(override.get("qlora")),
            full=self.full.merged(override.get("full")),
            objectives=objectives,
        )


class LicenseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    url: str
    hub_id: str | None = None  # value for the model card `license:` field
    name_prefix: str | None = None  # e.g. Llama requires derivative names to start with "Llama"
    attribution: str | None = None  # text that must be displayed
    notice: str | None = None  # notice to include in the README
    copy_files: list[str] = Field(default_factory=list)  # files from the base repo to ship along
    noncommercial: bool = False
    name_rules: str | None = None  # human-readable naming restrictions for derivatives


class Variant(BaseModel):
    """One concrete checkpoint on the Hugging Face Hub."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: Literal["instruct", "base", "thinking", "image"] = "instruct"
    params_b: float
    active_params_b: float | None = None
    layers: int | None = None
    hidden: int | None = None
    intermediate: int | None = None
    heads: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    vocab: int | None = None
    context: int | None = None
    tied_embeddings: bool = False
    experts: int | None = None
    experts_per_token: int | None = None
    expert_intermediate: int | None = None
    linear_attention_layers: int | None = None
    text_encoder_params_b: float | None = None
    vae_params_b: float | None = None
    gated: bool = False
    recommended: bool = False
    publisher: str | None = None
    license: LicenseInfo | None = None
    min_transformers: str | None = None
    fix_untrained_tokens: bool = False
    chat_template_from: str | None = None
    chat_format: dict[str, Any] | None = None
    guidelines: dict[str, Any] | None = None
    sampling: dict[str, dict[str, float]] | None = None
    lora_targets: dict[str, list[str]] | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def is_moe(self) -> bool:
        return bool(self.experts)


class ModelProfile(BaseModel):
    """A model family as shipped in ``profiles/<family>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    family: str
    display_name: str
    vendor: str
    modality: Modality = "text"
    summary: str = ""
    license: LicenseInfo
    requirements: dict[str, Any] = Field(default_factory=dict)
    loading: dict[str, Any] = Field(default_factory=dict)
    chat_format: dict[str, Any] | None = None
    stop_tokens: list[str] = Field(default_factory=list)
    lora_targets: dict[str, list[str]] = Field(default_factory=dict)
    sampling: dict[str, dict[str, float]] = Field(default_factory=dict)
    guidelines: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    variants: list[Variant]

    @model_validator(mode="after")
    def _check(self) -> "ModelProfile":
        if self.modality == "text" and not self.chat_format:
            raise ValueError(f"text family {self.family} needs a chat_format")
        if "default" not in self.lora_targets:
            raise ValueError(f"family {self.family} needs lora_targets.default")
        return self


class ResolvedModel(BaseModel):
    """A variant with its family settings and guidelines merged in."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: ModelProfile
    variant: Variant
    guidelines: Guidelines

    @property
    def id(self) -> str:
        return self.variant.id

    @property
    def family(self) -> str:
        return self.profile.family

    @property
    def modality(self) -> str:
        return self.profile.modality

    @property
    def license(self) -> LicenseInfo:
        return self.variant.license or self.profile.license

    @property
    def min_transformers(self) -> str | None:
        return self.variant.min_transformers or self.profile.requirements.get("transformers")

    def chat_format_dict(self) -> dict[str, Any]:
        return {**(self.profile.chat_format or {}), **(self.variant.chat_format or {})}

    def lora_targets(self, preset: str = "default") -> list[str]:
        targets = {**self.profile.lora_targets, **(self.variant.lora_targets or {})}
        if preset not in targets:
            raise KeyError(f"unknown LoRA target preset {preset!r}; options: {', '.join(targets)}")
        return targets[preset]

    def lora_presets(self) -> list[str]:
        return sorted({**self.profile.lora_targets, **(self.variant.lora_targets or {})})

    def sampling(self, mode: str = "default") -> dict[str, float | int]:
        merged = {**self.profile.sampling, **(self.variant.sampling or {})}
        params: dict[str, float | int] = dict(merged.get(mode) or merged.get("default") or {"temperature": 0.7, "top_p": 0.9})
        if "top_k" in params:
            params["top_k"] = int(params["top_k"])  # generate() rejects float top_k
        return params

    def notes(self) -> list[str]:
        return [*self.profile.notes, *self.variant.notes]
