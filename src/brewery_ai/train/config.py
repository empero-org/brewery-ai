"""Training job description (``job.yaml``) and the guideline checks behind it.

The agent never writes hyperparameters directly. It calls :func:`build_job`
with the base model, data statistics, hardware and a few optional overrides;
this module fills everything else from the model's guidelines, sizes batch and
accumulation to the GPU, and refuses values outside the hard bounds unless the
user explicitly granted an expert override.

Imported by both the control plane and the worker, so keep it light.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import secrets
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from brewery_ai import __version__

Method = Literal["lora", "qlora", "full"]
Objective = Literal["sft", "cpt", "dpo"]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: str
    eval: str | None = None
    num_train: int | None = None
    num_eval: int | None = None
    max_seq_len: int = 2048
    overflow: Literal["drop", "truncate"] = "drop"
    reasoning: Literal["auto", "native", "inline", "drop"] = "auto"


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    targets: str | list[str] = "default"
    use_rslora: bool = False
    use_dora: bool = False
    train_experts: bool = False
    train_embeddings: bool = False


class OptimSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float
    epochs: float = 2.0
    max_steps: int | None = None
    micro_batch_size: int = 1
    grad_accum: int = 16
    optimizer: str = "adamw_torch_fused"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    gradient_checkpointing: bool = True
    attn_implementation: str | None = None
    seed: int = 42
    logging_steps: int = 5
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 2
    num_gpus: int = 1


class DPOSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = 0.1
    label_smoothing: float = 0.0  # >0 = conservative DPO for noisy preferences
    sft_weight: float = 0.0  # adds an NLL term on the chosen answers (RPO-style)


class InitFrom(BaseModel):
    """Start from the result of an earlier stage instead of the base model."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["adapter", "full"] = "adapter"


class ImageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: int = 1024
    caption_dropout: float = 0.05
    trigger_word: str | None = None
    timestep_sampling: Literal["uniform", "shifted_logit_normal", "logit_normal"] = "uniform"
    sample_prompts: list[str] = Field(default_factory=list)  # previews during training (empty: a few training captions)
    sample_every: int | None = None  # steps between previews; None = with every checkpoint, 0 = off
    sample_steps: int = 30  # denoising steps per preview image
    cache_dtype: Literal["bf16", "fp16"] = "bf16"


class TrainJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brewery_version: str = Field(default=__version__, validation_alias=AliasChoices("brewery_version", "homebrew_version"))
    job_id: str
    project: str
    created: str = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    base_model: str
    family: str
    modality: Literal["text", "image"] = "text"
    objective: Objective = "sft"
    method: Method
    init_from: InitFrom | None = None
    stage: int | None = None
    data: DataConfig
    lora: LoraSettings | None = None
    dpo: DPOSettings | None = None
    optim: OptimSettings
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    image: ImageSettings | None = None
    expert_override: bool = False
    notes: list[str] = Field(default_factory=list)
    model_profile: dict[str, Any] | None = None  # snapshot of the model profile (see registry.snapshot)

    def resolved_model(self) -> Any:
        from brewery_ai.models.registry import from_snapshot, get_model

        if self.model_profile:
            return from_snapshot(self.model_profile)
        return get_model(self.base_model)

    @property
    def effective_batch(self) -> int:
        return self.optim.micro_batch_size * self.optim.grad_accum * max(self.runtime.num_gpus, 1)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainJob":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def make_job_id(project: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-")[:24] or "brew"
    return f"{stamp}-{slug}-{secrets.token_hex(3)}"


# --------------------------------------------------------------------------- #
# validation against guidelines
# --------------------------------------------------------------------------- #


def validate_job(job: TrainJob, model: Any) -> tuple[list[str], list[str]]:
    """Check a job against the model's guidelines. Returns ``(errors, warnings)``.

    ``model`` is a :class:`brewery_ai.models.schema.ResolvedModel`.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if job.objective != "sft" and not model.guidelines.objective_allowed(job.objective):
        errors.append(f"objective '{job.objective}' is not supported for {model.id}")
        return errors, warnings
    g = model.guidelines.for_method(job.method, job.objective)
    if not g.allowed:
        reason = " ".join(g.notes) or "not supported for this model"
        errors.append(f"method '{job.method}' is not allowed for {model.id}: {reason}")
        return errors, warnings
    if job.modality != model.modality:
        errors.append(f"{model.id} is a {model.modality} model but the job is for {job.modality}")

    def check(rng, value, name):
        if rng is None or value is None:
            return
        e, w = rng.check(value, name)
        errors.extend(e)
        warnings.extend(w)

    o = job.optim
    check(g.learning_rate, o.learning_rate, "learning_rate")
    check(g.warmup_ratio, o.warmup_ratio, "warmup_ratio")
    check(g.weight_decay, o.weight_decay, "weight_decay")
    check(g.max_grad_norm, o.max_grad_norm, "max_grad_norm")
    check(g.effective_batch, job.effective_batch, "effective_batch")
    if model.modality == "text":
        check(g.epochs, o.epochs, "epochs")
        check(g.max_seq_len, job.data.max_seq_len, "max_seq_len")
        ctx = model.variant.context
        if ctx and job.data.max_seq_len > ctx:
            errors.append(f"max_seq_len={job.data.max_seq_len} exceeds the model's context window ({ctx})")
    else:
        check(g.max_steps, o.max_steps, "max_steps")
        if job.image is not None:
            check(g.resolution, job.image.resolution, "resolution")
            check(g.caption_dropout, job.image.caption_dropout, "caption_dropout")
    if g.optimizers and o.optimizer not in g.optimizers:
        errors.append(f"optimizer '{o.optimizer}' is not one of {g.optimizers}")
    if g.schedulers and o.scheduler not in g.schedulers:
        errors.append(f"scheduler '{o.scheduler}' is not one of {g.schedulers}")

    if job.method in ("lora", "qlora"):
        if job.lora is None:
            errors.append("LoRA settings are missing")
        else:
            check(g.rank, job.lora.rank, "lora.rank")
            check(g.dropout, job.lora.dropout, "lora.dropout")
            if g.alpha_ratio is not None and job.lora.rank:
                check(g.alpha_ratio, round(job.lora.alpha / job.lora.rank, 3), "lora.alpha/rank")
            if isinstance(job.lora.targets, str):
                if job.lora.targets not in model.lora_presets():
                    errors.append(f"unknown LoRA target preset '{job.lora.targets}'; options: {model.lora_presets()}")
            if job.lora.train_experts and not model.variant.is_moe:
                errors.append("train_experts only applies to mixture-of-experts models")
            if job.lora.train_experts and (job.lora.use_dora or job.lora.dropout):
                errors.append("expert LoRA does not support DoRA or dropout")
    if job.objective == "dpo":
        if job.dpo is None:
            errors.append("DPO settings are missing")
        else:
            check(g.beta, job.dpo.beta, "dpo.beta")
            if not 0 <= job.dpo.label_smoothing < 0.5:
                errors.append("dpo.label_smoothing must be in [0, 0.5)")
    if job.runtime.precision == "fp16" and model.profile.loading.get("bf16_required"):
        errors.append(f"{model.profile.display_name} overflows in fp16; use a GPU with bf16 support")

    if job.expert_override and errors:
        warnings.extend(f"(expert override) {e}" for e in errors if "not allowed" not in e and "context window" not in e)
        errors = [e for e in errors if "not allowed" in e or "context window" in e]
    return errors, warnings


# --------------------------------------------------------------------------- #
# building a job from guidelines
# --------------------------------------------------------------------------- #


def _default(rng, fallback):
    return fallback if rng is None or rng.default is None else rng.default


def total_steps(job: TrainJob) -> int:
    if job.optim.max_steps:
        return int(job.optim.max_steps)
    n = job.data.num_train or 0
    per_step = max(job.effective_batch, 1)
    return max(1, math.ceil(n * job.optim.epochs / per_step))


def build_job(
    *,
    model: Any,
    project: str,
    method: str,
    objective: str = "sft",
    init_from: dict[str, Any] | None = None,
    stage: int | None = None,
    train_path: str,
    eval_path: str | None,
    num_train: int,
    num_eval: int | None = None,
    vram_gb: float | None = None,
    num_gpus: int = 1,
    bf16: bool = True,
    p95_tokens: int | None = None,
    overrides: dict[str, Any] | None = None,
    expert_override: bool = False,
) -> tuple[TrainJob, dict[str, Any]]:
    """Fill a :class:`TrainJob` from guidelines + overrides and size it to the hardware.

    Returns the job and a report with ``errors``, ``warnings``, ``fit`` and
    ``steps`` for the agent to present.
    """
    from brewery_ai.hardware.estimate import TrainShape, autofit
    from brewery_ai.models.registry import snapshot

    overrides = dict(overrides or {})
    g = model.guidelines.for_method(method, objective)
    image = model.modality == "image"
    report: dict[str, Any] = {"warnings": [], "errors": []}
    if image and objective != "sft":
        report["errors"].append("image models only support LoRA fine-tuning (objective 'sft')")
    if image and num_gpus > 1:  # the image trainer is single-process; extra GPUs would sit idle or collide
        report["warnings"].append(f"image LoRA training uses one GPU; the other {num_gpus - 1} stay idle")
        num_gpus = 1

    # sequence length: cover ~95% of samples, rounded up to a power-of-two-ish bucket
    if image:
        seq_len = int(overrides.pop("resolution", _default(g.resolution, 1024)))
    else:
        default_len = int(_default(g.max_seq_len, 2048))
        if p95_tokens:
            bucket = 256
            while bucket < p95_tokens and bucket < 32768:
                bucket *= 2
            default_len = max(512, bucket)
            if model.variant.context:
                default_len = min(default_len, model.variant.context)
        seq_len = int(overrides.pop("max_seq_len", default_len))

    rank = int(overrides.pop("lora_rank", _default(g.rank, 16)))
    alpha_ratio = float(_default(g.alpha_ratio, 2))
    alpha = int(overrides.pop("lora_alpha", max(1, round(rank * alpha_ratio))))
    optimizer = overrides.pop("optimizer", g.default_optimizer or "adamw_torch_fused")
    if not bf16 and optimizer == "adamw_torch_fused":
        optimizer = "adamw_torch"
    effective = int(overrides.pop("effective_batch", _default(g.effective_batch, 16)))
    if not image and num_train and effective > max(num_train // 4, 1):
        effective = max(1, min(effective, max(num_train // 4, 1)))
        report["warnings"].append(f"small dataset: effective batch lowered to {effective} so training still takes several steps")

    lora_targets = overrides.pop("lora_targets", "default")
    shape = TrainShape(
        method=method,
        objective=objective,
        seq_len=seq_len,
        lora_rank=rank,
        lora_preset=lora_targets if isinstance(lora_targets, str) else "default",
        optimizer=optimizer,
    )
    micro = overrides.pop("micro_batch_size", None)
    accum = overrides.pop("grad_accum", None)
    fit = None
    if vram_gb:
        fit = autofit(model, shape, vram_gb, effective, num_gpus)
        report["fit"] = fit.to_dict()
        if not fit.ok:
            report["errors"].append(
                f"estimated {fit.memory.total_gb:.0f} GB needed but the GPU has {vram_gb:.0f} GB; options: " + "; ".join(fit.suggestions)
            )
    if micro is None:
        micro = fit.micro_batch if fit else 1
    if accum is None:
        accum = max(1, math.ceil(effective / (int(micro) * max(num_gpus, 1))))

    epochs_default = _default(g.epochs, 2)
    if not image and objective == "sft" and num_train and num_train < 300:
        epochs_default = max(epochs_default, 3)
    optim = OptimSettings(
        learning_rate=float(overrides.pop("learning_rate", _default(g.learning_rate, 2e-4))),
        epochs=float(overrides.pop("epochs", epochs_default)),
        max_steps=overrides.pop("max_steps", int(_default(g.max_steps, 1500)) if image else None),
        micro_batch_size=int(micro),
        grad_accum=int(accum),
        optimizer=optimizer,
        scheduler=overrides.pop("scheduler", g.default_scheduler or "cosine"),
        warmup_ratio=float(overrides.pop("warmup_ratio", _default(g.warmup_ratio, 0.05))),
        weight_decay=float(overrides.pop("weight_decay", _default(g.weight_decay, 0.01))),
        max_grad_norm=float(overrides.pop("max_grad_norm", _default(g.max_grad_norm, 1.0))),
    )
    lora = None
    if method in ("lora", "qlora"):
        lora = LoraSettings(
            rank=rank,
            alpha=alpha,
            dropout=float(overrides.pop("lora_dropout", _default(g.dropout, 0.0))),
            targets=lora_targets,
            use_rslora=bool(overrides.pop("use_rslora", False)),
            use_dora=bool(overrides.pop("use_dora", False)),
            train_experts=bool(overrides.pop("train_experts", False)),
            train_embeddings=bool(overrides.pop("train_embeddings", False)),
        )
    dpo = None
    if objective == "dpo":
        dpo = DPOSettings(
            beta=float(overrides.pop("beta", _default(g.beta, 0.1))),
            label_smoothing=float(overrides.pop("label_smoothing", 0.0)),
            sft_weight=float(overrides.pop("sft_weight", 0.0)),
        )
    image_settings = None
    if image:
        image_settings = ImageSettings(
            resolution=seq_len,
            caption_dropout=float(overrides.pop("caption_dropout", _default(g.caption_dropout, 0.05))),
            trigger_word=overrides.pop("trigger_word", None),
            sample_prompts=list(overrides.pop("sample_prompts", [])),
            sample_every=overrides.pop("sample_every", None),
            sample_steps=int(overrides.pop("sample_steps", 30)),
        )
    precision = overrides.pop("precision", "auto")
    runtime = RuntimeSettings(
        precision=precision,
        gradient_checkpointing=bool(overrides.pop("gradient_checkpointing", True)),
        seed=int(overrides.pop("seed", 42)),
        num_gpus=num_gpus,
    )
    data = DataConfig(
        train=train_path,
        eval=eval_path,
        num_train=num_train,
        num_eval=num_eval,
        max_seq_len=seq_len if not image else 0,
        overflow=overrides.pop("overflow", "drop"),
        reasoning=overrides.pop("reasoning", "auto"),
    )
    if overrides:
        report["errors"].append(f"unknown settings: {', '.join(sorted(overrides))}")

    job = TrainJob(
        job_id=make_job_id(project),
        project=project,
        base_model=model.id,
        family=model.family,
        modality=model.modality,
        objective=objective,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        init_from=InitFrom(**init_from) if init_from else None,
        stage=stage,
        data=data,
        lora=lora,
        dpo=dpo,
        optim=optim,
        runtime=runtime,
        image=image_settings,
        expert_override=expert_override,
        model_profile=snapshot(model),
    )
    errors, warnings = validate_job(job, model)
    report["errors"].extend(errors)
    report["warnings"].extend(warnings)
    steps = total_steps(job)
    report["steps"] = steps
    if not image and steps < 10:
        report["warnings"].append(f"only {steps} optimizer steps: add more data or epochs for a noticeable effect")
    if not image and objective == "sft" and num_train and num_train < 50:
        report["warnings"].append("fewer than 50 examples: the model will mostly memorise them; aim for a few hundred")
    if objective == "dpo" and num_train and num_train < 100:
        report["warnings"].append("fewer than 100 preference pairs: DPO effects will be small or noisy; a few hundred to a few thousand work better")
    if objective == "dpo" and model.variant.kind == "base" and not init_from:
        report["warnings"].append("DPO on a raw base model rarely works well: run SFT first and start DPO from it (init_from)")
    if objective == "cpt" and not image:
        report["notes"] = ["CPT packs documents into full-length sequences; steps are estimated from token counts"]
    return job, report
