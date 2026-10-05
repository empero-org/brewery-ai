"""Back-of-the-envelope GPU memory, time and cost estimates.

These numbers steer the agent (which GPU to rent, which method fits, what
batch size to use). They are deliberately a little pessimistic: running out of
memory an hour into a rented session is the worst outcome for a beginner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from brewery_ai.hardware.gpus import GPU, CATALOG
from brewery_ai.models.schema import ResolvedModel

GB = 1024**3
CUDA_OVERHEAD_GB = 1.2
FRAGMENTATION = 1.08
HEADROOM = 0.92  # use at most 92% of a GPU's memory

OPTIMIZER_STATE_BYTES = {
    # bytes per *trainable* parameter for optimizer state
    "adamw_torch_fused": 8,
    "adamw_torch": 8,
    "adamw_8bit": 2,
    "adamw_bnb_8bit": 2,
    "paged_adamw_8bit": 2,
    "paged_adamw_32bit": 8,
    "adafactor": 1,
}


@dataclass
class TrainShape:
    method: str  # lora | qlora | full
    objective: str = "sft"
    seq_len: int = 2048  # text: tokens per sample; image: training resolution (px)
    micro_batch: int = 1
    lora_rank: int = 16
    lora_preset: str = "default"
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_torch_fused"


@dataclass
class MemoryEstimate:
    total_gb: float
    breakdown: dict[str, float]
    trainable_params: int
    notes: list[str] = field(default_factory=list)

    def fits(self, vram_gb: float) -> bool:
        return self.total_gb <= vram_gb * HEADROOM

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gb": round(self.total_gb, 1),
            "breakdown_gb": {k: round(v, 2) for k, v in self.breakdown.items()},
            "trainable_params_m": round(self.trainable_params / 1e6, 1),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# text models
# --------------------------------------------------------------------------- #


def _dims(model: ResolvedModel) -> dict[str, float]:
    v = model.variant
    hidden = v.hidden or 4096
    heads = v.heads or max(hidden // 128, 1)
    head_dim = v.head_dim or hidden // heads
    return {
        "P": v.params_b * 1e9,
        "P_active": (v.active_params_b or v.params_b) * 1e9,
        "H": hidden,
        "L": v.layers or 32,
        "V": v.vocab or 128000,
        "I": v.expert_intermediate if v.is_moe and v.expert_intermediate else (v.intermediate or hidden * 4),
        "q": heads * head_dim,
        "kv": (v.kv_heads or heads) * head_dim,
    }


def lora_param_count(model: ResolvedModel, rank: int, preset: str = "default") -> int:
    d = _dims(model)
    try:
        targets = set(model.lora_targets(preset))
    except KeyError:
        targets = set(model.lora_targets("default"))
    H, I, q, kv = d["H"], d["I"], d["q"], d["kv"]
    io = {
        "q_proj": H + q, "k_proj": H + kv, "v_proj": H + kv, "o_proj": q + H,
        "in_proj_qkv": H + 2 * q, "in_proj_z": H + q, "out_proj": q + H,
        "gate_proj": H + I, "up_proj": H + I, "down_proj": I + H,
    }
    per_layer = sum(v for k, v in io.items() if k in targets)
    if not per_layer:  # unknown names (image models etc.): assume attention-sized projections
        per_layer = 4 * 2 * H * max(len(targets), 1) / 4
    return int(rank * per_layer * d["L"])


def estimate_text_memory(model: ResolvedModel, shape: TrainShape) -> MemoryEstimate:
    d = _dims(model)
    v = model.variant
    P, H, L, V = d["P"], d["H"], d["L"], d["V"]
    notes: list[str] = []
    embed = V * H * (1 if v.tied_embeddings else 2)

    if shape.method == "qlora":
        if v.is_moe:
            weights = P * 2
            notes.append("MoE experts cannot be 4-bit quantized by bitsandbytes; they stay in bf16.")
        else:
            weights = (P - embed) * 0.56 + embed * 2
    else:
        weights = P * 2

    if shape.method == "full":
        trainable = int(P)
        per_param = 2 + OPTIMIZER_STATE_BYTES.get(shape.optimizer, 8) / 2  # bf16 grads + bf16-held states
        trainable_bytes = trainable * per_param
        notes.append("Full fine-tuning keeps weights, gradients and optimizer states for every parameter.")
    else:
        trainable = lora_param_count(model, shape.lora_rank, shape.lora_preset)
        trainable_bytes = trainable * (4 + 4 + OPTIMIZER_STATE_BYTES.get(shape.optimizer, 8))

    tokens = shape.seq_len * shape.micro_batch * (2 if shape.objective == "dpo" else 1)  # DPO runs chosen + rejected together
    if shape.gradient_checkpointing:
        activations = tokens * H * 2 * L + tokens * (18 * H + 6 * d["I"]) * 2
    else:
        activations = tokens * L * (34 * H + 8 * d["I"])
    logits = tokens * V * 10  # bf16 logits + fp32 upcast + fp32 grad in the loss
    if V > 200_000:
        notes.append("Large vocabulary: the loss alone needs several GB at long sequence lengths.")

    breakdown = {
        "weights": weights / GB,
        "trainable_and_optimizer": trainable_bytes / GB,
        "activations": activations / GB,
        "logits": logits / GB,
        "cuda_overhead": CUDA_OVERHEAD_GB,
    }
    total = sum(breakdown.values()) * FRAGMENTATION
    return MemoryEstimate(total, breakdown, trainable, notes)


# --------------------------------------------------------------------------- #
# image models (diffusion transformer + cached text/latent features)
# --------------------------------------------------------------------------- #


def estimate_image_memory(model: ResolvedModel, shape: TrainShape) -> MemoryEstimate:
    v = model.variant
    P = v.params_b * 1e9
    H = v.hidden or 3072
    L = v.layers or 60
    notes = ["Text embeddings and image latents are pre-computed, so the text encoder and VAE are not in memory while training."]
    weights = P * (0.56 if shape.method == "qlora" else 2.0)
    trainable = int(shape.lora_rank * L * 12 * 2 * H)  # ~12 projections per double-stream block
    trainable_bytes = trainable * (4 + 4 + OPTIMIZER_STATE_BYTES.get(shape.optimizer, 8))
    image_tokens = (shape.seq_len // 16) ** 2
    tokens = (image_tokens + 256) * shape.micro_batch
    if shape.gradient_checkpointing:
        activations = tokens * H * 2 * L * 2 + tokens * H * 40 * 2
    else:
        activations = tokens * L * H * 60
    encoder_peak = ((v.text_encoder_params_b or 7.0) * 1e9 * 2 + (v.vae_params_b or 0.2) * 1e9 * 2) / GB + CUDA_OVERHEAD_GB
    breakdown = {
        "transformer_weights": weights / GB,
        "trainable_and_optimizer": trainable_bytes / GB,
        "activations": activations / GB,
        "cuda_overhead": CUDA_OVERHEAD_GB,
    }
    total = sum(breakdown.values()) * FRAGMENTATION
    if encoder_peak > total:
        notes.append(f"Pre-computing features briefly needs ~{encoder_peak:.0f} GB (text encoder in bf16).")
        total = encoder_peak
    return MemoryEstimate(total, breakdown, trainable, notes)


def estimate_memory(model: ResolvedModel, shape: TrainShape) -> MemoryEstimate:
    if model.modality == "image":
        return estimate_image_memory(model, shape)
    return estimate_text_memory(model, shape)


# --------------------------------------------------------------------------- #
# throughput, time and cost
# --------------------------------------------------------------------------- #


def _mfu(gpu: GPU, method: str) -> float:
    mfu = 0.38 if gpu.consumer else 0.33
    if method == "qlora":
        mfu *= 0.55  # dequantization overhead
    if not gpu.bf16:
        mfu *= 0.6
    return mfu


def estimate_hours(model: ResolvedModel, shape: TrainShape, total_tokens: float, gpu: GPU, num_gpus: int = 1) -> tuple[float, float]:
    """Training wall-clock range (hours) for ``total_tokens`` processed tokens.

    For image models ``total_tokens`` is the number of training *steps* × batch.
    """
    v = model.variant
    p_active = (v.active_params_b or v.params_b) * 1e9
    if model.modality == "image":
        per_sample_tokens = (shape.seq_len // 16) ** 2 + 256
        work_tokens = total_tokens * per_sample_tokens
    else:
        work_tokens = total_tokens
    flops_per_token = (6 if shape.method == "full" else 4) * p_active
    if shape.gradient_checkpointing:
        flops_per_token += 2 * p_active
    if shape.objective == "dpo":
        flops_per_token = flops_per_token * 2 + 2 * p_active  # two sequences per pair + the reference pre-pass
    seconds = work_tokens * flops_per_token / (gpu.tflops * 1e12 * _mfu(gpu, shape.method) * max(num_gpus, 1) * (0.9 if num_gpus > 1 else 1.0))
    hours = seconds / 3600
    return hours * 0.75, hours * 1.6


def fitting_gpus(model: ResolvedModel, shape: TrainShape, provider: str | None = None, rentable_only: bool = True) -> list[dict[str, Any]]:
    """Catalog GPUs that fit this job, cheapest first.

    Rental recommendations skip GPUs without bf16 (T4, V100): they train in
    fp16, which is less stable, and recent PyTorch wheels no longer support them.
    """
    est = estimate_memory(model, shape)
    rows = []
    for gpu in CATALOG:
        if not est.fits(gpu.vram_gb):
            continue
        if not gpu.bf16 and (rentable_only or model.profile.loading.get("bf16_required")):
            continue
        if rentable_only and gpu.price(provider) is None:
            continue
        price = gpu.price(provider)
        rows.append({"gpu": gpu.name, "key": gpu.key, "vram_gb": gpu.vram_gb, "usd_per_hour": price, "notes": gpu.notes})
    rows.sort(key=lambda r: (r["usd_per_hour"] is None, r["usd_per_hour"] or 0, r["vram_gb"]))
    return rows


@dataclass
class FitResult:
    ok: bool
    method: str
    micro_batch: int
    grad_accum: int
    seq_len: int
    gradient_checkpointing: bool
    memory: MemoryEstimate
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": self.ok,
            "method": self.method,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "seq_len": self.seq_len,
            "gradient_checkpointing": self.gradient_checkpointing,
            "memory": self.memory.to_dict(),
            "suggestions": self.suggestions,
        }


def autofit(
    model: ResolvedModel,
    shape: TrainShape,
    vram_gb: float,
    effective_batch: int,
    num_gpus: int = 1,
) -> FitResult:
    """Pick the largest micro-batch (≤ 8) that fits, then derive gradient accumulation."""
    best: tuple[int, MemoryEstimate] | None = None
    for mb in (8, 4, 2, 1):
        if mb > effective_batch:
            continue
        trial = TrainShape(**{**shape.__dict__, "micro_batch": mb, "gradient_checkpointing": True})
        est = estimate_memory(model, trial)
        if est.fits(vram_gb):
            best = (mb, est)
            break
    if best is None:
        est = estimate_memory(model, TrainShape(**{**shape.__dict__, "micro_batch": 1, "gradient_checkpointing": True}))
        suggestions = []
        if shape.method == "full":
            suggestions.append("switch to LoRA (trains a small adapter instead of every weight)")
        if shape.method == "lora" and model.guidelines.qlora.allowed:
            suggestions.append("switch to QLoRA (4-bit base model)")
        if model.modality == "text" and shape.seq_len > 1024:
            suggestions.append(f"shorten max_seq_len (now {shape.seq_len})")
        if model.modality == "image" and shape.seq_len > 768:
            suggestions.append(f"train at a lower resolution (now {shape.seq_len}px)")
        suggestions.append(f"rent a GPU with at least {math.ceil(est.total_gb / HEADROOM)} GB")
        suggestions.append("pick a smaller base model")
        return FitResult(False, shape.method, 1, max(effective_batch, 1), shape.seq_len, True, est, suggestions)
    mb, est = best
    accum = max(1, math.ceil(effective_batch / (mb * max(num_gpus, 1))))
    return FitResult(True, shape.method, mb, accum, shape.seq_len, True, est)
