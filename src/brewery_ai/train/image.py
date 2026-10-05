"""Text-to-image LoRA training (worker side). v0.1 supports Qwen-Image 2.1.

The recipe follows diffusers' official ``train_dreambooth_lora_qwenimage21.py``:

1. encode every caption once with the pipeline's own ``encode_prompt`` (it
   returns the *pre-norm* text-encoder states the DiT was trained on), then
   free the 17 GB text encoder;
2. encode every image (RGBA, [-1, 1], aspect-ratio bucketed) with the VAE,
   keep the latent distribution, then free the VAE;
3. train LoRA adapters on the DiT with the flow-matching objective
   ``x_t = (1 - σ)·x0 + σ·ε``, target ``ε - x0``;
4. save the adapter in diffusers format (``pytorch_lora_weights.safetensors``,
   ``transformer.`` key prefix) so diffusers and ComfyUI can both load it.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from brewery_ai.etf.io import iter_records
from brewery_ai.train.config import TrainJob
from brewery_ai.train.status import StatusWriter

SIZE_MULTIPLE = 32
ASPECTS = (1.0, 4 / 3, 3 / 2, 16 / 9)
PIPELINES = {"qwen_image_21": "QwenImage21Pipeline"}  # pipeline kind (profile.pipeline.kind) -> diffusers class
TRANSFORMERS = {"qwen_image_21": "QwenImage21Transformer2DModel"}


def buckets_for(resolution: int) -> list[tuple[int, int]]:
    """~resolution² pixel buckets (height, width), multiples of 32, portrait and landscape."""
    area = resolution * resolution
    out: list[tuple[int, int]] = []
    for ar in ASPECTS:
        w = int(round(math.sqrt(area * ar) / SIZE_MULTIPLE)) * SIZE_MULTIPLE
        h = int(round(math.sqrt(area / ar) / SIZE_MULTIPLE)) * SIZE_MULTIPLE
        for hw in ((h, w), (w, h)):
            if hw not in out:
                out.append(hw)
    return out


def nearest_bucket(height: int, width: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    ratio = width / height
    return min(buckets, key=lambda b: abs(math.log((b[1] / b[0]) / ratio)))


def load_image_rgba(path: Path, size: tuple[int, int]):
    """Open → RGBA → cover-resize → center-crop → tensor (4, H, W) in [-1, 1]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    target_h, target_w = size
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    w, h = img.size
    scale = max(target_h / h, target_w / w)
    img = img.resize((max(target_w, round(w * scale)), max(target_h, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - target_w) // 2, (h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def calculate_shift(seq_len: int, base_seq: int = 256, max_seq: int = 8192, base_shift: float = 0.5, max_shift: float = 0.9) -> float:
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq_len * m + (base_shift - m * base_seq)


def sample_sigmas(mode: str, batch: int, seq_len: int, scheduler_config: dict[str, Any], generator=None):
    import torch

    if mode == "uniform":
        u = torch.rand(batch, generator=generator)
        n = int(scheduler_config.get("num_train_timesteps", 1000))
        # the scheduler's unshifted training sigmas are a uniform grid on (0, 1]
        return ((u * n).long().clamp(0, n - 1) + 1).float() / n
    u = torch.sigmoid(torch.randn(batch, generator=generator))
    if mode == "logit_normal":
        return u
    mu = calculate_shift(
        seq_len,
        scheduler_config.get("base_image_seq_len", 256),
        scheduler_config.get("max_image_seq_len", 8192),
        scheduler_config.get("base_shift", 0.5),
        scheduler_config.get("max_shift", 0.9),
    )
    return math.exp(mu) / (math.exp(mu) + (1 / u.clamp_min(1e-5) - 1))


def _release_memory() -> None:
    """Return freed GPU memory; callers must drop their own references (``del``) first."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_diffusers(kind: str):
    try:
        import diffusers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("diffusers is not installed on the training machine") from exc
    name_p, name_t = PIPELINES[kind], TRANSFORMERS[kind]
    if not hasattr(diffusers, name_p) or not hasattr(diffusers, name_t):
        raise RuntimeError(
            f"this diffusers version ({diffusers.__version__}) has no {name_p}; Qwen-Image 2.1 needs diffusers from the "
            "main branch (Brewery installs a pinned commit when it prepares a server)"
        )
    return getattr(diffusers, name_p), getattr(diffusers, name_t)


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def collect_items(job: TrainJob, run_dir: Path) -> list[dict[str, Any]]:
    data_path = run_dir / job.data.train
    items = []
    for rec in iter_records(data_path):
        if "image" not in rec:
            continue
        path = Path(rec["image"])
        if not path.is_absolute():
            path = data_path.parent / path
        if path.exists():
            variants = [c for c in (rec.get("captions") or []) if c] or [rec.get("caption", "")]
            for _ in range(int(rec.get("repeat", 1))):
                items.append({"path": path, "captions": variants})
    if not items:
        raise RuntimeError("no usable samples: no image records with existing image files")
    return items


def cache_features(job: TrainJob, info: Any, run_dir: Path, items: list[dict[str, Any]], status: StatusWriter, device: str, dtype) -> dict[str, Any]:
    import torch

    pipe_cls, _ = _import_diffusers(info.profile.pipeline.get("kind", "qwen_image_21"))
    settings = job.image
    trigger = (settings.trigger_word or "").strip()
    captions: list[list[str]] = []
    for it in items:
        variants = []
        for cap in it["captions"]:
            cap = cap.strip()
            if trigger and trigger.lower() not in cap.lower():
                cap = f"{trigger}, {cap}" if cap else trigger
            variants.append(cap)
        captions.append(variants)
    unique = sorted({c for vs in captions for c in vs} | {""} | set(settings.sample_prompts))

    status.set(state="preparing", message=f"encoding {len(unique)} captions with the text encoder")
    pipe = pipe_cls.from_pretrained(info.id, transformer=None, vae=None, torch_dtype=dtype)
    pipe.text_encoder.to(device)
    embeds: dict[str, tuple[Any, Any]] = {}
    with torch.no_grad():
        for cap in unique:
            pe, mask, _pad = pipe.encode_prompt(prompt=cap or "", device=device)
            embeds[cap] = (pe[0].to("cpu", dtype), None if mask is None else mask[0].to("cpu"))
    del pipe  # the text encoder alone is ~17 GB; drop it before the VAE pass
    _release_memory()

    status.set(message=f"encoding {len(items)} images with the VAE")
    import diffusers

    vae_cls = getattr(diffusers, info.profile.pipeline.get("vae_class", "AutoencoderKLQwenImage21"))
    vae = vae_cls.from_pretrained(info.id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
    buckets = buckets_for(settings.resolution)
    latents = []
    with torch.no_grad():
        for it in items:
            from PIL import Image

            with Image.open(it["path"]) as im:
                w, h = im.size
            size = nearest_bucket(h, w, buckets)
            pixels = load_image_rgba(it["path"], size).unsqueeze(0).unsqueeze(2).to(device, dtype)
            dist = vae.encode(pixels).latent_dist
            latents.append({"mean": dist.mean[0].to("cpu", dtype), "std": dist.std[0].to("cpu", dtype), "bucket": size})
    del vae
    _release_memory()
    return {"captions": captions, "embeds": embeds, "latents": latents, "latents_mean": mean, "latents_inv_std": inv_std}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _pad_prompt_batch(pairs: list[tuple[Any, Any]], device, dtype):
    import torch

    length = max(pe.shape[0] for pe, _ in pairs)
    embeds, masks = [], []
    for pe, mask in pairs:
        pad = length - pe.shape[0]
        m = mask if mask is not None else torch.ones(pe.shape[0], dtype=torch.long)
        embeds.append(torch.cat([pe, pe.new_zeros(pad, pe.shape[1])]) if pad else pe)
        masks.append(torch.cat([m, m.new_zeros(pad)]) if pad else m)
    embeds_t = torch.stack(embeds).to(device, dtype)
    mask_t = torch.stack(masks).to(device)
    return embeds_t, (None if bool(mask_t.all()) else mask_t)


def flow_matching_loss(transformer, pipe_cls, x0, prompt_embeds, prompt_mask, sigmas):
    """Rectified-flow loss for one batch of normalized latents ``x0`` with shape (B, C, 1, h, w)."""
    import torch

    bsz, ch, _, lh, lw = x0.shape
    device = x0.device
    noise = torch.randn_like(x0)
    s = sigmas.view(bsz, 1, 1, 1, 1)
    noisy = (1.0 - s) * x0 + s * noise
    packed = pipe_cls._pack_latents(noisy, batch_size=bsz, num_channels_latents=ch, height=lh, width=lw)
    img_mask = torch.cat(
        [
            torch.zeros(bsz, prompt_embeds.shape[1], dtype=torch.bool, device=device),
            torch.ones(bsz, (lh * lw) // 4, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    pred = transformer(
        hidden_states=packed,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        timestep=sigmas.view(bsz),
        img_shapes=[[(1, lh, lw)]] * bsz,
        img_mask=img_mask,
        return_dict=False,
    )[0]
    pred = pred[:, -packed.shape[1]:]
    pred = pipe_cls._unpack_latents(pred, lh * 16, lw * 16, 16)
    target = noise - x0
    return ((pred.float() - target.float()) ** 2).reshape(bsz, -1).mean(1).mean()


def run_image(job: TrainJob, run_dir: Path, status: StatusWriter, resume: bool = False) -> dict[str, Any]:
    import torch
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from brewery_ai.train import model as tmodel

    info = job.resolved_model()
    settings = job.image
    if settings is None:
        raise RuntimeError("image job without image settings")
    if not torch.cuda.is_available():
        raise RuntimeError("text-to-image training needs an NVIDIA GPU")
    device = "cuda"
    precision = tmodel.resolve_precision(job.runtime.precision, info)
    dtype = tmodel.torch_dtype(precision)
    rng = random.Random(job.runtime.seed)
    torch.manual_seed(job.runtime.seed)

    items = collect_items(job, run_dir)
    cache = cache_features(job, info, run_dir, items, status, device, dtype)

    status.set(state="loading_model", message=f"loading the {info.variant.label} transformer", precision=precision)
    pipe_cls, transformer_cls = _import_diffusers(info.profile.pipeline.get("kind", "qwen_image_21"))
    load_kwargs: dict[str, Any] = {"subfolder": "transformer", "torch_dtype": dtype}
    if job.method == "qlora":
        from diffusers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
        )
    transformer = transformer_cls.from_pretrained(info.id, **load_kwargs)
    transformer.to(device)
    transformer.requires_grad_(False)
    if job.runtime.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    lora = job.lora
    targets = lora.targets if isinstance(lora.targets, list) else info.lora_targets(lora.targets)
    transformer.add_adapter(
        LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout, init_lora_weights="gaussian", target_modules=targets)
    )
    params = [p for p in transformer.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    trainable = sum(p.numel() for p in params)
    status.set(trainable_params=trainable)

    o = job.optim
    from brewery_ai.train.runner import stop_requested, usable_optimizer

    o.optimizer = usable_optimizer(o.optimizer, status)
    if o.optimizer in ("adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit"):
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(params, lr=o.learning_rate, weight_decay=o.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=o.learning_rate, weight_decay=o.weight_decay)
    from transformers import get_scheduler

    max_steps = int(o.max_steps or 1000)
    scheduler = get_scheduler(o.scheduler, optimizer, num_warmup_steps=int(o.warmup_ratio * max_steps), num_training_steps=max_steps)

    sched_path = _scheduler_config(info.id)
    sched_cfg = json.loads(Path(sched_path).read_text()) if sched_path else {}
    by_bucket: dict[tuple[int, int], list[int]] = {}
    for i, lat in enumerate(cache["latents"]):
        by_bucket.setdefault(lat["bucket"], []).append(i)
    bucket_keys = list(by_bucket)
    weights = [len(by_bucket[k]) for k in bucket_keys]
    mean = cache["latents_mean"].to(device, dtype)
    inv_std = cache["latents_inv_std"].to(device, dtype)
    empty = cache["embeds"][""]

    final_dir = run_dir / "final"
    ckpt_dir = run_dir / "checkpoints"
    log_every = max(1, min(job.runtime.logging_steps, max_steps // 50 or 1))
    save_every = job.runtime.save_steps or max(100, max_steps // 4)
    # previews line up with the checkpoints by default, so a good-looking set is a checkpoint you can keep
    preview_every = save_every if settings.sample_every is None else max(0, int(settings.sample_every))
    previews = None
    prompts = preview_prompts(settings, cache["captions"])
    if prompts:
        status.set(state="preparing", message="setting up preview images")
        previews = Previews(pipe_cls, info, transformer, cache["embeds"], prompts, run_dir, status,
                            resolution=settings.resolution, steps=settings.sample_steps, device=device, dtype=dtype)
        if preview_every and previews.ready:
            previews.render(0)  # "before": the LoRA starts at zero, so this is the base model
    status.set(state="training", step=0, max_steps=max_steps, message="training")
    t0 = time.time()
    running = 0.0
    transformer.train()
    for step in range(1, max_steps + 1):
        loss_acc = 0.0
        for _ in range(o.grad_accum):
            bucket = rng.choices(bucket_keys, weights=weights)[0]
            idx = [rng.choice(by_bucket[bucket]) for _ in range(o.micro_batch_size)]
            pairs = []
            for i in idx:
                cap = rng.choice(cache["captions"][i])
                pairs.append(empty if rng.random() < settings.caption_dropout else cache["embeds"][cap])
            prompt_embeds, prompt_mask = _pad_prompt_batch(pairs, device, dtype)
            x0 = torch.stack([
                cache["latents"][i]["mean"] + cache["latents"][i]["std"] * torch.randn_like(cache["latents"][i]["std"])
                for i in idx
            ]).to(device, dtype)
            x0 = (x0 - mean) * inv_std
            sigmas = sample_sigmas(settings.timestep_sampling, x0.shape[0], x0.shape[3] * x0.shape[4], sched_cfg).to(device, dtype)
            loss = flow_matching_loss(transformer, pipe_cls, x0, prompt_embeds, prompt_mask, sigmas) / o.grad_accum
            loss.backward()
            loss_acc += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, o.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running = loss_acc if step == 1 else 0.9 * running + 0.1 * loss_acc
        if not math.isfinite(loss_acc):
            raise RuntimeError("nan loss: the loss became NaN")
        if step % log_every == 0 or step == max_steps:
            elapsed = time.time() - t0
            status.log_metrics({"step": step, "loss": loss_acc, "learning_rate": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm)})
            status.set(
                step=step, loss=round(running, 5), learning_rate=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                eta_s=round(elapsed / step * (max_steps - step), 1), gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                epoch=round(step * o.micro_batch_size * o.grad_accum / len(items), 2),
            )
        if step % save_every == 0 and step < max_steps:
            _save_lora(pipe_cls, transformer, ckpt_dir / f"checkpoint-{step}", get_peft_model_state_dict)
            # with previews, each checkpoint has a picture of what it does: keep them (LoRA files are small)
            _prune_checkpoints(ckpt_dir, max(job.runtime.save_total_limit, 12) if preview_every else job.runtime.save_total_limit)
            status.set(last_checkpoint_step=step)
        if previews is not None and preview_every and step % preview_every == 0 and step < max_steps:
            previews.render(step)
        if stop_requested(run_dir):
            break

    status.set(state="saving", message="saving the LoRA")
    _save_lora(pipe_cls, transformer, final_dir, get_peft_model_state_dict)
    stopped = step < max_steps
    results: dict[str, Any] = {"metrics": {"train_loss": running, "steps": step}, "images": len(items), "stopped_early": stopped}
    final_rows = previews.render(step, final=True) if previews is not None and previews.ready else []
    if final_rows:
        results["samples"] = [{"prompt": r["prompt"], "path": r["path"]} for r in final_rows]
    del transformer, optimizer, scheduler, params, cache, mean, inv_std, empty, previews
    _release_memory()
    (run_dir / "results.json").write_text(json.dumps(results, indent=1))
    if settings.sample_prompts and not final_rows:  # previews were unavailable: render with a fresh pipeline
        status.set(message="rendering sample images")
        try:
            rows = sample_images(run_dir, settings.sample_prompts[:4])
            results["samples"] = [{"prompt": r["prompt"], "path": r["finetuned"]} for r in rows]
            (run_dir / "results.json").write_text(json.dumps(results, indent=1))
        except Exception as exc:  # samples are nice to have
            status.set(sample_error=str(exc)[:300])
    return {"metrics": results["metrics"], "stopped": stopped}


class Previews:
    """Sample images rendered *during* training with the model being trained.

    Reuses the in-memory transformer (with its LoRA) and the prompt embeddings cached before training, so only
    the small VAE is loaded: no second copy of the 20B model and no text encoder. The same seeds are used every
    time, so the sets show how the LoRA changes the same picture. Each set goes to ``samples/step_NNNNNN/`` and
    is listed in ``samples/index.json`` and in ``status.json`` (``samples``, ``last_sample_step``).
    A failure only switches previews off; training goes on.
    """

    def __init__(self, pipe_cls, info: Any, transformer, embeds: dict[str, Any], prompts: list[str], run_dir: Path, status: StatusWriter,
                 *, resolution: int, steps: int, device: str, dtype, seed: int = 1234):
        self.transformer, self.embeds, self.prompts = transformer, embeds, prompts
        self.run_dir, self.status = run_dir, status
        self.size = max(256, resolution // 32 * 32)
        self.steps, self.device, self.dtype, self.seed = steps, device, dtype, seed
        self.pipe = None
        self.sets: list[dict[str, Any]] = []
        try:
            # the processor is just the tokenizer (tiny) but the pipeline needs it at start-up; only the big
            # text encoder is skipped, because the prompt embeddings were computed before training
            pipe = pipe_cls.from_pretrained(info.id, transformer=transformer, text_encoder=None, torch_dtype=dtype)
            pipe.vae.to(device)
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
        except Exception as exc:  # previews are a nicety
            self._fail(exc)

    @property
    def ready(self) -> bool:
        return self.pipe is not None and bool(self.prompts)

    def _fail(self, exc: Exception) -> None:
        self.pipe = None
        self.status.set(preview_error=f"{type(exc).__name__}: {exc}"[:300])

    def render(self, step: int, *, final: bool = False) -> list[dict[str, Any]]:
        if not self.ready:
            return []
        import torch

        rel = f"samples/step_{step:06d}"
        out_dir = self.run_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        was_training = self.transformer.training
        self.transformer.eval()
        self.status.set(message=f"rendering {len(self.prompts)} preview image(s) at step {step}")
        try:
            for i, prompt in enumerate(self.prompts):
                pe, mask = self.embeds[prompt]
                image = self.pipe(
                    prompt_embeds=pe.unsqueeze(0).to(self.device, self.dtype),
                    prompt_embeds_mask=None if mask is None else mask.unsqueeze(0).to(self.device),
                    height=self.size, width=self.size, num_inference_steps=self.steps,
                    generator=torch.Generator("cpu").manual_seed(self.seed + i),
                ).images[0]
                name = f"sample_{i:02d}.png"
                image.save(out_dir / name)
                row = {"prompt": prompt, "path": f"{rel}/{name}"}
                if final:  # the model card shows these
                    image.save(self.run_dir / "samples" / name)
                    row["path"] = f"samples/{name}"
                rows.append(row)
        except Exception as exc:
            self._fail(exc)
        finally:
            if was_training:
                self.transformer.train()
            _release_memory()
        if rows:
            self.sets = [s for s in self.sets if s["step"] != step] + [{"step": step, "dir": rel, "count": len(rows)}]
            index = {"prompts": self.prompts, "sets": self.sets}
            (self.run_dir / "samples" / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
            self.status.set(samples=self.sets, last_sample_step=step, **({} if final else {"message": "training"}))
        return rows


def preview_prompts(settings: Any, captions: list[list[str]], limit: int = 4) -> list[str]:
    """The user's sample prompts, or a few training captions so there is always something to look at."""
    if settings.sample_prompts:
        return list(dict.fromkeys(settings.sample_prompts))[:limit]
    seen: list[str] = []
    for variants in captions:
        cap = (variants[0] if variants else "").strip()
        if cap and cap not in seen:
            seen.append(cap)
        if len(seen) >= min(limit, 3):
            break
    return seen


def _scheduler_config(model_id: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model_id, "scheduler/scheduler_config.json")
    except Exception:
        return None


def _save_lora(pipe_cls, transformer, out_dir: Path, get_state) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = get_state(transformer)
    kwargs: dict[str, Any] = {}
    try:
        kwargs["transformer_lora_adapter_metadata"] = transformer.peft_config["default"].to_dict()
    except Exception:
        pass
    pipe_cls.save_lora_weights(save_directory=str(out_dir), transformer_lora_layers=state, **kwargs)


def _prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    import shutil

    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for old in ckpts[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def sample_images(run_dir: Path, prompts: list[Any], compare_base: bool = False, steps: int | None = None, prefix: str = "sample") -> list[dict[str, Any]]:
    """Render ``prompts`` with the trained LoRA into ``samples/<prefix>_NN.png`` (training samples use
    "sample", test_model uses "test" so the model card's samples are never overwritten)."""
    import torch

    job = TrainJob.load(run_dir / "job.yaml")
    info = job.resolved_model()
    pipe_cls, _ = _import_diffusers(info.profile.pipeline.get("kind", "qwen_image_21"))
    pipe = pipe_cls.from_pretrained(info.id, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(str(run_dir / "final"), adapter_name="brew")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    if total_gb >= 48:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    res = job.image.resolution if job.image else 1024
    out_dir = run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for i, prompt in enumerate(prompts):
        text = str(prompt)
        gen = torch.Generator("cpu").manual_seed(1234 + i)
        kwargs = dict(prompt=text, height=res, width=res, num_inference_steps=steps or (job.image.sample_steps if job.image else 30), generator=gen)
        image = pipe(**kwargs).images[0]
        name = f"{prefix}_{i:02d}.png"
        image.save(out_dir / name)
        row = {"prompt": text, "finetuned": f"samples/{name}"}
        if compare_base:
            pipe.disable_lora()
            base = pipe(**{**kwargs, "generator": torch.Generator("cpu").manual_seed(1234 + i)}).images[0]
            base_name = f"{prefix}_base_{i:02d}.png"
            base.save(out_dir / base_name)
            row["base"] = f"samples/{base_name}"
            pipe.enable_lora()
        rows.append(row)
    return rows
