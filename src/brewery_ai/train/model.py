"""Loading base models and attaching adapters (worker side)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch

log = logging.getLogger("brewery.train")

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>]+\|>|<[a-z_]+\|>|<\|[a-z_]+>|<(?:start|end)_of_turn>")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def device_info() -> dict[str, Any]:
    if torch.cuda.is_available():
        caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        return {"device": "cuda", "count": torch.cuda.device_count(), "bf16": all(c[0] >= 8 for c in caps)}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"device": "mps", "count": 1, "bf16": False}
    return {"device": "cpu", "count": 0, "bf16": False}


def resolve_precision(requested: str, model_info: Any) -> str:
    dev = device_info()
    if requested != "auto":
        return requested
    if dev["device"] == "cuda":
        if dev["bf16"]:
            return "bf16"
        if model_info.profile.loading.get("bf16_required"):
            raise RuntimeError(
                f"{model_info.profile.display_name} needs a GPU with bf16 support (Ampere or newer); "
                "this GPU only has fp16, where the model's activations overflow."
            )
        return "fp16"
    return "fp32"


def torch_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def load_tokenizer(model_info: Any, source: str | None = None) -> Any:
    """Tokenizer of the base model (or of an earlier stage's output folder ``source``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source or model_info.id, token=hf_token())
    donor_id = model_info.variant.chat_template_from
    if donor_id and donor_id != model_info.id and not getattr(tok, "chat_template", None):
        donor = AutoTokenizer.from_pretrained(donor_id, token=hf_token())
        tok.chat_template = donor.chat_template
    if not getattr(tok, "chat_template", None) and model_info.modality == "text":
        raise RuntimeError(f"{model_info.id} has no chat template; set chat_template_from in its profile")
    if tok.pad_token is None:
        wanted = model_info.profile.loading.get("pad_token")
        if wanted and tok.convert_tokens_to_ids(wanted) not in (None, tok.unk_token_id):
            tok.pad_token = wanted
        else:
            tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def stop_token_ids(tokenizer: Any, model_info: Any) -> list[int]:
    ids: list[int] = []
    for t in model_info.profile.stop_tokens:
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tokenizer.unk_token_id and i not in ids:
            ids.append(i)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def load_model(model_info: Any, method: str, precision: str, attn_implementation: str | None = None, source: str | None = None) -> Any:
    from transformers import AutoModelForCausalLM

    dtype = torch_dtype(precision)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attn_implementation or model_info.profile.loading.get("attn_implementation", "sdpa"),
        "token": hf_token(),
    }
    dev = device_info()
    if dev["device"] == "cuda":
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
    if method == "qlora":
        if dev["device"] != "cuda":
            raise RuntimeError("QLoRA needs an NVIDIA GPU (bitsandbytes). Use LoRA on this machine.")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(source or model_info.id, **kwargs)
    model.config.use_cache = False
    return model


def control_token_ids(tokenizer: Any, model_info: Any) -> list[int]:
    fmt = model_info.chat_format_dict()
    texts = [fmt.get("assistant_header", ""), *fmt.get("turn_end", []), *model_info.profile.stop_tokens]
    ids: set[int] = set()
    for text in texts:
        for tok in SPECIAL_TOKEN_RE.findall(text or ""):
            i = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(i, int) and i != tokenizer.unk_token_id:
                ids.add(i)
    return sorted(ids)


@torch.no_grad()
def fix_untrained_tokens(model: Any, tokenizer: Any, model_info: Any) -> list[int]:
    """Give *untrained* chat-control token rows of a base model the mean embedding.

    Base checkpoints (e.g. Llama 3.x) ship zero or identical placeholder rows
    for ``<|eot_id|>`` and friends; with the instruct chat template those rows
    would produce NaNs or never be learnable through LoRA. Rows that look
    trained (e.g. after an earlier full fine-tuning stage) are left alone.
    """
    ids = control_token_ids(tokenizer, model_info)
    if not ids:
        return []
    fixed: set[int] = set()
    for emb in {id(m): m for m in (model.get_input_embeddings(), model.get_output_embeddings()) if m is not None}.values():
        w = emb.weight
        if getattr(w, "dtype", None) not in (torch.float32, torch.float16, torch.bfloat16):
            continue  # quantized; leave it
        norms = w.float().norm(dim=-1)
        median = norms.median()
        rows = w[ids].float()
        untrained = []
        for j, tid in enumerate(ids):
            tiny = norms[tid] < 0.05 * median
            duplicate = any(k != j and torch.allclose(rows[j], rows[k]) for k in range(len(ids)))
            if tiny or duplicate:
                untrained.append(tid)
        if not untrained:
            continue
        keep = torch.ones(w.shape[0], dtype=torch.bool, device=w.device)
        keep[ids] = False
        mean = w[keep].float().mean(dim=0)
        w[untrained] = mean.to(w.dtype)
        fixed.update(untrained)
    if fixed:
        log.info("initialised %d untrained chat-control token embeddings to the mean", len(fixed))
    return sorted(fixed)


def lora_target_regex(model: Any, names: list[str]) -> str:
    """Build a PEFT full-match regex for the named projections.

    Multimodal checkpoints (Gemma 3/4 load as ConditionalGeneration) are
    restricted to the language model so vision/audio towers stay untouched.
    """
    alternatives = "|".join(re.escape(n) for n in names)
    multimodal = any(".language_model." in f".{n}." for n, _ in model.named_modules())
    if multimodal:
        return rf".*language_model.*\.({alternatives})"
    return rf"(.*\.)?({alternatives})"


def _module_name(model: Any, module: Any) -> str | None:
    return next((n.rsplit(".", 1)[-1] for n, m in model.named_modules() if m is module), None)


def apply_lora(model: Any, job: Any, model_info: Any, token_ids: list[int] | None = None) -> Any:
    """Wrap ``model`` with LoRA. ``token_ids`` (rows repaired by fix_untrained_tokens) become trainable
    token rows that are saved inside the adapter, so the repair survives loading onto the original base."""
    from peft import LoraConfig, get_peft_model

    settings = job.lora
    names = settings.targets if isinstance(settings.targets, list) else model_info.lora_targets(settings.targets)
    config_kwargs: dict[str, Any] = {
        "r": settings.rank,
        "lora_alpha": settings.alpha,
        "lora_dropout": settings.dropout,
        "target_modules": lora_target_regex(model, names),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_rslora": settings.use_rslora,
        "use_dora": settings.use_dora,
    }
    if settings.train_experts:
        params = model_info.profile.pipeline.get("expert_parameters") or ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
        config_kwargs["target_parameters"] = params
    if settings.train_embeddings:
        config_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
        if model_info.variant.tied_embeddings:
            config_kwargs["ensure_weight_tying"] = True
    elif token_ids:
        inp, out = model.get_input_embeddings(), model.get_output_embeddings()
        names = {_module_name(model, inp): list(token_ids)}
        if out is not None and out.weight is not inp.weight:  # untied head: its rows were repaired too
            names[_module_name(model, out)] = list(token_ids)
        if None not in names:
            config_kwargs["trainable_token_indices"] = names
    if job.runtime.gradient_checkpointing:
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, LoraConfig(**config_kwargs))
    return peft_model


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = total = 0
    for p in model.parameters():
        n = p.numel()
        if hasattr(p, "ds_numel"):
            n = p.ds_numel
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
