"""ETF files → tokenized samples with loss masks, without torch.

Shared by the torch worker (``train/data.py``) and the CPU backend (``train/cpuforge.py``), so the
chat template, masking and packing are identical whichever engine trains.
"""

from __future__ import annotations

import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from brewery_ai.etf.io import iter_records
from brewery_ai.etf.render import ChatFormat, Renderer, encode_all, encode_cpt, encode_preferences


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


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


def chat_format_for(model_info: Any, reasoning_mode: str = "auto") -> ChatFormat:
    fmt = ChatFormat.from_dict(model_info.chat_format_dict())
    if reasoning_mode == "drop":
        fmt = replace(fmt, reasoning="none")
    elif reasoning_mode == "inline":
        fmt = replace(fmt, reasoning="inline")
    elif reasoning_mode == "native" and fmt.reasoning != "native":
        raise ValueError(f"{model_info.id} has no native reasoning format; use 'inline' or 'drop'")
    return fmt


def _expand_repeats(path: str | Path):
    for record in iter_records(path):
        for _ in range(int(record.get("repeat", 1))):
            yield record


def build_samples(
    path: str | Path,
    tokenizer: Any,
    fmt: ChatFormat,
    max_len: int,
    overflow: str = "drop",
    seed: int = 42,
    objective: str = "sft",
) -> tuple[list[Any], dict[str, int]]:
    """Render an ETF file for the given objective (sft → masked samples, cpt → packed blocks, dpo → pairs)."""
    renderer = Renderer(tokenizer, fmt, max_len=max_len, overflow=overflow)
    if objective == "dpo":
        pairs, counts = encode_preferences(renderer, _expand_repeats(path))
        random.Random(seed).shuffle(pairs)
        counts["tokens"] = sum(p.num_tokens for p in pairs)
        return pairs, counts
    if objective == "cpt":
        records = list(_expand_repeats(path))
        random.Random(seed).shuffle(records)
        samples, counts = encode_cpt(renderer, records)
    else:
        samples, counts = encode_all(renderer, _expand_repeats(path))
        random.Random(seed).shuffle(samples)
    counts["trainable_tokens"] = sum(s.num_trainable for s in samples)
    counts["tokens"] = sum(s.num_tokens for s in samples)
    return samples, counts
