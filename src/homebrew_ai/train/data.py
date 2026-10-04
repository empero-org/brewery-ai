"""ETF files → tokenized torch datasets with loss masks (worker side)."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from homebrew_ai.etf.io import iter_records
from homebrew_ai.etf.render import IGNORE_INDEX, ChatFormat, PreferencePair, Renderer, Sample, encode_all, encode_cpt, encode_preferences


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        s = self.samples[i]
        return {"input_ids": s.input_ids, "labels": s.labels}


class PadCollator:
    """Right-pads a batch; padding positions never contribute to the loss."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id
        self.multiple = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        if self.multiple:
            longest = ((longest + self.multiple - 1) // self.multiple) * self.multiple
        ids, labels, mask = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = longest - n
            ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(f["labels"]) + [IGNORE_INDEX] * pad)
            mask.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[PreferencePair], ref: list[tuple[float, float]] | None = None):
        self.pairs = pairs
        self.ref = ref

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        p = self.pairs[i]
        item: dict[str, Any] = {
            "chosen_input_ids": p.chosen_ids,
            "chosen_labels": p.chosen_labels,
            "rejected_input_ids": p.rejected_ids,
            "rejected_labels": p.rejected_labels,
            "input_ids": p.chosen_ids,  # lets length-grouped sampling work
        }
        if self.ref is not None:
            item["ref_chosen_logps"], item["ref_rejected_logps"] = self.ref[i]
        return item


class PreferenceCollator:
    """Stacks chosen then rejected sequences into one right-padded batch of size 2B."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad = PadCollator(pad_token_id, pad_to_multiple_of)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [{"input_ids": f["chosen_input_ids"], "labels": f["chosen_labels"]} for f in features]
        rows += [{"input_ids": f["rejected_input_ids"], "labels": f["rejected_labels"]} for f in features]
        batch = self.pad(rows)
        if "ref_chosen_logps" in features[0]:
            batch["ref_chosen_logps"] = torch.tensor([f["ref_chosen_logps"] for f in features], dtype=torch.float32)
            batch["ref_rejected_logps"] = torch.tensor([f["ref_rejected_logps"] for f in features], dtype=torch.float32)
        return batch


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
