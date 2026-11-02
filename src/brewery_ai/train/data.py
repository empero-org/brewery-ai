"""ETF files → tokenized torch datasets with loss masks (worker side)."""

from __future__ import annotations

from typing import Any

import torch

from brewery_ai.etf.render import IGNORE_INDEX, PreferencePair, Sample
from brewery_ai.train.samples import build_samples, chat_format_for  # noqa: F401  (re-exported: torch-free home is train/samples.py)


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
