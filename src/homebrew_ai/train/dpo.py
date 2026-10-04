"""Direct Preference Optimization (worker side).

Loss (Rafailov et al., 2023), per pair::

    margin = β · [(log π(y_c|x) − log π_ref(y_c|x)) − (log π(y_r|x) − log π_ref(y_r|x))]
    loss   = −(1 − ε)·log σ(margin) − ε·log σ(−margin)   (+ optional NLL on y_c)

The reference model is the starting point of this stage (the base model, or the
merged result of the previous stage). Because the policy *equals* the
reference before the first update, reference log-probs are computed once in a
pre-pass with the very model we are about to train — no second copy in memory,
for LoRA and full fine-tuning alike.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from homebrew_ai.etf.render import IGNORE_INDEX


def selective_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p(label) per position without materialising a full fp32 log-softmax for the whole batch."""
    out = []
    for row_logits, row_labels in zip(logits, labels):
        row = row_logits.float()
        lse = torch.logsumexp(row, dim=-1)
        picked = row.gather(-1, row_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        out.append(picked - lse)
    return torch.stack(out)


def sequence_logps(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of log-probs of the labelled tokens per sequence, and the token counts."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    logps = selective_logprobs(logits, targets) * mask
    return logps.sum(-1), mask.sum(-1)


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta: float, label_smoothing: float = 0.0):
    logits = beta * ((policy_c - ref_c) - (policy_r - ref_r))
    loss = -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    rewards_c = beta * (policy_c - ref_c).detach()
    rewards_r = beta * (policy_r - ref_r).detach()
    return loss.mean(), rewards_c, rewards_r


@torch.no_grad()
def precompute_reference(model: Any, dataset: Any, collator: Any, batch_size: int, status: Any | None = None) -> list[tuple[float, float]]:
    """Reference log-probs for every pair, computed with the untrained starting model."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out: list[tuple[float, float]] = []
    total = len(dataset)
    for start in range(0, total, batch_size):
        feats = [dataset[i] for i in range(start, min(start + batch_size, total))]
        batch = collator(feats)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logps, _ = sequence_logps(model, ids, mask, labels)
        b = len(feats)
        for c, r in zip(logps[:b].tolist(), logps[b:].tolist()):
            out.append((float(c), float(r)))
        if status is not None and (start // batch_size) % 10 == 0:
            status.set(message=f"computing reference scores {min(start + batch_size, total)}/{total}")
    if was_training:
        model.train()
    return out


def make_dpo_trainer_class():
    from transformers import Trainer

    class DPOTrainer(Trainer):
        """HF Trainer with the DPO objective and DPO metrics in the logs."""

        def __init__(self, *args, beta: float = 0.1, label_smoothing: float = 0.0, sft_weight: float = 0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta
            self.label_smoothing = label_smoothing
            self.sft_weight = sft_weight
            self.model_accepts_loss_kwargs = False
            self._metrics: dict[str, list[float]] = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            ids, mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
            logps, ntok = sequence_logps(model, ids, mask, labels)
            b = ids.shape[0] // 2
            pc, pr = logps[:b], logps[b:]
            rc = inputs["ref_chosen_logps"].to(pc.dtype)
            rr = inputs["ref_rejected_logps"].to(pr.dtype)
            loss, rew_c, rew_r = dpo_loss(pc, pr, rc, rr, self.beta, self.label_smoothing)
            if self.sft_weight:
                nll = -(pc / ntok[:b].clamp_min(1)).mean()
                loss = loss + self.sft_weight * nll
            prefix = "eval_" if not model.training else ""
            self._metrics[prefix + "rewards/chosen"].append(rew_c.mean().item())
            self._metrics[prefix + "rewards/rejected"].append(rew_r.mean().item())
            self._metrics[prefix + "rewards/accuracy"].append((rew_c > rew_r).float().mean().item())
            self._metrics[prefix + "rewards/margin"].append((rew_c - rew_r).mean().item())
            return (loss, {"logits": logps}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def log(self, logs: dict[str, float], *args, **kwargs) -> None:
            eval_mode = any(k.startswith("eval_") for k in logs)
            for key in list(self._metrics):
                if key.startswith("eval_") == eval_mode and self._metrics[key]:
                    vals = self._metrics.pop(key)
                    logs[key] = round(sum(vals) / len(vals), 5)
            super().log(logs, *args, **kwargs)

    return DPOTrainer


def describe_progress(logs: dict[str, Any]) -> str | None:
    acc = logs.get("rewards/accuracy")
    if acc is None or (isinstance(acc, float) and math.isnan(acc)):
        return None
    return f"prefers the chosen answer {acc * 100:.0f}% of the time"
