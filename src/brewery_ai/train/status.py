"""Live job status shared between the worker and the control plane.

The worker writes ``status.json`` (atomically) and appends to
``metrics.jsonl`` in the run directory; the control plane reads them locally or
over SSH. Keep this module dependency-free.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STATUS_FILE = "status.json"
METRICS_FILE = "metrics.jsonl"
STOP_FILE = "STOP"  # created by the control plane to ask for a clean stop (works on every OS and with torchrun)
STATES = ("queued", "preparing", "loading_model", "training", "saving", "completed", "failed", "stopped")
TERMINAL = ("completed", "failed", "stopped")


def process_rank() -> int:
    """Global rank under torchrun (0 when training on one GPU)."""
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


class StatusWriter:
    """Only the main process (rank 0) reports progress; any process may report a failure."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.primary = process_rank() == 0
        self.path = self.run_dir / STATUS_FILE
        self.started = time.time()
        self.data: dict[str, Any] = {"state": "queued", "pid": os.getpid(), "started_at": _now(), "loss_history": []}
        if self.path.exists():
            try:
                previous = json.loads(self.path.read_text())
                self.data["loss_history"] = previous.get("loss_history", [])[-200:]
            except (OSError, ValueError):
                pass
        self.flush()

    def set(self, **fields: Any) -> None:
        self.data.update(fields)
        self.data["elapsed_s"] = round(time.time() - self.started, 1)
        self.flush()

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.primary:
            return
        record = {"time": _now(), **metrics}
        with open(self.run_dir / METRICS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if "loss" in metrics and "step" in metrics:
            history = self.data.setdefault("loss_history", [])
            history.append([metrics["step"], round(float(metrics["loss"]), 5)])
            if len(history) > 400:
                del history[: len(history) - 400]

    def fail(self, error: str, hint: str | None = None, kind: str = "error") -> None:
        self.data.update(state="failed", error=error[-4000:], error_kind=kind, hint=hint, finished_at=_now())
        self.flush(force=True)

    def flush(self, force: bool = False) -> None:
        if not (self.primary or force):
            return
        self.data["updated_at"] = _now()
        tmp = self.path.with_name(f".{STATUS_FILE}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def read_status(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / STATUS_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def classify_error(text: str) -> tuple[str, str]:
    """Map a traceback to ``(kind, beginner-friendly hint)``."""
    t = text.lower()
    if "out of memory" in t or "outofmemoryerror" in t or "cuda error: out of memory" in t:
        return "oom", "The GPU ran out of memory. Lower the micro batch size or max_seq_len, switch to QLoRA, or use a GPU with more memory."
    if "gatedrepoerror" in t or "401 client error" in t or "403 client error" in t or "access to model" in t and "restricted" in t:
        return "auth", "The base model is gated: accept its license on huggingface.co with your account and make sure an HF token is available on the training machine."
    if "no space left on device" in t:
        return "disk", "The training machine ran out of disk space. Free space or attach a bigger volume (models need roughly 2-3x their size)."
    if "nan" in t and "loss" in t:
        return "nan", "The loss became NaN. Lower the learning rate, and make sure the GPU supports bf16 for this model."
    if "no usable samples" in t:
        return "data", "None of the training examples could be rendered for this model. Check the dataset preview."
    if "connection" in t and ("huggingface" in t or "hf.co" in t):
        return "network", "Could not reach the Hugging Face Hub from the training machine. Check its internet connection."
    if "bitsandbytes" in t:
        return "deps", "bitsandbytes is missing or broken; QLoRA needs it on an NVIDIA GPU. Try LoRA instead, or reinstall bitsandbytes."
    return "error", "Something went wrong during training; the log above has the details."
