"""CPU training backend: hand the job to the ``cpuforge`` worker (a single static binary, no Python or torch).

Selected with ``runtime.backend: cpuforge`` in job.yaml. ``brewery train`` still starts, stops and
reports the job exactly as before; only the engine differs:

1. render the ETF data with the model's own tokenizer and chat format (same code as the torch worker),
2. write pre-tokenized ``data/*.tok.jsonl`` and a ``cpuforge.toml`` next to job.yaml,
3. run ``cpuforge worker --config cpuforge.toml`` as a child process. It writes status.json,
   metrics.jsonl and ``final/`` in the same layout, and reacts to the STOP file.

Torch-free on purpose: it runs on a machine that only has the cpuforge binary and a tokenizer.
Supported: sft and cpt with lora/full on llama, mistral and qwen3 style models. See docs/cpuforge.md.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from brewery_ai.train.config import TrainJob, total_steps
from brewery_ai.train.status import STOP_FILE, StatusWriter, read_status

BINARY_ENV = "CPUFORGE_BIN"
MODEL_DIR_ENV = "CPUFORGE_MODEL_DIR"  # use this local folder instead of downloading base_model
TRAIN_TOK = "data/train.tok.jsonl"
EVAL_TOK = "data/eval.tok.jsonl"


class WorkerFailed(RuntimeError):
    """The cpuforge worker ended with an error; carries its own classification for status.json."""

    def __init__(self, message: str, kind: str = "error", hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


# --------------------------------------------------------------------------- #
# what cpuforge can and cannot run
# --------------------------------------------------------------------------- #


def unsupported(job: TrainJob) -> list[str]:
    """Reasons this job cannot run on the cpuforge backend (empty = fine)."""
    why: list[str] = []
    if job.modality != "text":
        why.append("the cpuforge backend trains text models only")
    if job.objective == "dpo":
        why.append("DPO is not supported by the cpuforge backend (use sft or cpt)")
    if job.method == "qlora":
        why.append("QLoRA needs a GPU quantisation stack; use method 'lora' or 'full' with the cpuforge backend")
    if job.init_from is not None:
        why.append("continuing from an earlier stage (init_from) is not supported by the cpuforge backend yet")
    if job.runtime.num_gpus > 1:
        why.append("the cpuforge backend runs on one machine's CPU; set num_gpus to 1")
    if job.lora is not None:
        lo = job.lora
        if lo.use_dora:
            why.append("DoRA is not supported by the cpuforge backend")
        if lo.dropout:
            why.append("LoRA dropout is not supported by the cpuforge backend (set it to 0)")
        if lo.train_experts or lo.train_embeddings:
            why.append("train_experts / train_embeddings are not supported by the cpuforge backend")
    opt = job.optim.optimizer.lower()
    if "adamw" not in opt:
        why.append(f"optimizer '{job.optim.optimizer}' is not available on the cpuforge backend (it uses AdamW)")
    return why


FULL_MEMORY_MAX_SEQ = 512  # measured on a 0.6B model: 512 tokens keeps all activations in ~5 GB; 1024 runs out of 8 GB


def _memory_mode(job: TrainJob) -> str:
    """Keep every activation (about 40% faster) while sequences are short; recompute them (less RAM) on long ones."""
    if not job.runtime.gradient_checkpointing:
        return "full"
    return "full" if job.data.max_seq_len <= FULL_MEMORY_MAX_SEQ else "checkpoint"


def _lora_alpha(job: TrainJob) -> float:
    """rsLoRA scales by alpha/sqrt(r) instead of alpha/r; an equivalent plain alpha is alpha*sqrt(r)."""
    lo = job.lora
    assert lo is not None
    return float(lo.alpha) * math.sqrt(lo.rank) if lo.use_rslora else float(lo.alpha)


# --------------------------------------------------------------------------- #
# config for the worker
# --------------------------------------------------------------------------- #


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)  # a JSON string is a valid TOML basic string


def build_config(job: TrainJob, info: Any, *, steps: int, model_dir: str | None = None) -> tuple[str, list[str]]:
    """The ``cpuforge.toml`` text for this job, and any settings that were approximated."""
    notes: list[str] = []
    o = job.optim
    batch = max(1, o.micro_batch_size * o.grad_accum)
    warmup = max(0, math.ceil(o.warmup_ratio * steps)) if o.warmup_ratio else 0
    if o.scheduler not in ("cosine", "cosine_with_restarts"):
        notes.append(f"scheduler '{o.scheduler}' ignored: cpuforge uses warmup + cosine decay")
    if job.runtime.precision in ("bf16", "fp16"):
        notes.append("cpuforge trains in fp32 on the CPU (precision setting ignored)")
    eval_every = job.runtime.eval_steps or max(1, steps // 5)
    save_every = job.runtime.save_steps or min(100, max(5, steps // 10))
    lines = [
        "# written by brewery_ai.train.cpuforge",
        "[run]",
        f"job_id = {_q(job.job_id)}",
        f"base_model = {_q(job.base_model)}",
        f"objective = {_q(job.objective)}",
        "",
        "[model]",
    ]
    lines.append(f"path = {_q(model_dir)}" if model_dir else f"hf = {_q(job.base_model)}")
    lines += [
        "",
        "[data]",
        f"train = {_q(TRAIN_TOK)}",
        f"eval = {_q(EVAL_TOK)}",
        f"seq_len = {int(job.data.max_seq_len)}",
        "",
    ]
    if job.method in ("lora", "qlora") and job.lora is not None:
        names = job.lora.targets if isinstance(job.lora.targets, list) else info.lora_targets(job.lora.targets)
        lines += [
            "[lora]",
            f"rank = {int(job.lora.rank)}",
            f"alpha = {_lora_alpha(job):g}",
            "targets = [" + ", ".join(_q(str(n)) for n in names) + "]",
            f"seed = {int(job.runtime.seed)}",
            "",
        ]
    lines += [
        "[train]",
        f"steps = {int(steps)}",
        f"batch = {batch}",
        f"lr = {o.learning_rate:g}",
        f"warmup_steps = {warmup}",
        f"weight_decay = {o.weight_decay:g}",
        f"clip = {o.max_grad_norm:g}",
        f"memory = {_q(_memory_mode(job))}",
        f"seed = {int(job.runtime.seed)}",
        f"eval_every = {int(eval_every)}",
        f"save_every = {int(save_every)}",
        "",
    ]
    return "\n".join(lines), notes


def find_binary() -> str:
    found = os.environ.get(BINARY_ENV) or shutil.which("cpuforge")
    if not found:
        raise FileNotFoundError(
            "the cpuforge program was not found. Install it from https://github.com/Sm0r3s-source/cpuforge/releases "
            f"and put it on PATH, or set {BINARY_ENV} to its location"
        )
    return found


def write_tokenized(path: Path, samples: list[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps({"input_ids": [int(t) for t in s.input_ids], "labels": [int(t) for t in s.labels]}) + "\n")
    os.replace(tmp, path)
    return len(samples)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def run_cpuforge(job: TrainJob, run_dir: Path, status: StatusWriter, resume: bool, stop_requested) -> dict[str, Any]:
    """Run ``job`` on the cpuforge worker. Mirrors ``runner.run_text``: returns ``{"metrics", "stopped"}``."""
    problems = unsupported(job)
    if problems:
        raise RuntimeError("; ".join(problems))
    from brewery_ai.train.samples import build_samples, chat_format_for, load_tokenizer

    run_dir = Path(run_dir).resolve()
    info = job.resolved_model()
    binary = find_binary()
    counts: dict[str, int] = {}
    n_train = n_eval = 0
    have_data = (run_dir / TRAIN_TOK).exists() and (run_dir / "cpuforge.toml").exists()

    status.set(state="preparing", message="loading tokenizer and rendering data", objective=job.objective, engine="cpuforge")
    if resume and have_data:
        n_train = sum(1 for _ in open(run_dir / TRAIN_TOK, encoding="utf-8"))
        n_eval = sum(1 for _ in open(run_dir / EVAL_TOK, encoding="utf-8")) if (run_dir / EVAL_TOK).exists() else 0
    else:
        tok = load_tokenizer(info, None)
        fmt = chat_format_for(info, job.data.reasoning)
        train_items, counts = build_samples(run_dir / job.data.train, tok, fmt, job.data.max_seq_len, job.data.overflow, job.runtime.seed, job.objective)
        if not train_items:
            raise RuntimeError(f"no usable samples after rendering for objective '{job.objective}': {json.dumps(counts)}")
        eval_items: list[Any] = []
        if job.data.eval and (run_dir / job.data.eval).exists():
            eval_items, _ = build_samples(run_dir / job.data.eval, tok, fmt, job.data.max_seq_len, job.data.overflow, job.runtime.seed, job.objective)
        n_train = write_tokenized(run_dir / TRAIN_TOK, train_items)
        if eval_items:
            n_eval = write_tokenized(run_dir / EVAL_TOK, eval_items)
        else:
            (run_dir / EVAL_TOK).unlink(missing_ok=True)
    job.data.num_train = n_train
    steps = total_steps(job)
    model_dir = os.environ.get(MODEL_DIR_ENV) or None
    config_text, notes = build_config(job, info, steps=steps, model_dir=model_dir)
    if not n_eval:
        config_text = config_text.replace(f"eval = {_q(EVAL_TOK)}\n", 'eval = ""\n')
    (run_dir / "cpuforge.toml").write_text(config_text, encoding="utf-8")
    if counts:
        status.set(data=counts)
    status.set(num_train=n_train, num_eval=n_eval, max_steps=steps, message="starting the cpuforge worker", **({"warning": "; ".join(notes)} if notes else {}))

    # From here on the worker owns status.json; we only watch it (two writers would overwrite each other).
    cmd = [binary, "worker", "--config", str(run_dir / "cpuforge.toml")] + (["--resume"] if resume else [])
    env = dict(os.environ)
    proc = subprocess.Popen(cmd, cwd=run_dir, env=env)
    asked = False
    while True:
        try:
            code = proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if not asked and stop_requested(run_dir):
                (run_dir / STOP_FILE).touch()  # a SIGTERM sent to us becomes the worker's own stop request
                asked = True
    final = read_status(run_dir) or {}
    status.data.update({k: v for k, v in final.items() if k not in ("pid", "started_at", "elapsed_s")})
    if code != 0 or final.get("state") == "failed":
        raise WorkerFailed(str(final.get("error") or f"cpuforge exited with code {code}"), str(final.get("error_kind") or "error"), final.get("hint"))
    stopped = final.get("state") == "stopped"
    if not stopped and (run_dir / "final").is_dir():
        # Same tokenizer files the torch worker saves (chat template and pad token included), so export/test find them.
        load_tokenizer(info, None).save_pretrained(str(run_dir / "final"))
    metrics = dict(final.get("final_metrics") or {})
    results_path = run_dir / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        results = {}
    results.update({"objective": job.objective, "engine": "cpuforge", "data": counts or results.get("data"), "stopped_early": stopped, "metrics": metrics})
    results_path.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return {"metrics": metrics, "stopped": stopped}
