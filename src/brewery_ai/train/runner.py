"""Worker entry point: ``brewery train <run_dir>/job.yaml``.

Runs one training job to completion, keeping ``status.json`` up to date so
the control plane (possibly on another machine) can follow along. A SIGTERM
saves a checkpoint and stops cleanly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from brewery_ai.train.config import TrainJob, total_steps
from brewery_ai.train.status import STOP_FILE, StatusWriter, classify_error, process_rank

log = logging.getLogger("brewery.train")
_STOP = {"requested": False}


def _on_sigterm(signum, frame):  # pragma: no cover - signal handler
    _STOP["requested"] = True


def stop_requested(run_dir: Path | None = None) -> bool:
    """True once a stop was asked for by SIGTERM or the run's STOP file."""
    if not _STOP["requested"] and run_dir is not None and (run_dir / STOP_FILE).exists():
        _STOP["requested"] = True
    return _STOP["requested"]


def all_ranks_agree(flag: bool) -> bool:
    """With several GPUs every process must stop at the same step, or the others wait forever."""
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor([1.0 if flag else 0.0], device="cuda" if torch.cuda.is_available() else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item() > 0)
    except Exception:  # pragma: no cover - best effort
        pass
    return flag


def usable_optimizer(name: str, status: StatusWriter | None = None) -> str:
    """Fall back to plain AdamW when an 8-bit optimizer is asked for but bitsandbytes is missing."""
    if "8bit" not in name and "bnb" not in name:
        return name
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        log.warning("bitsandbytes is not available; using adamw_torch instead of %s", name)
        if status is not None:
            status.set(warning=f"bitsandbytes missing: used adamw_torch instead of {name} (needs more GPU memory)")
        return "adamw_torch"
    return name


def latest_checkpoint(run_dir: Path) -> str | None:
    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    return str(ckpts[-1]) if ckpts else None


def training_arguments(job: TrainJob, run_dir: Path, precision: str, n_train: int, has_eval: bool):
    """Build ``TrainingArguments`` for whichever transformers version is installed."""
    from transformers import TrainingArguments

    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    steps = total_steps(job)
    log_every = max(1, min(job.runtime.logging_steps, max(steps // 20, 1)))
    eval_every = job.runtime.eval_steps or max(10, steps // 8)
    save_every = job.runtime.save_steps or max(25, steps // 4)
    warmup = job.optim.warmup_ratio
    kw: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "per_device_train_batch_size": job.optim.micro_batch_size,
        "per_device_eval_batch_size": job.optim.micro_batch_size,
        "gradient_accumulation_steps": job.optim.grad_accum,
        "learning_rate": job.optim.learning_rate,
        "num_train_epochs": job.optim.epochs,
        "max_steps": job.optim.max_steps or -1,
        "lr_scheduler_type": job.optim.scheduler,
        "weight_decay": job.optim.weight_decay,
        "max_grad_norm": job.optim.max_grad_norm,
        "optim": job.optim.optimizer,
        "logging_steps": log_every,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": save_every,
        "save_total_limit": job.runtime.save_total_limit,
        "seed": job.runtime.seed,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "gradient_checkpointing": job.runtime.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "disable_tqdm": True,
        "ddp_find_unused_parameters": False,
    }
    if "warmup_ratio" in fields:
        kw["warmup_ratio"] = warmup
    else:  # transformers >= 5.15: a float below 1 means "ratio"
        kw["warmup_steps"] = warmup if warmup < 1 else int(warmup)
    if has_eval:
        kw["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "steps"
        kw["eval_steps"] = eval_every
    if job.optim.micro_batch_size > 1:
        if "train_sampling_strategy" in fields:
            kw["train_sampling_strategy"] = "group_by_length"
        elif "group_by_length" in fields:
            kw["group_by_length"] = True
    if precision == "fp32" and "use_cpu" in fields:
        import torch

        kw["use_cpu"] = not torch.cuda.is_available() and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return TrainingArguments(**{k: v for k, v in kw.items() if k in fields})


def make_status_callback(status: StatusWriter, max_steps_hint: int):
    from transformers import TrainerCallback

    class StatusCallback(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def on_train_begin(self, args, state, control, **kw):
            status.set(state="training", step=state.global_step, max_steps=state.max_steps or max_steps_hint)
            self.t0 = time.time()
            self.start_step = state.global_step

        def on_log(self, args, state, control, logs=None, **kw):
            logs = dict(logs or {})
            step = state.global_step
            done = max(step - getattr(self, "start_step", 0), 1)
            elapsed = time.time() - self.t0
            remaining = max((state.max_steps or max_steps_hint) - step, 0)
            fields: dict[str, Any] = {"step": step, "max_steps": state.max_steps or max_steps_hint, "epoch": round(state.epoch or 0, 3), "eta_s": round(elapsed / done * remaining, 1)}
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm", "rewards/accuracy", "rewards/margin", "eval_rewards/accuracy"):
                if key in logs:
                    fields[key.replace("/", "_")] = logs[key]
            try:
                import torch

                if torch.cuda.is_available():
                    fields["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            except Exception:
                pass
            loss = logs.get("loss")
            if loss is not None and (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))):
                status.fail("loss became NaN", classify_error("nan loss")[1], kind="nan")
                control.should_training_stop = True
                return
            status.log_metrics({"step": step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}})
            status.set(**fields)

        def on_step_end(self, args, state, control, **kw):
            if all_ranks_agree(stop_requested(status.run_dir)):
                _STOP["requested"] = True
                control.should_save = True
                control.should_training_stop = True

        def on_save(self, args, state, control, **kw):
            status.set(last_checkpoint_step=state.global_step)

    return StatusCallback()


def resolve_start(job: TrainJob, run_dir: Path, info: Any, status: StatusWriter | None = None) -> str | None:
    """Folder to load weights from when this stage continues an earlier one (None = the base model).

    Adapter stages are merged into their own starting weights first; the merged
    model is cached in ``<previous run>/merged`` so later stages can reuse it.
    """
    if job.init_from is None:
        return None
    run_dir = Path(run_dir).resolve()  # callers may pass "." (the worker runs inside the run folder)
    prev_dir = run_dir.parent / job.init_from.job_id
    prev_job_path = prev_dir / "job.yaml"
    if not (prev_dir / "final").is_dir() or not prev_job_path.exists():
        raise RuntimeError(f"stage {job.init_from.job_id} has no finished model in {prev_dir}")
    prev = TrainJob.load(prev_job_path)
    if prev.method == "full":
        return str(prev_dir / "final")
    merged = prev_dir / "merged"
    if (merged / "config.json").exists():
        return str(merged)
    if process_rank() != 0:  # with several GPUs only the main process merges; the others wait for it
        while not (merged / "config.json").exists():
            time.sleep(5)
        return str(merged)
    if status is not None:
        status.set(state="preparing", message=f"merging stage {prev.job_id} into its base model")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brewery_ai.train import model as tmodel

    base_source = resolve_start(prev, prev_dir, info, status)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and tmodel.device_info()["bf16"]) or info.profile.loading.get("bf16_required") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_source or info.id, dtype=dtype, token=tmodel.hf_token(), device_map="auto" if torch.cuda.is_available() else None)
    merged_model = PeftModel.from_pretrained(base, prev_dir / "final").merge_and_unload()
    tmp = prev_dir / "merged.tmp"
    shutil.rmtree(tmp, ignore_errors=True)  # leftovers of an interrupted merge
    merged_model.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(prev_dir / "final").save_pretrained(tmp)
    shutil.rmtree(merged, ignore_errors=True)
    tmp.rename(merged)
    del merged_model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(merged)


def run_text(job: TrainJob, run_dir: Path, status: StatusWriter, resume: bool) -> dict[str, Any]:
    import torch
    from transformers import Trainer

    from brewery_ai.train import data as tdata
    from brewery_ai.train import model as tmodel

    info = job.resolved_model()
    objective = job.objective
    source = resolve_start(job, run_dir, info, status)

    status.set(state="preparing", message="loading tokenizer and rendering data", objective=objective, init_from=source)
    tok = tmodel.load_tokenizer(info, source)
    fmt = tdata.chat_format_for(info, job.data.reasoning)
    train_items, counts = tdata.build_samples(run_dir / job.data.train, tok, fmt, job.data.max_seq_len, job.data.overflow, job.runtime.seed, objective)
    if not train_items:
        raise RuntimeError(f"no usable samples after rendering for objective '{objective}': {json.dumps(counts)}")
    eval_items: list = []
    if job.data.eval and (run_dir / job.data.eval).exists():
        eval_items, _ = tdata.build_samples(run_dir / job.data.eval, tok, fmt, job.data.max_seq_len, job.data.overflow, job.runtime.seed, objective)
    status.set(data=counts, num_train=len(train_items), num_eval=len(eval_items))

    precision = tmodel.resolve_precision(job.runtime.precision, info)
    status.set(state="loading_model", message=f"loading {source or info.id} ({precision})", precision=precision)
    model = tmodel.load_model(info, job.method, precision, job.runtime.attn_implementation, source)
    fixed_tokens: list[int] = []
    if info.variant.fix_untrained_tokens and objective in ("sft", "dpo"):
        fixed_tokens = tmodel.fix_untrained_tokens(model, tok, info)
    if job.method in ("lora", "qlora"):
        model = tmodel.apply_lora(model, job, info, token_ids=fixed_tokens)
    elif job.runtime.gradient_checkpointing:
        model.config.use_cache = False
    trainable, total = tmodel.count_parameters(model)
    status.set(trainable_params=trainable, total_params=total)

    job.data.num_train = len(train_items)
    job.optim.optimizer = usable_optimizer(job.optim.optimizer, status)
    args = training_arguments(job, run_dir, precision, len(train_items), bool(eval_items))
    callbacks = [make_status_callback(status, total_steps(job))]
    if objective == "dpo":
        from brewery_ai.train.dpo import make_dpo_trainer_class, precompute_reference

        collator = tdata.PreferenceCollator(tok.pad_token_id)
        if job.runtime.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        status.set(state="preparing", message="computing reference scores for DPO")
        ref = precompute_reference(model, tdata.PreferenceDataset(train_items), collator, max(1, job.optim.micro_batch_size), status)
        train_ds = tdata.PreferenceDataset(train_items, ref)
        eval_ds = None
        if eval_items:
            eval_ds = tdata.PreferenceDataset(eval_items, precompute_reference(model, tdata.PreferenceDataset(eval_items), collator, max(1, job.optim.micro_batch_size)))
        dpo = job.dpo
        trainer = make_dpo_trainer_class()(
            model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator, processing_class=tok,
            callbacks=callbacks, beta=dpo.beta if dpo else 0.1, label_smoothing=dpo.label_smoothing if dpo else 0.0, sft_weight=dpo.sft_weight if dpo else 0.0,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tdata.TokenizedDataset(train_items),
            eval_dataset=tdata.TokenizedDataset(eval_items) if eval_items else None,
            data_collator=tdata.PadCollator(tok.pad_token_id),
            processing_class=tok,
            callbacks=callbacks,
        )
    checkpoint = latest_checkpoint(run_dir) if resume else None
    result = trainer.train(resume_from_checkpoint=checkpoint)

    status.set(state="saving", message="saving the final model")
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))  # writes on the main process only
    main_process = trainer.is_world_process_zero()
    if main_process:
        tok.save_pretrained(str(final_dir))
        try:
            from transformers import GenerationConfig

            gen = GenerationConfig(eos_token_id=tmodel.stop_token_ids(tok, info), pad_token_id=tok.pad_token_id, **info.sampling("default"), do_sample=True)
            gen.save_pretrained(str(final_dir))
        except Exception as exc:  # generation config is a nicety, never fatal
            log.warning("could not write generation_config.json: %s", exc)

    metrics = dict(result.metrics or {})
    if eval_items:
        try:
            metrics.update(trainer.evaluate())
        except Exception as exc:
            log.warning("final evaluation failed: %s", exc)
    stopped = _STOP["requested"]
    if main_process:
        results = {"objective": objective, "metrics": metrics, "data": counts, "stopped_early": stopped, "started_from": source, "repaired_tokens": fixed_tokens or None}
        (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    return {"metrics": metrics, "stopped": stopped}


def run_chain(job_paths: list[str | Path], resume: bool = False) -> int:
    """Run stages one after another (e.g. CPT → SFT); stop at the first failure."""
    for path in job_paths:
        code = run(path, resume=resume)
        if code != 0:
            return code
        if _STOP["requested"]:
            return 0
    return 0


def run(job_path: str | Path, resume: bool = False) -> int:
    job_path = Path(job_path).resolve()
    run_dir = job_path.parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if resume and process_rank() == 0:
        (run_dir / STOP_FILE).unlink(missing_ok=True)  # the stop that paused this run is over
    status = StatusWriter(run_dir)
    try:
        job = TrainJob.load(job_path)
        status.set(state="preparing", job_id=job.job_id, base_model=job.base_model, method=job.method, modality=job.modality, objective=job.objective, stage=job.stage)
        if job.modality == "image":
            from brewery_ai.train.image import run_image

            outcome = run_image(job, run_dir, status, resume)
        else:
            outcome = run_text(job, run_dir, status, resume)
        final = {k: v for k, v in (outcome.get("metrics") or {}).items() if isinstance(v, (int, float))}
        status.set(state="stopped" if outcome.get("stopped") else "completed", final_metrics=final, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), message="done")
        return 0
    except BaseException as exc:  # noqa: BLE001 - we want every failure in status.json
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        kind, hint = classify_error(text)
        if isinstance(exc, KeyboardInterrupt):
            status.set(state="stopped", message="interrupted")
            return 130
        status.fail(f"{type(exc).__name__}: {exc}", hint, kind)
        return 1
