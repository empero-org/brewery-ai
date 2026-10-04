"""Recipe, brewing and tasting: training jobs, regimes and preference collection."""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

from homebrew_ai.agent.tools.base import ToolContext, ToolError, tool
from homebrew_ai.etf.io import iter_records, read_records, write_records
from homebrew_ai.etf.schema import content_text, normalize_record
from homebrew_ai.hardware.estimate import TrainShape, estimate_hours, fitting_gpus
from homebrew_ai.hardware.gpus import PRICES_CHECKED, generic, identify
from homebrew_ai.jobs.manager import active_job
from homebrew_ai.models.registry import get_model
from homebrew_ai.train.config import TrainJob, build_job, total_steps, validate_job

OBJECTIVE_WORDS = {"sft": "supervised fine-tuning (learn from example answers)", "cpt": "continued pretraining (read raw text)", "dpo": "preference tuning (prefer chosen over rejected answers)"}


def _compute_gpu(ctx: ToolContext) -> tuple[float | None, int, bool, Any]:
    hw = ctx.project.state.compute.hardware or {}
    gpus = hw.get("gpus") or []
    if not gpus:
        return None, 1, True, None
    vram = min(g.get("vram_gb") or 0 for g in gpus)
    known = identify(gpus[0].get("name", ""))
    torch = hw.get("torch") or {}
    bf16 = torch.get("bf16") if torch.get("bf16") is not None else (known.bf16 if known else True)
    gpu = known or generic(vram, bf16=bool(bf16), name=gpus[0].get("name", "GPU"))
    return float(vram), len(gpus), bool(bf16), gpu


def _training_set(ctx: ToolContext, name: str | None, objective: str) -> tuple[str, dict[str, Any]]:
    sets = ctx.project.state.training_sets
    if not sets:
        raise ToolError("no training set yet: import data and run build_training_set first")
    if name:
        if name not in sets:
            raise ToolError(f"unknown training set {name!r}; available: {', '.join(sets)}")
        chosen = name
    else:
        candidates = [k for k, v in sets.items() if objective in (v.get("objectives") or ["sft"])]
        if not candidates:
            raise ToolError(f"no training set is suitable for {objective}; build one (named e.g. '{objective}')")
        chosen = objective if objective in candidates else candidates[-1]
    entry = sets[chosen]
    if entry.get("objectives") and objective not in entry["objectives"]:
        raise ToolError(f"training set {chosen!r} cannot be used for {objective} (it supports {entry['objectives']})")
    return chosen, entry


def hf_token_for(model_id: str | None) -> str | None:
    """The user's HF token, but only when the worker needs it (gated base model or chat template)."""
    from homebrew_ai.data.hub import hf_token

    try:
        info = get_model(model_id) if model_id else None
    except Exception:
        info = None
    if info is not None and not info.variant.gated:
        source = info.variant.chat_template_from
        try:
            template_gated = bool(source) and get_model(source).variant.gated
        except Exception:
            template_gated = bool(source)
        if not template_gated:
            return None
    return hf_token()


def _requirements_for(info: Any, jobs: list[Any]) -> list[str]:
    from homebrew_ai.remote import bootstrap

    def field(j: Any, name: str) -> Any:
        return j.get(name) if isinstance(j, dict) else getattr(j, name, None)

    methods = [field(j, "method") for j in jobs]
    optimizers = [(field(j, "optim") or {}).get("optimizer") if isinstance(j, dict) else j.optim.optimizer for j in jobs]
    method = "qlora" if "qlora" in methods or not methods else methods[-1]
    return bootstrap.worker_requirements(info, method, optimizers)


def _known_job(ctx: ToolContext, job_id: str) -> dict[str, Any] | None:
    if job_id in ctx.project.state.drafts:
        return ctx.project.state.drafts[job_id]
    return ctx.project.job(job_id)


@tool(
    "propose_training_config",
    """Prepare one training stage with Homebrew's per-model guidelines: fills every hyperparameter, sizes batch and
gradient accumulation to the GPU, estimates memory/time/cost, and stores it as a draft. Never invent hyperparameters:
pass user wishes as `overrides` (e.g. {"epochs": 3, "lora_rank": 32, "learning_rate": 1e-4, "max_seq_len": 4096,
"beta": 0.2, "reasoning": "drop", "lora_targets": "attention", "train_embeddings": true, "trigger_word": "sks"}).
Image LoRAs render preview images while training (a "before" set at step 0, then one set per checkpoint) that the
user sees in watch_training: pass 2-4 "sample_prompts" with the trigger word (one close to the training images, one
new scene); optional "sample_every" (steps, 0 = off) and "sample_steps" (denoising steps, default 30).
objective: 'sft' (default), 'cpt' (continued pretraining on raw text) or 'dpo' (preference pairs).
For multi-stage regimes, set init_from_job to the job id of the previous stage (draft or finished).""",
    {
        "objective": {"type": "string", "enum": ["sft", "cpt", "dpo"]},
        "method": {"type": "string", "enum": ["lora", "qlora", "full"]},
        "training_set": {"type": "string"},
        "init_from_job": {"type": "string"},
        "overrides": {"type": "object"},
        "expert_override": {"type": "boolean", "description": "Allow values beyond the hard guidelines (only after the user explicitly asked)."},
    },
)
def propose_training_config(ctx: ToolContext, args: dict[str, Any]) -> Any:
    s = ctx.project.state
    if not s.base_model:
        raise ToolError("choose a base model first (select_base_model)")
    info = get_model(s.base_model)
    objective = args.get("objective") or "sft"
    set_name, ts = _training_set(ctx, args.get("training_set"), objective)
    if args.get("expert_override") and ctx.level != "expert":
        ctx.confirm_or_raise("Allow settings outside Homebrew's safety guidelines for this model?", details="Only do this if you know why you need it.")

    init_from = None
    stage = 1
    if args.get("init_from_job"):
        prev = _known_job(ctx, args["init_from_job"])
        if prev is None:
            raise ToolError(f"unknown job {args['init_from_job']!r}")
        if prev.get("base_model") != info.id:
            raise ToolError("a later stage must use the same base model as the stage it continues")
        init_from = {"job_id": args["init_from_job"], "kind": "full" if prev.get("method") == "full" else "adapter"}
        stage = int(prev.get("stage") or 1) + 1

    vram, n_gpus, bf16, gpu = _compute_gpu(ctx)
    num_train = int(ts.get("num_train") or 0)
    tok = ts.get("token_stats") or {}
    p95 = tok.get("p90") if objective != "cpt" else None
    if objective == "cpt" and tok.get("mean"):
        seq = int((args.get("overrides") or {}).get("max_seq_len") or info.guidelines.for_method("lora", "cpt").max_seq_len.default or 2048)
        num_train = max(1, math.ceil(num_train * tok["mean"] / seq))  # packed blocks

    methods = [args["method"]] if args.get("method") else ["lora", "qlora"]
    job = report = None
    for method in methods:
        if not info.guidelines.for_method(method, objective).allowed:
            continue
        job, report = build_job(
            model=info, project=s.name, method=method, objective=objective, init_from=init_from, stage=stage,
            train_path="data/train.jsonl", eval_path="data/eval.jsonl" if ts.get("eval") else None,
            num_train=num_train, num_eval=ts.get("num_eval"), vram_gb=vram, num_gpus=n_gpus, bf16=bf16,
            p95_tokens=p95, overrides=dict(args.get("overrides") or {}), expert_override=bool(args.get("expert_override")),
        )
        if not report["errors"]:
            break
    if job is None:
        raise ToolError(f"no allowed training method for {info.id} with objective {objective}")

    shape = TrainShape(method=job.method, objective=objective, seq_len=job.data.max_seq_len or (job.image.resolution if job.image else 1024), lora_rank=job.lora.rank if job.lora else 16, optimizer=job.optim.optimizer)
    estimate: dict[str, Any] = {}
    steps = total_steps(job)
    if info.modality == "image":
        work = steps * job.effective_batch
    else:
        mean_tokens = tok.get("mean") or (job.data.max_seq_len // 3)
        work = (ts.get("num_train") or num_train) * mean_tokens * job.optim.epochs * (2 if objective == "dpo" else 1)
    target_gpu = gpu
    rentals = fitting_gpus(info, shape, s.compute.provider)
    if target_gpu is None and rentals:
        from homebrew_ai.hardware.gpus import BY_KEY

        target_gpu = BY_KEY.get(rentals[0]["key"])
        estimate["assumes_rental"] = rentals[0]["gpu"]
    if target_gpu is not None:
        lo, hi = estimate_hours(info, shape, work, target_gpu, n_gpus)
        estimate.update(gpu=target_gpu.name, hours_low=round(lo, 2), hours_high=round(hi, 2))
        price = target_gpu.price(s.compute.provider)
        if price and (s.compute.kind != "local"):
            estimate.update(usd_low=round(lo * price, 2), usd_high=round(hi * price, 2), price_note=f"approx. on-demand prices checked {PRICES_CHECKED}")

    replaced: list[str] = []
    if not report["errors"]:
        draft = job.model_dump(mode="json")
        draft["_training_set"] = set_name
        # A new recipe for the same stage replaces the old draft, so start_training never trains both.
        key = (objective, stage, init_from["job_id"] if init_from else None)
        for old_id, old in list(s.drafts.items()):
            if (old.get("objective") or "sft", int(old.get("stage") or 1), (old.get("init_from") or {}).get("job_id")) == key:
                s.drafts.pop(old_id)
                replaced.append(old_id)
        for old in s.drafts.values():  # later stages now continue the replacement
            if (old.get("init_from") or {}).get("job_id") in replaced:
                old["init_from"] = {"job_id": job.job_id, "kind": "full" if job.method == "full" else "adapter"}
        s.drafts[job.job_id] = draft
        ctx.project.save()
    summary = {
        "job_id": job.job_id if not report["errors"] else None,
        "stored_as_draft": not report["errors"],
        "replaces_drafts": replaced or None,
        "objective": f"{objective}: {OBJECTIVE_WORDS[objective]}",
        "method": job.method,
        "base_model": info.id,
        "starts_from": init_from["job_id"] if init_from else "base model",
        "training_set": set_name,
        "examples": ts.get("num_train"),
        "settings": {
            "learning_rate": job.optim.learning_rate,
            "epochs": job.optim.epochs if info.modality == "text" else None,
            "max_steps": job.optim.max_steps,
            "effective_batch": job.effective_batch,
            "micro_batch": job.optim.micro_batch_size,
            "grad_accum": job.optim.grad_accum,
            "max_seq_len": job.data.max_seq_len or None,
            "lora": {"rank": job.lora.rank, "alpha": job.lora.alpha, "targets": job.lora.targets} if job.lora else None,
            "dpo_beta": job.dpo.beta if job.dpo else None,
            "image": {
                "resolution": job.image.resolution, "trigger_word": job.image.trigger_word,
                "preview_prompts": job.image.sample_prompts or "a few training captions",
                "preview_every": "with every checkpoint" if job.image.sample_every is None else (job.image.sample_every or "off"),
            } if job.image else None,
            "optimizer": job.optim.optimizer,
            "scheduler": job.optim.scheduler,
        },
        "steps": steps,
        "memory": report.get("fit", {}).get("memory") if report.get("fit") else None,
        "estimate": estimate or None,
        "errors": report["errors"],
        "warnings": report["warnings"],
        "guideline_notes": info.guidelines.for_method(job.method, objective).notes,
    }
    if ctx.level in ("builder", "expert") and not report["errors"]:
        rows = [[k, json.dumps(v) if isinstance(v, (dict, list)) else v] for k, v in summary["settings"].items() if v is not None]
        ctx.ui.table(f"Recipe {job.job_id} ({objective}, {job.method})", ["setting", "value"], rows)
    return summary


def _ordered_drafts(ctx: ToolContext, ids: list[str] | None) -> list[TrainJob]:
    drafts = ctx.project.state.drafts
    if not drafts:
        raise ToolError("no draft to start: use propose_training_config first")
    chosen = ids or list(drafts)
    jobs = []
    for jid in chosen:
        if jid not in drafts:
            raise ToolError(f"unknown draft {jid!r}; drafts: {', '.join(drafts)}")
        data = {k: v for k, v in drafts[jid].items() if not k.startswith("_")}
        jobs.append(TrainJob.model_validate(data))
    jobs.sort(key=lambda j: j.stage or 1)
    started = {j["job_id"]: j for j in ctx.project.state.jobs}
    in_batch = {j.job_id for j in jobs}
    for j in jobs:
        if j.init_from and j.init_from.job_id not in in_batch:
            prev = started.get(j.init_from.job_id)
            if prev is None or prev.get("state") != "completed":
                raise ToolError(f"stage {j.job_id} continues {j.init_from.job_id}, which has not finished; start them together or wait")
    return jobs


@tool(
    "start_training",
    """Start one or more drafted stages (asks the user to confirm, showing where it runs and the estimated time/cost).
Several stages (e.g. CPT then SFT) run back-to-back as a chain. Default: all drafts.""",
    {"job_ids": {"type": "array", "items": {"type": "string"}}},
    activity=None,
)
def start_training(ctx: ToolContext, args: dict[str, Any]) -> Any:
    s = ctx.project.state
    jobs = _ordered_drafts(ctx, args.get("job_ids"))
    info = get_model(jobs[0].base_model)
    for j in jobs:
        errors, _ = validate_job(j, info)
        if errors:
            raise ToolError(f"draft {j.job_id} has problems: {'; '.join(errors)}")
    compute = s.compute
    if compute.kind is None:
        raise ToolError("choose where to train first (use_local_computer or connect_server)")
    from homebrew_ai.remote import bootstrap

    needed = _requirements_for(info, jobs)
    if compute.kind == "ssh":
        if not compute.prepared:
            raise ToolError("the server is not prepared yet: run prepare_server")
        missing = bootstrap.missing_requirements(compute.requirements or [], needed)
        if missing:
            raise ToolError(
                f"the server was prepared for a different model or recipe and still needs: {', '.join(missing)}. "
                "Run prepare_server again (quick: what is installed stays)."
            )
    if compute.kind == "local":
        missing = bootstrap.check_local_worker(needed)["missing"]
        if missing:
            raise ToolError(f"training packages missing on this computer: {', '.join(missing)} (install_local_training_packages)")
    where = "this computer" if compute.kind == "local" else f"{compute.ssh['host']} ({compute.provider or 'SSH server'})"
    lines = [f"Where: {where}", f"Base model: {info.id}"]
    for j in jobs:
        lines.append(f"Stage {j.stage or 1}: {j.objective.upper()} with {j.method} — {total_steps(j)} steps" + (f" (continues {j.init_from.job_id})" if j.init_from else ""))
    if info.license.noncommercial:
        lines.append("Note: the base model's licence is NON-COMMERCIAL.")
    if compute.kind == "ssh":
        lines.append("The rented server keeps billing until you stop/destroy it.")
    ctx.confirm_or_raise(f"Start training {len(jobs)} stage(s)?", details="\n".join(lines))

    for j in jobs:
        draft = s.drafts[j.job_id]
        ts = s.training_sets[draft.get("_training_set", "train")]
        images_dir = ctx.project.abs(ts["images_dir"]) if ts.get("images_dir") else None
        ctx.jobs.stage(j, ctx.project.abs(ts["train"]), ctx.project.abs(ts["eval"]) if ts.get("eval") else None, images_dir)
    with ctx.ui.activity("Starting training" + (" (uploading data)" if compute.kind == "ssh" else "")):
        records = ctx.jobs.launch_chain(jobs, hf_token=hf_token_for(info.id))
    for rec in records:
        rec["training_set"] = s.drafts.get(rec["job_id"], {}).get("_training_set", "train")
    for j in jobs:
        s.drafts.pop(j.job_id, None)
    s.phase = "train"
    ctx.project.save()
    return {"started": [r["job_id"] for r in records], "where": where, "chain": len(records) > 1, "next": "use watch_training to follow progress"}


def _condense(status: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "state", "objective", "stage", "step", "max_steps", "epoch", "loss", "eval_loss", "learning_rate", "rewards_accuracy",
        "rewards_margin", "eta_s", "gpu_mem_gb", "message", "error", "error_kind", "hint", "final_metrics", "num_train", "precision",
        "last_sample_step", "preview_dir", "preview_error", "gallery_url",
    )
    out = {k: status[k] for k in keys if status.get(k) is not None}
    hist = status.get("loss_history") or []
    if hist:
        picks = hist[:: max(1, len(hist) // 8)][-8:]
        out["loss_trend"] = [round(v, 3) for _, v in picks]
    if status.get("state") in ("failed", "crashed"):
        out["log_tail"] = (status.get("log_tail") or "")[-1500:]
    return out


def _active_job_id(ctx: ToolContext, job_id: str | None) -> str:
    if job_id:
        return job_id
    job = active_job(ctx.project.state.jobs)
    if job is None:
        raise ToolError("no training job yet")
    return job["job_id"]


@tool("training_status", "Current state of a training job (default: the active or latest one): progress, loss, ETA, errors with a hint.", {"job_id": {"type": "string"}})
def training_status(ctx: ToolContext, args: dict[str, Any]) -> Any:
    jid = _active_job_id(ctx, args.get("job_id"))
    status = ctx.jobs.status(jid)
    return {"job_id": jid, **_condense(status)}


@tool(
    "watch_training",
    """Show the user a live progress view of training until the job (or chain) finishes, fails, or the user presses
Ctrl+C to stop watching (training continues). Returns the final status.""",
    {"job_id": {"type": "string"}, "max_minutes": {"type": "number"}},
)
def watch_training(ctx: ToolContext, args: dict[str, Any]) -> Any:
    jid = _active_job_id(ctx, args.get("job_id"))
    remote = ctx.project.state.compute.kind == "ssh"
    chain = [j["job_id"] for j in ctx.project.state.jobs if j.get("chain_id") and j.get("chain_id") == (ctx.project.job(jid) or {}).get("chain_id")] or [jid]

    fetched: dict[str, Any] = {}

    def with_previews(cid: str, st: dict[str, Any]) -> dict[str, Any]:
        """Download each new set of preview images so the user can open them while training runs."""
        step = st.get("last_sample_step")
        if step is None or (ctx.project.job(cid) or {}).get("modality") != "image":
            return st
        if fetched.get(cid) != step:
            fetched[cid] = step
            try:
                fetched[(cid, "dir")] = str(ctx.jobs.fetch(cid, f"samples/step_{int(step):06d}"))
            except Exception as exc:
                fetched[(cid, "dir")] = None
                st["preview_error"] = f"could not download the step {step} previews: {str(exc)[:120]}"
        if fetched.get((cid, "dir")):
            st["preview_dir"] = fetched[(cid, "dir")]
            from homebrew_ai.ui.gallery import get_gallery

            st["gallery_url"] = get_gallery(ctx.project).url
        return st

    def poll() -> dict[str, Any]:
        for cid in chain:
            st = ctx.jobs.status(cid)
            if st.get("state") not in ("completed",) or cid == chain[-1]:
                return {"job_id": cid, **with_previews(cid, st)}
        return {"job_id": chain[-1], **with_previews(chain[-1], ctx.jobs.status(chain[-1]))}

    final = ctx.ui.watch(poll, interval=20.0 if remote else 5.0, timeout_s=(args.get("max_minutes") or 0) * 60 or None)
    result = {"job_id": final.get("job_id", jid), **_condense(final)}
    if final.get("state") == "completed" and ctx.project.state.phase == "train":
        ctx.project.state.phase = "evaluate"
        ctx.project.save()
    return result


@tool(
    "open_preview_gallery",
    """Start (or show) the local preview gallery for image LoRAs: a web page on this computer where every checkpoint's
preview images sit side by side (step 0 = base model), refreshing while training runs. Returns the link to give the
user; they open it in their browser.""",
)
def open_preview_gallery(ctx: ToolContext, args: dict[str, Any]) -> Any:
    from homebrew_ai.ui.gallery import get_gallery

    jobs = [j["job_id"] for j in ctx.project.state.jobs if j.get("modality") == "image"]
    url = get_gallery(ctx.project).url
    ctx.ui.info(url, title="Preview gallery (open in your browser)", style="key")
    return {"url": url, "image_jobs": jobs, "note": "shown to the user" + ("" if jobs else "; no image LoRA run yet, previews appear once one trains")}


@tool("stop_training", "Stop a running training job after saving a checkpoint (asks the user first).", {"job_id": {"type": "string"}})
def stop_training(ctx: ToolContext, args: dict[str, Any]) -> Any:
    jid = _active_job_id(ctx, args.get("job_id"))
    ctx.confirm_or_raise(f"Stop training job {jid}? A checkpoint is saved first.")
    ctx.jobs.stop(jid)
    return {"stopping": jid}


def _completed(ctx: ToolContext, job_id: str | None) -> dict[str, Any]:
    jid = job_id or next((j["job_id"] for j in reversed(ctx.project.state.jobs) if j.get("state") == "completed"), None)
    if jid is None:
        raise ToolError("no finished training job yet")
    record = ctx.project.job(jid)
    if record is None:
        raise ToolError(f"unknown job {jid}")
    status = ctx.jobs.status(jid)
    if status.get("state") != "completed":
        raise ToolError(f"job {jid} is {status.get('state')}, not completed")
    return record


@tool(
    "test_model",
    """Taste the brew: run a few prompts through a finished model (on the machine that trained it) and show the answers,
optionally next to the original base model's answers. For image models, renders sample images.""",
    {
        "prompts": {"type": "array", "items": {"type": "string"}},
        "job_id": {"type": "string"},
        "compare_with_base": {"type": "boolean"},
        "system_prompt": {"type": "string"},
        "thinking": {"type": "boolean"},
        "max_new_tokens": {"type": "integer"},
    },
    ["prompts"],
)
def test_model(ctx: ToolContext, args: dict[str, Any]) -> Any:
    record = _completed(ctx, args.get("job_id"))
    prompts = [p for p in args["prompts"] if p.strip()][:8]
    if not prompts:
        raise ToolError("give at least one prompt")
    with tempfile.TemporaryDirectory() as tmp:
        pfile = Path(tmp) / "prompts.json"
        pfile.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        ctx.jobs.put_file(record["job_id"], pfile, "prompts.json")
    cmd = ["test", ".", "--prompts", "prompts.json", "--out", "test_out.json", "--max-new-tokens", str(int(args.get("max_new_tokens") or 300))]
    if args.get("compare_with_base"):
        cmd.append("--compare-base")
    if args.get("thinking"):
        cmd.append("--thinking")
    if args.get("system_prompt"):
        cmd += ["--system", args["system_prompt"]]
    with ctx.ui.activity("Loading the model and generating (this can take a minute)"):
        out = ctx.jobs.run_worker(record["job_id"], cmd, stdin=hf_token_for(record.get("base_model")), timeout=3600)
    rows = json.loads(out.strip().splitlines()[-1])
    if record.get("modality") == "image":
        local = ctx.jobs.fetch(record["job_id"], "samples")
        for r in rows:
            for k in ("finetuned", "base"):
                if r.get(k):
                    r[k] = str((local.parent / r[k]).resolve())
        ctx.ui.table("Sample images", ["prompt", "with LoRA", "base"], [[r["prompt"], r.get("finetuned"), r.get("base", "")] for r in rows])
        return {"samples": rows, "note": "images saved on this computer; the user can open them"}
    for r in rows:
        md = f"**Prompt:** {r['prompt']}\n\n**Brewed model:** {r['finetuned']}"
        if r.get("base"):
            md += f"\n\n**Base model:** {r['base']}"
        ctx.ui.markdown(md)
    return {"results": rows, "shown_to_user": True}


@tool(
    "fetch_results",
    "Download a finished run's files to this computer: 'final' (adapter/model), 'samples' (images), or 'results.json'.",
    {"job_id": {"type": "string"}, "what": {"type": "string", "enum": ["final", "samples", "results.json"]}},
)
def fetch_results(ctx: ToolContext, args: dict[str, Any]) -> Any:
    record = _completed(ctx, args.get("job_id"))
    with ctx.ui.activity("Downloading results"):
        path = ctx.jobs.fetch(record["job_id"], args.get("what") or "final")
    return {"downloaded_to": str(path)}


# --------------------------------------------------------------------------- #
# preference collection
# --------------------------------------------------------------------------- #


def _prompts_from_dataset(ctx: ToolContext, name: str, count: int, seed: int = 7) -> list[dict[str, Any]]:
    path = ctx.project.data_dir / f"{name}.jsonl"
    if not path.exists():
        raise ToolError(f"no dataset named {name!r}")
    prompts = []
    for rec in iter_records(path):
        if "messages" in rec:
            msgs = list(rec["messages"])
            while msgs and msgs[-1]["role"] != "user":
                msgs.pop()
            if msgs:
                prompts.append({"messages": [{"role": m["role"], "content": content_text(m.get("content"))} for m in msgs if m["role"] in ("system", "user", "assistant")]})
        elif "completion" in rec and rec["prompt"].strip():
            prompts.append({"messages": [{"role": "user", "content": rec["prompt"]}]})
    random.Random(seed).shuffle(prompts)
    return prompts[:count]


@tool(
    "generate_candidates",
    """Sample several different answers per prompt from a finished (SFT) model, as raw material for preference data.
Prompts come from a dataset (its user turns) or a given list. Then use review_candidates to pick winners.""",
    {
        "name": {"type": "string", "description": "Name for the candidate set."},
        "job_id": {"type": "string"},
        "from_dataset": {"type": "string"},
        "prompts": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer", "description": "Number of prompts (default 50)."},
        "candidates_per_prompt": {"type": "integer", "description": "Default 2."},
        "max_new_tokens": {"type": "integer"},
        "thinking": {"type": "boolean"},
    },
    ["name"],
)
def generate_candidates(ctx: ToolContext, args: dict[str, Any]) -> Any:
    from homebrew_ai.data.prepare import check_name

    name = check_name(args["name"])
    record = _completed(ctx, args.get("job_id"))
    count = max(1, min(int(args.get("count") or 50), 2000))
    if args.get("prompts"):
        prompts = [{"messages": [{"role": "user", "content": p}]} for p in args["prompts"][:count]]
    elif args.get("from_dataset"):
        prompts = _prompts_from_dataset(ctx, args["from_dataset"], count)
    else:
        raise ToolError("give from_dataset or prompts")
    if not prompts:
        raise ToolError("no usable prompts found")
    n = max(2, min(int(args.get("candidates_per_prompt") or 2), 4))
    with tempfile.TemporaryDirectory() as tmp:
        pfile = Path(tmp) / "cand_prompts.jsonl"
        write_records_raw(pfile, prompts)
        ctx.jobs.put_file(record["job_id"], pfile, "cand_prompts.jsonl")
    cmd = ["candidates", ".", "--prompts", "cand_prompts.jsonl", "--out", "candidates.jsonl", "--n", str(n), "--max-new-tokens", str(int(args.get("max_new_tokens") or 512))]
    if args.get("thinking"):
        cmd.append("--thinking")
    with ctx.ui.activity(f"Sampling {n} answers for {len(prompts)} prompts"):
        ctx.jobs.run_worker(record["job_id"], cmd, stdin=hf_token_for(record.get("base_model")), timeout=6 * 3600)
        local = ctx.jobs.fetch(record["job_id"], "candidates.jsonl")
    dest = ctx.project.data_dir / f"{name}.candidates.jsonl"
    dest.write_bytes(Path(local).read_bytes())
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines() if line.strip()]
    ex = rows[0] if rows else {}
    return {"candidates": name, "prompts": len(rows), "per_prompt": n, "example": {"prompt": content_text(ex.get("messages", [{}])[-1].get("content")) if ex else None, "answers": [c[:300] for c in ex.get("candidates", [])]}}


def write_records_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@tool(
    "review_candidates",
    """Turn sampled candidates into DPO preference pairs. mode='user': the user picks the better answer for each prompt
in the terminal. mode='ai': you judge every pair against the rubric. mode='ai_review': you judge, then the user checks
a sample of your verdicts. The rubric should describe what 'better' means for this user.""",
    {
        "candidates": {"type": "string", "description": "Name given to generate_candidates."},
        "mode": {"type": "string", "enum": ["user", "ai", "ai_review"]},
        "rubric": {"type": "string"},
        "output_dataset": {"type": "string"},
        "max_items": {"type": "integer"},
    },
    ["candidates", "mode", "output_dataset"],
)
def review_candidates(ctx: ToolContext, args: dict[str, Any]) -> Any:
    from homebrew_ai.data import synth
    from homebrew_ai.data.prepare import check_name

    src = ctx.project.data_dir / f"{check_name(args['candidates'])}.candidates.jsonl"
    if not src.exists():
        raise ToolError(f"no candidate set named {args['candidates']!r}")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: int(args.get("max_items") or len(rows))]
    out_name = check_name(args["output_dataset"])
    mode = args["mode"]
    rubric = args.get("rubric") or ctx.project.state.goal or "the more helpful, accurate and on-style answer"
    if mode in ("ai", "ai_review") and not args.get("rubric"):
        ctx.ui.info(f"Judging with this rubric: {rubric}", title="AI judge")
    pairs: list[dict[str, Any]] = []
    judged: list[tuple[int, str, str]] = []
    stats = {"A": 0, "B": 0, "tie": 0, "both_bad": 0, "skipped": 0}
    for i, row in enumerate(rows):
        cands = [c for c in row.get("candidates", []) if c.strip()]
        if len(cands) < 2:
            stats["skipped"] += 1
            continue
        prompt = content_text(row["messages"][-1].get("content")) if row.get("messages") else ""
        a, b = cands[0], cands[1]
        if mode == "user":
            verdict = ctx.ui.pick_better(prompt, a, b, i + 1, len(rows))
            if verdict == "stop":
                break
            judge = "user"
        else:
            try:
                res = synth.judge_pair(ctx.backend, prompt, a, b, rubric)
            except Exception:
                stats["skipped"] += 1
                continue
            verdict, judge = res["winner"], f"ai:{ctx.backend.label}"
            judged.append((len(pairs), prompt, res.get("reason", "")))
        if verdict not in ("A", "B"):
            stats["both_bad" if verdict == "both_bad" else "tie"] += 1
            continue
        stats[verdict] += 1
        chosen, rejected = (a, b) if verdict == "A" else (b, a)
        pairs.append({"messages": row["messages"], "chosen": chosen, "rejected": rejected, "meta": {"source": f"preferences:{args['candidates']}", "judge": judge}})
    if mode == "ai_review" and pairs:
        sample = random.Random(3).sample(range(len(pairs)), k=min(8, len(pairs)))
        flips = 0
        for k in sample:
            p = pairs[k]
            verdict = ctx.ui.pick_better(content_text(p["messages"][-1].get("content")), p["chosen"], p["rejected"], sample.index(k) + 1, len(sample))
            if verdict == "B":
                p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
                p["meta"]["judge"] = "user (corrected ai)"
                flips += 1
            elif verdict in ("tie", "both_bad"):
                p["meta"]["drop"] = True
            elif verdict == "stop":
                break
        pairs = [p for p in pairs if not p["meta"].get("drop")]
        stats["user_corrections"] = flips
    records = []
    for p in pairs:
        rec, _ = normalize_record(p)
        if rec is not None:
            records.append(rec)
    if not records:
        raise ToolError(f"no preference pairs were collected ({stats})")
    path = ctx.project.data_dir / f"{out_name}.jsonl"
    existing = read_records(path) if path.exists() else []
    write_records(path, existing + records)
    from homebrew_ai.agent.tools.datasets import register_dataset

    # the answers were written by the brewed model, so the pairs count as synthetic data
    register_dataset(ctx, out_name, path, "trace", len(existing) + len(records), {"type": "preferences", "from": args["candidates"], "mode": mode}, synthetic=True)
    return {"dataset": out_name, "pairs": len(records), "total": len(existing) + len(records), "verdicts": stats, "next": "build_training_set with this dataset (e.g. named 'dpo'), then propose_training_config(objective='dpo', init_from_job=...)"}
