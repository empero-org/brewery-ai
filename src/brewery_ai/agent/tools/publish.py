"""Bottling (packaging) and sharing (Hugging Face upload)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from brewery_ai.agent.tools.base import ToolContext, ToolError, tool
from brewery_ai.agent.tools.models import whoami
from brewery_ai.agent.tools.training import _completed
from brewery_ai.data.prepare import load_manifest
from brewery_ai.models.registry import get_model
from brewery_ai.package.card import check_repo_name, render_card, stage_entry, suggest_repo_name
from brewery_ai.train.config import TrainJob


def _stage_chain(ctx: ToolContext, job_id: str) -> list[dict[str, Any]]:
    chain = []
    current = ctx.project.job(job_id)
    while current is not None:
        chain.append(current)
        prev = current.get("init_from")
        current = ctx.project.job(prev) if prev else None
    return list(reversed(chain))


def _results(ctx: ToolContext, record: dict[str, Any]) -> dict[str, Any] | None:
    local = Path(record["run_dir"]) / "results.json"
    if not local.exists() and record.get("remote_run_dir"):
        try:
            ctx.jobs.fetch(record["job_id"], "results.json")
        except Exception:
            return None
    try:
        return json.loads(local.read_text())
    except (OSError, ValueError):
        return None


def _datasets_for(ctx: ToolContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for rec in records:
        ts = ctx.project.state.training_sets.get(rec.get("training_set") or "train") or {}
        for part in ts.get("parts") or []:
            name = part["name"] if isinstance(part, dict) else part
            if name in seen:
                continue
            seen.add(name)
            entry = ctx.project.state.datasets.get(name)
            manifest = load_manifest(ctx.project.data_dir, name) or {}
            source = manifest.get("source") or (entry.sources[0] if entry and entry.sources else {})
            out.append(
                {
                    "name": name,
                    "hf_id": source.get("id") if source.get("type") == "hf" else None,
                    "records": manifest.get("records") or (entry.records if entry else None),
                    "license": manifest.get("license") or source.get("license"),
                    "synthetic": bool(entry and entry.synthetic) or source.get("type") == "synthetic",
                    "generator": source.get("model"),
                    "kind": entry.kind if entry else None,
                }
            )
    return out


@tool(
    "package_model",
    """Bottle a finished model for sharing: writes a model card (README.md with training details, data, licence notices
and credits) and exports the weights on the machine that trained them — the LoRA adapter (small) or, with merge=true,
a merged standalone model. Later regime stages are always merged. The repo name must follow the base model's licence
rules (e.g. Llama derivatives start with 'Llama'); use the suggested name if unsure. Image LoRAs: checkpoint_step
exports the checkpoint whose preview images the user liked best instead of the final LoRA.""",
    {
        "repo_name": {"type": "string", "description": "Model name on Hugging Face (without the user/org prefix)."},
        "description": {"type": "string", "description": "2-4 sentences: what the model does and for whom."},
        "merge": {"type": "boolean"},
        "checkpoint_step": {"type": "integer", "description": "Image LoRAs only: export the checkpoint saved at this step."},
        "job_id": {"type": "string"},
        "title": {"type": "string"},
    },
    ["description"],
)
def package_model(ctx: ToolContext, args: dict[str, Any]) -> Any:
    record = _completed(ctx, args.get("job_id"))
    job = TrainJob.load(Path(record["run_dir"]) / "job.yaml")
    info = get_model(job.base_model)
    name = args.get("repo_name") or suggest_repo_name(ctx.project.state.name, info)
    problems = check_repo_name(name, info)
    if problems:
        raise ToolError("; ".join(problems) + f". Suggested name: {suggest_repo_name(name, info)}")
    who = whoami()
    repo_id = f"{who['user']}/{name}" if who.get("logged_in") else name
    chain = _stage_chain(ctx, record["job_id"])
    stages = []
    for rec in chain:
        stage_job = TrainJob.load(Path(rec["run_dir"]) / "job.yaml")
        res = _results(ctx, rec)
        ts = ctx.project.state.training_sets.get(rec.get("training_set") or "train") or {}
        data_summary = f"{ts.get('num_train', '?')} {'images' if stage_job.modality == 'image' else 'examples'}"
        stages.append(stage_entry(stage_job, res, data_summary))
    hw = ctx.project.state.compute.hardware or {}
    gpus = hw.get("gpus") or []
    hardware = ", ".join(f"{g.get('name')}" for g in gpus) or None
    export_kind = "lora" if job.modality == "image" else ("full" if job.method == "full" else ("merged" if (args.get("merge") or job.init_from) else "adapter"))
    samples = None
    checkpoint = args.get("checkpoint_step")
    if checkpoint is not None and job.modality != "image":
        raise ToolError("checkpoint_step is only available for image LoRAs")
    if job.modality == "image" and checkpoint is not None:  # that step's preview set becomes the card's samples
        try:
            index = json.loads(Path(ctx.jobs.fetch(record["job_id"], "samples/index.json")).read_text())
            step_set = next((s for s in index.get("sets", []) if s.get("step") == int(checkpoint)), None)
            if step_set:
                samples = [{"prompt": p, "path": f"samples/sample_{i:02d}.png"} for i, p in enumerate(index.get("prompts", [])[: step_set.get("count", 0)])]
        except Exception:
            samples = None
    elif job.modality == "image":  # only the samples that were actually rendered after training
        samples = (_results(ctx, record) or {}).get("samples") or None
    langs = sorted({lang for d in ctx.project.state.datasets.values() for src in d.sources for lang in ([src.get("lang")] if src.get("lang") else [])})
    description = args["description"]
    if checkpoint is not None:
        description += f"\n\nThis LoRA is the checkpoint saved at step {int(checkpoint)} of {job.optim.max_steps or '?'} training steps."
    card = render_card(
        repo_id=repo_id, title=args.get("title") or name, description=description, info=info, stages=stages,
        export_kind=export_kind, datasets=_datasets_for(ctx, chain), hardware=hardware, samples=samples, language=langs or None,
    )
    readme = Path(record["run_dir"]) / "README.md"
    readme.write_text(card, encoding="utf-8")
    ctx.jobs.put_file(record["job_id"], readme, "README.md")
    cmd = ["export", ".", "--out", "export", "--readme", "README.md"]
    if args.get("merge"):
        cmd.append("--merge")
    if checkpoint is not None:
        cmd += ["--checkpoint", str(int(checkpoint))]
    from brewery_ai.agent.tools.training import hf_token_for

    with ctx.ui.activity("Packaging the model" + (" (merging weights)" if export_kind == "merged" else "")):
        out = ctx.jobs.run_worker(record["job_id"], cmd, stdin=hf_token_for(record.get("base_model")), timeout=4 * 3600)
    result = json.loads(out.strip().splitlines()[-1])
    entry = {
        "job_id": record["job_id"], "repo_name": name, "repo_id": repo_id, "kind": result.get("kind"), "size_gb": result.get("size_gb"),
        "where": "local" if not record.get("remote_run_dir") else "remote", "path": result.get("out_dir"), "readme": str(readme),
        "checkpoint_step": checkpoint,
    }
    ctx.project.state.exports.append(entry)
    ctx.project.state.phase = "publish"
    ctx.project.save()
    preview = "\n".join(card.split("\n---\n", 1)[-1].splitlines()[:28])
    ctx.ui.markdown(preview + "\n\n_…(README preview)_")
    return {**entry, "files": result.get("files"), "license_files": result.get("license_files"), "readme_written": True}


@tool(
    "upload_to_hf",
    """Upload the packaged model to Hugging Face (asks the user to confirm repo name, visibility and licence terms).
Needs Brewery to be logged in with a token that has write access. Default is a private repository.""",
    {"repo_id": {"type": "string", "description": "user-or-org/name; default: the packaged name under the user's account"}, "private": {"type": "boolean"}, "job_id": {"type": "string"}},
)
def upload_to_hf(ctx: ToolContext, args: dict[str, Any]) -> Any:
    exports = [e for e in ctx.project.state.exports if not args.get("job_id") or e["job_id"] == args["job_id"]]
    if not exports:
        raise ToolError("nothing packaged yet: run package_model first")
    export = exports[-1]
    who = whoami()
    if not who.get("logged_in"):
        raise ToolError("not logged in to Hugging Face: use hf_login (with a token that has write access)")
    if who.get("token_role") == "read":
        raise ToolError("the Hugging Face token is read-only; create a token with write access and run hf_login again")
    repo_id = args.get("repo_id") or (export["repo_id"] if "/" in export["repo_id"] else f"{who['user']}/{export['repo_name']}")
    info = get_model(TrainJob.load(Path(ctx.project.job(export['job_id'])['run_dir']) / 'job.yaml').base_model)
    problems = check_repo_name(repo_id, info)
    if problems:
        raise ToolError("; ".join(problems))
    private = args.get("private", True)
    details = [
        f"Repository: https://huggingface.co/{repo_id} ({'private' if private else 'PUBLIC'})",
        f"Contents: {export['kind']} weights, {export.get('size_gb') or '?'} GB, model card, licence files",
        f"Base model licence: {info.license.name}" + (" — NON-COMMERCIAL" if info.license.noncommercial else ""),
    ]
    ctx.confirm_or_raise("Upload this model to Hugging Face?", details="\n".join(details))
    from brewery_ai.data.hub import hf_token

    cmd = ["push", "export", "--repo", repo_id, "--private" if private else "--public", "--card-repo", export["repo_id"]]
    with ctx.ui.activity(f"Uploading to {repo_id}"):
        out = ctx.jobs.run_worker(export["job_id"], cmd, stdin=hf_token(), timeout=6 * 3600)
    url = json.loads(out.strip().splitlines()[-1]).get("url", f"https://huggingface.co/{repo_id}")
    ctx.project.state.uploads.append({"repo_id": repo_id, "url": url, "private": private, "job_id": export["job_id"]})
    ctx.project.state.phase = "done"
    ctx.project.save()
    reminder = None
    if ctx.project.state.compute.kind == "ssh":
        reminder = "Remind the user to stop/destroy the rented GPU server now if they are done, so billing stops."
    return {"url": url, "private": private, "reminder": reminder}
