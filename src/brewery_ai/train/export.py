"""Bottling: turn a finished run into a publishable folder and push it (worker side).

Runs wherever the run lives (laptop or rented GPU box), so a 16 GB merged model
never has to travel over a home internet connection.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from brewery_ai.train.config import TrainJob

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _dir_size_gb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**3, 3)


def copy_license_files(model_id: str, names: list[str], out_dir: Path) -> list[str]:
    copied = []
    if not names:
        return copied
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return copied
    for name in names:
        try:
            path = hf_hub_download(model_id, name, token=os.environ.get("HF_TOKEN"))
        except Exception:
            continue
        _copy(Path(path), out_dir / name)
        copied.append(name)
    return copied


def export_run(run_dir: str | Path, out_dir: str | Path, *, merge: bool = False, readme: str | Path | None = None, checkpoint: int | None = None) -> dict[str, Any]:
    """Package a run. ``checkpoint`` (image LoRAs) exports ``checkpoints/checkpoint-N`` instead of the final LoRA,
    with that step's preview images as the samples."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    job = TrainJob.load(run_dir / "job.yaml")
    info = job.resolved_model()
    final = run_dir / "final"
    if checkpoint is not None:
        if job.modality != "image":
            raise ValueError("exporting an earlier checkpoint is supported for image LoRAs")
        final = run_dir / "checkpoints" / f"checkpoint-{int(checkpoint)}"
        if not final.is_dir():
            kept = sorted(p.name.split("-")[-1] for p in (run_dir / "checkpoints").glob("checkpoint-*"))
            raise FileNotFoundError(f"no checkpoint at step {checkpoint}; kept checkpoints: {', '.join(kept) or 'none'}")
    if not final.is_dir():
        raise FileNotFoundError(f"{final} does not exist: the run has not finished")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    kind = "adapter"

    if job.modality == "image":
        for f in final.iterdir():
            if f.is_file():
                _copy(f, out_dir / f.name)
        samples = run_dir / "samples" if checkpoint is None else run_dir / "samples" / f"step_{int(checkpoint):06d}"
        if samples.is_dir():
            for img in sorted(samples.glob("*.png"))[:8]:
                _copy(img, out_dir / "samples" / img.name)
        kind = "lora"
    elif job.method == "full":
        for f in final.iterdir():
            if f.is_file() and f.name != "training_args.bin":
                _copy(f, out_dir / f.name)
        kind = "full"
    elif merge or job.init_from is not None:
        # an adapter from a later stage only makes sense on top of the earlier stages, so always merge
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from brewery_ai.train import model as tmodel
        from brewery_ai.train.runner import resolve_start

        dtype = torch.bfloat16 if torch.cuda.is_available() or info.profile.loading.get("bf16_required") else torch.float32
        start = resolve_start(job, run_dir, info)
        base = AutoModelForCausalLM.from_pretrained(
            start or info.id, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, token=tmodel.hf_token()
        )
        merged = PeftModel.from_pretrained(base, final).merge_and_unload()
        merged.save_pretrained(out_dir, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(final).save_pretrained(out_dir)
        if (final / "generation_config.json").exists():
            _copy(final / "generation_config.json", out_dir / "generation_config.json")
        kind = "merged"
    else:  # plain adapter on the original base model
        for name in (*ADAPTER_FILES, *TOKENIZER_FILES):
            if (final / name).exists():
                _copy(final / name, out_dir / name)

    lic = info.license
    copied = copy_license_files(info.id, list(lic.copy_files), out_dir)
    if lic.notice:
        (out_dir / "NOTICE").write_text(lic.notice + "\n", encoding="utf-8")
    if readme:
        _copy(Path(readme), out_dir / "README.md")
    results = run_dir / "results.json"
    summary = {
        "brewery_version": job.brewery_version,
        "base_model": job.base_model,
        "method": job.method,
        "export": kind,
        "lora": job.lora.model_dump() if job.lora else None,
        "optim": job.optim.model_dump(),
        "max_seq_len": job.data.max_seq_len,
        "results": json.loads(results.read_text()) if results.exists() else None,
        "checkpoint_step": checkpoint,
    }
    (out_dir / "brewery.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return {"out_dir": str(out_dir), "kind": kind, "files": sorted(p.name for p in out_dir.rglob("*") if p.is_file()), "size_gb": _dir_size_gb(out_dir), "license_files": copied}


def fix_card_repo(folder: str | Path, card_repo: str | None, repo_id: str) -> bool:
    """The model card is written when packaging; point its usage code at the repo it is finally uploaded to."""
    readme = Path(folder) / "README.md"
    if not card_repo or card_repo == repo_id or not readme.is_file():
        return False
    text = readme.read_text(encoding="utf-8")
    fixed = text.replace(f'"{card_repo}"', f'"{repo_id}"').replace(f"huggingface.co/{card_repo})", f"huggingface.co/{repo_id})")
    if fixed != text:
        readme.write_text(fixed, encoding="utf-8")
    return fixed != text


def push_folder(folder: str | Path, repo_id: str, *, private: bool = True, commit_message: str | None = None, token: str | None = None, card_repo: str | None = None) -> str:
    from huggingface_hub import HfApi

    fix_card_repo(folder, card_repo, repo_id)

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    # create_repo(exist_ok=True) keeps an existing repo's visibility: never put a "private" upload into a public repo
    if private and not getattr(api.repo_info(repo_id=repo_id, repo_type="model"), "private", True):
        raise RuntimeError(f"{repo_id} already exists and is PUBLIC, so nothing was uploaded. Choose another name, or upload it publicly on purpose.")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message or "Upload model brewed with Brewery",
    )
    return f"https://huggingface.co/{repo_id}"
