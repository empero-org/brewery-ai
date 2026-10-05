"""Model cards (README.md) for brewed models.

The card documents what was trained and how (every stage of the regime), on
which data (sources, licences, synthetic share), the base model's licence
obligations, how to use the result, and credits Empero and everyone whose work
went into it.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import yaml

from brewery_ai import about

OBJECTIVE_NAMES = {"sft": "Supervised fine-tuning (SFT)", "cpt": "Continued pretraining (CPT)", "dpo": "Direct preference optimization (DPO)"}
METHOD_NAMES = {"lora": "LoRA", "qlora": "QLoRA (4-bit base)", "full": "full fine-tuning"}


def _front_matter(meta: dict[str, Any]) -> str:
    clean = {k: v for k, v in meta.items() if v not in (None, [], {}, "")}
    return "---\n" + yaml.safe_dump(clean, sort_keys=False, allow_unicode=True) + "---\n"


def _usage_text(repo_id: str, base_id: str, export_kind: str, trigger: str | None, modality: str, pipeline_class: str | None) -> str:
    if modality == "image":
        prompt = f"{trigger}, " if trigger else ""
        return f"""```python
import torch
from diffusers import {pipeline_class or "DiffusionPipeline"}

pipe = {pipeline_class or "DiffusionPipeline"}.from_pretrained("{base_id}", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}")
image = pipe("{prompt}a cozy reading nook, warm light", num_inference_steps=40).images[0]
image.save("brew.png")
```
Qwen-Image 2.1 needs a recent diffusers (`pip install git+https://github.com/huggingface/diffusers`)."""
    if export_kind == "adapter":
        return f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_id}", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [{{"role": "user", "content": "Hello!"}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```"""
    return f"""```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [{{"role": "user", "content": "Hello!"}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```
Also works with vLLM, SGLang and other runtimes that load Hugging Face checkpoints."""


def render_card(
    *,
    repo_id: str,
    title: str,
    description: str,
    info: Any,
    stages: list[dict[str, Any]],
    export_kind: str,
    datasets: list[dict[str, Any]],
    hardware: str | None = None,
    samples: list[dict[str, str]] | None = None,
    language: list[str] | None = None,
) -> str:
    """Render README.md. ``stages`` lists every stage (oldest first) with its job settings and results."""
    lic = info.license
    modality = info.modality
    last = stages[-1]
    objectives = sorted({s["objective"] for s in stages})
    methods = sorted({s["method"] for s in stages})
    hf_datasets = sorted({d["hf_id"] for d in datasets if d.get("hf_id")})
    trigger = (last.get("image") or {}).get("trigger_word") if modality == "image" else None

    meta: dict[str, Any] = {
        "base_model": info.id,
        "base_model_relation": "adapter" if export_kind in ("adapter", "lora") else "finetune",
        "library_name": "diffusers" if modality == "image" else ("peft" if export_kind == "adapter" else "transformers"),
        "license": lic.hub_id or lic.id,
        "pipeline_tag": "text-to-image" if modality == "image" else "text-generation",
        "datasets": hf_datasets,
        "language": language,
        "tags": sorted({*about.HUB_TAGS, *objectives, *("lora" if m in ("lora", "qlora") else "full-finetune" for m in methods), info.family.replace("_", "-"), modality}),
    }
    if (lic.hub_id or "") == "other":
        meta["license_name"] = lic.id
        meta["license_link"] = lic.url
    if modality == "image":
        meta["tags"] = sorted(set(meta["tags"]) | {"text-to-image", "diffusers", "template:sd-lora"})
        meta["instance_prompt"] = trigger
        if samples:
            meta["widget"] = [{"text": s["prompt"], "output": {"url": s["path"]}} for s in samples[:6]]

    out = [_front_matter(meta), f"# {title}\n"]
    out.append(f"{description.strip()}\n")
    out.append(f"> {about.CREDIT_LINE}, starting from [{info.variant.label}](https://huggingface.co/{info.id}).\n")

    notices = []
    if lic.noncommercial:
        notices.append(f"**Non-commercial use only.** The base model is released under the [{lic.name}]({lic.url}); this derivative and its outputs may not be used commercially.")
    if lic.attribution:
        notices.append(f"**{lic.attribution}.**")
    if lic.notice:
        notices.append(lic.notice)
    if lic.id == "gemma":
        notices.append("Use of this model is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which you must pass on to anyone you share it with.")
    if notices:
        out.append("## License notice\n\n" + "\n\n".join(notices) + "\n")

    out.append("## How to use\n\n" + _usage_text(repo_id, info.id, export_kind, trigger, modality, info.profile.pipeline.get("pipeline_class")) + "\n")
    if samples and modality == "image":
        out.append("## Samples\n\n<Gallery />\n")

    rows = ["| Stage | Objective | Method | Data | Steps | Result |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(stages, 1):
        res = s.get("results") or {}
        m = res.get("metrics") or {}
        bits = []
        for key, label in (("train_loss", "train loss"), ("eval_loss", "eval loss"), ("eval_rewards/accuracy", "pref. accuracy")):
            if isinstance(m.get(key), (int, float)):
                bits.append(f"{label} {m[key]:.3f}")
        rows.append(f"| {i} | {OBJECTIVE_NAMES.get(s['objective'], s['objective'])} | {METHOD_NAMES.get(s['method'], s['method'])} | {s.get('data_summary', '-')} | {s.get('steps', '-')} | {', '.join(bits) or '-'} |")
    out.append("## Training\n\n" + "\n".join(rows) + "\n")

    for i, s in enumerate(stages, 1):
        hp = {k: v for k, v in (s.get("hyperparameters") or {}).items() if v not in (None, {}, [])}
        out.append(f"<details><summary>Stage {i} hyperparameters</summary>\n\n```yaml\n{yaml.safe_dump(hp, sort_keys=False)}```\n</details>\n")
    if hardware:
        out.append(f"Trained on {hardware}.\n")

    if datasets:
        drows = ["| Dataset | Records | License | Notes |", "|---|---|---|---|"]
        for d in datasets:
            name = f"[{d['hf_id']}](https://huggingface.co/datasets/{d['hf_id']})" if d.get("hf_id") else d.get("name", "-")
            notes = []
            if d.get("synthetic"):
                notes.append(f"synthetic ({d.get('generator', 'AI-generated')})")
            if d.get("kind"):
                notes.append(d["kind"])
            drows.append(f"| {name} | {d.get('records', '-')} | {d.get('license') or 'unknown'} | {', '.join(notes) or '-'} |")
        out.append("## Data\n\n" + "\n".join(drows) + "\n\nTraining data was stored in ETF (Empero Trace Format) and rendered with the base model's own chat template.\n")
        if any(d.get("synthetic") for d in datasets):
            out.append("Part of the data was generated synthetically by an AI model and reviewed by the author.\n")

    out.append(
        "## Limitations\n\nThis model inherits the capabilities, biases and limitations of its base model and its training data. "
        "It can produce incorrect or inappropriate output; evaluate it for your use case before relying on it.\n"
    )
    credits = [f"- Brewed with [{about.PRODUCT}]({about.REPO_URL}) by [{about.ORG}]({about.WEBSITE}) — {about.ORG_TAGLINE}.", f"- Base model: [{info.id}](https://huggingface.co/{info.id}) by {info.variant.publisher or info.profile.vendor}."]
    credits += [f"- Dataset: [{h}](https://huggingface.co/datasets/{h})" for h in hf_datasets]
    out.append("## Credits\n\n" + "\n".join(credits) + "\n")
    out.append(f"<sub>Card generated by Brewery {about.VERSION} on {date.today().isoformat()}.</sub>\n")
    return "\n".join(out)


def stage_entry(job: Any, results: dict[str, Any] | None, data_summary: str) -> dict[str, Any]:
    """Compact description of one finished stage for :func:`render_card`."""
    from brewery_ai.train.config import total_steps

    hp: dict[str, Any] = {
        "objective": job.objective,
        "method": job.method,
        "learning_rate": job.optim.learning_rate,
        "epochs": job.optim.epochs if job.modality == "text" else None,
        "max_steps": job.optim.max_steps,
        "effective_batch_size": job.effective_batch,
        "optimizer": job.optim.optimizer,
        "scheduler": job.optim.scheduler,
        "warmup_ratio": job.optim.warmup_ratio,
        "max_seq_len": job.data.max_seq_len or None,
    }
    if job.lora:
        hp["lora"] = {"rank": job.lora.rank, "alpha": job.lora.alpha, "dropout": job.lora.dropout, "targets": job.lora.targets}
    if job.dpo:
        hp["dpo"] = job.dpo.model_dump()
    if job.image:
        hp["image"] = {"resolution": job.image.resolution, "trigger_word": job.image.trigger_word, "caption_dropout": job.image.caption_dropout}
    return {
        "objective": job.objective,
        "method": job.method,
        "steps": total_steps(job),
        "results": results,
        "data_summary": data_summary,
        "hyperparameters": hp,
        "image": hp.get("image"),
    }


def suggest_repo_name(project_name: str, info: Any) -> str:
    import re

    base = re.sub(r"[^A-Za-z0-9._-]+", "-", project_name).strip("-") or "brew"
    lic = info.license
    if lic.name_prefix and not base.lower().startswith(lic.name_prefix.lower()):
        size = info.variant.label.split()[-1] if info.variant.label else ""
        base = f"{lic.name_prefix}-{base}" if not size else f"{lic.name_prefix}-{size}-{base}"
    if info.license.id == "qwen-research" and base.lower().startswith("qwen"):
        base = base[4:].lstrip("-._") or "brew"
    return base[:96]


def check_repo_name(name: str, info: Any) -> list[str]:
    problems = []
    lic = info.license
    if lic.name_prefix and not name.split("/")[-1].lower().startswith(lic.name_prefix.lower()):
        problems.append(f"the {lic.name} requires the model name to start with '{lic.name_prefix}'")
    if lic.id == "qwen-research" and name.split("/")[-1].lower().startswith("qwen"):
        problems.append("the Qwen Research License does not allow 'Qwen' as the primary name of a derivative")
    return problems


def summarize_results(results: dict[str, Any] | None) -> str:
    if not results:
        return "-"
    return json.dumps({k: v for k, v in (results.get("metrics") or {}).items() if isinstance(v, (int, float))})[:200]
