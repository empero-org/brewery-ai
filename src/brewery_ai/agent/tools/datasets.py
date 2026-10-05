"""Ingredients: finding, importing, generating and checking training data."""

from __future__ import annotations

import json
import mimetypes
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from brewery_ai.agent.tools.base import ToolContext, ToolError, tool
from brewery_ai.data import hub, images, prepare, synth
from brewery_ai.etf.io import append_records, iter_raw, iter_records, read_records, write_records
from brewery_ai.etf.schema import content_text, normalize_record, record_fingerprint
from brewery_ai.models.registry import get_model
from brewery_ai.project import DatasetEntry

SYNTH_NOTICE = (
    "Synthetic examples are written by the AI model that guides you ({model}). Some AI providers' terms restrict using "
    "their outputs to train other models. Please check your provider's terms before publishing a model trained on this data."
)


def _base_model(ctx: ToolContext):
    mid = ctx.project.state.base_model
    return get_model(mid) if mid else None


def _register(ctx: ToolContext, name: str, path: Path, kind: str, records: int, source: dict[str, Any], synthetic: bool = False) -> None:
    register_dataset(ctx, name, path, kind, records, source, synthetic)


def register_dataset(ctx: ToolContext, name: str, path: Path, kind: str, records: int, source: dict[str, Any], synthetic: bool = False) -> None:
    """Record a dataset; extending an existing one keeps its earlier sources (licences, synthetic origin)."""
    old = ctx.project.state.datasets.get(name)
    sources = list(old.sources) if old and old.path == ctx.project.rel(path) else []
    if source not in sources:
        sources.append(source)
    was_synthetic = bool(old and old.path == ctx.project.rel(path) and old.synthetic)
    ctx.project.state.datasets[name] = DatasetEntry(path=ctx.project.rel(path), kind=kind, records=records, sources=sources, synthetic=synthetic or was_synthetic)
    ctx.project.save()


def _dataset_path(ctx: ToolContext, name: str) -> Path:
    """A dataset file, or the train file of a training set with that name (datasets win on a clash)."""
    path = ctx.project.data_dir / f"{name}.jsonl"
    if path.exists():
        return path
    sets = ctx.project.state.training_sets
    if name in sets:
        return ctx.project.abs(sets[name]["train"])
    folder = ctx.project.data_dir / name / "dataset.jsonl"
    if folder.exists():
        return folder
    known = sorted(ctx.project.state.datasets) + [f"{k} (training set)" for k in sorted(sets)]
    raise ToolError(f"no dataset named {name!r}; known: {', '.join(known) or 'none yet'}")


def _short_record(rec: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    out = json.loads(json.dumps(rec))
    for m in out.get("messages", []):
        for k in ("content", "reasoning"):
            if isinstance(m.get(k), str) and len(m[k]) > limit:
                m[k] = m[k][:limit] + "…"
    if isinstance(out.get("text"), str) and len(out["text"]) > limit:
        out["text"] = out["text"][:limit] + "…"
    out.pop("meta", None)
    return out


@tool(
    "search_datasets",
    """Search Hugging Face for datasets. Returns ids, downloads, licence, languages and tasks. Use short keyword queries
(e.g. 'pirate dialogue', 'medical qa', 'function calling', 'german instructions'); try several if needed.""",
    {
        "query": {"type": "string"},
        "language": {"type": "string", "description": "ISO code, e.g. en, de, fr"},
        "task": {"type": "string", "description": "e.g. text-generation, question-answering, text-to-image"},
        "modality": {"type": "string", "enum": ["text", "image"]},
        "limit": {"type": "integer"},
    },
    ["query"],
    activity="Searching Hugging Face",
)
def search_datasets(ctx: ToolContext, args: dict[str, Any]) -> Any:
    rows = hub.search_datasets(
        args["query"], limit=min(int(args.get("limit") or 8), 20), language=args.get("language"), task=args.get("task"), modality=args.get("modality"),
        safe=ctx.level != "expert",
    )
    return {"results": rows, "note": "check licences before importing; gated datasets need the user's acceptance on huggingface.co"}


@tool(
    "inspect_dataset",
    """Peek into a Hugging Face dataset without downloading it: configs, splits, row counts, columns, licence, a few
sample rows, and Brewery's guess of how to map it to ETF (text) or to image+caption (images).""",
    {"dataset_id": {"type": "string"}, "config": {"type": "string"}, "split": {"type": "string"}},
    ["dataset_id"],
    activity="Inspecting dataset",
)
def inspect_dataset(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return hub.inspect_dataset(args["dataset_id"], args.get("config"), args.get("split"))


@tool(
    "import_dataset",
    """Import a text dataset into the project as ETF: from Hugging Face (source='hf', dataset_id) or a local file/folder
(source='local', path: .jsonl/.json/.csv/.tsv/.parquet/.txt or a folder of .txt). Without a mapping the layout is
auto-detected (chat messages, ShareGPT, Alpaca, prompt/completion, preference, plain text). A mapping can name columns
and use templates, e.g. {"layout":"prompt_completion","prompt_template":"Translate to German: {en}","completion":"de"}.
Streams at most max_rows rows (default 5000), validates, de-duplicates and records the licence.""",
    {
        "name": {"type": "string", "description": "Short name for the imported dataset (letters, digits, - _ .)"},
        "source": {"type": "string", "enum": ["hf", "local"]},
        "dataset_id": {"type": "string"},
        "path": {"type": "string"},
        "config": {"type": "string"},
        "split": {"type": "string"},
        "mapping": {"type": "object"},
        "max_rows": {"type": "integer"},
        "license": {"type": "string"},
        "keep_reasoning": {"type": "boolean"},
        "max_chars": {"type": "integer", "description": "Drop records longer than this many characters."},
    },
    ["name", "source"],
    activity="Importing data",
)
def import_dataset(ctx: ToolContext, args: dict[str, Any]) -> Any:
    if args["source"] == "hf":
        if not args.get("dataset_id"):
            raise ToolError("dataset_id is required for source='hf'")
        source = {"type": "hf", "id": args["dataset_id"], "config": args.get("config"), "split": args.get("split") or "train"}
        license = args.get("license")
        if not license:
            try:
                license = hub.inspect_dataset(args["dataset_id"], args.get("config"), args.get("split"), n=1).get("license")
            except Exception:
                license = None
    else:
        if not args.get("path"):
            raise ToolError("path is required for source='local'")
        path = Path(args["path"]).expanduser()
        if not path.exists():
            raise ToolError(f"{path} does not exist")
        source = {"type": "local", "path": str(path)}
        license = args.get("license")
    try:
        report = prepare.import_dataset(
            ctx.project.data_dir, args["name"], source, mapping=args.get("mapping"), max_rows=args.get("max_rows", 5000),
            license=license, keep_reasoning=args.get("keep_reasoning", True), max_chars=args.get("max_chars"),
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    _register(ctx, args["name"], Path(report["path"]), "trace", report["records"], {**source, "license": license})
    report["samples"] = [_short_record(r) for r in report["samples"]]
    report["license"] = license
    return report


@tool(
    "import_images",
    """Import images for a text-to-image LoRA: a local folder (captions are read from same-named .txt files if present)
or a Hugging Face image dataset (needs image_column, optionally caption_column).""",
    {
        "name": {"type": "string"},
        "source": {"type": "string", "enum": ["local", "hf"]},
        "path": {"type": "string"},
        "dataset_id": {"type": "string"},
        "config": {"type": "string", "description": "Dataset config/subset, if it has several."},
        "split": {"type": "string", "description": "Default 'train'."},
        "image_column": {"type": "string"},
        "caption_column": {"type": "string"},
        "max_rows": {"type": "integer"},
        "trigger_word": {"type": "string"},
        "license": {"type": "string"},
    },
    ["name", "source"],
    activity="Importing images",
)
def import_images(ctx: ToolContext, args: dict[str, Any]) -> Any:
    try:
        if args["source"] == "local":
            report = images.import_image_folder(ctx.project.data_dir, args["name"], args.get("path") or "", trigger_word=args.get("trigger_word"))
            source = {"type": "local", "path": args.get("path")}
        else:
            if not args.get("dataset_id") or not args.get("image_column"):
                raise ToolError("dataset_id and image_column are required for source='hf'")
            with ctx.ui.activity("Importing images") as act:  # relabels the tool's spinner with progress
                report = images.import_image_hub(
                    ctx.project.data_dir, args["name"], args["dataset_id"], image_column=args["image_column"], caption_column=args.get("caption_column"),
                    config=args.get("config"), split=args.get("split") or "train", max_rows=int(args.get("max_rows") or 100),
                    trigger_word=args.get("trigger_word"), license=args.get("license"), on_progress=getattr(act, "update", None),
                )
            source = {"type": "hf", "id": args["dataset_id"], "config": args.get("config"), "license": args.get("license")}
    except (ValueError, FileNotFoundError) as exc:
        raise ToolError(str(exc)) from exc
    _register(ctx, args["name"], Path(report["path"]), "image", report["records"], source)
    return report


@tool(
    "add_examples",
    """Add hand-written training examples (ETF records) to a dataset, creating it if needed. Use it to capture examples
the user dictates, or a handful of 'golden' examples. Each record: {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}.""",
    {"name": {"type": "string"}, "records": {"type": "array", "items": {"type": "object"}}},
    ["name", "records"],
)
def add_examples(ctx: ToolContext, args: dict[str, Any]) -> Any:
    name = prepare.check_name(args["name"])
    path = ctx.project.data_dir / f"{name}.jsonl"
    existing = read_records(path) if path.exists() else []
    added, problems = [], []
    for i, raw in enumerate(args["records"]):
        raw = {**raw, "meta": {**(raw.get("meta") or {}), "source": "handwritten"}}
        rec, issues = normalize_record(raw)
        if rec is None:
            problems.append(f"record {i}: " + "; ".join(str(x) for x in issues if x.level == "error"))
        else:
            added.append(rec)
    if not added:
        raise ToolError("no valid records: " + " | ".join(problems[:5]))
    write_records(path, existing + added)
    _register(ctx, name, path, "trace", len(existing) + len(added), {"type": "handwritten"})
    return {"dataset": name, "added": len(added), "total": len(existing) + len(added), "problems": problems[:5]}


@tool(
    "preview_dataset",
    """Show the user a few records exactly as the chosen base model will see them during training (its chat template),
with the parts the model learns from highlighted. Also returns a compact version for you.""",
    {"name": {"type": "string", "description": "Dataset name, or 'train' for the built training set."}, "count": {"type": "integer"}},
    ["name"],
)
def preview_dataset(ctx: ToolContext, args: dict[str, Any]) -> Any:
    name = args["name"]
    path = _dataset_path(ctx, name)
    records = read_records(path, limit=max(1, min(int(args.get("count") or 2), 5)))
    if not records:
        raise ToolError("the dataset is empty")
    if "image" in records[0]:
        rows = [[r["image"], r.get("caption", "")] for r in records]
        ctx.ui.table(f"{name}: first images", ["image", "caption"], rows)
        return {"shown_to_user": len(rows), "records": records}
    info = _base_model(ctx)
    if info is None or info.modality != "text":
        for r in records:
            ctx.ui.markdown("```json\n" + json.dumps(_short_record(r, 600), ensure_ascii=False, indent=1) + "\n```")
        return {"shown_to_user": len(records), "note": "choose a text base model to see the rendered chat template", "records": [_short_record(r) for r in records]}
    from brewery_ai.etf.render import ChatFormat, Renderer, RenderError

    tok = _tokenizer(ctx, info)
    renderer = Renderer(tok, ChatFormat.from_dict(info.chat_format_dict()), max_len=10**9)
    out = []
    for i, rec in enumerate(records):
        try:
            segs = renderer.segments(rec)
        except RenderError as exc:
            out.append({"record": i, "error": str(exc)})
            continue
        ctx.ui.segments(f"Example {i + 1} as {info.variant.label} sees it (highlighted = what it learns)", segs)
        out.append({"record": i, "rendered_chars": sum(len(s) for s, _ in segs), "trainable_chars": sum(len(s) for s, t in segs if t)})
    return {"shown_to_user": len(records), "base_model": info.id, "details": out}


def _tokenizer(ctx: ToolContext, info):
    key = f"tok:{info.id}"
    if key not in ctx.cache:
        try:
            ctx.cache[key] = prepare.load_control_tokenizer(info)
        except Exception as exc:
            raise ToolError(f"could not load the {info.id} tokenizer ({str(exc)[:150]}). For gated models, accept the licence on huggingface.co and log in (hf_login).") from exc
    return ctx.cache[key]


@tool(
    "dataset_stats",
    """Statistics for a dataset: record counts, turns, share with reasoning/tool calls, duplicates, sources, licences, and
(with a base model chosen) exact token lengths under its chat template plus how many examples fit each max_seq_len.""",
    {"name": {"type": "string", "description": "Dataset name, or 'train'."}},
    ["name"],
    activity="Measuring the data",
)
def dataset_stats(ctx: ToolContext, args: dict[str, Any]) -> Any:
    from brewery_ai.etf.stats import file_stats

    name = args["name"]
    path = _dataset_path(ctx, name)
    out: dict[str, Any] = {"stats": file_stats(path)}
    info = _base_model(ctx)
    if info is not None and info.modality == "text":
        out["tokens"] = prepare.token_report(path, info)
    return out


def _texts(rec: dict[str, Any]) -> dict[str, list[str]]:
    """Text of a record by part: user/assistant turns, raw text, prompt/completion, chosen/rejected."""
    parts: dict[str, list[str]] = {"user": [], "assistant": [], "other": []}
    for m in rec.get("messages") or []:
        parts.get(m.get("role"), parts["other"]).append(content_text(m.get("content")))
    for key in ("chosen", "rejected", "completion"):
        if key in rec:
            value = rec[key]
            parts["assistant"].append(content_text(value) if not isinstance(value, list) else " ".join(content_text(m.get("content")) for m in value if isinstance(m, dict)))
    for key in ("text", "prompt", "system"):
        if isinstance(rec.get(key), str):
            parts["user" if key == "prompt" else "other"].append(rec[key])
    return parts


def _is_empty(rec: dict[str, Any]) -> bool:
    parts = _texts(rec)
    if rec.get("messages"):
        return any(not t.strip() for t in parts["user"]) or not parts["assistant"] or not parts["assistant"][-1].strip()
    if "text" in rec:
        return not str(rec.get("text") or "").strip()
    return not any(t.strip() for t in parts["assistant"]) or any(not t.strip() for t in parts["user"])


@tool(
    "clean_dataset",
    """Remove unwanted records from a text dataset; a backup is kept in data/_backups/. Use after looking at the data
(preview_dataset, dataset_stats): drop records with an empty prompt or answer, exact duplicates, answers shorter than
min_answer_chars, records longer than max_chars, records whose text matches drop_matching (a regular expression), or
records by 0-based position as shown by preview_dataset (drop_indices). Lines that are not valid ETF are dropped too.
Reports how many were removed and why.""",
    {
        "name": {"type": "string"},
        "drop_empty": {"type": "boolean", "description": "Drop records with an empty user prompt or answer (default true)."},
        "dedupe": {"type": "boolean", "description": "Drop exact duplicates (default true)."},
        "min_answer_chars": {"type": "integer"},
        "max_chars": {"type": "integer"},
        "drop_matching": {"type": "string", "description": "Regular expression, matched against all text of a record."},
        "drop_indices": {"type": "array", "items": {"type": "integer"}},
    },
    ["name"],
    activity="Cleaning the dataset",
)
def clean_dataset(ctx: ToolContext, args: dict[str, Any]) -> Any:
    name = prepare.check_name(args["name"])
    path = ctx.project.data_dir / f"{name}.jsonl"
    if not path.exists():
        raise ToolError(f"no text dataset named {name!r} (image datasets are re-imported instead)")
    try:
        pattern = re.compile(args["drop_matching"]) if args.get("drop_matching") else None
    except re.error as exc:
        raise ToolError(f"drop_matching is not a valid regular expression: {exc}") from exc
    drop_empty, dedupe = args.get("drop_empty", True), args.get("dedupe", True)
    min_answer, max_chars = int(args.get("min_answer_chars") or 0), int(args.get("max_chars") or 0)
    positions = {int(i) for i in args.get("drop_indices") or []}
    records = read_records(path)  # unreadable/invalid lines are skipped here and dropped on rewrite
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    invalid = sum(1 for _ in iter_raw(path)) - len(records)
    if invalid > 0:
        removed["invalid"] = invalid
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        parts = _texts(rec)
        everything = "\n".join(t for ts in parts.values() for t in ts)
        reason = None
        if i in positions:
            reason = "by_position"
        elif drop_empty and _is_empty(rec):
            reason = "empty"
        elif min_answer and sum(len(t) for t in parts["assistant"]) < min_answer:
            reason = "short_answer"
        elif max_chars and len(everything) > max_chars:
            reason = "too_long"
        elif pattern is not None and pattern.search(everything):
            reason = "matched_pattern"
        elif dedupe:
            fp = record_fingerprint(rec)
            reason = "duplicate" if fp in seen else None
            seen.add(fp)
        if reason:
            removed[reason] += 1
            if len(examples) < 3:
                examples.append({"position": i, "reason": reason, "record": _short_record(rec, 160)})
        else:
            kept.append(rec)
    if not removed:
        return {"dataset": name, "kept": len(kept), "removed": {}, "note": "nothing matched; the dataset is unchanged"}
    if not kept:
        raise ToolError(f"that would remove all {len(records)} records; nothing was changed")
    backups = ctx.project.data_dir / "_backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    path.replace(backup)
    write_records(path, kept)
    entry = ctx.project.state.datasets.get(name)
    if entry is not None:
        entry.records = len(kept)
        ctx.project.save()
    return {"dataset": name, "kept": len(kept), "removed": dict(removed), "examples_removed": examples, "backup": ctx.project.rel(backup)}


@tool(
    "build_training_set",
    """Combine one or more imported datasets into a named training set plus a small held-out eval set. Use different
names for different stages of a regime (e.g. 'cpt' with raw text, 'sft' with conversations, 'dpo' with preference
pairs). parts: [{"name": ..., "max_records": optional cap, "repeat": optional upsampling factor}].""",
    {
        "name": {"type": "string", "description": "Name of the training set (default 'train')."},
        "parts": {
            "type": "array",
            "items": {"type": "object", "properties": {"name": {"type": "string"}, "max_records": {"type": "integer"}, "repeat": {"type": "integer"}}, "required": ["name"]},
        },
        "eval_fraction": {"type": "number", "description": "Share held out for evaluation (default 0.05; text only)."},
    },
    ["parts"],
    activity="Building the training set",
)
def build_training_set(ctx: ToolContext, args: dict[str, Any]) -> Any:
    set_name = prepare.check_name(args.get("name") or "train")
    names = [p["name"] for p in args["parts"]]
    kinds = {ctx.project.state.datasets[n].kind for n in names if n in ctx.project.state.datasets}
    try:
        if kinds == {"image"}:
            report = images.build_image_training_set(ctx.project.data_dir, names, repeat=int(args["parts"][0].get("repeat") or 1), set_name=set_name)
        else:
            report = prepare.build_training_set(ctx.project.data_dir, args["parts"], eval_fraction=float(args.get("eval_fraction", 0.05)), set_name=set_name)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    info = _base_model(ctx)
    token_info = None
    if info is not None and info.modality == "text" and report["kind"] != "image":
        token_info = prepare.token_report(Path(report["train"]), info)
    entry = {
        "train": ctx.project.rel(report["train"]),
        "eval": ctx.project.rel(report["eval"]) if report.get("eval") else None,
        "num_train": report["num_train"],
        "num_eval": report["num_eval"],
        "kind": report["kind"],
        "objectives": report.get("objectives"),
        "parts": report.get("parts"),
        "token_stats": (token_info or {}).get("tokens"),
        "token_coverage": (token_info or {}).get("coverage"),
    }
    if report.get("images_dir"):
        entry["images_dir"] = ctx.project.rel(report["images_dir"])
    ctx.project.state.training_sets[set_name] = entry
    ctx.project.save()
    return {"training_set": set_name, **entry, "stats": report.get("stats")}


@tool(
    "generate_synthetic_data",
    """Generate training examples with your own model, shaped by the user's needs (opt-in; the user must agree).
mode='create': write `count` new conversations from a brief (what the target model should do, for whom, tone, topics),
with options for turns, reasoning traces, tool use, language, style, answer length, a system prompt and seed examples.
mode='transform': rewrite an existing dataset by an instruction (e.g. 'answer in the voice of a grumpy wizard; keep facts').
mode='preference': write DPO pairs (prompt, chosen answer meeting the brief, plausible worse answer; describe typical
flaws via `avoid`).
Start with ~10 to check quality with the user, then scale up.""",
    {
        "name": {"type": "string", "description": "Dataset to create (or extend)."},
        "mode": {"type": "string", "enum": ["create", "transform", "preference"]},
        "brief": {"type": "string", "description": "create: detailed description of the target behaviour. transform: the rewrite instruction."},
        "count": {"type": "integer"},
        "source_dataset": {"type": "string", "description": "transform: dataset to rewrite."},
        "turns": {"type": "integer"},
        "reasoning": {"type": "boolean"},
        "system_prompt": {"type": "string"},
        "language": {"type": "string"},
        "style": {"type": "string"},
        "answer_length": {"type": "string"},
        "avoid": {"type": "string"},
        "tools": {"type": "array", "items": {"type": "object"}},
        "seed_examples": {"type": "array", "items": {"type": "object"}},
        "effort": {
            "type": "string", "enum": ["off", "low", "medium", "high"],
            "description": "How hard the AI thinks per batch (reasoning models). Default: off for transform (rewriting "
                           "needs no deliberation; often 10-20x faster), low for create/preference. Raise it only when "
                           "a sample batch shows mistakes that more thinking would fix.",
        },
        "parallel": {"type": "integer", "description": "Requests at the same time (1-16, default 4 or BREWERY_AI_SYNTH_WORKERS)."},
    },
    ["name", "mode", "brief"],
)
def generate_synthetic_data(ctx: ToolContext, args: dict[str, Any]) -> Any:
    name = prepare.check_name(args["name"])
    count = max(1, min(int(args.get("count") or 10), 2000))
    s = ctx.project.state
    if not s.synthetic_terms_ack:
        ctx.ui.info(SYNTH_NOTICE.format(model=ctx.backend.label), title="Before generating data", style="warning")
        ctx.confirm_or_raise("Generate synthetic training data with this AI model?")
        s.synthetic_terms_ack = True
        ctx.project.save()
    if count > 50:
        calls = count // 4 + 2
        ctx.confirm_or_raise(f"Generate {count} examples? That is about {calls} requests to {ctx.backend.label} (this uses your API credit).")
    path = ctx.project.data_dir / f"{name}.jsonl"
    existing = len(read_records(path)) if path.exists() else 0
    origin = {"type": "synthetic", "model": ctx.backend.label, "brief": args["brief"][:300]}
    saved = {"n": 0}

    def save(batch: list[dict[str, Any]]) -> None:  # every finished batch goes to disk at once
        append_records(path, batch)
        saved["n"] += len(batch)
        _register(ctx, name, path, "trace", existing + saved["n"], origin, synthetic=True)

    speed = {"effort": args.get("effort") or ("off" if args["mode"] == "transform" else "low"), "workers": args.get("parallel")}
    with ctx.ui.activity("Brewing synthetic examples") as act:
        say = getattr(act, "update", lambda m: None)
        if args["mode"] == "transform":
            src_name = args.get("source_dataset") or ""
            src_path = ctx.project.data_dir / f"{prepare.check_name(src_name)}.jsonl"
            if not src_path.exists():
                raise ToolError(f"no dataset named {src_name!r} to transform")
            if src_path == path:
                raise ToolError("write the rewritten examples to a new dataset name (the source stays as it is)")
            source = read_records(src_path, limit=count)
            new, report = synth.transform(ctx.backend, source, args["brief"], on_progress=say, on_batch=save, **speed)
        elif args["mode"] == "preference":
            spec = synth.SynthSpec(brief=args["brief"], count=count, system_prompt=args.get("system_prompt"), language=args.get("language"), avoid=args.get("avoid"))
            new, report = synth.generate_preferences(ctx.backend, spec, on_progress=say, on_batch=save, **speed)
        else:
            spec = synth.SynthSpec(
                brief=args["brief"], count=count, turns=max(1, int(args.get("turns") or 1)), reasoning=bool(args.get("reasoning")),
                system_prompt=args.get("system_prompt"), language=args.get("language"), style=args.get("style"),
                answer_length=args.get("answer_length"), tools=list(args.get("tools") or []), seed_examples=list(args.get("seed_examples") or []),
                avoid=args.get("avoid"),
            )
            new, report = synth.generate(ctx.backend, spec, on_progress=say, on_batch=save, **speed)
    if not new:
        raise ToolError(f"no usable examples were produced ({report})")
    return {"dataset": name, "added": saved["n"], "total": existing + saved["n"], "report": report, "samples": [_short_record(r) for r in new[:2]]}


@tool(
    "caption_images",
    """Write captions for an image dataset with your own model's vision (asks the user first: the images are sent to the
AI provider). Includes the trigger word. Skips images that already have captions unless overwrite=true.""",
    {"name": {"type": "string"}, "trigger_word": {"type": "string"}, "style": {"type": "string"}, "overwrite": {"type": "boolean"}},
    ["name"],
)
def caption_images(ctx: ToolContext, args: dict[str, Any]) -> Any:
    folder = ctx.project.data_dir / prepare.check_name(args["name"])
    if not (folder / "dataset.jsonl").exists():
        raise ToolError(f"no image dataset named {args['name']!r}")
    if not ctx.backend.supports_vision():
        raise ToolError("the current AI model cannot see images; write captions manually or switch to a vision-capable model")
    records = list(iter_records(folder / "dataset.jsonl"))
    todo = [r for r in records if args.get("overwrite") or not r.get("caption")]
    if not todo:
        return {"captioned": 0, "note": "all images already have captions"}
    ctx.confirm_or_raise(f"Send {len(todo)} image(s) to {ctx.backend.label} to write captions?")
    captions: dict[str, str] = {}
    with ctx.ui.activity("Captioning images") as act:
        for i, r in enumerate(todo, 1):
            img_path = folder / r["image"]
            media = mimetypes.guess_type(img_path.name)[0] or "image/png"
            data = _downscaled(img_path)
            captions[r["image"]] = synth.caption_image(ctx.backend, data, "image/jpeg" if data[:3] == b"\xff\xd8\xff" else media, trigger_word=args.get("trigger_word"), style=args.get("style") or "concise, descriptive")
            if hasattr(act, "update"):
                act.update(f"{i}/{len(todo)} images")
    changed = images.set_captions(folder, captions)
    return {"captioned": changed, "examples": list(captions.items())[:3]}


def _downscaled(path: Path, max_side: int = 1024) -> bytes:
    import io

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
