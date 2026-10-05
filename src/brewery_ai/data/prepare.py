"""Importing datasets into ETF, mixing them, and splitting train/eval.

Every imported dataset becomes ``data/<name>.jsonl`` (ETF) plus
``data/<name>.meta.json`` recording where it came from, its license, the
mapping used and what was filtered — the model card is generated from these.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from brewery_ai.data.hub import hf_token
from brewery_ai.etf.convert import Mapping, convert_row, detect_mapping
from brewery_ai.etf.io import iter_records, write_records
from brewery_ai.etf.schema import normalize_record, record_fingerprint
from brewery_ai.etf.stats import compute_stats, record_chars

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
SETS_DIR = "_sets"  # training sets live in data/_sets/<name>/ (dataset names cannot start with "_")


def set_dir(data_dir: Path, set_name: str) -> Path:
    """Folder of a training set; separate from datasets so building a set never overwrites one."""
    return data_dir / SETS_DIR / check_name(set_name)


def check_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError("dataset names may only contain letters, digits, '-', '_' and '.' (max 64 chars)")
    return name


# --------------------------------------------------------------------------- #
# reading sources
# --------------------------------------------------------------------------- #


def _read_local(path: Path) -> Iterator[dict[str, Any]]:
    suffix = "".join(path.suffixes[-2:]) if path.suffix == ".gz" else path.suffix
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in (".txt", ".md") and f.is_file():
                yield {"text": f.read_text(encoding="utf-8", errors="replace"), "file": f.name}
        return
    if suffix in (".jsonl", ".jsonl.gz", ".ndjson"):
        from brewery_ai.etf.io import iter_raw

        for _, raw in iter_raw(path):
            if isinstance(raw, dict):
                yield raw
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        for row in data:
            if isinstance(row, dict):
                yield row
    elif suffix in (".csv", ".tsv"):
        with open(path, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
    elif suffix in (".txt", ".md"):
        yield {"text": path.read_text(encoding="utf-8", errors="replace"), "file": path.name}
    else:
        raise ValueError(f"unsupported file type {suffix!r} (use .jsonl, .json, .csv, .tsv, .parquet, .txt or a folder of .txt files)")


def read_source(source: dict[str, Any], *, max_rows: int | None = None, shuffle: bool = True, seed: int = 42) -> Iterator[dict[str, Any]]:
    kind = source.get("type", "hf")
    if kind == "local":
        rows: Iterable[dict[str, Any]] = _read_local(Path(source["path"]).expanduser())
        if max_rows is not None and shuffle:
            rows = list(rows)
            random.Random(seed).shuffle(rows)  # type: ignore[arg-type]
        for i, row in enumerate(rows):
            if max_rows is not None and i >= max_rows:
                break
            yield row
        return
    from datasets import load_dataset

    ds = load_dataset(source["id"], name=source.get("config"), split=source.get("split", "train"), streaming=True, token=hf_token(), revision=source.get("revision"))
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        yield row


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def import_dataset(
    data_dir: Path,
    name: str,
    source: dict[str, Any],
    *,
    mapping: dict[str, Any] | None = None,
    max_rows: int | None = 5000,
    min_chars: int = 1,
    max_chars: int | None = None,
    dedupe: bool = True,
    keep_reasoning: bool = True,
    license: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    check_name(name)
    rows_iter = read_source(source, max_rows=max_rows, seed=seed)
    head: list[dict[str, Any]] = []
    if mapping is None:
        for row in rows_iter:
            head.append(row)
            if len(head) >= 5:
                break
        guessed, why = detect_mapping(head)
        if guessed is None:
            raise ValueError(f"could not detect the dataset layout ({why}); pass an explicit mapping")
        map_obj = guessed
    else:
        map_obj = Mapping.from_dict(mapping)
        why = "explicit mapping"

    src_label = f"hf:{source['id']}" + (f"/{source['config']}" if source.get("config") else "") if source.get("type", "hf") == "hf" else f"local:{Path(source['path']).name}"
    meta = {"source": src_label}
    if license:
        meta["license"] = license

    counts: Counter = Counter()
    reasons: Counter = Counter()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def rows() -> Iterator[dict[str, Any]]:
        yield from head
        yield from rows_iter

    for row in rows():
        counts["read"] += 1
        try:
            raw = convert_row(row, map_obj, meta)
        except (KeyError, ValueError, TypeError) as exc:
            reasons[f"convert: {str(exc)[:60]}"] += 1
            continue
        record, issues = normalize_record(raw)
        if record is None:
            for issue in issues:
                if issue.level == "error":
                    reasons[issue.code] += 1
                    break
            continue
        if not keep_reasoning and "messages" in record:
            for m in record["messages"]:
                m.pop("reasoning", None)
        n = record_chars(record)
        if n < min_chars or (max_chars and n > max_chars):
            reasons["length_filter"] += 1
            continue
        if dedupe:
            fp = record_fingerprint(record)
            if fp in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(fp)
        out.append(record)

    if not out:
        raise ValueError(f"no usable records (read {counts['read']}; rejected: {dict(reasons.most_common(5))})")
    path = data_dir / f"{name}.jsonl"
    write_records(path, out)
    stats = compute_stats(out)
    manifest = {
        "name": name,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "source_label": src_label,
        "license": license,
        "mapping": map_obj.to_dict(),
        "mapping_explanation": why,
        "records": len(out),
        "read": counts["read"],
        "rejected": dict(reasons.most_common()),
        "stats": stats,
    }
    (data_dir / f"{name}.meta.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return {"path": str(path), "records": len(out), "read": counts["read"], "rejected": dict(reasons.most_common(8)), "mapping": map_obj.to_dict(), "stats": stats, "samples": out[:2]}


def load_manifest(data_dir: Path, name: str) -> dict[str, Any] | None:
    path = data_dir / f"{name}.meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# --------------------------------------------------------------------------- #
# mixing + split
# --------------------------------------------------------------------------- #


def suitable_objectives(records: list[dict[str, Any]]) -> list[str]:
    """Which training objectives a set of records can feed."""
    if not records:
        return []
    kinds = {("image" if "image" in r else "trace" if "messages" in r else "completion" if "completion" in r else "text") for r in records}
    if kinds == {"image"}:
        return ["sft"]
    out = ["cpt"]
    if kinds & {"trace", "completion"}:
        out.append("sft")
    if any(r.get("chosen") and r.get("rejected") for r in records):
        out.append("dpo")
    return out


def build_training_set(
    data_dir: Path,
    parts: list[dict[str, Any]],
    *,
    eval_fraction: float = 0.05,
    eval_max: int = 300,
    seed: int = 42,
    set_name: str = "train",
) -> dict[str, Any]:
    """Combine datasets (optionally capping each) into ``_sets/<set_name>/train.jsonl`` + ``eval.jsonl``."""
    rng = random.Random(seed)
    combined: list[dict[str, Any]] = []
    used = []
    for part in parts:
        name = check_name(part["name"])
        records = list(iter_records(data_dir / f"{name}.jsonl"))
        rng.shuffle(records)
        cap = part.get("max_records")
        if cap:
            records = records[: int(cap)]
        repeat = int(part.get("repeat", 1))
        combined.extend(records * max(repeat, 1))
        used.append({"name": name, "records": len(records), "repeat": repeat})
    if not combined:
        raise ValueError("no records to train on")
    kinds = {("image" if "image" in r else "trace" if "messages" in r else "completion" if "completion" in r else "text") for r in combined}
    if "image" in kinds and len(kinds) > 1:
        raise ValueError("image datasets cannot be mixed with text datasets")
    rng.shuffle(combined)
    n_eval = 0
    if "image" not in kinds and len(combined) >= 40 and eval_fraction > 0:
        n_eval = min(eval_max, max(1, int(len(combined) * eval_fraction)))
    eval_records, train_records = combined[:n_eval], combined[n_eval:]
    out = set_dir(data_dir, set_name)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    eval_file = out / "eval.jsonl"
    write_records(train_path, train_records)
    eval_path = None
    if eval_records:
        write_records(eval_file, eval_records)
        eval_path = str(eval_file)
    elif eval_file.exists():
        eval_file.unlink()
    return {
        "train": str(train_path),
        "eval": eval_path,
        "num_train": len(train_records),
        "num_eval": len(eval_records),
        "parts": used,
        "kind": "mixed" if len(kinds) > 1 else kinds.pop(),
        "objectives": suitable_objectives(train_records),
        "stats": compute_stats(train_records),
    }


# --------------------------------------------------------------------------- #
# token statistics with the real tokenizer + chat template
# --------------------------------------------------------------------------- #


def token_report(path: Path, model_info: Any, *, sample: int = 1500, candidate_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)) -> dict[str, Any]:
    """Render a sample of records exactly like training will and measure token lengths."""
    from brewery_ai.etf.render import ChatFormat, Renderer, RenderError
    from brewery_ai.etf.stats import token_stats

    records = []
    for i, rec in enumerate(iter_records(path)):
        if i >= sample:
            break
        records.append(rec)
    try:
        tok = load_control_tokenizer(model_info)
    except Exception as exc:
        chars = [record_chars(r) for r in records]
        approx = [max(1, int(c / 3.6)) + 32 for c in chars]
        return {"exact": False, "reason": f"tokenizer unavailable ({str(exc)[:120]}); lengths are estimated from characters", "tokens": token_stats(approx)}
    renderer = Renderer(tok, ChatFormat.from_dict(model_info.chat_format_dict()), max_len=10**9)
    lengths: list[int] = []
    trainable: list[int] = []
    failed: Counter = Counter()
    for rec in records:
        try:
            for s in renderer.encode(rec):
                lengths.append(s.num_tokens)
                trainable.append(s.num_trainable)
        except RenderError as exc:
            failed[str(exc)[:80]] += 1
    report = {"exact": True, "sampled_records": len(records), "tokens": token_stats(lengths), "trainable_tokens_mean": round(sum(trainable) / max(len(trainable), 1))}
    report["coverage"] = {str(L): round(sum(1 for n in lengths if n <= L) / max(len(lengths), 1), 4) for L in candidate_lengths}
    if failed:
        report["render_failures"] = dict(failed.most_common(5))
    return report


def load_control_tokenizer(model_info: Any) -> Any:
    """Tokenizer + chat template on the laptop (no torch needed)."""
    import os
    import warnings

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings("ignore", message=".*PyTorch.*")
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_info.id, token=hf_token())
    if model_info.variant.chat_template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(model_info.variant.chat_template_from, token=hf_token()).chat_template
    return tok
