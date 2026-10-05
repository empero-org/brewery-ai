"""Quick statistics over ETF datasets (no tokenizer needed)."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from brewery_ai.etf.io import iter_records
from brewery_ai.etf.schema import record_fingerprint, record_kind


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    values = sorted(values)

    def pct(p: float) -> int:
        idx = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
        return values[idx]

    return {"min": values[0], "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": values[-1]}


def record_chars(record: dict[str, Any]) -> int:
    kind = record_kind(record)
    if kind == "trace":
        return sum(len(m.get("content", "")) + len(m.get("reasoning", "")) for m in record["messages"])
    if kind == "image":
        return len(record.get("caption", ""))
    return len(record["text"])


def compute_stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    kinds: Counter = Counter()
    sources: Counter = Counter()
    licenses: Counter = Counter()
    languages: Counter = Counter()
    turns: list[int] = []
    chars: list[int] = []
    answer_chars: list[int] = []
    with_reasoning = with_tools = with_system = synthetic = 0
    seen: set[str] = set()
    duplicates = 0
    total = 0
    for record in records:
        total += 1
        kind = record_kind(record)
        kinds[kind] += 1
        fp = record_fingerprint(record)
        if fp in seen:
            duplicates += 1
        seen.add(fp)
        chars.append(record_chars(record))
        meta = record.get("meta") or {}
        if meta.get("source"):
            sources[str(meta["source"])] += 1
        if meta.get("license"):
            licenses[str(meta["license"])] += 1
        if meta.get("lang"):
            languages[str(meta["lang"])] += 1
        if meta.get("synthetic"):
            synthetic += 1
        if kind == "trace":
            msgs = record["messages"]
            turns.append(sum(1 for m in msgs if m["role"] == "assistant"))
            if any(m.get("reasoning") for m in msgs):
                with_reasoning += 1
            if any(m.get("tool_calls") for m in msgs):
                with_tools += 1
            if msgs and msgs[0]["role"] == "system":
                with_system += 1
            answer_chars.extend(len(m.get("content", "")) for m in msgs if m["role"] == "assistant")

    def share(n: int) -> float:
        traces = kinds.get("trace", 0)
        return round(n / traces, 3) if traces else 0.0

    return {
        "records": total,
        "kinds": dict(kinds),
        "duplicates": duplicates,
        "chars": _percentiles(chars),
        "assistant_turns": _percentiles(turns),
        "answer_chars_mean": round(statistics.fmean(answer_chars)) if answer_chars else 0,
        "share_with_reasoning": share(with_reasoning),
        "share_with_tool_calls": share(with_tools),
        "share_with_system_prompt": share(with_system),
        "synthetic": synthetic,
        "sources": dict(sources.most_common(10)),
        "licenses": dict(licenses.most_common(10)),
        "languages": dict(languages.most_common(10)),
    }


def file_stats(path: str | Path, limit: int | None = None) -> dict[str, Any]:
    def gen():
        for i, record in enumerate(iter_records(path)):
            if limit is not None and i >= limit:
                break
            yield record

    stats = compute_stats(gen())
    stats["path"] = str(path)
    return stats


def token_stats(lengths: list[int], max_len: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = _percentiles(lengths)
    out["mean"] = round(statistics.fmean(lengths)) if lengths else 0
    out["total"] = int(sum(lengths))
    if max_len:
        over = sum(1 for n in lengths if n > max_len)
        out["over_max_len"] = over
        out["share_over_max_len"] = round(over / len(lengths), 4) if lengths else 0.0
    return out
