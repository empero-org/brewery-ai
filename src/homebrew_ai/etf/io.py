"""Reading, writing and validating ETF ``.jsonl`` files."""

from __future__ import annotations

import gzip
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from homebrew_ai.etf.schema import Issue, normalize_record


def _open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_raw(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, parsed_json_or_exception)`` for every non-blank line."""
    path = Path(path)
    with _open_text(path, "rt") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, exc


def iter_records(path: str | Path, *, skip_invalid: bool = True) -> Iterator[dict[str, Any]]:
    """Yield canonical records. Invalid lines are skipped (or raise if ``skip_invalid`` is False)."""
    for lineno, raw in iter_raw(path):
        if isinstance(raw, Exception):
            if skip_invalid:
                continue
            raise ValueError(f"{path}:{lineno}: invalid JSON: {raw}")
        record, issues = normalize_record(raw)
        if record is None:
            if skip_invalid:
                continue
            raise ValueError(f"{path}:{lineno}: " + "; ".join(str(i) for i in issues))
        yield record


def read_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in iter_records(path):
        out.append(record)
        if limit is not None and len(out) >= limit:
            break
    return out


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as ``.jsonl`` (gzip if the name ends in ``.gz``). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    if path.suffix == ".gz":
        raw = gzip.open(tmp, "wb")
        fh: Any = io.TextIOWrapper(raw, encoding="utf-8")
    else:
        fh = open(tmp, "w", encoding="utf-8")
    with fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1
    tmp.replace(path)
    return count


def append_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a plain ``.jsonl`` file (used to save progress batch by batch)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(lines))
    return len(lines)


def count_lines(path: str | Path) -> int:
    with _open_text(Path(path), "rt") as fh:
        return sum(1 for line in fh if line.strip())


@dataclass
class ValidationReport:
    path: str
    total: int = 0
    valid: int = 0
    invalid: int = 0
    traces: int = 0
    texts: int = 0
    images: int = 0
    completions: int = 0
    errors: Counter = field(default_factory=Counter)
    warnings: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.valid > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "traces": self.traces,
            "texts": self.texts,
            "images": self.images,
            "completions": self.completions,
            "errors": dict(self.errors.most_common()),
            "warnings": dict(self.warnings.most_common()),
            "examples": self.examples,
        }


def validate_file(path: str | Path, max_examples: int = 8) -> ValidationReport:
    report = ValidationReport(path=str(path))
    for lineno, raw in iter_raw(path):
        report.total += 1
        if isinstance(raw, Exception):
            report.invalid += 1
            report.errors["invalid_json"] += 1
            if len(report.examples) < max_examples:
                report.examples.append(f"line {lineno}: invalid JSON ({raw.msg})")
            continue
        record, issues = normalize_record(raw)
        for issue in issues:
            bucket = report.errors if issue.level == "error" else report.warnings
            bucket[issue.code] += 1
            if issue.level == "error" and len(report.examples) < max_examples:
                report.examples.append(f"line {lineno}: {issue}")
        if record is None:
            report.invalid += 1
        else:
            report.valid += 1
            if "messages" in record:
                report.traces += 1
            elif "image" in record:
                report.images += 1
            elif "completion" in record:
                report.completions += 1
            else:
                report.texts += 1
    return report


def summarize_issues(issues: list[Issue]) -> str:
    return "; ".join(str(i) for i in issues)
