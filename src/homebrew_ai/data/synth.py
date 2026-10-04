"""Synthetic training data, generated with the agent's own LLM (opt-in).

Generation is *dynamic*: instead of fixed templates, the agent passes a brief
distilled from what the user asked for, and the generator

1. plans a diverse list of concrete scenarios for that brief, then
2. writes one ETF trace per scenario in small batches, following the spec
   (turns, reasoning, tool use, tone, length, language, system prompt), or
3. rewrites existing records according to an instruction (``transform``),
   e.g. "answer like a 1920s radio host, keep every fact".

Everything is validated as ETF; anything malformed is dropped.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from homebrew_ai.backends.base import Backend, BackendError, parse_json_loose
from homebrew_ai.etf.schema import normalize_record, record_fingerprint

_MSG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["user", "assistant", "tool"]},
        "content": {"type": "string"},
        "reasoning": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "arguments_json"],
                "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}},
            },
        },
    },
}
EXAMPLES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["examples"],
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["messages"],
                "properties": {"messages": {"type": "array", "items": _MSG_SCHEMA}},
            },
        }
    },
}
SCENARIOS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenarios"],
    "properties": {"scenarios": {"type": "array", "items": {"type": "string"}}},
}

WRITER_SYSTEM = """You write training data for fine-tuning a language model. The data teaches the *target model* how to behave.
Write realistic, varied, high-quality conversations. The user turns should sound like real people (different
phrasings, levels of detail, occasional typos when natural). The assistant turns show exactly the behaviour the target
model should learn. Never mention that the data is synthetic or that you are generating training data.
Return JSON only, matching the requested schema."""


@dataclass
class SynthSpec:
    brief: str
    count: int = 20
    turns: int = 1
    reasoning: bool = False
    system_prompt: str | None = None
    language: str | None = None
    style: str | None = None
    answer_length: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    seed_examples: list[dict[str, Any]] = field(default_factory=list)
    avoid: str | None = None


def _ask_json(backend: Backend, system: str, prompt: str, schema: dict[str, Any], max_tokens: int, on_progress: Callable[[str, int], None] | None = None, effort: str | None = None) -> Any:
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "json_schema": schema}
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    if effort:
        kwargs["effort"] = effort
    text = backend.complete(system, prompt, **kwargs)
    try:
        return parse_json_loose(text)
    except ValueError as exc:
        raise BackendError(f"the model did not return valid JSON: {text[:200]}") from exc


def _workers(requested: int | None = None) -> int:
    if requested:
        return max(1, min(int(requested), 16))
    try:
        return max(1, min(int(os.environ.get("HOMEBREW_AI_SYNTH_WORKERS", "4")), 16))
    except ValueError:
        return 4


def _k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 100_000:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


class BatchProgress:
    """One short status line for requests running in parallel, e.g.
    ``312/1180 rewritten · 8 running · 298k written · 5:40 · ~16 min left``."""

    def __init__(self, say: Callable[[str], None], target: int, unit: str):
        self.say, self.target, self.unit = say, target, unit
        self.items = 0
        self.live: dict[int, dict[str, int]] = {}
        self.chars = {"thinking": 0, "writing": 0}  # finished batches
        self.lock = threading.Lock()
        self.last = 0.0
        self.started = time.time()

    def callback(self, batch: int) -> Callable[[str, int], None]:
        def update(kind: str, chars: int) -> None:
            with self.lock:
                if batch in self.live:
                    self.live[batch][kind] = chars
                    self._emit()
        return update

    def begin(self, batch: int) -> None:
        with self.lock:
            self.live[batch] = {}
            self._emit(force=True)

    def finish(self, batch: int, added: int) -> None:
        with self.lock:
            for kind, chars in self.live.pop(batch, {}).items():
                self.chars[kind] = self.chars.get(kind, 0) + chars
            self.items += added
            self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < 0.25:
            return
        self.last = now
        writing = self.chars["writing"] + sum(v.get("writing", 0) for v in self.live.values())
        thinking = self.chars["thinking"] + sum(v.get("thinking", 0) for v in self.live.values())
        parts = [f"{min(self.items, self.target)}/{self.target} {self.unit}"]
        if self.live:
            parts.append(f"{len(self.live)} running")
        if writing:
            parts.append(f"{_k(writing)} written")
        elif thinking:
            parts.append(f"thinking ({_k(thinking)})")
        elif self.live:
            parts.append("waiting for the model")
        elapsed = now - self.started
        parts.append(_clock(elapsed))
        if 0 < self.items < self.target:
            left = elapsed / self.items * (self.target - self.items)
            parts.append(f"~{max(1, round(left / 60))} min left")
        self.say(" · ".join(parts))


def _pipeline(next_task: Callable[[int], Callable[[], Any] | None], on_done: Callable[[int, Any], None], workers: int) -> bool:
    """Run tasks with up to ``workers`` in flight.

    ``next_task(n)`` returns the n-th task, or None when there is no more work right now; ``on_done(n, result)``
    runs in the calling thread as soon as a task finishes (``result`` is the exception if it failed), so finished
    work is counted and saved immediately. Returns False when interrupted with Ctrl+C; requests already sent
    finish in the background and are discarded.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="synth")
    running: dict[Future, int] = {}
    n = 0
    try:
        while True:
            while len(running) < max(1, workers):
                task = next_task(n)
                if task is None:
                    break
                running[pool.submit(task)] = n
                n += 1
            if not running:
                return True
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                i = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - one failed batch must not stop the others
                    result = exc
                on_done(i, result)
    except KeyboardInterrupt:
        return False
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def plan_scenarios(backend: Backend, spec: SynthSpec, n: int) -> list[str]:
    prompt = (
        f"Brief for the target model:\n{spec.brief}\n\n"
        f"List {n} distinct, concrete scenarios (one sentence each) for user requests the target model should handle. "
        "Cover different topics, difficulty levels, user types and tones; include a few edge cases where the right "
        "behaviour is to clarify or politely decline. Return {\"scenarios\": [...]}."
    )
    if spec.language:
        prompt += f"\nThe conversations will be in {spec.language}."
    if spec.avoid:
        prompt += f"\nAvoid: {spec.avoid}"
    data = _ask_json(backend, WRITER_SYSTEM, prompt, SCENARIOS_SCHEMA, max_tokens=4000)
    scenarios = [s.strip() for s in (data.get("scenarios") if isinstance(data, dict) else data) or [] if isinstance(s, str) and s.strip()]
    seen, unique = set(), []
    for s in scenarios:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    return unique[:n]


def _spec_text(spec: SynthSpec) -> str:
    lines = [f"Target behaviour: {spec.brief}"]
    lines.append(f"Each conversation has {spec.turns} user turn(s), each followed by an assistant answer.")
    if spec.reasoning:
        lines.append("Every assistant message includes a 'reasoning' field with the step-by-step thinking that leads to the answer (the answer itself stays in 'content').")
    if spec.tools:
        lines.append("The assistant may call these tools (put calls in 'tool_calls' with 'arguments_json' as a JSON object string, then add a 'tool' message with a plausible result before the final answer):")
        lines.extend("  " + json.dumps(t, ensure_ascii=False) for t in spec.tools)
    if spec.style:
        lines.append(f"Assistant style: {spec.style}")
    if spec.answer_length:
        lines.append(f"Answer length: {spec.answer_length}")
    if spec.language:
        lines.append(f"Language: {spec.language}")
    if spec.system_prompt:
        lines.append(f"The target model will run with this system prompt (do not include it in the messages): {spec.system_prompt}")
    if spec.avoid:
        lines.append(f"Avoid: {spec.avoid}")
    for ex in spec.seed_examples[:3]:
        lines.append("Example of the desired format and quality:\n" + json.dumps(ex, ensure_ascii=False)[:2000])
    return "\n".join(lines)


def _to_records(data: Any, spec: SynthSpec, source: str) -> list[dict[str, Any]]:
    examples = data.get("examples") if isinstance(data, dict) else data
    out = []
    for ex in examples or []:
        if not isinstance(ex, dict):
            continue
        messages = []
        for m in ex.get("messages") or []:
            if not isinstance(m, dict):
                continue
            msg = {"role": m.get("role"), "content": m.get("content", "")}
            if m.get("reasoning"):
                msg["reasoning"] = m["reasoning"]
            calls = []
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c.get("arguments_json") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls.append({"name": c.get("name"), "arguments": args if isinstance(args, dict) else {}})
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
        raw: dict[str, Any] = {"messages": messages, "meta": {"source": source, "synthetic": True}}
        if spec.system_prompt:
            raw["system"] = spec.system_prompt
        if spec.tools:
            raw["tools"] = spec.tools
        record, _ = normalize_record(raw)
        if record is not None:
            out.append(record)
    return out


def generate(
    backend: Backend,
    spec: SynthSpec,
    *,
    on_progress: Callable[[str], None] | None = None,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
    batch_size: int = 4,
    effort: str | None = None,
    workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create ``spec.count`` new traces. Returns ``(records, report)``.

    ``on_batch`` receives each batch's new records as soon as it is done (to save progress)."""
    say = on_progress or (lambda _m: None)
    source = f"synthetic:{backend.label}"
    n_scen = max(3, min(spec.count, 200))
    say(f"planning {n_scen} scenarios")
    scenarios = plan_scenarios(backend, spec, n_scen)
    if not scenarios:
        raise BackendError("could not plan any scenarios for this brief")

    def prompt_for(chunk: list[str]) -> str:
        prompt = _spec_text(spec) + "\n\nWrite one conversation for each scenario:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
        return prompt + '\n\nReturn {"examples": [{"messages": [...]}, ...]} with exactly one example per scenario.'

    return _scenario_batches(
        backend, spec, scenarios, "examples", prompt_for, EXAMPLES_SCHEMA, lambda data: _to_records(data, spec, source),
        say, on_batch, batch_size, source, effort, workers,
    )


def _scenario_batches(
    backend: Backend,
    spec: SynthSpec,
    scenarios: list[str],
    unit: str,
    prompt_for: Callable[[list[str]], str],
    schema: dict[str, Any],
    to_records: Callable[[Any], list[dict[str, Any]]],
    say: Callable[[str], None],
    on_batch: Callable[[list[dict[str, Any]]], None] | None,
    batch_size: int,
    source: str,
    effort: str | None = None,
    workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shared loop of generate() and generate_preferences(): batches of scenarios, deduplicated, in parallel."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    state = {"failures": 0, "budget": math.ceil(spec.count / batch_size) + 3, "idx": 0, "pending": 0}
    sizes: dict[int, int] = {}
    progress = BatchProgress(say, spec.count, unit)

    def next_task(n: int) -> Callable[[], Any] | None:
        need = spec.count - len(records) - state["pending"]
        if need <= 0 or state["budget"] <= 0 or state["failures"] > 3:
            return None
        size = min(batch_size, need)
        chunk = [scenarios[(state["idx"] + i) % len(scenarios)] for i in range(size)]
        state["idx"] += size
        state["budget"] -= 1
        state["pending"] += size
        sizes[n] = size

        def task() -> Any:
            progress.begin(n)
            return _ask_json(backend, WRITER_SYSTEM, prompt_for(chunk), schema, max_tokens=12000, on_progress=progress.callback(n), effort=effort)
        return task

    def on_done(n: int, data: Any) -> None:
        state["pending"] -= sizes.pop(n, 0)
        fresh: list[dict[str, Any]] = []
        if isinstance(data, BaseException):
            state["failures"] += 1
        else:
            for rec in to_records(data):
                fp = record_fingerprint(rec)
                if fp not in seen and len(records) < spec.count:
                    seen.add(fp)
                    records.append(rec)
                    fresh.append(rec)
        progress.finish(n, len(fresh))
        if fresh and on_batch:
            on_batch(fresh)

    finished = _pipeline(next_task, on_done, _workers(workers))
    report: dict[str, Any] = {"scenarios": len(scenarios), "generated": len(records), "failed_batches": state["failures"], "source": source}
    if not finished:
        report["stopped_early"] = "interrupted by the user (Ctrl+C); everything finished before that was kept"
    return records, report


def transform(
    backend: Backend,
    records: list[dict[str, Any]],
    instruction: str,
    *,
    rewrite: tuple[str, ...] = ("assistant",),
    on_progress: Callable[[str], None] | None = None,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
    batch_size: int = 4,
    effort: str | None = None,
    workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rewrite existing traces according to ``instruction`` (only the roles in ``rewrite``).

    ``on_batch`` receives each batch's rewritten records as soon as it is done (to save progress)."""
    say = on_progress or (lambda _m: None)
    batches = [b for b in ([r for r in records[i : i + batch_size] if "messages" in r] for i in range(0, len(records), batch_size)) if b]
    progress = BatchProgress(say, sum(len(b) for b in batches), "rewritten")
    done: dict[int, list[dict[str, Any]]] = {}
    state = {"failures": 0}

    def next_task(n: int) -> Callable[[], Any] | None:
        if n >= len(batches):
            return None
        batch = batches[n]

        def task() -> Any:
            progress.begin(n)
            payload = [{"messages": [{k: v for k, v in m.items() if k in ("role", "content", "reasoning")} for m in r["messages"] if m["role"] != "system"]} for r in batch]
            prompt = (
                f"Rewrite these conversations. Instruction: {instruction}\n"
                f"Only change messages with these roles: {', '.join(rewrite)}; copy all other messages exactly. "
                "Keep the same number of conversations, in the same order, with the same number of messages each.\n\n"
                + json.dumps({"examples": payload}, ensure_ascii=False)
                + '\n\nReturn {"examples": [...]} in the same structure.'
            )
            return _ask_json(backend, WRITER_SYSTEM, prompt, EXAMPLES_SCHEMA, max_tokens=16000, on_progress=progress.callback(n), effort=effort)
        return task

    def on_done(n: int, data: Any) -> None:
        out: list[dict[str, Any]] = []
        if isinstance(data, BaseException):
            state["failures"] += 1
        else:
            examples = data.get("examples") if isinstance(data, dict) else None
            for original, new in zip(batches[n], examples or []):
                new_msgs = new.get("messages") if isinstance(new, dict) else None
                old_msgs = [m for m in original["messages"] if m["role"] != "system"]
                if not new_msgs or len(new_msgs) != len(old_msgs):
                    state["failures"] += 1
                    continue
                merged = [m for m in original["messages"] if m["role"] == "system"]
                for old, nm in zip(old_msgs, new_msgs):
                    m = dict(old)
                    if old["role"] in rewrite and isinstance(nm, dict):
                        m["content"] = nm.get("content", old.get("content", ""))
                        if old.get("reasoning") and nm.get("reasoning"):
                            m["reasoning"] = nm["reasoning"]
                    merged.append(m)
                meta = {**(original.get("meta") or {}), "synthetic": True, "transformed_by": backend.label, "transform": instruction[:200]}
                record, _ = normalize_record({**original, "messages": merged, "meta": meta})
                if record is not None:
                    out.append(record)
        done[n] = out
        progress.finish(n, len(out))
        if out and on_batch:
            on_batch(out)

    finished = _pipeline(next_task, on_done, _workers(workers))
    rewritten = [r for n in sorted(done) for r in done[n]]  # source order
    report: dict[str, Any] = {"rewritten": len(rewritten), "failed": state["failures"]}
    if not finished:
        report["stopped_early"] = "interrupted by the user (Ctrl+C); everything finished before that was kept"
    return rewritten, report


PAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pairs"],
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prompt", "chosen", "rejected"],
                "properties": {"prompt": {"type": "string"}, "chosen": {"type": "string"}, "rejected": {"type": "string"}, "why": {"type": "string"}},
            },
        }
    },
}


def generate_preferences(
    backend: Backend,
    spec: SynthSpec,
    *,
    on_progress: Callable[[str], None] | None = None,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
    batch_size: int = 4,
    effort: str | None = None,
    workers: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preference pairs for DPO: a prompt, an answer showing the wanted behaviour, and a plausible worse one.

    ``spec.brief`` describes what "better" means (e.g. "concise, cites the manual, never invents part numbers");
    ``spec.avoid`` can describe the failure modes the rejected answers should show.
    """
    say = on_progress or (lambda _m: None)
    source = f"synthetic:{backend.label}"
    say("planning scenarios")
    scenarios = plan_scenarios(backend, spec, max(3, min(spec.count, 200)))
    if not scenarios:
        raise BackendError("could not plan any scenarios for this brief")

    def prompt_for(chunk: list[str]) -> str:
        return (
            f"What a better answer looks like: {spec.brief}\n"
            + (f"Typical flaws of worse answers: {spec.avoid}\n" if spec.avoid else "Worse answers should be plausible but clearly worse on the criteria above (not absurd).\n")
            + (f"Language: {spec.language}\n" if spec.language else "")
            + "For each scenario write a realistic user prompt, a 'chosen' answer that fully meets the criteria and a 'rejected' answer that falls short.\n"
            + "\n".join(f"{i + 1}. {sc}" for i, sc in enumerate(chunk))
            + '\nReturn {"pairs": [{"prompt", "chosen", "rejected", "why"}...]}.'
        )

    def to_records(data: Any) -> list[dict[str, Any]]:
        out = []
        for pair in (data.get("pairs") if isinstance(data, dict) else []) or []:
            raw: dict[str, Any] = {
                "messages": [{"role": "user", "content": pair.get("prompt", "")}],
                "chosen": pair.get("chosen", ""),
                "rejected": pair.get("rejected", ""),
                "meta": {"source": source, "synthetic": True, "why": pair.get("why", "")[:300]},
            }
            if spec.system_prompt:
                raw["system"] = spec.system_prompt
            rec, _ = normalize_record(raw)
            if rec is not None:
                out.append(rec)
        return out

    return _scenario_batches(backend, spec, scenarios, "pairs", prompt_for, PAIR_SCHEMA, to_records, say, on_batch, batch_size, source, effort, workers)


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "reason"],
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie", "both_bad"]}, "reason": {"type": "string"}},
}


def judge_pair(backend: Backend, prompt: str, a: str, b: str, rubric: str) -> dict[str, Any]:
    """Ask the guiding LLM which of two candidate answers better fits the rubric."""
    system = "You compare two answers from a model being fine-tuned and pick the one that better matches the rubric. Be strict and consistent. Return JSON only."
    text = f"Rubric (what the user wants): {rubric}\n\nPrompt:\n{prompt}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nReturn {{\"winner\": \"A\"|\"B\"|\"tie\"|\"both_bad\", \"reason\": short}}."
    data = _ask_json(backend, system, text, JUDGE_SCHEMA, max_tokens=1500)
    if not isinstance(data, dict) or data.get("winner") not in ("A", "B", "tie", "both_bad"):
        raise BackendError("the judge returned an unexpected answer")
    return data


def caption_image(backend: Backend, image_bytes: bytes, media_type: str, *, trigger_word: str | None, style: str) -> str:
    system = "You write captions for training a text-to-image model. Describe what is visible: subject, pose, setting, lighting, colours, style. One or two sentences, no preamble."
    prompt = f"Caption style: {style}."
    if trigger_word:
        prompt += f" Refer to the main subject as '{trigger_word}'."
    return backend.complete(system, prompt, images=[(image_bytes, media_type)], max_tokens=600).strip().strip('"')
