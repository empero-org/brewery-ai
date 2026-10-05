"""Turn rows of arbitrary datasets into ETF records.

Most public datasets follow one of a handful of layouts (OpenAI-style
``messages``, ShareGPT ``conversations``, Alpaca ``instruction/input/output``,
prompt/completion pairs, preference pairs, plain text). :func:`detect_mapping`
guesses the layout from a few sample rows; :func:`convert_row` applies a
:class:`Mapping`. The agent can always override the guess with an explicit
mapping, including Python format templates over any columns.
"""

from __future__ import annotations

import json
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any

LAYOUTS = ("messages", "sharegpt", "alpaca", "prompt_completion", "preference", "text")

CHAT_COLUMNS = ("messages", "conversations", "conversation", "chat", "dialog", "dialogue", "turns", "conversation_a")
PAIR_CANDIDATES = (
    ("instruction", "output"),
    ("prompt", "completion"),
    ("prompt", "response"),
    ("prompt", "answer"),
    ("prompt", "output"),
    ("question", "answer"),
    ("question", "response"),
    ("query", "response"),
    ("query", "answer"),
    ("input", "output"),
    ("input", "target"),
    ("problem", "solution"),
    ("instruction", "response"),
    ("user", "assistant"),
    ("source", "target"),
    ("context", "response"),
)
INPUT_COLUMNS = ("input", "context")
REASONING_COLUMNS = ("reasoning", "reasoning_content", "thinking", "chain_of_thought", "cot", "rationale", "thought")
SYSTEM_COLUMNS = ("system", "system_prompt", "system_message")
TEXT_COLUMNS = ("text", "content", "document", "story", "body", "markdown")
TOOLS_COLUMNS = ("tools", "functions")


@dataclass
class Mapping:
    """How to read one dataset row. Column names refer to the source dataset."""

    layout: str
    messages: str | None = None
    role_key: str = "role"
    content_key: str = "content"
    role_map: dict[str, str] = field(default_factory=dict)
    prompt: str | None = None
    completion: str | None = None
    input: str | None = None
    reasoning: str | None = None
    system: str | None = None
    system_text: str | None = None
    prompt_template: str | None = None
    completion_template: str | None = None
    text: str | None = None
    chosen: str | None = None
    rejected: str | None = None
    tools: str | None = None
    keep_meta: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for f in fields(self):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                defaults[f.name] = f.default_factory()  # type: ignore[misc]
        return {k: v for k, v in asdict(self).items() if k == "layout" or v != defaults.get(k)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mapping":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown mapping keys: {', '.join(sorted(unknown))}")
        if data.get("layout") not in LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
        return cls(**data)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"template refers to unknown column {key!r}")


def _render_template(template: str, row: dict[str, Any]) -> str:
    values = {k: ("" if v is None else v) for k, v in row.items()}
    return template.format_map(_SafeDict(values))


def _as_list(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    return json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def _chat_item_keys(value: Any) -> tuple[str, str] | None:
    value = _as_list(value)
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return None
    first = value[0]
    if "role" in first and "content" in first:
        return "role", "content"
    if "from" in first and "value" in first:
        return "from", "value"
    if "speaker" in first and "text" in first:
        return "speaker", "text"
    if "author" in first and "text" in first:
        return "author", "text"
    return None


def detect_mapping(rows: list[dict[str, Any]]) -> tuple[Mapping | None, str]:
    """Guess a mapping from sample rows. Returns ``(mapping or None, explanation)``."""
    if not rows:
        return None, "no rows to inspect"
    row = rows[0]
    cols = {c.lower(): c for c in row}

    def col(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    system = col(*SYSTEM_COLUMNS)
    tools = col(*TOOLS_COLUMNS)

    # 1. preference pairs (checked first: chosen/rejected often hold whole conversations)
    chosen, rejected = col("chosen"), col("rejected")
    if chosen and rejected:
        prompt = col("prompt", "question", "instruction", "input")
        convo = next((cols[n] for n in CHAT_COLUMNS if n in cols and _chat_item_keys(row[cols[n]])), None)
        return (
            Mapping(layout="preference", chosen=chosen, rejected=rejected, prompt=prompt, messages=convo, system=system),
            "preference pairs (chosen/rejected): usable for DPO, or for SFT on the chosen answers",
        )

    # 2. chat-style list columns
    for name in CHAT_COLUMNS:
        if name in cols:
            keys = _chat_item_keys(row[cols[name]])
            if keys:
                layout = "messages" if keys == ("role", "content") else "sharegpt"
                m = Mapping(layout=layout, messages=cols[name], role_key=keys[0], content_key=keys[1], system=system, tools=tools)
                return m, f"column {cols[name]!r} holds a conversation list ({keys[0]}/{keys[1]})"
    for original, value in row.items():
        keys = _chat_item_keys(value)
        if keys:
            layout = "messages" if keys == ("role", "content") else "sharegpt"
            return (
                Mapping(layout=layout, messages=original, role_key=keys[0], content_key=keys[1], system=system, tools=tools),
                f"column {original!r} looks like a conversation list",
            )

    # 3. instruction / prompt-completion pairs
    reasoning = col(*REASONING_COLUMNS)
    for p, c in PAIR_CANDIDATES:
        if p in cols and c in cols:
            extra = None
            if p in ("instruction", "question", "prompt", "query"):
                extra = next((cols[i] for i in INPUT_COLUMNS if i in cols and cols[i] != cols[p]), None)
            layout = "alpaca" if (p, c) == ("instruction", "output") else "prompt_completion"
            m = Mapping(layout=layout, prompt=cols[p], completion=cols[c], input=extra, reasoning=reasoning, system=system)
            why = f"prompt column {cols[p]!r} → answer column {cols[c]!r}"
            if extra:
                why += f" (+ extra input {extra!r})"
            if reasoning:
                why += f", reasoning from {reasoning!r}"
            return m, why

    # 4. plain text
    text = col(*TEXT_COLUMNS)
    if text and isinstance(row[text], str):
        return Mapping(layout="text", text=text), f"plain text in column {text!r} (continued pretraining)"

    return None, f"could not recognise the layout of columns: {', '.join(row)}"


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

_HH_SPLIT = re.compile(r"\n\n(Human|Assistant|H|A):\s?")


def _parse_hh_string(text: str) -> list[dict[str, str]]:
    """Parse Anthropic-HH style ``\\n\\nHuman: ... \\n\\nAssistant: ...`` strings."""
    parts = _HH_SPLIT.split("\n\n" + text.lstrip())
    messages = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] in ("Human", "H") else "assistant"
        messages.append({"role": role, "content": parts[i + 1].strip()})
    return messages


def _convert_chat(value: Any, mapping: Mapping) -> list[dict[str, Any]]:
    value = _as_list(value)
    if not isinstance(value, list):
        raise ValueError(f"column {mapping.messages!r} is not a list")
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("conversation items must be objects")
        role_raw = str(item.get(mapping.role_key, "")).strip()
        role = mapping.role_map.get(role_raw, role_raw)
        content = item.get(mapping.content_key)
        if role in ("function_call", "tool_call"):
            call = _as_list(content)
            calls = call if isinstance(call, list) else [call]
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            continue
        msg = {k: v for k, v in item.items() if k not in (mapping.role_key, mapping.content_key)}
        msg["role"] = role
        msg["content"] = content
        messages.append(msg)
    return messages


def convert_row(row: dict[str, Any], mapping: Mapping, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert one row into a raw ETF record (validate it with ``normalize_record``)."""
    layout = mapping.layout
    record: dict[str, Any]

    if layout == "text":
        key = mapping.text or "text"
        text = _render_template(mapping.prompt_template, row) if mapping.prompt_template else _stringify(row.get(key))
        record = {"text": text}
    elif layout in ("messages", "sharegpt"):
        messages = _convert_chat(row.get(mapping.messages or "messages"), mapping)
        record = {"messages": messages}
    elif layout == "preference":
        prompt_text = _stringify(row.get(mapping.prompt)) if mapping.prompt else ""

        def to_msgs(raw: Any) -> list[dict[str, Any]]:
            raw = _as_list(raw)
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                keys = _chat_item_keys(raw) or ("role", "content")
                return _convert_chat(raw, Mapping(layout="messages", role_key=keys[0], content_key=keys[1]))
            if isinstance(raw, str) and ("\n\nHuman:" in "\n\n" + raw.lstrip() or "\n\nAssistant:" in raw):
                return _parse_hh_string(raw)
            return [{"role": "assistant", "content": _stringify(raw)}]

        def split(msgs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            roles = [str(m.get("role", "")).lower() for m in msgs]
            last = max((i for i, r in enumerate(roles) if r in ("assistant", "gpt", "model", "bot")), default=-1)
            return (msgs[:last], msgs[last:]) if last >= 0 else (msgs, [])

        prompt_c, chosen = split(to_msgs(row.get(mapping.chosen or "chosen")))
        _prompt_r, rejected = split(to_msgs(row.get(mapping.rejected or "rejected")))
        if mapping.messages and row.get(mapping.messages):
            keys = _chat_item_keys(row[mapping.messages]) or ("role", "content")
            prompt_c = _convert_chat(row[mapping.messages], Mapping(layout="messages", role_key=keys[0], content_key=keys[1]))
        prompt_msgs = prompt_c or ([{"role": "user", "content": prompt_text}] if prompt_text else [])
        if prompt_text and not any(str(m.get("role")).lower() in ("user", "human") for m in prompt_msgs):
            prompt_msgs = [{"role": "user", "content": prompt_text}, *prompt_msgs]
        record = {"messages": prompt_msgs, "chosen": chosen, "rejected": rejected}
    elif layout in ("alpaca", "prompt_completion"):
        if mapping.prompt_template:
            prompt = _render_template(mapping.prompt_template, row)
        else:
            prompt = _stringify(row.get(mapping.prompt or "prompt"))
            extra = _stringify(row.get(mapping.input)) if mapping.input else ""
            if extra.strip():
                prompt = f"{prompt.rstrip()}\n\n{extra.strip()}"
        if mapping.completion_template:
            answer = _render_template(mapping.completion_template, row)
        else:
            answer = _stringify(row.get(mapping.completion or "completion"))
        assistant: dict[str, Any] = {"role": "assistant", "content": answer}
        if mapping.reasoning and row.get(mapping.reasoning):
            assistant["reasoning"] = _stringify(row.get(mapping.reasoning))
        record = {"messages": [{"role": "user", "content": prompt}, assistant]}
    else:
        raise ValueError(f"unknown layout {layout!r}")

    if "messages" in record:
        system = mapping.system_text
        if mapping.system and row.get(mapping.system):
            system = _stringify(row[mapping.system])
        if system and not (record["messages"] and str(record["messages"][0].get("role")).lower() in ("system", "developer")):
            record["messages"] = [{"role": "system", "content": system}, *record["messages"]]
        if mapping.tools and row.get(mapping.tools):
            record["tools"] = _as_list(row[mapping.tools])

    extra_meta = {k: row[k] for k in mapping.keep_meta if k in row}
    if meta or extra_meta:
        record["meta"] = {**(meta or {}), **extra_meta}
    return record
