"""Empero Trace Format (ETF), version 1.

ETF is Homebrew's native training-data layout: plain ``.jsonl``, one JSON
object ("record") per line. Every field except the one that defines the record
kind is optional — leave out what you don't need. Four record kinds exist:

* **trace** — a conversation (``messages``), optionally with a system prompt,
  tool schemas, tool calls/results, reasoning, RAG documents, per-message loss
  control, chat-template options and preference alternatives::

      {"messages": [
          {"role": "system", "content": "You are Captain Byte."},
          {"role": "user", "content": "What is 7 x 8?"},
          {"role": "assistant", "reasoning": "7 x 8 = 56", "content": "Arr, 56!"}],
       "tools": [{"name": "calc", "description": "...", "parameters": {...}}],
       "meta": {"source": "...", "license": "..."}}

* **text** — a document for continued pretraining: ``{"text": "..."}``
* **completion** — raw prompt/completion without a chat template (loss on the
  completion only): ``{"prompt": "def add(a, b):", "completion": " return a + b"}``
* **image** — an image for text-to-image training:
  ``{"image": "images/0001.png", "caption": "a photo of sks corgi"}``

The full specification lives in ``docs/etf.md`` and as JSON Schema in
``etf.schema.json``. This module validates raw records and produces the
canonical form the rest of Homebrew relies on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

ETF_VERSION = 1
KINDS = ("trace", "text", "completion", "image")
ROLES = ("system", "user", "assistant", "tool")
TRAIN_ON = ("assistant", "last", "all")

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "ipython": "tool",
    "function_response": "tool",
    "tool_response": "tool",
}

REASONING_KEYS = ("reasoning", "reasoning_content", "thinking", "thought")
MEDIA_TYPES = ("image", "audio", "video", "file")

Level = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    level: Level
    code: str
    message: str
    index: int | None = None

    def __str__(self) -> str:
        where = f" (message {self.index})" if self.index is not None else ""
        return f"{self.level}: {self.code}{where}: {self.message}"


class ETFError(ValueError):
    """Raised when a record cannot be used at all."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues if i.level == "error"))


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #


def _normalize_part(part: Any) -> dict[str, Any] | None:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return None
    ptype = part.get("type", "text")
    if ptype in ("text", "input_text", "output_text"):
        text = part.get("text")
        return {"type": "text", "text": text} if isinstance(text, str) else None
    if ptype in ("image_url", "input_image"):
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else (url or part.get("url"))
        return {"type": "image", "image": url} if isinstance(url, str) and url else None
    if ptype == "input_audio":
        audio = part.get("input_audio") or {}
        data = audio.get("data") if isinstance(audio, dict) else None
        return {"type": "audio", "audio": data or part.get("audio"), **({"format": audio.get("format")} if isinstance(audio, dict) and audio.get("format") else {})}
    if ptype in MEDIA_TYPES:
        ref = part.get(ptype) or part.get("url") or part.get("path")
        if not isinstance(ref, str) or not ref:
            return None
        out = {"type": ptype, ptype: ref}
        for extra in ("mime", "format", "name"):
            if isinstance(part.get(extra), str):
                out[extra] = part[extra]
        return out
    return None


def normalize_content(value: Any, issues: list[Issue], index: int | None) -> str | list[dict[str, Any]]:
    """Strings stay strings; part lists are canonicalised and collapse to a string if text-only."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict) and "type" in value:
        value = [value]
    if isinstance(value, list):
        parts = []
        for raw in value:
            part = _normalize_part(raw)
            if part is None:
                issues.append(Issue("warning", "bad_content_part", f"ignored an unrecognised content part: {str(raw)[:80]}", index))
                continue
            parts.append(part)
        if all(p["type"] == "text" for p in parts):
            return "".join(p["text"] for p in parts) if len(parts) <= 1 else "\n".join(p["text"] for p in parts)
        return parts
    issues.append(Issue("error", "bad_content", f"content must be a string or a list of parts, got {type(value).__name__}", index))
    return ""


def content_text(content: Any) -> str:
    """Plain text of a (canonical) content value; media parts are skipped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def media_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict) and p.get("type") in MEDIA_TYPES]
    return []


def split_think_tags(content: str) -> tuple[str | None, str]:
    """Split ``<think>...</think>`` at the start of an answer into (reasoning, answer)."""
    stripped = content.lstrip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        head, _, tail = stripped[len("<think>"):].partition("</think>")
        return head.strip("\n").strip(), tail.lstrip("\n")
    if "</think>" in content and "<think>" not in content and content.count("</think>") == 1:
        head, _, tail = content.partition("</think>")
        if head.strip():
            return head.strip(), tail.lstrip("\n")
    return None, content


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


def _parse_arguments(raw: Any, issues: list[Issue], index: int | None) -> dict[str, Any] | str:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            issues.append(Issue("warning", "unparsed_arguments", "tool-call arguments are not valid JSON; kept as text", index))
            return raw
        if isinstance(parsed, dict):
            return parsed
        issues.append(Issue("warning", "unparsed_arguments", "tool-call arguments are not a JSON object; kept as text", index))
        return raw
    issues.append(Issue("error", "bad_arguments", f"tool-call arguments must be an object or JSON string, got {type(raw).__name__}", index))
    return {}


def _normalize_tool_calls(raw: Any, issues: list[Issue], index: int, counter: list[int]) -> list[dict[str, Any]]:
    if raw in (None, []):
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        issues.append(Issue("error", "bad_tool_calls", "tool_calls must be a list", index))
        return []
    calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            issues.append(Issue("error", "bad_tool_calls", "each tool call must be an object", index))
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(Issue("error", "bad_tool_calls", "tool call without a name", index))
            continue
        args = fn.get("arguments", fn.get("parameters", fn.get("args", fn.get("input"))))
        call_id = item.get("id") or item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            counter[0] += 1
            call_id = f"call_{counter[0]}"
        call = {"id": call_id, "name": name.strip(), "arguments": _parse_arguments(args, issues, index)}
        ctype = item.get("type")
        if isinstance(ctype, str) and ctype not in ("function", ""):
            call["type"] = ctype
        calls.append(call)
    return calls


def normalize_tools(raw: Any, issues: list[Issue] | None = None) -> list[dict[str, Any]]:
    """Canonical tool schemas: ``{"name", "description", "parameters"}`` plus optional
    ``returns`` (output schema), ``type`` (non-function tools), ``strict`` and ``examples``."""
    issues = issues if issues is not None else []
    if raw in (None, []):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            issues.append(Issue("error", "bad_tools", "tools is a string that is not valid JSON"))
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        issues.append(Issue("error", "bad_tools", "tools must be a list"))
        return []
    tools: list[dict[str, Any]] = []
    for spec in raw:
        if not isinstance(spec, dict):
            issues.append(Issue("error", "bad_tools", "each tool schema must be an object"))
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            issues.append(Issue("error", "bad_tools", "tool schema without a name"))
            continue
        params = fn.get("parameters", fn.get("input_schema")) or {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            issues.append(Issue("error", "bad_tools", f"parameters of tool {name!r} must be a JSON schema object"))
            continue
        tool: dict[str, Any] = {"name": name, "description": str(fn.get("description") or ""), "parameters": params}
        returns = fn.get("returns", fn.get("output_schema"))
        if isinstance(returns, dict):
            tool["returns"] = returns
        ttype = spec.get("type")
        if isinstance(ttype, str) and ttype not in ("function", ""):
            tool["type"] = ttype
        if isinstance(fn.get("strict"), bool):
            tool["strict"] = fn["strict"]
        if isinstance(fn.get("examples"), list):
            tool["examples"] = fn["examples"]
        tools.append(tool)
    return tools


def _normalize_documents(raw: Any, issues: list[Issue]) -> list[dict[str, Any]]:
    if raw in (None, []):
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        issues.append(Issue("error", "bad_documents", "documents must be a list"))
        return []
    docs = []
    for d in raw:
        if isinstance(d, str):
            docs.append({"text": d})
        elif isinstance(d, dict) and isinstance(d.get("text", d.get("content")), str):
            doc = {"text": d.get("text", d.get("content"))}
            for k in ("title", "source", "id"):
                if isinstance(d.get(k), (str, int)):
                    doc[k] = str(d[k])
            docs.append(doc)
        else:
            issues.append(Issue("warning", "bad_document", "ignored a document without text"))
    return docs


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #


def _loss_weight(raw: dict[str, Any]) -> float:
    if "weight" in raw and isinstance(raw["weight"], (int, float)) and not isinstance(raw["weight"], bool):
        return float(raw["weight"])
    train = raw.get("train", True)
    if train in (False, 0, "false", "False", "no"):
        return 0.0
    return 1.0


def _normalize_message(raw: Any, index: int, issues: list[Issue], counter: list[int]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        issues.append(Issue("error", "bad_message", "message must be an object", index))
        return None
    role_raw = raw.get("role", raw.get("from"))
    role = ROLE_ALIASES.get(str(role_raw).strip().lower()) if role_raw is not None else None
    if role is None:
        issues.append(Issue("error", "bad_role", f"unknown role {role_raw!r}", index))
        return None
    raw_content = raw.get("content", raw.get("value"))
    if role == "tool" and isinstance(raw_content, (dict, list)) and not (isinstance(raw_content, list) and raw_content and isinstance(raw_content[0], dict) and "type" in raw_content[0]):
        content: Any = raw_content  # structured tool output (JSON object/array) is kept as-is
    else:
        content = normalize_content(raw_content, issues, index)
    msg: dict[str, Any] = {"role": role, "content": content}
    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        msg["name"] = name.strip()

    if role == "assistant":
        reasoning = next((raw[k] for k in REASONING_KEYS if isinstance(raw.get(k), str) and raw[k].strip()), None)
        if isinstance(content, str):
            if reasoning is None:
                extracted, answer = split_think_tags(content)
                if extracted is not None:
                    reasoning, msg["content"] = extracted, answer
            elif "<think>" in content:
                _, msg["content"] = split_think_tags(content)
        if reasoning:
            msg["reasoning"] = reasoning.strip()
        calls = _normalize_tool_calls(raw.get("tool_calls", raw.get("function_call")), issues, index, counter)
        if calls:
            msg["tool_calls"] = calls
        weight = _loss_weight(raw)
        if weight <= 0:
            msg["train"] = False
        elif weight != 1.0:
            msg["weight"] = weight
        if not content_text(msg["content"]).strip() and not media_parts(msg["content"]) and not calls and not msg.get("reasoning"):
            issues.append(Issue("error", "empty_assistant", "assistant message has no content, reasoning or tool calls", index))
    elif role == "tool":
        call_id = raw.get("tool_call_id") or raw.get("call_id") or raw.get("id")
        if isinstance(call_id, str) and call_id:
            msg["tool_call_id"] = call_id
        if raw.get("is_error") is True or raw.get("status") in ("error", "failed"):
            msg["is_error"] = True
        if isinstance(content, str) and not content.strip():
            issues.append(Issue("warning", "empty_tool_result", "tool result is empty", index))
    else:
        if not content_text(content).strip() and not media_parts(content):
            issues.append(Issue("error", "empty_message", f"{role} message is empty", index))
    return msg


def _normalize_assistant_alt(raw: Any, issues: list[Issue], counter: list[int], label: str) -> list[dict[str, Any]] | None:
    """Preference alternatives: an assistant message, a string, or a list of messages."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [{"role": "assistant", "content": raw}]
    elif isinstance(raw, dict):
        raw = [{"role": "assistant", **raw}] if "role" not in raw else [raw]
    if not isinstance(raw, list) or not raw:
        issues.append(Issue("error", f"bad_{label}", f"{label} must be an assistant message, a string or a list of messages"))
        return None
    out = []
    for m in raw:
        norm = _normalize_message(m, -1, issues, counter)
        if norm is not None:
            out.append(norm)
    if not any(m["role"] == "assistant" for m in out):
        issues.append(Issue("error", f"bad_{label}", f"{label} contains no assistant message"))
        return None
    return out


def _normalize_trace(raw: dict[str, Any], issues: list[Issue]) -> dict[str, Any] | None:
    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        issues.append(Issue("error", "no_messages", "messages must be a non-empty list"))
        return None
    counter = [0]
    messages: list[dict[str, Any]] = []
    system = raw.get("system")
    if system is not None:
        sys_content = normalize_content(system, issues, None)
        if content_text(sys_content).strip():
            messages.append({"role": "system", "content": sys_content})
    for i, m in enumerate(messages_raw):
        norm = _normalize_message(m, i, issues, counter)
        if norm is not None:
            messages.append(norm)
    if any(i.level == "error" for i in issues):
        return None

    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            issues.append(Issue("warning", "system_not_first", "system message in the middle of the conversation (merged into the next user turn for templates that cannot show it)", i))

    first_body = next((i for i, m in enumerate(messages) if m["role"] != "system"), None)
    if first_body is None:
        issues.append(Issue("error", "no_messages", "trace has only system messages"))
        return None
    if messages[first_body]["role"] != "user":
        issues.append(Issue("warning", "starts_without_user", "conversation does not start with a user message", first_body))

    # tool results must answer an earlier assistant tool call
    pending: list[str] = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if pending:
                issues.append(Issue("warning", "unanswered_tool_calls", f"{len(pending)} tool call(s) got no tool result", i))
            pending = [c["id"] for c in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            prev = next((p for p in reversed(messages[:i]) if p["role"] not in ("tool", "system")), None)
            if prev is None or prev["role"] != "assistant" or not prev.get("tool_calls"):
                issues.append(Issue("error", "orphan_tool_result", "tool result without a preceding assistant tool call", i))
                return None
            if "tool_call_id" not in m:
                if pending:
                    m["tool_call_id"] = pending[0]
                else:
                    issues.append(Issue("warning", "extra_tool_result", "more tool results than tool calls", i))
            if m.get("tool_call_id") in pending:
                pending.remove(m["tool_call_id"])
            if "name" not in m:
                call = next((c for c in prev["tool_calls"] if c["id"] == m.get("tool_call_id")), None)
                if call:
                    m["name"] = call["name"]
        elif m["role"] == "user":
            pending = []

    for i in range(first_body + 1, len(messages)):
        a, b = messages[i - 1]["role"], messages[i]["role"]
        if a == b and a in ("user", "assistant"):
            issues.append(Issue("warning", "repeated_role", f"two {a} messages in a row (merged for strict chat templates)", i))

    record: dict[str, Any] = {"messages": messages}
    chosen = _normalize_assistant_alt(raw.get("chosen"), issues, counter, "chosen")
    rejected = _normalize_assistant_alt(raw.get("rejected"), issues, counter, "rejected")
    if any(i.level == "error" for i in issues):
        return None
    if chosen:
        record["chosen"] = chosen
    if rejected:
        record["rejected"] = rejected
    if isinstance(raw.get("label"), bool):
        record["label"] = raw["label"]

    if not chosen:
        while messages and messages[-1]["role"] not in ("assistant",):
            if messages[-1]["role"] == "system" and len(messages) == 1:
                break
            messages.pop()
            issues.append(Issue("warning", "trailing_messages", "dropped trailing non-assistant message (nothing to learn from it)"))

    train_on = raw.get("train_on", "assistant")
    if train_on not in TRAIN_ON:
        issues.append(Issue("error", "bad_train_on", f"train_on must be one of {TRAIN_ON}"))
        return None
    if train_on != "assistant":
        record["train_on"] = train_on

    trainable = [m for m in messages if m["role"] == "assistant" and m.get("train", True)]
    if not trainable and not chosen and train_on != "all":
        issues.append(Issue("error", "nothing_to_learn", "no assistant message to train on"))
        return None

    tools = normalize_tools(raw.get("tools"), issues)
    documents = _normalize_documents(raw.get("documents"), issues)
    if any(i.level == "error" for i in issues):
        return None
    if tools:
        record["tools"] = tools
    if raw.get("tool_choice") is not None:
        record["tool_choice"] = raw["tool_choice"]
    if documents:
        record["documents"] = documents
    kwargs = raw.get("template_kwargs", raw.get("chat_template_kwargs"))
    if isinstance(kwargs, dict) and kwargs:
        record["template_kwargs"] = kwargs
    return record


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


def detect_kind(raw: dict[str, Any]) -> str | None:
    found = []
    if "messages" in raw:
        found.append("trace")
    if "text" in raw:
        found.append("text")
    if "completion" in raw:
        found.append("completion")
    if "image" in raw:
        found.append("image")
    if len(found) != 1:
        return None if not found else "ambiguous"
    return found[0]


def normalize_record(raw: Any) -> tuple[dict[str, Any] | None, list[Issue]]:
    """Validate one raw record.

    Returns ``(record, issues)``. ``record`` is ``None`` when the record has at
    least one error and cannot be trained on; warnings never block a record.
    """
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [Issue("error", "not_an_object", "record must be a JSON object")]
    kind = detect_kind(raw)
    if kind == "ambiguous":
        return None, [Issue("error", "ambiguous", "record must have exactly one of 'messages', 'text', 'completion' or 'image'")]
    if kind is None:
        return None, [Issue("error", "unknown_record", "record needs 'messages' (trace), 'text', 'prompt'+'completion' or 'image'")]

    record: dict[str, Any] | None
    if kind == "trace":
        record = _normalize_trace(raw, issues)
    elif kind == "text":
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return None, [Issue("error", "empty_text", "text record is empty")]
        record = {"text": text}
    elif kind == "completion":
        prompt, completion = raw.get("prompt", ""), raw.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            return None, [Issue("error", "bad_completion", "prompt and completion must be strings")]
        if not completion.strip():
            return None, [Issue("error", "empty_completion", "completion is empty")]
        record = {"prompt": prompt, "completion": completion}
    else:
        image = raw.get("image")
        if not isinstance(image, str) or not image.strip():
            return None, [Issue("error", "bad_image", "image must be a non-empty path or URL")]
        caption = raw.get("caption", "")
        caption = "" if caption is None else caption
        if not isinstance(caption, str):
            return None, [Issue("error", "bad_caption", "caption must be a string")]
        record = {"image": image.strip(), "caption": caption.strip()}
        variants = raw.get("captions")
        if isinstance(variants, list):
            clean = [c.strip() for c in variants if isinstance(c, str) and c.strip()]
            if clean:
                record["captions"] = clean
                if not record["caption"]:
                    record["caption"] = clean[0]
        refs = raw.get("references")
        if isinstance(refs, list):
            refs = [r for r in refs if isinstance(r, str) and r.strip()]
            if refs:
                record["references"] = refs
        if not record["caption"]:
            issues.append(Issue("warning", "empty_caption", "image has no caption"))

    if record is None:
        return None, issues
    if isinstance(raw.get("id"), (str, int)):
        record["id"] = str(raw["id"])
    repeat = raw.get("repeat")
    if isinstance(repeat, int) and not isinstance(repeat, bool) and repeat > 1:
        record["repeat"] = min(repeat, 100)
    if isinstance(raw.get("meta"), dict) and raw["meta"]:
        record["meta"] = raw["meta"]
    return record, issues


def ensure_record(raw: Any) -> dict[str, Any]:
    record, issues = normalize_record(raw)
    if record is None:
        raise ETFError(issues)
    return record


def record_kind(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "trace"
    if "image" in record:
        return "image"
    if "completion" in record:
        return "completion"
    return "text"


def is_trace(record: dict[str, Any]) -> bool:
    return "messages" in record


def is_image(record: dict[str, Any]) -> bool:
    return "image" in record


def record_fingerprint(record: dict[str, Any]) -> str:
    """Stable hash of the trainable content (ignores id/meta) for deduplication."""
    kind = record_kind(record)
    if kind == "trace":
        payload: Any = [
            [m["role"], content_text(m.get("content")).strip(), m.get("reasoning", ""), m.get("tool_calls", []), media_parts(m.get("content"))]
            for m in record["messages"]
        ] + [record.get("chosen"), record.get("rejected")]
    elif kind == "image":
        payload = [record["image"], record.get("caption", ""), record.get("captions"), record.get("references")]
    elif kind == "completion":
        payload = [record["prompt"], record["completion"]]
    else:
        payload = record["text"].strip()
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()
