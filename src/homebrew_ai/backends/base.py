"""Provider-neutral chat types for the agent's own LLM ("the brewmaster").

History is append-only: the agent loop only ever appends messages, and each
backend renders the same history to the same request bytes. Assistant turns
keep the provider's native payload in ``raw`` so it can be replayed verbatim
(Anthropic requires thinking blocks to come back unchanged).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    parts: list[str] = field(default_factory=list)  # user/tool text parts; assistant text
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    raw: Any = None
    raw_provider: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "parts": self.parts,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "is_error": self.is_error,
            "raw": self.raw,
            "raw_provider": self.raw_provider,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            parts=list(d.get("parts") or []),
            tool_calls=[ToolCall(c["id"], c["name"], c.get("arguments") or {}) for c in d.get("tool_calls") or []],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            is_error=bool(d.get("is_error")),
            raw=d.get("raw"),
            raw_provider=d.get("raw_provider"),
        )


@dataclass
class Turn:
    message: Message
    stop_reason: str  # end | tool_use | max_tokens | refusal
    usage: dict[str, int] = field(default_factory=dict)
    refusal: str | None = None


ProgressCallback = Callable[[str, int], None]
TextCallback = Callable[[str], None]


class BackendError(RuntimeError):
    """A problem talking to the LLM provider, phrased for humans."""


class Backend(ABC):
    provider: str = "base"

    def __init__(self, model: str, capability: str = "medium"):
        self.model = model
        self.capability = capability

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_text: TextCallback | None = None,
        on_progress: TextCallback | None = None,
    ) -> Turn: ...

    @abstractmethod
    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 8000,
        json_schema: dict[str, Any] | None = None,
        images: list[tuple[bytes, str]] | None = None,
        on_progress: ProgressCallback | None = None,
        effort: str | None = None,
    ) -> str:
        """One-shot completion without tools (used for synthetic data and captions).

        ``on_progress(kind, chars)`` is called while the answer streams in, with kind "thinking" or
        "writing" and the number of characters received so far for that kind. ``effort`` ("off", "low",
        "medium", "high") asks reasoning models to think less or more, where the provider supports it.
        """

    def supports_vision(self) -> bool:
        return False

    def check(self) -> str:
        """Cheap connectivity check; returns a short reply or raises BackendError."""
        return self.complete("Reply with exactly: ok", "ping", max_tokens=200)


# --------------------------------------------------------------------------- #
# tool-input validation (subset of JSON Schema used by Homebrew's tools)
# --------------------------------------------------------------------------- #

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: dict[str, Any], value: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_check_type(t, value) for t in expected):
            errors.append(f"{path or 'input'} should be one of {expected}")
            return errors
    elif expected and not _check_type(expected, value):
        errors.append(f"{path or 'input'} should be {expected}, got {type(value).__name__}")
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or 'input'} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"missing required field {path + '.' if path else ''}{req}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errors.extend(validate_arguments(props[k], v, f"{path}.{k}" if path else k))
            elif schema.get("additionalProperties") is False:
                errors.append(f"unknown field {path + '.' if path else ''}{k}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_arguments(schema["items"], item, f"{path}[{i}]"))
    return errors


def _check_type(name: str, value: Any) -> bool:
    if name == "null":
        return value is None
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(name)
    return True if py is None else isinstance(value, py)


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in prose or a ```json fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for chunk in text.split("```")[1::2]:
            chunk = chunk.split("\n", 1)[1] if chunk.lstrip().startswith(("json", "JSON")) else chunk
            try:
                return json.loads(chunk.strip())
            except json.JSONDecodeError:
                continue
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if starts:
        start = min(starts)
        closer = "]" if text[start] == "[" else "}"
        end = text.rfind(closer)
        if end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("no JSON found in the model's reply")
