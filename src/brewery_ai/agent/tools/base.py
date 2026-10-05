"""Tool plumbing: registry, context, input validation and confirmation gates.

Side effects that cost money, publish things or touch the user's machine are
confirmed by the user through ``ctx.ui`` *inside* the tool, so the model can
never skip a confirmation by phrasing.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from brewery_ai.backends.base import ToolCall, ToolSpec, validate_arguments

if TYPE_CHECKING:  # pragma: no cover
    from brewery_ai.backends.base import Backend
    from brewery_ai.jobs.manager import JobManager
    from brewery_ai.project import Project
    from brewery_ai.settings import Settings


class UI(Protocol):
    """What tools may do with the user's terminal."""

    def choose(self, question: str, options: list[dict[str, str]], *, allow_other: bool = True, multi: bool = False) -> Any: ...
    def confirm(self, question: str, *, default: bool = False, details: str | None = None) -> bool: ...
    def text(self, question: str, *, default: str | None = None) -> str: ...
    def secret(self, question: str) -> str: ...
    def info(self, message: str, *, title: str | None = None, style: str = "info") -> None: ...
    def markdown(self, text: str) -> None: ...
    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None: ...
    def activity(self, label: str) -> Any: ...  # context manager showing a spinner
    def segments(self, title: str, segments: list[tuple[str, bool]]) -> None: ...
    def watch(self, poll: Callable[[], dict[str, Any]], *, interval: float = 10.0, timeout_s: float | None = None) -> dict[str, Any]: ...
    def pick_better(self, prompt: str, a: str, b: str, index: int, total: int) -> str: ...  # "A" | "B" | "tie" | "both_bad" | "stop"


class ToolError(Exception):
    """An expected failure; the message is shown to the agent (and usually the user)."""


class Declined(ToolError):
    """The user said no at a confirmation prompt."""


@dataclass
class ToolContext:
    project: "Project"
    ui: UI
    settings: "Settings"
    backend: "Backend"
    jobs: "JobManager"
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        return self.project.state.level or "hobbyist"

    def confirm_or_raise(self, question: str, details: str | None = None) -> None:
        if not self.ui.confirm(question, default=False, details=details):
            raise Declined("the user declined")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Any]
    levels: tuple[str, ...] | None = None
    activity: str | None = None  # spinner label while running

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None, *, levels: tuple[str, ...] | None = None, activity: str | None = None):
    """Register a tool. ``properties`` is a JSON-schema properties mapping."""

    schema = {"type": "object", "properties": properties or {}, "additionalProperties": False}
    if required:
        schema["required"] = required

    def deco(fn: Callable[[ToolContext, dict[str, Any]], Any]):
        if name in REGISTRY:
            raise ValueError(f"duplicate tool {name}")
        REGISTRY[name] = Tool(name, description.strip(), schema, fn, levels, activity)
        return fn

    return deco


def _to_text(result: Any, limit: int) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) > limit:
        text = text[:limit] + f"… [truncated {len(text) - limit} chars]"
    return text


def run_tool(call: ToolCall, ctx: ToolContext, *, limit: int = 8000) -> tuple[str, bool]:
    """Execute one tool call. Returns ``(result_text, is_error)``."""
    tool_obj = REGISTRY.get(call.name)
    if tool_obj is None:
        return f"unknown tool {call.name!r}", True
    if call.parse_error:
        return json.dumps({"INVALID_JSON": call.parse_error}), True
    problems = validate_arguments(tool_obj.parameters, call.arguments)
    if problems:
        return "invalid arguments: " + "; ".join(problems[:6]), True
    if tool_obj.levels and ctx.level not in tool_obj.levels:
        return f"{call.name} is only available at these levels: {', '.join(tool_obj.levels)}", True
    try:
        if tool_obj.activity:
            with ctx.ui.activity(tool_obj.activity):
                result = tool_obj.handler(ctx, call.arguments)
        else:
            result = tool_obj.handler(ctx, call.arguments)
        return _to_text(result, limit), False
    except Declined:
        return "The user declined at the confirmation prompt. Ask what they would like to change.", True
    except ToolError as exc:
        return f"error: {exc}", True
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # unexpected: keep the session alive, give the agent something useful
        tail = traceback.format_exc().strip().splitlines()[-3:]
        return f"error: {type(exc).__name__}: {exc}\n" + "\n".join(tail), True


def specs(names: set[str] | None = None, level: str | None = None) -> list[ToolSpec]:
    out = []
    for t in REGISTRY.values():
        if names is not None and t.name not in names:
            continue
        if t.levels and level and level not in t.levels:
            continue
        out.append(t.spec())
    return out
