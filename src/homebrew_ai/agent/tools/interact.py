"""Talking to the user and keeping the project state."""

from __future__ import annotations

from typing import Any

from homebrew_ai.agent.glossary import explain
from homebrew_ai.agent.tools.base import ToolContext, ToolError, tool
from homebrew_ai.project import PHASES


@tool(
    "ask_user",
    """Ask the user a question in the terminal and wait for the answer. Use it whenever the user must choose between
options (2-4 options, recommended option first) or provide a short piece of information. One question per call.""",
    {
        "question": {"type": "string", "description": "The question, phrased for the user's level."},
        "options": {
            "type": "array",
            "description": "Choices to offer. Leave empty for a free-text answer.",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                "required": ["label"],
            },
        },
        "allow_other": {"type": "boolean", "description": "Let the user type their own answer instead (default true)."},
        "multi_select": {"type": "boolean", "description": "Allow choosing several options (default false)."},
    },
    ["question"],
)
def ask_user(ctx: ToolContext, args: dict[str, Any]) -> Any:
    options = args.get("options") or []
    if not options:
        answer = ctx.ui.text(args["question"])
        return {"answer": answer}
    answer = ctx.ui.choose(
        args["question"],
        [{"label": o["label"], "description": o.get("description", "")} for o in options],
        allow_other=args.get("allow_other", True),
        multi=args.get("multi_select", False),
    )
    if answer is None:
        return {"answer": None, "note": "the user skipped the question"}
    return {"answer": answer}


@tool(
    "update_project",
    """Record decisions in the project state: the goal (what the model should do), the modality (text or image), the
current workflow phase, the project name, and short notes about user preferences worth remembering.
Call it when the goal becomes clear and whenever you move to a new phase.""",
    {
        "goal": {"type": "string", "description": "One or two sentences describing what the brewed model should do."},
        "modality": {"type": "string", "enum": ["text", "image"]},
        "phase": {"type": "string", "enum": list(PHASES)},
        "name": {"type": "string", "description": "Display name of the project."},
        "add_notes": {"type": "array", "items": {"type": "string"}, "description": "Short facts to remember (preferences, constraints)."},
    },
)
def update_project(ctx: ToolContext, args: dict[str, Any]) -> Any:
    s = ctx.project.state
    for key in ("goal", "modality", "phase", "name"):
        if args.get(key):
            setattr(s, key, args[key])
    for note in args.get("add_notes") or []:
        if note and note not in s.notes:
            s.notes.append(note[:300])
    ctx.project.save()
    return {"ok": True, "state": ctx.project.summary()}


@tool("get_project_state", "Return the full current project state (decisions, datasets, training data, jobs, exports).")
def get_project_state(ctx: ToolContext, args: dict[str, Any]) -> Any:
    s = ctx.project.state
    data = ctx.project.summary()
    if s.drafts:
        data["drafts"] = {
            jid: {
                "objective": d.get("objective"), "method": d.get("method"), "base_model": d.get("base_model"), "init_from": d.get("init_from"),
                "optim": d.get("optim"), "lora": d.get("lora"), "dpo": d.get("dpo"), "max_seq_len": d.get("data", {}).get("max_seq_len"),
                "training_data": d.get("data", {}).get("train"), "image": d.get("image"),
            }
            for jid, d in s.drafts.items()
        }
    if s.training_sets:
        data["training_sets"] = s.training_sets
    if s.compute.hardware:
        hw = s.compute.hardware
        data["hardware"] = {"gpus": hw.get("gpus"), "ram_gb": hw.get("ram_gb"), "disk": hw.get("disk"), "os": hw.get("os")}
    return data


@tool(
    "explain_term",
    "Get a short, reliable explanation of a fine-tuning term (LoRA, epoch, learning rate, VRAM, ...) at the user's level.",
    {"term": {"type": "string"}},
    ["term"],
)
def explain_term(ctx: ToolContext, args: dict[str, Any]) -> Any:
    entry = explain(args["term"], ctx.level)
    if entry is None:
        return {"term": args["term"], "explanation": None, "note": "not in the glossary; explain it yourself in one or two plain sentences"}
    return entry


def require(value: Any, message: str) -> Any:
    if not value:
        raise ToolError(message)
    return value
