import json

import pytest

from brewery_ai.agent.loop import Agent
from brewery_ai.agent.tools.base import REGISTRY
from brewery_ai.backends.base import Backend, Message, ToolCall, Turn
from brewery_ai.jobs.manager import JobManager
from brewery_ai.settings import Settings


class ScriptedBackend(Backend):
    provider = "scripted"

    def __init__(self, turns):
        super().__init__("scripted-model", "high")
        self.turns = list(turns)
        self.calls = []

    def chat(self, system, messages, tools, *, on_text=None, on_progress=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        turn = self.turns.pop(0)
        if callable(turn):
            turn = turn(messages)
        if on_text and turn.message.text:
            on_text(turn.message.text)
        return turn

    def complete(self, system, prompt, **kw):
        return "summary of the conversation"


def say(text):
    return Turn(Message("assistant", [text]), "end")


def call(*calls):
    return Turn(Message("assistant", ["One moment."], [ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)]), "tool_use")


def make_agent(project, ui, turns):
    backend = ScriptedBackend(turns)
    return Agent(project, Settings(), backend, ui, JobManager(project)), backend


def test_tool_round_trip_updates_project_and_keeps_history_valid(project, fake_ui):
    fake_ui.answers = ["Qwen3 0.6B"]
    agent, backend = make_agent(project, fake_ui, [
        say("Welcome!"),
        call(("update_project", {"goal": "a pirate chatbot", "modality": "text", "phase": "model"}), ("list_base_models", {"max_params_b": 2})),
        call(("ask_user", {"question": "Which model?", "options": [{"label": "Qwen3 0.6B"}, {"label": "Gemma 3 1B"}]})),
        say("Great choice."),
    ])
    agent.start()
    agent.send("I want a pirate chatbot")
    assert project.state.goal == "a pirate chatbot" and project.state.phase == "model"
    roles = [m.role for m in agent.history]
    assert roles == ["user", "assistant", "user", "assistant", "tool", "tool", "assistant", "tool", "assistant"]
    # every tool call is answered, in order
    for i, m in enumerate(agent.history):
        if m.role == "assistant" and m.tool_calls:
            answered = [r.tool_call_id for r in agent.history[i + 1 : i + 1 + len(m.tool_calls)]]
            assert answered == [c.id for c in m.tool_calls]
    models = json.loads(agent.history[5].parts[0])["models"]
    assert all(r["params_b"] <= 2 for r in models)
    assert json.loads(agent.history[7].parts[0]) == {"answer": "Qwen3 0.6B"}
    # state travels in user/tool messages, the system prompt never changes
    assert "<brewery_state>" in agent.history[2].parts[-1]
    assert len({c["system"] for c in backend.calls}) == 1
    assert all(len(c["tools"]) == len(REGISTRY) for c in backend.calls)
    # session persisted and reloadable
    again, _ = make_agent(project, fake_ui, [])
    assert [m.role for m in again.history] == roles


def test_invalid_arguments_are_reported_not_executed(project, fake_ui):
    agent, _ = make_agent(project, fake_ui, [call(("update_project", {"phase": "nonsense"})), say("ok")])
    agent.send("go")
    result = agent.history[2]
    assert result.is_error and "must be one of" in result.parts[0]


def test_level_gated_tool(project, fake_ui):
    project.state.level = "beginner"
    project.save()
    agent, _ = make_agent(project, fake_ui, [call(("run_shell", {"command": "ls", "where": "local"})), say("ok")])
    agent.send("run ls")
    assert agent.history[2].is_error and "only available" in agent.history[2].parts[0]


def test_interrupt_during_tools_keeps_history_consistent(project, fake_ui, monkeypatch):
    agent, _ = make_agent(project, fake_ui, [call(("get_project_state", {}), ("explain_term", {"term": "lora"}))])

    def boom(ctx, args):
        raise KeyboardInterrupt

    monkeypatch.setattr(REGISTRY["explain_term"], "handler", boom)
    with pytest.raises(KeyboardInterrupt):
        agent.send("hi")
    tail = agent.history[-2:]
    assert [m.role for m in tail] == ["tool", "tool"] and tail[1].is_error


def test_refusal_and_compaction(project, fake_ui):
    agent, backend = make_agent(project, fake_ui, [Turn(Message("assistant", []), "refusal", refusal="cyber"), say("after compaction")])
    agent.send("something")
    assert any(kind == "warn" for kind, _ in fake_ui.log)
    agent.capability = type(agent.capability)("tiny", False, 10, 1000, 5)
    agent.send("continue")
    assert agent.history[0].parts[0].startswith("[Brewery] The conversation was getting long")
