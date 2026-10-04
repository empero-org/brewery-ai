"""The agent loop: user turn → model → tools → model → … until it answers.

History is append-only (see ``backends/base.py``). When it grows past the
backend's budget it is compacted the simple way: the whole conversation is
summarised into one message and a fresh history starts from it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from typing import Any

from homebrew_ai.agent import tools as _tools  # noqa: F401  (registers all tools)
from homebrew_ai.agent.levels import get_level
from homebrew_ai.agent.phases import tools_for_phase
from homebrew_ai.agent.prompts import build_system_prompt, kickoff_message, state_block
from homebrew_ai.agent.tools.base import ToolContext, run_tool, specs
from homebrew_ai.backends.base import Backend, BackendError, Message
from homebrew_ai.backends.presets import CAPABILITIES

SESSION_FILE = "session.json"


class Agent:
    def __init__(self, project: Any, settings: Any, backend: Backend, ui: Any, jobs: Any):
        self.project = project
        self.settings = settings
        self.backend = backend
        self.ui = ui
        self.capability = CAPABILITIES.get(backend.capability, CAPABILITIES["medium"])
        self.system = build_system_prompt(backend.capability)
        self.ctx = ToolContext(project=project, ui=ui, settings=settings, backend=backend, jobs=jobs)
        self.history: list[Message] = []
        self.usage: Counter = Counter()
        self.pending_notes: list[str] = []
        self._last_state_hash: str | None = None
        self.load()

    # -- persistence ---------------------------------------------------------

    @property
    def session_path(self):
        return self.project.session_dir / SESSION_FILE

    def load(self) -> None:
        if not self.session_path.exists():
            return
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.history = [Message.from_dict(m) for m in data.get("history", [])]
        self.usage.update(data.get("usage", {}))
        if data.get("system_hash") != _hash(self.system):
            # a different prompt (new Homebrew version or backend tier): start from a summary-free fresh history
            # but keep the old transcript on disk
            self.archive("prompt-changed")

    def save(self) -> None:
        data = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": self.backend.label,
            "system_hash": _hash(self.system),
            "usage": dict(self.usage),
            "history": [m.to_dict() for m in self.history],
        }
        tmp = self.session_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.session_path)

    def archive(self, reason: str) -> None:
        if self.history:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = self.project.session_dir / f"session-{stamp}-{reason}.json"
            path.write_text(json.dumps([m.to_dict() for m in self.history], ensure_ascii=False), encoding="utf-8")
        self.history = []

    # -- helpers -------------------------------------------------------------

    def _summary(self) -> dict[str, Any]:
        s = self.project.summary()
        level = get_level(self.project.state.level)
        s["level"] = level.key
        return s

    def _state(self, force: bool = False) -> str | None:
        summary = self._summary()
        digest = _hash(json.dumps(summary, sort_keys=True, default=str))
        if not force and digest == self._last_state_hash and not self.pending_notes:
            return None
        self._last_state_hash = digest
        block = state_block(summary, self.pending_notes)
        self.pending_notes = []
        return block

    def tool_specs(self):
        if self.capability.scoped_tools:
            return specs(tools_for_phase(self.project.state.phase), self.project.state.level)
        return specs()  # every tool, always the same list: keeps the prompt cache and thinking valid

    def note(self, text: str) -> None:
        """Queue a Homebrew notice for the model (e.g. 'the user switched level')."""
        self.pending_notes.append(f"[Homebrew notice] {text}")

    def history_chars(self) -> int:
        return sum(len(p) for m in self.history for p in m.parts) + sum(len(json.dumps(c.arguments)) for m in self.history for c in m.tool_calls)

    # -- conversation --------------------------------------------------------

    def start(self) -> None:
        resumed = bool(self.history) or self.project.state.phase not in ("welcome", "goal")
        if self.history:
            self.send(kickoff_message(self.project.state.level or "hobbyist", resumed=True), from_user=False)
        else:
            self.send(kickoff_message(self.project.state.level or "hobbyist", resumed=resumed), from_user=False)

    def send(self, text: str, *, from_user: bool = True) -> None:
        if self.history_chars() > self.capability.history_budget_chars:
            self.compact()
        parts = [text]
        block = self._state(force=True)
        if block:
            parts.append(block)
        self.history.append(Message("user", parts))
        self.save()
        self._loop()

    def _loop(self) -> None:
        for _ in range(self.capability.max_steps):
            try:
                turn = self.backend.chat(self.system, self.history, self.tool_specs(), on_text=self.ui.stream, on_progress=self.ui.progress)
            except BackendError as exc:
                self.ui.end_stream()
                self.ui.error(str(exc))
                self.save()
                return
            except Exception as exc:  # never let a provider hiccup take the whole session down
                self.ui.end_stream()
                self.ui.error(f"unexpected problem talking to the AI ({type(exc).__name__}: {str(exc)[:300]}). Your project is saved; try again.")
                self.save()
                return
            finally:
                self.ui.end_stream()
            self.history.append(turn.message)
            for k, v in turn.usage.items():
                self.usage[k] += v
            if turn.stop_reason == "refusal":
                self.ui.warn(f"The AI declined to continue ({turn.refusal}). Try rephrasing your request.")
                break
            calls = turn.message.tool_calls
            if not calls:
                if turn.stop_reason == "max_tokens":
                    self.ui.warn("The reply was cut off. Say 'continue' if you want the rest.")
                break
            results: list[Message] = []
            if turn.stop_reason == "max_tokens":
                for call in calls:
                    results.append(Message("tool", ["error: your reply hit the length limit before this tool call was complete; try again with a shorter call"], tool_call_id=call.id, name=call.name, is_error=True))
                self.history.extend(results)
                continue
            try:
                for call in calls:
                    self.ui.tool_started(call.name, call.arguments)
                    text, is_error = run_tool(call, self.ctx, limit=self.capability.tool_result_chars)
                    self.ui.tool_finished(call.name, is_error, text)
                    results.append(Message("tool", [text], tool_call_id=call.id, name=call.name, is_error=is_error))
            except KeyboardInterrupt:
                done = {r.tool_call_id for r in results}
                for call in calls:
                    if call.id not in done:
                        results.append(Message("tool", ["interrupted by the user (Ctrl+C)"], tool_call_id=call.id, name=call.name, is_error=True))
                self.history.extend(results)
                self.save()
                raise
            block = self._state()
            if block:
                results[-1].parts.append(block)
            self.history.extend(results)
            self.save()
        else:
            self.ui.warn("I paused after many steps in a row — tell me how you'd like to continue.")
        self.save()

    def compact(self) -> None:
        """Summarise the conversation and start a fresh history from the summary."""
        transcript = []
        for m in self.history:
            if m.role == "user":
                transcript.append("USER: " + m.parts[0][:2000])
            elif m.role == "assistant":
                if m.text:
                    transcript.append("ASSISTANT: " + m.text[:2000])
                for c in m.tool_calls:
                    transcript.append(f"TOOL CALL {c.name}: {json.dumps(c.arguments)[:300]}")
            else:
                transcript.append(f"TOOL RESULT {m.name}: {(m.parts[0] if m.parts else '')[:400]}")
        text = "\n".join(transcript)[-120_000:]
        try:
            summary = self.backend.complete(
                "You summarise a conversation between a user and the Homebrew fine-tuning assistant so it can continue seamlessly.",
                "Summarise the conversation below in at most 400 words. Keep: the user's goal and preferences, decisions made, "
                "what was tried and failed, open questions, and the next planned step. The project state is tracked separately.\n\n" + text,
                max_tokens=2000,
            )
        except BackendError:
            summary = "(The earlier conversation could not be summarised; rely on the project state.)"
        self.archive("compacted")
        self.history.append(Message("user", ["[Homebrew] The conversation was getting long, so here is a summary of it so far:\n" + summary]))
        self.history.append(Message("assistant", ["Thanks — I have the summary and the project state, and I'll continue from there."]))
        self._last_state_hash = None


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
