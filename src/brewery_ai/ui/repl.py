"""The interactive session: chat with the brewmaster, plus a few slash commands."""

from __future__ import annotations

from pathlib import Path

from brewery_ai.agent.levels import LEVELS, get_level
from brewery_ai.agent.loop import Agent
from brewery_ai.agent.phases import PHASE_BY_KEY
from brewery_ai.backends.presets import describe
from brewery_ai.jobs.manager import JobManager, active_job
from brewery_ai.project import Project
from brewery_ai.ui.console import ConsoleUI

HELP = """
**Just type** to talk to the brewmaster. Commands:

| command | what it does |
|---|---|
| `/status` | project overview |
| `/jobs` | training jobs and their state |
| `/watch` | live view of the running training job |
| `/gallery` | web page with the preview images of image-LoRA checkpoints |
| `/level` | change how technical the conversation is |
| `/usage` | tokens used by the guiding AI this project |
| `/thinking` | show the AI's thinking: `on` (shortened), `full` or `off` |
| `/help` | this help |
| `/quit` | leave (training keeps running; reopen with `brewery` in this folder) |

Ctrl+C interrupts the brewmaster; Ctrl+D leaves.
"""


def _enable_line_editing(project: Project) -> Path | None:
    """Arrow keys and history for the input bar (readline, where available)."""
    try:
        import readline
    except ImportError:  # e.g. Windows without pyreadline
        return None
    history = project.session_dir / "input_history"
    try:
        readline.read_history_file(history)
    except (OSError, ValueError):
        pass
    readline.set_history_length(500)
    return history


def _save_history(path: Path | None) -> None:
    if path is None:
        return
    try:
        import readline

        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(path)
    except (ImportError, OSError):
        pass


def _short(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def footer(project: Project, agent: Agent, model: str) -> str:
    """Status line under the input bar: project · model · level · phase · tokens · context."""
    s = project.state
    phase = PHASE_BY_KEY.get(s.phase)
    level = get_level(s.level)
    u = agent.usage
    tokens_in = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    parts = [s.name, model, level.title, phase.title if phase else s.phase]
    if tokens_in or u.get("output_tokens"):
        parts.append(f"↑{_short(tokens_in)} ↓{_short(u.get('output_tokens', 0))}")
    budget = agent.capability.history_budget_chars
    if budget:
        parts.append(f"context {min(99, round(100 * agent.history_chars() / budget))}%")
    return " · ".join(p for p in parts if p)


def run_repl(project: Project, settings, backend, ui: ConsoleUI) -> None:
    jobs = JobManager(project)
    agent = Agent(project, settings, backend, ui, jobs)
    info = describe(settings.backend)
    ui.agent_label = info["model"]
    ui.banner(f"project: {project.state.name} · guide: {info['model']} · level: {get_level(project.state.level).title}")
    ui.console.print("[dim] Type to chat · /help for commands · Ctrl+C interrupts · Ctrl+D leaves[/]")
    history = _enable_line_editing(project) if ui.interactive else None
    try:
        agent.start()
    except KeyboardInterrupt:
        ui.end_stream()
        ui.console.print("[dim](interrupted)[/]")
    while True:
        try:
            text = ui.read_input(footer(project, agent, info["model"])).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not text:
            continue
        if text.startswith("/"):
            if handle_command(text, project, agent, jobs, ui):
                break
            continue
        try:
            agent.send(text)
        except KeyboardInterrupt:
            ui.end_stream()
            ui.console.print("[dim](interrupted — say what you'd like instead)[/]")
    _save_history(history)
    agent.save()
    ui.console.print("[dim]Saved. Run `brewery` in this folder to continue later.[/]")


def handle_command(text: str, project: Project, agent: Agent, jobs: JobManager, ui: ConsoleUI) -> bool:
    cmd, _, arg = text[1:].partition(" ")
    cmd = cmd.lower()
    if cmd in ("quit", "exit", "q"):
        return True
    if cmd in ("help", "h", "?"):
        ui.markdown(HELP)
    elif cmd == "status":
        s = project.state
        phase = PHASE_BY_KEY.get(s.phase)
        rows = [
            ["phase", phase.title if phase else s.phase],
            ["goal", s.goal or "–"],
            ["base model", s.base_model or "–"],
            ["compute", (s.compute.kind or "–") + (f" ({s.compute.ssh['host']})" if s.compute.ssh else "")],
            ["datasets", ", ".join(f"{k} ({v.records})" for k, v in s.datasets.items()) or "–"],
            ["training sets", ", ".join(f"{k} ({v.get('num_train')})" for k, v in s.training_sets.items()) or "–"],
            ["drafts", ", ".join(s.drafts) or "–"],
            ["jobs", ", ".join(f"{j['job_id']} [{j.get('state')}]" for j in s.jobs[-4:]) or "–"],
        ]
        ui.table(f"Project {s.name}", ["", ""], rows)
    elif cmd == "jobs":
        if not project.state.jobs:
            ui.console.print("No training jobs yet.")
        for j in project.state.jobs[-8:]:
            try:
                st = jobs.status(j["job_id"], log_lines=0)
            except Exception as exc:
                st = {"state": f"unknown ({str(exc)[:60]})"}
            ui.console.print(f"• {j['job_id']}  {j.get('objective', 'sft')}/{j.get('method')}  [bold]{st.get('state')}[/]  step {st.get('step', '–')}/{st.get('max_steps', '–')}  loss {st.get('loss', '–')}")
    elif cmd == "watch":
        active = active_job(project.state.jobs)
        if active is None:
            ui.console.print("No training job to watch.")
        else:
            remote = project.state.compute.kind == "ssh"
            ui.watch(lambda: {"job_id": active["job_id"], **jobs.status(active["job_id"])}, interval=20.0 if remote else 5.0)
    elif cmd == "level":
        from brewery_ai.ui.setup import choose_level

        old = project.state.level
        project.state.level = arg if arg in LEVELS else choose_level(ui, old)
        project.save()
        if project.state.level != old:
            agent.note(f"The user switched their level from {old} to {project.state.level}; adapt your style from now on.")
            ui.console.print(f"[ok]✓ level: {get_level(project.state.level).title}[/]")
    elif cmd in ("gallery", "samples"):
        from brewery_ai.ui.gallery import get_gallery

        url = get_gallery(project).url
        has_images = any(j.get("modality") == "image" for j in project.state.jobs)
        ui.console.print(f"[key]🖼  Preview gallery: {url}[/]")
        ui.console.print("[dim]Open it in your browser; it updates while training runs." + ("" if has_images else " (No image LoRA run yet.)") + "[/]")
    elif cmd == "thinking":
        from brewery_ai.ui.console import THINKING_MODES

        if arg.strip().lower() in THINKING_MODES:
            ui.thinking_mode = arg.strip().lower()
        else:
            ui.thinking_mode = THINKING_MODES[(THINKING_MODES.index(ui.thinking_mode) + 1) % len(THINKING_MODES)]
        words = {"on": "shown (shortened)", "full": "shown in full", "off": "hidden"}
        ui.console.print(f"[ok]✓ thinking: {words[ui.thinking_mode]}[/]")
    elif cmd == "usage":
        u = agent.usage
        ui.console.print(f"input {u.get('input_tokens', 0):,} · output {u.get('output_tokens', 0):,} · cache read {u.get('cache_read_input_tokens', 0):,} · cache write {u.get('cache_creation_input_tokens', 0):,}")
    else:
        ui.console.print(f"Unknown command /{cmd}. Type /help.")
    return False


def open_project_dir(path: Path) -> Project:
    return Project.load(path)
