"""Terminal UI: rich output + questionary prompts, with plain fallbacks when not on a TTY."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
from typing import Any, Callable

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from brewery_ai import about

THEME = Theme(
    {
        "brand": "bold #f2a541",
        "agent": "bold #f2a541",
        "agent.model": "grey50",
        "user": "bold #7fb8ff",
        "user.bar": "#e6edf3 on #2d333b",
        "user.mark": "bold #f2a541 on #2d333b",
        "think.head": "italic #b392f0",
        "think": "italic grey62",
        "tool": "#7aa2c7",
        "dim": "grey58",
        "ok": "green",
        "warn": "yellow",
        "err": "bold red",
        "train": "bold green",
        "key": "cyan",
    }
)

# Raw ANSI for the input bar (drawn around readline's input(), outside rich).
_ACCENT = "\x1b[38;2;242;165;65m"
_BAR_BG = "\x1b[48;2;45;51;59m"
_BAR_FG = "\x1b[38;2;230;237;243m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
THINKING_MODES = ("on", "full", "off")
THINK_CAP = 1500  # characters of thinking shown per block in "on" mode

TOOL_LABELS = {
    "ask_user": None,
    "update_project": None,
    "get_project_state": None,
    "explain_term": None,
    "detect_hardware": "checking hardware",
    "use_local_computer": "setting up this computer",
    "install_local_training_packages": "installing training packages",
    "estimate_requirements": "estimating memory, time and cost",
    "gpu_rental_guide": "preparing a GPU rental guide",
    "create_ssh_key": "setting up Brewery's SSH key",
    "connect_server": "connecting to your server",
    "prepare_server": "preparing the server",
    "list_base_models": "looking up base models",
    "get_model_details": "reading the model's guidelines",
    "select_base_model": "checking model access",
    "hf_login_status": "checking your Hugging Face login",
    "hf_login": "logging in to Hugging Face",
    "search_datasets": "searching Hugging Face datasets",
    "inspect_dataset": "inspecting the dataset",
    "import_dataset": "importing data",
    "import_images": "importing images",
    "add_examples": "adding examples",
    "preview_dataset": "rendering a preview",
    "dataset_stats": "measuring the data",
    "clean_dataset": "cleaning the dataset",
    "build_training_set": "building the training set",
    "generate_synthetic_data": "generating synthetic data",
    "caption_images": "captioning images",
    "propose_training_config": "preparing the recipe",
    "start_training": "starting training",
    "training_status": "checking on training",
    "watch_training": None,
    "stop_training": "stopping training",
    "open_preview_gallery": "opening the preview gallery",
    "test_model": "taste-testing the model",
    "fetch_results": "downloading results",
    "generate_candidates": "sampling candidate answers",
    "review_candidates": "collecting preferences",
    "package_model": "packaging the model",
    "upload_to_hf": "uploading to Hugging Face",
    "run_shell": "running a command",
}

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 32) -> str:
    if not values:
        return ""
    values = values[-width:]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in values)


def _fmt_eta(seconds: float | None) -> str:
    if not seconds:
        return "–"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


class _Activity:
    def __init__(self, status: Any):
        self.status = status

    def update(self, message: str) -> None:
        width = max(20, self.status.console.width - 4)  # one line: a wrapped spinner line redraws badly
        if len(message) > width:
            message = message[: width - 1] + "…"
        self.status.update(Text(message, style="dim"))


class ConsoleUI:
    def __init__(self, *, color: bool = True, emoji: bool = True, simple: bool | None = None):
        self.console = Console(theme=THEME, no_color=not color, emoji=emoji, highlight=False)
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self._simple = simple  # None: decide per prompt (the terminal size can change after start-up)
        # The framed input bar needs a little cursor movement; BREWERY_AI_PLAIN_INPUT=1 turns it off.
        self.fancy_input = self.interactive and color and not os.environ.get("BREWERY_AI_PLAIN_INPUT")
        mode = os.environ.get("BREWERY_AI_THINKING", "on").lower()
        self.thinking_mode = mode if mode in THINKING_MODES else "on"
        self.agent_label = ""
        self._buffer = ""
        self._streaming = False
        self._live: Live | None = None
        self._header_done = False
        self._think_active = False
        self._think_buf = ""
        self._think_shown = 0
        self._think_hidden = 0
        self._think_started = 0.0
        self._status: Any = None  # the spinner of the running activity, paused while asking the user

    @property
    def simple(self) -> bool:
        """Numbered prompts instead of arrow-key menus, which redraw badly in narrow or embedded terminals."""
        if self._simple is not None:
            return self._simple
        return (
            bool(os.environ.get("BREWERY_AI_SIMPLE_PROMPTS"))
            or os.environ.get("TERM_PROGRAM") == "claude-desktop"
            or self.console.width < 90
        )

    @simple.setter
    def simple(self, value: bool | None) -> None:
        self._simple = value

    # -- turns -------------------------------------------------------------

    def _header(self) -> None:
        """Print the brewmaster's name once per reply."""
        if self._header_done:
            return
        self._header_done = True
        line = Text("● brewmaster", style="agent")
        if self.agent_label:
            line.append(f"  {self.agent_label}", style="agent.model")
        self.console.print()
        self.console.print(line)

    def user_message(self, text: str) -> None:
        """Show what the user said as a shaded, full-width block."""
        line = Text("› ", style="user.mark")
        line.append(text, style="user.bar")
        self.console.print(Padding(line, (0, 1), style="user.bar", expand=True))

    def read_input(self, footer: str = "") -> str:
        """Read one message in a framed input bar with a status footer (like pi's editor).

        Layout while typing: accent rule / shaded input line / accent rule / dim footer. Afterwards the
        frame is cleared and the message stays in the transcript as a shaded block. Raises
        KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D) like ``input()``.
        """
        self.end_stream()
        self._header_done = False
        if not self.fancy_input:
            text = input("you ▸ ")
            return text
        out = sys.stdout
        width = max(20, self.console.width)
        rule = _ACCENT + "─" * width + _RESET
        foot = footer
        while cell_len(foot) > width - 2:
            foot = foot[:-2] + "…"
        out.write("\n" + rule + "\n" + _BAR_BG + " " * (width - 1) + _RESET + "\n" + rule + "\n " + _DIM + foot + _RESET + "\x1b[2A\r")
        out.flush()
        prompt = f"\001{_BAR_BG}{_ACCENT}\x1b[1m\002› \001\x1b[22m{_BAR_FG}\002"
        try:
            text = input(prompt)
        except (KeyboardInterrupt, EOFError):
            out.write(_RESET + "\r\x1b[1A\x1b[J")  # drop the frame (cursor is on the input line)
            out.flush()
            raise
        lines = text.split("\n")  # pasted text can contain line breaks
        rows = sum(max(1, math.ceil(((2 if i == 0 else 0) + cell_len(line)) / width)) for i, line in enumerate(lines))
        out.write(_RESET + f"\x1b[{rows + 1}A\r\x1b[J")  # back up to the top rule and clear the frame
        out.flush()
        if text.strip():
            self.user_message(text.strip())
        return text

    # -- streaming agent text ----------------------------------------------

    def stream(self, chunk: str) -> None:
        """Agent text arrives in pieces; finished paragraphs are rendered as Markdown (append-only)."""
        if not chunk:
            return
        self._end_thinking()
        self._header()
        self._streaming = True
        self._buffer += chunk
        self._flush_paragraphs()

    def _flush_paragraphs(self) -> None:
        start = 0
        while True:
            idx = self._buffer.find("\n\n", start)
            if idx < 0:
                return
            block = self._buffer[:idx]
            if block.count("```") % 2 == 1:  # inside a code block: wait until it is closed
                start = idx + 2
                continue
            self._buffer = self._buffer[idx + 2:]
            start = 0
            if block.strip():
                self._print_markdown(block)

    def _print_markdown(self, text: str) -> None:
        self.console.print(Padding(Markdown(text), (0, 0, 0, 2)))

    def end_stream(self) -> None:
        self._end_thinking()
        if self._streaming and self._buffer.strip():
            self._print_markdown(self._buffer)
        self._streaming = False
        self._buffer = ""

    # -- thinking ------------------------------------------------------------

    def thinking(self, delta: str) -> None:
        """The model's thinking (or Claude's progress updates), shown dim and italic, separate from the reply."""
        if not delta or self.thinking_mode == "off":
            return
        if self._streaming:  # thinking after some reply text: finish that text first
            if self._buffer.strip():
                self._print_markdown(self._buffer)
            self._buffer = ""
            self._streaming = False
        self._header()
        if not self._think_active:
            self._think_active = True
            self._think_started = time.time()
            self._think_buf, self._think_shown, self._think_hidden = "", 0, 0
            self.console.print(Text("  ✻ thinking", style="think.head"))
        self._think_buf += delta
        while "\n" in self._think_buf:
            line, self._think_buf = self._think_buf.split("\n", 1)
            self._think_line(line)
        if len(self._think_buf) > 400:  # long paragraph without line breaks: show it sentence by sentence
            cut = self._think_buf.rfind(". ", 0, len(self._think_buf) - 1)
            if cut > 100:
                self._think_line(self._think_buf[: cut + 1])
                self._think_buf = self._think_buf[cut + 2:]

    progress = thinking  # backends call on_progress with thinking/progress text

    def _think_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self.thinking_mode != "full":
            room = THINK_CAP - self._think_shown
            if room <= 0:
                self._think_hidden += len(line)
                return
            if len(line) > room:
                self._think_hidden += len(line) - room
                line = line[:room].rstrip() + "…"
        self._think_shown += len(line)
        self.console.print(Padding(Text(line, style="think"), (0, 0, 0, 4)))

    def _end_thinking(self) -> None:
        if not self._think_active:
            return
        self._think_active = False
        if self._think_buf.strip():
            self._think_line(self._think_buf)
        self._think_buf = ""
        took = time.time() - self._think_started
        note = f"  ✻ thought for {took:.0f}s" if took >= 1 else "  ✻ done thinking"
        if self._think_hidden:
            note += f" · {self._think_hidden:,} more characters hidden (/thinking full shows everything)"
        self.console.print(Text(note, style="think.head"))

    # -- tools ---------------------------------------------------------------

    def tool_started(self, name: str, args: dict[str, Any]) -> None:
        self.end_stream()
        label = TOOL_LABELS.get(name, name.replace("_", " "))
        if label:
            self._header()
            self.console.print(Text(f"  ⚙ {label}…", style="tool"))

    def tool_finished(self, name: str, is_error: bool, text: str) -> None:
        if is_error and TOOL_LABELS.get(name, name):
            first = text.splitlines()[0][:160] if text else ""
            self.console.print(Text(f"  ✗ {first}", style="warn"))

    def warn(self, message: str) -> None:
        self.end_stream()
        self.console.print(Text(f"⚠ {message}", style="warn"))

    def error(self, message: str) -> None:
        self.end_stream()
        self.console.print(Panel(message, title="Problem", border_style="red", box=box.ROUNDED))

    # -- prompts -------------------------------------------------------------

    def _q(self):
        import questionary

        return questionary

    def choose(self, question: str, options: list[dict[str, str]], *, allow_other: bool = True, multi: bool = False) -> Any:
        with self._asking():
            return self._choose(question, options, allow_other=allow_other, multi=multi)

    def _choose(self, question: str, options: list[dict[str, str]], *, allow_other: bool = True, multi: bool = False) -> Any:
        self.end_stream()
        labels = [o["label"] for o in options]
        other = "✍  Something else (type it)"
        if not self.interactive or self.simple:
            self.console.print(Text(f"? {question}", style="bold"))
            for i, o in enumerate(options, 1):
                line = Text(f"  {i}) ", style="key")
                line.append(o["label"], style="bold")
                if o.get("description"):
                    line.append(f" — {o['description']}", style="dim")
                self.console.print(line)
            hint = "numbers separated by commas" if multi else f"1-{len(labels)}" + (", or type your own answer" if allow_other else "")
            while True:
                raw = input(f"  choose ({hint}): ").strip()
                if multi:
                    picks = [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit() and 1 <= int(x) <= len(labels)]
                    if picks:
                        return [labels[i - 1] for i in picks]
                elif raw.isdigit() and 1 <= int(raw) <= len(labels):
                    return labels[int(raw) - 1]
                elif raw and allow_other:
                    return raw
                self.console.print(Text("  please enter one of the numbers above", style="warn"))
        q = self._q()
        width = max(40, self.console.width - 10)

        def title(o: dict[str, str]) -> str:
            text = o["label"] + (f"  — {o['description']}" if o.get("description") else "")
            return text if len(text) <= width else text[: width - 1] + "…"  # wrapped lines garble the menu

        choices = [q.Choice(title=title(o), value=o["label"]) for o in options]
        if allow_other and not multi:
            choices.append(q.Choice(title=other, value=other))
        if multi:
            answer = q.checkbox(question, choices=choices, qmark="?").ask()
        else:
            answer = q.select(question, choices=choices, qmark="?", use_shortcuts=len(choices) <= 9).ask()
        if answer == other:
            answer = q.text("Your answer:", qmark="✍").ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def confirm(self, question: str, *, default: bool = False, details: str | None = None) -> bool:
        with self._asking():
            return self._confirm(question, default=default, details=details)

    def _confirm(self, question: str, *, default: bool = False, details: str | None = None) -> bool:
        self.end_stream()
        if details:
            self.console.print(Panel(details, border_style="key", box=box.ROUNDED))
        if not self.interactive or self.simple:
            raw = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
            return default if not raw else raw in ("y", "yes", "j", "ja")
        answer = self._q().confirm(question, default=default, qmark="?").ask()
        if answer is None:
            raise KeyboardInterrupt
        return bool(answer)

    def text(self, question: str, *, default: str | None = None) -> str:
        with self._asking():
            return self._text(question, default=default)

    def _text(self, question: str, *, default: str | None = None) -> str:
        self.end_stream()
        if not self.interactive or self.simple:
            suffix = f" [{default}]" if default else ""
            return input(f"? {question}{suffix} ").strip() or (default or "")
        answer = self._q().text(question, default=default or "", qmark="?").ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def secret(self, question: str) -> str:
        with self._asking():
            return self._secret(question)

    def _secret(self, question: str) -> str:
        self.end_stream()
        if not self.interactive or self.simple:
            import getpass

            return getpass.getpass(f"{question}: ")
        answer = self._q().password(question, qmark="🔑").ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer

    # -- output --------------------------------------------------------------

    def info(self, message: str, *, title: str | None = None, style: str = "info") -> None:
        self.end_stream()
        border = {"warning": "yellow", "key": "cyan", "error": "red"}.get(style, "#f2a541")
        self.console.print(Panel(message, title=title, border_style=border, box=box.ROUNDED))

    def markdown(self, text: str) -> None:
        self.end_stream()
        self.console.print(Markdown(text))

    def table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        self.end_stream()
        t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold", header_style="key")
        for c in columns:
            t.add_column(str(c), overflow="fold")
        for r in rows:
            t.add_row(*[("" if v is None else str(v)) for v in r])
        self.console.print(t)

    @contextlib.contextmanager
    def activity(self, label: str):
        self.end_stream()
        if self._status is not None:  # already inside an activity (e.g. a tool's own): just relabel
            yield _Activity(self._status)
            return
        with self.console.status(f"[dim]{label}…[/]", spinner="dots") as status:
            self._status = status
            try:
                yield _Activity(status)
            finally:
                self._status = None

    @contextlib.contextmanager
    def _asking(self):
        """Pause the spinner while a question waits for an answer (it would redraw over the prompt)."""
        status = self._status
        if status is not None:
            status.stop()
        try:
            yield
        finally:
            if status is not None and self._status is status:
                status.start()

    def segments(self, title: str, segments: list[tuple[str, bool]]) -> None:
        self.end_stream()
        text = Text()
        for piece, trainable in segments:
            if not trainable and len(piece) > 700:
                piece = piece[:300] + f"\n… [{len(piece) - 600} characters] …\n" + piece[-300:]
            text.append(piece, style="train" if trainable else "dim")
        self.console.print(Panel(text, title=title, subtitle="[train]green[/] = what the model learns", border_style="dim", box=box.ROUNDED))

    def pick_better(self, prompt: str, a: str, b: str, index: int, total: int) -> str:
        with self._asking():
            return self._pick_better(prompt, a, b, index, total)

    def _pick_better(self, prompt: str, a: str, b: str, index: int, total: int) -> str:
        self.end_stream()
        self.console.rule(f"[key]Preference {index}/{total}[/]")
        self.console.print(Panel(Markdown(prompt[:3000]), title="Prompt", border_style="cyan", box=box.ROUNDED))
        self.console.print(Panel(Markdown(a[:4000]), title="Answer A", border_style="#7fb8ff", box=box.ROUNDED))
        self.console.print(Panel(Markdown(b[:4000]), title="Answer B", border_style="#c792ea", box=box.ROUNDED))
        options = [("A is better", "A"), ("B is better", "B"), ("About the same", "tie"), ("Both are bad", "both_bad"), ("Stop reviewing", "stop")]
        if not self.interactive or self.simple:
            raw = input("Better answer? [a/b/t(ie)/x(both bad)/s(top)] ").strip().lower()[:1]
            return {"a": "A", "b": "B", "t": "tie", "x": "both_bad", "s": "stop"}.get(raw, "tie")
        q = self._q()
        answer = q.select("Which answer is better?", choices=[q.Choice(t, value=v) for t, v in options], qmark="?").ask()
        return answer or "stop"

    def watch(self, poll: Callable[[], dict[str, Any]], *, interval: float = 10.0, timeout_s: float | None = None) -> dict[str, Any]:
        self.end_stream()
        start = time.time()
        try:
            status: dict[str, Any] = poll()
        except Exception as exc:  # server unreachable right now: keep trying instead of crashing
            status = {"state": "unknown", "message": f"(could not reach the job yet: {str(exc)[:80]})"}
        self.console.print(Text("  watching training — press Ctrl+C to stop watching (training keeps running)", style="dim"))
        if self.simple:
            return self._watch_lines(poll, status, interval, timeout_s, start)
        try:
            with Live(self._watch_panel(status), console=self.console, refresh_per_second=2) as live:
                while status.get("state") not in ("completed", "failed", "stopped", "crashed", "cancelled"):
                    if timeout_s and time.time() - start > timeout_s:
                        status["detached"] = True
                        break
                    time.sleep(interval)
                    try:
                        status = poll()
                    except Exception as exc:  # flaky SSH: keep watching
                        status = {**status, "message": f"(connection hiccup: {str(exc)[:80]})"}
                    live.update(self._watch_panel(status))
        except KeyboardInterrupt:
            status["detached"] = True
        return status

    def _status_line(self, st: dict[str, Any]) -> Text:
        state = st.get("state", "?")
        step, total = st.get("step") or 0, st.get("max_steps") or 0
        parts = [time.strftime("%H:%M"), (st.get("objective") or "").upper(), state]
        if total:
            parts.append(f"step {step}/{total}")
        if st.get("loss") is not None:
            parts.append(f"loss {st['loss']:.4f}")
        if st.get("eval_loss") is not None:
            parts.append(f"eval {st['eval_loss']:.4f}")
        if st.get("rewards_accuracy") is not None:
            parts.append(f"prefers chosen {st['rewards_accuracy'] * 100:.0f}%")
        if total and st.get("eta_s"):
            parts.append(f"ETA {_fmt_eta(st.get('eta_s'))}")
        if st.get("message") and state not in ("training",):
            parts.append(str(st["message"])[:80])
        return Text("  " + " · ".join(p for p in parts if p), style="train" if state == "completed" else "dim")

    def _preview_line(self, st: dict[str, Any]) -> Text | None:
        if not st.get("preview_dir"):
            return None
        step = st.get("last_sample_step")
        label = "before training" if step == 0 else f"step {step}"
        line = Text(f"  🖼  previews ({label}): {st['preview_dir']}", style="key")
        if st.get("gallery_url"):
            line.append(f"\n     compare all checkpoints: {st['gallery_url']}", style="key")
        return line

    def _watch_lines(self, poll, status, interval, timeout_s, start) -> dict[str, Any]:
        last = None
        shown_preview = None
        try:
            while True:
                line = self._status_line(status)
                key = (status.get("state"), status.get("step"), status.get("message"))
                if key != last:
                    self.console.print(line)
                    last = key
                if status.get("preview_dir") and status.get("preview_dir") != shown_preview:
                    shown_preview = status["preview_dir"]
                    self.console.print(self._preview_line(status))
                if status.get("state") in ("completed", "failed", "stopped", "crashed", "cancelled"):
                    if status.get("state") in ("failed", "crashed"):
                        self.console.print(Text(f"  {status.get('error') or 'the process stopped unexpectedly'}", style="err"))
                        if status.get("hint"):
                            self.console.print(Text(f"  {status['hint']}", style="warn"))
                    break
                if timeout_s and time.time() - start > timeout_s:
                    status["detached"] = True
                    break
                time.sleep(interval)
                try:
                    status = poll()
                except Exception as exc:  # flaky SSH: keep watching
                    status = {**status, "message": f"(connection hiccup: {str(exc)[:60]})"}
        except KeyboardInterrupt:
            status["detached"] = True
        return status

    def _watch_panel(self, st: dict[str, Any]) -> Panel:
        state = st.get("state", "?")
        step, total = st.get("step") or 0, st.get("max_steps") or 0
        title = f"{st.get('job_id', '')}  ·  {(st.get('objective') or '').upper()}  ·  {state}"
        rows: list[Any] = []
        if total:
            width = 40
            filled = int(width * min(step, total) / total)
            bar = Text("█" * filled, style="train")
            bar.append("░" * (width - filled), style="dim")
            bar.append(f"  step {step}/{total}   ETA {_fmt_eta(st.get('eta_s'))}", style="dim")
            rows.append(bar)
        line = []
        if st.get("loss") is not None:
            line.append(f"loss {st['loss']:.4f}")
        if st.get("eval_loss") is not None:
            line.append(f"eval {st['eval_loss']:.4f}")
        if st.get("rewards_accuracy") is not None:
            line.append(f"prefers chosen {st['rewards_accuracy'] * 100:.0f}%")
        if st.get("gpu_mem_gb"):
            line.append(f"GPU {st['gpu_mem_gb']} GB")
        if line:
            rows.append(Text("   ".join(line)))
        hist = [v for _, v in (st.get("loss_history") or [])]
        if hist:
            rows.append(Text(sparkline(hist, 48), style="train"))
        if st.get("message") and state not in ("training",):
            rows.append(Text(str(st["message"]), style="dim"))
        preview = self._preview_line(st)
        if preview is not None:
            rows.append(preview)
        if state in ("failed", "crashed"):
            rows.append(Text(str(st.get("error") or "the process stopped unexpectedly"), style="err"))
            if st.get("hint"):
                rows.append(Text(str(st["hint"]), style="warn"))
        return Panel(Group(*rows) if rows else Text("starting…", style="dim"), title=title, border_style="green" if state == "completed" else "#f2a541", box=box.ROUNDED)

    # -- misc ----------------------------------------------------------------

    def banner(self, subtitle: str = "") -> None:
        self.console.print(Text(about.BANNER.rstrip("\n"), style="brand"))
        self.console.print(Text(f" Brewery {about.VERSION} · by {about.ORG} · brew your own AI model", style="bold"))
        if subtitle:
            self.console.print(Text(f" {subtitle}", style="dim"))
        self.console.print()
