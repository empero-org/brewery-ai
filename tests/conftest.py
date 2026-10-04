from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import pytest

warnings.filterwarnings("ignore", message=".*PyTorch.*")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never touch the real ~/.config/homebrew-ai during tests."""
    monkeypatch.setenv("HOMEBREW_AI_HOME", str(tmp_path / "config"))
    from homebrew_ai.models import registry

    registry.load_profiles.cache_clear()
    yield
    registry.load_profiles.cache_clear()


@pytest.fixture(scope="session")
def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, qwen_tokenizer):
    """A 2-layer random Qwen3 with the real Qwen3 tokenizer (CPU-friendly)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    path = tmp_path_factory.mktemp("tiny") / "tiny-qwen3"
    cfg = Qwen3Config(vocab_size=len(qwen_tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    qwen_tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_model(tiny_model_dir):
    """A ResolvedModel describing the tiny model with the qwen3 family profile."""
    from homebrew_ai.models.registry import from_snapshot, get_model, snapshot

    snap = snapshot(get_model("Qwen/Qwen3-0.6B"))
    snap["variants"][0].update(id=str(tiny_model_dir), label="tiny", params_b=0.01, layers=2, hidden=64, intermediate=128, heads=4, kv_heads=2, head_dim=16)
    return from_snapshot(snap)


class FakeUI:
    """Scripted stand-in for ConsoleUI."""

    interactive = False

    def __init__(self, answers: list[Any] | None = None, confirm: bool = True):
        self.answers = list(answers or [])
        self.confirm_answer = confirm
        self.log: list[tuple[str, Any]] = []

    def _next(self, default=None):
        return self.answers.pop(0) if self.answers else default

    def stream(self, chunk): self.log.append(("stream", chunk))
    def end_stream(self): pass
    def progress(self, note): self.log.append(("progress", note))
    def tool_started(self, name, args): self.log.append(("tool", name))
    def tool_finished(self, name, is_error, text): self.log.append(("tool_done", (name, is_error)))
    def warn(self, m): self.log.append(("warn", m))
    def error(self, m): self.log.append(("error", m))
    def choose(self, question, options, *, allow_other=True, multi=False): return self._next(options[0]["label"] if options else None)
    def confirm(self, question, *, default=False, details=None):
        self.log.append(("confirm", question))
        return self.confirm_answer
    def text(self, question, *, default=None): return self._next(default or "")
    def secret(self, question): return self._next("")
    def info(self, message, *, title=None, style="info"): self.log.append(("info", message))
    def markdown(self, text): self.log.append(("markdown", text))
    def table(self, title, columns, rows): self.log.append(("table", title))
    def segments(self, title, segments): self.log.append(("segments", segments))
    def pick_better(self, prompt, a, b, index, total): return self._next("A")
    def watch(self, poll, *, interval=10.0, timeout_s=None):
        import time
        for _ in range(600):
            st = poll()
            if st.get("state") in ("completed", "failed", "stopped", "crashed", "cancelled"):
                return st
            time.sleep(0.5)
        return st

    from contextlib import contextmanager

    @contextmanager
    def activity(self, label):
        class _A:
            def update(self, m): pass
        yield _A()


@pytest.fixture
def fake_ui():
    return FakeUI()


@pytest.fixture
def project(tmp_path):
    from homebrew_ai.project import Project

    p = Project.create(tmp_path / "proj", "test-brew")
    p.state.level = "builder"
    p.save()
    return p
