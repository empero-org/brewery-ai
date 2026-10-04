"""Regression tests for the correctness review (one test per finding where practical)."""

import io
import json
import types
from pathlib import Path

import pytest

from homebrew_ai.agent.tools.base import run_tool
from homebrew_ai.backends.base import ToolCall
from homebrew_ai.jobs.manager import JobManager, active_job
from homebrew_ai.models.registry import get_model
from homebrew_ai.remote import bootstrap
from homebrew_ai.remote.sshutil import SSHParseError, SSHSpec, parse_ssh_command
from homebrew_ai.remote.target import SSHTarget
from homebrew_ai.train.config import build_job, make_job_id
from homebrew_ai.train.status import STOP_FILE, StatusWriter, read_status

from test_tools import make_ctx, run


# --- #4 ssh options --------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    "ssh -p 41022 root@203.0.113.40 -L 8080:localhost:8080",  # Vast's default command
    "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@1.2.3.4 -p 2222",
    "ssh -oServerAliveInterval=30 -lroot 1.2.3.4",
])
def test_provider_commands_still_parse(cmd):
    spec = parse_ssh_command(cmd)
    assert spec.host and spec.port


@pytest.mark.parametrize("cmd", [
    'ssh root@1.2.3.4 -o ProxyCommand="sh -c id"',
    "ssh -oProxyCommand=id root@h",
    "ssh -l '-oProxyCommand=id' host",
    "ssh -o LocalCommand=id -o PermitLocalCommand=yes root@h",
    "ssh -J jump root@h",
    "ssh -F /tmp/evil root@h",
    "ssh -o UserKnownHostsFile=/home/x/.bashrc root@h",
    "ssh root@h rm -rf /",
])
def test_dangerous_ssh_commands_are_refused(cmd):
    with pytest.raises(SSHParseError):
        parse_ssh_command(cmd)


def test_stored_spec_is_rechecked():
    with pytest.raises(SSHParseError):
        SSHTarget(SSHSpec(host="h", options=["ProxyCommand=id"]))
    with pytest.raises(SSHParseError):
        SSHTarget(SSHSpec(host="-oProxyCommand=id"))


def test_connect_server_asks_first(project, fake_ui):
    fake_ui.confirm_answer = False
    text, err = run_tool(ToolCall("c", "connect_server", {"ssh_command": "ssh -p 2222 root@203.0.113.9"}), make_ctx(project, fake_ui))
    assert err and "declined" in text and ("confirm", "Connect to the GPU server 203.0.113.9?") in fake_ui.log


# --- #5 training sets never overwrite datasets ------------------------------


def _pirate(tmp_path, n=60):
    data = tmp_path / "pirate.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"Arr {i}"}]} for i in range(n)]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    return data


def test_training_set_named_like_dataset_keeps_dataset(project, fake_ui, tmp_path, qwen_tokenizer):
    ctx = make_ctx(project, fake_ui)
    run(ctx, "import_dataset", name="pirate", source="local", path=str(_pirate(tmp_path)), license="mit")
    for _ in range(3):
        ts = run(ctx, "build_training_set", name="pirate", parts=[{"name": "pirate"}])
        assert ts["num_train"] + ts["num_eval"] == 60
    assert sum(1 for _ in open(project.data_dir / "pirate.jsonl")) == 60
    assert ts["train"].startswith("data/_sets/pirate/")


# --- #13 / #19 drafts ---------------------------------------------------------


def test_reproposing_replaces_the_draft_and_repoints_later_stages(project, fake_ui, tmp_path, qwen_tokenizer):
    ctx = make_ctx(project, fake_ui)
    run(ctx, "select_base_model", model_id="Qwen/Qwen3-0.6B")
    run(ctx, "import_dataset", name="pirate", source="local", path=str(_pirate(tmp_path)), license="mit")
    run(ctx, "build_training_set", name="sft", parts=[{"name": "pirate"}])
    first = run(ctx, "propose_training_config", objective="sft")["job_id"]
    second = run(ctx, "propose_training_config", objective="sft", overrides={"epochs": 2})
    assert second["replaces_drafts"] == [first] and list(project.state.drafts) == [second["job_id"]]
    stage2 = run(ctx, "propose_training_config", objective="sft", init_from_job=second["job_id"], overrides={"epochs": 1})
    third = run(ctx, "propose_training_config", objective="sft", overrides={"epochs": 3})
    assert set(project.state.drafts) == {third["job_id"], stage2["job_id"]}
    assert project.state.drafts[stage2["job_id"]]["init_from"]["job_id"] == third["job_id"]


def test_job_ids_are_unique_within_a_second():
    assert len({make_job_id("brew") for _ in range(50)}) == 50


# --- #9 which job the user means ---------------------------------------------


def test_active_job_skips_cancelled_and_prefers_running():
    jobs = [{"job_id": "a", "state": "cancelled"}, {"job_id": "b", "state": "training"}, {"job_id": "c", "state": "queued"}]
    assert active_job(jobs)["job_id"] == "b"
    assert active_job([{"job_id": "a", "state": "cancelled"}, {"job_id": "b", "state": "completed"}])["job_id"] == "b"
    assert active_job([]) is None


# --- #7 / #11 requirements ----------------------------------------------------


def test_requirements_cover_8bit_optimizers_and_model_changes():
    info = get_model("Qwen/Qwen3-0.6B")
    assert not any("bitsandbytes" in r for r in bootstrap.worker_requirements(info, "lora", ["adamw_torch"]))
    assert any("bitsandbytes" in r for r in bootstrap.worker_requirements(info, "full", ["adamw_8bit"]))
    prepared = bootstrap.worker_requirements(info, "qlora")
    qwen35 = bootstrap.worker_requirements(get_model("Qwen/Qwen3.5-2B"), "lora")
    assert bootstrap.missing_requirements(prepared, qwen35) == ["flash-linear-attention>=0.4.2"]
    assert bootstrap.missing_requirements(qwen35, qwen35) == []


def test_local_worker_check_reads_url_requirements(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "diffusers" else real(name, *a))
    out = bootstrap.check_local_worker(["diffusers @ https://github.com/huggingface/diffusers/archive/abc.zip"])
    assert "diffusers" in out["missing"]


def test_start_training_notices_a_server_prepared_for_another_model(project, fake_ui, tmp_path, qwen_tokenizer):
    ctx = make_ctx(project, fake_ui)
    run(ctx, "select_base_model", model_id="Qwen/Qwen3-0.6B")
    run(ctx, "import_dataset", name="pirate", source="local", path=str(_pirate(tmp_path)), license="mit")
    run(ctx, "build_training_set", name="sft", parts=[{"name": "pirate"}])
    run(ctx, "propose_training_config", objective="sft", method="qlora")
    c = project.state.compute
    c.kind, c.ssh, c.prepared = "ssh", {"host": "203.0.113.9", "user": "root", "port": 22}, True
    c.requirements = bootstrap.BASE_REQUIREMENTS  # prepared before QLoRA was chosen: no bitsandbytes
    project.save()
    text, err = run_tool(ToolCall("s", "start_training", {}), ctx)
    assert err and "bitsandbytes" in text and "prepare_server" in text


# --- #2 / #15 status and stopping ---------------------------------------------


def test_only_rank_zero_reports_progress_but_any_rank_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    w = StatusWriter(tmp_path)
    w.set(state="training", step=3)
    w.log_metrics({"step": 3, "loss": 1.0})
    assert read_status(tmp_path) is None and not (tmp_path / "metrics.jsonl").exists()
    w.fail("boom")
    assert read_status(tmp_path)["state"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_uses_a_stop_file_for_the_whole_chain(project):
    runs = project.runs_dir
    for jid in ("s1", "s2"):
        (runs / jid).mkdir(parents=True)
        project.state.jobs.append({"job_id": jid, "run_dir": str(runs / jid), "chain_id": "s1", "pid": 0, "target_spec": {"kind": "local"}, "state": "training"})
    project.save()
    JobManager(project).stop("s2")
    assert (runs / "s1" / STOP_FILE).exists() and (runs / "s2" / STOP_FILE).exists()


def test_worker_notices_the_stop_file(tmp_path):
    from homebrew_ai.train import runner

    runner._STOP["requested"] = False
    assert not runner.stop_requested(tmp_path)
    (tmp_path / STOP_FILE).touch()
    assert runner.stop_requested(tmp_path)
    runner._STOP["requested"] = False


# --- #1 later stages resolve their start from "." ------------------------------


def test_resolve_start_works_from_inside_the_run_folder(tiny_model, tmp_path, monkeypatch):
    from homebrew_ai.train.runner import resolve_start

    def make(name, method, init=None):
        job, _ = build_job(model=tiny_model, project="t", method=method, init_from=init, stage=2 if init else 1, train_path="data/train.jsonl",
                           eval_path=None, num_train=8, bf16=False, overrides={"max_seq_len": 64, "optimizer": "adamw_torch"}, expert_override=True)
        job.job_id = name
        job.save(tmp_path / name / "job.yaml")
        return job

    make("cpt", "full")
    (tmp_path / "cpt" / "final").mkdir()
    sft = make("sft", "lora", {"job_id": "cpt", "kind": "full"})
    monkeypatch.chdir(tmp_path / "sft")
    assert resolve_start(sft, Path("."), tiny_model) == str(tmp_path / "cpt" / "final")


# --- #3 / #20 backends -----------------------------------------------------------


class _Block(types.SimpleNamespace):
    def model_dump(self, mode="json", by_alias=False, exclude_none=True):
        d = dict(self.__dict__)
        if by_alias and "from_" in d:
            d["from"] = d.pop("from_")
        return d


def _final(content, stop="end_turn"):
    return types.SimpleNamespace(content=content, stop_reason=stop, usage=None, stop_details=None)


def test_claude_fallback_drops_the_declined_models_tool_calls():
    pytest.importorskip("anthropic")
    from homebrew_ai.backends.anthropic_backend import AnthropicBackend

    b = AnthropicBackend("claude-opus-5-5", api_key="test")
    content = [
        _Block(type="thinking", thinking="…", signature="sig"),
        _Block(type="text", text="Let me "),
        _Block(type="tool_use", id="t1", name="run_shell", input={"command": "x"}),
        _Block(type="fallback", from_={"model": "claude-opus-5-5"}, to={"model": "claude-opus-5"}),
        _Block(type="text", text="check the data."),
        _Block(type="tool_use", id="t2", name="dataset_stats", input={"name": "pirate"}),
    ]
    turn = b._to_turn(_final(content, "tool_use"))
    assert [c.id for c in turn.message.tool_calls] == ["t2"]
    assert [blk["type"] for blk in turn.message.raw] == ["text", "text", "tool_use"]


def test_claude_refusal_leaves_a_valid_history_entry():
    pytest.importorskip("anthropic")
    from homebrew_ai.backends.anthropic_backend import AnthropicBackend

    b = AnthropicBackend("claude-opus-5-5", api_key="test")
    turn = b._to_turn(_final([_Block(type="tool_use", id="t1", name="x", input={})], "refusal"))
    assert turn.stop_reason == "refusal" and not turn.message.tool_calls
    assert turn.message.raw == [{"type": "text", "text": turn.message.parts[0]}]


def test_unknown_blocks_are_replayed_with_wire_names():
    pytest.importorskip("anthropic")
    from homebrew_ai.backends.anthropic_backend import _block_to_param

    assert _block_to_param(_Block(type="future_block", from_={"model": "m"})) == {"type": "future_block", "from": {"model": "m"}}


@pytest.mark.parametrize("text,expected", [
    ("This model's maximum context length is 8192 tokens (including 300 in the functions).", False),
    ("Invalid schema for function 'start_training'", False),
    ("tools is not supported for this model", True),
    ("No endpoints found that support tool use.", True),
    ("Unrecognized request argument supplied: tools", True),
])
def test_text_tool_mode_only_for_real_tool_errors(text, expected):
    from homebrew_ai.backends.openai_backend import _no_tool_support

    assert _no_tool_support(text) is expected


# --- UI ----------------------------------------------------------------------------


def _console_ui():
    from homebrew_ai.ui.console import ConsoleUI

    ui = ConsoleUI(color=False)
    ui.console.file = io.StringIO()
    ui.console.width = 80
    return ui


def test_reply_streaming_keeps_code_blocks_whole_and_shows_thinking_apart():
    ui = _console_ui()
    ui.thinking("Planning the recipe.\n")
    for piece in ["Here is code:\n\n```python\nx = 1\n", "\ny = 2\n```\n\nDone."]:
        ui.stream(piece)
    ui.end_stream()
    out = ui.console.file.getvalue()
    assert out.count("● brewmaster") == 1
    assert out.index("✻ thinking") < out.index("Planning the recipe.") < out.index("Here is code")
    assert "x = 1" in out and "y = 2" in out and out.rstrip().endswith("Done.")


def test_long_thinking_is_shortened_unless_full():
    ui = _console_ui()
    ui.thinking("word " * 1000 + "\n")
    ui.end_stream()
    assert "more characters hidden" in ui.console.file.getvalue()
    ui = _console_ui()
    ui.thinking_mode = "off"
    ui.thinking("secret plan\n")
    ui.end_stream()
    assert "secret plan" not in ui.console.file.getvalue()


# --- synthetic data: parallel batches with visible progress ----------------------


class _SlowWriter:
    label = "fake:writer"

    def supports_vision(self):
        return False

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.active = self.peak = 0

    def complete(self, system, prompt, *, max_tokens=8000, json_schema=None, images=None, on_progress=None, effort=None):
        self.efforts = getattr(self, "efforts", []) + [effort]
        import time

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            for i in range(3):
                time.sleep(0.05)
                if on_progress:
                    on_progress("thinking" if i < 2 else "writing", 100 * (i + 1))
            data = json.loads(prompt.split("\n\n")[1])  # the payload between the instruction and "Return ..."
            return json.dumps({"examples": [{"messages": [dict(m, content=m["content"].upper()) for m in ex["messages"]]} for ex in data["examples"]]})
        finally:
            with self.lock:
                self.active -= 1


def test_transform_runs_batches_in_parallel_and_reports_progress():
    from homebrew_ai.data import synth

    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer {i}"}]} for i in range(10)]
    backend, said = _SlowWriter(), []
    out, report = synth.transform(backend, rows, "shout", on_progress=said.append)
    assert report == {"rewritten": 10, "failed": 0}
    assert [r["messages"][1]["content"] for r in out] == [f"ANSWER {i}" for i in range(10)]  # order kept
    assert [r["messages"][0]["content"] for r in out] == [f"q{i}" for i in range(10)]  # only assistant turns change
    assert backend.peak > 1
    assert any("running" in s and ("written" in s or "thinking" in s) for s in said) and said[-1].startswith("10/10 rewritten ·")
    assert any(s.startswith(("4/10", "8/10")) and "running" in s for s in said)  # counted while others still run
    assert all(len(s) < 70 for s in said)


def test_ctrl_c_keeps_finished_batches():
    from homebrew_ai.data import synth

    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer {i}"}]} for i in range(12)]
    saved = []

    def save(batch):
        saved.extend(batch)
        raise KeyboardInterrupt  # the user presses Ctrl+C right after the first batch is saved

    out, report = synth.transform(_SlowWriter(), rows, "shout", on_batch=save)
    assert "stopped_early" in report and len(out) == len(saved) == 4


def test_synthetic_batches_are_saved_as_they_finish(project, fake_ui):
    from homebrew_ai.agent.tools.base import ToolContext
    from homebrew_ai.settings import Settings

    ctx = make_ctx(project, fake_ui)
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer {i}"}]} for i in range(10)]
    run(ctx, "add_examples", name="plain", records=rows)
    ctx = ToolContext(project=project, ui=fake_ui, settings=Settings(), backend=_SlowWriter(), jobs=JobManager(project))
    out = run(ctx, "generate_synthetic_data", name="loud", mode="transform", source_dataset="plain", brief="shout", count=10)
    assert out["added"] == 10 and project.state.datasets["loud"].records == 10 and project.state.datasets["loud"].synthetic
    assert sum(1 for _ in open(project.data_dir / "loud.jsonl")) == 10
    text, err = run_tool(ToolCall("x", "generate_synthetic_data", {"name": "plain", "mode": "transform", "source_dataset": "plain", "brief": "x"}), ctx)
    assert err and "new dataset name" in text


def test_clean_dataset_drops_empty_duplicate_and_matching_records(project, fake_ui):
    ctx = make_ctx(project, fake_ui)
    good = [{"messages": [{"role": "user", "content": f"Recipe for dish {i}"}, {"role": "assistant", "content": f"Grandma's dish {i}, step by step."}]} for i in range(6)]
    rows = good + [good[0], {"messages": [{"role": "user", "content": "Write me a recipe for: "}, {"role": "assistant", "content": "Ingredients: c(\"egg\")"}]}]
    rows.append({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "hello"}]})
    run(ctx, "add_examples", name="recipes", records=rows[:-1])
    (project.data_dir / "recipes.jsonl").open("a").write(json.dumps(rows[-1]) + "\n")  # add_examples would reject it
    out = run(ctx, "clean_dataset", name="recipes", drop_matching=r'c\("')
    assert out["kept"] == 6 and out["removed"] == {"duplicate": 1, "matched_pattern": 1, "invalid": 1}
    assert sum(1 for _ in open(project.data_dir / "recipes.jsonl")) == 6 and (project.root / out["backup"]).exists()
    assert project.state.datasets["recipes"].records == 6
    again = run(ctx, "clean_dataset", name="recipes")
    assert again["removed"] == {} and again["kept"] == 6
    text, err = run_tool(ToolCall("x", "clean_dataset", {"name": "recipes", "drop_matching": "Grandma|Recipe"}), ctx)
    assert err and "all 6 records" in text


def test_numbered_prompts_in_the_desktop_terminal(monkeypatch):
    ui = _console_ui()
    ui.console.width = 120
    monkeypatch.delenv("HOMEBREW_AI_SIMPLE_PROMPTS", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "other")
    assert not ui.simple
    monkeypatch.setenv("TERM_PROGRAM", "claude-desktop")
    assert ui.simple
    monkeypatch.setenv("TERM_PROGRAM", "other")
    ui.console.width = 72  # decided per prompt: a resize after start-up counts
    assert ui.simple
    ui.simple = False
    assert not ui.simple


# --- image previews during training -------------------------------------------------


class _FakeImagePipe:
    """Stands in for QwenImage21Pipeline.from_pretrained(...) without diffusers or a GPU."""

    fail_on_call = False
    calls: list[dict] = []

    def __init__(self, transformer):
        self.transformer = transformer
        self.vae = types.SimpleNamespace(to=lambda device: None)

    @classmethod
    def from_pretrained(cls, model_id, *, transformer, text_encoder, torch_dtype, **kw):
        assert text_encoder is None  # no second text encoder, no second transformer
        assert "processor" not in kw  # the real pipeline needs its (tiny) processor at start-up
        return cls(transformer)

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, *, prompt_embeds, prompt_embeds_mask, height, width, num_inference_steps, generator):
        from PIL import Image

        assert not self.transformer.training  # rendered in eval mode
        if self.fail_on_call:
            raise RuntimeError("CUDA out of memory")
        _FakeImagePipe.calls.append({"shape": tuple(prompt_embeds.shape), "size": (height, width), "steps": num_inference_steps})
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "orange")])


class _FakeTransformer:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


def test_previews_render_with_the_model_in_memory(tmp_path):
    import torch

    from homebrew_ai.train.image import Previews, preview_prompts
    from homebrew_ai.train.config import ImageSettings

    settings = ImageSettings(trigger_word="sks")
    captions = [["sks dog on grass"], ["sks dog on grass"], ["sks dog in snow"], ["sks dog asleep"], ["sks dog running"]]
    prompts = preview_prompts(settings, captions)
    assert prompts == ["sks dog on grass", "sks dog in snow", "sks dog asleep"]
    assert preview_prompts(ImageSettings(sample_prompts=["a", "b", "a"]), captions) == ["a", "b"]

    embeds = {p: (torch.zeros(5, 8), None) for p in prompts}
    status = StatusWriter(tmp_path)
    transformer = _FakeTransformer()
    _FakeImagePipe.calls = []
    previews = Previews(_FakeImagePipe, types.SimpleNamespace(id="Qwen/Qwen-Image-2.1"), transformer, embeds, prompts, tmp_path, status,
                        resolution=1000, steps=20, device="cpu", dtype=torch.float32)
    assert previews.ready
    rows0 = previews.render(0)
    rows250 = previews.render(250)
    final = previews.render(300, final=True)
    assert transformer.training  # back to training mode
    assert len(rows0) == len(rows250) == 3 and (tmp_path / "samples/step_000250/sample_02.png").exists()
    assert final[0]["path"] == "samples/sample_00.png" and (tmp_path / "samples/sample_00.png").exists()
    assert _FakeImagePipe.calls[0] == {"shape": (1, 5, 8), "size": (992, 992), "steps": 20}
    st = read_status(tmp_path)
    assert st["last_sample_step"] == 300 and [s["step"] for s in st["samples"]] == [0, 250, 300]
    assert json.loads((tmp_path / "samples/index.json").read_text())["prompts"] == prompts


def test_a_failing_preview_turns_previews_off_but_not_training(tmp_path):
    import torch

    from homebrew_ai.train.image import Previews

    status = StatusWriter(tmp_path)
    transformer = _FakeTransformer()
    _FakeImagePipe.fail_on_call = True
    try:
        previews = Previews(_FakeImagePipe, types.SimpleNamespace(id="m"), transformer, {"p": (torch.zeros(2, 4), None)}, ["p"], tmp_path, status,
                            resolution=512, steps=10, device="cpu", dtype=torch.float32)
        assert previews.render(0) == [] and not previews.ready and transformer.training
        assert "out of memory" in read_status(tmp_path)["preview_error"]
    finally:
        _FakeImagePipe.fail_on_call = False


def test_watching_an_image_job_downloads_and_shows_new_previews(project, fake_ui):
    run_dir = project.runs_dir / "img1"
    (run_dir / "samples/step_000250").mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "completed", "step": 300, "max_steps": 300, "last_sample_step": 250}))
    project.state.jobs.append({"job_id": "img1", "run_dir": str(run_dir), "modality": "image", "pid": 0, "target_spec": {"kind": "local"}, "state": "training"})
    project.save()
    out = run(make_ctx(project, fake_ui), "watch_training")
    assert out["last_sample_step"] == 250 and out["preview_dir"].endswith("samples/step_000250")

    ui = _console_ui()
    st = {"state": "completed", "step": 300, "max_steps": 300, "last_sample_step": 0, "preview_dir": "/x/samples/step_000000"}
    ui.watch(lambda: st, interval=0.01)
    assert "previews (before training): /x/samples/step_000000" in ui.console.file.getvalue()


def test_image_lora_can_be_exported_from_a_checkpoint(tmp_path, monkeypatch):
    from homebrew_ai.train import export

    monkeypatch.setattr(export, "copy_license_files", lambda *a, **k: [])
    info = get_model("Qwen/Qwen-Image-2.1")
    job, _ = build_job(model=info, project="t", method="lora", train_path="data/train.jsonl", eval_path=None, num_train=12, vram_gb=96,
                       overrides={"trigger_word": "sks"})
    job.save(tmp_path / "job.yaml")
    for folder, payload in (("final", "final"), ("checkpoints/checkpoint-250", "step250")):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "pytorch_lora_weights.safetensors").write_text(payload)
    (tmp_path / "samples/step_000250").mkdir(parents=True)
    (tmp_path / "samples/step_000250/sample_00.png").write_text("png@250")
    (tmp_path / "samples/sample_00.png").write_text("png@final")
    out = export.export_run(tmp_path, tmp_path / "out", checkpoint=250)
    assert (tmp_path / "out/pytorch_lora_weights.safetensors").read_text() == "step250"
    assert (tmp_path / "out/samples/sample_00.png").read_text() == "png@250"
    assert json.loads((tmp_path / "out/homebrew.json").read_text())["checkpoint_step"] == 250
    with pytest.raises(FileNotFoundError, match="kept checkpoints: 250"):
        export.export_run(tmp_path, tmp_path / "out2", checkpoint=500)


def test_private_upload_never_lands_in_an_existing_public_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from homebrew_ai.train.export import push_folder

    uploads = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def repo_info(self, repo_id, repo_type):
            return types.SimpleNamespace(private=repo_id.endswith("secret"))

        def upload_folder(self, **kw):
            uploads.append(kw["repo_id"])

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(RuntimeError, match="PUBLIC"):
        push_folder(tmp_path, "me/already-public", private=True)
    assert uploads == []
    assert push_folder(tmp_path, "me/secret", private=True).endswith("me/secret") and uploads == ["me/secret"]
    assert push_folder(tmp_path, "me/already-public", private=False) and uploads[-1] == "me/already-public"


@pytest.mark.parametrize("base_url,expected", [
    ("https://openrouter.ai/api/v1", {"extra_body": {"reasoning": {"effort": "low"}}}),
    (None, {"reasoning_effort": "low"}),
    ("http://localhost:11434/v1", {}),
])
def test_synthetic_data_can_ask_for_less_reasoning(base_url, expected, monkeypatch):
    from homebrew_ai.backends.openai_backend import OpenAICompatBackend as OpenAIBackend

    b = OpenAIBackend("xiaomi/mimo-v2.6-pro", api_key="x", base_url=base_url)
    seen = {}
    monkeypatch.setattr(b, "_stream_text", lambda req, on_progress: seen.update(req) or "{}")
    b.complete("sys", "prompt", effort="low")
    assert {k: seen[k] for k in ("extra_body", "reasoning_effort") if k in seen} == expected
    seen.clear()
    b.complete("sys", "prompt", effort="off")
    off = {"https://openrouter.ai/api/v1": {"extra_body": {"reasoning": {"enabled": False}}}, None: {"reasoning_effort": "minimal"}}.get(base_url, {})
    assert {k: seen[k] for k in ("extra_body", "reasoning_effort") if k in seen} == off


def test_rewrites_default_to_no_thinking(project, fake_ui):
    from homebrew_ai.agent.tools.base import ToolContext
    from homebrew_ai.settings import Settings

    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer {i}"}]} for i in range(4)]
    run(make_ctx(project, fake_ui), "add_examples", name="plain", records=rows)
    writer = _SlowWriter()
    ctx = ToolContext(project=project, ui=fake_ui, settings=Settings(), backend=writer, jobs=JobManager(project))
    run(ctx, "generate_synthetic_data", name="loud", mode="transform", source_dataset="plain", brief="shout", count=4, parallel=2)
    assert writer.efforts == ["off"]


def test_questions_pause_the_spinner(monkeypatch):
    ui = _console_ui()
    events = []

    class FakeStatus:
        def stop(self):
            events.append("stop")

        def start(self):
            events.append("start")

        def update(self, *a, **k):
            pass

        console = ui.console

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ui.console, "status", lambda *a, **k: FakeStatus())
    monkeypatch.setattr("builtins.input", lambda prompt="": events.append(("asked", prompt)) or "y")
    ui._simple = True
    with ui.activity("Connecting to the server"):
        assert ui.confirm("Connect to the GPU server 1.2.3.4?") is True
        with ui.activity("nested"):  # a tool's own activity inside the tool spinner: no second live display
            pass
    assert events == ["stop", ("asked", "Connect to the GPU server 1.2.3.4? [y/N] "), "start"]


def test_image_import_keeps_original_files_and_reports_progress(tmp_path, monkeypatch):
    import datasets
    from PIL import Image

    from homebrew_ai.data import images

    def encoded(fmt, color):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(buf, format=fmt)
        return buf.getvalue()

    rows = [
        {"image": {"bytes": encoded("JPEG", "red"), "path": "a.jpg"}, "text": "y2k digicam photo of a mall"},
        {"image": {"bytes": encoded("PNG", "blue"), "path": "b.png"}, "text": "frosted glass phone"},
        {"image": {"bytes": encoded("GIF", "green"), "path": "c.gif"}, "text": "animated sparkle"},
        {"image": {"bytes": encoded("JPEG", "red"), "path": "dup.jpg"}, "text": "same picture again"},
        {"image": {"bytes": b"not an image", "path": "x.jpg"}, "text": "broken"},
    ]

    class FakeStream:
        def cast_column(self, column, feature):
            assert column == "image" and feature.decode is False
            return self

        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            return iter(rows)

    calls = {}
    monkeypatch.setattr(datasets, "load_dataset", lambda ds_id, name=None, split=None, streaming=None, token=None: calls.update(name=name, split=split) or FakeStream())
    said = []
    report = images.import_image_hub(tmp_path, "y2k", "someone/y2k", image_column="image", caption_column="text", config="cameraphone",
                                     max_rows=10, on_progress=said.append)
    assert calls == {"name": "cameraphone", "split": "train"}
    files = sorted(p.suffix for p in (tmp_path / "y2k" / "images").iterdir())
    assert files == [".jpg", ".png", ".png"]  # JPEG/PNG kept as they were, GIF converted once
    jpg = next((tmp_path / "y2k" / "images").glob("*.jpg"))
    assert jpg.read_bytes() == rows[0]["image"]["bytes"]  # byte-identical, not re-encoded
    assert report["records"] == 3 and said[-1] == "3/10 images"


def test_card_usage_code_points_at_the_final_repo(tmp_path):
    from homebrew_ai.train.export import fix_card_repo

    (tmp_path / "README.md").write_text('model = PeftModel.from_pretrained(base, "Grandmas-Kitchen")\ntok = AutoTokenizer.from_pretrained("Grandmas-Kitchen")\n')
    assert fix_card_repo(tmp_path, "Grandmas-Kitchen", "empero-ai/Grandmas-Kitchen")
    assert (tmp_path / "README.md").read_text().count('"empero-ai/Grandmas-Kitchen"') == 2
    assert not fix_card_repo(tmp_path, "empero-ai/Grandmas-Kitchen", "empero-ai/Grandmas-Kitchen")
