"""The cpuforge (CPU) training backend: config mapping, support checks, and real runs through the worker.

The end-to-end tests need the ``cpuforge`` binary (``CPUFORGE_BIN`` or on PATH) and skip without it.
They use a tiny local tokenizer and a random byte-level model, so no download is involved.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time

import pytest

from brewery_ai.etf.io import write_records
from brewery_ai.jobs.manager import JobManager
from brewery_ai.train import cpuforge as cf
from brewery_ai.train.config import TrainJob, build_job, validate_job

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

BINARY = os.environ.get("CPUFORGE_BIN") or shutil.which("cpuforge")
needs_binary = pytest.mark.skipif(not BINARY, reason="cpuforge binary not available")

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    """A character-level tokenizer (vocab < 256) with Qwen-style chat markers."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["<unk>"] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]})
    fast.chat_template = TEMPLATE
    path = tmp_path_factory.mktemp("tok") / "tok"
    fast.save_pretrained(path)
    return path


@pytest.fixture
def model(tok_dir):
    """A ResolvedModel with the real qwen3 profile (chat format, LoRA presets) whose id is the local tokenizer folder."""
    from brewery_ai.models.registry import from_snapshot, get_model, snapshot

    snap = snapshot(get_model("Qwen/Qwen3-0.6B"))
    snap["variants"][0].update(id=str(tok_dir), label="tiny", params_b=0.001)
    return from_snapshot(snap)


@pytest.fixture
def byte_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("cfm") / "base"
    subprocess.run([BINARY, "init", "--out", str(out), "--preset", "tiny"], check=True, capture_output=True)
    return out


def make_job(model, *, method="lora", num_train=40, **over):
    over = {"max_seq_len": 96, "effective_batch": 4, "micro_batch_size": 2, "optimizer": "adamw_torch", "epochs": 1, "backend": "cpuforge", **over}
    job, report = build_job(model=model, project="t", method=method, train_path="data/train.jsonl", eval_path="data/eval.jsonl", num_train=num_train, num_eval=4, bf16=False, overrides=over, expert_override=True)
    assert not report["errors"], report
    return job


def records(n):
    return [{"messages": [{"role": "user", "content": f"say hi {i % 5}"}, {"role": "assistant", "content": "hello there, friend."}]} for i in range(n)]


# --------------------------------------------------------------------------- #
# config mapping and validation (no binary needed)
# --------------------------------------------------------------------------- #


def test_backend_defaults_to_torch_and_old_job_files_still_load():
    old = "job_id: j\nproject: p\nbase_model: m\nfamily: f\nmethod: lora\ndata: {train: a}\noptim: {learning_rate: 0.0001}\nlora: {rank: 8}\nruntime: {seed: 1}\n"
    import yaml

    job = TrainJob.model_validate(yaml.safe_load(old))
    assert job.runtime.backend == "torch"


def test_build_job_accepts_the_backend_override(model):
    assert make_job(model).runtime.backend == "cpuforge"
    job, report = build_job(model=model, project="t", method="lora", train_path="d", eval_path=None, num_train=40, bf16=False, overrides={"max_seq_len": 96, "optimizer": "adamw_torch"}, expert_override=True)
    assert job.runtime.backend == "torch"


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda j: setattr(j, "objective", "dpo"), "DPO"),
        (lambda j: setattr(j, "method", "qlora"), "QLoRA"),
        (lambda j: setattr(j.lora, "use_dora", True), "DoRA"),
        (lambda j: setattr(j.lora, "dropout", 0.1), "dropout"),
        (lambda j: setattr(j.runtime, "num_gpus", 2), "num_gpus"),
        (lambda j: setattr(j.optim, "optimizer", "adafactor"), "optimizer"),
        (lambda j: setattr(j, "init_from", {"job_id": "x"}), "init_from"),
    ],
)
def test_unsupported_settings_are_refused_with_a_reason(model, change, fragment):
    job = make_job(model)
    change(job)
    if isinstance(job.init_from, dict):
        from brewery_ai.train.config import InitFrom

        job.init_from = InitFrom(**job.init_from)
    why = cf.unsupported(job)
    assert any(fragment in w for w in why), why
    errors, _ = validate_job(job, model)
    assert any(fragment in e for e in errors), errors


def test_a_plain_lora_or_full_job_is_supported(model):
    assert cf.unsupported(make_job(model)) == []
    assert cf.unsupported(make_job(model, method="full")) == []
    assert cf.unsupported(make_job(model, use_rslora=True)) == []


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (Python 3.11+)")
def test_worker_config_matches_the_job(model):
    job = make_job(model, learning_rate=3e-4, lora_rank=8, lora_alpha=16, warmup_ratio=0.1, weight_decay=0.02, max_grad_norm=0.5)
    text, notes = cf.build_config(job, model, steps=50, model_dir=None)
    c = tomllib.loads(text)
    assert c["model"] == {"hf": job.base_model}
    assert c["train"]["steps"] == 50 and c["train"]["batch"] == 4
    assert math.isclose(c["train"]["lr"], 3e-4) and c["train"]["warmup_steps"] == 5
    assert math.isclose(c["train"]["weight_decay"], 0.02) and math.isclose(c["train"]["clip"], 0.5)
    assert c["train"]["memory"] == "full"  # short sequences: keep activations (faster)
    long = tomllib.loads(cf.build_config(make_job(model, max_seq_len=2048), model, steps=5)[0])
    assert long["train"]["memory"] == "checkpoint"  # long sequences: recompute to save RAM
    off = make_job(model, max_seq_len=2048, gradient_checkpointing=False)
    assert tomllib.loads(cf.build_config(off, model, steps=5)[0])["train"]["memory"] == "full"
    assert c["lora"]["rank"] == 8 and c["lora"]["alpha"] == 16
    assert c["lora"]["targets"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert c["data"] == {"train": cf.TRAIN_TOK, "eval": cf.EVAL_TOK, "seq_len": 96}
    assert any("fp32" in n for n in notes) or job.runtime.precision == "auto"

    full = tomllib.loads(cf.build_config(make_job(model, method="full"), model, steps=5, model_dir="/models/x")[0])
    assert "lora" not in full and full["model"] == {"path": "/models/x"}

    rs = tomllib.loads(cf.build_config(make_job(model, use_rslora=True, lora_rank=16, lora_alpha=32), model, steps=5)[0])
    assert math.isclose(rs["lora"]["alpha"], 32 * 4)  # alpha/sqrt(16) == (alpha*4)/16


@pytest.mark.skipif(tomllib is None, reason="needs tomllib (Python 3.11+)")
def test_awkward_strings_survive_into_toml(model):
    job = make_job(model)
    job.job_id = 'a "quoted" \\ id\nnewline'
    c = tomllib.loads(cf.build_config(job, model, steps=3)[0])
    assert c["run"]["job_id"] == job.job_id


def test_missing_binary_gives_an_actionable_error(monkeypatch):
    monkeypatch.delenv(cf.BINARY_ENV, raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError, match="CPUFORGE_BIN"):
        cf.find_binary()


# --------------------------------------------------------------------------- #
# real runs
# --------------------------------------------------------------------------- #


def launch(project, model, byte_model, tmp_path, monkeypatch, job, n=40):
    monkeypatch.setenv(cf.BINARY_ENV, BINARY)
    monkeypatch.setenv(cf.MODEL_DIR_ENV, str(byte_model))
    data = tmp_path / "train.jsonl"
    write_records(data, records(n))
    evalf = tmp_path / "eval.jsonl"
    write_records(evalf, records(4))
    project.state.compute.kind = "local"
    project.save()
    jm = JobManager(project)
    jm.stage(job, data, evalf)
    return jm, jm.launch(job)


def wait(jm, job_id, states=("completed", "failed", "crashed", "stopped"), timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = jm.status(job_id)
        if st["state"] in states:
            return st
        time.sleep(0.5)
    raise AssertionError(f"timed out: {st}")


@needs_binary
def test_lora_job_runs_end_to_end_through_brewery_train(project, model, byte_model, tmp_path, monkeypatch):
    job = make_job(model, learning_rate=5e-3, epochs=3)
    jm, rec = launch(project, model, byte_model, tmp_path, monkeypatch, job)
    st = wait(jm, job.job_id)
    assert st["state"] == "completed", st.get("log_tail") or st
    run = project.runs_dir / job.job_id

    # what Brewery's export/test/UI code expects to find
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        assert (run / "final" / name).exists(), name
    results = json.loads((run / "results.json").read_text())
    assert results["engine"] == "cpuforge" and results["stopped_early"] is False and results["data"]["tokens"] > 0
    final = json.loads((run / "status.json").read_text())
    assert final["state"] == "completed" and final["engine"] == "cpuforge"
    assert final["num_train"] == 40 and final["max_steps"] == final["step"] > 1
    assert final["trainable_params"] < final["total_params"]
    losses = [r["loss"] for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines()) if "loss" in r]
    assert len(losses) == final["step"] and losses[-1] < losses[0]

    # the data really is masked: prompt tokens are -100, answer tokens are learned
    first = json.loads((run / cf.TRAIN_TOK).read_text().splitlines()[0])
    assert -100 in first["labels"] and any(t != -100 for t in first["labels"]) and len(first["labels"]) == len(first["input_ids"])
    assert max(first["input_ids"]) < 256


@needs_binary
def test_full_finetune_writes_a_model_folder(project, model, byte_model, tmp_path, monkeypatch):
    job = make_job(model, method="full", num_train=16, learning_rate=1e-3)
    jm, _ = launch(project, model, byte_model, tmp_path, monkeypatch, job, n=16)
    st = wait(jm, job.job_id)
    assert st["state"] == "completed", st
    final = project.runs_dir / job.job_id / "final"
    assert (final / "model.safetensors").exists() and (final / "config.json").exists() and not (final / "adapter_config.json").exists()


@needs_binary
def test_stop_pauses_cleanly_and_resume_completes(project, model, byte_model, tmp_path, monkeypatch):
    job = make_job(model, num_train=400, epochs=6, learning_rate=2e-3)
    jm, _ = launch(project, model, byte_model, tmp_path, monkeypatch, job, n=400)
    run = project.runs_dir / job.job_id
    deadline = time.time() + 120
    while time.time() < deadline:
        s = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        if s.get("state") == "training" and s.get("step", 0) >= 2:
            break
        time.sleep(0.3)
    else:
        raise AssertionError(f"never started training: {s}")
    jm.stop(job.job_id)
    st = wait(jm, job.job_id, states=("stopped", "completed", "failed", "crashed"))
    assert st["state"] == "stopped", st
    assert (run / "state.safetensors").exists() and not (run / "final").exists()

    # resume (what `brewery train --resume` does): finish a few steps later, from the saved state
    stopped_at = json.loads((run / "status.json").read_text())["step"]
    j = TrainJob.load(run / "job.yaml")
    j.optim.max_steps = stopped_at + 3
    j.save(run / "job.yaml")
    from brewery_ai.train.runner import run as run_job

    assert run_job(run / "job.yaml", resume=True) == 0
    done = json.loads((run / "status.json").read_text())
    assert (done["state"], done["step"]) == ("completed", stopped_at + 3)
    assert (run / "final" / "adapter_model.safetensors").exists() and (run / "final" / "tokenizer_config.json").exists()
    assert len(done["loss_history"]) >= stopped_at  # the curve survived the restart


@needs_binary
def test_failures_reach_status_json_with_the_workers_hint(project, model, byte_model, tmp_path, monkeypatch):
    job = make_job(model)
    monkeypatch.setenv(cf.MODEL_DIR_ENV, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv(cf.BINARY_ENV, BINARY)
    data = tmp_path / "train.jsonl"
    write_records(data, records(8))
    project.state.compute.kind = "local"
    project.save()
    jm = JobManager(project)
    jm.stage(job, data, None)
    jm.launch(job)
    st = wait(jm, job.job_id)
    assert st["state"] == "failed"
    assert "config.json" in json.loads((project.runs_dir / job.job_id / "status.json").read_text())["error"]


def test_status_write_survives_a_reader_holding_the_file(tmp_path, monkeypatch):
    """Windows refuses os.replace while another process has status.json open; the writer must retry, not crash."""
    from brewery_ai.train import status as st

    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(st.os, "replace", flaky)
    monkeypatch.setattr(st.time, "sleep", lambda s: None)
    w = st.StatusWriter(tmp_path)
    w.set(state="preparing")
    w.fail("boom", "hint", "data")
    assert calls["n"] > 3
    assert st.read_status(tmp_path)["state"] == "failed"


def test_unsupported_job_fails_before_touching_the_binary(project, model, tmp_path, monkeypatch):
    monkeypatch.setenv(cf.BINARY_ENV, "/nonexistent/cpuforge")
    job = make_job(model)
    job.objective = "dpo"
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    job.save(run / "job.yaml")
    from brewery_ai.train.runner import run as run_job
    from brewery_ai.train.status import read_status

    assert run_job(run / "job.yaml") == 1
    st = read_status(run)
    assert st["state"] == "failed" and "DPO" in st["error"]
