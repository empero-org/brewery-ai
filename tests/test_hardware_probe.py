"""GPU discovery without vendor command-line tools, including native Windows ROCm."""

import contextlib
import io
import json
import subprocess
import sys
import types

import pytest

from homebrew_ai.hardware import probe


def rocm_device():
    return {
        "index": 0, "vendor": "amd", "backend": "rocm", "name": "AMD Radeon RX 7900 XT",
        "vram_gb": 20.0, "vram_free_gb": 18.0, "driver": None,
        "compute_capability": "gfx1100", "bf16": True,
    }


def without_smi(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_gpus", lambda: ([], None))
    monkeypatch.setattr(probe, "amd_gpus", lambda: [])


def test_rocm_gpu_reaches_local_compute_and_memory_planning(monkeypatch, project):
    from homebrew_ai.agent.tools.compute import summarize_hardware, use_local_computer
    from homebrew_ai.agent.tools.training import _compute_gpu
    from homebrew_ai.remote import bootstrap

    without_smi(monkeypatch)
    device = rocm_device()
    torch = {"installed": True, "hip": "7.2", "cuda_available": True, "cuda_works": True,
             "bf16": True, "devices": [device]}
    monkeypatch.setattr(probe, "find_torch_python", lambda: sys.executable)
    monkeypatch.setattr(probe, "torch_info", lambda **kw: torch.copy())
    monkeypatch.setattr(bootstrap, "check_local_worker", lambda reqs: {"missing": []})

    report = probe.probe(with_torch=True, path=project.root)
    assert report["gpus"] == [device]
    assert report["cuda_driver_version"] is None
    summary = summarize_hardware(report)
    assert summary["gpus"][0]["bf16"] is True
    assert summary["torch"]["hip"] == "7.2"
    assert summary["verdict"] == "usable GPU with 20.0 GB"

    ctx = types.SimpleNamespace(project=project, cache={"local_hardware": report})
    assert use_local_computer(ctx, {})["missing_packages"] == []
    assert project.state.compute.prepared
    vram, count, bf16, gpu = _compute_gpu(ctx)
    assert (vram, count, bf16, gpu.name) == (20.0, 1, True, device["name"])
    assert project.load(project.root).state.compute.hardware["gpus"] == [device]


def test_smi_data_is_preserved_when_torch_is_probed(monkeypatch, tmp_path):
    nvidia = {"name": "NVIDIA RTX 4090", "index": 0, "vendor": "nvidia", "vram_gb": 24.0,
              "vram_free_gb": 22.0, "driver": "580", "compute_capability": "8.9"}
    monkeypatch.setattr(probe, "nvidia_gpus", lambda: ([nvidia], "13.0"))
    monkeypatch.setattr(probe, "find_torch_python", lambda: sys.executable)
    monkeypatch.setattr(probe, "torch_info", lambda **kw: {"installed": True, "devices": [rocm_device()]})
    report = probe.probe(with_torch=True, path=tmp_path)
    assert report["gpus"] == [nvidia]
    assert report["cuda_driver_version"] == "13.0"


def test_cpu_torch_does_not_invent_a_gpu(monkeypatch, tmp_path):
    without_smi(monkeypatch)
    monkeypatch.setattr(probe, "find_torch_python", lambda: sys.executable)
    monkeypatch.setattr(probe, "torch_info", lambda **kw: {"installed": True, "cuda_available": False, "devices": []})
    assert probe.probe(with_torch=True, path=tmp_path)["gpus"] == []


def test_lightweight_probe_does_not_import_torch(monkeypatch, tmp_path):
    without_smi(monkeypatch)
    monkeypatch.setattr(probe, "find_torch_python", lambda: pytest.fail("unexpected PyTorch import"))
    assert "torch" not in probe.probe(path=tmp_path)


def torch_snapshot(monkeypatch, *, hip="7.2", bf16=True, capability=(11, 0), operation_error=None):
    synchronized = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_capability=lambda i: capability,
        get_device_properties=lambda i: types.SimpleNamespace(name="test GPU", total_memory=20 * 1024**3, gcnArchName="gfx1100"),
        mem_get_info=lambda i: (18 * 1024**3, 20 * 1024**3),
        device=lambda i: contextlib.nullcontext(), is_bf16_supported=lambda *, including_emulation: bf16,
        synchronize=lambda i=None: synchronized.append(i),
    )
    def zeros(*args, **kwargs):
        if operation_error:
            raise RuntimeError(operation_error)
        return object()

    fake_torch = types.SimpleNamespace(
        __version__="2.9", version=types.SimpleNamespace(cuda=None if hip else "12.8", hip=hip), cuda=cuda,
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        zeros=zeros,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def run(args, **kw):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(args[-1], {})
        return subprocess.CompletedProcess(args, 0, stdout.getvalue(), "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    return probe.torch_info(), synchronized


@pytest.mark.parametrize("hip,bf16,capability,vendor", [("7.2", False, (11, 0), "amd"), (None, True, (8, 6), "nvidia")])
def test_torch_snapshot_uses_backend_memory_and_bf16_api(monkeypatch, hip, bf16, capability, vendor):
    report, synchronized = torch_snapshot(monkeypatch, hip=hip, bf16=bf16, capability=capability)
    assert report["bf16"] is bf16  # HIP's numeric capability is not an NVIDIA architecture.
    assert report["cuda_works"] is True
    assert synchronized
    device = report["devices"][0]
    assert (device["vendor"], device["vram_gb"], device["vram_free_gb"], device["bf16"]) == (vendor, 20.0, 18.0, bf16)
    assert device["compute_capability"] == ("gfx1100" if hip else "8.6")


def test_failed_gpu_operation_blocks_local_training(monkeypatch, project):
    from homebrew_ai.agent.tools.base import ToolError
    from homebrew_ai.agent.tools.compute import summarize_hardware, use_local_computer
    from homebrew_ai.agent.tools.training import start_training
    from homebrew_ai.models.registry import get_model
    from homebrew_ai.remote import bootstrap
    from homebrew_ai.train.config import build_job

    snapshot, _ = torch_snapshot(monkeypatch, operation_error="HIP device operation failed")
    without_smi(monkeypatch)
    monkeypatch.setattr(probe, "find_torch_python", lambda: sys.executable)
    monkeypatch.setattr(probe, "torch_info", lambda **kw: snapshot)
    monkeypatch.setattr(bootstrap, "check_local_worker", lambda reqs: {"missing": []})
    report = probe.probe(with_torch=True, path=project.root)
    assert report["gpus"][0]["name"] == "test GPU"
    assert report["torch"]["cuda_works"] is False
    assert summarize_hardware(report)["verdict"] == "GPU training not ready: HIP device operation failed"

    ctx = types.SimpleNamespace(project=project, cache={"local_hardware": report})
    result = use_local_computer(ctx, {})
    assert result["missing_packages"] == []
    assert result["gpu_error"] == "HIP device operation failed"
    assert not project.load(project.root).state.compute.prepared

    model = get_model("Qwen/Qwen3-0.6B")
    job, config = build_job(model=model, project="blocked", method="lora", train_path="data/train.jsonl", eval_path=None, num_train=4)
    assert not config["errors"], config
    project.state.drafts[job.job_id] = job.model_dump(mode="json")
    project.state.compute.prepared = True  # A stale ready flag must not bypass the failed operation check.
    ctx.confirm_or_raise = lambda *a, **kw: pytest.fail("failed GPU requested training confirmation")
    ctx.jobs = types.SimpleNamespace(stage=lambda *a, **kw: pytest.fail("failed GPU staged training"))
    with pytest.raises(ToolError, match="HIP device operation failed"):
        start_training(ctx, {})
    assert not project.load(project.root).state.compute.prepared
    assert job.job_id in project.state.drafts
    assert not project.state.jobs


def test_gpu_operation_check_is_required_but_cpu_training_remains_available(monkeypatch, project):
    from homebrew_ai.agent.tools.compute import use_local_computer
    from homebrew_ai.remote import bootstrap

    monkeypatch.setattr(bootstrap, "check_local_worker", lambda reqs: {"missing": []})
    ctx = types.SimpleNamespace(project=project, cache={"local_hardware": {"gpus": [rocm_device()], "torch": {"installed": True}}})
    assert use_local_computer(ctx, {})["gpu_error"]
    assert not project.state.compute.prepared
    ctx.cache["local_hardware"] = {"gpus": [], "torch": {"installed": True, "cuda_available": False}}
    assert use_local_computer(ctx, {})["gpu_error"] is None
    assert project.load(project.root).state.compute.prepared


def test_package_install_does_not_override_failed_gpu_check(monkeypatch, project):
    from homebrew_ai.agent.tools.compute import install_local_training_packages

    project.state.compute.hardware = {"gpus": [rocm_device()], "torch": {"installed": True, "cuda_works": False,
                                                                                "cuda_error": "HIP device operation failed"}}
    ctx = types.SimpleNamespace(project=project, cache={"local_hardware": project.state.compute.hardware},
                                confirm_or_raise=lambda *a, **kw: None)
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    result = install_local_training_packages(ctx, {})
    assert not result["prepared"] and result["gpu_error"] == "HIP device operation failed"
    assert not project.load(project.root).state.compute.prepared
    assert "local_hardware" not in ctx.cache


def test_timeout_retains_completed_snapshot(monkeypatch):
    snapshot = {"installed": True, "devices": [rocm_device()], "cuda_available": True, "cuda_works": True}

    def run(args, **kw):
        raise subprocess.TimeoutExpired(args, kw["timeout"], output=(json.dumps(snapshot) + "\n").encode())

    monkeypatch.setattr(probe.subprocess, "run", run)
    report = probe.torch_info(timeout=1)
    assert report["installed"] is True
    assert report["devices"] == snapshot["devices"]
    assert "timed out" in report["error"]


def test_training_precision_does_not_treat_hip_capability_as_nvidia(monkeypatch):
    torch = pytest.importorskip("torch")
    from homebrew_ai.train.model import device_info

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (11, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda i: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *, including_emulation: False)
    assert device_info() == {"device": "cuda", "count": 1, "bf16": False}


@pytest.mark.slow
def test_detected_gpu_runs_homebrew_lora_training(tmp_path, tiny_model, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm GPU")

    from homebrew_ai.etf.io import write_records
    from homebrew_ai.train import model as tmodel
    from homebrew_ai.train.config import build_job
    from homebrew_ai.train.runner import run

    report = probe.probe(with_torch=True, path=tmp_path)
    assert report["gpus"] and report["torch"]["cuda_works"], report
    records = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}] * 4
    write_records(tmp_path / "data/train.jsonl", records)
    job, config = build_job(
        model=tiny_model, project="gpu-test", method="lora", train_path="data/train.jsonl", eval_path=None, num_train=4,
        vram_gb=report["gpus"][0]["vram_gb"], bf16=report["torch"]["bf16"],
        overrides={"max_seq_len": 64, "effective_batch": 2, "micro_batch_size": 2, "optimizer": "adamw_torch", "epochs": 1},
        expert_override=True,
    )
    assert not config["errors"], config
    job.runtime.save_steps = 10_000
    job_path = job.save(tmp_path / "job.yaml")
    original = tmodel.load_model
    devices = []

    def load(*args, **kwargs):
        model = original(*args, **kwargs)
        devices.append(str(next(model.parameters()).device))
        return model

    monkeypatch.setattr(tmodel, "load_model", load)
    result = run(job_path)
    torch.cuda.synchronize()
    assert result == 0, (tmp_path / "status.json").read_text()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "completed" and status["step"] >= 2
    assert devices == ["cuda:0"]  # PyTorch exposes ROCm GPUs through the cuda device API.
    assert (tmp_path / "final/adapter_model.safetensors").is_file()
