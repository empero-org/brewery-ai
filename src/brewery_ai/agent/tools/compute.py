"""Where training runs: this computer, or a GPU server over SSH."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any

from brewery_ai.agent.tools.base import ToolContext, ToolError, tool
from brewery_ai.hardware import probe as hw_probe
from brewery_ai.hardware.estimate import TrainShape, estimate_hours, estimate_memory, fitting_gpus
from brewery_ai.hardware.gpus import BY_KEY, PRICES_CHECKED, generic, identify
from brewery_ai.models.registry import get_model
from brewery_ai.remote import bootstrap
from brewery_ai.remote.guides import free_options, rental_guide
from brewery_ai.remote.sshutil import DEFAULT_KEY, SSHParseError, ensure_key, parse_ssh_command
from brewery_ai.remote.target import SSHTarget


def summarize_hardware(report: dict[str, Any]) -> dict[str, Any]:
    gpus = []
    for g in report.get("gpus", []):
        known = identify(g.get("name", ""))
        gpus.append(
            {
                "name": g.get("name"),
                "vendor": g.get("vendor"),
                "backend": g.get("backend"),
                "vram_gb": g.get("vram_gb"),
                "free_gb": g.get("vram_free_gb"),
                "bf16": g.get("bf16", known.bf16 if known else None),
                "catalog": known.key if known else None,
            }
        )
    torch = report.get("torch") or {}
    out = {
        "os": f"{report.get('os')} {report.get('arch')}",
        "python": report.get("python"),
        "cpu": report.get("cpu", {}).get("model"),
        "ram_gb": report.get("ram_gb"),
        "disk_free_gb": (report.get("disk") or {}).get("free_gb"),
        "gpus": gpus,
        "cuda_driver": report.get("cuda_driver_version"),
        "apple_silicon": report.get("apple_silicon"),
        "container": report.get("container"),
    }
    if torch:
        out["torch"] = {k: torch.get(k) for k in ("installed", "version", "hip", "cuda_available", "cuda_works", "cuda_error", "bf16", "error") if k in torch}
    if not gpus:
        out["verdict"] = "no NVIDIA/AMD GPU found: training here is not practical (tiny test models only); rent a GPU server"
    elif error := hw_probe.gpu_runtime_error(report):
        out["verdict"] = f"GPU training not ready: {error}"
    else:
        best = max(g["vram_gb"] or 0 for g in gpus)
        out["verdict"] = f"usable GPU with {best} GB" if best >= 8 else f"GPU has only {best} GB: only very small models"
    return out


@tool(
    "detect_hardware",
    """Check the hardware of this computer (where='local') or of the connected SSH server (where='remote'): GPUs, VRAM,
RAM, disk, Python/PyTorch. Call this early to decide where training can run.""",
    {"where": {"type": "string", "enum": ["local", "remote"]}},
    ["where"],
    activity="Checking hardware",
)
def detect_hardware(ctx: ToolContext, args: dict[str, Any]) -> Any:
    if args["where"] == "local":
        report = hw_probe.probe(with_torch=True, path=str(ctx.project.root))
        ctx.cache["local_hardware"] = report
        return summarize_hardware(report)
    compute = ctx.project.state.compute
    if compute.kind != "ssh" or not compute.ssh:
        raise ToolError("no server connected yet; use connect_server first")
    target = SSHTarget(bootstrap_spec(compute.ssh))
    report = bootstrap.probe_remote(target, with_torch=True, path=compute.remote_root or "~")
    compute.hardware = report
    ctx.project.save()
    return summarize_hardware(report)


def bootstrap_spec(data: dict[str, Any]):
    from brewery_ai.remote.sshutil import SSHSpec

    return SSHSpec(host=data["host"], user=data.get("user", "root"), port=int(data.get("port", 22)), key=data.get("key"), options=list(data.get("options") or []), provider=data.get("provider"))


@tool(
    "use_local_computer",
    """Choose this computer for training. Checks the GPU and which training packages (PyTorch, transformers, peft, ...)
are missing. Use an NVIDIA (CUDA) or AMD (ROCm) GPU with 8 GB+, or tiny CPU test runs.""",
    activity="Checking this computer",
)
def use_local_computer(ctx: ToolContext, args: dict[str, Any]) -> Any:
    report = ctx.cache.get("local_hardware") or hw_probe.probe(with_torch=True, path=str(ctx.project.root))
    compute = ctx.project.state.compute
    compute.kind = "local"
    compute.ssh = None
    compute.hardware = report
    compute.prepared = False
    model_id = ctx.project.state.base_model
    reqs = bootstrap.BASE_REQUIREMENTS
    if model_id:
        info = get_model(model_id)
        reqs = bootstrap.worker_requirements(info, "lora")
    check = bootstrap.check_local_worker(reqs)
    gpu_error = hw_probe.gpu_runtime_error(report)
    compute.prepared = not check["missing"] and gpu_error is None
    ctx.project.save()
    return {"hardware": summarize_hardware(report), "missing_packages": check["missing"], "gpu_error": gpu_error,
            "install_command": check["install_command"] if check["missing"] else None}


@tool(
    "install_local_training_packages",
    "Install the missing training packages into Brewery's Python on this computer (asks the user first).",
    activity="Installing training packages",
)
def install_local_training_packages(ctx: ToolContext, args: dict[str, Any]) -> Any:
    model_id = ctx.project.state.base_model
    reqs = list(bootstrap.BASE_REQUIREMENTS)
    if model_id:
        reqs = bootstrap.worker_requirements(get_model(model_id), "lora")
    report = ctx.project.state.compute.hardware or {}
    pkgs = ["torch", *reqs] if not (report.get("torch") or {}).get("installed") else reqs
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    ctx.confirm_or_raise("Install the training packages on this computer?", details=" ".join(cmd) + "\n(PyTorch alone is a few GB.)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolError("pip install failed:\n" + "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:]))
    gpu_error = hw_probe.gpu_runtime_error(report)
    ctx.project.state.compute.prepared = gpu_error is None
    ctx.cache.pop("local_hardware", None)
    ctx.project.save()
    return {"installed": pkgs, "prepared": ctx.project.state.compute.prepared, "gpu_error": gpu_error}


def _shape_from(info, method: str | None, seq_len: int | None) -> TrainShape:
    method = method or ("lora" if info.guidelines.lora.allowed else "qlora")
    g = info.guidelines.for_method(method)
    if info.modality == "image":
        seq = int(seq_len or (g.resolution.default if g.resolution and g.resolution.default else 1024))
    else:
        seq = int(seq_len or 2048)
    return TrainShape(method=method, seq_len=seq, lora_rank=int(g.rank.default) if g.rank and g.rank.default else 16, optimizer=g.default_optimizer or "adamw_torch")


@tool(
    "estimate_requirements",
    """Estimate GPU memory, suitable rental GPUs with prices, and (if dataset size is known) training time and cost for a
base model + method. Use it to compare options before choosing a model, a method or a GPU.""",
    {
        "base_model": {"type": "string", "description": "Hugging Face id of a supported base model."},
        "method": {"type": "string", "enum": ["lora", "qlora", "full"]},
        "seq_len": {"type": "integer", "description": "Text: max tokens per example. Image: training resolution in pixels."},
        "train_tokens": {"type": "integer", "description": "Text: total tokens to process (examples × avg tokens × epochs). Image: steps × batch size."},
        "gpu": {"type": "string", "description": "Catalog key of a GPU to estimate time on (e.g. rtx4090, a100_80, h100)."},
        "provider": {"type": "string", "enum": ["runpod", "vast"]},
    },
    ["base_model"],
)
def estimate_requirements(ctx: ToolContext, args: dict[str, Any]) -> Any:
    info = get_model(args["base_model"])
    shape = _shape_from(info, args.get("method"), args.get("seq_len"))
    if not info.guidelines.for_method(shape.method).allowed:
        raise ToolError(f"{shape.method} is not allowed for {info.id}: " + " ".join(info.guidelines.for_method(shape.method).notes))
    mem = estimate_memory(info, shape)
    options = fitting_gpus(info, shape, args.get("provider"))
    out: dict[str, Any] = {
        "base_model": info.id,
        "method": shape.method,
        "seq_len_or_resolution": shape.seq_len,
        "memory": mem.to_dict(),
        "cheapest_fitting_gpus": options[:5],
        "prices_note": f"approximate on-demand $/h, checked {PRICES_CHECKED}",
    }
    local = ctx.project.state.compute.hardware
    if local and local.get("gpus"):
        vram = max(g.get("vram_gb") or 0 for g in local["gpus"])
        out["fits_current_compute"] = mem.fits(vram)
    tokens = args.get("train_tokens")
    if tokens:
        gpu = BY_KEY.get(args.get("gpu") or "") or (BY_KEY.get(options[0]["key"]) if options else None)
        if gpu is None and local and local.get("gpus"):
            g0 = local["gpus"][0]
            gpu = identify(g0.get("name", "")) or generic(g0.get("vram_gb") or 24, name=g0.get("name", "GPU"))
        if gpu is not None:
            lo, hi = estimate_hours(info, shape, float(tokens), gpu)
            price = gpu.price(args.get("provider"))
            out["time_estimate"] = {"gpu": gpu.name, "hours_low": round(lo, 2), "hours_high": round(hi, 2)}
            if price:
                out["time_estimate"]["cost_usd_low"] = round(lo * price, 2)
                out["time_estimate"]["cost_usd_high"] = round(hi * price, 2)
    return out


@tool(
    "gpu_rental_guide",
    """Step-by-step instructions for renting a GPU server on Runpod or Vast.ai, tailored to the chosen model, including
recommended GPUs with approximate prices and the SSH public key to paste. Creates Brewery's SSH key if needed (asks first).""",
    {"provider": {"type": "string", "enum": ["runpod", "vast"]}, "method": {"type": "string", "enum": ["lora", "qlora", "full"]}, "seq_len": {"type": "integer"}},
    ["provider"],
)
def gpu_rental_guide(ctx: ToolContext, args: dict[str, Any]) -> Any:
    public_key = None
    key_path = DEFAULT_KEY
    if not key_path.exists():
        if shutil.which("ssh-keygen") and ctx.ui.confirm(
            "Create an SSH key for Brewery (~/.ssh/brewery_ed25519)? It lets Brewery log in to your GPU server securely.", default=True
        ):
            key_path, public_key, _ = ensure_key(key_path)
    else:
        key_path, public_key, _ = ensure_key(key_path)
    options: list[dict[str, Any]] = []
    disk_gb = 60
    if ctx.project.state.base_model:
        info = get_model(ctx.project.state.base_model)
        shape = _shape_from(info, args.get("method"), args.get("seq_len"))
        options = fitting_gpus(info, shape, args["provider"])
        disk_gb = int(max(40, info.variant.params_b * 2 * 3 + 30 + (info.variant.text_encoder_params_b or 0) * 2))
    guide = rental_guide(args["provider"], disk_gb=disk_gb, public_key=public_key, gpu_options=options, level=ctx.level)
    if public_key:
        ctx.ui.info(public_key, title="Your Brewery SSH public key (safe to share; paste it into the provider)", style="key")
        guide["public_key"] = "shown to the user above"
        guide["key_file"] = str(key_path)
    else:
        guide["public_key"] = None
        guide["key_status"] = (
            "Brewery's SSH key was not created (the user declined or ssh-keygen is missing). Ask whether to create it "
            "(create_ssh_key) or to use an SSH key they already have (then include its -i path in the SSH command)."
        )
    guide["free_alternatives"] = free_options()
    ctx.project.state.compute.provider = args["provider"]
    ctx.project.save()
    return guide


@tool(
    "create_ssh_key",
    """Create (or show) Brewery's SSH key pair (~/.ssh/brewery_ed25519) and display the public key for the user to
paste into their GPU provider's SSH-key settings. Asks the user first if the key does not exist yet.""",
)
def create_ssh_key(ctx: ToolContext, args: dict[str, Any]) -> Any:
    if not DEFAULT_KEY.exists():
        if not shutil.which("ssh-keygen"):
            raise ToolError("ssh-keygen is not installed on this computer; the user needs OpenSSH (or can use an existing key)")
        ctx.confirm_or_raise("Create an SSH key for Brewery (~/.ssh/brewery_ed25519)? It lets Brewery log in to your GPU server securely.")
    key_path, public_key, created = ensure_key(DEFAULT_KEY)
    ctx.ui.info(public_key, title="Your Brewery SSH public key (safe to share; paste it into the provider)", style="key")
    return {"created": created, "key_file": str(key_path), "public_key": "shown to the user above", "next": "the user adds it to the provider BEFORE renting (Vast: cloud.vast.ai/manage-keys)"}


@tool(
    "connect_server",
    """Connect to a GPU server using the SSH command the provider shows (e.g. 'ssh root@1.2.3.4 -p 22022 -i ~/.ssh/key').
Tests the connection, checks the hardware and remembers the server for this project.""",
    {"ssh_command": {"type": "string"}},
    ["ssh_command"],
    activity="Connecting to the server",
)
def connect_server(ctx: ToolContext, args: dict[str, Any]) -> Any:
    try:
        spec = parse_ssh_command(args["ssh_command"])
    except SSHParseError as exc:
        raise ToolError(str(exc)) from exc
    details = [f"user {spec.user}, port {spec.port}"]
    if spec.key:
        details.append(f"key {spec.key}")
    if spec.options:
        details.append("options " + " ".join(spec.options))
    details.append("Brewery will copy its training code and your training data to this server.")
    ctx.confirm_or_raise(f"Connect to the GPU server {spec.host}?", details="\n".join(details))
    target = SSHTarget(spec)
    res = target.test()
    if not res.ok and spec.key is None and DEFAULT_KEY.exists() and "permission denied" in (res.stderr or "").lower():
        # the provider may only know Brewery's own key; ssh's defaults (agent, ~/.ssh/id_*) were tried first
        spec.key = str(DEFAULT_KEY)
        target = SSHTarget(spec)
        res = target.test()
    if not res.ok:
        hint = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["no output"]
        raise ToolError(
            f"could not log in to {spec.label}: {hint[0]}. Check that the server is running, the command is the 'direct/exposed TCP' one, "
            "and that your public key was added to the provider before the server started."
        )
    root = bootstrap.remote_root(target)
    report = bootstrap.probe_remote(target, with_torch=True, path=root)
    compute = ctx.project.state.compute
    compute.kind = "ssh"
    compute.ssh = spec.to_dict()
    compute.provider = spec.provider or compute.provider
    compute.remote_root = root
    compute.hardware = report
    compute.prepared = False
    ctx.project.save()
    moved = ctx.jobs.adopt_server(target, root)
    out = {"connected": spec.label, "work_folder": root, "hardware": summarize_hardware(report), "next": "run prepare_server to install the training environment"}
    if moved:
        out["existing_jobs_found_here"] = moved
    return out


@tool(
    "prepare_server",
    """Install Brewery's training environment on the connected server (uploads the worker, creates a virtualenv,
installs PyTorch/transformers/peft as needed). Takes a few minutes; asks the user first. Safe to run again.""",
)
def prepare_server(ctx: ToolContext, args: dict[str, Any]) -> Any:
    compute = ctx.project.state.compute
    if compute.kind != "ssh" or not compute.ssh:
        raise ToolError("no server connected; use connect_server first")
    from brewery_ai.agent.tools.training import _requirements_for

    info = get_model(ctx.project.state.base_model) if ctx.project.state.base_model else None
    reqs = _requirements_for(info, list(ctx.project.state.drafts.values()))
    ctx.confirm_or_raise(f"Set up the training environment on {compute.ssh['host']}?", details="Installs into " + (compute.remote_root or "~/brewery") + ": " + ", ".join(reqs))
    target = SSHTarget(bootstrap_spec(compute.ssh))
    lines: list[str] = []
    with ctx.ui.activity("Preparing the server") as act:
        def say(msg: str) -> None:
            lines.append(msg)
            if hasattr(act, "update"):
                act.update(msg)

        result = bootstrap.prepare_remote(target, reqs, say=say)
    compute.prepared = True
    compute.remote_root = result["root"]
    compute.hardware = result["hardware"]
    compute.requirements = reqs
    ctx.project.save()
    return {"ready": True, "versions": result["versions"], "steps": lines}
