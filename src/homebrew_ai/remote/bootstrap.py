"""Preparing a machine to run Homebrew training jobs ("the worker").

On a rented GPU box Homebrew keeps everything under one root folder
(``/workspace/homebrew`` when a persistent volume exists, else
``~/homebrew``)::

    <root>/pkg/homebrew_ai   the worker code, uploaded from this laptop
    <root>/venv              a virtualenv that reuses the image's PyTorch
    <root>/hf-cache          Hugging Face downloads (HF_HOME)
    <root>/runs/<job_id>     one folder per training run
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from importlib import resources
from pathlib import Path
from typing import Any, Callable

import homebrew_ai
from homebrew_ai.remote.target import LocalTarget, SSHTarget

BASE_REQUIREMENTS = [
    "transformers>=5.10,<6",
    "peft>=0.21",
    "accelerate>=1.10",
    "safetensors>=0.4",
    "pyyaml>=6",
    "pydantic>=2.6",
    "huggingface_hub>=1.0,<3",
    "jinja2>=3.1",
]
QLORA_REQUIREMENTS = ["bitsandbytes>=0.46.1"]


def _req_name(req: str) -> str:
    return re.split(r"[<>=!~ @\[;]", req.strip(), maxsplit=1)[0].lower()


def _lower_bound(spec: str) -> tuple[int, ...]:
    m = re.search(r">=\s*([0-9.]+)", spec or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def worker_requirements(model_info: Any, method: str = "qlora", optimizers: Any = ()) -> list[str]:
    """pip requirements for the worker; family packages replace same-named defaults.

    bitsandbytes is needed for QLoRA and for every 8-bit optimizer (the default for full fine-tuning
    and image LoRAs).
    """
    reqs = list(BASE_REQUIREMENTS)
    if method == "qlora" or any("8bit" in (o or "") or "bnb" in (o or "") for o in optimizers):
        reqs += QLORA_REQUIREMENTS
    if model_info is None:
        return reqs
    wanted_tf = model_info.min_transformers
    if wanted_tf and _lower_bound(wanted_tf) > _lower_bound(reqs[0]):
        reqs[0] = f"transformers{wanted_tf},<6"
    for pkg in model_info.profile.requirements.get("packages", []):
        name = _req_name(pkg)
        reqs = [r for r in reqs if _req_name(r) != name]
        reqs.append(pkg)
    return reqs


def probe_source() -> str:
    return resources.files("homebrew_ai.hardware").joinpath("probe.py").read_text(encoding="utf-8")


def remote_root(target: SSHTarget) -> str:
    res = target.run('if [ -d /workspace ] && [ -w /workspace ]; then echo /workspace/homebrew; else echo "$HOME/homebrew"; fi', timeout=60)
    res.raise_for_error("finding a work folder")
    return res.stdout.strip().splitlines()[-1]


def probe_remote(target: SSHTarget, with_torch: bool = True, path: str = ".", python: str = "python3") -> dict[str, Any]:
    flags = " --torch" if with_torch else ""
    res = target.run(f"{python} - --path {shlex.quote(path)}{flags}", input=probe_source(), timeout=180)
    res.raise_for_error("hardware probe")
    return json.loads(res.stdout.strip().splitlines()[-1])


def package_dir() -> Path:
    return Path(homebrew_ai.__file__).resolve().parent


def package_hash() -> str:
    """Fingerprint of the worker code, so each version gets its own folder on the server."""
    h = hashlib.sha256()
    base = package_dir()
    for path in sorted(base.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            h.update(str(path.relative_to(base)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()[:12]


def sync_worker(target: SSHTarget, root: str) -> str:
    """Make sure this exact worker version is on the server; returns the folder for PYTHONPATH.

    Versions live side by side (``<root>/pkg-<hash>``), so a job that is still running keeps its own code.
    """
    pkg = f"{root}/pkg-{package_hash()}"
    check = target.run(f"test -f {shlex.quote(pkg)}/homebrew_ai/__init__.py && echo present", timeout=60)
    if "present" not in check.stdout:
        partial = pkg + ".partial"
        target.run(f"rm -rf {shlex.quote(partial)}", timeout=60)
        target.put(package_dir(), f"{partial}/homebrew_ai")
        target.run(f"rm -rf {shlex.quote(pkg)} && mv {shlex.quote(partial)} {shlex.quote(pkg)}", timeout=60).raise_for_error("installing the worker code")
    return pkg


def worker_prefix(root: str, pkg: str | None = None) -> str:
    """Shell prefix that runs ``python -m homebrew_ai`` inside the worker environment."""
    return (
        f"export HF_HOME={shlex.quote(root + '/hf-cache')} PYTHONPATH={shlex.quote(pkg or root + '/pkg')} "
        f"PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1; {shlex.quote(root + '/venv/bin/python')} -m homebrew_ai"
    )


def _torch_index(cuda_driver: str | None) -> str | None:
    try:
        major, minor = (int(x) for x in (cuda_driver or "0.0").split(".")[:2])
    except ValueError:
        return None
    if (major, minor) >= (13, 0):
        return None  # default PyPI wheels (CUDA 13)
    if (major, minor) >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    return "https://download.pytorch.org/whl/cu126"


def prepare_remote(target: SSHTarget, requirements: list[str], say: Callable[[str], None] = print) -> dict[str, Any]:
    """Make an SSH machine ready for training. Safe to run again (idempotent)."""
    say(f"Connecting to {target.label} ...")
    target.test().raise_for_error("SSH connection")
    root = remote_root(target)
    target.run(f"mkdir -p {shlex.quote(root)}/runs {shlex.quote(root)}/hf-cache {shlex.quote(root)}/pkg && touch ~/.no_auto_tmux", timeout=60).raise_for_error("creating folders")
    say("Checking the hardware ...")
    hw = probe_remote(target, with_torch=True, path=root)
    py = hw.get("python", "0")
    if tuple(int(x) for x in py.split(".")[:2]) < (3, 10):
        raise RuntimeError(f"the server has Python {py}; Homebrew needs 3.10 or newer (pick a newer PyTorch template)")
    if not hw.get("gpus"):
        say("Warning: no GPU detected on the server.")

    say("Uploading the Homebrew worker ...")
    target.run(f"rm -rf {shlex.quote(root)}/pkg/homebrew_ai", timeout=60)
    target.put(package_dir(), f"{root}/pkg/homebrew_ai")

    venv_py = f"{root}/venv/bin/python"
    base_py = (hw.get("torch") or {}).get("python") or "python3"  # reuse the image's PyTorch (venv/conda) when present
    res = target.run(f"test -x {shlex.quote(venv_py)} || {shlex.quote(base_py)} -m venv --system-site-packages {shlex.quote(root)}/venv", timeout=300)
    if not res.ok:
        raise RuntimeError("could not create a Python virtualenv on the server: " + (res.stderr or res.stdout)[-400:])

    torch = hw.get("torch") or {}
    if not torch.get("installed") or (hw.get("gpus") and not torch.get("cuda_works", torch.get("cuda_available"))):
        index = _torch_index(hw.get("cuda_driver_version"))
        say("Installing PyTorch (this can take a few minutes) ...")
        extra = f" --index-url {index}" if index else ""
        res = target.run(f"{shlex.quote(venv_py)} -m pip install -q torch{extra}", timeout=1800)
        res.raise_for_error("installing PyTorch")

    say("Installing training libraries ...")
    pkgs = " ".join(shlex.quote(r) for r in requirements)
    res = target.run(f"{shlex.quote(venv_py)} -m pip install -q --upgrade pip >/dev/null 2>&1; {shlex.quote(venv_py)} -m pip install -q {pkgs}", timeout=2400)
    res.raise_for_error("installing training libraries")

    say("Verifying ...")
    check = (
        "import json, torch, transformers, peft, homebrew_ai;"
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpus': torch.cuda.device_count(),"
        "'transformers': transformers.__version__, 'peft': peft.__version__, 'homebrew': homebrew_ai.__version__}))"
    )
    res = target.run(f"PYTHONPATH={shlex.quote(root + '/pkg')} {shlex.quote(venv_py)} -c {shlex.quote(check)}", timeout=300)
    res.raise_for_error("verifying the worker")
    versions = json.loads(res.stdout.strip().splitlines()[-1])
    return {"root": root, "hardware": hw, "versions": versions, "requirements": requirements}


def missing_requirements(installed: list[str], needed: list[str]) -> list[str]:
    """Requirements in ``needed`` that a machine prepared with ``installed`` may not satisfy."""
    have = {r.replace(" ", "") for r in installed}
    return [r for r in needed if r.replace(" ", "") not in have]


def check_local_worker(requirements: list[str]) -> dict[str, Any]:
    """Which training packages are missing on this machine?"""
    import importlib.util

    names = {
        "transformers": "transformers", "peft": "peft", "accelerate": "accelerate", "torch": "torch", "bitsandbytes": "bitsandbytes",
        "diffusers": "diffusers", "flash-linear-attention": "fla", "pillow": "PIL",
    }
    wanted = ["torch"] + [_req_name(r) for r in requirements]
    missing = sorted({w for w in wanted if w in names and importlib.util.find_spec(names[w]) is None})
    return {"missing": missing, "install_command": f"pip install 'homebrew-ai[train]'" + (" diffusers pillow" if "diffusers" in missing else "")}


def local_target() -> LocalTarget:
    return LocalTarget()
