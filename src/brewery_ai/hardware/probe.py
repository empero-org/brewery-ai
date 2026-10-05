#!/usr/bin/env python3
"""Standalone hardware probe. Standard library only.

Brewery runs this file locally (imported) and on remote machines by piping
its source into ``python3 -`` over SSH, so it must never import anything
outside the standard library. It prints one JSON object.

Usage: python3 probe.py [--torch] [--path DIR]
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys


def _run(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def nvidia_gpus():
    gpus = []
    if not shutil.which("nvidia-smi"):
        return gpus, None
    fields = "index,name,memory.total,memory.used,driver_version,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    has_cc = out is not None
    if out is None:
        out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"])
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total = float(parts[2]) / 1024.0
            used = float(parts[3]) / 1024.0
        except ValueError:
            continue
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else len(gpus),
                "vendor": "nvidia",
                "name": parts[1],
                "vram_gb": round(total, 1),
                "vram_free_gb": round(max(total - used, 0.0), 1),
                "driver": parts[4],
                "compute_capability": parts[5] if has_cc and len(parts) > 5 else None,
            }
        )
    cuda = None
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if m:
            cuda = m.group(1)
    return gpus, cuda


def amd_gpus():
    gpus = []
    if not shutil.which("rocm-smi"):
        return gpus
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return gpus
    try:
        data = json.loads(out)
    except ValueError:
        return gpus
    for i, (card, info) in enumerate(sorted(data.items())):
        if not card.startswith("card"):
            continue
        total = info.get("VRAM Total Memory (B)") or info.get("vram Total Memory (B)")
        used = info.get("VRAM Total Used Memory (B)") or info.get("vram Total Used Memory (B)") or 0
        try:
            total_gb = float(total) / 1024**3
            used_gb = float(used) / 1024**3
        except (TypeError, ValueError):
            continue
        gpus.append(
            {
                "index": i,
                "vendor": "amd",
                "name": info.get("Card series") or info.get("Card model") or card,
                "vram_gb": round(total_gb, 1),
                "vram_free_gb": round(max(total_gb - used_gb, 0.0), 1),
                "driver": None,
                "compute_capability": info.get("GFX Version"),
            }
        )
    return gpus


def ram_gb():
    total = avail = None
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 1024**2
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        pass
    if total is None and sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            total = int(out.strip()) / 1024**3
    if total is None and sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total, avail = stat.ullTotalPhys / 1024**3, stat.ullAvailPhys / 1024**3
        except Exception:
            pass
    return (round(total, 1) if total else None), (round(avail, 1) if avail else None)


def cpu_model():
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def torch_info(timeout=90, python=None):
    code = (
        "import json, torch\n"
        "d = {'installed': True, 'version': torch.__version__, 'cuda': torch.version.cuda,\n"
        "     'hip': getattr(torch.version, 'hip', None), 'cuda_available': torch.cuda.is_available(),\n"
        "     'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,\n"
        "     'mps': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}\n"
        "if d['cuda_available']:\n"
        "    caps = [torch.cuda.get_device_capability(i) for i in range(d['device_count'])]\n"
        "    d['capabilities'] = ['%d.%d' % c for c in caps]\n"
        "    d['bf16'] = all(c[0] >= 8 for c in caps)\n"
        "    try:\n"
        "        torch.zeros(1, device='cuda'); d['cuda_works'] = True\n"
        "    except Exception as e:\n"
        "        d['cuda_works'] = False; d['cuda_error'] = str(e)[:300]\n"
        "print(json.dumps(d))\n"
    )
    try:
        out = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"installed": False, "error": str(exc)[:200]}
    if out.returncode != 0:
        err = (out.stderr or "").strip().splitlines()
        missing = any("No module named 'torch'" in line for line in err)
        return {"installed": not missing, "error": (err[-1] if err else "torch import failed")[:300]}
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"installed": True, "error": "could not parse torch probe output"}


def container_hint():
    env = os.environ
    if env.get("RUNPOD_POD_ID"):
        return "runpod"
    if env.get("VAST_CONTAINERLABEL") or env.get("VAST_TCP_PORT_22") or os.path.exists("/.vast"):
        return "vast"
    if env.get("COLAB_GPU") or env.get("COLAB_RELEASE_TAG"):
        return "colab"
    if env.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/.dockerenv"):
        return "docker"
    return None


TORCH_PYTHONS = ("/venv/main/bin/python", "/opt/conda/bin/python", "/usr/local/bin/python3", "python3")


def find_torch_python():
    """Interpreter that already has PyTorch (cloud templates often keep it in a venv/conda env)."""
    for cand in (sys.executable, *TORCH_PYTHONS):
        exe = cand if os.path.isabs(cand) else shutil.which(cand)
        if not exe or not os.path.exists(exe):
            continue
        try:
            out = subprocess.run([exe, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            return exe
    return None


def probe(with_torch=False, path="."):
    gpus, cuda = nvidia_gpus()
    if not gpus:
        gpus = amd_gpus()
    total, avail = ram_gb()
    check = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(check) and os.path.dirname(check) != check:
        check = os.path.dirname(check)  # the work folder may not exist yet
    try:
        usage = shutil.disk_usage(check)
        disk = {"path": check, "free_gb": round(usage.free / 1024**3, 1), "total_gb": round(usage.total / 1024**3, 1)}
    except OSError:
        disk = None
    machine = platform.machine().lower()
    apple_silicon = sys.platform == "darwin" and machine in ("arm64", "aarch64")
    report = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": machine,
        "python": platform.python_version(),
        "cpu": {"model": cpu_model(), "cores": os.cpu_count()},
        "ram_gb": total,
        "ram_available_gb": avail,
        "disk": disk,
        "gpus": gpus,
        "cuda_driver_version": cuda,
        "apple_silicon": apple_silicon,
        "container": container_hint(),
    }
    if with_torch:
        py = find_torch_python()
        report["torch"] = torch_info(python=py) if py else {"installed": False, "error": "no Python with PyTorch found"}
        if py:
            report["torch"]["python"] = py
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    with_torch = "--torch" in argv
    path = "."
    if "--path" in argv:
        i = argv.index("--path")
        if i + 1 < len(argv):
            path = os.path.expanduser(argv[i + 1])
    print(json.dumps(probe(with_torch=with_torch, path=path)))


if __name__ == "__main__":
    main()
