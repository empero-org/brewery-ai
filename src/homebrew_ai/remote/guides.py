"""Step-by-step guides for renting a GPU (Runpod, Vast.ai).

Homebrew does not create accounts, spend money or start machines for the user.
It explains how, recommends a GPU that fits the job, and takes over once the
user pastes the SSH command of their running machine.
"""

from __future__ import annotations

from typing import Any

from homebrew_ai.hardware.gpus import PRICES_CHECKED

PROVIDERS = {
    "runpod": {
        "name": "Runpod",
        "url": "https://www.runpod.io",
        "keys_url": "https://www.runpod.io/console/user/settings",
        "steps": [
            "Create an account at runpod.io and add credit (Billing). $10 is plenty for a first small brew.",
            "Open Settings → SSH Public Keys and paste the Homebrew public key shown below. Save.",
            "Go to Pods → Deploy. Pick the GPU recommended below. 'Community Cloud' is cheaper, 'Secure Cloud' is more reliable.",
            "Choose the official 'Runpod PyTorch' template (any recent 2.x version).",
            "Set Container Disk to at least 30 GB and Volume Disk to {disk_gb} GB (this job needs room for the model, checkpoints and an export).",
            "Make sure 'SSH over exposed TCP' / 'Expose TCP port 22' is enabled, then press Deploy and wait until the pod shows as Running.",
            "Click Connect and copy the command under 'SSH over exposed TCP'. It looks like: ssh root@203.0.113.7 -p 22093 -i ~/.ssh/id_ed25519 (do NOT use the ssh.runpod.io one).",
            "Paste that command back here.",
        ],
        "billing": [
            "Pods bill per second while running.",
            "A *stopped* pod still bills for its volume disk. When you are done (model uploaded or downloaded), TERMINATE the pod.",
            "If your balance hits $0 the pod is stopped and, without a network volume, its data is lost.",
        ],
    },
    "vast": {
        "name": "Vast.ai",
        "url": "https://vast.ai",
        "keys_url": "https://cloud.vast.ai/manage-keys/",
        "steps": [
            "Create an account at vast.ai and add credit (Billing). $10 is plenty for a first small brew.",
            "Open cloud.vast.ai/manage-keys and add the Homebrew public key shown below. Do this BEFORE renting: keys only apply to new instances.",
            "In Search, pick the 'PyTorch' template and filter for the GPU recommended below.",
            "Set disk space to at least {disk_gb} GB, and prefer offers with reliability above 98% and 'Max CUDA' 12.6 or newer.",
            "Rent the offer ('on-demand' is safest; 'interruptible' is cheaper but can be paused at any time).",
            "When the instance is running, open its Connect/SSH dialog and copy the 'Direct ssh connect' command (the proxy one also works). It looks like: ssh -p 41234 root@203.0.113.7 -L 8080:localhost:8080",
            "Paste that command back here.",
        ],
        "billing": [
            "Instances bill by the second while running, plus disk storage for as long as the instance exists.",
            "Stopping does NOT stop storage billing. DESTROY the instance when you are done.",
            "A stopped instance may not restart if someone else rented the GPU meanwhile — finish and download your model first.",
        ],
    },
}

BUDGET_CHECK = (
    "Renting a GPU costs real money: you pay per second while the machine runs, plus storage until you delete it. "
    "Decide on a budget first (a small first brew usually costs well under $5) and set a spending limit with the provider."
)


def rental_guide(provider: str, *, disk_gb: int, public_key: str | None, gpu_options: list[dict[str, Any]], level: str) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    info = PROVIDERS[provider]
    steps = [s.format(disk_gb=disk_gb) for s in info["steps"]]
    guide: dict[str, Any] = {
        "provider": info["name"],
        "website": info["url"],
        "ssh_keys_page": info["keys_url"],
        "steps": steps,
        "billing_warnings": info["billing"],
        "recommended_gpus": gpu_options[:4],
        "prices_note": f"Approximate on-demand prices checked {PRICES_CHECKED}; check the live price before renting.",
        "public_key": public_key,
    }
    if level in ("beginner", "hobbyist"):
        guide["budget_check"] = BUDGET_CHECK
    return guide


def free_options() -> list[dict[str, str]]:
    return [
        {"name": "Your own NVIDIA GPU", "notes": "Free if you have one with 8 GB+ (RTX 3060 and newer). Homebrew trains locally."},
        {"name": "Google Colab (free tier)", "notes": "A T4 (16 GB, no bf16) when available. Not yet automated by Homebrew; planned."},
        {"name": "Kaggle notebooks", "notes": "2x T4 for ~30 h/week. Not yet automated by Homebrew; planned."},
    ]
