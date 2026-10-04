"""GPU catalog: memory, speed and approximate rental prices.

Prices are on-demand $/hour for one GPU, checked on 2026-10-04 (Runpod pricing
page; Vast.ai live offers API). Markets move — Homebrew always labels these as
estimates and tells users to check the live price before renting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRICES_CHECKED = "2026-10-04"


@dataclass(frozen=True)
class GPU:
    key: str
    name: str
    vram_gb: float
    tflops: float  # dense bf16 tensor TFLOPS (fp16 for GPUs without bf16; fp32-accumulate rate on GeForce)
    bf16: bool
    arch: str
    match: tuple[str, ...]  # lowercase substrings that identify it in nvidia-smi names
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # provider -> (low, typical)
    consumer: bool = False
    notes: str = ""

    def price(self, provider: str | None = None) -> float | None:
        options = [self.prices[provider]] if provider and provider in self.prices else list(self.prices.values())
        typical = [p[1] for p in options]
        return min(typical) if typical else None


CATALOG: tuple[GPU, ...] = (
    GPU("t4", "NVIDIA T4", 16, 65, False, "Turing", ("t4",), {"vast": (0.12, 0.16)},
        notes="No bf16; free on Google Colab and Kaggle. Fine for tiny models only."),
    GPU("v100", "NVIDIA V100", 16, 125, False, "Volta", ("v100",), {"vast": (0.10, 0.13)},
        notes="No bf16 and dropped by CUDA 13 (needs cu126 PyTorch wheels)."),
    GPU("rtx3060", "RTX 3060 12GB", 12, 25.5, True, "Ampere", ("3060",), consumer=True),
    GPU("rtx3090", "RTX 3090", 24, 71, True, "Ampere", ("3090",), {"runpod": (0.22, 0.22), "vast": (0.11, 0.18)}, consumer=True,
        notes="Cheapest 24 GB card to rent."),
    GPU("rtx4090", "RTX 4090", 24, 165, True, "Ada", ("4090",), {"runpod": (0.34, 0.34), "vast": (0.27, 0.42)}, consumer=True,
        notes="Great value for LoRA/QLoRA up to ~14B."),
    GPU("rtx5090", "RTX 5090", 32, 209, True, "Blackwell", ("5090",), {"runpod": (0.69, 0.69), "vast": (0.28, 0.54)}, consumer=True),
    GPU("a10g", "NVIDIA A10G", 24, 70, True, "Ampere", ("a10g", "a10 "), {}),
    GPU("l4", "NVIDIA L4", 24, 121, True, "Ada", ("l4",), {"runpod": (0.44, 0.44), "vast": (0.33, 0.33)}),
    GPU("a6000", "RTX A6000", 48, 155, True, "Ampere", ("a6000",), {"runpod": (0.33, 0.33), "vast": (0.36, 0.40)},
        notes="48 GB for little money; slower than newer cards."),
    GPU("rtx6000ada", "RTX 6000 Ada", 48, 364, True, "Ada", ("6000 ada",), {"runpod": (0.74, 0.74), "vast": (0.52, 0.76)}),
    GPU("l40s", "NVIDIA L40S", 48, 362, True, "Ada", ("l40s",), {"runpod": (0.79, 0.79), "vast": (0.47, 0.80)}),
    GPU("a100_40", "A100 40GB", 40, 312, True, "Ampere", ("a100-sxm4-40", "a100-pcie-40", "a100 40")),
    GPU("a100_80", "A100 80GB", 80, 312, True, "Ampere", ("a100-sxm4-80", "a100 80", "a100-pcie-80", "a100 80gb"),
        {"runpod": (1.19, 1.19), "vast": (0.66, 0.95)}),
    GPU("rtxpro5000", "RTX PRO 5000 Blackwell (48GB)", 48, 250, True, "Blackwell", ("rtx pro 5000",), {},
        notes="48 GB Blackwell workstation card; plenty for small-model SFT and image LoRAs."),
    GPU("rtxpro6000", "RTX PRO 6000 (96GB)", 96, 470, True, "Blackwell", ("rtx pro 6000",), {"runpod": (1.69, 1.69), "vast": (1.00, 1.45)},
        notes="96 GB of memory at a fraction of H100 prices; good for MoE LoRA."),
    GPU("h100_pcie", "H100 PCIe", 80, 756, True, "Hopper", ("h100 pcie",), {"runpod": (1.99, 1.99), "vast": (2.13, 2.35)}),
    GPU("h100", "H100 SXM", 80, 989, True, "Hopper", ("h100 80gb hbm3", "h100 sxm", "h100"), {"runpod": (2.69, 2.69), "vast": (2.02, 3.07)}),
    GPU("h100_nvl", "H100 NVL", 94, 835, True, "Hopper", ("h100 nvl",), {"runpod": (2.59, 2.59), "vast": (2.54, 2.82)}),
    GPU("h200", "H200", 141, 989, True, "Hopper", ("h200",), {"runpod": (3.59, 3.59), "vast": (2.63, 4.50)}),
    GPU("b200", "B200", 180, 2250, True, "Blackwell", ("b200",), {"runpod": (5.98, 5.98), "vast": (6.25, 7.66)}),
    GPU("mi300x", "AMD MI300X", 192, 1307, True, "CDNA3", ("mi300x",), {},
        notes="ROCm: works with PyTorch, but bitsandbytes/QLoRA support is limited."),
)

BY_KEY = {g.key: g for g in CATALOG}


def identify(name: str) -> GPU | None:
    """Match an nvidia-smi / rocm-smi product name to a catalog entry."""
    lowered = name.lower()
    best: tuple[int, GPU] | None = None
    for gpu in CATALOG:
        for needle in gpu.match:
            if needle in lowered and (best is None or len(needle) > best[0]):
                best = (len(needle), gpu)
    return best[1] if best else None


def rentable(provider: str | None = None) -> list[GPU]:
    """GPUs with a known rental price, cheapest first."""
    gpus = [g for g in CATALOG if g.price(provider) is not None]
    return sorted(gpus, key=lambda g: g.price(provider) or 0)


def generic(vram_gb: float, bf16: bool = True, tflops: float | None = None, name: str = "GPU") -> GPU:
    """Stand-in for an unknown GPU so estimates still work."""
    return GPU("unknown", name, vram_gb, tflops or (80.0 if bf16 else 30.0), bf16, "unknown", ())
