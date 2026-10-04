"""Regenerate docs/models.md from the shipped model profiles: python scripts/gen_models_doc.py"""

from __future__ import annotations

from pathlib import Path

from homebrew_ai.models.registry import families, get_model

OUT = Path(__file__).resolve().parents[1] / "docs" / "models.md"


def fmt_range(r) -> str:
    if r is None or r.default is None:
        return "–"
    lo = "" if r.min is None else f"{r.min:g}"
    hi = "" if r.max is None else f"{r.max:g}"
    rec = f" (rec. {r.recommended[0]:g}–{r.recommended[1]:g})" if r.recommended else ""
    bounds = f" [{lo}…{hi}]" if (lo or hi) else ""
    return f"**{r.default:g}**{rec}{bounds}"


def main() -> None:
    lines = [
        "# Supported models",
        "",
        "Generated from `src/homebrew_ai/models/profiles/*.yaml` by `scripts/gen_models_doc.py`. Every number below is a",
        "guardrail the agent must stay inside (`propose_training_config` rejects values outside the hard bounds unless the",
        "user grants an expert override; leaving the recommended band only produces a warning).",
        "",
    ]
    for fam in families():
        lines += [f"## {fam.display_name}", "", fam.summary.strip(), "", f"Licence: [{fam.license.name}]({fam.license.url})", ""]
        lines += ["| Model | Kind | Params (B) | Context | Licence | Notes |", "|---|---|---|---|---|---|"]
        for v in fam.variants:
            m = get_model(v.id)
            params = f"{v.params_b:g}" + (f" ({v.active_params_b:g} active)" if v.active_params_b else "")
            notes = []
            if v.recommended:
                notes.append("recommended")
            if v.gated:
                notes.append("gated")
            if v.publisher:
                notes.append(v.publisher)
            lines.append(f"| `{v.id}` | {v.kind} | {params} | {v.context or '–'} | {m.license.id} | {', '.join(notes)} |")
        sample = get_model(fam.variants[0].id)
        objectives = ("sft",) if fam.modality == "image" else ("sft", "cpt", "dpo")
        lines += ["", "Guidelines (first variant; individual variants may override):", ""]
        lines += ["| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |", "|---|---|---|---|---|---|---|"]
        for obj in objectives:
            for method in ("lora", "qlora", "full"):
                g = sample.guidelines.for_method(method, obj)
                if not g.allowed:
                    lines.append(f"| {obj} | {method} | not allowed | | | | {' '.join(g.notes)[:120]} |")
                    continue
                extra = []
                if g.beta is not None:
                    extra.append(f"β {fmt_range(g.beta)}")
                if g.resolution is not None:
                    extra.append(f"resolution {fmt_range(g.resolution)}")
                if g.max_seq_len is not None:
                    extra.append(f"max_seq_len {fmt_range(g.max_seq_len)}")
                steps = fmt_range(g.epochs) if fam.modality == "text" else fmt_range(g.max_steps)
                lines.append(f"| {obj} | {method} | {fmt_range(g.learning_rate)} | {fmt_range(g.rank)} | {steps} | {fmt_range(g.effective_batch)} | {'; '.join(extra)} |")
        if fam.notes:
            lines += ["", "Notes:", ""] + [f"- {n}" for n in fam.notes]
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
