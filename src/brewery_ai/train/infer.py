"""Quick generation with a finished run, to taste the brew (worker side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brewery_ai.train.config import TrainJob


def _messages(prompt: Any, system: str | None) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return prompt
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": str(prompt)}]


def load_finetuned(run_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brewery_ai.train import model as tmodel

    job = TrainJob.load(run_dir / "job.yaml")
    info = job.resolved_model()
    final = run_dir / "final"
    precision = tmodel.resolve_precision("auto", info)
    dtype = tmodel.torch_dtype(precision)
    device_map = "auto" if torch.cuda.is_available() else None
    tok = AutoTokenizer.from_pretrained(final)
    if job.method == "full":
        model = AutoModelForCausalLM.from_pretrained(final, dtype=dtype, device_map=device_map)
        is_adapter = False
    else:
        from peft import PeftModel

        from brewery_ai.train.runner import resolve_start

        start = resolve_start(job, run_dir, info)  # earlier stage's merged weights, or None for the base model
        base = AutoModelForCausalLM.from_pretrained(start or info.id, dtype=dtype, device_map=device_map, token=tmodel.hf_token())
        model = PeftModel.from_pretrained(base, final)
        is_adapter = True
    model.eval()
    return job, info, tok, model, is_adapter


def generate(
    run_dir: str | Path,
    prompts: list[Any],
    *,
    system: str | None = None,
    compare_base: bool = False,
    max_new_tokens: int = 256,
    thinking: bool = False,
) -> list[dict[str, Any]]:
    import torch

    from brewery_ai.train import model as tmodel

    run_dir = Path(run_dir)
    job, info, tok, model, is_adapter = load_finetuned(run_dir)
    sampling = info.sampling("thinking" if thinking else "default")
    stop = tmodel.stop_token_ids(tok, info)
    fmt = info.chat_format_dict()
    device = next(model.parameters()).device
    results = []
    for prompt in prompts:
        kwargs: dict[str, Any] = {}
        if fmt.get("thinking_flag"):
            kwargs[fmt["thinking_flag"]] = thinking
        enc = tok.apply_chat_template(
            _messages(prompt, system), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        n = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=stop, pad_token_id=tok.pad_token_id, **sampling)

        def run_once() -> str:
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            return tok.decode(out[0][n:], skip_special_tokens=True).strip()

        row = {"prompt": prompt, "finetuned": run_once()}
        if compare_base and is_adapter:
            with model.disable_adapter():
                row["base"] = run_once()
        results.append(row)
    return results


def main_test(run_dir: str, prompts_file: str, out: str, compare_base: bool, system: str | None, thinking: bool, max_new_tokens: int) -> int:
    prompts = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    job = TrainJob.load(Path(run_dir) / "job.yaml")
    if job.modality == "image":
        from brewery_ai.train.image import sample_images

        rows = sample_images(Path(run_dir), prompts, compare_base=compare_base, prefix="test")
    else:
        rows = generate(run_dir, prompts, system=system, compare_base=compare_base, thinking=thinking, max_new_tokens=max_new_tokens)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))
    return 0


def candidates(run_dir: str | Path, prompts: list[Any], *, n: int = 2, max_new_tokens: int = 512, thinking: bool = False) -> list[dict[str, Any]]:
    """Sample ``n`` different answers per prompt from a finished run (for preference collection)."""
    import torch

    from brewery_ai.train import model as tmodel

    run_dir = Path(run_dir)
    job, info, tok, model, _ = load_finetuned(run_dir)
    sampling = dict(info.sampling("thinking" if thinking else "default"))
    sampling["temperature"] = max(float(sampling.get("temperature", 0.7)), 0.8)  # a little extra variety between candidates
    stop = tmodel.stop_token_ids(tok, info)
    fmt = info.chat_format_dict()
    device = next(model.parameters()).device
    rows = []
    for prompt in prompts:
        messages = prompt.get("messages") if isinstance(prompt, dict) else _messages(prompt, None)
        while messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        kwargs: dict[str, Any] = {}
        if fmt.get("thinking_flag"):
            kwargs[fmt["thinking_flag"]] = thinking
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kwargs)
        enc = {k: v.to(device) for k, v in enc.items()}
        length = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, num_return_sequences=n, eos_token_id=stop, pad_token_id=tok.pad_token_id, **sampling)
        texts = [tok.decode(seq[length:], skip_special_tokens=True).strip() for seq in out]
        rows.append({"messages": messages, "candidates": texts})
    return rows


def main_candidates(run_dir: str, prompts_file: str, out: str, n: int, max_new_tokens: int, thinking: bool) -> int:
    from brewery_ai.etf.io import iter_raw

    prompts = [raw for _, raw in iter_raw(prompts_file) if not isinstance(raw, Exception)]
    rows = candidates(run_dir, prompts, n=n, max_new_tokens=max_new_tokens, thinking=thinking)
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(rows), "candidates_per_prompt": n, "out": out}))
    return 0
