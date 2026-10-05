"""Tiny end-to-end training runs on CPU (slow-ish, ~1-2 minutes)."""

import json

import pytest

from brewery_ai.etf.io import write_records
from brewery_ai.train.config import build_job

pytestmark = pytest.mark.slow


def stage(root, model, name, objective, records, method="lora", init=None, over=None, stage_no=1):
    rd = root / name
    write_records(rd / "data/train.jsonl", records)
    job, rep = build_job(model=model, project="t", method=method, objective=objective, init_from=init, stage=stage_no, train_path="data/train.jsonl",
                         eval_path=None, num_train=len(records), bf16=False,
                         overrides={"max_seq_len": 128, "effective_batch": 4, "micro_batch_size": 2, "optimizer": "adamw_torch", "epochs": 1, **(over or {})},
                         expert_override=True)
    job.job_id = name
    job.runtime.save_steps = 10_000  # no intermediate checkpoints: keeps the test light on disk
    job.save(rd / "job.yaml")
    return rd / "job.yaml"


SFT = [{"messages": [{"role": "user", "content": f"What is {i}+{i}?"}, {"role": "assistant", "content": f"Arr, {2 * i}!"}]} for i in range(16)]


def status(path):
    return json.loads((path.parent / "status.json").read_text())


def test_sft_lora_full_and_generation(tmp_path, tiny_model):
    from brewery_ai.train.export import export_run
    from brewery_ai.train.infer import generate
    from brewery_ai.train.runner import run

    lora = stage(tmp_path, tiny_model, "lora", "sft", SFT, over={"learning_rate": 1e-3})
    assert run(lora) == 0 and status(lora)["state"] == "completed"
    assert (lora.parent / "final/adapter_model.safetensors").exists()
    out = generate(lora.parent, ["What is 2+2?"], compare_base=True, max_new_tokens=8)
    assert {"prompt", "finetuned", "base"} <= set(out[0])
    exp = export_run(lora.parent, tmp_path / "export-adapter")
    assert exp["kind"] == "adapter" and "adapter_model.safetensors" in exp["files"] and "brewery.json" in exp["files"]
    merged = export_run(lora.parent, tmp_path / "export-merged", merge=True)
    assert merged["kind"] == "merged" and any(f.endswith(".safetensors") for f in merged["files"])

    full = stage(tmp_path, tiny_model, "full", "sft", SFT, method="full", over={"learning_rate": 1e-4})
    assert run(full) == 0 and status(full)["state"] == "completed"
    assert (full.parent / "final/config.json").exists()


def test_regime_cpt_sft_dpo(tmp_path, tiny_model):
    from brewery_ai.train.runner import run_chain

    docs = [{"text": f"Chapter {i}. The brewer mixed {i} measures of malt and waited for the yeast."} for i in range(30)]
    pref = [{"messages": [{"role": "user", "content": f"Describe batch {i}."}], "chosen": f"Batch {i} was golden.", "rejected": "No idea."} for i in range(12)]
    j1 = stage(tmp_path, tiny_model, "s1", "cpt", docs, over={"learning_rate": 1e-3})
    j2 = stage(tmp_path, tiny_model, "s2", "sft", SFT, init={"job_id": "s1", "kind": "adapter"}, over={"learning_rate": 1e-3}, stage_no=2)
    j3 = stage(tmp_path, tiny_model, "s3", "dpo", pref, init={"job_id": "s2", "kind": "adapter"}, over={"learning_rate": 1e-4}, stage_no=3)
    assert run_chain([j1, j2, j3]) == 0
    s3 = status(j3)
    assert s3["state"] == "completed" and s3["init_from"].endswith("s2/merged") and s3["data"]["pairs"] == 12
    assert (tmp_path / "s1/merged/config.json").exists() and (tmp_path / "s2/merged/config.json").exists()


def test_failure_is_reported(tmp_path, tiny_model):
    from brewery_ai.train.runner import run

    bad = stage(tmp_path, tiny_model, "bad", "sft", SFT)
    import yaml

    data = yaml.safe_load(bad.read_text())
    data["data"]["max_seq_len"] = 4
    bad.write_text(yaml.safe_dump(data))
    assert run(bad) == 1
    st = status(bad)
    assert st["state"] == "failed" and st["error_kind"] == "data"


def test_later_stage_can_be_tested_and_exported_from_its_run_folder(tmp_path, tiny_model, monkeypatch):
    """Review #1: the worker runs test/export inside the run folder and passes "."."""
    from pathlib import Path

    from brewery_ai.train.export import export_run
    from brewery_ai.train.infer import load_finetuned
    from brewery_ai.train.runner import run_chain

    j1 = stage(tmp_path, tiny_model, "c1", "sft", SFT, over={"learning_rate": 1e-3})
    j2 = stage(tmp_path, tiny_model, "c2", "sft", SFT, init={"job_id": "c1", "kind": "adapter"}, over={"learning_rate": 1e-3}, stage_no=2)
    assert run_chain([j1, j2]) == 0
    monkeypatch.chdir(tmp_path / "c2")
    job, _info, _tok, _model, is_adapter = load_finetuned(Path("."))
    assert job.job_id == "c2" and is_adapter
    assert export_run(Path("."), tmp_path / "c2-export")["kind"] == "merged"


def test_repaired_token_rows_travel_inside_the_adapter(tmp_path):
    """Review #6: rows fixed by fix_untrained_tokens must survive loading the adapter onto the original base."""
    import types

    import torch
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    from brewery_ai.train.model import apply_lora

    ids = [60, 61]
    for tied in (False, True):
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied)
        base = LlamaForCausalLM(cfg)
        with torch.no_grad():  # what an untrained base checkpoint looks like
            base.get_input_embeddings().weight[ids] = 0
            base.get_output_embeddings().weight[ids] = 0
        base.save_pretrained(tmp_path / f"base-{tied}")
        model = LlamaForCausalLM.from_pretrained(tmp_path / f"base-{tied}")
        with torch.no_grad():  # the repair
            for emb in {id(e): e for e in (model.get_input_embeddings(), model.get_output_embeddings())}.values():
                emb.weight[ids] = 0.5
        job = types.SimpleNamespace(
            lora=types.SimpleNamespace(rank=4, alpha=8, dropout=0.0, targets=["q_proj", "v_proj"], use_rslora=False, use_dora=False, train_experts=False, train_embeddings=False),
            runtime=types.SimpleNamespace(gradient_checkpointing=False),
        )
        info = types.SimpleNamespace(lora_targets=lambda t: t, variant=types.SimpleNamespace(tied_embeddings=tied), profile=types.SimpleNamespace(pipeline={}))
        apply_lora(model, job, info, token_ids=ids).save_pretrained(tmp_path / f"adapter-{tied}")
        fresh = LlamaForCausalLM.from_pretrained(tmp_path / f"base-{tied}")
        merged = PeftModel.from_pretrained(fresh, tmp_path / f"adapter-{tied}").merge_and_unload()
        assert torch.allclose(merged.get_input_embeddings().weight[ids], torch.full((2, 32), 0.5))
        assert torch.allclose(merged.get_output_embeddings().weight[ids], torch.full((2, 32), 0.5))
