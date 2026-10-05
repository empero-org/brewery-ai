import pytest

from brewery_ai.hardware.estimate import TrainShape, autofit, estimate_memory, fitting_gpus
from brewery_ai.models.registry import get_model
from brewery_ai.train.config import TrainJob, build_job, total_steps, validate_job


def make(model_id="Qwen/Qwen3-8B", method="lora", objective="sft", **kw):
    m = get_model(model_id)
    args = dict(model=m, project="t", method=method, objective=objective, train_path="data/train.jsonl", eval_path=None, num_train=1000, vram_gb=80.0)
    args.update(kw)
    return m, *build_job(**args)


def test_sft_defaults_follow_guidelines():
    m, job, report = make()
    assert not report["errors"]
    assert job.optim.learning_rate == 2e-4 and job.lora.rank == 16 and job.lora.alpha == 32
    assert job.model_profile["variants"][0]["id"] == "Qwen/Qwen3-8B"
    assert total_steps(job) == report["steps"]


def test_dpo_and_cpt_defaults():
    _, dpo, rep = make(objective="dpo", init_from={"job_id": "x", "kind": "adapter"})
    assert dpo.optim.learning_rate == 5e-6 and dpo.dpo.beta == 0.1 and dpo.init_from.job_id == "x" and not rep["errors"]
    _, cpt, rep = make(objective="cpt")
    assert cpt.lora.rank == 64 and cpt.optim.epochs == 1 and not rep["errors"]


def test_bounds_are_enforced_and_expert_override():
    _, job, report = make(overrides={"learning_rate": 0.5})
    assert any("learning_rate" in e for e in report["errors"])
    _, job, report = make(overrides={"learning_rate": 0.002}, expert_override=True)
    assert not report["errors"] and any("expert override" in w for w in report["warnings"])
    _, job, report = make("Qwen/Qwen3.5-35B-A3B", method="qlora")
    assert any("not allowed" in e for e in report["errors"])
    _, job, report = make(overrides={"max_seq_len": 10**6})
    assert any("context window" in e or "maximum" in e for e in report["errors"])


def test_image_job():
    m, job, report = make("Qwen/Qwen-Image-2.1", overrides={"trigger_word": "sks corgi"}, num_train=20, vram_gb=24.0)
    assert job.modality == "image" and job.image.trigger_word == "sks corgi" and job.optim.max_steps == 1000
    _, job2, report2 = make("Qwen/Qwen-Image-2.1", objective="dpo", num_train=20)
    assert report2["errors"]


def test_unknown_override_and_yaml_roundtrip(tmp_path):
    _, job, report = make(overrides={"warp_speed": 9})
    assert any("unknown settings" in e for e in report["errors"])
    _, job, _ = make()
    path = job.save(tmp_path / "job.yaml")
    assert TrainJob.load(path) == job


def test_memory_estimates_are_monotonic_and_fit():
    m = get_model("Qwen/Qwen3-8B")
    small = estimate_memory(m, TrainShape("lora", seq_len=1024)).total_gb
    big = estimate_memory(m, TrainShape("lora", seq_len=4096)).total_gb
    q = estimate_memory(m, TrainShape("qlora", seq_len=1024)).total_gb
    full = estimate_memory(m, TrainShape("full", seq_len=1024, optimizer="adamw_8bit")).total_gb
    assert q < small < big < full
    dpo = estimate_memory(m, TrainShape("lora", objective="dpo", seq_len=1024)).total_gb
    assert dpo > small
    fit = autofit(m, TrainShape("lora", seq_len=1024), vram_gb=80, effective_batch=16)
    assert fit.ok and fit.micro_batch * fit.grad_accum >= 16
    assert not autofit(m, TrainShape("full", seq_len=4096), vram_gb=24, effective_batch=16).ok
    rentals = fitting_gpus(m, TrainShape("qlora", seq_len=1024))
    assert rentals and all("T4" not in r["gpu"] and "V100" not in r["gpu"] for r in rentals)
