"""JobManager launching a real detached worker process on this machine."""

import time

import pytest

from brewery_ai.etf.io import write_records
from brewery_ai.jobs.manager import JobManager
from brewery_ai.train.config import build_job

pytestmark = pytest.mark.slow


def test_launch_and_follow_local_job(project, tiny_model, tmp_path):
    data = tmp_path / "train.jsonl"
    write_records(data, [{"messages": [{"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": f"ahoy {i}"}]} for i in range(8)])
    job, report = build_job(model=tiny_model, project="t", method="lora", train_path="data/train.jsonl", eval_path=None, num_train=8, bf16=False,
                            overrides={"max_seq_len": 64, "effective_batch": 2, "micro_batch_size": 2, "optimizer": "adamw_torch", "epochs": 1}, expert_override=True)
    job.runtime.save_steps = 10_000
    project.state.compute.kind = "local"
    project.save()
    jm = JobManager(project)
    jm.stage(job, data, None)
    record = jm.launch(job)
    assert record["pid"] and project.job(job.job_id)
    deadline = time.time() + 300
    while time.time() < deadline:
        st = jm.status(job.job_id)
        if st["state"] in ("completed", "failed", "crashed"):
            break
        time.sleep(2)
    assert st["state"] == "completed", st.get("log_tail")
    assert project.job(job.job_id)["state"] == "completed"
