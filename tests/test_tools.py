import json

from homebrew_ai.agent.tools.base import ToolContext, run_tool
from homebrew_ai.backends.base import ToolCall
from homebrew_ai.jobs.manager import JobManager
from homebrew_ai.settings import Settings


def run(ctx, tool_name, **args):
    text, err = run_tool(ToolCall("x", tool_name, args), ctx)
    assert not err, text
    return json.loads(text)


def make_ctx(project, ui):
    return ToolContext(project=project, ui=ui, settings=Settings(), backend=None, jobs=JobManager(project))


def test_local_import_build_and_propose(project, fake_ui, tmp_path, qwen_tokenizer):
    data = tmp_path / "pirate.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"Arr {i}"}]} for i in range(60)]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    ctx = make_ctx(project, fake_ui)
    run(ctx, "select_base_model", model_id="Qwen/Qwen3-0.6B")
    rep = run(ctx, "import_dataset", name="pirate", source="local", path=str(data), license="cc-by-4.0")
    assert rep["records"] == 60 and rep["license"] == "cc-by-4.0"
    ts = run(ctx, "build_training_set", name="sft", parts=[{"name": "pirate"}])
    assert ts["num_train"] + ts["num_eval"] == 60 and "sft" in ts["objectives"] and ts["token_stats"]["max"] > 0
    prop = run(ctx, "propose_training_config", objective="sft", overrides={"epochs": 2})
    assert prop["stored_as_draft"] and prop["settings"]["epochs"] == 2 and prop["job_id"] in project.state.drafts
    bad, err = run_tool(ToolCall("y", "propose_training_config", {"objective": "dpo"}), ctx)
    assert err and "dpo" in bad
    preview = run(ctx, "preview_dataset", name="pirate", count=1)
    assert preview["shown_to_user"] == 1 and any(k == "segments" for k, _ in fake_ui.log)


def test_add_examples_and_stats(project, fake_ui):
    ctx = make_ctx(project, fake_ui)
    out = run(ctx, "add_examples", name="golden", records=[
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Ahoy!"}]},
        {"messages": [{"role": "user", "content": "broken"}]},
    ])
    assert out["added"] == 1 and out["problems"]
    stats = run(ctx, "dataset_stats", name="golden")
    assert stats["stats"]["records"] == 1


def test_estimate_and_explain(project, fake_ui):
    ctx = make_ctx(project, fake_ui)
    est = run(ctx, "estimate_requirements", base_model="Qwen/Qwen3-8B", method="qlora", train_tokens=2_000_000)
    assert est["memory"]["total_gb"] > 4 and est["cheapest_fitting_gpus"] and est["time_estimate"]["hours_low"] > 0
    assert run(ctx, "explain_term", term="LR")["term"] == "learning rate"
