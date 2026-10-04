import pytest

from homebrew_ai.etf.render import ChatFormat
from homebrew_ai.models.registry import all_models, families, get_model, search, summarize


def test_profiles_load_and_cover_requested_families():
    fams = {f.family for f in families()}
    assert {"qwen3", "qwen3_5", "qwen3_5_moe", "llama3", "gemma3", "gemma4", "qwen_image"} <= fams


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.id)
def test_every_variant_is_consistent(model):
    v = model.variant
    assert v.params_b > 0
    if model.modality == "text":
        fmt = ChatFormat.from_dict(model.chat_format_dict())
        assert fmt.assistant_header and fmt.turn_end
        assert model.profile.stop_tokens
        assert v.vocab and v.layers and v.hidden
    for preset in model.lora_presets():
        assert model.lora_targets(preset)
    for objective in ("sft", "cpt", "dpo"):
        if not model.guidelines.objective_allowed(objective):
            continue
        for method in ("lora", "qlora", "full"):
            g = model.guidelines.for_method(method, objective)
            if not g.allowed:
                continue
            for knob in ("learning_rate", "rank", "epochs", "max_steps", "effective_batch", "max_seq_len", "beta", "warmup_ratio", "resolution"):
                rng = getattr(g, knob)
                if rng is None or rng.default is None:
                    continue
                errors, _ = rng.check(rng.default, knob)
                assert not errors, (model.id, objective, method, knob, errors)
            if g.default_optimizer and g.optimizers:
                assert g.default_optimizer in g.optimizers


def test_licences():
    llama = get_model("meta-llama/Llama-3.1-8B-Instruct")
    assert llama.license.name_prefix == "Llama" and llama.license.hub_id == "llama3.1"
    assert get_model("meta-llama/Llama-3.2-3B-Instruct").license.hub_id == "llama3.2"
    img = get_model("Qwen/Qwen-Image-2.1")
    assert img.license.noncommercial and img.modality == "image"
    assert [o for o in ("sft", "cpt", "dpo") if img.guidelines.objective_allowed(o)] == ["sft"]


def test_moe_restrictions_and_search():
    moe = get_model("Qwen/Qwen3.5-35B-A3B")
    assert moe.variant.is_moe and not moe.guidelines.qlora.allowed and not moe.guidelines.full.allowed
    small = search(modality="text", max_params_b=2, include_base=False)
    assert small and all(m.variant.params_b <= 2 for m in small)
    rows = summarize(small)
    assert {"id", "license", "methods", "objectives"} <= set(rows[0])


def test_user_profiles_dir(tmp_path, monkeypatch):
    import yaml

    from homebrew_ai.models import registry

    d = tmp_path / "profiles"
    d.mkdir()
    prof = dict(registry._read_yaml("qwen3.yaml"))
    prof["family"] = "my_qwen"
    prof["variants"] = [{**prof["variants"][0], "id": "me/my-model", "label": "Mine"}]
    (d / "mine.yaml").write_text(yaml.safe_dump(prof))
    monkeypatch.setenv("HOMEBREW_AI_PROFILES", str(d))
    registry.load_profiles.cache_clear()
    assert get_model("me/my-model").family == "my_qwen"
