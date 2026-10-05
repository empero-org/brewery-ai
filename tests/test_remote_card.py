import pytest

from brewery_ai.models.registry import get_model
from brewery_ai.package.card import check_repo_name, render_card, suggest_repo_name
from brewery_ai.remote.sshutil import SSHParseError, parse_ssh_command


def test_parse_ssh_commands():
    s = parse_ssh_command("ssh root@203.0.113.7 -p 22093 -i ~/.ssh/id_ed25519")
    assert (s.user, s.host, s.port) == ("root", "203.0.113.7", 22093) and s.key.endswith("id_ed25519")
    v = parse_ssh_command("ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080")
    assert v.port == 41234 and v.provider == "vast"
    assert parse_ssh_command("ubuntu@myserver:2222").port == 2222
    with pytest.raises(SSHParseError):
        parse_ssh_command("ssh abc-123@ssh.runpod.io -i ~/.ssh/key")


def test_repo_name_rules():
    llama = get_model("meta-llama/Llama-3.1-8B-Instruct")
    assert check_repo_name("pirate-bot", llama)
    assert suggest_repo_name("pirate-bot", llama).startswith("Llama")
    img = get_model("Qwen/Qwen-Image-2.1")
    assert check_repo_name("user/qwen-style", img) and not check_repo_name("user/sketch-style", img)


def test_render_card_has_notices_and_stages():
    llama = get_model("meta-llama/Llama-3.2-1B-Instruct")
    stages = [
        {"objective": "sft", "method": "lora", "steps": 100, "data_summary": "500 examples", "hyperparameters": {"learning_rate": 2e-4}, "results": {"metrics": {"train_loss": 1.2}}},
        {"objective": "dpo", "method": "lora", "steps": 50, "data_summary": "300 pairs", "hyperparameters": {"learning_rate": 5e-6}, "results": None},
    ]
    card = render_card(repo_id="me/Llama-pirate", title="Llama pirate", description="Talks like a pirate.", info=llama, stages=stages,
                       export_kind="merged", datasets=[{"name": "x", "hf_id": "org/x", "records": 500, "license": "mit"}])
    assert card.startswith("---\n") and "license: llama3.2" in card and "Built with Llama" in card
    assert "Direct preference optimization" in card and "org/x" in card and "empero" in card.lower()
    img = get_model("Qwen/Qwen-Image-2.1")
    card2 = render_card(repo_id="me/sketch", title="Sketch", description="Pencil sketches.", info=img,
                        stages=[{"objective": "sft", "method": "lora", "steps": 1000, "image": {"trigger_word": "skt"}, "hyperparameters": {}}],
                        export_kind="lora", datasets=[], samples=[{"prompt": "skt cat", "path": "samples/sample_00.png"}])
    assert "license: other" in card2 and "license_name: qwen-research" in card2 and "Non-commercial" in card2 and "instance_prompt: skt" in card2
