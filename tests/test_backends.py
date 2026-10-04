import json

from homebrew_ai.backends.base import Message, ToolCall, parse_json_loose, validate_arguments
from homebrew_ai.backends.openai_backend import OpenAICompatBackend, _calls_from_text, _ToolBlockFilter
from homebrew_ai.backends.presets import capability_for


def test_validate_arguments():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]}}, "required": ["n"], "additionalProperties": False}
    assert validate_arguments(schema, {"n": 1, "mode": "a"}) == []
    errs = validate_arguments(schema, {"mode": "c", "x": 1})
    assert any("missing required" in e for e in errs) and any("must be one of" in e for e in errs) and any("unknown field" in e for e in errs)
    assert validate_arguments({"type": "integer"}, True)  # bools are not integers


def test_parse_json_loose():
    assert parse_json_loose('here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('prefix [1, 2] suffix') == [1, 2]


def test_anthropic_message_rendering_groups_results_and_replays_raw():
    from homebrew_ai.backends.anthropic_backend import AnthropicBackend

    b = AnthropicBackend("claude-opus-5-5", api_key="sk-test")
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}, {"type": "tool_use", "id": "t2", "name": "y", "input": {}}]
    history = [
        Message("user", ["hello", "<homebrew_state>s</homebrew_state>"]),
        Message("assistant", [], [ToolCall("t1", "x", {}), ToolCall("t2", "y", {})], raw=raw, raw_provider="anthropic"),
        Message("tool", ["r1"], tool_call_id="t1", name="x"),
        Message("tool", ["r2", "<homebrew_state>s2</homebrew_state>"], tool_call_id="t2", name="y", is_error=True),
    ]
    msgs = b._messages(history)
    assert msgs[1]["content"] == raw
    assert [c["type"] for c in msgs[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert msgs[2]["content"][1]["is_error"] is True
    params, betas = b._params("sys", history, [])
    assert params["thinking"]["type"] == "adaptive" and params["output_config"]["effort"] == "medium"
    assert "fallbacks" in params and betas
    assert b._degrade("block_binding: Extra inputs are not permitted") and not b.features["binding"]


def test_openai_text_mode_and_tool_block_parsing():
    b = OpenAICompatBackend("qwen3.5:9b", None, "http://localhost:11434/v1", tool_mode="text")
    history = [Message("user", ["hi"]), Message("assistant", ["ok"], [ToolCall("c1", "f", {"a": 1})]), Message("tool", ["done"], tool_call_id="c1", name="f")]
    msgs = b._messages("sys", history, [])
    assert msgs[2]["role"] == "assistant" and "<tool_call>" in msgs[2]["content"]
    assert msgs[3]["role"] == "user" and "<tool_result" in msgs[3]["content"]
    calls, rest = _calls_from_text('Sure.\n<tool_call>{"name": "f", "arguments": {"a": 2}}</tool_call>')
    assert calls[0].name == "f" and calls[0].arguments == {"a": 2} and rest == "Sure."
    shown = []
    filt = _ToolBlockFilter(shown.append)
    for piece in ["Hello <tool", "_call>{\"name\":1}</tool_c", "all> bye"]:
        filt.feed(piece)
    filt.flush()
    assert "".join(shown) == "Hello  bye"


def test_capability_heuristics():
    assert capability_for("claude-opus-5-5") == "high"
    assert capability_for("qwen3.5:9b") == "low"
    assert capability_for("gpt-5-mini") == "medium"
    assert capability_for("anything", "low") == "low"


def test_message_roundtrip():
    m = Message("assistant", ["x"], [ToolCall("1", "t", {"a": 1})], raw=[{"type": "text", "text": "x"}], raw_provider="anthropic")
    assert Message.from_dict(json.loads(json.dumps(m.to_dict()))) == m
