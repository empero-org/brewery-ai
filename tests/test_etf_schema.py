from homebrew_ai.etf.schema import content_text, ensure_record, normalize_record, record_fingerprint, record_kind


def codes(issues, level=None):
    return {i.code for i in issues if level is None or i.level == level}


def test_basic_trace_and_aliases():
    rec, issues = normalize_record({"conversations": [], "messages": [
        {"from": "human", "value": "hi"}, {"role": "gpt", "content": "hello"}]})
    assert rec["messages"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert not codes(issues, "error")


def test_system_shorthand_and_mid_system_warning():
    rec, issues = normalize_record({"system": "Be brief.", "messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "developer", "content": "Now in German."}, {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]})
    assert rec["messages"][0] == {"role": "system", "content": "Be brief."}
    assert rec["messages"][3]["role"] == "system"
    assert "system_not_first" in codes(issues, "warning")


def test_think_tags_and_reasoning_fields():
    rec = ensure_record({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nplan\n</think>\n\nanswer"}]})
    assert rec["messages"][1] == {"role": "assistant", "content": "answer", "reasoning": "plan"}
    rec = ensure_record({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "reasoning_content": "r", "content": "a"}]})
    assert rec["messages"][1]["reasoning"] == "r"


def test_tool_calls_ids_and_results():
    rec, issues = normalize_record({
        "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}, "description": "math"}}],
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": "{\"x\": 2}"}}]},
            {"role": "tool", "content": {"result": 4}},
            {"role": "assistant", "content": "4"}]})
    call = rec["messages"][1]["tool_calls"][0]
    assert call == {"id": "call_1", "name": "calc", "arguments": {"x": 2}}
    assert rec["messages"][2]["tool_call_id"] == "call_1" and rec["messages"][2]["name"] == "calc"
    assert rec["messages"][2]["content"] == {"result": 4}
    assert rec["tools"] == [{"name": "calc", "description": "math", "parameters": {"type": "object"}}]


def test_orphan_tool_result_is_error():
    rec, issues = normalize_record({"messages": [{"role": "user", "content": "x"}, {"role": "tool", "content": "y"}, {"role": "assistant", "content": "z"}]})
    assert rec is None and "orphan_tool_result" in codes(issues)


def test_content_parts_and_media():
    rec = ensure_record({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "x.png"}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]}]})
    assert rec["messages"][0]["content"][1] == {"type": "image", "image": "x.png"}
    assert content_text(rec["messages"][0]["content"]) == "what is this?"
    assert rec["messages"][1]["content"] == "a cat"


def test_loss_control_and_extras():
    rec = ensure_record({"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "train": False},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d", "weight": 2.5}],
        "train_on": "last", "repeat": 3, "template_kwargs": {"enable_thinking": False},
        "documents": ["plain doc", {"title": "T", "content": "body"}], "tool_choice": "auto", "meta": {"license": "mit"}})
    assert rec["messages"][1]["train"] is False and rec["messages"][3]["weight"] == 2.5
    assert rec["train_on"] == "last" and rec["repeat"] == 3 and rec["tool_choice"] == "auto"
    assert rec["documents"] == [{"text": "plain doc"}, {"text": "body", "title": "T"}]


def test_nothing_to_learn():
    rec, issues = normalize_record({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "train": False}]})
    assert rec is None and "nothing_to_learn" in codes(issues)
    rec, _ = normalize_record({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "train": False}], "train_on": "all"})
    assert rec is not None


def test_preference_record():
    rec = ensure_record({"messages": [{"role": "user", "content": "joke?"}], "chosen": "a good joke", "rejected": {"content": "no"}})
    assert rec["chosen"] == [{"role": "assistant", "content": "a good joke"}]
    assert rec["rejected"] == [{"role": "assistant", "content": "no"}]
    assert record_kind(rec) == "trace"


def test_text_completion_image_kinds():
    assert record_kind(ensure_record({"text": "doc"})) == "text"
    comp = ensure_record({"prompt": "p", "completion": "c"})
    assert record_kind(comp) == "completion"
    img, issues = normalize_record({"image": "a.png", "captions": ["x", "y"], "references": ["r.png"]})
    assert img == {"image": "a.png", "caption": "x", "captions": ["x", "y"], "references": ["r.png"]}
    assert normalize_record({"text": "a", "image": "b"})[0] is None


def test_trailing_user_dropped_and_fingerprint():
    rec, issues = normalize_record({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]})
    assert rec["messages"][-1]["role"] == "assistant" and "trailing_messages" in codes(issues)
    r1 = ensure_record({"id": 1, "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
    r2 = ensure_record({"id": 2, "meta": {"x": 1}, "messages": [{"role": "user", "content": "a "}, {"role": "assistant", "content": "b"}]})
    assert record_fingerprint(r1) == record_fingerprint(r2)
