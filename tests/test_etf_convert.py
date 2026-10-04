import pytest

from homebrew_ai.etf.convert import Mapping, convert_row, detect_mapping
from homebrew_ai.etf.schema import ensure_record


def conv(rows, mapping=None):
    m = mapping or detect_mapping(rows)[0]
    return [ensure_record(convert_row(r, m, {"source": "test"})) for r in rows], m


def test_alpaca():
    recs, m = conv([{"instruction": "Translate", "input": "Hallo", "output": "Hello"}])
    assert m.layout == "alpaca"
    assert recs[0]["messages"][0]["content"] == "Translate\n\nHallo"
    assert recs[0]["meta"] == {"source": "test"}


def test_sharegpt_with_function_calls():
    rows = [{"conversations": [
        {"from": "system", "value": "sys"}, {"from": "human", "value": "weather?"},
        {"from": "function_call", "value": "{\"name\": \"w\", \"arguments\": {\"c\": \"Paris\"}}"},
        {"from": "observation", "value": "21C"}, {"from": "gpt", "value": "21C in Paris"}], "tools": "[{\"name\": \"w\"}]"}]
    recs, m = conv(rows)
    msgs = recs[0]["messages"]
    assert [x["role"] for x in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[2]["tool_calls"][0]["name"] == "w" and recs[0]["tools"][0]["name"] == "w"


def test_prompt_completion_with_reasoning_and_system_text():
    rows = [{"question": "2+2", "answer": "4", "rationale": "add"}]
    m, _ = detect_mapping(rows)
    m.system_text = "You are a calculator."
    recs, _ = conv(rows, m)
    assert recs[0]["messages"][0] == {"role": "system", "content": "You are a calculator."}
    assert recs[0]["messages"][2]["reasoning"] == "add"


def test_templates():
    m = Mapping(layout="prompt_completion", prompt_template="Translate to German: {en}", completion="de")
    recs, _ = conv([{"en": "cat", "de": "Katze"}], m)
    assert recs[0]["messages"][0]["content"] == "Translate to German: cat"
    with pytest.raises(KeyError):
        convert_row({"x": 1}, Mapping(layout="prompt_completion", prompt_template="{missing}", completion="x"))


def test_preference_layouts():
    recs, m = conv([{"prompt": "q", "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}], "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}]}])
    assert m.layout == "preference"
    assert recs[0]["messages"] == [{"role": "user", "content": "q"}]
    assert recs[0]["chosen"][0]["content"] == "good" and recs[0]["rejected"][0]["content"] == "bad"
    hh, _ = conv([{"chosen": "\n\nHuman: hi\n\nAssistant: hello!", "rejected": "\n\nHuman: hi\n\nAssistant: go away"}])
    assert hh[0]["messages"][0]["content"] == "hi" and hh[0]["chosen"][0]["content"] == "hello!"


def test_text_and_mapping_roundtrip():
    recs, m = conv([{"content": "a long document"}])
    assert m.layout == "text" and recs[0]["text"] == "a long document"
    assert Mapping.from_dict(m.to_dict()) == m
    with pytest.raises(ValueError):
        Mapping.from_dict({"layout": "nope"})
