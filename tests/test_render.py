import pytest

from homebrew_ai.etf.render import ChatFormat, Renderer, encode_cpt, pack_sequences
from homebrew_ai.etf.schema import ensure_record
from homebrew_ai.models.registry import get_model


@pytest.fixture
def renderer(qwen_tokenizer):
    m = get_model("Qwen/Qwen3-0.6B")
    return Renderer(qwen_tokenizer, ChatFormat.from_dict(m.chat_format_dict()), max_len=4096)


def trained_text(renderer, rec):
    return "".join(s for s, t in renderer.segments(ensure_record(rec)) if t)


def test_assistant_only_loss_and_empty_think_excluded(renderer):
    rec = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]}
    assert trained_text(renderer, rec) == "Hello!<|im_end|>"
    sample = renderer.encode(ensure_record(rec))[0]
    decoded = renderer.tokenizer.decode([t for t, l in zip(sample.input_ids, sample.labels) if l != -100])
    assert decoded == "Hello!<|im_end|>"


def test_history_reasoning_split(renderer):
    rec = ensure_record({"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "reasoning": "r1", "content": "A"},
        {"role": "user", "content": "b"}, {"role": "assistant", "reasoning": "r2", "content": "B"}]})
    assert renderer.drops_history_reasoning
    samples = renderer.encode(rec)
    assert len(samples) == 2
    texts = [renderer.tokenizer.decode([t for t, l in zip(s.input_ids, s.labels) if l != -100]) for s in samples]
    assert "r1" in texts[0] and "r2" in texts[1]


def test_tools_documents_and_mid_system(renderer):
    rec = {"documents": [{"title": "Doc", "text": "Hops add bitterness."}],
           "tools": [{"name": "calc", "description": "math", "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}],
           "messages": [
               {"role": "user", "content": "2x2?"},
               {"role": "assistant", "content": "", "tool_calls": [{"name": "calc", "arguments": {"x": 4}}]},
               {"role": "tool", "content": "4"},
               {"role": "assistant", "content": "It is 4."},
               {"role": "system", "content": "Answer in German."},
               {"role": "user", "content": "Danke"},
               {"role": "assistant", "content": "Bitte!"}]}
    segs = renderer.segments(ensure_record(rec))
    full = "".join(s for s, _ in segs)
    trained = "".join(s for s, t in segs if t)
    assert "Hops add bitterness." in full and "# Tools" in full
    assert "[System note] Answer in German." in full
    assert '"name": "calc"' in trained and "It is 4." in trained and "Bitte!" in trained
    assert "<tool_response>" not in trained


def test_train_on_last_and_all(renderer):
    base = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "B1"}]}
    assert trained_text(renderer, {**base, "train_on": "last"}).startswith("B1")
    assert "A1" not in trained_text(renderer, {**base, "train_on": "last"})
    everything = renderer.segments(ensure_record({**base, "train_on": "all"}))
    assert all(t for _, t in everything)


def test_preference_pair(renderer):
    pair = renderer.encode_preference(ensure_record({"messages": [{"role": "user", "content": "joke"}], "chosen": "funny", "rejected": "no"}))
    tok = renderer.tokenizer
    chosen = tok.decode([t for t, l in zip(pair.chosen_ids, pair.chosen_labels) if l != -100])
    rejected = tok.decode([t for t, l in zip(pair.rejected_ids, pair.rejected_labels) if l != -100])
    assert chosen == "funny<|im_end|>" and rejected == "no<|im_end|>"


def test_completion_and_cpt(renderer):
    s = renderer.encode(ensure_record({"prompt": "def f():", "completion": " return 1"}))[0]
    assert renderer.tokenizer.decode([t for t, l in zip(s.input_ids, s.labels) if l != -100]).startswith(" return 1")
    docs = [ensure_record({"text": "word " * 50}) for _ in range(10)]
    renderer.max_len = 64
    samples, counts = encode_cpt(renderer, docs)
    assert all(len(x.input_ids) <= 64 for x in samples) and counts["records"] == 10
    assert pack_sequences([[1, 2, 3]] * 3, max_len=4, min_tail=1)[0].input_ids == [1, 2, 3, 1]


def test_overflow_drop_and_truncate(renderer):
    rec = ensure_record({"messages": [{"role": "user", "content": "x " * 400}, {"role": "assistant", "content": "y"}]})
    renderer.max_len = 64
    assert renderer.encode(rec) == []
    renderer.overflow = "truncate"
    assert renderer.encode(rec) == []  # the answer itself is cut off -> nothing to learn, dropped
