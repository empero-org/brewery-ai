"""ETF → model input: chat-template rendering and loss masking.

Homebrew always renders with the model's *official* chat template (the one
shipped in the model repo), so training data looks exactly like what the model
sees at inference time. Loss masking works on the rendered text:

1. the ETF trace is adapted to the message dialect the family's template
   expects (reasoning field name, tool-call shape, role alternation, ...);
2. ``tokenizer.apply_chat_template(..., tokenize=False)`` renders it;
3. every assistant turn is located between the family's assistant header and
   its end-of-turn token — that character span is "trainable";
4. the text is tokenized once with offsets, and each token inside a trainable
   span keeps its label, everything else gets ``-100``.

This module is shared by the worker (training) and the control plane (data
previews and token statistics). It imports nothing heavy at module level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from homebrew_ai.etf.schema import content_text, media_parts

IGNORE_INDEX = -100


class RenderError(ValueError):
    """A record cannot be rendered for this model family (it is skipped)."""


@dataclass
class ChatFormat:
    """How a model family's chat template marks and wants things.

    Values come from the model profiles (``homebrew_ai/models/profiles``).
    """

    assistant_header: str
    turn_end: tuple[str, ...]
    reasoning: str = "native"  # native | inline | none
    reasoning_field: str = "reasoning_content"
    tools: str = "native"  # native | inline
    tool_args: str = "dict"  # dict | json_string
    strict_alternation: bool = False
    empty_think: str | None = None  # an empty reasoning block the template adds; excluded from the loss
    empty_think_inject: str | None = None  # empty reasoning block to *insert* for non-reasoning answers
    thinking_flag: str | None = None  # template kwarg set to True when a sample has reasoning
    tool_turns_merged: bool = False  # tool results render inside the same assistant turn
    exclude_within_turn: list[tuple[str, str]] = field(default_factory=list)  # (start, end) markers
    max_tool_calls_per_turn: int | None = None
    mid_system: str = "merge"  # native | merge (into the next user turn)
    documents: str = "inline"  # native (template `documents=` kwarg) | inline (into the system prompt)
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    document_end: str | None = None  # token appended to text records
    document_bos: bool = False  # make sure text records start with BOS

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatFormat":
        data = dict(data)
        turn_end = data.pop("turn_end")
        if isinstance(turn_end, str):
            turn_end = [turn_end]
        if "exclude_within_turn" in data:
            data["exclude_within_turn"] = [tuple(pair) for pair in data["exclude_within_turn"]]
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown chat_format keys: {', '.join(sorted(unknown))}")
        return cls(turn_end=tuple(turn_end), **data)


@dataclass
class PreferencePair:
    chosen_ids: list[int]
    chosen_labels: list[int]
    rejected_ids: list[int]
    rejected_labels: list[int]

    @property
    def num_tokens(self) -> int:
        return len(self.chosen_ids) + len(self.rejected_ids)


@dataclass
class Sample:
    input_ids: list[int]
    labels: list[int]

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def num_trainable(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


# --------------------------------------------------------------------------- #
# ETF trace → HF chat messages
# --------------------------------------------------------------------------- #

INLINE_TOOLS_PREAMBLE = (
    "You can call tools. To call one, reply with one or more blocks of the form\n"
    "<tool_call>\n{\"name\": <tool-name>, \"arguments\": <json-object>}\n</tool_call>\n"
    "Tool results come back inside <tool_response></tool_response>.\n\nAvailable tools:\n"
)


def _hf_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
        for t in tools
    ]


def _arguments(call: dict[str, Any], fmt: ChatFormat) -> Any:
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            if fmt.tool_args == "dict" and fmt.tools == "native":
                raise RenderError(f"tool call {call.get('name')!r} has non-object arguments, which this chat template cannot render")
            return args
        args = parsed
    if fmt.tool_args == "json_string":
        return json.dumps(args, ensure_ascii=False)
    return args


def _text_of(content: Any, *, role: str) -> str:
    if role == "tool" and isinstance(content, (dict, list)) and not media_parts(content):
        return json.dumps(content, ensure_ascii=False)
    return content_text(content)


def _documents_block(docs: list[dict[str, Any]]) -> str:
    lines = ["Use the following documents to answer:"]
    for i, d in enumerate(docs, 1):
        title = f" {d['title']}" if d.get("title") else ""
        lines.append(f"[{i}]{title}\n{d['text']}")
    return "\n\n".join(lines)


def conversation(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages to render: preference records train on their ``chosen`` continuation."""
    msgs = list(record["messages"])
    if record.get("chosen"):
        msgs = msgs + list(record["chosen"])
    return msgs


def adapt_messages(record: dict[str, Any], fmt: ChatFormat) -> tuple[list[dict[str, Any]], list[bool], list[bool], list[dict[str, Any]] | None, dict[str, Any]]:
    """Turn a canonical ETF trace into HF chat messages for ``fmt``.

    Returns ``(messages, train_flags, has_reasoning, tools, template_kwargs)``;
    the two flag lists are aligned with the assistant messages in ``messages``.
    """
    out: list[dict[str, Any]] = []
    train: list[bool] = []
    reasoning_flags: list[bool] = []
    tools = record.get("tools") or []
    hf_tools = _hf_tool_specs(tools) if tools and fmt.tools == "native" else None
    kwargs: dict[str, Any] = dict(record.get("template_kwargs") or {})
    msgs = conversation(record)
    train_on = record.get("train_on", "assistant")
    last_assistant = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    pending_system: list[str] = []

    for idx, m in enumerate(msgs):
        role = m["role"]
        text = _text_of(m.get("content"), role=role)
        if role == "system":
            if not out:
                out.append({"role": "system", "content": text})
            elif fmt.mid_system == "native":
                out.append({"role": "system", "content": text})
            else:
                pending_system.append(text)
            continue
        if role == "user":
            if pending_system:
                text = "\n\n".join(f"[System note] {t}" for t in pending_system) + "\n\n" + text
                pending_system = []
            if not text.strip():
                raise RenderError("a user turn has no text (media-only turns need a multimodal trainer)")
            out.append({"role": "user", "content": text})
            continue
        if role == "assistant":
            content = text
            msg: dict[str, Any] = {"role": "assistant"}
            reasoning = m.get("reasoning")
            if reasoning and fmt.reasoning == "native":
                msg[fmt.reasoning_field] = reasoning
            elif reasoning and fmt.reasoning == "inline":
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            calls = m.get("tool_calls") or []
            if fmt.max_tool_calls_per_turn is not None and len(calls) > fmt.max_tool_calls_per_turn:
                raise RenderError(f"this chat template allows at most {fmt.max_tool_calls_per_turn} tool call(s) per assistant turn")
            if calls and fmt.tools == "native":
                msg["tool_calls"] = [
                    {"type": "function", "id": c["id"], "function": {"name": c["name"], "arguments": _arguments(c, fmt)}}
                    for c in calls
                ]
            elif calls:
                blocks = [
                    "<tool_call>\n" + json.dumps({"name": c["name"], "arguments": _arguments(c, fmt)}, ensure_ascii=False) + "\n</tool_call>"
                    for c in calls
                ]
                content = (content.rstrip() + "\n" if content.strip() else "") + "\n".join(blocks)
            msg["content"] = content
            out.append(msg)
            trainable = bool(m.get("train", True))
            if train_on == "last":
                trainable = trainable and idx == last_assistant
            train.append(trainable)
            reasoning_flags.append(bool(reasoning) and fmt.reasoning != "none")
            continue
        if role == "tool":
            if fmt.tools == "native":
                tool_msg = {"role": "tool", "content": text}
                if m.get("tool_call_id"):
                    tool_msg["tool_call_id"] = m["tool_call_id"]
                if m.get("name"):
                    tool_msg["name"] = m["name"]
                out.append(tool_msg)
            else:
                out.append({"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"})
            continue
        raise RenderError(f"unknown role {role!r}")

    preamble_parts = []
    docs = record.get("documents") or []
    if docs:
        if fmt.documents == "native":
            kwargs["documents"] = [{"title": d.get("title", ""), "text": d["text"]} for d in docs]
        else:
            preamble_parts.append(_documents_block(docs))
    if tools and fmt.tools != "native":
        preamble_parts.append(INLINE_TOOLS_PREAMBLE + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools))
    if preamble_parts:
        preamble = "\n\n".join(preamble_parts)
        if out and out[0]["role"] == "system":
            out[0] = {"role": "system", "content": out[0]["content"].rstrip() + "\n\n" + preamble}
        else:
            out.insert(0, {"role": "system", "content": preamble})

    if fmt.strict_alternation:
        out, train, reasoning_flags = _merge_same_roles(out, train, reasoning_flags)
    return out, train, reasoning_flags, hf_tools, kwargs


def _merge_same_roles(messages: list[dict[str, Any]], train: list[bool], reasoning: list[bool]):
    merged: list[dict[str, Any]] = []
    new_train: list[bool] = []
    new_reasoning: list[bool] = []
    a_idx = 0
    for m in messages:
        is_assistant = m["role"] == "assistant"
        if merged and merged[-1]["role"] == m["role"] and m["role"] in ("user", "assistant"):
            merged[-1] = {**merged[-1], "content": merged[-1]["content"].rstrip() + "\n\n" + m["content"]}
            if is_assistant:
                new_train[-1] = new_train[-1] or train[a_idx]
                new_reasoning[-1] = new_reasoning[-1] or reasoning[a_idx]
        else:
            merged.append(dict(m))
            if is_assistant:
                new_train.append(train[a_idx])
                new_reasoning.append(reasoning[a_idx])
        if is_assistant:
            a_idx += 1
    return merged, new_train, new_reasoning


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    """Renders ETF records into training samples for one tokenizer + chat format."""

    def __init__(self, tokenizer: Any, fmt: ChatFormat, max_len: int = 4096, overflow: str = "drop"):
        if overflow not in ("drop", "truncate"):
            raise ValueError("overflow must be 'drop' or 'truncate'")
        self.tokenizer = tokenizer
        self.fmt = fmt
        self.max_len = max_len
        self.overflow = overflow
        self._drops_history: bool | None = None

    # -- template probing ---------------------------------------------------

    @property
    def drops_history_reasoning(self) -> bool:
        """True when the template removes reasoning from earlier assistant turns."""
        if self._drops_history is None:
            self._drops_history = self._probe_history_reasoning()
        return self._drops_history

    def _probe_history_reasoning(self) -> bool:
        if self.fmt.reasoning != "native":
            return False
        probe = {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "reasoning": "HB_PROBE_R1", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "reasoning": "HB_PROBE_R2", "content": "a2"},
            ]
        }
        try:
            text = self.render_text(probe)[0]
        except Exception:
            return False
        return "HB_PROBE_R1" not in text

    # -- records → texts ----------------------------------------------------

    def expand(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Split multi-turn reasoning traces when the template would drop earlier reasoning.

        Each trainable assistant turn becomes its own sample whose history is
        rendered exactly like at inference time; only that turn is trained.
        """
        if "messages" not in record or record.get("train_on") == "all":
            return [record]
        if record.get("chosen"):
            record = {k: v for k, v in record.items() if k not in ("chosen", "rejected", "label")} | {"messages": conversation(record)}
        msgs = record["messages"]
        assistant_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        reasoning_before_last = any(msgs[i].get("reasoning") for i in assistant_idx[:-1])
        if not reasoning_before_last or not self.drops_history_reasoning:
            return [record]
        out = []
        for k in assistant_idx:
            if not msgs[k].get("train", True):
                continue
            prefix = []
            for i, m in enumerate(msgs[: k + 1]):
                if m["role"] == "assistant" and i != k:
                    m = {**m, "train": False}
                prefix.append(m)
            out.append({**record, "messages": prefix})
        return out

    def render_text(self, record: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
        """Render one trace; returns ``(text, trainable_char_spans)``."""
        fmt = self.fmt
        messages, train, reasoning, tools, record_kwargs = adapt_messages(record, fmt)
        kwargs = {**fmt.template_kwargs, **record_kwargs}
        if tools:
            kwargs["tools"] = tools
        if fmt.thinking_flag:
            kwargs[fmt.thinking_flag] = any(reasoning)
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
        except Exception as exc:  # jinja raise_exception(...) and friends
            raise RenderError(f"chat template rejected the conversation: {exc}") from exc
        if not isinstance(text, str):
            raise RenderError("chat template did not return text")

        groups = self._assistant_groups(messages, train, reasoning)
        if record.get("train_on") == "all":
            return text, [(0, len(text))]
        turns = self.find_assistant_turns(text)
        if len(turns) != len(groups):
            raise RenderError(f"found {len(turns)} assistant turns in the rendered text but expected {len(groups)}")

        # Insert an empty reasoning block where the template expects one but does
        # not render it for training (e.g. Gemma 4 26B/31B non-thinking answers).
        inject = fmt.empty_think_inject
        if inject:
            for gi in range(len(groups) - 1, -1, -1):
                trainable, had_reasoning, current = groups[gi]
                if current and not had_reasoning:
                    at = turns[gi][0]
                    text = text[:at] + inject + text[at:]
                    turns = [
                        (a + len(inject), b + len(inject)) if j > gi else ((a, b + len(inject)) if j == gi else (a, b))
                        for j, (a, b) in enumerate(turns)
                    ]

        spans: list[tuple[int, int]] = []
        for (start, end), (trainable, had_reasoning, _current) in zip(turns, groups):
            if not trainable:
                continue
            for empty in (inject, fmt.empty_think):
                if empty and not had_reasoning and text.startswith(empty, start):
                    start += len(empty)
                    break
            spans.extend(self._minus_exclusions(text, start, end))
        return text, spans

    def _assistant_groups(self, messages, train, reasoning) -> list[tuple[bool, bool, bool]]:
        """One ``(trainable, had_reasoning, after_last_user)`` entry per rendered assistant turn."""
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        groups: list[tuple[bool, bool, bool]] = []
        a_idx = 0
        prev_role = None
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                merge = self.fmt.tool_turns_merged and groups and prev_role in ("tool", "assistant")
                if merge:
                    t, r, c = groups[-1]
                    groups[-1] = (t or train[a_idx], r, c)
                else:
                    groups.append((train[a_idx], reasoning[a_idx], i > last_user))
                a_idx += 1
            prev_role = m["role"] if m["role"] != "system" else prev_role
        return groups

    def _minus_exclusions(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split ``[start, end)`` around excluded regions such as embedded tool responses.

        The start marker itself stays trainable (the model emits it to hand over
        to the tool); everything after it up to and including the end marker is
        environment output and gets no loss.
        """
        spans = [(start, end)]
        for open_tok, close_tok in self.fmt.exclude_within_turn:
            out = []
            for a, b in spans:
                pos = a
                while True:
                    i = text.find(open_tok, pos, b)
                    if i < 0:
                        out.append((pos, b))
                        break
                    cut = i + len(open_tok)
                    j = text.find(close_tok, cut, b)
                    resume = b if j < 0 else j + len(close_tok)
                    out.append((pos, cut))
                    pos = resume
                    if pos >= b:
                        break
            spans = [(a, b) for a, b in out if b > a]
        return spans

    def find_assistant_turns(self, text: str) -> list[tuple[int, int]]:
        header = self.fmt.assistant_header
        turns = []
        pos = 0
        while True:
            h = text.find(header, pos)
            if h < 0:
                break
            start = h + len(header)
            ends = [(text.find(tok, start), tok) for tok in self.fmt.turn_end]
            ends = [(i, tok) for i, tok in ends if i >= 0]
            if not ends:
                if not text[start:].strip():
                    break  # some templates always append an empty generation prompt
                raise RenderError("assistant turn without an end-of-turn token")
            i, tok = min(ends)
            end = i + len(tok)
            turns.append((start, end))
            pos = end
        return turns

    # -- texts → token samples ---------------------------------------------

    def _tokenize(self, text: str, spans: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        labels = [IGNORE_INDEX] * len(ids)
        if spans:
            span_iter = iter(sorted(spans))
            cur = next(span_iter, None)
            for i, (s, e) in enumerate(offsets):
                while cur is not None and s >= cur[1]:
                    cur = next(span_iter, None)
                if cur is None:
                    break
                if e > s and s >= cur[0]:
                    labels[i] = ids[i]
        return ids, labels

    def encode(self, record: dict[str, Any]) -> list[Sample]:
        """Encode one ETF record into zero or more samples (see ``overflow``)."""
        if "image" in record:
            raise RenderError("image records belong to text-to-image training")
        if "text" in record:
            return self._encode_text(record["text"])
        if "completion" in record:
            return self._encode_completion(record["prompt"], record["completion"])
        samples = []
        for sub in self.expand(record):
            text, spans = self.render_text(sub)
            ids, labels = self._tokenize(text, spans)
            if len(ids) > self.max_len:
                if self.overflow == "drop":
                    continue
                ids, labels = ids[: self.max_len], labels[: self.max_len]
            sample = Sample(ids, labels)
            if sample.num_trainable:
                samples.append(sample)
        return samples

    def _end_id(self) -> int | None:
        tok = self.tokenizer
        end_id = tok.convert_tokens_to_ids(self.fmt.document_end) if self.fmt.document_end else None
        if end_id is None or end_id == getattr(tok, "unk_token_id", None):
            end_id = tok.eos_token_id
        return end_id

    def _encode_completion(self, prompt: str, completion: str) -> list[Sample]:
        tok = self.tokenizer
        p_ids = list(tok(prompt, add_special_tokens=False)["input_ids"]) if prompt else []
        c_ids = list(tok(completion, add_special_tokens=False)["input_ids"])
        bos = getattr(tok, "bos_token_id", None)
        if self.fmt.document_bos and bos is not None:
            p_ids = [bos] + p_ids
        end = self._end_id()
        if end is not None:
            c_ids.append(end)
        ids = p_ids + c_ids
        labels = [IGNORE_INDEX] * len(p_ids) + c_ids
        if len(ids) > self.max_len:
            if self.overflow == "drop":
                return []
            ids, labels = ids[: self.max_len], labels[: self.max_len]
        sample = Sample(ids, labels)
        return [sample] if sample.num_trainable else []

    def _encode_text(self, text: str) -> list[Sample]:
        tok = self.tokenizer
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
        bos = getattr(tok, "bos_token_id", None)
        if self.fmt.document_bos and bos is not None and (not ids or ids[0] != bos):
            ids.insert(0, bos)
        end_id = self._end_id()
        if end_id is not None and (not ids or ids[-1] != end_id):
            ids.append(end_id)
        out = []
        for i in range(0, len(ids), self.max_len):
            chunk = ids[i : i + self.max_len]
            if len(chunk) >= 16 or i == 0:
                out.append(Sample(chunk, list(chunk)))
        return out

    # -- objectives other than SFT -------------------------------------------

    def encode_preference(self, record: dict[str, Any]) -> PreferencePair | None:
        """Tokenize prompt+chosen and prompt+rejected; only the alternatives carry labels."""
        if not record.get("chosen") or not record.get("rejected"):
            raise RenderError("not a preference record (needs 'chosen' and 'rejected')")
        prompt = [{**m, "train": False} if m["role"] == "assistant" else m for m in record["messages"]]
        base = {k: v for k, v in record.items() if k not in ("chosen", "rejected", "label", "messages", "train_on")}
        encoded = []
        for alt in (record["chosen"], record["rejected"]):
            alt_msgs = [{**m, "train": True} if m["role"] == "assistant" else m for m in alt]
            text, spans = self.render_text({**base, "messages": prompt + alt_msgs})
            ids, labels = self._tokenize(text, spans)
            if len(ids) > self.max_len:
                if self.overflow == "drop":
                    return None
                ids, labels = ids[: self.max_len], labels[: self.max_len]
            if not any(x != IGNORE_INDEX for x in labels):
                return None
            encoded.append((ids, labels))
        return PreferencePair(encoded[0][0], encoded[0][1], encoded[1][0], encoded[1][1])

    def cpt_tokens(self, record: dict[str, Any]) -> list[int]:
        """Token ids of a record as raw training text (for continued pretraining)."""
        tok = self.tokenizer
        bos = getattr(tok, "bos_token_id", None)
        if "text" in record or "completion" in record:
            text = record["text"] if "text" in record else record["prompt"] + record["completion"]
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
            if self.fmt.document_bos and bos is not None and (not ids or ids[0] != bos):
                ids.insert(0, bos)
        elif "messages" in record:
            text, _ = self.render_text({**record, "train_on": "all"})
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
        else:
            raise RenderError("image records cannot be used for continued pretraining")
        end = self._end_id()
        if end is not None and (not ids or ids[-1] != end):
            ids.append(end)
        return ids

    def segments(self, record: dict[str, Any]) -> list[tuple[str, bool]]:
        """Rendered text split into ``(piece, trainable)`` segments, for previews."""
        if "text" in record:
            return [(record["text"], True)]
        if "completion" in record:
            return [(record["prompt"], False), (record["completion"], True)]
        sub = self.expand(record)[-1]
        text, spans = self.render_text(sub)
        out: list[tuple[str, bool]] = []
        pos = 0
        for a, b in spans:
            if a > pos:
                out.append((text[pos:a], False))
            out.append((text[a:b], True))
            pos = b
        if pos < len(text):
            out.append((text[pos:], False))
        return out


def pack_sequences(token_lists: Iterable[list[int]], max_len: int, min_tail: int = 64) -> list[Sample]:
    """Concatenate documents (each ending in EOS) and cut them into ``max_len`` blocks."""
    out: list[Sample] = []
    buf: list[int] = []
    for ids in token_lists:
        buf.extend(ids)
        while len(buf) >= max_len:
            block, buf = buf[:max_len], buf[max_len:]
            out.append(Sample(block, list(block)))
    if len(buf) >= min_tail:
        out.append(Sample(buf, list(buf)))
    return out


def encode_cpt(renderer: Renderer, records: Iterable[dict[str, Any]]) -> tuple[list[Sample], dict[str, int]]:
    counts = {"records": 0, "skipped_render": 0, "documents_tokens": 0}

    def gen():
        for record in records:
            counts["records"] += 1
            try:
                ids = renderer.cpt_tokens(record)
            except RenderError:
                counts["skipped_render"] += 1
                continue
            counts["documents_tokens"] += len(ids)
            yield ids

    samples = pack_sequences(gen(), renderer.max_len)
    counts["samples"] = len(samples)
    return samples, counts


def encode_preferences(renderer: Renderer, records: Iterable[dict[str, Any]]) -> tuple[list[PreferencePair], dict[str, int]]:
    pairs: list[PreferencePair] = []
    counts = {"records": 0, "skipped_render": 0, "skipped_long": 0, "not_preference": 0}
    for record in records:
        counts["records"] += 1
        if not record.get("chosen") or not record.get("rejected"):
            counts["not_preference"] += 1
            continue
        try:
            pair = renderer.encode_preference(record)
        except RenderError:
            counts["skipped_render"] += 1
            continue
        if pair is None:
            counts["skipped_long"] += 1
            continue
        pairs.append(pair)
    counts["pairs"] = len(pairs)
    return pairs, counts


def encode_all(renderer: Renderer, records: Iterable[dict[str, Any]]) -> tuple[list[Sample], dict[str, int]]:
    """Encode many records; returns samples plus counters of what was skipped and why."""
    samples: list[Sample] = []
    counts = {"records": 0, "samples": 0, "skipped_render": 0, "skipped_long": 0}
    for record in records:
        counts["records"] += 1
        try:
            out = renderer.encode(record)
        except RenderError:
            counts["skipped_render"] += 1
            continue
        if not out:
            counts["skipped_long"] += 1
        samples.extend(out)
    counts["samples"] = len(samples)
    return samples, counts
