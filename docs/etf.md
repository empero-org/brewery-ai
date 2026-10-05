# ETF — Empero Trace Format (v1)

ETF is Brewery's native training-data layout. Files are plain **`.jsonl`**: one JSON object (a *record*) per line.
"ETF" is just the name of the layout — there is no special file extension.

Design goals:

- **Everything optional except the essentials.** A record needs exactly one field that defines its kind; every other
  field may be left out. Small datasets stay small, rich agent traces fit too.
- **One format for every objective.** The same file can feed SFT, continued pretraining (CPT) and preference tuning
  (DPO); Brewery picks what each objective needs.
- **Model-agnostic.** ETF never contains chat-template tokens. Brewery renders records with each model's own official
  chat template at training time, so one dataset works for Qwen, Llama and Gemma alike.
- **Lenient in, canonical out.** Common variants (OpenAI/ShareGPT message shapes, `<think>` tags, string-encoded tool
  arguments, `developer` roles, …) are accepted and normalised.

The machine-readable definition is [`src/brewery_ai/etf/etf.schema.json`](../src/brewery_ai/etf/etf.schema.json)
(`brewery etf schema` prints it). `brewery etf validate FILE` checks a file and explains every problem.

## Record kinds

| Kind | Defining field | Used for |
|---|---|---|
| **trace** | `messages` | conversations: SFT, DPO (with `chosen`/`rejected`), CPT (rendered as text) |
| **text** | `text` | raw documents: CPT (also usable in SFT as full-loss text) |
| **completion** | `completion` (+ optional `prompt`) | raw prompt → continuation without a chat template (base models, code completion, formats) |
| **image** | `image` | text-to-image LoRA training (image + caption) |

A record with more than one defining field is invalid.

### Fields on every record

| Field | Type | Meaning |
|---|---|---|
| `id` | string / int | your identifier |
| `meta` | object | free-form metadata; Brewery reads `source`, `license`, `lang`, `synthetic`, `tags` |
| `repeat` | int ≥ 1 | use this record N times (upsampling) |
| `etf` | `1` | optional version marker |

## Traces

```json
{"system": "You are Captain Byte, a pirate who teaches maths.",
 "messages": [
   {"role": "user", "content": "What is 7 x 8?"},
   {"role": "assistant", "reasoning": "7 x 8 = 56.", "content": "Arr, that be 56, matey!"}],
 "meta": {"source": "handwritten", "license": "cc-by-4.0"}}
```

### Messages

| Field | Roles | Meaning |
|---|---|---|
| `role` | all | `system`, `user`, `assistant`, `tool` (aliases accepted: `developer`→system, `human`/`prompter`→user, `gpt`/`model`/`bot`→assistant, `function`/`observation`/`ipython`→tool) |
| `content` | all | a string, or a list of parts (see *Multimodal content*). May be empty for an assistant turn that only calls tools |
| `name` | all | speaker name for multi-party chats; for tool messages, the tool's name |
| `reasoning` | assistant | the model's thinking before the answer. `<think>…</think>` at the start of `content`, `reasoning_content` and `thinking` are understood too |
| `tool_calls` | assistant | list of calls (several = parallel tool use) |
| `tool_call_id` | tool | which call this result answers (filled in automatically when omitted) |
| `is_error` | tool | the tool failed |
| `train` | assistant | `false` = context only: no loss on this turn (e.g. few-shot examples) |
| `weight` | assistant | loss weight; `0` = no loss (v1 trainers treat any weight > 0 as 1) |

System messages may appear anywhere. The first one becomes the system prompt; later ones are kept for templates that
support them and otherwise merged into the next user turn as a `[System note]`. The `system` field is a shorthand for
a leading system message.

### Tools

```json
{"tools": [{"name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            "returns": {"type": "object", "properties": {"temp_c": {"type": "number"}}}}],
 "messages": [
   {"role": "user", "content": "Weather in Paris?"},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}}]},
   {"role": "tool", "tool_call_id": "c1", "content": {"temp_c": 21}},
   {"role": "assistant", "content": "It's 21 °C in Paris."}]}
```

- Tool schemas: `name`, `description`, `parameters` (JSON Schema), optional `returns`, `strict`, `examples`, `type`
  (omit for function tools). OpenAI's `{"type": "function", "function": {...}}` wrapper is accepted.
- Tool calls: `id` (generated if missing), `name`, `arguments` (object preferred; JSON strings are parsed).
- Tool results: `content` may be a string or structured JSON (it is serialised for the template).
- `tool_choice` is kept for reference.
- Models whose templates have no tool format (e.g. Gemma 3) learn tools through a compact `<tool_call>{json}</tool_call>`
  convention described in the system prompt; Llama templates allow one call per turn (records with parallel calls are
  skipped for Llama).

### Multimodal content

```json
{"role": "user", "content": [
  {"type": "text", "text": "What breed is this?"},
  {"type": "image", "image": "images/dog_01.jpg"}]}
```

Part types: `text`, `image`, `audio`, `video`, `file` (with `mime`, `format`, `name` where useful). Paths are relative
to the `.jsonl` file; URLs and data URIs are allowed. OpenAI's `image_url` / `input_audio` parts are accepted. Text-only
trainers (all v1 text trainers) use the text parts; media-only user turns are skipped.

### RAG documents

```json
{"documents": [{"title": "Brewing 101", "text": "Hops add bitterness and aroma."}],
 "messages": [{"role": "user", "content": "What do hops do?"},
              {"role": "assistant", "content": "They add bitterness and aroma."}]}
```

Documents are passed to templates that support them and otherwise added to the system prompt.

### Chat-template options

`template_kwargs` passes options to the model's chat template for this record, e.g. `{"enable_thinking": false}` or
`{"reasoning_effort": "low"}`. Brewery sets `enable_thinking` automatically from whether a record has reasoning.

### Loss control

| Mechanism | Effect |
|---|---|
| default | loss on every assistant turn (plus its end-of-turn token) |
| `"train": false` on an assistant message | that turn is context only |
| `"train_on": "last"` | only the final assistant turn is trained |
| `"train_on": "all"` | every token is trained (CPT-style) |

When a model's template drops reasoning from earlier turns (Qwen3, Qwen3.5, Gemma 4), a multi-turn trace with
reasoning in several turns is split into one sample per turn, so every reasoning trace is trained in exactly the form
the model sees at inference time. Empty reasoning blocks that templates insert are excluded from the loss.

### Preference pairs (DPO)

```json
{"messages": [{"role": "user", "content": "Tell me a joke about yeast."}],
 "chosen": "Why did the yeast break up with the dough? It needed more space to rise.",
 "rejected": "No."}
```

`chosen` and `rejected` are continuations of `messages`: a string, an assistant message, or a list of messages (e.g. an
answer that uses tools). For SFT, Brewery trains on `messages` + `chosen`; for DPO, on the pair. `label` (bool) is
reserved for unpaired preference methods.

## Text documents (CPT)

```json
{"text": "Chapter 1. The brewery stood at the edge of the village…", "meta": {"source": "local:novel.txt"}}
```

For CPT, documents are tokenised, separated by the end-of-text token and packed into full-length sequences.

## Raw completions

```json
{"prompt": "def fibonacci(n):", "completion": "\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"}
```

No chat template is applied; the loss covers the completion (plus an end-of-text token).

## Images

```json
{"image": "images/0001.png", "caption": "sks corgi sitting on a red sofa, studio light",
 "captions": ["sks corgi on a sofa", "a photo of sks corgi indoors"],
 "references": ["images/ref_0001.png"]}
```

- `image`: path relative to the `.jsonl`, absolute path, or URL.
- `caption` (or several `captions`: one is chosen at random per training step).
- `references`: condition images (reserved for edit training).

## Validation

`brewery etf validate FILE` reports errors (the record is skipped) and warnings (the record is used). Errors include:
no trainable assistant turn, a tool result without a preceding tool call, empty user turns, unknown roles, records with
no or several defining fields. Warnings include: conversations that don't start with a user turn, unanswered tool calls,
repeated roles (merged for strict templates), trailing non-assistant messages (dropped), empty captions.

## Converting other datasets

`brewery etf convert SOURCE --out data.jsonl` (or the agent's `import_dataset` tool) detects common layouts:
OpenAI-style `messages`, ShareGPT `conversations` (incl. `function_call`/`observation` turns), Alpaca
`instruction/input/output`, prompt/completion and question/answer pairs, preference `chosen/rejected` (including
Anthropic-HH strings), and plain `text`. A mapping can rename columns and use Python format templates:

```json
{"layout": "prompt_completion", "prompt_template": "Translate to German: {en}", "completion": "de", "system_text": "You are a translator."}
```
