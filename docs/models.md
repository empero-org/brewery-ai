# Supported models

Generated from `src/brewery_ai/models/profiles/*.yaml` by `scripts/gen_models_doc.py`. Every number below is a
guardrail the agent must stay inside (`propose_training_config` rejects values outside the hard bounds unless the
user grants an expert override; leaving the recommended band only produces a warning).

**max_seq_len is sized from your data**: Brewery measures every training set with the model's own tokenizer and
chat template and picks the smallest power of two that fits ~90% of the examples (at least 512, at most the model's
context window from the tables below). 2048 is only the fallback when no measurement is available.

## Gemma 3

Google's Gemma 3 (March 2025), from a tiny 270M model up to 27B. 4B and larger can also see images; Brewery fine-tunes the language part and leaves vision untouched. Repos are gated: accept Google's terms on Hugging Face first.

Licence: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `google/gemma-3-270m-it` | instruct | 0.268 | 32768 | gemma | recommended, gated |
| `google/gemma-3-1b-it` | instruct | 1 | 32768 | gemma | gated |
| `google/gemma-3-4b-it` | instruct | 4.3 | 131072 | gemma | recommended, gated |
| `google/gemma-3-12b-it` | instruct | 12.19 | 131072 | gemma | gated |
| `google/gemma-3-27b-it` | instruct | 27.43 | 131072 | gemma | gated |
| `google/gemma-3-270m` | base | 0.268 | 32768 | gemma | gated |
| `google/gemma-3-1b-pt` | base | 1 | 32768 | gemma | gated |
| `google/gemma-3-4b-pt` | base | 4.3 | 131072 | gemma | gated |
| `google/gemma-3-12b-pt` | base | 12.19 | 131072 | gemma | gated |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | full | **5e-05** (rec. 1e-05–0.0001) [1e-07…0.0001] | – | **2** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | full | **2e-05** (rec. 5e-06–5e-05) [1e-07…0.0001] | – | **1** (rec. 1–2) [0.1…10] | **64** (rec. 32–256) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | full | **5e-07** (rec. 1e-07–5e-06) [1e-08…2e-05] | – | **1** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |

Notes:

- Needs a GPU with bf16 (RTX 30xx/40xx/50xx, A100, H100, L4, ...). On T4/V100 Gemma's activations overflow fp16.
- Gemma 3 has no native tool-calling or thinking format; Brewery teaches tool calls as <tool_call> JSON blocks and reasoning as inline <think> blocks.
- There is no system role: the system prompt is merged into the first user turn by the chat template.
- Terms of Use: derivatives must pass on Google's use restrictions and include a NOTICE file. Brewery's packager writes both.
- Gemma's 262k vocabulary makes the logits large; keep sequence length and batch moderate on small GPUs.

## Gemma 4

Google's Gemma 4 (April 2026): Apache-2.0, native system role, thinking mode and tool calling. E2B/E4B are efficient on-device models, 12B/31B dense, 26B-A4B a mixture of experts. Brewery fine-tunes the language part; vision/audio stay as-is.

Licence: [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | instruct | 5.12 | 131072 | apache-2.0 | recommended |
| `google/gemma-4-E4B-it` | instruct | 8 | 131072 | apache-2.0 | recommended |
| `google/gemma-4-12B-it` | instruct | 11.95 | 262144 | apache-2.0 |  |
| `google/gemma-4-26B-A4B-it` | instruct | 25.8 (3.8 active) | 262144 | apache-2.0 |  |
| `google/gemma-4-31B-it` | instruct | 31.27 | 262144 | apache-2.0 |  |
| `google/gemma-4-E2B` | base | 5.12 | 131072 | apache-2.0 |  |
| `google/gemma-4-E4B` | base | 8 | 131072 | apache-2.0 |  |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–32) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | full | **1e-05** (rec. 2e-06–3e-05) [1e-07…0.0001] | – | **2** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | full | **2e-05** (rec. 5e-06–5e-05) [1e-07…0.0001] | – | **1** (rec. 1–2) [0.1…10] | **64** (rec. 32–256) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–32) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | full | **5e-07** (rec. 1e-07–5e-06) [1e-08…2e-05] | – | **1** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |

Notes:

- Needs a GPU with bf16; FlashAttention-2 does not support Gemma 4's 512-dim global heads, so Brewery uses PyTorch SDPA.
- Tool results render inside the model's own turn; Brewery masks them out of the loss so the model only learns its own calls and answers.
- To keep thinking ability, keep at least ~75% of samples with reasoning traces, or train without thinking entirely.

## Llama 3.x

Meta's Llama 3.1 / 3.2 / 3.3 text models. Very widely supported by tools and runtimes. Repos are gated: accept Meta's license on Hugging Face first.

Licence: [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | instruct | 1.236 | 131072 | llama3.2 | recommended, gated |
| `meta-llama/Llama-3.2-3B-Instruct` | instruct | 3.213 | 131072 | llama3.2 | recommended, gated |
| `meta-llama/Llama-3.1-8B-Instruct` | instruct | 8.03 | 131072 | llama3.1 | recommended, gated |
| `meta-llama/Llama-3.1-70B-Instruct` | instruct | 70.55 | 131072 | llama3.1 | gated |
| `meta-llama/Llama-3.3-70B-Instruct` | instruct | 70.55 | 131072 | llama3.3 | gated |
| `meta-llama/Llama-3.2-1B` | base | 1.236 | 131072 | llama3.2 | gated |
| `meta-llama/Llama-3.2-3B` | base | 3.213 | 131072 | llama3.2 | gated |
| `meta-llama/Llama-3.1-8B` | base | 8.03 | 131072 | llama3.1 | gated |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | full | **1e-05** (rec. 2e-06–3e-05) [1e-07…0.0001] | – | **2** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | full | **2e-05** (rec. 5e-06–5e-05) [1e-07…0.0001] | – | **1** (rec. 1–2) [0.1…10] | **64** (rec. 32–256) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | full | **5e-07** (rec. 1e-07–5e-06) [1e-08…2e-05] | – | **1** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |

Notes:

- License: a published fine-tune must have a name starting with 'Llama', show 'Built with Llama', and ship a copy of the license. Brewery's packager handles all three.
- Llama has no native reasoning format: reasoning traces are taught as inline <think>...</think> blocks (or dropped, if you prefer).
- The Llama chat template allows one tool call per assistant turn; traces with parallel calls are skipped.
- Base (non-instruct) models have untrained chat-control tokens. Brewery initialises them to the mean embedding before training.

## Qwen3

Dense Qwen3 models (April 2025, plus the 2507 refresh). Strong all-rounders with a "hybrid thinking" mode: the same model can answer with or without a <think> block.

Licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | instruct | 0.6 | 40960 | apache-2.0 | recommended |
| `Qwen/Qwen3-1.7B` | instruct | 1.72 | 40960 | apache-2.0 |  |
| `Qwen/Qwen3-4B` | instruct | 4.02 | 40960 | apache-2.0 |  |
| `Qwen/Qwen3-4B-Instruct-2507` | instruct | 4.02 | 262144 | apache-2.0 | recommended |
| `Qwen/Qwen3-4B-Thinking-2507` | thinking | 4.02 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3-8B` | instruct | 8.19 | 40960 | apache-2.0 | recommended |
| `Qwen/Qwen3-14B` | instruct | 14.77 | 40960 | apache-2.0 |  |
| `Qwen/Qwen3-32B` | instruct | 32.76 | 40960 | apache-2.0 |  |
| `Qwen/Qwen3-0.6B-Base` | base | 0.6 | 32768 | apache-2.0 |  |
| `Qwen/Qwen3-1.7B-Base` | base | 1.72 | 32768 | apache-2.0 |  |
| `Qwen/Qwen3-4B-Base` | base | 4.02 | 32768 | apache-2.0 |  |
| `Qwen/Qwen3-8B-Base` | base | 8.19 | 32768 | apache-2.0 |  |
| `Qwen/Qwen3-14B-Base` | base | 14.77 | 32768 | apache-2.0 |  |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | full | **1e-05** (rec. 2e-06–3e-05) [1e-07…0.0001] | – | **2** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | full | **2e-05** (rec. 5e-06–5e-05) [1e-07…0.0001] | – | **1** (rec. 1–2) [0.1…10] | **64** (rec. 32–256) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | full | **5e-07** (rec. 1e-07–5e-06) [1e-08…2e-05] | – | **1** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |

Notes:

- Hybrid thinking: answers without reasoning get an empty <think></think> block from the chat template; Brewery excludes it from the loss.
- To keep thinking ability, keep at least ~75% of samples with reasoning traces; training only on plain answers weakens thinking mode.
- Multi-turn conversations with reasoning in several turns are split into one sample per turn, because the template drops earlier reasoning.

## Qwen3.5

Qwen3.5 dense models (Feb 2026) and architecture-identical successors (Qwen3.6, Qwen3.8, Empero distills). Hybrid attention: 3 of every 4 layers use Gated DeltaNet linear attention, so long contexts are cheap. 262K native context, thinking by default (>=4B).

Licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | instruct | 0.75 | 262144 | apache-2.0 | recommended |
| `Qwen/Qwen3.5-2B` | instruct | 1.88 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-4B` | instruct | 4.21 | 262144 | apache-2.0 | recommended |
| `Qwen/Qwen3.5-9B` | instruct | 8.95 | 262144 | apache-2.0 | recommended |
| `Qwen/Qwen3.5-27B` | instruct | 26.9 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.6-27B` | instruct | 26.9 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.8-27B` | instruct | 26.9 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-0.8B-Base` | base | 0.75 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-2B-Base` | base | 1.88 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-4B-Base` | base | 4.21 | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-9B-Base` | base | 8.95 | 262144 | apache-2.0 |  |
| `empero-ai/Qwen3.8-9B-Distill` | instruct | 8.95 | 262144 | apache-2.0 | Empero |
| `empero-ai/Qwen3.8-4B-Distill` | instruct | 4.21 | 262144 | apache-2.0 | Empero |
| `empero-ai/Qwen3.8-2B-Distill` | instruct | 1.88 | 262144 | apache-2.0 | Empero |
| `empero-ai/Qwythos-9B-v2` | instruct | 8.95 | 1048576 | apache-2.0 | Empero |
| `empero-ai/Qwythos-27B-v1` | instruct | 26.9 | 1048576 | apache-2.0 | Empero |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–64) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | full | **1e-05** (rec. 2e-06–3e-05) [1e-07…0.0001] | – | **2** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | full | **2e-05** (rec. 5e-06–5e-05) [1e-07…0.0001] | – | **1** (rec. 1–2) [0.1…10] | **64** (rec. 32–256) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–64) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | full | **5e-07** (rec. 1e-07–5e-06) [1e-08…2e-05] | – | **1** (rec. 1–3) [0.1…10] | **32** (rec. 16–128) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |

Notes:

- Brewery trains the text model only: the vision encoder and the multi-token-prediction head are dropped on load, so exports are text-only Qwen3_5ForCausalLM checkpoints.
- The default LoRA targets include the Gated DeltaNet projections (in_proj_qkv, in_proj_z, out_proj), which make up 75% of the token-mixing layers. 'classic' targets only attention + MLP like Unsloth.
- Install flash-linear-attention on the training machine (Brewery does it automatically). Without it the linear-attention layers fall back to a much slower PyTorch path.
- Tool calls use Qwen's XML format and tool arguments must be JSON objects.
- Small Qwen3.5 repos ship no generation_config.json; Brewery writes one with the right stop tokens when exporting.

## Qwen3.5 MoE

Sparse mixture-of-experts Qwen3.5 (35B total, ~3.5B active per token). Fast inference for its quality, but every expert has to sit in GPU memory while training.

Licence: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-35B-A3B` | instruct | 34.66 (3.46 active) | 262144 | apache-2.0 | recommended |
| `Qwen/Qwen3.6-35B-A3B` | instruct | 34.66 (3.46 active) | 262144 | apache-2.0 |  |
| `Qwen/Qwen3.5-35B-A3B-Base` | base | 34.66 (3.46 active) | 262144 | apache-2.0 |  |
| `empero-ai/Qwen3.8-35B-A3B-Distill` | instruct | 34.66 (3.46 active) | 262144 | apache-2.0 | Empero |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0002** (rec. 5e-05–0.0003) [1e-06…0.001] | **16** (rec. 8–32) | **2** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | max_seq_len **auto**, rec. 512–8192 |
| sft | qlora | not allowed | | | | QLoRA keeps the frozen base model in 4-bit NF4. It needs an NVIDIA GPU (bitsandbytes) and is slower than LoRA. bitsandby |
| sft | full | not allowed | | | | Full fine-tuning updates every weight: best quality ceiling, highest memory, easiest to overfit. Full fine-tuning a 35B  |
| cpt | lora | **0.0001** (rec. 3e-05–0.0002) [1e-06…0.001] | **64** (rec. 16–128) | **1** (rec. 1–2) [0.1…10] | **32** (rec. 16–128) [1…1024] | max_seq_len **auto**, rec. 1024–8192 |
| cpt | qlora | not allowed | | | | QLoRA keeps the frozen base model in 4-bit NF4. It needs an NVIDIA GPU (bitsandbytes) and is slower than LoRA. bitsandby |
| cpt | full | not allowed | | | | Full fine-tuning updates every weight: best quality ceiling, highest memory, easiest to overfit. Full fine-tuning a 35B  |
| dpo | lora | **5e-06** (rec. 1e-06–5e-05) [1e-07…0.0001] | **16** (rec. 8–32) | **1** (rec. 1–3) [0.1…10] | **16** (rec. 8–64) [1…1024] | β **0.1** (rec. 0.05–0.5) [0.01…1]; max_seq_len **auto**, rec. 512–8192 |
| dpo | qlora | not allowed | | | | QLoRA keeps the frozen base model in 4-bit NF4. It needs an NVIDIA GPU (bitsandbytes) and is slower than LoRA. bitsandby |
| dpo | full | not allowed | | | | Full fine-tuning updates every weight: best quality ceiling, highest memory, easiest to overfit. Full fine-tuning a 35B  |

Notes:

- LoRA trains attention, the Gated DeltaNet projections and the shared expert; the router and the 256 routed experts stay frozen (the stable, recommended setup).
- Expert LoRA (PEFT target_parameters) is available as an expert option but is experimental and multiplies memory use.
- Needs a single GPU with at least 80 GB (A100/H100 80GB, H200, B200, RTX PRO 6000 96GB): all 35B weights are loaded in bf16.
- Brewery trains the text model only; exports are text-only checkpoints without the vision encoder and MTP head.

## Qwen-Image

Qwen-Image 2.1 (September 2026): a 7B single-stream diffusion transformer with a Qwen3-VL-8B text encoder. Excellent at rendering text inside images, outputs RGBA. Brewery trains LoRA adapters that teach it a style, a character or an object.

Licence: [Qwen Research License (non-commercial)](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

| Model | Kind | Params (B) | Context | Licence | Notes |
|---|---|---|---|---|---|
| `Qwen/Qwen-Image-2.1` | image | 7.115 | – | qwen-research | recommended |

Guidelines (first variant; individual variants may override):

| Objective | Method | Learning rate | Rank | Epochs / steps | Effective batch | Extra |
|---|---|---|---|---|---|---|
| sft | lora | **0.0001** (rec. 5e-05–0.00015) [1e-06…0.0003] | **16** (rec. 8–64) | **1000** (rec. 300–2500) [50…10000] | **1** (rec. 1–8) [1…64] | resolution **1024** |
| sft | qlora | **0.0001** (rec. 5e-05–0.00015) [1e-06…0.0003] | **16** (rec. 8–64) | **1000** (rec. 300–2500) [50…10000] | **1** (rec. 1–8) [1…64] | resolution **1024** |
| sft | full | not allowed | | | | Full fine-tuning of image models is not supported in this version; use LoRA. |

Notes:

- LICENSE: Qwen-Image 2.1 is NON-COMMERCIAL (Qwen Research License). LoRAs trained on it, and images made with them, may not be sold or used commercially. Published LoRAs must show 'Built with Qwen' and include the license.
- Use 10-50 good, varied images. Give every image a caption that describes it, plus one unique trigger word (e.g. 'sks corgi').
- Less is more: outputs start to drift after ~1000 steps at lr 1e-4, and lr 3e-4 visibly damages unrelated prompts. Check samples every ~250 steps.
- Needs an NVIDIA GPU with bf16 and 24 GB+. On 24 GB cards use QLoRA (4-bit transformer) or train at 512px.
- Never train on photos of real people without their permission, and never on anyone under 18.

