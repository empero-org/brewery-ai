# Architecture

Homebrew has two halves that share one package (`homebrew_ai`) but never import each other's heavy dependencies:

```text
┌──────────────── control plane (your laptop) ────────────────┐        ┌────────── worker (GPU machine) ──────────┐
│ ui/        terminal REPL, prompts, live training view        │        │ train/runner.py  CPT / SFT / DPO jobs     │
│ agent/     loop, system prompt, levels, phases, 41 tools     │  SSH   │ train/image.py   Qwen-Image 2.1 LoRA       │
│ backends/  Claude (Anthropic SDK), OpenAI-compatible         │ ─────▶ │ train/dpo.py     DPO loss + reference pass │
│ data/      HF search/inspect/import, images, synthetic data  │  tar   │ train/infer.py   test prompts, candidates  │
│ jobs/      launch, follow, stop, fetch (local or SSH)        │        │ train/export.py  adapter/merged export,    │
│ remote/    SSH targets, server bootstrap, rental guides      │        │                  Hugging Face upload       │
│ package/   model cards, licence rules                        │        │ status.json / metrics.jsonl / train.log    │
└──────────────────────────────────────────────────────────────┘        └───────────────────────────────────────────┘
          shared and light: etf/ (format, converters, renderer), models/ (profiles + guidelines), train/config.py
```

## Control plane

- **Agent loop** (`agent/loop.py`): user message → model → tool calls → results → … until the model answers. History
  is **append-only**. The system prompt is fixed for the session, and the live project state travels in a
  `<homebrew_state>` block on new user/tool messages. This keeps prompt caches warm and Claude's thinking blocks valid
  (they are replayed verbatim). Long sessions are compacted by summarising the whole conversation into one message.
- **Tools** (`agent/tools/`): validated against their JSON schemas before running. Side effects are confirmed by the
  user through the UI inside the tool (money, installs, uploads, sending images to an AI provider, writing SSH keys,
  shell commands). Expert-only tools are gated by level at execution time, so the tool list never changes mid-session.
- **Backends** (`backends/`): Claude uses adaptive thinking with `display: "updates"` (progress notes shown between
  tool calls), effort control, refusal handling, server-side fallbacks, eager tool-input streaming and prompt caching.
  Optional features switch themselves off if an API rejects them. The OpenAI-compatible backend covers OpenAI,
  OpenRouter, Ollama, LM Studio, vLLM and llama.cpp. It falls back to a text tool protocol for models without
  function calling, and understands `<tool_call>` blocks that local servers leave in plain text.
- **Capability tiers** (`backends/presets.py`): strong models see all tools; small local models get phase-scoped tools,
  stricter instructions and a smaller context budget.

## Shared core

- **ETF** (`etf/`): schema/normalisation, converters from common dataset layouts, and the **renderer**. The renderer
  adapts a trace to each family's dialect (reasoning field, tool-call shape, strict alternation, mid-conversation
  system notes, documents) and renders it with the model's official chat template. It finds assistant turns between the
  family's header and end-of-turn markers, excludes template-inserted empty reasoning blocks and in-turn tool responses
  (Gemma 4), and tokenises once with offsets to build labels. Templates that drop earlier reasoning trigger per-turn
  splitting, which is detected automatically by probing the template.
- **Model profiles** (`models/profiles/*.yaml`): loading options, chat format, stop tokens, LoRA target presets,
  sampling, licence obligations and **guidelines**: hard bounds, defaults and recommended bands per method (LoRA, QLoRA,
  full), with per-objective overrides (CPT, DPO). Jobs embed a snapshot of the profile, so the worker never depends on
  the control plane's registry (custom profiles work remotely too).
- **Jobs** (`train/config.py`): `build_job` fills a `TrainJob` from guidelines and overrides, sizes micro-batch and
  gradient accumulation to the GPU with the memory estimator (`hardware/estimate.py`), and validates it.

## Worker

- `homebrew train job.yaml` runs one job; `homebrew chain a/job.yaml b/job.yaml …` runs stages back-to-back.
- A stage with `init_from` starts from the previous stage's result. Adapters are merged into that stage's own starting
  weights first, recursively, and cached in `<run>/merged`.
- SFT uses masked samples, CPT packs documents into full-length blocks, DPO tokenises chosen/rejected pairs and
  precomputes reference log-probs with the untrained starting model.
- Progress goes to `status.json` (state, step, loss, ETA, GPU memory, DPO reward accuracy, errors with a hint) and
  `metrics.jsonl`. SIGTERM saves a checkpoint and stops cleanly.

## Remote execution

- Targets (`remote/target.py`) share one interface for local runs and SSH. File transfer streams a tar archive through
  SSH, so it works through any SSH server that allows commands (no rsync or SFTP needed). Runpod's proxy SSH is
  detected and refused, with instructions to use "SSH over exposed TCP".
- `prepare_remote` uploads the worker package, creates a virtualenv that reuses the image's PyTorch (installing a
  CUDA-matched build only if needed) and installs the training libraries. It keeps everything under
  `/workspace/homebrew` when a persistent volume exists.
- The Hugging Face token is passed to remote processes over **stdin** and exported into the process environment; it is
  never written to the remote disk or placed on a command line.
