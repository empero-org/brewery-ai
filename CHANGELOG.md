# Changelog

## Unreleased

**Fixed**
- Hardware detection uses ROCm PyTorch to report AMD GPUs, VRAM and BF16 support on Windows without `rocm-smi`.
- When a GPU operation check fails or is incomplete, Homebrew blocks local GPU training.
- File operations and SSH commands handle paths with spaces, Unicode and Windows backslashes.
- Project-relative paths and worker fingerprints use portable separators and consistent ordering.

**Changed**
- Local commands use the Windows system shell on Windows and Bash or `sh` on POSIX systems.

**Added**
- Windows ROCm setup instructions include tested dependency constraints that preserve the AMD PyTorch build.

## 0.1.0 — 2026-10-05 · first public release

**The brewmaster**
- Guided agent REPL with four experience levels (Beginner, Hobbyist, Builder, Expert), a phase-based workflow
  (goal → compute → base model → data → recipe → training → taste test → bottling → sharing) and 41 tools.
- Guiding AI backends: Claude (Anthropic SDK; adaptive thinking, server-side fallbacks, refusal handling) and any
  OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio, vLLM, llama.cpp), with a text tool protocol for
  models without function calling. Capability tiers adapt tools and guidance to the model.
- Chat UI: framed input bar with a status footer (model · level · phase · tokens · context), shaded user turns, the
  model's thinking in a separate dim block (`/thinking on|full|off`), line editing and history, numbered menus in
  narrow or embedded terminals.

**Training**
- Text model families: Qwen3, Qwen3.5 (+ Qwen3.6/3.8 and Empero distills), Qwen3.5 MoE, Llama 3.1/3.2/3.3,
  Gemma 3, Gemma 4. Methods: LoRA, QLoRA, full fine-tuning.
- Objectives CPT, SFT and DPO, chainable into regimes (e.g. CPT → SFT → DPO); DPO without a second model in memory.
- Image LoRAs for Qwen-Image 2.1, with preview images rendered during training (a "before" set, then one per
  checkpoint) from the model in memory, a local one-page preview gallery (`/gallery`), and packaging of any kept
  checkpoint.
- Per-model hyperparameter guidelines; recipes are filled and sized to the GPU, with memory, time and cost estimates.
- Runs locally or on any SSH server (Runpod and Vast.ai guides): versioned worker sync, detached jobs, live status,
  clean stops (STOP file, works with torchrun and on Windows), rank-aware multi-GPU text training.

**Data**
- ETF (Empero Trace Format) v1: one `.jsonl` format for chats, system prompts, tools, reasoning, documents,
  preference pairs, raw text, completions and images; JSON Schema, validator and converters.
- Hugging Face search, inspection and import (text and images), local files, hand-written examples, `clean_dataset`.
- Opt-in synthetic data (create / transform / preference) in parallel batches with live progress, saved as each
  batch finishes; adjustable reasoning effort. AI captioning; preference collection from your model's own samples.

**Sharing**
- Model cards with training regime, data, licence notices, usage code and credits; licence compliance (Llama
  naming, Gemma terms, non-commercial notices); Hugging Face upload (private by default, never into an existing
  public repo by accident).

**Safety**
- Everything that costs money, installs software, sends data elsewhere or publishes asks for confirmation inside
  Homebrew. SSH commands are parsed with an allow-list (no local command execution through ssh options); the HF
  token only goes to a server when a model needs it; secrets never pass through the chat.
