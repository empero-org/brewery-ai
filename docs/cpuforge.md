# Training on the CPU with cpuforge

Brewery can hand a training job to [cpuforge](https://github.com/Sm0r3s-source/cpuforge), a single
binary that fine-tunes Llama-style models on a CPU. Nothing else changes: the agent, data preparation,
`brewery status|stop`, the STOP file, `final/`, export and publishing all work as for GPU jobs.

Use it when there is no GPU and the model is small. Measured with cpuforge on a 2-core AVX2 machine: a 0.6B model
(Qwen3-0.6B shape, LoRA rank 16) trains at about 22 tokens/s using 3.5 GB of RAM, so ten hours cover roughly 800k
tokens, e.g. 1,000 examples of ~400 tokens for two epochs. A 135M model is about 5x faster. Bigger models are
possible but slow. Run a few steps first to see the speed of your CPU.

## Setup (on the machine that trains)

1. Install cpuforge: download `cpuforge-linux-x64` (or the Windows `.exe`) from the
   [releases page](https://github.com/Sm0r3s-source/cpuforge/releases), make it executable and put it on `PATH`
   (or set `CPUFORGE_BIN` to its location).
2. Install Brewery as usual. No `train` extra is needed: the CPU path never imports torch.
3. Models are downloaded by cpuforge itself from Hugging Face. Gated models need `HF_TOKEN`, which Brewery already
   passes to its worker. To use a folder you already have, set `CPUFORGE_MODEL_DIR=/path/to/model`.

## Using it

Ask for it in the conversation ("train on the CPU"), or set the override when proposing a config:

```json
{"overrides": {"backend": "cpuforge"}}
```

In `job.yaml` this is `runtime: {backend: cpuforge}`. For a remote machine over SSH, install cpuforge there.

## What is supported

| | cpuforge backend |
|---|---|
| Objectives | `sft`, `cpt` |
| Methods | `lora`, `full` (not `qlora`) |
| Models | Llama, Mistral and Qwen3 architectures with safetensors weights (Qwen3 0.6B / 1.7B, Llama 3.2 1B / 3B, SmolLM2, ...) |
| Not yet | DPO, image models, Gemma, MoE, `init_from` (multi-stage), DoRA, LoRA dropout, several GPUs |
| Precision | fp32 on the CPU (the `precision` setting is ignored) |
| Optimizer | AdamW (any `adamw*` name); the scheduler is warmup + cosine |
| `use_rslora` | supported (mapped to an equivalent alpha) |
| `gradient_checkpointing` | on: cpuforge keeps all activations for sequences up to 512 tokens (about 40% faster) and recomputes them above that to save RAM; off: always keeps them. Measured on a 0.6B model: at 1024 tokens keeping everything needed more than 6 GB |

Unsupported settings are refused when the config is proposed, with the reason.

## How it works

`brewery train job.yaml` starts as usual. With `backend: cpuforge` it renders the data with the model's tokenizer and
chat format (the same code the torch worker uses, `train/samples.py`), writes `data/train.tok.jsonl` and
`data/eval.tok.jsonl` plus a `cpuforge.toml` into the run folder, then runs `cpuforge worker --config cpuforge.toml`
and watches it. The worker writes `status.json` and `metrics.jsonl` and, on success, a PEFT adapter in `final/`
(or a full model folder for `method: full`); Brewery adds the tokenizer files and `results.json`.

Stopping writes the STOP file (and SIGTERM is forwarded), the worker saves `state.safetensors` and exits; resuming
continues from that exact step.

## Limits to know about

- The cpuforge numerics are verified against finite differences and a reference implementation of each operation,
  but a full run on a real Qwen3 or Llama checkpoint has not been compared token for token with PyTorch/PEFT.
  Check the first adapter with `brewery test` before relying on it.
- Tokenizer and chat template come from Brewery (not from cpuforge), so special tokens and masks match the GPU path.
