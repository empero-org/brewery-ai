# 🍺 Homebrew

**Brew your own AI model.** Homebrew is a guided agent for fine-tuning language and image models. You chat with the
*brewmaster* — an AI of your choice (Claude, any OpenAI-compatible model, or a local model) — and it walks you through
everything: what you want the model to do, where to train it, which base model fits, finding and preparing data,
picking safe hyperparameters, running the training (on your GPU or a rented server), testing the result, and
publishing it on Hugging Face with a proper model card.

It is built for everyone from complete beginners to experts who just want the busywork automated.

<sub>Made by [Empero](https://empero.org) — independent AI research lab, open by default. · [GitHub](https://github.com/empero-org) · [Hugging Face](https://huggingface.co/empero-ai)</sub>

---

## Brewed with Homebrew

Two demo models, each made start to finish in one guided session on a single rented GPU (an RTX PRO 5000 on
Vast.ai), with `xiaomi/mimo-v2.6-pro` on OpenRouter as the brewmaster:

| Model | What it is | How it was brewed |
|---|---|---|
| [**Homebrew-Qwen3.5-2B-Grandmas-Kitchen**](https://huggingface.co/empero-ai/Homebrew-Qwen3.5-2B-Grandmas-Kitchen) | Qwen3.5 2B that writes home-cooking recipes in a warm, chatty grandma voice | public recipe dataset → cleaned → rewritten in the grandma voice with opt-in synthetic data → LoRA SFT (826 examples, ~2 epochs) |
| [**Homebrew-Qwen-Image-2.1-Y2K**](https://huggingface.co/empero-ai/Homebrew-Qwen-Image-2.1-Y2K) | Qwen-Image 2.1 LoRA for the early-2000s digicam snapshot look (trigger `y2kphoto`) | Hugging Face image dataset (160 photos with captions) + trigger word → LoRA with preview images at every checkpoint → the step-500 checkpoint picked from the previews |

The model cards (training regime, data, licences, credits, before/after samples) were written by Homebrew too.

## What it can do

- **Talks at your level.** Pick 🌱 Beginner, 🍺 Hobbyist, 🛠️ Builder or 🧪 Expert; explanations, questions and the
  amount of detail adapt.
- **Text models:** Qwen3, Qwen3.5 (plus Qwen3.6/3.8 and Empero's Qwen3.8 / Qwythos models on the same architecture),
  Qwen3.5 MoE, Llama 3.1/3.2/3.3, Gemma 3, Gemma 4 — with **LoRA, QLoRA or full fine-tuning**.
- **Training objectives you can chain into regimes:** continued pretraining (**CPT**) on raw text, supervised
  fine-tuning (**SFT**) on conversations, and preference tuning (**DPO**). For example CPT → SFT → DPO on a base model,
  plain SFT on an instruct model, or SFT → collect preferences → DPO.
- **Image models:** LoRA adapters for **Qwen-Image 2.1** (styles, characters, objects).
- **Guardrails, not guesswork.** Every supported model ships with hyperparameter guidelines (per objective and method).
  The agent can only propose recipes inside them; memory, time and cost are estimated before anything runs.
- **Data from anywhere:** Hugging Face search and preview without downloading, local files, examples you write
  together, opt-in synthetic data shaped to your request, and preference collection where you pick the better of two
  answers from your own model.
- **ETF (Empero Trace Format):** one `.jsonl` format for chats, system prompts, tool schemas and calls, reasoning, RAG
  documents, preference pairs, raw text, completions and images. See [docs/etf.md](docs/etf.md).
- **Train anywhere:** your own NVIDIA GPU, or any SSH server. Step-by-step guides for renting on **Runpod** and
  **Vast.ai** (with price estimates). Homebrew prepares the server and runs jobs detached; it follows progress live.
- **Bottle and share:** export adapters or merged models, a generated model card (training regime, data, licences,
  credits), licence compliance (Llama naming, Gemma terms, non-commercial notices) and upload to Hugging Face.

## Quick start

```bash
pip install "homebrew-ai @ git+https://github.com/empero-org/homebrew-ai"
```

```bash
homebrew
```

On the first run Homebrew asks which AI should guide you and for its API key (stored only on your computer, readable
only by you). Then it asks your experience level and a project name, creates a project folder and the conversation
begins:

```text
● brewmaster
  Ahoy! I'm here to help you brew your own AI model. Tell me what you'd like it to do —
  chat in a certain style, know about a topic, use tools, draw in a style…

 › a chatbot that explains chemistry like a friendly pirate

● brewmaster
  ✻ thinking
    No GPU on this laptop; a 2-4B model with LoRA fits a rented 24 GB card…
  ⚙ checking hardware…
  Your laptop has no NVIDIA GPU, so we'll rent one for about an hour (~$0.40 on a 24 GB card)…
```

Training happens on the machine with the GPU, so the laptop only needs the light control-plane install. If you have an
NVIDIA GPU and want to train locally, also install the training extras:

```bash
pip install "homebrew-ai[train] @ git+https://github.com/empero-org/homebrew-ai"
```

Close the terminal at any time: training keeps running, and `homebrew` in the project folder picks up where you left off.

## How a brew works

| Phase | What happens |
|---|---|
| 🎯 Goal | the brewmaster asks what the model should do until it's concrete |
| 🖥️ Compute | hardware detection; your GPU, or a rented server (guided), prepared automatically over SSH |
| 🧠 Base model | a model that fits the goal, the hardware and your licence needs; gated access is checked |
| 🌾 Ingredients | find, import, write or generate data; preview exactly what the model will learn; build training sets |
| ⚙️ Recipe | `propose_training_config` fills every hyperparameter from the model's guidelines and sizes it to the GPU |
| 🔥 Brewing | training runs detached; live progress with loss curve and ETA; failures come with a plain-language fix |
| 👅 Taste test | try prompts against the new model (side by side with the base); optionally collect preferences for DPO |
| 🍾 Bottling | adapter or merged export, model card, licence files |
| 🚀 Sharing | upload to Hugging Face (private by default), and a reminder to switch off rented servers |

Anything that costs money, installs software, sends your data elsewhere or publishes something asks for your
confirmation inside Homebrew. The AI cannot skip these prompts. Passwords, API keys and tokens never go through the chat.

## Choosing the guiding AI

| Preset | Notes |
|---|---|
| Claude (Anthropic) | best guidance (`claude-opus-5-5` default; Sonnet 5.5 and Haiku 4.5 selectable) |
| OpenAI | any chat model with tool calling |
| OpenRouter | many models behind one key |
| Ollama / LM Studio | free and local; use a 14B+ model with tool calling for good results |
| Custom | any OpenAI-compatible server (vLLM, llama.cpp, …) |

Homebrew adapts to the model: strong models get the full toolset, small local models get step-by-step guidance and
phase-scoped tools. Models without native tool calling are driven through a text protocol automatically. Change the
guide any time with `homebrew setup`.

## Regimes: CPT → SFT → DPO

Each training job has an **objective** (`cpt`, `sft`, `dpo`) and can start from a previous stage (`init_from`). LoRA
stages are merged into their starting weights before the next stage begins, so stages stack cleanly:

- **SFT** — the common case: an instruct model learns your behaviour, style, format or tools.
- **CPT → SFT** — teach a base model a domain or language from raw text, then how to chat.
- **SFT → DPO** — sharpen preferences. Preference pairs can come from Hugging Face, from synthetic generation, or from
  your model itself: Homebrew samples two answers per prompt (`generate_candidates`) and you pick the better one in
  the terminal (`review_candidates`), or let the guiding AI judge with a rubric and spot-check its verdicts.

DPO uses reference log-probabilities computed once from the stage's starting model, so it needs no second model in
memory, for LoRA and full fine-tuning alike.

## Watching an image LoRA learn

Qwen-Image LoRA runs render preview images while they train: a "before" set at step 0 (the base model), then one
set with every checkpoint, always with the same seeds so you can compare. While you watch, Homebrew downloads each
new set to `runs/<job>/samples/step_NNNNNN/` and prints the folder; ask the brewmaster for specific preview prompts
(include the trigger word) or a different interval. Every preview set has a saved checkpoint next to it, so if
step 500 looks better than the end, ask the brewmaster to package that checkpoint instead.

For a side-by-side view, type `/gallery` (or ask the brewmaster): Homebrew serves a one-page gallery on your own
computer (`http://127.0.0.1:8765/<random token>/`, never reachable from outside). Rows are checkpoints, columns are
the preview prompts; it refreshes while training runs and downloads new sets from the server by itself. Click a
picture to enlarge it, use ←/→ to walk through the checkpoints for one prompt, and C to compare with the base model.

## Making data

Most brews start from a Hugging Face dataset, your own files, or examples you write with the brewmaster. Synthetic
data is opt-in (Homebrew shows your provider's terms first) and always shaped to your request: write new
conversations from a brief, rewrite an existing dataset (for example "answer like a cozy grandma"), or create
preference pairs for DPO. Batches run in parallel with a live progress line and are saved as they finish, so a
cancelled run keeps its work. Reasoning models are asked not to think for simple rewrites (several times faster in
our tests). `clean_dataset` drops empty, duplicate or broken records (with a backup).

## Renting a GPU

Homebrew never creates accounts or spends money for you. It recommends a GPU that fits (with approximate Runpod and
Vast.ai prices), creates an SSH key, and walks you through renting. Then you paste the provider's SSH command and
Homebrew takes over: hardware check, environment setup, data upload, training, results. Billing notes are part of the
guide. Stopped pods and instances still bill for storage, so destroy them when you're done.

## Command line

| Command | Purpose |
|---|---|
| `homebrew` | start or resume the guided session in the current project |
| `homebrew new NAME` | new project |
| `homebrew setup` | choose/change the guiding AI |
| `homebrew doctor` | check installation, guiding AI, Hugging Face login, hardware |
| `homebrew hardware [--ssh "ssh …"]` | hardware report of this computer or a server |
| `homebrew models [--modality text\|image] [--all]` | supported base models |
| `homebrew status [JOB]` | training jobs of the current project |
| `homebrew etf validate\|stats\|convert\|schema` | work with ETF files |

Inside a session: `/status`, `/jobs`, `/watch`, `/gallery`, `/level`, `/usage`, `/thinking`, `/help`, `/quit`.

You type into a framed input bar; the line under it shows project · guiding model · level · phase · tokens ·
context. The brewmaster's thinking appears as a dim, italic block above its reply (`/thinking full` shows all of
it, `/thinking off` hides it). Environment switches:

| Variable | Effect |
|---|---|
| `HOMEBREW_AI_PLAIN_INPUT=1` | plain `you ▸` prompt instead of the framed input bar (for very limited terminals) |
| `HOMEBREW_AI_SIMPLE_PROMPTS=1` | numbered choices instead of arrow-key menus (automatic in narrow terminals) |
| `HOMEBREW_AI_THINKING=on\|full\|off` | initial thinking display |
| `HOMEBREW_AI_SYNTH_WORKERS=4` | parallel requests for synthetic data (1–16) |

Worker commands (run on the training machine, used by Homebrew itself): `homebrew train JOB.yaml`,
`homebrew chain JOB.yaml…`, `homebrew test`, `homebrew candidates`, `homebrew export`, `homebrew push`.

## Project layout

```text
my-pirate-bot/
  homebrew.yaml        decisions and state (goal, model, compute, datasets, drafts, jobs, exports)
  data/                ETF datasets + manifests (source, licence, mapping)
  data/_sets/<name>/   training sets built from datasets (train.jsonl, eval.jsonl, images/)
  runs/<job_id>/       job.yaml, data copy, status.json, metrics.jsonl, train.log, final/ (adapter/model)
  .homebrew/           conversation history (to resume)
```

User settings live in `~/.config/homebrew-ai/` (`settings.yaml`, `credentials.yaml` with mode 600). Extra model
profiles can be dropped into `~/.config/homebrew-ai/profiles/` (same schema as the shipped ones).

## Documentation

- [docs/etf.md](docs/etf.md): the Empero Trace Format
- [docs/models.md](docs/models.md): supported models and their hyperparameter guidelines (generated)
- [docs/architecture.md](docs/architecture.md): how the pieces fit together

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest -m "not slow"   # fast suite
pytest -m slow         # tiny CPU training runs: SFT/full/CPT→SFT→DPO, export, a detached local job
```

## Roadmap

GGUF export for llama.cpp/Ollama, free-tier notebooks (Colab/Kaggle), more image models and image editing LoRAs,
multi-GPU sharding (FSDP) for large full fine-tunes, KTO/ORPO, evaluation suites, and automatic GPU rental via provider
APIs (opt-in, with spending limits).

## Licence

Homebrew is released under the **Homebrew License**: MIT terms for individuals and organisations up to USD 2,000,000
gross monthly revenue (averaged over twelve months, including affiliates). Larger organisations need a commercial licence
from Empero (hello@empero.org). Models and data you create with Homebrew are yours, subject to the licences of the base
models and datasets you used. See [LICENSE](LICENSE).
