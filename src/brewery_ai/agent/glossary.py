"""Short, curated explanations of fine-tuning terms at two depths.

The agent can always explain things itself; the glossary keeps explanations
correct and consistent, which matters most with small local backend models.
"""

from __future__ import annotations

GLOSSARY: dict[str, tuple[str, str]] = {
    # term: (plain explanation, technical explanation)
    "fine-tuning": (
        "Taking a model that already knows a lot and giving it extra practice on your examples, so it picks up your style or knowledge.",
        "Continued training of a pretrained model on a task-specific dataset, usually supervised (SFT) on prompt/answer pairs with loss only on the answers.",
    ),
    "base model": (
        "The ready-made model you start from. Brewery teaches it new tricks instead of building one from scratch.",
        "The pretrained checkpoint whose weights are adapted. 'Instruct' variants are already chat-tuned; 'base'/'pt' variants are raw pretrained models.",
    ),
    "lora": (
        "A way to teach a model by adding a small 'add-on' instead of changing the whole model. Much cheaper, and you can switch the add-on on and off.",
        "Low-Rank Adaptation: frozen base weights plus trainable low-rank matrices (rank r, scale alpha/r) injected into linear layers; typically <1% trainable parameters.",
    ),
    "qlora": (
        "LoRA on a compressed copy of the model, so it fits on smaller, cheaper GPUs. A bit slower, nearly as good.",
        "LoRA over a 4-bit NF4-quantized frozen base (bitsandbytes) with bf16 compute; cuts weight memory ~4x at some speed cost.",
    ),
    "full fine-tuning": (
        "Changing every part of the model. Can give the best results but needs much more GPU memory and can easily 'forget' old skills.",
        "Updating all parameters; needs weights + gradients + optimizer states in memory (roughly 6-16 bytes per parameter).",
    ),
    "epoch": (
        "One full pass over all your examples. Two epochs means the model sees every example twice.",
        "One pass over the training set; with small data, more epochs raise the risk of overfitting.",
    ),
    "learning rate": (
        "How big each learning step is. Too big and the model gets confused, too small and it barely learns.",
        "Step size of the optimizer; LoRA typically uses ~1e-4 to 2e-4, full fine-tuning ~5e-6 to 2e-5, with warmup and decay.",
    ),
    "batch size": (
        "How many examples the model looks at before it updates itself once.",
        "Effective batch = micro batch per GPU × gradient accumulation steps × number of GPUs.",
    ),
    "gradient accumulation": (
        "A trick to act like a big batch on a small GPU: look at a few examples, remember the lesson, repeat, then update.",
        "Summing gradients over several micro-batches before an optimizer step to reach a larger effective batch size within memory limits.",
    ),
    "loss": (
        "A score for how wrong the model's guesses are on your examples. It should go down during training.",
        "Mean token-level cross-entropy on trainable (assistant) tokens; eval loss on held-out data reveals overfitting.",
    ),
    "overfitting": (
        "When the model memorises your examples instead of learning the pattern, so it does worse on new questions.",
        "Training loss keeps falling while eval loss rises; mitigate with more/varied data, fewer epochs, lower LR or rank.",
    ),
    "token": (
        "A piece of a word. Models read and write text in tokens; 1,000 tokens is roughly 750 English words.",
        "Subword unit from the tokenizer; sequence length, memory and cost all scale with token counts.",
    ),
    "context length": (
        "How much text the model can read at once during training. Longer needs more GPU memory.",
        "max_seq_len: samples longer than this are dropped (default) or truncated; memory grows roughly linearly with it.",
    ),
    "gpu": (
        "A graphics card, used here as a very fast calculator. Training needs one with enough memory (VRAM).",
        "Accelerator for the matrix multiplications; VRAM bounds model size, batch and sequence length; bf16 support (Ampere+) matters for stability.",
    ),
    "vram": (
        "The GPU's own memory. The model, its add-on and the examples being processed all have to fit in it.",
        "Device memory holding weights, adapter/optimizer states, activations and logits; Brewery estimates it before training.",
    ),
    "dataset": (
        "Your collection of example conversations (or images with captions) the model learns from.",
        "Training corpus, stored by Brewery as ETF (Empero Trace Format) .jsonl records.",
    ),
    "etf": (
        "Brewery's file format for training examples: one example conversation (or image + caption) per line.",
        "Empero Trace Format: JSONL with trace records (messages incl. reasoning/tool calls), text records and image records; see docs/etf.md.",
    ),
    "chat template": (
        "The exact way a model expects a conversation to be written down. Brewery always uses the model's own.",
        "Jinja template shipped with the tokenizer that serialises messages with role markers; training and inference must match.",
    ),
    "reasoning": (
        "Some models 'think out loud' before answering. You can train that thinking too, if your examples include it.",
        "Chain-of-thought traces rendered in the model's native format (e.g. <think> blocks for Qwen) or inline for families without one.",
    ),
    "tool calling": (
        "Teaching a model to use tools, like a calculator or a weather lookup, by showing example conversations where it does.",
        "Assistant turns that emit structured function calls; results come back as tool messages, rendered per the model's template.",
    ),
    "adapter": (
        "The small add-on file LoRA produces. You load it on top of the base model.",
        "PEFT adapter weights (adapter_model.safetensors + adapter_config.json) applied to the frozen base at load time.",
    ),
    "merge": (
        "Baking the add-on into the model, so you get one normal model file that works everywhere.",
        "merge_and_unload(): W' = W + (alpha/r)·B·A, saved as a standalone checkpoint (larger download, no PEFT needed).",
    ),
    "quantization": (
        "Storing the model's numbers in a shorter, compressed form so it needs less memory.",
        "Lower-precision weight storage (e.g. NF4 4-bit, int8); QLoRA quantizes the frozen base only.",
    ),
    "trigger word": (
        "A special word (like 'sks corgi') you put in every caption, so you can call up what the image model learned.",
        "Rare token sequence bound to the concept during LoRA training; used in prompts to activate it.",
    ),
    "steps": (
        "For image models, training is counted in steps: each step the model practises on a few of your pictures.",
        "Optimizer updates; for image LoRAs a few hundred to a few thousand steps at batch 1-4 is typical.",
    ),
    "hugging face": (
        "A website where people share AI models and datasets, like a library. Brewery can download from it and publish your model there.",
        "huggingface.co Hub: git-backed model/dataset repositories with model cards, gated access and an API.",
    ),
}

ALIASES = {
    "finetuning": "fine-tuning", "fine tuning": "fine-tuning", "low-rank adaptation": "lora", "lr": "learning rate",
    "epochs": "epoch", "tokens": "token", "seq len": "context length", "max_seq_len": "context length",
    "sequence length": "context length", "hf": "hugging face", "huggingface": "hugging face", "gpu memory": "vram",
    "thinking": "reasoning", "chain of thought": "reasoning", "function calling": "tool calling", "tools": "tool calling",
    "empero trace format": "etf", "4-bit": "quantization", "full finetune": "full fine-tuning", "full ft": "full fine-tuning",
}


def explain(term: str, level: str) -> dict[str, str] | None:
    key = term.strip().lower()
    key = ALIASES.get(key, key)
    entry = GLOSSARY.get(key)
    if entry is None:
        return None
    plain, technical = entry
    if level in ("beginner", "hobbyist"):
        return {"term": key, "explanation": plain}
    return {"term": key, "explanation": technical, "plain": plain}
