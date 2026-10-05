"""Experience levels: Brewery asks once, then talks to everyone at their level."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    key: str
    emoji: str
    label: str
    blurb: str
    style: str

    @property
    def title(self) -> str:
        return f"{self.emoji} {self.label}"


LEVELS: dict[str, Level] = {
    "beginner": Level(
        "beginner", "🌱", "Beginner", "I've never trained an AI model before",
        "Plain language, no jargon without a one-line explanation. Use everyday comparisons where they help "
        "(a dataset is a stack of example conversations, fine-tuning is practice for a model that already knows a lot). "
        "Ask one question at a time, as ask_user with 2-4 choices and a clear recommendation. Don't show raw configs, "
        "file paths or hyperparameters unless asked; give the gist ('about 20 minutes on a rented GPU, roughly $0.50'). "
        "Always state costs and anything that needs an account up front, and confirm before spending money.",
    ),
    "hobbyist": Level(
        "hobbyist", "🍺", "Hobbyist", "I use AI tools a lot, training one is new to me",
        "Friendly and clear. Explain a technical term in one short sentence the first time you use it. Offer 2-4 clear "
        "options with a recommendation. Show the numbers that matter (model size, time, cost, dataset size) and keep "
        "detailed hyperparameters behind a 'want the details?' offer.",
    ),
    "builder": Level(
        "builder", "🛠️", "Builder", "I can code, but I'm new to fine-tuning",
        "Assume programming literacy; explain ML concepts briefly and precisely. Show the training config summary, "
        "memory and time estimates, and the main trade-offs (LoRA vs QLoRA vs full, rank, sequence length). Accept "
        "overrides and explain their consequences.",
    ),
    "expert": Level(
        "expert", "🧪", "Expert", "I know LoRA, learning rates & co. — just automate it",
        "Terse and technical; skip basics. Show complete configs, memory breakdowns and every knob. Apply requested "
        "overrides directly within the guidelines; values beyond the hard guidelines need an explicit expert override "
        "confirmed by the user. The run_shell tool is available for debugging.",
    ),
}

DEFAULT_LEVEL = "hobbyist"


def get_level(key: str | None) -> Level:
    return LEVELS.get(key or DEFAULT_LEVEL, LEVELS[DEFAULT_LEVEL])
