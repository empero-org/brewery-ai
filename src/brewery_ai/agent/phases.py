"""The brewing workflow, phase by phase.

Phases are a guide, not a cage: the agent may jump back (e.g. to fix data)
whenever needed. For small backend models the visible tools are scoped to the
current phase to keep them focused.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    key: str
    title: str
    goal: str
    done_when: str
    tools: tuple[str, ...]


CORE_TOOLS = ("ask_user", "update_project", "get_project_state", "explain_term")

PHASES: tuple[Phase, ...] = (
    Phase("welcome", "👋 Welcome", "Greet the user briefly and make sure the project has a name.", "you know what the project is called", ()),
    Phase(
        "goal", "🎯 Goal",
        "Find out what the model should do: text (chat style, persona, domain expert, tool use, reasoning, a language) or "
        "images (a style, a character, an object). Ask follow-ups until the goal is concrete. Save it with update_project.",
        "project.goal and project.modality are set",
        (),
    ),
    Phase(
        "compute", "🖥️ Compute",
        "Find out where training can run: detect this computer's hardware; if it has no suitable GPU, explain renting one "
        "(gpu_rental_guide), then connect_server and prepare_server.",
        "compute is local with a usable GPU, or a prepared SSH server",
        ("detect_hardware", "use_local_computer", "install_local_training_packages", "estimate_requirements", "gpu_rental_guide", "create_ssh_key", "connect_server", "prepare_server"),
    ),
    Phase(
        "model", "🧠 Base model",
        "Pick a base model that fits the goal, the hardware and the user's licence needs (list_base_models, "
        "get_model_details, estimate_requirements). Check gated access (select_base_model).",
        "project.base_model is set and accessible",
        ("list_base_models", "get_model_details", "estimate_requirements", "select_base_model", "hf_login_status", "hf_login"),
    ),
    Phase(
        "data", "🌾 Ingredients (data)",
        "Get training data for each stage of the regime: raw text for CPT, conversations for SFT, preference pairs for "
        "DPO. Search and inspect Hugging Face datasets, import them (or local files), add hand-written examples, or — if "
        "the user opts in — generate synthetic data shaped by their needs. Preview what the model will see, check "
        "statistics, then build a named training set per stage.",
        "a training set exists for each planned stage, with enough good examples",
        ("search_datasets", "inspect_dataset", "import_dataset", "import_images", "add_examples", "preview_dataset", "dataset_stats", "clean_dataset", "generate_synthetic_data", "caption_images", "build_training_set"),
    ),
    Phase(
        "config", "⚙️ Recipe",
        "Plan the regime (e.g. SFT only; SFT then DPO; CPT then SFT then DPO for base models) and prepare each stage "
        "with propose_training_config (never invent hyperparameters). Later stages start from earlier ones via "
        "init_from_job. Present the plan at the user's level, adjust on request, make sure it fits the hardware.",
        "draft jobs without errors exist for the next stage(s)",
        ("propose_training_config", "estimate_requirements", "get_model_details"),
    ),
    Phase(
        "train", "🔥 Brewing",
        "Start training (start_training asks the user to confirm; consecutive stages can run as one chain), then watch "
        "it and explain progress in plain words. Handle failures using the hint in the status.",
        "the job(s) completed",
        ("start_training", "watch_training", "training_status", "stop_training", "open_preview_gallery", "propose_training_config"),
    ),
    Phase(
        "evaluate", "👅 Taste test",
        "Try the model with test_model on a few prompts (compare with the base model) and discuss whether it does what "
        "the user wanted. To improve it further, collect preferences: generate_candidates from the trained model, let "
        "the user (or an AI judge the user reviews) pick the better answers with review_candidates, then run a DPO stage.",
        "the user is happy with the samples",
        ("test_model", "fetch_results", "training_status", "open_preview_gallery", "generate_candidates", "review_candidates", "build_training_set", "propose_training_config"),
    ),
    Phase(
        "package", "🍾 Bottling",
        "Pick a repository name that follows the base model's licence rules, write a short description, and package the "
        "model with package_model (adapter or merged).",
        "an export exists",
        ("package_model", "hf_login_status", "hf_login"),
    ),
    Phase(
        "publish", "🚀 Sharing",
        "Offer to upload to Hugging Face with upload_to_hf (it asks the user to confirm). Remind people who rented a GPU "
        "to stop/destroy the server afterwards so billing ends.",
        "uploaded, or the user decided to keep it private/local",
        ("upload_to_hf", "hf_login_status", "hf_login"),
    ),
    Phase("done", "🎉 Done", "Summarise what was made and suggest next experiments.", "-", ()),
)

PHASE_BY_KEY = {p.key: p for p in PHASES}


def tools_for_phase(key: str) -> set[str]:
    phase = PHASE_BY_KEY.get(key)
    names = set(CORE_TOOLS)
    if phase:
        names.update(phase.tools)
        idx = PHASES.index(phase)
        if idx + 1 < len(PHASES):  # let the agent move on without a phase-switch round-trip
            names.update(PHASES[idx + 1].tools)
    return names
