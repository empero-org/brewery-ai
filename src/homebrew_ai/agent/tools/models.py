"""Choosing a base model, and the user's Hugging Face account."""

from __future__ import annotations

from typing import Any

from homebrew_ai.agent.tools.base import ToolContext, ToolError, tool
from homebrew_ai.models import registry


def _range(r) -> dict[str, Any] | None:
    if r is None:
        return None
    d = {k: v for k, v in {"min": r.min, "max": r.max, "default": r.default, "recommended": r.recommended, "choices": r.choices}.items() if v is not None}
    return d or None


@tool(
    "list_base_models",
    """List the base models this Homebrew version can fine-tune, with size, licence, gating and allowed methods.
Filter by modality (text/image), family, or maximum size in billions of parameters.""",
    {
        "modality": {"type": "string", "enum": ["text", "image"]},
        "family": {"type": "string", "description": "qwen3, qwen3_5, qwen3_5_moe, llama3, gemma3, gemma4, qwen_image"},
        "max_params_b": {"type": "number"},
        "include_base": {"type": "boolean", "description": "Include raw pretrained (non-chat) checkpoints. Default false."},
    },
)
def list_base_models(ctx: ToolContext, args: dict[str, Any]) -> Any:
    models = registry.search(
        modality=args.get("modality") or ctx.project.state.modality,
        family=args.get("family"),
        max_params_b=args.get("max_params_b"),
        include_base=bool(args.get("include_base", False)),
    )
    rows = registry.summarize(models)
    families = {m.family: m.profile for m in models}
    return {
        "models": rows,
        "families": {k: {"name": p.display_name, "summary": p.summary.strip(), "license": p.license.name} for k, p in families.items()},
    }


@tool(
    "get_model_details",
    """Everything Homebrew knows about one base model: architecture size, context length, chat features, licence
obligations, hyperparameter guidelines per method, LoRA target presets and important notes.""",
    {"model_id": {"type": "string"}},
    ["model_id"],
)
def get_model_details(ctx: ToolContext, args: dict[str, Any]) -> Any:
    try:
        m = registry.get_model(args["model_id"])
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    v = m.variant
    lic = m.license
    guidelines = {}
    for method in ("lora", "qlora", "full"):
        g = m.guidelines.for_method(method)
        if not g.allowed:
            guidelines[method] = {"allowed": False, "why": " ".join(g.notes)}
            continue
        guidelines[method] = {
            k: val for k, val in {
                "learning_rate": _range(g.learning_rate), "rank": _range(g.rank), "epochs": _range(g.epochs), "max_steps": _range(g.max_steps),
                "effective_batch": _range(g.effective_batch), "max_seq_len": _range(g.max_seq_len), "resolution": _range(g.resolution),
                "optimizers": g.optimizers, "default_optimizer": g.default_optimizer, "notes": g.notes or None,
            }.items() if val
        }
    chat = m.chat_format_dict() if m.modality == "text" else {}
    return {
        "id": v.id,
        "label": v.label,
        "family": m.profile.display_name,
        "modality": m.modality,
        "kind": v.kind,
        "params_b": v.params_b,
        "active_params_b": v.active_params_b,
        "context": v.context,
        "moe": v.is_moe,
        "gated": v.gated,
        "publisher": v.publisher or m.profile.vendor,
        "chat": {"reasoning": chat.get("reasoning"), "tools": chat.get("tools")} if chat else None,
        "license": {
            "name": lic.name, "url": lic.url, "noncommercial": lic.noncommercial, "name_prefix_required": lic.name_prefix,
            "attribution_required": lic.attribution, "name_rules": lic.name_rules,
        },
        "requirements": m.profile.requirements,
        "lora_target_presets": {p: m.lora_targets(p) for p in m.lora_presets()},
        "guidelines": guidelines,
        "notes": m.notes(),
    }


@tool(
    "select_base_model",
    """Choose the base model for this project. Checks whether the user can download it (gated models like Llama and
Gemma 3 need the licence accepted on huggingface.co and a logged-in token) and records the choice.""",
    {"model_id": {"type": "string"}},
    ["model_id"],
    activity="Checking model access",
)
def select_base_model(ctx: ToolContext, args: dict[str, Any]) -> Any:
    try:
        m = registry.get_model(args["model_id"])
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    access = check_access(m.id)
    s = ctx.project.state
    s.base_model = m.id
    s.modality = m.modality
    s.drafts.clear()
    ctx.project.save()
    out: dict[str, Any] = {"selected": m.id, "modality": m.modality, "access": access, "license": m.license.name}
    if m.license.noncommercial:
        out["important"] = "NON-COMMERCIAL licence: the brewed model and its outputs may not be used commercially. Make sure the user knows."
    if access["status"] != "ok":
        out["todo"] = access.get("how_to_fix")
    return out


def check_access(model_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover
        return {"status": "unknown"}
    from homebrew_ai.data.hub import hf_token

    token = hf_token()
    try:
        auth_check(model_id, token=token)
        return {"status": "ok"}
    except GatedRepoError:
        fix = f"Open https://huggingface.co/{model_id}, log in, and accept the licence (approval can take a few minutes)."
        if not token:
            fix += " Then log Homebrew in to Hugging Face (hf_login)."
        return {"status": "gated", "logged_in": bool(token), "how_to_fix": fix}
    except RepositoryNotFoundError:
        return {"status": "not_found", "how_to_fix": "The repository was not found or is private."}
    except Exception as exc:
        return {"status": "unknown", "detail": str(exc)[:200]}


@tool("hf_login_status", "Is Homebrew logged in to Hugging Face, and as whom? (Needed for gated models and uploads.)")
def hf_login_status(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return whoami()


def whoami() -> dict[str, Any]:
    from homebrew_ai.data.hub import hf_token

    token = hf_token()
    if not token:
        return {"logged_in": False}
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).whoami()
    except Exception as exc:
        return {"logged_in": False, "problem": f"the stored token did not work: {str(exc)[:120]}"}
    auth = info.get("auth", {}).get("accessToken", {}) if isinstance(info.get("auth"), dict) else {}
    return {"logged_in": True, "user": info.get("name"), "orgs": [o.get("name") for o in info.get("orgs", [])][:10], "token_role": auth.get("role")}


@tool(
    "hf_login",
    """Log Homebrew in to Hugging Face. The user pastes an access token into a hidden prompt (it never goes through the
chat). For uploads the token needs 'write' permission. Explain where to get one: https://huggingface.co/settings/tokens""",
)
def hf_login(ctx: ToolContext, args: dict[str, Any]) -> Any:
    token = ctx.ui.secret("Paste your Hugging Face access token (input is hidden)").strip()
    if not token:
        raise ToolError("no token entered")
    try:
        from huggingface_hub import HfApi, login

        info = HfApi(token=token).whoami()
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        raise ToolError(f"that token did not work: {str(exc)[:150]}") from exc
    return {"logged_in": True, "user": info.get("name")}
