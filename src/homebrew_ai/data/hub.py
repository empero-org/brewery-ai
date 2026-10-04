"""Finding and peeking at datasets on the Hugging Face Hub.

Searching uses the Hub API; peeking uses the Dataset Viewer API, so nothing is
downloaded until the user decides to import a dataset.
"""

from __future__ import annotations

from typing import Any

import httpx

from homebrew_ai.about import user_agent
from homebrew_ai.etf.convert import detect_mapping

VIEWER = "https://datasets-server.huggingface.co"
NSFW_TAG = "not-for-all-audiences"


def hf_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _headers() -> dict[str, str]:
    headers = {"User-Agent": user_agent()}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    return [t.split(":", 1)[1] for t in tags if t.startswith(prefix + ":")]


def _short(text: Any, limit: int = 300) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"… [{len(text) - limit} more chars]"
    if isinstance(text, list):
        return [_short(t, limit) for t in text[:6]] + ([f"… {len(text) - 6} more items"] if len(text) > 6 else [])
    if isinstance(text, dict):
        return {k: _short(v, limit) for k, v in list(text.items())[:12]}
    return text


def search_datasets(
    query: str,
    *,
    limit: int = 8,
    language: str | None = None,
    task: str | None = None,
    modality: str | None = None,
    safe: bool = True,
    sort: str = "downloads",
) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    filters = []
    if language:
        filters.append(f"language:{language}")
    if task:
        filters.append(f"task_categories:{task}")
    if modality:
        filters.append(f"modality:{modality}")
    api = HfApi(token=hf_token())
    infos = api.list_datasets(
        search=query or None,
        filter=filters or None,
        sort=sort,
        limit=limit * 3,
        expand=["downloads", "likes", "tags", "gated", "cardData", "lastModified"],
    )
    rows = []
    for info in infos:
        tags = list(info.tags or [])
        nsfw = NSFW_TAG in tags
        if safe and nsfw:
            continue
        card = info.card_data.to_dict() if getattr(info, "card_data", None) is not None and hasattr(info.card_data, "to_dict") else (info.card_data or {})
        rows.append(
            {
                "id": info.id,
                "downloads": info.downloads,
                "likes": info.likes,
                "gated": bool(info.gated),
                "license": (_tag_values(tags, "license") or [card.get("license")])[0] if (_tag_values(tags, "license") or card.get("license")) else None,
                "languages": _tag_values(tags, "language")[:5],
                "tasks": _tag_values(tags, "task_categories")[:4],
                "size": (_tag_values(tags, "size_categories") or [None])[0],
                "modalities": _tag_values(tags, "modality"),
                "pretty_name": card.get("pretty_name"),
                "not_for_all_audiences": nsfw,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _get(path: str, **params: Any) -> dict[str, Any]:
    r = httpx.get(f"{VIEWER}/{path}", params=params, headers=_headers(), timeout=30)
    if r.status_code == 401 or r.status_code == 404:
        raise PermissionError("the dataset viewer cannot read this dataset (it may be gated, private, or not support previews)")
    r.raise_for_status()
    return r.json()


def _image_columns(features: Any) -> list[str]:
    cols = []
    if isinstance(features, list):  # first-rows format: [{"name", "type": {...}}]
        for f in features:
            if isinstance(f.get("type"), dict) and f["type"].get("_type") == "Image":
                cols.append(f["name"])
    elif isinstance(features, dict):  # info format: {name: {"_type": ...}}
        for name, spec in features.items():
            if isinstance(spec, dict) and spec.get("_type") == "Image":
                cols.append(name)
    return cols


def inspect_dataset(dataset_id: str, config: str | None = None, split: str | None = None, n: int = 3) -> dict[str, Any]:
    """Splits, columns, size, license and a few sample rows — plus a mapping guess."""
    from huggingface_hub import HfApi

    out: dict[str, Any] = {"id": dataset_id}
    try:
        meta = HfApi(token=hf_token()).dataset_info(dataset_id)
        tags = list(meta.tags or [])
        out["license"] = (_tag_values(tags, "license") or [None])[0]
        out["gated"] = bool(meta.gated)
        out["not_for_all_audiences"] = NSFW_TAG in tags
        out["languages"] = _tag_values(tags, "language")[:5]
    except Exception as exc:
        out["hub_error"] = str(exc)[:200]

    try:
        splits = _get("splits", dataset=dataset_id).get("splits", [])
    except PermissionError as exc:
        out["error"] = str(exc)
        return out
    if not splits:
        out["error"] = "no splits found (the dataset viewer may still be processing it)"
        return out
    configs = sorted({s["config"] for s in splits})
    config = config or ("default" if "default" in configs else configs[0])
    cfg_splits = [s["split"] for s in splits if s["config"] == config]
    split = split or ("train" if "train" in cfg_splits else cfg_splits[0])
    out.update(configs=configs[:20], config=config, splits=cfg_splits, split=split)

    try:
        info = _get("info", dataset=dataset_id, config=config).get("dataset_info", {})
        out["num_rows"] = {k: v.get("num_examples") for k, v in (info.get("splits") or {}).items()}
        out["download_size_mb"] = round((info.get("download_size") or 0) / 1e6, 1)
        out["columns"] = {k: (v.get("_type") if isinstance(v, dict) else "list") for k, v in (info.get("features") or {}).items()}
        image_cols = _image_columns(info.get("features"))
    except Exception:
        image_cols = []

    first = _get("first-rows", dataset=dataset_id, config=config, split=split)
    rows = [r["row"] for r in first.get("rows", [])]
    image_cols = image_cols or _image_columns(first.get("features"))
    out["image_columns"] = image_cols
    out["sample_rows"] = [_short({k: ("<image>" if k in image_cols else v) for k, v in row.items()}) for row in rows[:n]]
    if image_cols:
        text_cols = [k for k, v in (rows[0].items() if rows else []) if isinstance(v, str)]
        out["suggested"] = {"kind": "image", "image_column": image_cols[0], "caption_column": next((c for c in text_cols if c.lower() in ("text", "caption", "prompt", "description")), text_cols[0] if text_cols else None)}
    else:
        mapping, why = detect_mapping(rows)
        out["suggested"] = {"kind": "text", "mapping": mapping.to_dict() if mapping else None, "explanation": why}
    return out
