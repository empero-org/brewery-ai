"""Image datasets for text-to-image LoRAs.

An image dataset is a folder ``data/<name>/`` with ``dataset.jsonl`` (ETF
image records) and the image files under ``images/``. Captions can come from
same-named ``.txt`` files (the common kohya convention), a dataset column, the
user, or an AI captioner (see ``data/synth.py``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from homebrew_ai.data.prepare import check_name, set_dir
from homebrew_ai.etf.io import iter_records, write_records

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIN_SIDE = 512


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def import_image_folder(data_dir: Path, name: str, folder: str | Path, *, trigger_word: str | None = None, default_caption: str = "") -> dict[str, Any]:
    check_name(name)
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not files:
        raise ValueError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {folder}")
    out_dir = data_dir / name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records, seen = [], set()
    skipped: Counter = Counter()
    for src in files:
        size = _image_size(src)
        if size is None:
            skipped["unreadable"] += 1
            continue
        digest = _hash_file(src)
        if digest in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(digest)
        dest = img_dir / f"{digest[:12]}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        caption_file = src.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8", errors="replace").strip() if caption_file.exists() else default_caption
        records.append({"image": f"images/{dest.name}", "caption": caption, "meta": {"source": f"local:{folder.name}", "width": size[0], "height": size[1]}})
    return _finish(out_dir, name, records, skipped, {"type": "local", "path": str(folder)}, trigger_word)


KEEP_FORMATS = {"jpeg": ".jpg", "mpo": ".jpg", "png": ".png", "webp": ".webp"}  # stored as downloaded


def _store_image(img_dir: Path, data: bytes) -> tuple[Path, tuple[int, int]] | None:
    """Save image bytes without re-encoding (JPEG/PNG/WebP); other formats become PNG. None if unreadable."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as probe:
            size, fmt = probe.size, (probe.format or "").lower()
            ext = KEEP_FORMATS.get(fmt)
            digest = hashlib.sha1(data).hexdigest()[:12]
            if ext is None:  # GIF, TIFF, BMP ...: convert once
                dest = img_dir / f"{digest}.png"
                probe.convert("RGBA" if probe.mode in ("P", "LA", "RGBA") else "RGB").save(dest)
                return dest, size
    except Exception:
        return None
    dest = img_dir / f"{digest}{ext}"
    dest.write_bytes(data)
    return dest, size


def import_image_hub(
    data_dir: Path, name: str, dataset_id: str, *, image_column: str, caption_column: str | None, config: str | None = None,
    split: str = "train", max_rows: int = 200, trigger_word: str | None = None, license: str | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    check_name(name)
    from datasets import Image as ImageFeature
    from datasets import load_dataset

    from homebrew_ai.data.hub import hf_token

    say = on_progress or (lambda _m: None)
    out_dir = data_dir / name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    say("connecting to the dataset")
    ds = load_dataset(dataset_id, name=config, split=split, streaming=True, token=hf_token())
    try:  # raw bytes: no decode/re-encode of every photo (that made imports crawl)
        ds = ds.cast_column(image_column, ImageFeature(decode=False))
    except Exception:
        pass
    ds = ds.shuffle(seed=42, buffer_size=max(100, min(2000, max_rows * 4)))
    records, seen = [], set()
    skipped: Counter = Counter()
    say(f"0/{max_rows} images")
    for i, row in enumerate(ds):
        if len(records) >= max_rows or i > max_rows * 4:
            break
        img = row.get(image_column)
        data = None
        if isinstance(img, dict):
            data = img.get("bytes")
            if data is None and img.get("path"):
                try:
                    data = Path(img["path"]).read_bytes()
                except OSError:
                    data = None
        elif img is not None and hasattr(img, "save"):  # already decoded (cast not possible)
            buf = io.BytesIO()
            (img.convert("RGBA") if img.mode in ("P", "LA") else img).save(buf, format="PNG")
            data = buf.getvalue()
        if not data:
            skipped["no_image"] += 1
            continue
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(digest)
        stored = _store_image(img_dir, data)
        if stored is None:
            skipped["unreadable"] += 1
            continue
        dest, size = stored
        caption = str(row.get(caption_column) or "") if caption_column else ""
        meta = {"source": f"hf:{dataset_id}", "width": size[0], "height": size[1]}
        if license:
            meta["license"] = license
        records.append({"image": f"images/{dest.name}", "caption": caption.strip(), "meta": meta})
        say(f"{len(records)}/{max_rows} images")
    return _finish(out_dir, name, records, skipped, {"type": "hf", "id": dataset_id, "config": config, "split": split}, trigger_word, license)


def _finish(out_dir: Path, name: str, records: list[dict[str, Any]], skipped: Counter, source: dict[str, Any], trigger_word: str | None, license: str | None = None) -> dict[str, Any]:
    if not records:
        raise ValueError(f"no usable images (skipped: {dict(skipped)})")
    write_records(out_dir / "dataset.jsonl", records)
    report = check_images(out_dir)
    manifest = {
        "name": name,
        "kind": "image",
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "license": license,
        "trigger_word": trigger_word,
        "records": len(records),
        "skipped": dict(skipped),
        "report": report,
    }
    (out_dir / "dataset.meta.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"path": str(out_dir / "dataset.jsonl"), "records": len(records), "skipped": dict(skipped), "report": report}


def check_images(folder: Path) -> dict[str, Any]:
    records = list(iter_records(folder / "dataset.jsonl"))
    sizes, small, uncaptioned = [], 0, 0
    aspects: Counter = Counter()
    for r in records:
        meta = r.get("meta") or {}
        w, h = meta.get("width"), meta.get("height")
        if not (w and h):
            size = _image_size(folder / r["image"])
            w, h = size if size else (0, 0)
        if min(w, h) < MIN_SIDE:
            small += 1
        sizes.append(min(w, h))
        ratio = w / h if h else 1
        aspects["square" if 0.9 <= ratio <= 1.1 else "landscape" if ratio > 1.1 else "portrait"] += 1
        if not r.get("caption"):
            uncaptioned += 1
    tips = []
    if len(records) < 10:
        tips.append("fewer than 10 images: results will be weak; 15-50 varied images work best")
    if small:
        tips.append(f"{small} image(s) are smaller than {MIN_SIDE}px on the short side and will look soft")
    if uncaptioned:
        tips.append(f"{uncaptioned} image(s) have no caption yet")
    sizes.sort()
    return {"images": len(records), "short_side_median": sizes[len(sizes) // 2] if sizes else 0, "small_images": small, "uncaptioned": uncaptioned, "aspects": dict(aspects), "tips": tips}


def set_captions(folder: Path, captions: dict[str, str]) -> int:
    """Replace captions by image path (``images/xyz.png`` → caption)."""
    records = list(iter_records(folder / "dataset.jsonl"))
    changed = 0
    for r in records:
        if r["image"] in captions:
            r["caption"] = captions[r["image"]].strip()
            changed += 1
    write_records(folder / "dataset.jsonl", records)
    return changed


def build_image_training_set(data_dir: Path, names: list[str], *, repeat: int = 1, seed: int = 42, set_name: str = "train") -> dict[str, Any]:
    """Combine image datasets into ``data/_sets/<set_name>/train.jsonl`` with the images next to it."""
    out = set_dir(data_dir, set_name)
    img_out = out / "images"
    if img_out.exists():
        shutil.rmtree(img_out)  # only this training set's own copies (hard links) of the images
    img_out.mkdir(parents=True)
    records = []
    for name in names:
        folder = data_dir / check_name(name)
        for r in iter_records(folder / "dataset.jsonl"):
            src = folder / r["image"]
            dest = img_out / f"{name}-{Path(r['image']).name}"
            try:
                dest.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dest)
            records.append({**r, "image": f"images/{dest.name}"})
    if not records:
        raise ValueError("no images to train on")
    random.Random(seed).shuffle(records)
    write_records(out / "train.jsonl", records * max(repeat, 1))
    return {"train": str(out / "train.jsonl"), "eval": None, "num_train": len(records), "num_eval": 0, "kind": "image", "objectives": ["sft"], "images_dir": str(img_out), "parts": names}
