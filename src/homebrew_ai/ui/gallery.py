"""A local one-page gallery of image-LoRA preview sets.

Each row is a checkpoint (step 0 = the base model "before"), each column one preview prompt. Homebrew serves the page
on 127.0.0.1 only, behind a random token in the URL, from the run folders on this computer. Remote runs are synced on
demand: while the page is open it asks every few seconds, and Homebrew checks the server at most every
``SYNC_EVERY_S`` seconds per job (one SSH call, plus a download for each new set).
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import shlex
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from homebrew_ai.project import Project
from homebrew_ai.remote.target import target_from_dict

SYNC_EVERY_S = 20
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
_galleries: dict[str, "Gallery"] = {}


def get_gallery(project: Project) -> "Gallery":
    """The project's gallery, started on first use (it lives as long as this Homebrew session)."""
    key = str(project.root.resolve())
    if key not in _galleries:
        _galleries[key] = Gallery(project)
    gallery = _galleries[key]
    gallery.start()
    return gallery


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Gallery:
    def __init__(self, project: Project):
        self.project = project
        self.token = secrets.token_urlsafe(9)
        self.server: ThreadingHTTPServer | None = None
        self._locks: dict[str, threading.Lock] = {}
        self._last_sync: dict[str, float] = {}
        self._remote: dict[str, dict[str, Any]] = {}

    # -- server --------------------------------------------------------------

    @property
    def url(self) -> str:
        port = self.server.server_address[1] if self.server else 0
        return f"http://127.0.0.1:{port}/{self.token}/"

    def start(self, port: int = 8765) -> str:
        if self.server is None:
            handler = _make_handler(self)
            for candidate in [*range(port, port + 20), 0]:
                try:
                    self.server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
                    break
                except OSError:
                    continue
            assert self.server is not None
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, name="homebrew-gallery", daemon=True).start()
        return self.url

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    # -- data ----------------------------------------------------------------

    def image_jobs(self) -> list[dict[str, Any]]:
        return [j for j in list(self.project.state.jobs) if j.get("modality") == "image"]

    def state(self) -> dict[str, Any]:
        return {"project": self.project.state.name, "jobs": [self.job_view(r) for r in self.image_jobs()], "now": time.time()}

    def job_view(self, record: dict[str, Any]) -> dict[str, Any]:
        job_id = record["job_id"]
        run_dir = Path(record["run_dir"])
        samples = run_dir / "samples"
        self.sync(record)
        index = _read_json(samples / "index.json") or {}
        remote = self._remote.get(job_id) or {}
        status = remote.get("status") or _read_json(run_dir / "status.json") or {}
        if "checkpoints" in remote:
            checkpoints = set(remote["checkpoints"])
        else:
            checkpoints = {int(p.name.split("-")[-1]) for p in (run_dir / "checkpoints").glob("checkpoint-*") if p.name.split("-")[-1].isdigit()}
        sets = []
        for entry in index.get("sets", []):
            step = int(entry.get("step", 0))
            folder = f"step_{step:06d}"
            count = int(entry.get("count", 0))
            images = [
                f"img/{job_id}/{folder}/sample_{i:02d}.png" if (samples / folder / f"sample_{i:02d}.png").is_file() else None
                for i in range(count)
            ]
            sets.append({"step": step, "images": images, "checkpoint": step in checkpoints})
        trigger = None
        job_yaml = _read_json_yaml(run_dir / "job.yaml")
        if job_yaml:
            trigger = (job_yaml.get("image") or {}).get("trigger_word")
        return {
            "job_id": job_id,
            "base_model": record.get("base_model"),
            "trigger": trigger,
            "state": status.get("state") or record.get("state"),
            "step": status.get("step"),
            "max_steps": status.get("max_steps"),
            "prompts": index.get("prompts", []),
            "sets": sets,
            "remote": bool(record.get("remote_run_dir")),
            "sync_error": remote.get("error"),
            "preview_error": status.get("preview_error"),
        }

    def sync(self, record: dict[str, Any]) -> None:
        """Copy new preview sets of a remote run to this computer (rate-limited; never raises)."""
        spec = record.get("target_spec") or {}
        if spec.get("kind") != "ssh" or not record.get("remote_run_dir"):
            return
        job_id = record["job_id"]
        lock = self._locks.setdefault(job_id, threading.Lock())
        if time.time() - self._last_sync.get(job_id, 0.0) < SYNC_EVERY_S or not lock.acquire(blocking=False):
            return
        try:
            self._last_sync[job_id] = time.time()
            target = target_from_dict(spec)
            rd = shlex.quote(record["remote_run_dir"])
            res = target.run(
                f"cat {rd}/samples/index.json 2>/dev/null || echo '{{}}'; echo '<<HB>>'; "
                f"cat {rd}/status.json 2>/dev/null || echo '{{}}'; echo '<<HB>>'; ls -1 {rd}/checkpoints 2>/dev/null; true",
                timeout=60,
            )
            if not res.ok:
                raise RuntimeError((res.stderr or res.stdout).strip()[-200:] or "the server did not answer")
            parts = res.stdout.split("<<HB>>")
            index = json.loads(parts[0].strip() or "{}")
            status = json.loads(parts[1].strip() or "{}") if len(parts) > 1 else {}
            names = parts[2].split() if len(parts) > 2 else []
            checkpoints = [int(n.split("-")[-1]) for n in names if n.startswith("checkpoint-") and n.split("-")[-1].isdigit()]
            local = Path(record["run_dir"]) / "samples"
            local.mkdir(parents=True, exist_ok=True)
            for entry in index.get("sets", []):
                folder = f"step_{int(entry.get('step', 0)):06d}"
                have = len(list((local / folder).glob("*.png"))) if (local / folder).is_dir() else 0
                if have < int(entry.get("count", 0)):
                    target.get(f"{record['remote_run_dir']}/samples/{folder}", local / folder)
            if index:  # written after the images, so the page never points at missing files
                (local / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
            self._remote[job_id] = {"status": status, "checkpoints": checkpoints, "error": None}
        except Exception as exc:  # noqa: BLE001 - shown on the page instead
            self._remote.setdefault(job_id, {})["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            lock.release()

    def image_path(self, job_id: str, rel: str) -> Path | None:
        record = next((j for j in self.image_jobs() if j["job_id"] == job_id), None)
        if record is None:
            return None
        samples = (Path(record["run_dir"]) / "samples").resolve()
        path = (samples / rel).resolve()
        if not path.is_relative_to(samples) or path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            return None
        return path


def _read_json_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _make_handler(gallery: Gallery):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HomebrewGallery/1"

        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal clean
            pass

        def _send(self, code: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            port = self.server.server_address[1]
            if (self.headers.get("Host") or "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return self._send(403, b"forbidden", "text/plain")  # blocks DNS-rebinding pages
            path = urlparse(self.path).path
            prefix = f"/{gallery.token}/"
            if path == prefix.rstrip("/"):
                self.send_response(302)
                self.send_header("Location", prefix)
                self.end_headers()
                return None
            if not path.startswith(prefix):
                return self._send(404, b"not found", "text/plain")
            rest = path[len(prefix):]
            if rest in ("", "index.html"):
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if rest == "api/jobs":
                return self._send(200, json.dumps(gallery.state()).encode("utf-8"), "application/json")
            if rest.startswith("img/") and rest.count("/") >= 2:
                job_id, rel = rest[4:].split("/", 1)
                file = gallery.image_path(unquote(job_id), unquote(rel))
                if file is None:
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                return self._send(200, file.read_bytes(), ctype, cache="max-age=86400")  # preview files never change
            return self._send(404, b"not found", "text/plain")

    return Handler


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA previews</title>
<style>
:root{--bg:#faf8f5;--panel:#fff;--text:#1d1b19;--muted:#6b645c;--line:#e7e1d8;--accent:#b8690b;--accent-soft:#f6e7cf;--ok:#2f7d4a;--warn:#9a5b00;--shadow:0 1px 2px rgba(0,0,0,.06),0 6px 20px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root{--bg:#14120f;--panel:#1c1915;--text:#f1ece4;--muted:#a59c90;--line:#2e2a24;--accent:#f2a541;--accent-soft:#3a2c17;--ok:#6cc58b;--warn:#f2c04a;--shadow:none}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:22px 24px 6px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
h1{margin:0;font-size:20px;font-weight:650;letter-spacing:-.01em}
h1 span{color:var(--accent)}
.meta{color:var(--muted);font-size:13px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:6px 24px 0}
.tab{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:5px 12px;border-radius:999px;cursor:pointer;font:inherit;font-size:13px}
.tab[aria-selected=true]{border-color:var(--accent);background:var(--accent-soft)}
main{padding:12px 24px 40px;max-width:100%}
.summary{display:flex;flex-wrap:wrap;gap:6px 18px;margin:4px 0 14px;font-size:14px;align-items:center}
.summary b{font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--muted);margin-right:6px;vertical-align:1px}
.dot.live{background:var(--ok);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(108,197,139,.55)}70%{box-shadow:0 0 0 8px rgba(108,197,139,0)}100%{box-shadow:0 0 0 0 rgba(108,197,139,0)}}
.note{color:var(--warn);font-size:13px}
.scroller{overflow:auto;max-width:100%;border:1px solid var(--line);border-radius:12px;background:var(--panel);box-shadow:var(--shadow)}
table{border-collapse:separate;border-spacing:0}
th,td{padding:10px;vertical-align:top;border-bottom:1px solid var(--line)}
tr:last-child th,tr:last-child td{border-bottom:0}
thead th{position:sticky;top:0;background:var(--panel);z-index:2;text-align:left;font-size:12px;font-weight:600;width:240px;max-width:240px}
thead th .p{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;color:var(--muted);font-weight:450;font-size:13px;margin-top:2px}
tbody th{position:sticky;left:0;background:var(--panel);z-index:1;text-align:left;white-space:nowrap;min-width:118px}
thead th:first-child{left:0;z-index:3;width:132px;max-width:132px}
.step{font-weight:650;font-size:15px}
.badge{display:inline-block;margin-top:5px;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:500;background:var(--accent-soft);color:var(--accent)}
.badge.ok{background:transparent;border:1px solid var(--ok);color:var(--ok)}
td{width:240px}
.cell{width:220px;height:220px;border-radius:8px;background:var(--line);display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;overflow:hidden}
.cell img{width:100%;height:100%;object-fit:cover;cursor:zoom-in;display:block}
.cell img:focus-visible{outline:3px solid var(--accent);outline-offset:-3px}
.empty{padding:56px 24px;text-align:center;color:var(--muted)}
.hint{color:var(--muted);font-size:13px;margin-top:14px;max-width:820px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font:12px ui-monospace,monospace;color:var(--text)}
#lb{position:fixed;inset:0;background:rgba(12,10,8,.93);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:10;padding:16px}
#lb.open{display:flex}
#lbframe{display:flex;gap:14px;align-items:flex-start;justify-content:center;max-width:100%}
#lbframe figure{margin:0;text-align:center;color:#a59c90;font-size:12px}
#lbframe img{max-width:min(90vw,1024px);max-height:74vh;border-radius:10px;display:block}
#lb.compare #lbframe img{max-width:min(45vw,900px)}
#lbcap{color:#f1ece4;margin-top:12px;text-align:center;max-width:900px;font-size:14px}
#lbcap b{color:#f2a541}
.lbbar{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.lbbar button{background:#2a2620;color:#f1ece4;border:1px solid #3a352d;border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit;font-size:13px}
.lbbar button:hover,.lbbar button:focus-visible{border-color:#f2a541;outline:none}
.lbbar button[aria-pressed=true]{background:#3a2c17;border-color:#f2a541}
@media (max-width:640px){header,main,.tabs{padding-left:16px;padding-right:16px}td,thead th{width:170px;max-width:170px}.cell{width:150px;height:150px}tbody th{min-width:92px}}
</style>
</head>
<body>
<header><h1>Homebrew · <span>LoRA previews</span></h1><div class="meta" id="meta">loading…</div></header>
<nav class="tabs" id="tabs" role="tablist" aria-label="Training runs"></nav>
<main id="main"><div class="empty">Loading…</div></main>
<div id="lb" role="dialog" aria-modal="true" aria-label="Preview image">
  <div id="lbframe"></div>
  <div id="lbcap"></div>
  <div class="lbbar">
    <button data-act="prev-step">← earlier</button>
    <button data-act="next-step">later →</button>
    <button data-act="prev-prompt">↑ prompt</button>
    <button data-act="next-prompt">↓ prompt</button>
    <button data-act="compare" aria-pressed="false">compare with before (C)</button>
    <button data-act="close">close (Esc)</button>
  </div>
</div>
<script>
const S = {data: null, job: null, lb: null, compare: false, updated: 0, timer: null};
const $ = id => document.getElementById(id);
const DONE = ["completed", "failed", "stopped", "crashed", "cancelled"];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const current = () => S.data && S.data.jobs.find(j => j.job_id === S.job);
const isFinal = (job, set) => job.state === "completed" && set === job.sets[job.sets.length - 1] && set.step > 0;
const label = (job, set) => set.step === 0 ? "before" : (isFinal(job, set) ? "final" : "step " + set.step);

async function refresh() {
  try {
    const res = await fetch("api/jobs", {cache: "no-store"});
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    const changed = JSON.stringify(data.jobs) !== JSON.stringify(S.data && S.data.jobs);
    S.data = data;
    S.updated = Date.now();
    if (!current()) S.job = data.jobs.length ? data.jobs[data.jobs.length - 1].job_id : null;
    if (changed) render();
  } catch (e) {
    $("meta").textContent = "Homebrew is not answering — is it still running?";
  }
  const job = current();
  clearTimeout(S.timer);
  S.timer = setTimeout(refresh, job && !DONE.includes(job.state) ? 8000 : 60000);
}

function render() {
  const jobs = S.data.jobs;
  $("tabs").innerHTML = jobs.length > 1 ? jobs.map(j =>
    `<button class="tab" role="tab" aria-selected="${j.job_id === S.job}" data-job="${esc(j.job_id)}">${esc(j.job_id)} · ${esc(j.state || "?")}</button>`).join("") : "";
  const job = current();
  if (!job) {
    $("main").innerHTML = `<div class="empty">No image LoRA runs in <b>${esc(S.data.project)}</b> yet.<br>Previews show up here as soon as a Qwen-Image LoRA starts training.</div>`;
    return;
  }
  const live = !DONE.includes(job.state);
  const progress = job.max_steps ? ` · step ${job.step || 0} / ${job.max_steps}` : "";
  let html = `<div class="summary"><span><span class="dot ${live ? "live" : ""}"></span><b>${esc(job.state || "unknown")}</b>${progress}</span>
    <span>${esc(job.base_model || "")}</span>${job.trigger ? `<span>trigger <b>${esc(job.trigger)}</b></span>` : ""}
    <span>${job.sets.length} preview set${job.sets.length === 1 ? "" : "s"}</span></div>`;
  if (job.sync_error) html += `<p class="note">Could not reach the training server: ${esc(job.sync_error)}</p>`;
  if (job.preview_error) html += `<p class="note">Previews stopped: ${esc(job.preview_error)}</p>`;
  if (!job.sets.length) {
    html += `<div class="scroller"><div class="empty">Waiting for the first previews — Homebrew renders a “before” set right before training starts.</div></div>`;
  } else {
    html += `<div class="scroller"><table><thead><tr><th scope="col">checkpoint</th>` +
      job.prompts.map((p, k) => `<th scope="col" title="${esc(p)}">prompt ${k + 1}<div class="p">${esc(p)}</div></th>`).join("") +
      `</tr></thead><tbody>`;
    job.sets.forEach((set, i) => {
      const badge = set.step === 0 ? `<span class="badge">base model</span>` :
        isFinal(job, set) ? `<span class="badge">final LoRA</span>` :
        set.checkpoint ? `<span class="badge ok" title="Tell the brewmaster: package the checkpoint from step ${set.step}">✓ checkpoint saved</span>` : "";
      html += `<tr><th scope="row"><div class="step">${esc(label(job, set))}</div>${badge}</th>` +
        job.prompts.map((p, k) => {
          const src = set.images[k];
          return `<td><div class="cell">${src ? `<img src="${esc(src)}" loading="lazy" tabindex="0" alt="${esc(label(job, set))}: ${esc(p)}" data-s="${i}" data-p="${k}">` : "downloading…"}</div></td>`;
        }).join("") + `</tr>`;
    });
    html += `</tbody></table></div>
      <p class="hint">Click a picture to enlarge it. In the big view <kbd>←</kbd>/<kbd>→</kbd> move through the checkpoints for the same prompt,
      <kbd>↑</kbd>/<kbd>↓</kbd> switch prompts and <kbd>C</kbd> puts the “before” image next to it. Like one checkpoint best?
      Tell the brewmaster “package the checkpoint from step N”.</p>`;
  }
  $("main").innerHTML = html;
  if (S.lb) showLb();
}

function openLb(s, p) { S.lb = {s, p}; $("lb").classList.add("open"); showLb(); }
function closeLb() { S.lb = null; $("lb").classList.remove("open"); }
function showLb() {
  const job = current();
  if (!job || !S.lb) return closeLb();
  const s = Math.max(0, Math.min(S.lb.s, job.sets.length - 1)), p = Math.max(0, Math.min(S.lb.p, job.prompts.length - 1));
  S.lb = {s, p};
  const set = job.sets[s], src = set.images[p];
  const before = job.sets[0] && job.sets[0].step === 0 ? job.sets[0].images[p] : null;
  const fig = (url, cap) => url ? `<figure><img src="${esc(url)}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>` : "";
  const showCompare = S.compare && before && s > 0;
  $("lb").classList.toggle("compare", !!showCompare);
  $("lbframe").innerHTML = (showCompare ? fig(before, "before (base model)") : "") + (src ? fig(src, label(job, set)) : `<figure>downloading…</figure>`);
  $("lbcap").innerHTML = `<b>${esc(label(job, set))}</b> · prompt ${p + 1} of ${job.prompts.length} — ${esc(job.prompts[p])}`;
  document.querySelector('[data-act=compare]').setAttribute("aria-pressed", String(S.compare));
}
function moveLb(ds, dp) { if (S.lb) { S.lb = {s: S.lb.s + ds, p: S.lb.p + dp}; showLb(); } }

document.addEventListener("click", e => {
  const t = e.target;
  if (t.dataset && t.dataset.job) { S.job = t.dataset.job; render(); return; }
  if (t.matches && t.matches(".cell img")) { openLb(+t.dataset.s, +t.dataset.p); return; }
  const act = t.dataset && t.dataset.act;
  if (act === "prev-step") moveLb(-1, 0);
  else if (act === "next-step") moveLb(1, 0);
  else if (act === "prev-prompt") moveLb(0, -1);
  else if (act === "next-prompt") moveLb(0, 1);
  else if (act === "compare") { S.compare = !S.compare; showLb(); }
  else if (act === "close" || t.id === "lb") closeLb();
});
document.addEventListener("keydown", e => {
  if (!S.lb) {
    if (e.key === "Enter" && e.target.matches && e.target.matches(".cell img")) openLb(+e.target.dataset.s, +e.target.dataset.p);
    return;
  }
  const keys = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  if (keys[e.key]) { e.preventDefault(); moveLb(...keys[e.key]); }
  else if (e.key === "Escape") closeLb();
  else if (e.key === "c" || e.key === "C") { S.compare = !S.compare; showLb(); }
});
setInterval(() => {
  if (!S.updated) return;
  const s = Math.round((Date.now() - S.updated) / 1000);
  const job = current();
  $("meta").textContent = `${S.data.project}${job ? " · " + job.job_id : ""} · updated ${s < 2 ? "just now" : s + " s ago"}`;
}, 1000);
refresh();
</script>
</body>
</html>
"""
