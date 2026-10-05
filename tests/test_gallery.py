"""The local preview gallery: page, API, images, safety checks and remote sync."""

import http.client
import json
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from brewery_ai.ui.gallery import Gallery

from test_ssh_target import fake_ssh  # noqa: F401 - fixture


def _make_sets(root: Path, steps=(0, 250, 500), prompts=("sks dog on a sofa", "sks dog on a beach")):
    samples = root / "samples"
    for step in steps:
        folder = samples / f"step_{step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, _ in enumerate(prompts):
            Image.new("RGB", (16, 16), (step % 255, 80, 120)).save(folder / f"sample_{i:02d}.png")
    (samples / "index.json").write_text(json.dumps({"prompts": list(prompts), "sets": [{"step": s, "dir": f"samples/step_{s:06d}", "count": len(prompts)} for s in steps]}))
    (root / "status.json").write_text(json.dumps({"state": "training", "step": 600, "max_steps": 1000}))
    (root / "checkpoints" / "checkpoint-500").mkdir(parents=True, exist_ok=True)
    (root / "job.yaml").write_text("image:\n  trigger_word: sks\n")


def _get(url: str, path: str = "", host: str | None = None) -> tuple[int, bytes, str]:
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request("GET", u.path + path, headers={"Host": host or f"{u.hostname}:{u.port}"})
    res = conn.getresponse()
    return res.status, res.read(), res.getheader("Content-Type") or ""


def test_gallery_serves_page_api_and_images_safely(project):
    run_dir = project.runs_dir / "img1"
    _make_sets(run_dir)
    project.state.jobs.append({"job_id": "img1", "run_dir": str(run_dir), "modality": "image", "base_model": "Qwen/Qwen-Image-2.1", "target_spec": {"kind": "local"}, "state": "training"})
    project.save()
    g = Gallery(project)
    url = g.start(port=0)
    try:
        status, body, ctype = _get(url)
        assert status == 200 and "text/html" in ctype and b"LoRA previews" in body
        status, body, _ = _get(url, "api/jobs")
        job = json.loads(body)["jobs"][0]
        assert job["trigger"] == "sks" and job["prompts"] == ["sks dog on a sofa", "sks dog on a beach"]
        assert [s["step"] for s in job["sets"]] == [0, 250, 500] and [s["checkpoint"] for s in job["sets"]] == [False, False, True]
        status, body, ctype = _get(url, job["sets"][1]["images"][0])
        assert status == 200 and ctype == "image/png" and body[:4] == b"\x89PNG"
        assert _get(url, "img/img1/../status.json")[0] == 404  # nothing outside samples/, only images
        assert _get(url, "img/img1/../../../brewery.yaml")[0] == 404
        assert _get(url.replace(g.token, "wrong-token"))[0] == 404
        assert _get(url, host="evil.example:80")[0] == 403  # DNS rebinding
    finally:
        g.stop()


def test_gallery_downloads_new_sets_from_a_remote_run(project, fake_ssh, tmp_path):  # noqa: F811
    remote = tmp_path / "server" / "runs" / "img2"
    _make_sets(remote, steps=(0, 250))
    run_dir = project.runs_dir / "img2"
    run_dir.mkdir(parents=True)
    (run_dir / "job.yaml").write_text("image:\n  trigger_word: sks\n")
    project.state.jobs.append({
        "job_id": "img2", "run_dir": str(run_dir), "modality": "image", "base_model": "Qwen/Qwen-Image-2.1",
        "target_spec": fake_ssh.to_dict(), "remote_run_dir": str(remote), "state": "training",
    })
    project.save()
    g = Gallery(project)
    view = g.state()["jobs"][0]
    assert view["remote"] and view["sync_error"] is None and view["step"] == 600
    assert [s["step"] for s in view["sets"]] == [0, 250] and all(view["sets"][1]["images"])
    assert (run_dir / "samples/step_000250/sample_01.png").exists()
    assert {s["step"]: s["checkpoint"] for s in view["sets"]} == {0: False, 250: False}
