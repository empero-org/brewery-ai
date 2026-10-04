"""``homebrew`` command line.

Interactive use:  ``homebrew`` (start or resume a project), ``homebrew new NAME``, ``homebrew setup``.
Utilities:        ``homebrew doctor``, ``homebrew hardware``, ``homebrew models``, ``homebrew etf ...``.
Worker commands:  ``homebrew train|chain|test|candidates|export|push`` — run on the training machine.

Worker commands import only the light training modules (no UI, no LLM SDKs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from homebrew_ai import __version__


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="homebrew", description="Homebrew by Empero — brew your own AI model.")
    p.add_argument("--version", action="version", version=f"homebrew-ai {__version__}")
    p.add_argument("--project", help="project folder (default: the current folder or its parents)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("chat", help="start or resume the guided session (default)")
    n = sub.add_parser("new", help="create a new project and start")
    n.add_argument("name", nargs="?")
    sub.add_parser("setup", help="choose or change the AI that guides you")
    sub.add_parser("doctor", help="check the installation, the guiding AI and this computer")
    h = sub.add_parser("hardware", help="show this computer's (or a server's) hardware")
    h.add_argument("--ssh", help="SSH command of a server to inspect instead")
    m = sub.add_parser("models", help="list supported base models")
    m.add_argument("--modality", choices=["text", "image"])
    m.add_argument("--all", action="store_true", help="include raw pretrained checkpoints")
    s = sub.add_parser("status", help="training jobs of this project")
    s.add_argument("job", nargs="?")

    etf = sub.add_parser("etf", help="work with ETF (Empero Trace Format) files")
    etf_sub = etf.add_subparsers(dest="etf_cmd")
    v = etf_sub.add_parser("validate", help="check an ETF .jsonl file")
    v.add_argument("file")
    st = etf_sub.add_parser("stats", help="statistics of an ETF file")
    st.add_argument("file")
    c = etf_sub.add_parser("convert", help="convert a dataset (HF id or local file) to ETF")
    c.add_argument("source", help="Hugging Face dataset id or a local file/folder")
    c.add_argument("--out", required=True)
    c.add_argument("--config")
    c.add_argument("--split", default="train")
    c.add_argument("--mapping", help="JSON mapping (see docs/etf.md); default: auto-detect")
    c.add_argument("--max-rows", type=int, default=None)
    etf_sub.add_parser("schema", help="print the ETF JSON Schema")

    # worker commands
    t = sub.add_parser("train", help="[worker] run a training job")
    t.add_argument("job")
    t.add_argument("--resume", action="store_true")
    ch = sub.add_parser("chain", help="[worker] run several stages back-to-back")
    ch.add_argument("jobs", nargs="+")
    ts = sub.add_parser("test", help="[worker] generate with a finished run")
    ts.add_argument("run_dir")
    ts.add_argument("--prompts", required=True)
    ts.add_argument("--out", required=True)
    ts.add_argument("--compare-base", action="store_true")
    ts.add_argument("--system")
    ts.add_argument("--thinking", action="store_true")
    ts.add_argument("--max-new-tokens", type=int, default=300)
    cd = sub.add_parser("candidates", help="[worker] sample several answers per prompt")
    cd.add_argument("run_dir")
    cd.add_argument("--prompts", required=True)
    cd.add_argument("--out", required=True)
    cd.add_argument("--n", type=int, default=2)
    cd.add_argument("--max-new-tokens", type=int, default=512)
    cd.add_argument("--thinking", action="store_true")
    ex = sub.add_parser("export", help="[worker] package a finished run")
    ex.add_argument("run_dir")
    ex.add_argument("--out", required=True)
    ex.add_argument("--merge", action="store_true")
    ex.add_argument("--readme")
    ex.add_argument("--checkpoint", type=int, help="image LoRAs: export checkpoints/checkpoint-N instead of final")
    pu = sub.add_parser("push", help="[worker] upload a folder to the Hugging Face Hub")
    pu.add_argument("folder")
    pu.add_argument("--repo", required=True)
    pu.add_argument("--card-repo", help="repo id the model card was written for (rewritten to --repo)")
    vis = pu.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true", default=True)
    vis.add_argument("--public", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")  # embedded terminals often lack cursor-position reports
    args = _parser().parse_args(argv)
    cmd = args.cmd or "chat"
    if cmd in ("chat", "new"):  # download bars would scribble over the chat; Homebrew shows its own progress
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    try:
        if cmd == "train":
            from homebrew_ai.train.runner import run

            return run(args.job, resume=args.resume)
        if cmd == "chain":
            from homebrew_ai.train.runner import run_chain

            return run_chain(args.jobs)
        if cmd == "test":
            from homebrew_ai.train.infer import main_test

            return main_test(args.run_dir, args.prompts, args.out, args.compare_base, args.system, args.thinking, args.max_new_tokens)
        if cmd == "candidates":
            from homebrew_ai.train.infer import main_candidates

            return main_candidates(args.run_dir, args.prompts, args.out, args.n, args.max_new_tokens, args.thinking)
        if cmd == "export":
            from homebrew_ai.train.export import export_run

            print(json.dumps(export_run(args.run_dir, args.out, merge=args.merge, readme=args.readme, checkpoint=args.checkpoint)))
            return 0
        if cmd == "push":
            from homebrew_ai.train.export import push_folder

            url = push_folder(args.folder, args.repo, private=not args.public, card_repo=args.card_repo)
            print(json.dumps({"url": url}))
            return 0
        if cmd == "etf":
            return _etf(args)
        if cmd == "models":
            return _models(args)
        if cmd == "hardware":
            return _hardware(args)
        if cmd == "doctor":
            return _doctor(args)
        if cmd == "status":
            return _status(args)
        if cmd == "setup":
            from homebrew_ai.settings import load_settings
            from homebrew_ai.ui.console import ConsoleUI
            from homebrew_ai.ui.setup import setup_backend

            setup_backend(ConsoleUI(), load_settings())
            return 0
        return _chat(args, new_name=getattr(args, "name", None), force_new=cmd == "new")
    except KeyboardInterrupt:
        print()
        return 130


def _chat(args, new_name: str | None, force_new: bool) -> int:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from homebrew_ai.backends.base import BackendError
    from homebrew_ai.backends.presets import make_backend
    from homebrew_ai.paths import find_project_root
    from homebrew_ai.project import Project
    from homebrew_ai.settings import load_settings
    from homebrew_ai.ui.console import ConsoleUI
    from homebrew_ai.ui.repl import run_repl
    from homebrew_ai.ui.setup import choose_level, create_project, setup_backend

    settings = load_settings()
    ui = ConsoleUI(color=settings.ui.color, emoji=settings.ui.emoji)
    if not settings.configured:
        ui.banner("first run")
        settings = setup_backend(ui, settings)
    root = Path(args.project).expanduser() if args.project else (None if force_new else find_project_root())
    if root is not None and (root / "homebrew.yaml").exists():
        project = Project.load(root)
    else:
        if not force_new and not args.project:
            ui.console.print("[dim]No Homebrew project here — let's start one.[/]")
        base = Path(args.project).expanduser() if args.project else Path.cwd()
        project = create_project(ui, base if args.project is None else base.parent, new_name or (base.name if args.project else None))
    if not project.state.level:
        project.state.level = choose_level(ui)
        project.save()
    try:
        backend = make_backend(settings.backend)
    except BackendError as exc:
        ui.error(str(exc))
        return 1
    run_repl(project, settings, backend, ui)
    return 0


def _etf(args) -> int:
    from homebrew_ai.etf.io import validate_file

    if args.etf_cmd == "validate":
        report = validate_file(args.file)
        print(json.dumps(report.to_dict(), indent=1))
        return 0 if report.invalid == 0 else 1
    if args.etf_cmd == "stats":
        from homebrew_ai.etf.stats import file_stats

        print(json.dumps(file_stats(args.file), indent=1))
        return 0
    if args.etf_cmd == "convert":
        from homebrew_ai.data.prepare import import_dataset

        out = Path(args.out)
        src = {"type": "local", "path": args.source} if Path(args.source).expanduser().exists() else {"type": "hf", "id": args.source, "config": args.config, "split": args.split}
        name = out.name.removesuffix(".jsonl")
        report = import_dataset(out.parent if str(out.parent) else Path("."), name, src, mapping=json.loads(args.mapping) if args.mapping else None, max_rows=args.max_rows)
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=1, default=str))
        return 0
    if args.etf_cmd == "schema":
        from importlib import resources

        print(resources.files("homebrew_ai.etf").joinpath("etf.schema.json").read_text(encoding="utf-8"))
        return 0
    print("usage: homebrew etf {validate,stats,convert,schema} ...")
    return 2


def _models(args) -> int:
    from rich.console import Console
    from rich.table import Table

    from homebrew_ai.models.registry import search, summarize

    rows = summarize(search(modality=args.modality, include_base=args.all))
    table = Table(title="Supported base models")
    for col in ("id", "params (B)", "license", "methods", "objectives", "notes"):
        table.add_column(col)
    for r in rows:
        notes = ", ".join(x for x in ("recommended" if r["recommended"] else "", "gated" if r["gated"] else "", r["kind"] if r["kind"] != "instruct" else "") if x)
        params = f"{r['params_b']}" + (f" ({r['active_params_b']} active)" if r.get("active_params_b") else "")
        table.add_row(r["id"], params, r["license"], ",".join(r["methods"]), ",".join(r["objectives"]), notes)
    Console().print(table)
    return 0


def _hardware(args) -> int:
    from homebrew_ai.agent.tools.compute import summarize_hardware

    if args.ssh:
        from homebrew_ai.remote.bootstrap import probe_remote
        from homebrew_ai.remote.sshutil import parse_ssh_command
        from homebrew_ai.remote.target import SSHTarget

        report = probe_remote(SSHTarget(parse_ssh_command(args.ssh)))
    else:
        from homebrew_ai.hardware.probe import probe

        report = probe(with_torch=True)
    print(json.dumps(summarize_hardware(report), indent=1))
    return 0


def _status(args) -> int:
    from homebrew_ai.jobs.manager import JobManager
    from homebrew_ai.paths import find_project_root
    from homebrew_ai.project import Project

    root = Path(args.project).expanduser() if args.project else find_project_root()
    if root is None:
        print("not inside a Homebrew project")
        return 1
    project = Project.load(root)
    jm = JobManager(project)
    for j in project.state.jobs if not args.job else [project.job(args.job)]:
        if j is None:
            continue
        st = jm.status(j["job_id"], log_lines=3)
        print(f"{j['job_id']}  {j.get('objective', 'sft')}/{j.get('method')}  {st.get('state')}  step {st.get('step', '-')}/{st.get('max_steps', '-')}  loss {st.get('loss', '-')}")
    return 0


def _doctor(args) -> int:
    from rich.console import Console

    from homebrew_ai.hardware.probe import probe
    from homebrew_ai.settings import load_settings

    con = Console()
    con.print(f"[bold]Homebrew {__version__}[/] · Python {sys.version.split()[0]}")
    settings = load_settings()
    if settings.backend:
        from homebrew_ai.backends.base import BackendError
        from homebrew_ai.backends.presets import make_backend

        try:
            make_backend(settings.backend).check()
            con.print(f"[green]✓[/] guiding AI: {settings.backend.preset} / {settings.backend.model}")
        except BackendError as exc:
            con.print(f"[red]✗[/] guiding AI: {exc}")
    else:
        con.print("[yellow]![/] no guiding AI configured yet (run `homebrew setup`)")
    try:
        from huggingface_hub import get_token

        con.print(("[green]✓[/] Hugging Face token found" if get_token() else "[yellow]![/] not logged in to Hugging Face (needed for gated models and uploads)"))
    except Exception:
        pass
    hw = probe(with_torch=True)
    gpus = hw.get("gpus") or []
    con.print(f"{'[green]✓[/]' if gpus else '[yellow]![/]'} GPUs: " + (", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in gpus) or "none — training will run on a rented server"))
    torch = hw.get("torch") or {}
    con.print(f"  PyTorch: {torch.get('version') or 'not installed'}" + (" (CUDA ok)" if torch.get("cuda_available") else ""))
    con.print(f"  RAM {hw.get('ram_gb')} GB · free disk {(hw.get('disk') or {}).get('free_gb')} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
