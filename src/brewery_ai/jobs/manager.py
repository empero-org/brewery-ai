"""Launching and following training jobs, locally or on an SSH server.

Each job lives in ``<project>/runs/<job_id>/`` locally. For remote jobs the
same folder is mirrored to ``<remote root>/runs/<job_id>/`` and results are
fetched back on demand. Jobs survive the REPL being closed: the worker runs
detached and reports through ``status.json``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from brewery_ai.project import Project
from brewery_ai.remote import bootstrap
from brewery_ai.remote.target import LocalTarget, SSHTarget, target_from_dict
from brewery_ai.train.config import TrainJob
from brewery_ai.train.status import STOP_FILE, TERMINAL, read_status


FINISHED = ("completed", "failed", "stopped", "crashed", "cancelled")
RUNNING = ("preparing", "loading_model", "training", "saving", "stopping")


def active_job(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The job the user most likely means: the newest running one, else the newest unfinished one, else the latest."""
    for j in reversed(jobs):
        if j.get("state") in RUNNING:
            return j
    for j in reversed(jobs):
        if j.get("state") not in FINISHED:
            return j
    return jobs[-1] if jobs else None


def _pid_alive(pid: int | None) -> bool:
    """Is ``pid`` still one of our training processes? (PIDs get reused after a restart.)"""
    if not pid:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        try:
            line = " ".join(proc.cmdline())
            return "brewery_ai" in line or "homebrew_ai" in line
        except psutil.AccessDenied:
            return True
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class JobManager:
    def __init__(self, project: Project):
        self.project = project

    # -- targets -------------------------------------------------------------

    def target(self, record: dict[str, Any] | None = None) -> LocalTarget | SSHTarget:
        if record and record.get("target_spec"):
            return target_from_dict(record["target_spec"])
        compute = self.project.state.compute
        if compute.kind == "ssh" and compute.ssh:
            return target_from_dict({"kind": "ssh", **compute.ssh})
        return LocalTarget()

    def local_run_dir(self, job_id: str) -> Path:
        return self.project.runs_dir / job_id

    # -- staging -------------------------------------------------------------

    def stage(self, job: TrainJob, train_file: Path, eval_file: Path | None, images_dir: Path | None = None) -> Path:
        """Create ``runs/<job_id>/`` with job.yaml and a copy of the data."""
        run_dir = self.local_run_dir(job.job_id)
        (run_dir / "data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_file, run_dir / job.data.train)
        if eval_file and job.data.eval:
            shutil.copy2(eval_file, run_dir / job.data.eval)
        if images_dir is not None and images_dir.is_dir():
            dest = run_dir / Path(job.data.train).parent / "images"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(images_dir, dest)
        job.save(run_dir / "job.yaml")
        return run_dir

    # -- launching -----------------------------------------------------------

    def launch(self, job: TrainJob, *, hf_token: str | None = None) -> dict[str, Any]:
        return self.launch_chain([job], hf_token=hf_token)[0]

    def launch_chain(self, jobs: list[TrainJob], *, hf_token: str | None = None) -> list[dict[str, Any]]:
        """Start one job, or several stages that run back-to-back (``brewery chain``)."""
        for job in jobs:
            if not (self.local_run_dir(job.job_id) / "job.yaml").exists():
                raise FileNotFoundError(f"job {job.job_id} is not staged")
        target = self.target()
        chain_id = jobs[0].job_id
        nproc = max(jobs[0].runtime.num_gpus, 1)
        job_files = [f"{j.job_id}/job.yaml" for j in jobs]
        verb = ["train", job_files[0]] if len(jobs) == 1 else ["chain", *job_files]
        log_name = f"{chain_id}/train.log" if len(jobs) == 1 else f"chain-{chain_id}.log"
        base_record = {
            "base_model": jobs[0].base_model,
            "modality": jobs[0].modality,
            "target": target.label,
            "target_spec": target.to_dict(),
            "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": "queued",
            "chain_id": chain_id if len(jobs) > 1 else None,
            "num_gpus": nproc,
        }
        if isinstance(target, LocalTarget):
            runs = self.project.runs_dir
            cmd = [sys.executable, "-m", "brewery_ai", *verb]
            if nproc > 1:
                cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", "-m", "brewery_ai", *verb]
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
            if hf_token:
                env["HF_TOKEN"] = hf_token
            log = open(runs / log_name, "ab")
            kwargs: dict[str, Any] = {"cwd": runs, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **kwargs)
            extra = {"pid": proc.pid, "log": str(runs / log_name)}
        else:
            root = self.project.state.compute.remote_root or bootstrap.remote_root(target)
            pkg = bootstrap.sync_worker(target, root)  # the worker always matches this version of Brewery
            for job in jobs:
                target.put(self.local_run_dir(job.job_id), f"{root}/runs/{job.job_id}")
            launcher = "-m brewery_ai"
            if nproc > 1:
                launcher = f"-m torch.distributed.run --nproc_per_node={nproc} -m brewery_ai"
            prefix = bootstrap.worker_prefix(root, pkg).replace("-m brewery_ai", launcher)
            inner = prefix + " " + " ".join(shlex.quote(v) for v in verb)
            cmd = (
                f"cd {shlex.quote(root + '/runs')} && IFS= read -r HF_TOKEN; export HF_TOKEN; "
                f'[ -n "$HF_TOKEN" ] || unset HF_TOKEN; '
                f"nohup setsid bash -c {shlex.quote(inner)} > {shlex.quote(log_name)} 2>&1 < /dev/null & echo $!"
            )
            res = target.run(cmd, input=(hf_token or "") + "\n", timeout=300)
            res.raise_for_error("starting the training job")
            extra = {"pid": int(res.stdout.strip().splitlines()[-1]), "remote_root": root, "remote_log": f"{root}/runs/{log_name}"}
        records = []
        for job in jobs:
            record = {
                **base_record,
                "job_id": job.job_id,
                "objective": job.objective,
                "method": job.method,
                "stage": job.stage,
                "init_from": job.init_from.job_id if job.init_from else None,
                "run_dir": str(self.local_run_dir(job.job_id)),
                **extra,
            }
            if isinstance(target, SSHTarget):
                record["remote_run_dir"] = f"{extra['remote_root']}/runs/{job.job_id}"
            self.project.state.jobs.append(record)
            records.append(record)
        self.project.save()
        return records

    # -- status --------------------------------------------------------------

    def _remote_status(self, target: SSHTarget, record: dict[str, Any], log_lines: int) -> tuple[dict | None, bool, str]:
        rd = shlex.quote(record["remote_run_dir"])
        pid = int(record.get("pid") or 0)
        script = (
            f"cat {rd}/status.json 2>/dev/null || echo '{{}}'; echo '<<HB>>'; "
            # alive = the PID exists AND is still ours (after a server restart PIDs get reused)
            f"if kill -0 {pid} 2>/dev/null && tr '\\0' ' ' < /proc/{pid}/cmdline 2>/dev/null | grep -qE 'brewery_ai|homebrew_ai'; "
            f"then echo alive; else echo dead; fi; echo '<<HB>>'; "
            f"tail -n {int(log_lines)} {shlex.quote(record.get('remote_log') or record['remote_run_dir'] + '/train.log')} 2>/dev/null"
        )
        res = target.run(script, timeout=60)
        if not res.ok:
            raise RuntimeError(f"could not reach {target.label}: {(res.stderr or res.stdout).strip()[-300:]}")
        parts = res.stdout.split("<<HB>>")
        try:
            status = json.loads(parts[0].strip() or "{}") or None
        except ValueError:
            status = None
        alive = len(parts) > 1 and parts[1].strip() == "alive"
        log = parts[2].strip("\n") if len(parts) > 2 else ""
        return status, alive, log

    def status(self, job_id: str | None = None, log_lines: int = 15) -> dict[str, Any]:
        record = self.project.job(job_id)
        if record is None:
            raise KeyError("no training job yet" if job_id is None else f"unknown job {job_id}")
        target = self.target(record)
        if isinstance(target, SSHTarget):
            status, alive, log = self._remote_status(target, record, log_lines)
        else:
            run_dir = Path(record["run_dir"])
            status = read_status(run_dir)
            alive = _pid_alive(record.get("pid"))
            log_path = Path(record.get("log") or run_dir / "train.log")
            log = ""
            if log_path.exists():
                with open(log_path, "rb") as fh:
                    fh.seek(0, 2)
                    size = fh.tell()
                    fh.seek(max(0, size - 6000))
                    log = "\n".join(fh.read().decode(errors="replace").splitlines()[-log_lines:])
        state = (status or {}).get("state", "queued")
        if state not in TERMINAL and not alive:
            state = "crashed" if status else ("cancelled" if record.get("chain_id") else "crashed")
        status = dict(status or {})
        status["state"] = state
        status["alive"] = alive
        status["log_tail"] = log
        status.pop("loss_history_full", None)
        if record.get("state") != state:
            self.project.update_job(record["job_id"], state=state)
        return status

    def stop(self, job_id: str | None = None) -> str:
        """Ask a job (and the rest of its chain) to save a checkpoint and stop.

        The STOP file works everywhere (Windows, several GPUs); single-GPU jobs also get SIGTERM so a
        stop is noticed even while the model is still loading.
        """
        record = self.project.job(job_id)
        if record is None:
            raise KeyError("no training job to stop")
        pid = int(record.get("pid") or 0)
        signal_too = pid and int(record.get("num_gpus") or 1) <= 1  # torchrun would SIGKILL workers mid-save
        chain = [j for j in self.project.state.jobs if record.get("chain_id") and j.get("chain_id") == record["chain_id"]] or [record]
        target = self.target(record)
        if isinstance(target, SSHTarget):
            cmd = "; ".join(f"touch {shlex.quote(j['remote_run_dir'] + '/' + STOP_FILE)} 2>/dev/null" for j in chain if j.get("remote_run_dir"))
            cmd += f"; test -d {shlex.quote(record['remote_run_dir'])}"
            if signal_too:
                cmd += f"; kill -TERM -- -{pid} 2>/dev/null || kill -TERM {pid} 2>/dev/null; pkill -TERM -P {pid} 2>/dev/null; true"
            res = target.run(cmd, timeout=60)
            if not res.ok:
                raise RuntimeError(f"could not reach {target.label} to stop the job (it may still be running and billing): {(res.stderr or res.stdout).strip()[-300:]}")
        else:
            for j in chain:
                run_dir = Path(j["run_dir"])
                if run_dir.is_dir():
                    (run_dir / STOP_FILE).touch()
            if signal_too and sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
        self.project.update_job(record["job_id"], state="stopping")
        return record["job_id"]

    def adopt_server(self, target: SSHTarget, root: str) -> list[str]:
        """Re-point job records at ``target`` when their run folders live there.

        Rented pods often come back with a new IP/port after a restart; without this, status, fetch and
        packaging for existing jobs would keep dialling the old address.
        """
        ids = [j["job_id"] for j in self.project.state.jobs if (j.get("target_spec") or {}).get("kind") == "ssh"]
        if not ids:
            return []
        script = "; ".join(f"[ -d {shlex.quote(root + '/runs/' + jid)} ] && echo {shlex.quote(jid)}" for jid in ids) + "; true"
        res = target.run(script, timeout=60)
        found = set(res.stdout.splitlines()) if res.ok else set()
        spec = target.to_dict()
        moved = []
        for j in self.project.state.jobs:
            if j["job_id"] in found and j.get("target_spec") != spec:
                j.update(target_spec=spec, target=target.label, remote_root=root, remote_run_dir=f"{root}/runs/{j['job_id']}")
                if j.get("remote_log"):
                    j["remote_log"] = f"{root}/runs/{j['remote_log'].rsplit('/', 1)[-1]}"
                moved.append(j["job_id"])
        if moved:
            self.project.save()
        return moved

    # -- results -------------------------------------------------------------

    def fetch(self, job_id: str | None = None, what: str = "final") -> Path:
        """Copy ``what`` (e.g. ``final``, ``samples``, ``results.json``) from the run to this computer."""
        record = self.project.job(job_id)
        if record is None:
            raise KeyError("no training job")
        local = Path(record["run_dir"]) / what
        target = self.target(record)
        if isinstance(target, SSHTarget):
            target.get(f"{record['remote_run_dir']}/{what}", local)
        return local

    def put_file(self, job_id: str | None, local_file: Path, name: str) -> None:
        """Place a file into the run folder (on whichever machine holds the run)."""
        record = self.project.job(job_id)
        if record is None:
            raise KeyError("no training job")
        target = self.target(record)
        if isinstance(target, SSHTarget):
            target.put(local_file, f"{record['remote_run_dir']}/{name}")
        else:
            dest = Path(record["run_dir"]) / name
            if local_file.resolve() != dest.resolve():
                shutil.copy2(local_file, dest)

    def run_worker(self, job_id: str | None, args: list[str], *, stdin: str | None = None, timeout: float = 3600) -> str:
        """Run ``python -m brewery_ai <args>`` inside the run folder on the job's machine."""
        record = self.project.job(job_id)
        if record is None:
            raise KeyError("no training job")
        target = self.target(record)
        if isinstance(target, SSHTarget):
            root = record.get("remote_root") or self.project.state.compute.remote_root
            prefix = bootstrap.worker_prefix(root, bootstrap.sync_worker(target, root))
            secret = "IFS= read -r HF_TOKEN; export HF_TOKEN; [ -n \"$HF_TOKEN\" ] || unset HF_TOKEN; "
            cmd = f"cd {shlex.quote(record['remote_run_dir'])} && {secret}{prefix} " + " ".join(shlex.quote(a) for a in args)
            res = target.run(cmd, input=(stdin or "") + "\n", timeout=timeout)
        else:
            env = dict(os.environ, PYTHONUTF8="1")
            if stdin:
                env["HF_TOKEN"] = stdin
            proc = subprocess.run([sys.executable, "-m", "brewery_ai", *args], cwd=record["run_dir"], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
            res = type("R", (), {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr, "code": proc.returncode})()
        if not res.ok:
            tail = (res.stderr or res.stdout).strip().splitlines()[-20:]
            raise RuntimeError(f"`brewery {' '.join(args[:2])}` failed on {target.label}:\n" + "\n".join(tail))
        return res.stdout
