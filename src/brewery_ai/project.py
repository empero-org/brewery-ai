"""A Brewery project: one folder per model being brewed.

::

    my-pirate-bot/
      brewery.yaml        project state (what the agent knows and decided)
      data/                ETF datasets (.jsonl) + their manifests (.meta.json)
      runs/<job_id>/       training runs (job.yaml, data copy, logs, adapters)
      export/<name>/       bottled models ready for the Hub
      .brewery/           conversation history for resuming
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from brewery_ai import __version__
from brewery_ai.paths import PROJECT_FILE, SESSION_DIR, migrate_project

PHASES = ("welcome", "goal", "compute", "model", "data", "config", "train", "evaluate", "package", "publish", "done")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class ComputeState(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str | None = None  # local | ssh
    ssh: dict[str, Any] | None = None
    provider: str | None = None
    remote_root: str | None = None
    prepared: bool = False
    hardware: dict[str, Any] | None = None


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    kind: str = "trace"
    records: int = 0
    sources: list[dict[str, Any]] = Field(default_factory=list)
    synthetic: bool = False
    created: str = Field(default_factory=_now)


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    created: str = Field(default_factory=_now)
    brewery_version: str = Field(default=__version__, validation_alias=AliasChoices("brewery_version", "homebrew_version"))
    level: str | None = None
    phase: str = "welcome"
    goal: str | None = None
    modality: str | None = None
    notes: list[str] = Field(default_factory=list)
    compute: ComputeState = Field(default_factory=ComputeState)
    base_model: str | None = None
    datasets: dict[str, DatasetEntry] = Field(default_factory=dict)
    training_sets: dict[str, dict[str, Any]] = Field(default_factory=dict)  # name -> train/eval paths, counts, kind, token stats
    drafts: dict[str, dict[str, Any]] = Field(default_factory=dict)  # job_id -> TrainJob dump (not started yet)
    regime: list[dict[str, Any]] = Field(default_factory=list)  # planned stages: {"objective", "training_set", "status", "job_id"}
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    exports: list[dict[str, Any]] = Field(default_factory=list)
    uploads: list[dict[str, Any]] = Field(default_factory=list)
    synthetic_terms_ack: bool = False


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-._")
    return slug[:64] or "my-brew"


class Project:
    def __init__(self, root: Path, state: ProjectState):
        self.root = root
        self.state = state

    # -- persistence ---------------------------------------------------------

    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> "Project":
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / PROJECT_FILE).exists():
            return cls.load(root)
        project = cls(root, ProjectState(name=name or root.name))
        for sub in ("data", "runs", "export", SESSION_DIR):
            (root / sub).mkdir(exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        root = Path(root).expanduser().resolve()
        migrate_project(root)
        data = yaml.safe_load((root / PROJECT_FILE).read_text(encoding="utf-8")) or {}
        return cls(root, ProjectState.model_validate(data))

    def save(self) -> None:
        path = self.root / PROJECT_FILE
        tmp = path.with_suffix(".tmp")
        header = "# Brewery project state. Edited by the Brewery agent; safe to read, careful when editing.\n"
        tmp.write_text(header + yaml.safe_dump(self.state.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(path)

    # -- paths ---------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def session_dir(self) -> Path:
        path = self.root / SESSION_DIR
        path.mkdir(exist_ok=True)
        return path

    def rel(self, path: str | Path) -> str:
        path = Path(path).expanduser().resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def abs(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.root / p

    # -- helpers -------------------------------------------------------------

    def job(self, job_id: str | None = None) -> dict[str, Any] | None:
        if not self.state.jobs:
            return None
        if job_id is None:
            return self.state.jobs[-1]
        return next((j for j in self.state.jobs if j["job_id"] == job_id), None)

    def update_job(self, job_id: str, **fields: Any) -> None:
        job = self.job(job_id)
        if job is not None:
            job.update(fields)
            self.save()

    def summary(self) -> dict[str, Any]:
        """Compact state for the agent's context (keeps tokens low)."""
        s = self.state
        out: dict[str, Any] = {
            "project": s.name,
            "phase": s.phase,
            "level": s.level,
            "goal": s.goal,
            "modality": s.modality,
            "base_model": s.base_model,
            "compute": {k: v for k, v in s.compute.model_dump().items() if v not in (None, False, {}) and k != "hardware"},
            "datasets": {k: {"path": v.path, "records": v.records, "kind": v.kind, "synthetic": v.synthetic} for k, v in s.datasets.items()},
            "training_sets": {k: {kk: v.get(kk) for kk in ("kind", "num_train", "num_eval", "objectives")} for k, v in s.training_sets.items()},
        }
        if s.compute.hardware:
            hw = s.compute.hardware
            out["compute"]["gpus"] = [f"{g.get('name')} ({g.get('vram_gb')} GB)" for g in hw.get("gpus", [])] or "none"
        if s.drafts:
            out["drafts"] = [
                {"job_id": d.get("job_id"), "objective": d.get("objective"), "method": d.get("method"), "init_from": (d.get("init_from") or {}).get("job_id")}
                for d in list(s.drafts.values())[-4:]
            ]
        if s.regime:
            out["regime"] = s.regime
        if s.jobs:
            out["jobs"] = [{k: j.get(k) for k in ("job_id", "state", "objective", "method", "target")} for j in s.jobs[-4:]]
        if s.exports:
            out["exports"] = s.exports[-2:]
        if s.uploads:
            out["uploads"] = s.uploads[-2:]
        if s.notes:
            out["notes"] = s.notes[-10:]
        return {k: v for k, v in out.items() if v not in (None, {}, [])}
