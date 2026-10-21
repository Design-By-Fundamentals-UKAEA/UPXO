"""The runs held in the data directory: what each one was and how far it got."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from upxo.meshing.gbconformant.d3v2p0.paths import data_directory
from upxo.meshing.gbconformant.pipeline.checkpoints import FOLDER as CHECKPOINTS, CheckpointStore
from upxo.meshing.gbconformant.pipeline.runfiles import DEFAULT_RUN_NAME
from upxo.meshing.gbconformant.pipeline.stages import STAGES

SUBSET_NAME = re.compile(r"^(?P<parent>.+)_subset_(?P<index>\d+)$")


@dataclass
class RunInfo:
    """One run folder."""

    name: str
    path: Path
    modified: float
    parent: Optional[str] = None             # the run a subset run was cut from, by name
    subset_index: Optional[int] = None
    dataset: Optional[str] = None
    level: Optional[str] = None
    subset_box: Optional[str] = None
    stages_saved: int = 0
    last_stage: Optional[str] = None
    last_stage_title: Optional[str] = None
    finished: bool = False                   # the Abaqus export stage was saved
    saved_bytes: int = 0
    has_manifest: bool = False
    has_report: bool = False
    voxels: Optional[int] = None
    tets: Optional[int] = None
    seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    disk_bytes: int = 0                      # everything in the folder
    history_bytes: int = 0                   # older copies of results that "drop history" would free
    older_export_bytes: int = 0              # earlier exports "delete older exports" would remove

    @property
    def older_exports_text(self) -> str:
        return _megabytes(self.older_export_bytes)

    @property
    def folder_text(self) -> str:
        return _megabytes(self.disk_bytes)

    @property
    def history_text(self) -> str:
        return _megabytes(self.history_bytes)

    @property
    def disk_text(self) -> str:
        """Disk use for a list: the folder, and how much of it is history."""
        text = _megabytes(self.disk_bytes)
        parts = ([f"{_megabytes(self.history_bytes)} history"] if self.history_bytes else []) + (
            [f"{_megabytes(self.older_export_bytes)} older exports"] if self.older_export_bytes else [])
        return text + (f" ({', '.join(parts)})" if parts else "")

    @property
    def size_text(self) -> str:
        """What the run came to, for a list: tetrahedra and total stage time."""
        from upxo.meshing.gbconformant.pipeline.estimate import duration_text
        parts = []
        if self.tets:
            parts.append(f"{self.tets:,} tets")
        if self.seconds:
            parts.append(duration_text(self.seconds))
        return ", ".join(parts)

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")

    @property
    def progress_text(self) -> str:
        if not self.stages_saved:
            return "nothing saved"
        if self.finished:
            return "finished"
        return f"stopped after {self.last_stage_title}"


def _megabytes(size: int) -> str:
    size = float(size)
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.1f} GB"
    return f"{size / 1024 ** 2:.0f} MB" if size >= 1024 ** 2 else f"{size / 1024:.0f} KB"


def folder_size(path) -> int:
    """Bytes of every file under ``path``."""
    total = 0
    for base, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def runs_directory(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else data_directory() / "runs"


def _box_text(subset: Optional[dict]) -> Optional[str]:
    box = ((subset or {}).get("box") or {}).get("index")
    if not box:
        return None
    lo, hi = box["lo"], box["hi"]
    return "x{}-{}  y{}-{}  z{}-{}".format(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _read_manifest(folder: Path) -> Optional[dict]:
    path = folder / "run_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def describe_run(folder: Path) -> RunInfo:
    """What is known about the run in ``folder`` (from its manifest and saved results)."""
    folder = Path(folder)
    info = RunInfo(name=folder.name, path=folder, modified=folder.stat().st_mtime)
    match = SUBSET_NAME.match(folder.name)
    if match:
        info.parent, info.subset_index = match["parent"], int(match["index"])
    manifest = _read_manifest(folder)
    info.has_manifest = manifest is not None
    info.has_report = (folder / "report" / "report.html").is_file()
    stages = {}
    if manifest:
        config = manifest.get("config") or {}
        info.dataset = config.get("DATASET_TYPE")
        info.level = config.get("HIERARCHY_LEVEL") or (manifest.get("input") or {}).get("hierarchy_level") or None
        info.subset_box = _box_text(manifest.get("subset"))
        stages = (manifest.get("saved_results") or {}).get("stages") or {}
        info.modified = max(info.modified, _stamp(manifest.get("created_utc")))
        from upxo.meshing.gbconformant.pipeline.estimate import observation_from_folder
        seen = observation_from_folder(folder, manifest)
        if seen is not None:
            info.voxels, info.tets, info.seconds, info.peak_memory_mb = (
                seen.voxels, seen.tets, seen.seconds, seen.peak_memory_mb)
    elif (folder / CHECKPOINTS).is_dir():
        stages = {p.name: {"bytes": 0} for p in (folder / CHECKPOINTS).iterdir()
                  if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").is_file()}
    info.disk_bytes = folder_size(folder)
    info.older_export_bytes = sum(folder_size(p) for p in older_exports(folder))
    if (folder / CHECKPOINTS).is_dir():
        try:
            info.history_bytes = CheckpointStore(folder).history_bytes([s.id for s in STAGES])
        except (OSError, ValueError, KeyError):
            info.history_bytes = 0
    order = [s for s in STAGES if s.id in stages]
    info.stages_saved = len(order)
    info.saved_bytes = sum(int((stages[s.id] or {}).get("bytes") or 0) for s in order)
    if order:
        info.last_stage, info.last_stage_title = order[-1].id, order[-1].title
        info.finished = order[-1].id == STAGES[-1].id
    return info


def _stamp(text) -> float:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def list_runs(root: Optional[Path] = None) -> list[RunInfo]:
    """Every run folder under ``root`` (default: the data directory), newest first.

    A folder counts as a run when it holds a manifest, saved stage results, or is named after the default run.
    """
    base = runs_directory(root)
    if not base.is_dir():
        return []
    runs = []
    for folder in base.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if not ((folder / "run_manifest.json").is_file() or (folder / CHECKPOINTS).is_dir()
                or folder.name == DEFAULT_RUN_NAME):
            continue
        runs.append(describe_run(folder))
    return sorted(runs, key=lambda r: -r.modified)


# --- housekeeping -------------------------------------------------------------------------------------
class RunDeletionError(ValueError):
    """A run folder was not deleted; the message says why."""


def is_run_folder(folder: Path) -> bool:
    folder = Path(folder)
    return folder.is_dir() and ((folder / "run_manifest.json").is_file() or (folder / CHECKPOINTS).is_dir()
                                or folder.name == DEFAULT_RUN_NAME)


def _writable(function, path, _info):
    os.chmod(path, stat.S_IWRITE)               # a read-only file does not stop the deletion of a run
    function(path)


def delete_run(folder, *, root: Optional[Path] = None, keep=None) -> int:
    """Delete a run folder and return the bytes freed.

    Only a folder directly inside the runs directory (``root``, by default the data directory's) that is a run
    (it holds a manifest or saved stage results) is deleted; the runs directory itself, any other folder and the
    run ``keep`` (the one that is loaded) are refused with :class:`RunDeletionError`.
    """
    base = runs_directory(root).resolve()
    target = Path(folder).resolve()
    if target == base:
        raise RunDeletionError("The runs folder itself cannot be deleted.")
    if target.parent != base:
        raise RunDeletionError(f"{target} is not a run folder inside {base}.")
    if keep is not None and Path(keep).resolve() == target:
        raise RunDeletionError("This run is the one currently loaded; load another run first.")
    if not is_run_folder(target):
        raise RunDeletionError(f"{target.name} does not look like a run (no manifest and no saved stage results).")
    size = folder_size(target)
    shutil.rmtree(target, onerror=_writable)
    return size


EXPORT_FOLDERS = ("abaqus", "surfaces", "volume")


def _export_stamp(name: str) -> str:
    for suffix in ("_fm", "_mesh"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def older_exports(folder) -> list:
    """The export folders of a run that a later export replaced.

    Each export writes a new numbered (or time-stamped) folder under ``abaqus``, ``surfaces`` and ``volume``.
    In each of these the newest export (a deck and its mesh-only deck count as one) is kept; the others are older.
    """
    older = []
    for name in EXPORT_FOLDERS:
        base = Path(folder) / name
        if not base.is_dir():
            continue
        groups = {}
        for child in base.iterdir():
            if child.is_dir():
                groups.setdefault(_export_stamp(child.name), []).append(child)
        for stamp in sorted(groups)[:-1]:
            older.extend(sorted(groups[stamp]))
    return older


def delete_older_exports(folder) -> int:
    """Delete the older exports of a run (see :func:`older_exports`); returns the bytes freed."""
    freed = 0
    for path in older_exports(folder):
        freed += folder_size(path)
        shutil.rmtree(path, onerror=_writable)
    return freed


def drop_run_history(folder) -> int:
    """Drop the older copies of results in a run's checkpoints; returns the bytes freed (see ``CheckpointStore.drop_history``)."""
    return CheckpointStore(Path(folder)).drop_history([s.id for s in STAGES])
