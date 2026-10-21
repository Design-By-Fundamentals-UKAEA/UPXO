"""Run folders, JSON audits and setting readers used by the stage handlers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from upxo.meshing.gbconformant.d3v2p0.paths import run_directory

DEFAULT_RUN_NAME = "conformal_run"


def subset_run_active(config: dict) -> bool:
    """True when the subset stage is on and has allocated its indexed run folder."""
    return bool(config.get("STAGE_SUBSET") and config.get("_RUN_FOLDER"))


def export_folder(config: dict, kind: str) -> Path:
    """Folder for one export of ``kind``: the run folder, or a new numbered subfolder (created) of a subset run."""
    run = ensure_run_dir(config)
    if not subset_run_active(config):
        return run
    from upxo.meshing.gbconformant.fm_steel_abaqus.subset import indexed_name, next_index

    base = run / kind
    base.mkdir(parents=True, exist_ok=True)
    folder = base / indexed_name(next_index(base))
    folder.mkdir()
    return folder


def current_run_dir(config: dict) -> Path:
    """The folder the settings point at; nothing is created."""
    if config.get("_OPEN_FOLDER"):                     # a state opened from a given run folder
        return Path(str(config["_OPEN_FOLDER"]))
    if subset_run_active(config):
        return Path(str(config["_RUN_FOLDER"]))
    name = str(config.get("RUN_NAME") or DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    return run_directory(name)


def ensure_run_dir(config: dict) -> Path:
    path = current_run_dir(config)
    if not config.get("_OPEN_FOLDER"):
        path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def spacing_tuple(config: dict) -> tuple[float, float, float]:
    spacing = config.get("SPACING", (1.0, 1.0, 1.0))
    arr = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    return float(arr[0]), float(arr[1]), float(arr[2])


def cfg_float(config: dict, key: str, default: float) -> float:
    """Read a float setting; treat missing/None as ``default``.

    Settings that may be left automatic hold an explicit ``None``, so
    ``config.get(key, default)`` alone would still return ``None``.
    """
    val = config.get(key, default)
    if val is None:
        val = default
    return float(val)


def cfg_int(config: dict, key: str, default: int) -> int:
    """Read an int setting; treat missing/None as ``default``."""
    val = config.get(key, default)
    if val is None:
        val = default
    return int(val)


def grain_summary(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    return {
        "shape": list(map(int, labels.shape)),
        "n_grains": int(len(ids)),
        "min_label": int(ids.min()) if len(ids) else None,
        "max_label": int(ids.max()) if len(ids) else None,
        "n_voxels": int(labels.size),
        "label_counts": {int(g): int(n) for g, n in zip(ids, counts)},
    }


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON-serialisable: {type(obj)!r}")
