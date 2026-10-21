"""Run manifest: the settings, inputs, seeds and environment a run used, for reproducibility."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upxo.meshing.gbconformant.pipeline.runfiles import ensure_run_dir, write_json
from upxo.meshing.gbconformant.pipeline.saved_results import (
    input_signature, signatures, store_for)
from upxo.meshing.gbconformant.pipeline.settings import CONFIG_VERSION, default_config, differences
from upxo.meshing.gbconformant.pipeline.stages import stage_ids

FORMAT = "upxo.gbconformant.run_manifest"
VERSION = 3
MANIFEST_NAME = "run_manifest.json"


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        pass
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "pyvista": _pkg_version("pyvista"),
        "matplotlib": _pkg_version("matplotlib"),
    }
    try:
        import gmsh  # noqa: F401
        out["gmsh"] = getattr(gmsh, "__version__", _pkg_version("gmsh") or "importable")
    except Exception:
        out["gmsh"] = None
    out["upxo"] = _pkg_version("upxo")
    return out


def _subset_record(cfg: dict) -> Any:
    """The crop description a subset run wrote into its folder, if any."""
    try:
        path = ensure_run_dir(cfg) / "subset.json"
        if cfg.get("STAGE_SUBSET") and path.is_file():
            return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        pass
    return None


def _saved_results(state) -> dict[str, Any]:
    """What the run folder holds: per stage its signature, status and size."""
    try:
        store = store_for(state)
        out = {}
        for stage in store.stages():
            meta = store.meta(stage) or {}
            out[stage] = {
                "status": meta.get("status"), "signature": meta.get("signature"),
                "bytes": meta.get("bytes"), "written_utc": meta.get("written_utc"),
                "artefacts": sorted(meta.get("artefacts", {})),
                "not_stored": sorted(meta.get("skipped", {})),
            }
        return {"enabled": bool(getattr(state, "checkpoints_enabled", False)),
                "current_signatures": signatures(state.config), "stages": out}
    except Exception as error:
        return {"error": str(error)}


RESULT_SETTINGS = ("TET_MESH_SIZE_FACTOR", "TET_MESH_SIZE", "GMSH_SIZE_FACTOR", "GMSH_SIZE", "FINAL_GMSH_SIZE",
                   "UPSCALE_FACTOR", "LAPLACE_ITERATIONS")


def _run_results(state) -> dict[str, Any]:
    """How big the run was and what it took, for the pre-run estimate and the run list.

    Sizes come from the latest stage that reported them, so a run stopped early holds what it reached.
    """
    reports = state.reports or {}
    out: dict[str, Any] = {"voxels": None, "grains": None, "tets": None, "nodes": None}
    try:
        labels = state.get_artefact("labels")
        if labels is not None:
            out["voxels"] = int(labels.size)
            out["grains"] = int(len(set(labels.ravel().tolist()) - {0}))
    except Exception:
        pass
    for sid in reversed(stage_ids()):
        report = reports.get(sid) or {}
        if report.get("n_tets"):
            out["tets"] = int(report["n_tets"])
            out["nodes"] = report.get("n_points")
            break
    durations = [r["duration_s"] for r in reports.values() if isinstance(r, dict) and "duration_s" in r]
    out["seconds"] = round(float(sum(durations)), 2) if durations else None
    peaks = [r["peak_memory_mb"] for r in reports.values() if isinstance(r, dict) and "peak_memory_mb" in r]
    out["peak_memory_mb"] = max(peaks) if peaks else None
    out["settings"] = {key: state.config.get(key) for key in RESULT_SETTINGS if key in state.config}
    return out


def build_run_manifest(state, *, defaults: dict | None = None) -> dict[str, Any]:
    """The manifest of ``state``'s run; settings are compared with ``defaults`` (the pipeline's own when omitted)."""
    cfg = state.config
    defaults = default_config() if defaults is None else defaults
    timings = []
    for sid, report in (state.reports or {}).items():
        if isinstance(report, dict) and "duration_s" in report:
            timings.append({"stage_id": sid, "duration_s": report["duration_s"]})
    return {
        "format": FORMAT,
        "version": VERSION,
        "config_version": CONFIG_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": cfg.get("RUN_NAME"),
        "stages": stage_ids(),
        "seeds": {"grain_id_seed": cfg.get("GRAIN_ID_SEED"),
                  "identity_seed": cfg.get("IDENTITY_SEED"),
                  "texture_unit_normalize": cfg.get("TEXTURE_UNIT_NORMALIZE")},
        "input": {
            "dataset_type": cfg.get("DATASET_TYPE"),
            "raw_folder": cfg.get("RAW_FOLDER"), "sample_path": cfg.get("SAMPLE_PATH"),
            "hierarchy_level": cfg.get("HIERARCHY_LEVEL"),
            "files": input_signature(cfg),
        },
        "subset": _subset_record(cfg),
        "saved_results": _saved_results(state),
        "config": dict(cfg),
        "config_diff_vs_defaults": differences(defaults, cfg),
        "stage_timings": timings,
        "results": _run_results(state),
        "environment": collect_environment(),
    }


def write_run_manifest(state, path: Path | str | None = None, *, manifest: dict | None = None) -> Path:
    """Write ``run_manifest.json`` under the run directory (or ``path``)."""
    if path is None:
        path = ensure_run_dir(state.config) / MANIFEST_NAME
    path = Path(path)
    write_json(path, manifest if manifest is not None else build_run_manifest(state))
    return path
