"""Input stages: load the dataset, crop it to a subset, map the grain ids."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from upxo.meshing.gbconformant.pipeline.rolling import current_labels, set_labels
from upxo.meshing.gbconformant.pipeline.runfiles import (
    DEFAULT_RUN_NAME, cfg_int, ensure_run_dir, grain_summary, spacing_tuple, write_json,
)
from upxo.meshing.gbconformant.pipeline.settings import clean_path_text
from upxo.meshing.gbconformant.pipeline.state import PipelineState

HIERARCHY_KEY = "load_lgi.hierarchy"
SUBSET_HIERARCHY_KEY = "subset_extract.hierarchy"
SAME_AS_SOURCE = "same as source"
DATASET_TYPES = ("fm_steel", "labels", "twinned_fcc")
HIERARCHY_TYPES = ("fm_steel", "twinned_fcc")    # datasets that carry orientations and parentage


def active_hierarchy(state: PipelineState):
    """The hierarchy the pipeline works on: the subset when one was extracted."""
    return state.get_artefact(SUBSET_HIERARCHY_KEY) or state.get_artefact(HIERARCHY_KEY)


def handle_load_lgi(state: PipelineState) -> dict:
    cfg = state.config
    dataset = str(cfg.get("DATASET_TYPE", "fm_steel"))
    if dataset == "fm_steel":
        return load_fm_steel(state)
    if dataset == "twinned_fcc":
        return load_twinned_fcc(state)
    path = Path(clean_path_text(cfg.get("SAMPLE_PATH"))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SAMPLE_PATH is not a file: {path}")
    labels = np.load(path, allow_pickle=False)
    if labels.ndim != 3 or labels.dtype.kind not in "iu":
        raise ValueError("LGI must be a 3D integer array")
    if np.any(labels < 0):
        raise ValueError("LGI labels must be nonnegative")
    spacing = spacing_tuple(cfg)
    cfg["SPACING"] = spacing
    set_labels(state, labels, also_before=True)
    state.store_artefact("labels_original", labels.copy())
    state.store_artefact("load_lgi.labels_full", labels.copy())
    state.store_artefact("grain_id_map", {})
    summary = grain_summary(labels)
    report = {
        "path": str(path.resolve()),
        "spacing": list(spacing),
        **{k: summary[k] for k in ("shape", "n_grains", "min_label", "max_label", "n_voxels")},
    }
    write_json(ensure_run_dir(cfg) / "load_lgi.json", report)
    return report


def load_fm_steel(state: PipelineState) -> dict:
    """Load one hierarchy level of an FM-steel raw export folder."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import detect_levels, load_hierarchy

    cfg = state.config
    raw = clean_path_text(cfg.get("RAW_FOLDER"))
    folder = Path(raw).expanduser()
    if not raw or not folder.is_dir():
        raise FileNotFoundError(f"RAW_FOLDER is not a folder: {folder}")
    levels = detect_levels(folder)
    if not levels:
        raise FileNotFoundError(
            f"No hierarchy level found in {folder}: expected <prefix>_lgi.npy "
            "with <prefix>_orientations.pkl.")
    level = str(cfg.get("HIERARCHY_LEVEL") or "") or levels[-1]
    if level not in levels:
        raise ValueError(f"Level {level!r} is not in the folder; available: {levels}")
    hierarchy = load_hierarchy(folder, level)
    cfg["HIERARCHY_LEVEL"] = level
    return install_hierarchy(state, hierarchy, folder, "fm_steel",
                             {"level": level, "available_levels": levels})


def load_twinned_fcc(state: PipelineState) -> dict:
    """Load a twinned FCC structure folder: one feature per grain, with its role and orientation."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import load_twinned

    cfg = state.config
    raw = clean_path_text(cfg.get("TWINNED_FOLDER"))
    folder = Path(raw).expanduser()
    if not raw or not folder.is_dir():
        raise FileNotFoundError(f"TWINNED_FOLDER is not a folder: {folder}")
    hierarchy = load_twinned(folder, float(cfg.get("TWINNED_VOXEL_SIZE", 1.0)),
                             str(cfg.get("TWINNED_UNITS", "microns")))
    roles = {name: len(labels) for name, labels in hierarchy.groups.items() if name.startswith("es_role_")}
    return install_hierarchy(state, hierarchy, folder, "twinned_fcc",
                             {"grains_by_role": roles,
                              "host_families": sum(1 for n in hierarchy.groups if n.startswith("es_family_"))})


def install_hierarchy(state: PipelineState, hierarchy, folder: Path, dataset: str, extra: dict) -> dict:
    """Make ``hierarchy`` the input: its label array, the hierarchy itself, and the load report."""
    cfg = state.config
    cfg["SPACING"] = (1.0, 1.0, 1.0)  # meshed in voxel units; units applied at export
    labels = hierarchy.lgi.copy()
    state.store_artefact("labels_before", labels.copy())
    state.store_artefact("labels", labels)
    state.store_artefact("labels_original", labels.copy())
    state.store_artefact("load_lgi.labels_full", labels.copy())
    state.store_artefact("grain_id_map", {})
    state.store_artefact(HIERARCHY_KEY, hierarchy)
    summary = grain_summary(labels)
    report = {
        "path": str(folder.resolve()),
        "dataset_type": dataset,
        **extra,
        "source_units": hierarchy.units,
        "voxel_size": hierarchy.voxel_size,
        **{k: summary[k] for k in ("shape", "n_grains", "min_label", "max_label", "n_voxels")},
    }
    write_json(ensure_run_dir(cfg) / "load_lgi.json", report)
    return report


def subset_spec(cfg: dict):
    """The box entered in the settings, as a :class:`BoxSpec`."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import BoxSpec

    units = str(cfg.get("SUBSET_UNITS") or SAME_AS_SOURCE)
    hi = cfg.get("SUBSET_HI")
    return BoxSpec(
        mode=str(cfg.get("SUBSET_MODE", "index")),
        lo=tuple(cfg.get("SUBSET_LO") or (0, 0, 0)),
        hi=None if hi is None else tuple(hi),
        units=None if units == SAME_AS_SOURCE else units)


def dataset_geometry(state: PipelineState):
    """``(parent labels, hierarchy or None, voxel size, source units)`` of the input dataset."""
    parent = state.get_artefact("labels_before_subset")
    if parent is None:
        parent = state.get_artefact("load_lgi.labels_full")
    if parent is None:
        parent = state.get_artefact("labels")
    hierarchy = state.get_artefact(HIERARCHY_KEY)
    if hierarchy is not None:
        return parent, hierarchy, hierarchy.voxel_size, hierarchy.units
    return parent, None, float(spacing_tuple(state.config)[0]), None


def crop_hierarchy_report(cfg: dict, hierarchy, labels) -> dict:
    """``{'counts': {'crop', 'parent'}, 'warnings'}``: PAGs, packets and blocks in the crop and in the structure.

    The warnings are for the levels the person set a minimum for (``RVE_MIN_PAGS``, ``RVE_MIN_PACKETS``,
    ``RVE_MIN_BLOCKS``); with none set there are none.
    """
    if hierarchy is None:
        return {"counts": None, "warnings": []}
    from upxo.meshing.gbconformant.fm_steel_abaqus.subset import crop_adequacy, hierarchy_counts

    counts = {"crop": hierarchy_counts(hierarchy, labels), "parent": hierarchy_counts(hierarchy)}
    minimums = {"pag": cfg.get("RVE_MIN_PAGS"), "packet": cfg.get("RVE_MIN_PACKETS"), "block": cfg.get("RVE_MIN_BLOCKS")}
    return {"counts": counts, "warnings": crop_adequacy(counts["crop"], minimums)}


def handle_subset_extract(state: PipelineState) -> dict:
    """Crop the input dataset to the entered box; later stages see only the crop."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import (
        cut_statistics, describe_box, extract_labels, extract_subset, new_indexed_directory,
        resolve_box, subset_record, write_subset_record)
    from upxo.meshing.gbconformant.d3v2p0.paths import run_directory

    cfg = state.config
    parent, hierarchy, voxel_size, units = dataset_geometry(state)
    if parent is None:
        raise RuntimeError("No labels artefact — run Load first.")
    if not cfg.get("STAGE_SUBSET"):
        return {"skipped": True, "reason": "Subset extract is switched off."}
    parent = np.asarray(parent)
    spec = subset_spec(cfg)
    lo, hi = resolve_box(spec, parent.shape, voxel_size, units)
    crop = extract_labels(parent, lo, hi)
    ids, counts = np.unique(crop, return_counts=True)
    parent_counts = (hierarchy.voxel_counts if hierarchy is not None else
                     dict(zip(*[a.tolist() for a in np.unique(parent, return_counts=True)])))
    stats = cut_statistics(parent_counts, dict(zip(ids.tolist(), counts.tolist())),
                           cfg_int(cfg, "MIN_GRAIN_VOXELS", 4))
    stats.update(crop_hierarchy_report(cfg, hierarchy, ids.tolist()))
    state.invalidate_after("subset_extract")     # later stages were built on another dataset
    state.store_artefact("labels_before_subset", parent.copy())
    state.store_artefact("labels_before", crop.copy())
    state.store_artefact("labels", crop)
    state.store_artefact("labels_original", crop.copy())
    state.store_artefact("grain_id_map", {})
    if hierarchy is not None:
        state.store_artefact(SUBSET_HIERARCHY_KEY, extract_subset(hierarchy, lo, hi))
    shown = None if units is None else (
        cfg.get("FM_OUTPUT_UNITS") if cfg.get("FM_OUTPUT_UNITS") not in (None, SAME_AS_SOURCE) else units)
    record = subset_record(spec, lo, hi, parent.shape, voxel_size, units, shown, stats)
    name = str(cfg.get("RUN_NAME") or DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    folder = new_indexed_directory(run_directory(name).parent, f"{name}_subset")
    cfg["_RUN_FOLDER"] = str(folder)
    write_subset_record(folder, record)
    return {
        "run_folder": str(folder),
        "origin_voxels": list(lo), "shape": list(crop.shape), "voxels": int(crop.size),
        "parent_shape": list(parent.shape), **{k: stats[k] for k in (
            "features_in_parent", "features_in_crop", "cut_features",
            "fragments_below_threshold", "smallest_in_crop", "counts", "warnings")},
    }


def handle_grain_id_map(state: PipelineState) -> dict:
    cfg = state.config
    if not cfg.get("STAGE_GRAIN_ID_SHUFFLE", False):
        return {"enabled": False, "skipped": True}
    labels_original = state.get_artefact("labels_original")
    if labels_original is None:
        labels_original = current_labels(state).copy()
        state.store_artefact("labels_original", labels_original)
    labels_original = np.asarray(labels_original)
    seed = cfg_int(cfg, "GRAIN_ID_SEED", 0)
    original_ids = np.unique(labels_original)
    shuffled = np.random.default_rng(seed).permutation(original_ids)
    mapping = {int(a): int(b) for a, b in zip(original_ids.tolist(), shuffled.tolist())}
    labels = shuffled[np.searchsorted(original_ids, labels_original)]
    set_labels(state, labels, also_before=True)
    state.store_artefact("grain_id_map", mapping)
    report = {
        "enabled": True,
        "seed": seed,
        "n_grains": int(len(original_ids)),
        "mapping_sample": dict(list(mapping.items())[:20]),
    }
    write_json(ensure_run_dir(cfg) / "grain_id_map.json",
               {"report": report, "mapping": mapping})
    return report
