"""Cleaning stages: small grains, spikes, pinches, voxel topology, thin grains, upscale."""
from __future__ import annotations

import numpy as np

from upxo.meshing.gbconformant.pipeline.rolling import current_labels, set_labels
from upxo.meshing.gbconformant.pipeline.runfiles import (
    cfg_float, cfg_int, ensure_run_dir, spacing_tuple, write_json,
)
from upxo.meshing.gbconformant.pipeline.state import PipelineState


def handle_small_grains(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.small_grains import remove_small_grains

    cfg = state.config
    enabled = bool(cfg.get("STAGE_SMALL_GRAINS", True) and cfg.get("REMOVE_SMALL_GRAINS", True))
    labels_in = current_labels(state)
    state.store_artefact("labels_before_small_grains", labels_in.copy())
    labels_out, removal_report = remove_small_grains(
        labels_in,
        enabled=enabled,
        min_voxels=cfg_int(cfg, "MIN_GRAIN_VOXELS", 4),
        protected_ids=tuple(cfg.get("PROTECTED_GRAIN_IDS") or ()),
        protect_rve=bool(cfg.get("PROTECT_RVE_GRAINS", False)),
        max_passes=cfg_int(cfg, "REMOVAL_MAX_PASSES", 10),
        selection=str(cfg.get("RECIPIENT_SELECTION", "shared_area")),
        spacing=spacing_tuple(cfg),
    )
    set_labels(state, labels_out)
    state.store_artefact("small_grains.survivor_map",
                         dict(removal_report["original_to_survivor"]))
    report = {
        "enabled": enabled,
        "n_grains_before": int(len(np.unique(labels_in))),
        "n_grains_after": int(len(np.unique(labels_out))),
        **removal_report,
    }
    write_json(ensure_run_dir(cfg) / "small_grains.json",
               {"report": report})
    return report


def handle_spikes(state: PipelineState) -> dict:
    cfg = state.config
    enabled = bool(cfg.get("STAGE_SPIKES", True) and cfg.get("CLEAN_SPIKES", True))
    labels_in = current_labels(state).copy()
    state.store_artefact("labels_before_spikes", labels_in.copy())
    if enabled:
        from upxo.meshing.gbconformant.d3v2p0.spike_cleaning import clean_spikes

        labels_cleaned, result = clean_spikes(
            labels_in, max_passes=cfg_int(cfg, "SPIKE_MAX_PASSES", 100),
            progress=state.report_progress)
    else:
        labels_cleaned = labels_in
        result = {"stop_reason": "disabled", "history": [], "remaining_candidates": None,
                  "net_changed_voxels": 0}
    set_labels(state, labels_cleaned)
    report = {"enabled": enabled, **result}
    write_json(ensure_run_dir(cfg) / "spike_cleaning.json", report)
    return report


def handle_pinches(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.pinch_cleaning import clean_pinch_contacts

    cfg = state.config
    enabled = bool(cfg.get("STAGE_PINCHES", True) and cfg.get("CLEAN_PINCHES", True))
    labels_in = current_labels(state)
    state.store_artefact("labels_before_pinches", labels_in.copy())
    labels_out, pinch_report, pinch_changes = clean_pinch_contacts(
        labels_in,
        enabled=enabled,
        max_passes=cfg_int(cfg, "PINCH_MAX_PASSES", 100),
        max_changes=cfg_int(cfg, "PINCH_MAX_CHANGES", 200),
        max_volume_fraction=cfg_float(cfg, "PINCH_MAX_VOLUME_FRACTION", 0.03),
        protect_grains_up_to=cfg_int(cfg, "PINCH_PROTECT_GRAINS_UP_TO", 8),
        progress=state.report_progress,
    )
    set_labels(state, labels_out)
    report = dict(pinch_report)
    report["enabled"] = enabled
    write_json(ensure_run_dir(cfg) / "pinch_cleaning.json",
               {"report": report, "changes": pinch_changes})
    return report


def handle_voxel_topology(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.voxel_topology import clean_voxel_topology

    cfg = state.config
    enabled = bool(cfg.get("STAGE_VOXEL_TOPOLOGY", True)
                   and cfg.get("CLEAN_VOXEL_TOPOLOGY", True))
    labels_in = current_labels(state)
    state.store_artefact("labels_before_topology", labels_in.copy())
    labels_out, topology_report, topology_changes = clean_voxel_topology(
        labels_in,
        enabled=enabled,
        max_passes=cfg_int(cfg, "TOPOLOGY_MAX_PASSES", 20),
        max_changes=cfg_int(cfg, "TOPOLOGY_MAX_CHANGES", 5000),
        max_volume_fraction=cfg_float(cfg, "TOPOLOGY_MAX_VOLUME_FRACTION", 0.20),
        protect_grains_up_to=cfg_int(cfg, "TOPOLOGY_PROTECT_GRAINS_UP_TO", 0),
        preserve_components=bool(cfg.get("TOPOLOGY_PRESERVE_COMPONENTS", False)),
        max_pair_trials=cfg_int(cfg, "TOPOLOGY_MAX_PAIR_TRIALS", 20000),
        progress=state.report_progress,
    )
    if (cfg.get("REQUIRE_CLEAN_TOPOLOGY", False)
            and topology_report.get("remaining_defects")):
        write_json(ensure_run_dir(cfg) / "voxel_topology.json",
                   {"report": topology_report, "changes": topology_changes})
        raise RuntimeError(
            "Unresolved voxel boundary defects "
            f"({topology_report.get('remaining_defects')}); "
            "inspect voxel_topology.json or disable REQUIRE_CLEAN_TOPOLOGY."
        )
    set_labels(state, labels_out)
    state.store_artefact("labels_meshing", labels_out.copy())
    report = dict(topology_report)
    report["enabled"] = enabled
    write_json(ensure_run_dir(cfg) / "voxel_topology.json",
               {"report": report, "changes": topology_changes})
    return report


def handle_freeze_thin(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.thin_freeze import (
        FrozenGeometry, detect_thin_grains,
    )

    cfg = state.config
    enabled = bool(cfg.get("STAGE_FREEZE_THIN", False)
                   and cfg.get("FREEZE_THIN_GRAINS", False))
    labels = state.get_artefact("labels_meshing")
    if labels is None:
        labels = current_labels(state)
    spacing = spacing_tuple(cfg)
    frozen_ids, freezing_report = detect_thin_grains(
        labels,
        enabled=enabled,
        automatic=bool(cfg.get("AUTO_FREEZE_THIN_GRAINS", True)),
        thickness=cfg_int(cfg, "THIN_THICKNESS_VOXELS", 1),
        explicit_ids=tuple(cfg.get("EXPLICIT_FROZEN_GRAIN_IDS") or ()),
        criterion=str(cfg.get("THIN_CRITERION", "inscribed")),
    )
    frozen_geometry = FrozenGeometry(labels, frozen_ids, spacing)
    state.store_artefact("frozen_grain_ids", list(frozen_ids))
    state.store_artefact("frozen_geometry", frozen_geometry)
    report = dict(freezing_report)
    report["n_frozen"] = len(frozen_ids)
    write_json(ensure_run_dir(cfg) / "frozen_grains.json", report)
    return report


def handle_upscale(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.interfaces import upscale_labels

    cfg = state.config
    factor = cfg_int(cfg, "UPSCALE_FACTOR", 1)
    if cfg.get("STAGE_UPSCALE", True) is False:
        factor = 1
    if factor not in (1, 2, 3):
        raise ValueError("UPSCALE_FACTOR must be 1, 2, or 3")
    labels = state.get_artefact("labels_meshing")
    if labels is None:
        labels = current_labels(state)
    labels = np.asarray(labels)
    spacing = spacing_tuple(cfg)
    labels_up, effective = upscale_labels(labels, factor=factor, spacing=spacing)
    effective_t = tuple(float(x) for x in np.asarray(effective, dtype=float).ravel()[:3])
    cfg["EFFECTIVE_SPACING"] = effective_t
    set_labels(state, labels_up)
    state.store_artefact("labels_upscaled", labels_up.copy())
    # Ensure FrozenGeometry exists even when the freeze stage was skipped.
    if state.get_artefact("frozen_geometry") is None:
        from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry
        ids = list(state.get_artefact("frozen_grain_ids") or [])
        state.store_artefact("frozen_geometry", FrozenGeometry(labels, ids, spacing))
        state.store_artefact("frozen_grain_ids", ids)
    report = {
        "factor": factor,
        "shape_before": list(map(int, labels.shape)),
        "shape_after": list(map(int, labels_up.shape)),
        "spacing_before": list(spacing),
        "spacing_after": list(effective_t),
    }
    write_json(ensure_run_dir(cfg) / "upscale.json", report)
    return report
