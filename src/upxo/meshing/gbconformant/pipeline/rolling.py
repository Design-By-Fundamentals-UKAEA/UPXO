"""The rolling artefacts the stages hand on (labels, frozen geometry, upscaled labels).

A stage that is switched off still has to leave these coherent for the stages after
it; the skip side effects here do that.
"""
from __future__ import annotations

import numpy as np

from upxo.meshing.gbconformant.pipeline.runfiles import spacing_tuple
from upxo.meshing.gbconformant.pipeline.stages import STAGES
from upxo.meshing.gbconformant.pipeline.state import PipelineState, StageStatus


def current_labels(state: PipelineState) -> np.ndarray:
    labels = state.get_artefact("labels")
    if labels is None:
        raise RuntimeError("No labels artefact — run Load LGI first.")
    return np.asarray(labels)


def set_labels(state: PipelineState, labels: np.ndarray, *, also_before: bool = False) -> None:
    labels = np.asarray(labels)
    if also_before:
        state.store_artefact("labels_before", labels.copy())
    state.store_artefact("labels", labels)


def ensure_frozen_geometry(state: PipelineState):
    """Return FrozenGeometry, creating an empty (no frozen grains) one if needed.

    Optional freeze/upscale stages may be switched off; later stages still need this
    artefact the way the notebook always constructs one.
    """
    existing = state.get_artefact("frozen_geometry")
    if existing is not None:
        return existing
    from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry

    labels = state.get_artefact("labels_meshing")
    if labels is None:
        labels = state.get_artefact("labels_upscaled")
    if labels is None:
        labels = current_labels(state)
    labels = np.asarray(labels)
    ids = list(state.get_artefact("frozen_grain_ids") or [])
    spacing = state.config.get("EFFECTIVE_SPACING") or spacing_tuple(state.config)
    fg = FrozenGeometry(labels, ids, spacing)
    state.store_artefact("frozen_geometry", fg)
    state.store_artefact("frozen_grain_ids", ids)
    return fg


def ensure_labels_upscaled(state: PipelineState) -> np.ndarray:
    """Ensure ``labels_upscaled`` / EFFECTIVE_SPACING exist (identity if skipped)."""
    labels = state.get_artefact("labels_upscaled")
    if labels is not None:
        if state.config.get("EFFECTIVE_SPACING") is None:
            state.config["EFFECTIVE_SPACING"] = spacing_tuple(state.config)
        return np.asarray(labels)
    labels = state.get_artefact("labels_meshing")
    if labels is None:
        labels = current_labels(state)
        state.store_artefact("labels_meshing", np.asarray(labels).copy())
    labels = np.asarray(labels)
    spacing = spacing_tuple(state.config)
    state.config["EFFECTIVE_SPACING"] = spacing
    state.store_artefact("labels_upscaled", labels.copy())
    return labels


def ensure_skipped_optionals(state: PipelineState, before_stage_id: str) -> None:
    """Apply skip side effects for disabled optional stages before ``before_stage_id``.

    A caller that runs one stage at a time never visits the optional stages that are
    switched off, so the artefacts those stages would have written (e.g.
    ``frozen_geometry``) are synthesised here.
    """
    for stage in STAGES:
        if stage.id == before_stage_id:
            break
        flag = stage.optional_flag
        if not flag or state.config.get(flag, True):
            continue
        if stage.id not in state.status:
            continue
        status = state.status.get(stage.id, StageStatus.NOT_RUN)
        if status in (StageStatus.OK, StageStatus.SKIPPED, StageStatus.RUNNING):
            continue
        mark_skipped(state, stage.id)


def mark_skipped(state: PipelineState, stage_id: str) -> None:
    """Record ``stage_id`` as skipped (its switch is off) and keep the rolling artefacts coherent."""
    from upxo.meshing.gbconformant.pipeline.saved_results import config_changes, save_after_skip

    flag = next((s.optional_flag for s in STAGES if s.id == stage_id), None)
    before = dict(state.config)
    apply_skip_side_effects(state, stage_id)
    state.set_status(stage_id, StageStatus.SKIPPED)
    state.reports[stage_id] = {"skipped": True, "flag": flag}
    state.append_log(f"skipped (flag off): {stage_id}")
    save_after_skip(state, stage_id, config_changes(before, state.config))


def apply_skip_side_effects(state: PipelineState, stage_id: str) -> None:
    """Keep rolling artefacts coherent when an optional stage is skipped."""
    previous = state.current_stage
    state.current_stage = stage_id
    try:
        _skip_side_effects(state, stage_id)
    finally:
        state.current_stage = previous


def _skip_side_effects(state: PipelineState, stage_id: str) -> None:
    if stage_id == "freeze_thin":
        from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry
        labels = state.get_artefact("labels_meshing")
        if labels is None:
            labels = current_labels(state)
            state.store_artefact("labels_meshing", np.asarray(labels).copy())
        state.store_artefact("frozen_grain_ids", [])
        state.store_artefact(
            "frozen_geometry",
            FrozenGeometry(labels, [], spacing_tuple(state.config)),
        )
    elif stage_id == "upscale":
        labels = state.get_artefact("labels_meshing")
        if labels is None:
            labels = current_labels(state)
            state.store_artefact("labels_meshing", np.asarray(labels).copy())
        labels = np.asarray(labels)
        spacing = spacing_tuple(state.config)
        state.config["EFFECTIVE_SPACING"] = spacing
        state.config["UPSCALE_FACTOR"] = 1
        set_labels(state, labels)
        state.store_artefact("labels_upscaled", labels.copy())
        if state.get_artefact("frozen_geometry") is None:
            from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry
            state.store_artefact("frozen_grain_ids",
                                 list(state.get_artefact("frozen_grain_ids") or []))
            state.store_artefact(
                "frozen_geometry",
                FrozenGeometry(labels, state.get_artefact("frozen_grain_ids") or [],
                               spacing),
            )
    elif stage_id in (
        "small_grains", "spikes", "pinches", "voxel_topology",
    ):
        # Later stages expect labels_meshing after the cleaning chain.
        if stage_id == "voxel_topology" or state.get_artefact("labels_meshing") is None:
            state.store_artefact("labels_meshing", current_labels(state).copy())
    elif stage_id in ("guard_relax", "unpinch"):
        # Keep current interfaces as the rolling surface.
        iface = state.get_artefact("interfaces")
        if iface is not None and stage_id == "unpinch":
            state.store_artefact("interfaces_before_unpinch", iface)
    elif stage_id in ("contact_repair", "crossing_repair", "surface_repair"):
        remeshed = state.get_artefact("remeshed_rve")
        if remeshed is not None and state.get_artefact("repaired_rve") is None:
            state.store_artefact("repaired_rve", remeshed)
