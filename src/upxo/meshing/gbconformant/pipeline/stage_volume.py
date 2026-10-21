"""Volume stages: tetrahedral meshing, dihedral smoothing, swaps, the gate and node insertion."""
from __future__ import annotations

from typing import Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.progress import adapt_dict_progress, phase_progress
from upxo.meshing.gbconformant.pipeline.quality_snapshots import (
    compare_snapshots, tet_quality_snapshot,
)
from upxo.meshing.gbconformant.pipeline.runfiles import (
    cfg_float, cfg_int, ensure_run_dir, spacing_tuple, write_json,
)
from upxo.meshing.gbconformant.pipeline.stage_surface import (
    backend_kwargs, frozen, save_surface_vtp,
)
from upxo.meshing.gbconformant.pipeline.state import PipelineState

PRE_TET_POINTS = "pre_tet_surface_points"


def handle_tet_mesh(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.gmsh_tets import (
        mesh_repaired_rve_gmsh, save_grain_tetrahedra,
    )

    cfg = state.config
    prog = state.report_progress
    surface = state.get_artefact("repaired_rve")
    if surface is None:
        raise RuntimeError("repaired_rve missing")
    state.store_artefact(PRE_TET_POINTS, np.array(surface.points, dtype=float, copy=True))
    smin = min(spacing_tuple(cfg))
    mesh_size = cfg_float(cfg, "TET_MESH_SIZE",
                          cfg_float(cfg, "TET_MESH_SIZE_FACTOR", 1.5) * smin)
    opt_thresh = cfg_float(cfg, "TET_OPTIMIZE_THRESHOLD", 0.5)
    phase_progress(prog, 1, 3, "Tet mesh: Gmsh volume meshing…")
    tets = mesh_repaired_rve_gmsh(
        surface,
        mesh_size=mesh_size,
        optimize_netgen=bool(cfg.get("TET_OPTIMIZE_NETGEN", True)),
        minimum_quality=cfg_float(cfg, "MIN_TET_QUALITY", 0.05),
        netgen_all_volumes=bool(cfg.get("TET_NETGEN_ALL_VOLUMES", True)),
        optimize_threshold=opt_thresh,
        minimum_dihedral=cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0),
        **backend_kwargs(cfg),
    )
    state.store_artefact("verified_tets", tets)
    out = None
    if cfg.get("WRITE_STAGE_TET_FILES", False):
        phase_progress(prog, 2, 3, "Tet mesh: saving VTU…")
        out = ensure_run_dir(cfg) / "verified_tets.vtu"
        save_grain_tetrahedra(tets, out)
    else:
        phase_progress(prog, 2, 3, "Tet mesh: kept in the saved stage results")
    phase_progress(prog, 3, 3, "Tet mesh: quality summary…")
    snap = tet_quality_snapshot(state)
    state.store_artefact("tet_quality_after", snap)
    state.store_artefact("tet_quality_after_tet_mesh", snap)
    report = dict(tets.report)
    return {
        "status": report.get("status"),
        "ready": report.get("ready"),
        "minimum_quality": report.get("minimum_quality"),
        "dihedral_angles": report.get("dihedral_angles"),
        "path": None if out is None else str(out),
        "quality_after": snap,
    }


def _tets(state: PipelineState):
    tets = state.get_artefact("verified_tets")
    if tets is None:
        raise RuntimeError("verified_tets missing — run tetrahedral meshing first.")
    return tets


def _surface(state: PipelineState):
    surface = state.get_artefact("repaired_rve")
    if surface is None:
        raise RuntimeError("repaired_rve missing — surface required for tet improvement.")
    return surface


def _save_tets(state: PipelineState, tets) -> Optional[str]:
    """Write the stage's mesh and report to the run folder when WRITE_STAGE_TET_FILES is on; else nothing.

    The final mesh is written by the export stages; the stage results saved for resuming hold every
    intermediate mesh, so this is only a convenience copy.
    """
    if not state.config.get("WRITE_STAGE_TET_FILES", False):
        return None
    from upxo.meshing.gbconformant.d3v2p1.gmsh_tets import save_grain_tetrahedra

    out = ensure_run_dir(state.config) / "verified_tets.vtu"
    save_grain_tetrahedra(tets, out)
    write_json(out.with_suffix(".json"), tets.report)
    return str(out)


def _tet_summary(tets) -> dict:
    report = dict(tets.report)
    return {
        "status": report.get("status"),
        "ready": report.get("ready"),
        "minimum_quality": report.get("minimum_quality"),
        "dihedral_angles": report.get("dihedral_angles"),
        "n_tets": int(len(tets.tetrahedra)),
        "n_points": int(len(tets.points)),
    }


def _capture_quality_before(state: PipelineState, stage_id: str) -> dict:
    snap = tet_quality_snapshot(state)
    state.store_artefact("tet_quality_before", snap)
    state.store_artefact(f"tet_quality_before_{stage_id}", snap)
    return snap


def _attach_quality_compare(
    state: PipelineState, summary: dict, *, before: dict, stage_id: str,
) -> dict:
    after = tet_quality_snapshot(state)
    state.store_artefact("tet_quality_after", after)
    state.store_artefact(f"tet_quality_after_{stage_id}", after)
    summary["quality_before"] = before
    summary["quality_after"] = after
    summary["quality_delta"] = compare_snapshots(before, after)
    return summary


def smooth_kwargs(cfg: dict, *, gate: bool = False) -> dict:
    if gate:
        target = cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0)
        neighbour = cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0)
        tri_target = cfg_float(cfg, "GATE_TRIANGLE_TARGET", 20.0)
        tri_floor = cfg_float(cfg, "GATE_TRIANGLE_FLOOR", 15.0)
    else:
        target = cfg_float(cfg, "TET_TARGET_MIN_DIHEDRAL", 30.0)
        neighbour = cfg_float(cfg, "TET_NEIGHBOUR_FLOOR", 20.0)
        tri_target = cfg_float(cfg, "SURFACE_TRIANGLE_TARGET", 30.0)
        tri_floor = cfg_float(cfg, "SURFACE_TRIANGLE_FLOOR", 20.0)
    kwargs = {
        "target": target,
        "max_angle": cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0),
        "neighbour_floor": neighbour,
        "max_passes": cfg_int(cfg, "TET_SMOOTHING_PASSES", 10),
        "frozen_grain_ids": (),  # filled by caller
        "max_deviation": cfg.get("SURFACE_MOVE_MAX_DEVIATION"),
        "minimum_facet_angle": cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        "triangle_target": tri_target,
        "triangle_floor": tri_floor,
        "wedge_limit": cfg_float(cfg, "SMOOTH_MIN_WEDGE_DEGREES", 30.0),
        **backend_kwargs(cfg),
    }
    if gate:
        jdev = cfg.get("JUNCTION_POINT_MAX_DEVIATION")
        if jdev is None and cfg.get("MOVE_SURFACE_NODES", True):
            jdev = 0.15 * min(spacing_tuple(cfg))
        kwargs["corner_max_deviation"] = (
            jdev if cfg.get("MOVE_SURFACE_NODES", True) else None)
    return kwargs


def _run_smooth(
    state: PipelineState,
    *,
    enabled: bool,
    gate: bool = False,
    stage_id: str = "tet_smooth",
    capture_quality: bool = True,
) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.tet_smoothing import (
        smooth_grain_tetrahedra, surface_after_smoothing,
    )

    cfg = state.config
    before = _capture_quality_before(state, stage_id) if capture_quality else None
    tets = _tets(state)
    surface = _surface(state)
    move_surface = bool(cfg.get("MOVE_SURFACE_NODES", True))
    kwargs = smooth_kwargs(cfg, gate=gate)
    kwargs["enabled"] = enabled
    kwargs["frozen_grain_ids"] = tuple(state.get_artefact("frozen_grain_ids") or ())
    kwargs["surface"] = surface if move_surface else None
    max_passes = int(kwargs.get("max_passes") or cfg_int(cfg, "TET_SMOOTHING_PASSES", 10))
    label = "Gate smooth" if gate else "Tet smooth"
    kwargs["progress"] = adapt_dict_progress(
        state, label=label, total=max_passes)
    tets = smooth_grain_tetrahedra(tets, len(surface.points), **kwargs)
    if move_surface and enabled:
        surface = surface_after_smoothing(surface, tets)
        frozen(state).check(surface)
        state.store_artefact("repaired_rve", surface)
        save_surface_vtp(ensure_run_dir(cfg) / "repaired_RVE.vtp", surface)
    state.store_artefact("verified_tets", tets)
    path = _save_tets(state, tets)
    summary = _tet_summary(tets)
    summary["path"] = path
    summary["enabled"] = enabled
    summary["gate"] = gate
    summary["dihedral_smoothing"] = tets.report.get("dihedral_smoothing")
    if capture_quality and before is not None:
        _attach_quality_compare(
            state, summary, before=before, stage_id=stage_id)
    return summary


def handle_tet_smooth(state: PipelineState) -> dict:
    cfg = state.config
    enabled = bool(cfg.get("STAGE_TET_SMOOTH", True)
                   and cfg.get("SMOOTH_TET_DIHEDRALS", True))
    return _run_smooth(
        state, enabled=enabled, gate=False, stage_id="tet_smooth")


def handle_tet_swaps(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.tet_swaps import swap_grain_tetrahedra

    cfg = state.config
    enabled = bool(cfg.get("STAGE_TET_SWAPS", True) and cfg.get("SWAP_TETS", True))
    before = _capture_quality_before(state, "tet_swaps")
    max_passes = cfg_int(cfg, "TET_SWAP_PASSES", 5)
    tets = swap_grain_tetrahedra(
        _tets(state),
        _surface(state),
        enabled=enabled,
        target=cfg_float(cfg, "TET_TARGET_MIN_DIHEDRAL", 30.0),
        max_angle=cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0),
        max_passes=max_passes,
        progress=adapt_dict_progress(
            state, label="Tet swaps", total=max_passes),
        **backend_kwargs(cfg),
    )
    state.store_artefact("verified_tets", tets)
    path = _save_tets(state, tets)
    summary = _tet_summary(tets)
    summary["path"] = path
    summary["enabled"] = enabled
    summary["tet_swaps"] = tets.report.get("tet_swaps")
    return _attach_quality_compare(
        state, summary, before=before, stage_id="tet_swaps")


def handle_tet_smooth_after(state: PipelineState) -> dict:
    cfg = state.config
    enabled = bool(
        cfg.get("STAGE_SMOOTH_AFTER_SWAPS", True)
        and cfg.get("SMOOTH_AFTER_SWAPS", True)
        and cfg.get("SMOOTH_TET_DIHEDRALS", True)
    )
    return _run_smooth(
        state, enabled=enabled, gate=False, stage_id="tet_smooth_after")


def handle_dihedral_gate(state: PipelineState) -> dict:
    """Gate smooth + gate swaps (notebook cells combined)."""
    from upxo.meshing.gbconformant.d3v2p1.tet_swaps import swap_grain_tetrahedra

    cfg = state.config
    enabled = bool(cfg.get("STAGE_DIHEDRAL_GATE", True) and cfg.get("GATE_PASS", True))
    before = _capture_quality_before(state, "dihedral_gate")
    # The inner smooth must not overwrite the gate's before snapshot or deltas.
    smooth_report = _run_smooth(
        state, enabled=enabled, gate=True, stage_id="dihedral_gate",
        capture_quality=False)
    max_passes = cfg_int(cfg, "TET_SWAP_PASSES", 5)
    tets = swap_grain_tetrahedra(
        _tets(state),
        _surface(state),
        enabled=enabled,
        target=cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0),
        max_angle=cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0),
        max_passes=max_passes,
        progress=adapt_dict_progress(
            state, label="Gate swaps", total=max_passes),
        **backend_kwargs(cfg),
    )
    state.store_artefact("verified_tets", tets)
    path = _save_tets(state, tets)
    summary = _tet_summary(tets)
    summary["path"] = path
    summary["enabled"] = enabled
    summary["gate_smooth"] = smooth_report
    summary["tet_swaps"] = tets.report.get("tet_swaps")
    return _attach_quality_compare(
        state, summary, before=before, stage_id="dihedral_gate")


def handle_insert_nodes(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.tet_swaps import insert_grain_tetrahedra

    cfg = state.config
    enabled = bool(cfg.get("STAGE_INSERT_NODES", True) and cfg.get("INSERT_NODES", True))
    before = _capture_quality_before(state, "insert_nodes")
    max_passes = 3
    tets = insert_grain_tetrahedra(
        _tets(state),
        _surface(state),
        enabled=enabled,
        target=cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0),
        max_angle=cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0),
        progress=adapt_dict_progress(
            state, label="Node insertion", total=max_passes),
    )
    state.store_artefact("verified_tets", tets)
    path = _save_tets(state, tets)
    insertion = {
        k: v for k, v in dict(tets.report.get("node_insertion") or {}).items()
        if k != "history"
    }
    summary = _tet_summary(tets)
    summary["path"] = path
    summary["enabled"] = enabled
    summary["node_insertion"] = insertion
    return _attach_quality_compare(
        state, summary, before=before, stage_id="insert_nodes")
