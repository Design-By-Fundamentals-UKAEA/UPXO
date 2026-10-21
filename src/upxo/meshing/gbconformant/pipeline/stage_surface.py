"""Surface stages: interface smoothing, geometry guard, remesh, RVE faces, validation and repair."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from upxo.meshing.gbconformant.pipeline.progress import phase_progress
from upxo.meshing.gbconformant.pipeline.rolling import (
    current_labels, ensure_frozen_geometry, ensure_labels_upscaled,
)
from upxo.meshing.gbconformant.pipeline.runfiles import (
    cfg_float, cfg_int, ensure_run_dir, spacing_tuple, write_json,
)
from upxo.meshing.gbconformant.pipeline.state import PipelineState


def backend_kwargs(cfg: dict) -> dict:
    return {
        "backend": str(cfg.get("BACKEND", "auto")),
        "n_workers": cfg.get("N_WORKERS"),
    }


def effective_spacing(cfg: dict) -> tuple[float, float, float]:
    spacing = cfg.get("EFFECTIVE_SPACING") or spacing_tuple(cfg)
    arr = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    return float(arr[0]), float(arr[1]), float(arr[2])


def rve_dimensions(state: PipelineState) -> np.ndarray:
    labels = ensure_labels_upscaled(state)
    spacing = np.asarray(effective_spacing(state.config), dtype=float)
    return np.asarray(labels.shape, dtype=float) * spacing


def frozen(state: PipelineState):
    return ensure_frozen_geometry(state)


def current_interfaces(state: PipelineState):
    iface = state.get_artefact("interfaces")
    if iface is None:
        raise RuntimeError("interfaces missing — run smooth_interfaces first.")
    return iface


def save_surface_vtp(path: Path, surface, *, exterior=None, rve_face=None) -> None:
    # Visualisation libraries may bind to the thread that first imported them, so
    # callers write surfaces from the thread that owns the session.
    from upxo.meshing.gbconformant.d3v2p0.surface_export import save_surface

    save_surface(path, surface, exterior=exterior, rve_face=rve_face)


def handle_smooth_interfaces(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.interfaces import smooth_interfaces

    cfg = state.config
    labels = ensure_labels_upscaled(state)
    spacing = cfg.get("EFFECTIVE_SPACING") or spacing_tuple(cfg)
    ensure_frozen_geometry(state)
    frozen_ids = tuple(state.get_artefact("frozen_grain_ids") or ())
    jmax = cfg.get("JUNCTION_MAX_DISPLACEMENT")
    if jmax is None:
        jmax = 0.75 * min(spacing_tuple(cfg))
    interfaces = smooth_interfaces(
        labels,
        spacing=spacing,
        iterations=cfg_int(cfg, "LAPLACE_ITERATIONS", 50),
        relaxation=cfg_float(cfg, "LAPLACE_RELAXATION", 0.75),
        junction_iterations=cfg_int(cfg, "JUNCTION_ITERATIONS", 10),
        junction_relaxation=cfg_float(cfg, "JUNCTION_RELAXATION", 0.35),
        junction_max_displacement=jmax,
        frozen_grain_ids=frozen_ids,
        iterations_by_pair=cfg.get("SMOOTH_ITERATIONS_BY_PAIR"),
        ramp_rings=cfg_int(cfg, "SMOOTH_RAMP_RINGS", 2),
        thickness_cap_fraction=cfg_float(cfg, "THICKNESS_CAP_FRACTION", 0.5),
        minimum_wedge_angle=cfg_float(cfg, "SMOOTH_MIN_WEDGE_DEGREES", 30.0),
        minimum_corner_angle=cfg_float(cfg, "SMOOTH_MIN_CORNER_DEGREES", 20.0),
        progress=state.report_progress,
    )
    frozen_geometry = state.get_artefact("frozen_geometry")
    if frozen_geometry is not None:
        frozen_geometry.check(interfaces)
    state.store_artefact("interfaces", interfaces)
    state.store_artefact("interfaces_smoothed", interfaces)
    report = {
        "n_points": int(len(interfaces.points)),
        "n_triangles": int(len(interfaces.triangles)),
        "accepted_iterations": int(interfaces.accepted_iterations),
        "frozen_junction_points": int(np.count_nonzero(interfaces.node_kind == 3)),
        "junction_metrics": interfaces.junction_metrics(),
        "smoothing_report": interfaces.smoothing_report,
    }
    write_json(ensure_run_dir(cfg) / "smooth_interfaces.json", report)
    return report


def handle_geometry_guard(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.geometry_guard import stabilize_interfaces

    cfg = state.config
    smoothed = state.get_artefact("interfaces_smoothed") or current_interfaces(state)
    spacing = effective_spacing(cfg)
    clearance = cfg.get("GEOMETRY_BOUNDARY_CLEARANCE")
    if clearance is None:
        clearance = 0.05 * min(spacing)
    result = frozen(state).run(
        stabilize_interfaces,
        smoothed,
        rve_dimensions(state),
        enabled=bool(cfg.get("GUARD_SMOOTHED_GEOMETRY", True)),
        boundary_clearance=float(clearance),
        max_passes=cfg_int(cfg, "GEOMETRY_MAX_PASSES", 100),
        relaxation=cfg_float(cfg, "GEOMETRY_ROLLBACK_RELAXATION", 0.5),
        max_correction=cfg.get("GEOMETRY_MAX_CORRECTION"),
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        progress=state.report_progress,
        **backend_kwargs(cfg),
    )
    interfaces, report = result
    state.store_artefact("interfaces", interfaces)
    state.store_artefact("interfaces_before_unpinch", interfaces)
    write_json(ensure_run_dir(cfg) / "geometry_guard.json", report)
    return dict(report)


def handle_guard_relax(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.guard_relaxation import (
        relax_guarded_interfaces,
    )

    cfg = state.config
    enabled = bool(cfg.get("STAGE_GUARD_RELAX", True)
                   and cfg.get("RELAX_GUARD_ROLLBACK", True))
    spacing = effective_spacing(cfg)
    clearance = cfg.get("GEOMETRY_BOUNDARY_CLEARANCE")
    if clearance is None:
        clearance = 0.05 * min(spacing)
    jmax = cfg.get("JUNCTION_MAX_DISPLACEMENT")
    if jmax is None:
        jmax = 0.75 * min(spacing_tuple(cfg))
    wedge = cfg_float(cfg, "SMOOTH_MIN_WEDGE_DEGREES", 0.0)
    min_facet = max(cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0), wedge)
    result = frozen(state).run(
        relax_guarded_interfaces,
        current_interfaces(state),
        state.get_artefact("interfaces_smoothed") or current_interfaces(state),
        rve_dimensions(state),
        enabled=enabled,
        iterations=cfg_int(cfg, "GUARD_RELAX_ITERATIONS", 3),
        margin_rings=cfg_int(cfg, "GUARD_RELAX_MARGIN_RINGS", 2),
        minimum_facet_angle=min_facet,
        boundary_clearance=float(clearance),
        junction_max_displacement=jmax,
        exclude_grain_ids=tuple(state.get_artefact("frozen_grain_ids") or ()),
        progress=state.report_progress,
        **backend_kwargs(cfg),
    )
    interfaces, report = result
    state.store_artefact("interfaces", interfaces)
    state.store_artefact("interfaces_before_unpinch", interfaces)
    write_json(ensure_run_dir(cfg) / "guard_relaxation.json", report)
    return {k: v for k, v in dict(report).items() if k != "history"}


def handle_unpinch(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.unpinch import separate_pinch_fans

    cfg = state.config
    enabled = bool(cfg.get("STAGE_UNPINCH", True) and cfg.get("SEPARATE_PINCHES", True))
    base = spacing_tuple(cfg)
    smin = min(base)
    before = state.get_artefact("interfaces_before_unpinch") or current_interfaces(state)
    result = frozen(state).run(
        separate_pinch_fans,
        before,
        enabled=enabled,
        radius=cfg_float(cfg, "PINCH_PRESS_RADIUS", 1.5 * smin),
        max_displacement=cfg_float(cfg, "PINCH_PRESS_MAX_DISPLACEMENT", 0.5 * smin),
        iterations=cfg_int(cfg, "PINCH_PRESS_ITERATIONS", 30),
        relaxation=cfg_float(cfg, "PINCH_PRESS_RELAXATION", 0.25),
        min_separation=cfg_float(cfg, "PINCH_MIN_OPENING", 0.02 * smin),
        progress=state.report_progress,
    )
    interfaces, report, audit = result
    state.store_artefact("interfaces", interfaces)
    write_json(ensure_run_dir(cfg) / "unpinch.json",
               {"report": report, "contacts": audit})
    return dict(report)


def handle_gmsh_interfaces(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.gmsh_interfaces import remesh_interfaces_gmsh

    cfg = state.config
    spacing = spacing_tuple(cfg)
    default_size = cfg_float(cfg, "GMSH_SIZE_FACTOR", 0.75) * min(spacing)
    mesh_size = cfg_float(cfg, "GMSH_SIZE", default_size)
    chart_tris = cfg.get("GMSH_CHART_TRIANGLES", 128)
    if chart_tris is None:
        chart_tris = 128
    remesh_opening = cfg.get("GMSH_MIN_REMESH_OPENING", 30.0)
    if remesh_opening is None:
        remesh_opening = 30.0
    prog = state.report_progress
    phase_progress(prog, 1, 4, "Gmsh remesh: preparing…")
    phase_progress(prog, 2, 4, "Gmsh remesh: meshing interfaces (may take a while)…")
    gmsh_result = frozen(state).run(
        remesh_interfaces_gmsh,
        current_interfaces(state),
        mesh_size=mesh_size,
        verbose=bool(cfg.get("GMSH_VERBOSE", False)),
        algorithm=cfg_int(cfg, "GMSH_ALGORITHM", 6),
        check_intersections=bool(cfg.get("GMSH_CHECK_INTERSECTIONS", True)),
        intersection_retries=cfg_int(cfg, "GMSH_INTERSECTION_RETRIES", 4),
        max_chart_triangles=int(chart_tris),
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        minimum_remesh_opening=float(remesh_opening),
        rve_dimensions=rve_dimensions(state),
        retry_failed_only=bool(cfg.get("GMSH_RETRY_FAILED_ONLY", False)),
        **backend_kwargs(cfg),
    )
    phase_progress(prog, 3, 4, "Gmsh remesh: writing VTP…")
    state.store_artefact("gmsh_result", gmsh_result)
    state.store_artefact("interfaces", gmsh_result)
    out = ensure_run_dir(cfg) / "gmsh.vtp"
    save_surface_vtp(out, gmsh_result)
    write_json(out.with_suffix(".json"), gmsh_result.report)
    phase_progress(prog, 4, 4, "Gmsh remesh: done")
    return dict(gmsh_result.report)


def handle_close_rve(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces

    cfg = state.config
    gmsh_result = state.get_artefact("gmsh_result")
    if gmsh_result is None:
        raise RuntimeError("gmsh_result missing — run Gmsh remesh interfaces first.")
    labels = state.get_artefact("labels_upscaled")
    if labels is None:
        labels = current_labels(state)
    spacing = effective_spacing(cfg)
    base = spacing_tuple(cfg)
    default_size = cfg_float(cfg, "GMSH_SIZE_FACTOR", 0.75) * min(base)
    default_size = cfg_float(cfg, "GMSH_SIZE", default_size)
    mesh_size = cfg_float(cfg, "CAP_MESH_SIZE", default_size)
    prog = state.report_progress
    phase_progress(prog, 1, 3, "Close RVE: preparing caps…")
    phase_progress(prog, 2, 3, "Close RVE: meshing six faces…")
    closed_rve = frozen(state).run(
        close_rve_faces,
        gmsh_result,
        labels,
        spacing=spacing,
        mesh_size=mesh_size,
        edge_size=cfg_float(cfg, "RVE_EDGE_SIZE", mesh_size),
    )
    phase_progress(prog, 3, 3, "Close RVE: writing VTP…")
    state.store_artefact("closed_rve", closed_rve)
    out = ensure_run_dir(cfg) / "closed_RVE.vtp"
    save_surface_vtp(out, closed_rve)
    write_json(out.with_suffix(".json"), closed_rve.report)
    report = {k: v for k, v in dict(closed_rve.report).items()
              if k != "enclosed_grain_volumes"}
    return report


def handle_pre_joint_quality(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.surface_quality import improve_surface_quality

    cfg = state.config
    closed = state.get_artefact("closed_rve")
    if closed is None:
        raise RuntimeError("closed_rve missing")
    enabled = bool(cfg.get("STAGE_PRE_JOINT_QUALITY", True)
                   and cfg.get("PRE_JOINT_QUALITY_REPAIR", True))
    closed = frozen(state).run(
        improve_surface_quality,
        closed,
        enabled=enabled,
        minimum_quality=cfg_float(cfg, "PRE_JOINT_QUALITY_TARGET", 0.05),
        max_passes=cfg_int(cfg, "PRE_JOINT_QUALITY_MAX_PASSES", 10),
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        progress=state.report_progress,
        **backend_kwargs(cfg),
    )
    state.store_artefact("closed_rve", closed)
    repair = closed.report.get("surface_quality_repair", {})
    write_json(ensure_run_dir(cfg) / "pre_joint_quality.json", repair)
    return dict(repair)


def handle_joint_remesh(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.gmsh_closed import remesh_closed_rve_gmsh

    cfg = state.config
    closed = state.get_artefact("closed_rve")
    if closed is None:
        raise RuntimeError("closed_rve missing")
    base = spacing_tuple(cfg)
    default_size = cfg_float(cfg, "GMSH_SIZE_FACTOR", 0.75) * min(base)
    default_size = cfg_float(cfg, "GMSH_SIZE", default_size)
    mesh_size = cfg_float(cfg, "FINAL_GMSH_SIZE", default_size)
    chart_tris = cfg.get("GMSH_CHART_TRIANGLES", 128)
    if chart_tris is None:
        chart_tris = 128
    remesh_opening = cfg.get("GMSH_MIN_REMESH_OPENING", 30.0)
    if remesh_opening is None:
        remesh_opening = 30.0
    prog = state.report_progress
    phase_progress(prog, 1, 3, "Joint remesh: preparing…")
    phase_progress(prog, 2, 3, "Joint remesh: Gmsh closed RVE…")
    remeshed = frozen(state).run(
        remesh_closed_rve_gmsh,
        closed,
        mesh_size=mesh_size,
        algorithm=cfg_int(cfg, "FINAL_GMSH_ALGORITHM",
                          cfg_int(cfg, "GMSH_ALGORITHM", 6)),
        check_intersections=bool(cfg.get("GMSH_CHECK_INTERSECTIONS", True)),
        intersection_retries=cfg_int(cfg, "GMSH_INTERSECTION_RETRIES", 4),
        max_chart_triangles=int(chart_tris),
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        minimum_remesh_opening=float(remesh_opening),
        retry_failed_only=bool(cfg.get("GMSH_RETRY_FAILED_ONLY", False)),
        **backend_kwargs(cfg),
    )
    phase_progress(prog, 3, 3, "Joint remesh: writing VTP…")
    state.store_artefact("remeshed_rve", remeshed)
    state.store_artefact("repaired_rve", remeshed)
    out = ensure_run_dir(cfg) / "closed_RVE_remeshed.vtp"
    save_surface_vtp(out, remeshed)
    write_json(out.with_suffix(".json"), remeshed.report)
    return {k: v for k, v in dict(remeshed.report).items()
            if k != "enclosed_grain_volumes"}


def handle_pre_tet_validation(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.tet_validation import validate_tet_surfaces

    cfg = state.config
    surface = state.get_artefact("remeshed_rve")
    if surface is None:
        raise RuntimeError("remeshed_rve missing")
    labels = state.get_artefact("labels_upscaled")
    if labels is None:
        labels = current_labels(state)
    prog = state.report_progress
    phase_progress(prog, 1, 3, "Pre-tet validation: quality checks…")
    phase_progress(prog, 2, 3, "Pre-tet validation: intersections / facets…")
    report = validate_tet_surfaces(
        surface,
        expected_grains=np.unique(labels),
        min_triangle_quality=cfg_float(cfg, "MIN_SURFACE_TRIANGLE_QUALITY", 0.05),
        check_intersections=True,
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        **backend_kwargs(cfg),
    )
    phase_progress(prog, 3, 3, "Pre-tet validation: writing report…")
    state.store_artefact("tet_validation", report)
    write_json(ensure_run_dir(cfg) / "pre_tet_validation.json", report)
    if cfg.get("STOP_ON_VALIDATION_ERRORS") and report.get("blockers"):
        raise RuntimeError("Surface validation blocked: " + "; ".join(report["blockers"]))
    return {
        "status": report.get("status"),
        "n_blockers": len(report.get("blockers") or []),
        "n_warnings": len(report.get("warnings") or []),
        "triangle_quality": report.get("triangle_quality"),
    }


def handle_contact_repair(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.tet_repair import repair_tet_contacts

    cfg = state.config
    surface = state.get_artefact("repaired_rve") or state.get_artefact("remeshed_rve")
    if surface is None:
        raise RuntimeError("remeshed_rve missing")
    enabled = bool(cfg.get("STAGE_CONTACT_REPAIR", True) and cfg.get("REPAIR_CONTACTS", True))
    prog = state.report_progress
    phase_progress(prog, 1, 3, "Contact repair: preparing…")
    if enabled:
        sep = cfg.get("REPAIR_SEPARATION")
        if sep is None:
            sep = cfg_float(cfg, "REPAIR_SEPARATION_FACTOR", 0.15) * min(spacing_tuple(cfg))
        phase_progress(prog, 2, 3, "Contact repair: separating contacts…")
        surface = frozen(state).run(repair_tet_contacts, surface, separation=float(sep))
    else:
        phase_progress(prog, 2, 3, "Contact repair: skipped (disabled)")
    phase_progress(prog, 3, 3, "Contact repair: writing VTP…")
    state.store_artefact("repaired_rve", surface)
    out = ensure_run_dir(cfg) / "repaired_RVE.vtp"
    save_surface_vtp(out, surface)
    repair = surface.report.get("contact_repair", "disabled")
    return {"enabled": enabled, "contact_repair": repair}


def handle_crossing_repair(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.intersection_repair import (
        repair_interface_crossings,
    )

    cfg = state.config
    surface = state.get_artefact("repaired_rve")
    if surface is None:
        raise RuntimeError("repaired_rve missing")
    enabled = bool(cfg.get("STAGE_CROSSING_REPAIR", True)
                   and cfg.get("REPAIR_INTERFACE_CROSSINGS", True))
    smin = min(spacing_tuple(cfg))
    prog = state.report_progress
    phase_progress(prog, 1, 3, "Crossing repair: preparing…")
    if enabled:
        phase_progress(prog, 2, 3, "Crossing repair: resolving intersections…")
        surface = frozen(state).run(
            repair_interface_crossings,
            surface,
            max_displacement=cfg_float(cfg, "CROSSING_MAX_DISPLACEMENT", 0.35 * smin),
            radius=cfg_float(cfg, "CROSSING_NEIGHBOURHOOD_RADIUS", 1.0 * smin),
            opening=cfg_float(cfg, "CROSSING_OPENING", 0.10),
            **backend_kwargs(cfg),
        )
        out = ensure_run_dir(cfg) / "repaired_RVE.vtp"
        save_surface_vtp(out, surface)
    else:
        phase_progress(prog, 2, 3, "Crossing repair: skipped (disabled)")
    phase_progress(prog, 3, 3, "Crossing repair: done")
    state.store_artefact("repaired_rve", surface)
    return {
        "enabled": enabled,
        "intersection_repair": surface.report.get("intersection_repair", "disabled"),
    }


def handle_surface_repair(state: PipelineState) -> dict:
    """Quality flips + angle repair (notebook cells combined)."""
    from upxo.meshing.gbconformant.d3v2p1.surface_quality import improve_surface_quality
    from upxo.meshing.gbconformant.d3v2p1.surface_angles import improve_surface_angles

    cfg = state.config
    surface = state.get_artefact("repaired_rve")
    if surface is None:
        raise RuntimeError("repaired_rve missing")
    do_quality = bool(cfg.get("REPAIR_SURFACE_QUALITY", True))
    do_angles = bool(cfg.get("STAGE_SURFACE_ANGLE_REPAIR", True)
                     and cfg.get("REPAIR_SURFACE_ANGLES", True))
    quality_target = cfg_float(
        cfg, "SURFACE_QUALITY_TARGET",
        cfg_float(cfg, "MIN_SURFACE_TRIANGLE_QUALITY", 0.05))
    if do_quality:
        surface = frozen(state).run(
            improve_surface_quality,
            surface,
            enabled=True,
            minimum_quality=quality_target,
            max_passes=cfg_int(cfg, "SURFACE_QUALITY_MAX_PASSES", 10),
            minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
            progress=state.report_progress,
            **backend_kwargs(cfg),
        )
    if do_angles:
        surface = frozen(state).run(
            improve_surface_angles,
            surface,
            enabled=True,
            minimum_angle=cfg_float(cfg, "SURFACE_MIN_ANGLE_DEGREES", 30.0),
            max_passes=cfg_int(cfg, "SURFACE_ANGLE_MAX_PASSES", 12),
            max_deviation=cfg.get("SURFACE_ANGLE_MAX_DEVIATION"),
            minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
            frozen_grain_ids=tuple(state.get_artefact("frozen_grain_ids") or ()),
            wedge_limit=cfg.get("SMOOTH_MIN_WEDGE_DEGREES", 30.0),
            progress=state.report_progress,
            **backend_kwargs(cfg),
        )
    state.store_artefact("repaired_rve", surface)
    out = ensure_run_dir(cfg) / "repaired_RVE.vtp"
    save_surface_vtp(out, surface)
    return {
        "quality": surface.report.get("surface_quality_repair"),
        "angles": {
            k: v for k, v in dict(surface.report.get("surface_angle_repair") or {}).items()
            if k != "history"
        },
    }


def handle_surface_gate(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.tet_validation import validate_tet_surfaces

    cfg = state.config
    prog = state.report_progress
    surface = state.get_artefact("repaired_rve")
    if surface is None:
        raise RuntimeError("repaired_rve missing")
    labels = state.get_artefact("labels_upscaled")
    if labels is None:
        labels = current_labels(state)
    phase_progress(prog, 1, 3, "Surface gate: validating repaired RVE…")
    validation = validate_tet_surfaces(
        surface,
        expected_grains=np.unique(labels),
        min_triangle_quality=cfg_float(cfg, "MIN_SURFACE_TRIANGLE_QUALITY", 0.05),
        check_intersections=True,
        minimum_facet_angle=cfg_float(cfg, "MIN_FACET_OPENING_DEGREES", 15.0),
        **backend_kwargs(cfg),
    )
    state.store_artefact("repaired_validation", validation)
    phase_progress(prog, 2, 3, "Surface gate: writing validation report…")
    write_json(ensure_run_dir(cfg) / "repaired_RVE.json", {
        "repair": surface.report.get("contact_repair"),
        "surface_quality_repair": surface.report.get("surface_quality_repair"),
        "surface_angle_repair": surface.report.get("surface_angle_repair"),
        "intersection_repair": surface.report.get("intersection_repair"),
        "validation": validation,
    })
    if validation.get("blockers"):
        raise RuntimeError(
            "Repair has not resolved all surface defects: "
            + "; ".join(validation["blockers"]))
    # Surface-angle gate (notebook cell 48)
    phase_progress(prog, 3, 3, "Surface gate: angle / quality check…")
    min_angle = cfg_float(cfg, "SURFACE_MIN_ANGLE_DEGREES", 30.0)
    tq = validation.get("triangle_quality") or {}
    worst = tq.get("minimum_angle_degrees")
    poor = worst is not None and float(worst) < min_angle
    approved = bool(cfg.get("APPROVE_TET_MESHING_WITH_POOR_SURFACE", False))
    if poor and not approved:
        raise RuntimeError(
            f"Surface gate: minimum triangle angle {worst}° < {min_angle}°; "
            "set APPROVE_TET_MESHING_WITH_POOR_SURFACE to continue.")
    return {
        "status": validation.get("status"),
        "n_blockers": len(validation.get("blockers") or []),
        "poor_surface": poor,
        "approved_poor_surface": approved,
        "triangle_quality": tq,
        "minimum_angle_degrees": worst,
        "surface_min_angle_limit": min_angle,
        "quality_bounds": {
            "minimum_angle_degrees": worst,
            "mean_angle_degrees": tq.get("mean_angle_degrees"),
            "minimum_quality": tq.get("minimum_quality") or tq.get("min_quality"),
            "mean_quality": tq.get("mean_quality"),
        },
    }
