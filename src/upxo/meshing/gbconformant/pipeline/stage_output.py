"""Output stages: the before/after comparison and the surface, volume and Abaqus exports."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.rolling import current_labels
from upxo.meshing.gbconformant.pipeline.runfiles import (
    ensure_run_dir, export_folder, spacing_tuple, subset_run_active, write_json,
)
from upxo.meshing.gbconformant.pipeline.settings import clean_path_text
from upxo.meshing.gbconformant.pipeline.stage_input import SAME_AS_SOURCE, active_hierarchy
from upxo.meshing.gbconformant.pipeline.stage_surface import save_surface_vtp
from upxo.meshing.gbconformant.pipeline.state import PipelineState

SURVIVOR_KEY = "small_grains.survivor_map"
COMPARISON_KEY = "compare_before_after.result"


def _isotropic_spacing(cfg: dict) -> float:
    spacing = spacing_tuple(cfg)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("The FM-steel steps need equal voxel spacing along all axes.")
    return float(spacing[0])


def deck_settings(cfg: dict):
    """Settings -> :class:`DeckSettings`."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings

    units = str(cfg.get("FM_OUTPUT_UNITS") or SAME_AS_SOURCE)
    return DeckSettings(
        material_format=str(cfg.get("FM_MATERIAL_FORMAT", "reference_umat")),
        n_depvar=int(cfg.get("FM_N_DEPVAR", 1)),
        load_axis=str(cfg.get("FM_LOAD_AXIS", "z")),
        load_mode=str(cfg.get("FM_LOAD_MODE", "displacement")),
        load_case=str(cfg.get("FM_LOAD_CASE", "monotonic")),
        load_sense=str(cfg.get("FM_LOAD_SENSE", "tension")),
        target_strain=float(cfg.get("FM_TARGET_STRAIN", 0.02)),
        strain_rate=float(cfg.get("FM_STRAIN_RATE", 0.0004)),
        hold_time=float(cfg.get("FM_HOLD_TIME", 0.0)),
        cycle_definition=str(cfg.get("FM_CYCLE_DEFINITION", "min_max")),
        strain_min=float(cfg.get("FM_STRAIN_MIN", -0.01)),
        strain_max=float(cfg.get("FM_STRAIN_MAX", 0.01)),
        strain_amplitude=float(cfg.get("FM_STRAIN_AMPLITUDE", 0.01)),
        r_ratio=float(cfg.get("FM_R_RATIO", -1.0)),
        cycles=int(cfg.get("FM_CYCLES", 5)),
        output_elset=str(cfg.get("FM_OUTPUT_ELSET", "") or "").strip(),
        job_scripts=bool(cfg.get("JOB_SCRIPTS", True)),
        abaqus_command=str(cfg.get("ABAQUS_COMMAND", "abaqus") or "abaqus").strip(),
        umat_file=clean_path_text(cfg.get("UMAT_FILE")),
        job_name=str(cfg.get("JOB_NAME", "fm_job") or "fm_job").strip(),
        job_cpus=int(cfg.get("JOB_CPUS", 4)),
        job_memory_gb=float(cfg.get("JOB_MEMORY_GB", 0.0)),
        job_time_hours=float(cfg.get("JOB_TIME_HOURS", 24.0)),
        slurm_partition=str(cfg.get("SLURM_PARTITION", "") or "").strip(),
        applied_displacement=float(cfg.get("FM_APPLIED_DISPLACEMENT", 0.02)),
        step_time=float(cfg.get("FM_STEP_TIME", 50.0)),
        write_step=bool(cfg.get("FM_WRITE_STEP", True)),
        initial_inc=float(cfg.get("FM_INITIAL_INC", 0.01)),
        min_inc=float(cfg.get("FM_MIN_INC", 1e-8)),
        max_inc=float(cfg.get("FM_MAX_INC", 0.5)),
        max_increments=int(cfg.get("FM_MAX_INCREMENTS", 10000)),
        output_interval=float(cfg.get("FM_OUTPUT_INTERVAL", 2.5)),
        element_outputs=str(cfg.get("FM_ELEMENT_OUTPUTS", "LE, S, SDV")),
        node_outputs=str(cfg.get("FM_NODE_OUTPUTS", "U, RF")),
        length_scale=float(cfg.get("FM_LENGTH_SCALE", 1.0)),
        output_units=None if units == SAME_AS_SOURCE else units,
        header_message=str(cfg.get("FM_HEADER_MESSAGE", "")),
        boundary_sets=bool(cfg.get("EXPORT_BOUNDARY_SETS", True)),
        write_msh=bool(cfg.get("EXPORT_MSH", True)),
        write_vtu=bool(cfg.get("EXPORT_VTU", True)),
        layer_sets=bool(cfg.get("EXPORT_BOUNDARY_LAYER", True)),
        boundary_classes=bool(cfg.get("EXPORT_BOUNDARY_CLASSES", True)),
        low_angle_degrees=float(cfg.get("BOUNDARY_LOW_ANGLE_DEGREES", 15.0)),
        crystal_symmetry=str(cfg.get("TEXTURE_CRYSTAL_SYMMETRY", "cubic")),
        feature_table=bool(cfg.get("EXPORT_FEATURE_TABLE", True)),
        deck_check=bool(cfg.get("EXPORT_DECK_CHECK", True)),
        elset_prefixes={
            "pag": str(cfg.get("FM_PREFIX_PAG", "es_pag_")),
            "pck": str(cfg.get("FM_PREFIX_PCK", "es_pck_")),
            "blk": str(cfg.get("FM_PREFIX_BLK", "es_blk_")),
            "sblk": str(cfg.get("FM_PREFIX_SBLK", "es_subblk_")),
        },
        nset_names={
            "XMIN": str(cfg.get("FM_NSET_XMIN", "ns_xminus")),
            "XMAX": str(cfg.get("FM_NSET_XMAX", "ns_xplus")),
            "YMIN": str(cfg.get("FM_NSET_YMIN", "ns_yminus")),
            "YMAX": str(cfg.get("FM_NSET_YMAX", "ns_yplus")),
            "ZMIN": str(cfg.get("FM_NSET_ZMIN", "ns_zminus")),
            "ZMAX": str(cfg.get("FM_NSET_ZMAX", "ns_zplus")),
        },
    )


def build_identity(state: PipelineState, hierarchy):
    """Raw identity of every meshed grain from the stored shuffle and merge maps."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import resolve_identities

    cfg = state.config
    return resolve_identities(
        hierarchy.voxel_counts,
        shuffle_map=state.get_artefact("grain_id_map") or None,
        survivor_map=state.get_artefact(SURVIVOR_KEY),
        rule=str(cfg.get("IDENTITY_RULE", "larger")),
        seed=int(cfg.get("IDENTITY_SEED", 0)))


def export_fm_deck(state: PipelineState, tets, folder: Path) -> dict:
    """Write the FM-steel deck of ``tets`` into the new ``folder``."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import write_deck

    cfg = state.config
    hierarchy = active_hierarchy(state)
    if hierarchy is None:
        raise RuntimeError(
            "The FM-steel deck needs an FM-steel raw folder as the input (or a twinned FCC folder): "
            "set the dataset type to FM steel or twinned FCC and rerun, or use the mesh-only deck.")
    spacing = _isotropic_spacing(cfg)
    identity = build_identity(state, hierarchy)
    result = write_deck(
        folder, tets.points, tets.tetrahedra, tets.grain_ids, hierarchy, identity,
        deck_settings(cfg), mesh_voxel_length=spacing,
        meshed_labels=state.get_artefact("labels_meshing"))
    info = result["info"]
    realloc = result["reallocation"] or {}
    merges = _write_merge_log(cfg, hierarchy, identity, result["directory"])
    if cfg.get("EXPORT_PROVENANCE", True):
        from upxo.meshing.gbconformant.pipeline.provenance import write_provenance
        write_provenance(result["directory"], cfg)
    check = None
    if info.get("deck_check") is not None:
        check = json.loads((Path(result["directory"]) / "deck_check.json").read_text(encoding="utf8"))
        if cfg.get("REQUIRE_DECK_CHECK") and not check["ok"]:
            raise RuntimeError(f"Deck check failed: {', '.join(check['failed'])}. "
                               f"See deck_check.json in {result['directory']}.")
    return {
        "deck_check": None if check is None else {"ok": check["ok"], "failed": check["failed"],
                                                  "checks": check["checks"]},
        "layer_sets": info.get("layer_sets"),
        "boundary_classes": info.get("boundary_classes"),
        "load_case": info.get("load_case"),
        "job_scripts": info.get("job_scripts"),
        "output_estimate": info.get("output_estimate"),
        "boundary_sets": info.get("boundary_sets"),
        "feature_table": info.get("feature_table"),
        "folder": str(result["directory"]),
        "merged_features": merges,
        "level": hierarchy.level,
        "tetrahedra": info["tetrahedra"],
        "features_with_elements": info["features_with_elements"],
        "features_without_elements": info["features_without_elements"],
        "output_units": info["output_units"],
        "coordinate_scale": info["coordinate_scale"],
        "identity_rule": identity.rule,
        "voxels_moved": realloc.get("voxels_moved"),
        "volume_over_own_raw_label": (result["audit"] or {}).get("volume_over_own_raw_label"),
    }


def _write_merge_log(cfg: dict, hierarchy, identity, folder) -> Optional[int]:
    """Write the merge log beside the deck when features were merged; returns the number absorbed."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import merge_log, write_merge_log

    if not identity.report.get("merged_groups"):
        return 0
    try:
        log = merge_log(hierarchy, identity, symmetry=str(cfg.get("TEXTURE_CRYSTAL_SYMMETRY", "cubic")),
                        prefixes=deck_settings(cfg).elset_prefixes)
    except ValueError:                           # misorientation is only available for cubic crystals
        return None
    write_merge_log(folder, log)
    return log["summary"]["absorbed_features"]


def comparison_settings(cfg: dict):
    from upxo.meshing.gbconformant.fm_steel_abaqus import ComparisonSettings

    units = str(cfg.get("FM_OUTPUT_UNITS") or SAME_AS_SOURCE)
    return ComparisonSettings(
        include_rve_faces=bool(cfg.get("COMPARE_INCLUDE_RVE_FACES", False)),
        output_units=None if units == SAME_AS_SOURCE else units,
        length_scale=float(cfg.get("FM_LENGTH_SCALE", 1.0)),
        mesh_voxel_length=_isotropic_spacing(cfg),
        symmetry=str(cfg.get("TEXTURE_CRYSTAL_SYMMETRY", "cubic")))


def texture_settings(cfg: dict):
    """Settings -> :class:`TextureSettings` (pole figure options)."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import TextureSettings

    pole = str(cfg.get("TEXTURE_POLE", "100"))
    if pole == "custom":
        values = [float(v) for v in str(cfg.get("TEXTURE_HKL", "")).replace(",", " ").split()]
        if len(values) != 3:
            raise ValueError("Custom pole needs exactly three numbers (h k l).")
        pole = values
    grid = str(cfg.get("TEXTURE_GRID", "auto")).strip().lower() or "auto"
    return TextureSettings(
        pole_family=pole,
        crystal_symmetry=str(cfg.get("TEXTURE_CRYSTAL_SYMMETRY", "cubic")),
        apply_sample_symmetry=bool(cfg.get("TEXTURE_SAMPLE_SYMMETRY", True)),
        use_rd=bool(cfg.get("TEXTURE_USE_RD", True)),
        use_td=bool(cfg.get("TEXTURE_USE_TD", True)),
        use_nd=bool(cfg.get("TEXTURE_USE_ND", True)),
        half_width_deg=float(cfg.get("TEXTURE_HALF_WIDTH", 7.5)),
        grid_points="auto" if grid == "auto" else int(grid),
        unit_normalize=bool(cfg.get("TEXTURE_UNIT_NORMALIZE", True)),
        plot_mode=str(cfg.get("TEXTURE_MODE", "density")),
        projection=str(cfg.get("TEXTURE_PROJECTION", "equal_area")),
        hemisphere=str(cfg.get("TEXTURE_HEMISPHERE", "upper")),
        cmap=str(cfg.get("TEXTURE_CMAP", "viridis")),
        difference_cmap=str(cfg.get("TEXTURE_DIFF_CMAP", "RdBu_r")),
        levels=int(cfg.get("TEXTURE_LEVELS", 10)),
        contour_scale=str(cfg.get("TEXTURE_SCALE", "linear")))


def handle_compare_before_after(state: PipelineState) -> dict:
    """Junction line lengths and equivalent diameters, raw voxels against the mesh."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import (
        compare_before_after, format_summary, report_summary)

    cfg = state.config
    hierarchy = active_hierarchy(state)
    if hierarchy is None:
        return {"skipped": True,
                "reason": "The comparison needs an FM-steel raw folder as the input."}
    tets = state.get_artefact("verified_tets")
    if tets is None:
        raise RuntimeError("verified_tets missing")
    identity = build_identity(state, hierarchy)
    result = compare_before_after(
        hierarchy, tets.points, tets.tetrahedra, tets.grain_ids, identity,
        comparison_settings(cfg))
    state.store_artefact(COMPARISON_KEY, result)
    summary = report_summary(result)
    lines = result["junction_lines"]
    write_json(ensure_run_dir(cfg) / "comparison_summary.json", {
        "summary": summary,
        "junction_sets_matched": lines["by_region_set"]["matched"],
        "junction_sets_only_before": [
            {"set": list(k), "length": v} for k, v in lines["by_region_set"]["only_before"].items()],
        "junction_sets_only_after": [
            {"set": list(k), "length": v} for k, v in lines["by_region_set"]["only_after"].items()],
        "diameters_matched": result["diameters"]["matched"],
    })
    return {**summary, "text": format_summary(result)}


def handle_export_surfaces(state: PipelineState) -> dict:
    cfg = state.config
    run = export_folder(cfg, "surfaces")
    written = []
    for key, name in (
        ("gmsh_result", "gmsh.vtp"),
        ("closed_rve", "closed_RVE.vtp"),
        ("remeshed_rve", "closed_RVE_remeshed.vtp"),
        ("repaired_rve", "repaired_RVE.vtp"),
    ):
        surface = state.get_artefact(key)
        if surface is None:
            continue
        path = run / name
        save_surface_vtp(path, surface)
        written.append(str(path))
    report = {"written": written}
    write_json(run / "export_surfaces.json", report)
    return report


def handle_export_volume(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p1.gmsh_tets import save_grain_tetrahedra

    cfg = state.config
    tets = state.get_artefact("verified_tets")
    if tets is None:
        raise RuntimeError("verified_tets missing — run tetrahedral meshing first.")
    out = export_folder(cfg, "volume") / "verified_tets.vtu"
    save_grain_tetrahedra(tets, out)
    write_json(out.with_suffix(".json"), tets.report)
    return {"path": str(out), "ready": tets.report.get("ready")}


def handle_export_abaqus(state: PipelineState) -> dict:
    from upxo.meshing.gbconformant.d3v2p0.abaqus_export import export_tets_abaqus

    cfg = state.config
    want_deck = bool(cfg.get("EXPORT_FM_DECK", True))
    want_mesh_only = bool(cfg.get("EXPORT_MESH_ONLY", False))
    if not (want_deck or want_mesh_only):
        raise ValueError("Select the FM-steel deck, the mesh-only deck, or both.")
    tets = state.get_artefact("verified_tets")
    if tets is None:
        raise RuntimeError("verified_tets missing")
    if cfg.get("REQUIRE_READY_TET_MESH") and not tets.report.get("ready"):
        raise RuntimeError("Tet mesh is below quality thresholds; refuse export.")
    base = ensure_run_dir(cfg) / "abaqus"
    if subset_run_active(cfg):
        from upxo.meshing.gbconformant.fm_steel_abaqus.subset import indexed_name, next_index
        stamp = indexed_name(next_index(base))      # 001, 002, ... one number per export
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report: dict = {}
    folder = None
    if want_deck:
        deck_report = export_fm_deck(state, tets, base / (stamp + "_fm"))
        report["fm_deck"] = deck_report
        folder = Path(deck_report["folder"])
    if want_mesh_only:
        labels = state.get_artefact("labels_meshing")
        if labels is None:
            labels = state.get_artefact("labels_upscaled")
            if labels is None:
                labels = current_labels(state)
        mesh_only = Path(export_tets_abaqus(
            tets, base / (stamp + "_mesh" if subset_run_active(cfg) else stamp),
            prefix=str(cfg.get("ABAQUS_ELSET_PREFIX", "grn_")),
            units=str(cfg.get("ABAQUS_UNITS", "unspecified")),
            expected_grains=np.unique(labels)))
        report["mesh_only_folder"] = str(mesh_only)
        folder = folder or mesh_only
    state.store_artefact("abaqus_folder", folder)
    report["folder"] = str(folder)
    return report
