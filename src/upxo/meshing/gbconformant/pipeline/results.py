"""What a run produced, as tables, text, figures and files: the material of a run summary and report.

Everything here reads a :class:`PipelineState` (a run that just finished, or one restored from its
run folder) and builds plain tables of ``(title, columns, rows)``, matplotlib figures, and the list
of files written. :func:`populate_report` puts the standard sections into a
:class:`upxo.reporting.ReportSession`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.runfiles import ensure_run_dir
from upxo.meshing.gbconformant.pipeline.settings import default_config, differences
from upxo.meshing.gbconformant.pipeline.stage_input import active_hierarchy
from upxo.meshing.gbconformant.pipeline.stage_output import (
    COMPARISON_KEY, build_identity, comparison_settings, deck_settings, texture_settings)
from upxo.meshing.gbconformant.pipeline.stage_quality import REGION_KEY
from upxo.meshing.gbconformant.pipeline.stages import STAGES, stage_def
from upxo.meshing.gbconformant.pipeline.state import PipelineState, StageStatus

SECTIONS = ("summary", "settings", "quality", "fidelity", "deliverables")
SECTION_TITLES = {
    "summary": "Run summary",
    "settings": "Settings changed from the defaults",
    "quality": "Mesh quality",
    "fidelity": "Fidelity to the voxel structure",
    "deliverables": "Deliverables",
}
FIDELITY_PLOTS = ("line_length", "line_length_per_grain", "diameter", "matched_lines",
                  "matched_grains", "matched_diameters", "pole_figures", "grain_volume", "grain_area",
                  "grain_aspect", "coordination", "misorientation_boundary", "misorientation_grain")
GRAIN_PLOTS = ("grain_volume", "grain_area", "grain_aspect", "coordination", "misorientation_boundary",
               "misorientation_grain")
# plot key -> (parameter of the comparison, view)
PLOT_PARAMETERS = {
    "line_length": ("line_length", "distribution"),
    "line_length_per_grain": ("line_length_per_grain", "distribution"),
    "diameter": ("diameter", "distribution"),
    "matched_lines": ("line_length", "matched"),
    "matched_grains": ("line_length_per_grain", "matched"),
    "matched_diameters": ("diameter", "matched"),
    "pole_figures": ("pole_figures", "pole"),
    "grain_volume": ("grain_volume", "matched"),
    "grain_area": ("grain_area", "matched"),
    "grain_aspect": ("aspect", "matched"),
    "coordination": ("coordination", "distribution"),
    "misorientation_boundary": ("misorientation_boundary", "distribution"),
    "misorientation_grain": ("misorientation_grain", "distribution"),
}
WORST_GRAINS = 10


@dataclass
class Block:
    """A titled table."""

    title: str
    columns: list
    rows: list

    def text(self) -> str:
        cells = [[str(c) for c in self.columns]] + [[_cell(v) for v in row] for row in self.rows]
        widths = [max(len(r[i]) for r in cells) for i in range(len(self.columns))]
        lines = [self.title, "-" * len(self.title)]
        for n, row in enumerate(cells):
            lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
            if n == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)


@dataclass
class ResultFigure:
    key: str
    title: str
    figure: Any
    note: str = ""


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _facts(title: str, pairs) -> Optional[Block]:
    rows = [[label, value] for label, value in pairs if value is not None]
    return Block(title, ["Item", "Value"], rows) if rows else None


def _report(state: PipelineState, stage_id: str) -> dict:
    report = state.reports.get(stage_id)
    return report if isinstance(report, dict) else {}


def _ran(state: PipelineState, stage_id: str) -> bool:
    return state.status.get(stage_id) in (StageStatus.OK, StageStatus.SKIPPED) and bool(
        _report(state, stage_id))


def _done(state: PipelineState, stage_id: str) -> bool:
    return state.status.get(stage_id) == StageStatus.OK


def _count_or_none(values) -> Any:
    """A list of ids as "none" or "N: id, id, ..." (the first few)."""
    if not isinstance(values, (list, tuple)):
        return values
    if not values:
        return "none"
    shown = ", ".join(str(v) for v in list(values)[:8])
    return f"{len(values)}: {shown}" + (", ..." if len(values) > 8 else "")


def _interval(band) -> Any:
    if isinstance(band, (list, tuple)) and len(band) == 2:
        return f"[{band[0]:.4g}, {band[1]:.4g}]"
    return band


def _shape(values) -> Optional[str]:
    return None if not values else " x ".join(str(int(v)) for v in values)


# --- run summary ---------------------------------------------------------------------------------
def stage_rows(state: PipelineState) -> list[list]:
    """``[stage, title, status, seconds]`` for every stage, in run order."""
    rows = []
    for stage in STAGES:
        status = state.status.get(stage.id)
        seconds = _report(state, stage.id).get("duration_s")
        rows.append([stage.id, stage.title, status.value if status else "Not run", seconds])
    return rows


def input_block(state: PipelineState) -> Optional[Block]:
    report = _report(state, "load_lgi")
    if not report:
        return None
    labels = state.get_artefact("load_lgi.labels_full")
    return _facts("Input", [
        ("Source", report.get("path")),
        ("Dataset type", report.get("dataset_type", state.config.get("DATASET_TYPE"))),
        ("Hierarchy level", report.get("level")),
        ("Levels available", ", ".join(report.get("available_levels") or []) or None),
        ("Voxels (x y z)", _shape(report.get("shape") or getattr(labels, "shape", None))),
        ("Voxel count", report.get("n_voxels")),
        ("Grains", report.get("n_grains")),
        ("Voxel size", report.get("voxel_size")),
        ("Source units", report.get("source_units")),
    ])


def _count_rows(counts) -> list:
    """``("<name> in the subset", "<crop> of <structure>")`` for each level the structure has."""
    if not counts:
        return []
    from upxo.meshing.gbconformant.fm_steel_abaqus.subset import COUNT_NAMES
    return [(f"{COUNT_NAMES[name][0].upper() + COUNT_NAMES[name][1:]} in the subset", f"{n} of {counts['parent'].get(name)}")
            for name, n in counts["crop"].items()]


def subset_block(state: PipelineState) -> Optional[Block]:
    report = _report(state, "subset_extract")
    if not report.get("run_folder"):
        return None
    return _facts("Subset of the input", [
        ("Origin in the parent (voxels)", ", ".join(str(v) for v in report.get("origin_voxels") or [])),
        ("Subset (voxels)", _shape(report.get("shape"))),
        ("Parent (voxels)", _shape(report.get("parent_shape"))),
        ("Features in the parent", report.get("features_in_parent")),
        ("Features in the subset", report.get("features_in_crop")),
        ("Features cut by the box", report.get("cut_features")),
        *_count_rows(report.get("counts")),
        ("Warnings about the crop", "; ".join(report.get("warnings") or []) or None),
        ("Fragments below the size threshold", report.get("fragments_below_threshold")),
        ("Smallest feature in the subset (voxels)", report.get("smallest_in_crop")),
        ("Run folder", report.get("run_folder")),
    ])


def cleaning_block(state: PipelineState) -> Optional[Block]:
    rows = []
    r = _report(state, "small_grains")
    if r:
        if r.get("skipped") or r.get("enabled") is False:
            rows.append(["Small grains", "off"])
        else:
            rows.append(["Small grains", f"{r.get('n_grains_before')} -> {r.get('n_grains_after')} grains; "
                                         f"{len(r.get('removed_ids') or [])} absorbed (below {r.get('min_voxels')} voxels)"])
    r = _report(state, "spikes")
    if r:
        rows.append(["Spikes", "off" if r.get("enabled") is False else
                     f"{r.get('net_changed_voxels')} voxels changed; {r.get('stop_reason')}"])
    r = _report(state, "pinches")
    if r:
        rows.append(["Pinch contacts", "off" if r.get("enabled") is False else
                     f"{r.get('initial_candidates')} found, {r.get('accepted_changes')} repaired, "
                     f"{r.get('remaining_candidates')} left; {r.get('net_changed_voxels')} voxels changed"])
    r = _report(state, "voxel_topology")
    if r:
        rows.append(["Voxel topology", "off" if r.get("enabled") is False else
                     f"defects {r.get('initial_defects')} -> {r.get('remaining_defects')}; "
                     f"{r.get('net_changed_voxels')} voxels changed"])
    r = _report(state, "freeze_thin")
    if r:
        rows.append(["Thin grains", "off" if r.get("enabled") is False else
                     f"{r.get('n_frozen')} of {r.get('total_grains')} kept fixed ({r.get('criterion')})"])
    r = _report(state, "upscale")
    if r:
        rows.append(["Upscale", f"factor {r.get('factor')}: {_shape(r.get('shape_before'))} -> "
                                f"{_shape(r.get('shape_after'))} voxels"])
    return Block("Cleaning of the voxel structure", ["Step", "Result"], rows) if rows else None


def surface_block(state: PipelineState) -> Optional[Block]:
    rows = []
    r = _report(state, "smooth_interfaces")
    if r:
        rows.append(["Smoothed interfaces", f"{r.get('n_points')} points, {r.get('n_triangles')} triangles, "
                                             f"{r.get('accepted_iterations')} iterations"])
    r = _report(state, "gmsh_interfaces")
    if r:
        rows.append(["Gmsh remesh of the interfaces", f"{r.get('input_triangles')} -> {r.get('output_triangles')} triangles; "
                                                       f"{r.get('remaining_intersections')} intersections left"])
    r = _report(state, "close_rve")
    if r:
        rows.append(["RVE faces closed", f"{r.get('cap_triangles')} cap triangles, {r.get('total_triangles')} in total; "
                                         f"all six faces closed: {_cell(r.get('all_six_faces_closed'))}"])
    r = _report(state, "joint_remesh")
    if r:
        rows.append(["Joint remesh", f"{r.get('input_triangles')} -> {r.get('output_triangles')} triangles; "
                                     f"{r.get('remaining_intersections')} intersections left"])
    r = _report(state, "pre_tet_validation")
    if r:
        rows.append(["Validation before repair", f"{r.get('status')}; {r.get('n_blockers')} blockers, "
                                                 f"{r.get('n_warnings')} warnings"])
    r = _report(state, "surface_gate")
    if r:
        rows.append(["Surface gate", f"{r.get('status')}; {r.get('n_blockers')} blockers; "
                                     f"poor surface: {_cell(r.get('poor_surface'))}"])
    return Block("Interface surface", ["Step", "Result"], rows) if rows else None


def mesh_block(state: PipelineState) -> Optional[Block]:
    tets = state.get_artefact("verified_tets")
    if tets is None:
        return None
    report = dict(getattr(tets, "report", None) or {})
    dihedral = report.get("dihedral_angles") or {}
    percentiles = report.get("quality_percentiles_1_5_50") or []
    return _facts("Tetrahedral mesh", [
        ("Tetrahedra", int(len(tets.tetrahedra))),
        ("Nodes", int(len(tets.points))),
        ("Grains", report.get("grains")),
        ("Total volume", report.get("total_volume")),
        ("Status", report.get("status")),
        ("Meets the quality targets", report.get("ready")),
        ("Quality measure", report.get("quality_metric")),
        ("Lowest quality", report.get("minimum_quality")),
        ("Quality percentiles 1 / 5 / 50", ", ".join(f"{v:.4g}" for v in percentiles) or None),
        ("Smallest dihedral angle (deg)", dihedral.get("minimum")),
        ("Largest dihedral angle (deg)", dihedral.get("maximum")),
        ("Positive element volumes", report.get("positive_volumes")),
        ("Conforms to the surface", report.get("source_surface_conformity_verified")),
    ])


def tet_stage_block(state: PipelineState) -> Optional[Block]:
    rows = []
    for stage_id in ("tet_mesh", "tet_smooth", "tet_swaps", "tet_smooth_after", "dihedral_gate",
                     "insert_nodes"):
        r = _report(state, stage_id)
        if not r or r.get("enabled") is False or r.get("skipped"):
            continue
        dihedral = r.get("dihedral_angles") or {}
        n_tets = r.get("n_tets") or (r.get("quality_after") or {}).get("n")
        rows.append([stage_def(stage_id).title, n_tets, r.get("minimum_quality"),
                     dihedral.get("minimum"), dihedral.get("maximum")])
    return Block("Tetrahedral mesh after each stage",
                 ["Stage", "Tetrahedra", "Lowest quality", "Smallest dihedral (deg)",
                  "Largest dihedral (deg)"], rows) if rows else None


def export_block(state: PipelineState) -> Optional[Block]:
    r = _report(state, "export_abaqus")
    if not r:
        return None
    deck = r.get("fm_deck") or {}
    return _facts("Abaqus export", [
        ("Folder", r.get("folder")),
        ("Hierarchy level", deck.get("level")),
        ("Tetrahedra", deck.get("tetrahedra")),
        ("Features with elements", deck.get("features_with_elements")),
        ("Features without elements", _count_or_none(deck.get("features_without_elements"))),
        ("Output units", deck.get("output_units")),
        ("Coordinate scale", deck.get("coordinate_scale")),
        ("Orientation rule when grains merged", deck.get("identity_rule")),
        ("Voxels reassigned by the merge rule", deck.get("voxels_moved")),
        ("Merged features in the merge log", deck.get("merged_features") or None),
        ("Load case", _load_text(deck.get("load_case"), deck.get("output_units"))),
        ("Analysis output", _output_text(deck.get("output_estimate"))),
        ("Deck check", _deck_check_text(deck.get("deck_check"))),
        ("Elements next to a grain boundary", _layer_text(deck.get("layer_sets"))),
        ("Mesh-only deck", r.get("mesh_only_folder")),
    ])


def _load_text(summary, units) -> Optional[str]:
    if not summary:
        return None
    from upxo.meshing.gbconformant.fm_steel_abaqus.loading import describe
    return "; ".join(describe(summary, units or ""))


def _output_text(figures) -> Optional[str]:
    if not figures:
        return None
    from upxo.meshing.gbconformant.fm_steel_abaqus.outputs import text
    return text(figures)


def boundary_block(state: PipelineState) -> Optional[Block]:
    """The grain boundaries by kind: how many, how much area, how far apart in orientation."""
    summary = (_report(state, "export_abaqus").get("fm_deck") or {}).get("boundary_classes")
    if not summary or not summary.get("by_kind"):
        return None
    rows = []
    for name, entry in summary["by_kind"].items():
        low = entry.get("low_angle_area_fraction")
        rows.append([name, entry["pairs"], entry["area"], round(100.0 * entry["area_fraction"], 2),
                     entry.get("mean_misorientation_deg"), None if low is None else round(100.0 * low, 2)])
    return Block(f"Grain boundaries by kind (low angle: below {summary['low_angle_degrees']:g} degrees)",
                 ["Kind", "Boundaries", "Area", "Share of area (%)", "Mean misorientation (deg)",
                  "Low-angle share (%)"], rows)


def _deck_check_text(check) -> Optional[str]:
    if not check:
        return None
    return "all checks passed" if check["ok"] else "FAILED: " + ", ".join(check["failed"])


def _layer_text(layers) -> Optional[str]:
    if not layers:
        return None
    total = layers['gb_layer_elements'] + layers['interior_elements']
    return f"{layers['gb_layer_elements']:,} of {total:,} ({100 * layers['gb_layer_fraction']:.1f} %)"


def timing_block(state: PipelineState) -> Optional[Block]:
    rows = [[title, status, seconds] for _, title, status, seconds in stage_rows(state)
            if seconds is not None]
    if not rows:
        return None
    total = sum(r[2] for r in rows)
    rows.append(["Total", "", round(total, 3)])
    return Block("Time per stage", ["Stage", "Status", "Seconds"], rows)


def _g(r: dict, key: str, default="?"):
    value = r.get(key)
    return default if value is None else value


_NOTES = {
    "load_lgi": lambda r: f"{_g(r, 'n_grains')} grains, {_shape(r.get('shape')) or '?'} voxels",
    "subset_extract": lambda r: f"{_shape(r.get('shape')) or '?'} voxels, {_g(r, 'features_in_crop')} features",
    "grain_id_map": lambda r: f"{_g(r, 'n_grains')} ids shuffled (seed {_g(r, 'seed')})",
    "small_grains": lambda r: f"{_g(r, 'n_grains_before')} -> {_g(r, 'n_grains_after')} grains",
    "spikes": lambda r: f"{_g(r, 'net_changed_voxels')} voxels changed",
    "pinches": lambda r: f"{_g(r, 'accepted_changes')} repaired, {_g(r, 'remaining_candidates')} left",
    "voxel_topology": lambda r: f"defects {_g(r, 'initial_defects')} -> {_g(r, 'remaining_defects')}",
    "freeze_thin": lambda r: f"{_g(r, 'n_frozen')} kept fixed",
    "upscale": lambda r: f"factor {_g(r, 'factor')}",
    "smooth_interfaces": lambda r: f"{_g(r, 'n_triangles')} triangles",
    "geometry_guard": lambda r: f"{_g(r, 'moved_nodes')} nodes moved, {_g(r, 'remaining_intersections')} intersections left",
    "guard_relax": lambda r: f"{_g(r, 'moved_nodes')} nodes moved",
    "unpinch": lambda r: f"{_g(r, 'split_contacts')} contacts split",
    "gmsh_interfaces": lambda r: f"{_g(r, 'input_triangles')} -> {_g(r, 'output_triangles')} triangles",
    "close_rve": lambda r: f"{_g(r, 'total_triangles')} triangles",
    "pre_joint_quality": lambda r: f"{_g(r, 'accepted_flips')} flips",
    "joint_remesh": lambda r: f"{_g(r, 'input_triangles')} -> {_g(r, 'output_triangles')} triangles",
    "pre_tet_validation": lambda r: f"{_g(r, 'status')}, {_g(r, 'n_blockers')} blockers",
    "surface_gate": lambda r: f"{_g(r, 'status')}, {_g(r, 'n_blockers')} blockers",
    "region_quality": lambda r: (f"{_g(r, 'junction_lines')} junction lines, {_g(r, 'junction_points')} "
                                 f"junction points, {_g(r, 'order')} layers"),
    "compare_before_after": lambda r: f"{_g(r, 'features_before')} -> {_g(r, 'features_after')} features",
    "export_surfaces": lambda r: f"{len(r.get('written') or [])} files",
    "export_volume": lambda r: Path(str(r.get("path", ""))).name,
    "export_abaqus": lambda r: Path(str(r.get("folder", ""))).name,
}
for _tet_stage in ("tet_mesh", "tet_smooth", "tet_swaps", "tet_smooth_after", "dihedral_gate", "insert_nodes"):
    _NOTES[_tet_stage] = lambda r: (
        f"{(_g(r, 'n_tets', None) or (r.get('quality_after') or {}).get('n') or '?')} tets, "
        f"lowest minSICN {_cell(r.get('minimum_quality'))}, {_g(r, 'status')}")


def stage_note(state: PipelineState, stage_id: str) -> str:
    """A line of key numbers for a stage that ran (``""`` when it has none to show)."""
    report = _report(state, stage_id)
    if state.status.get(stage_id) == StageStatus.FAILED:
        return str(state.errors.get(stage_id) or report.get("error") or "failed")
    if not report or report.get("skipped") or report.get("enabled") is False:
        return "switched off" if report.get("skipped") or report.get("enabled") is False else ""
    try:
        return _NOTES[stage_id](report) if stage_id in _NOTES else ""
    except Exception:
        return ""


def stage_summary_block(state: PipelineState) -> Optional[Block]:
    """One table of every stage that has a status: how it ended, how long it took, its key numbers."""
    rows, total = [], 0.0
    for stage in STAGES:
        status = state.status.get(stage.id)
        if status is None or status == StageStatus.NOT_RUN:
            continue
        seconds = _report(state, stage.id).get("duration_s")
        total += float(seconds) if isinstance(seconds, (int, float)) else 0.0
        rows.append([stage.title, status.value, seconds, stage_note(state, stage.id)])
    if not rows:
        return None
    rows.append(["Total", "", round(total, 3), ""])
    return Block("Stages", ["Stage", "Status", "Seconds", "Key numbers"], rows)


def summary_blocks(state: PipelineState) -> list[Block]:
    builders = (input_block, subset_block, cleaning_block, surface_block, mesh_block, tet_stage_block,
                export_block, boundary_block, stage_summary_block)
    return [block for block in (b(state) for b in builders) if block is not None]


def summary_text(state: PipelineState) -> str:
    blocks = summary_blocks(state)
    if not blocks:
        return "Nothing has been run yet."
    return "\n\n".join(block.text() for block in blocks)


# --- settings ------------------------------------------------------------------------------------------
def settings_block(state: PipelineState) -> Block:
    changes = [row for row in differences(default_config(), state.config) if not str(row["key"]).startswith("_")]
    rows = [[row["key"], row["default"], row["current"]] for row in changes]
    return Block(SECTION_TITLES["settings"], ["Setting", "Default", "This run"], rows)


# --- quality -------------------------------------------------------------------------------------------
def quality_blocks(state: PipelineState) -> list[Block]:
    from upxo.meshing.gbconformant.pipeline.quality_figures import (
        surface_distribution_arrays, tet_distribution_arrays)

    blocks = []
    for kind, arrays_of, measures in (
        ("Surface triangles", surface_distribution_arrays,
         (("smallest_angle", "Smallest angle (deg)"), ("largest_angle", "Largest angle (deg)"),
          ("shape_quality", "Shape quality"))),
        ("Tetrahedra", tet_distribution_arrays,
         (("smallest_dihedral", "Smallest dihedral (deg)"), ("largest_dihedral", "Largest dihedral (deg)"),
          ("minSICN", "Gmsh minSICN"))),
    ):
        arrays, stats = arrays_of(state)
        if arrays is None:
            continue
        rows = []
        for key, label in measures:
            entry = stats.get(key)
            if isinstance(entry, dict):
                rows.append([label, entry.get("worst"), _interval(entry.get("band")), entry.get("count")])
        extra = {
            "Surface triangles": [("below the minimum-angle limit", stats.get("below_min_angle_limit"))],
            "Tetrahedra": [("below the minimum dihedral limit", stats.get("below_min_limit")),
                           ("above the maximum dihedral limit", stats.get("above_max_limit"))],
        }[kind]
        title = f"{kind}: {stats.get('triangles', stats.get('tetrahedra'))} elements, worst band = " \
                f"{stats.get('fraction')} of them" + "".join(f"; {n} {label}" for label, n in extra)
        blocks.append(Block(title, ["Measure", "Worst value", "Worst band", "Elements in band"], rows))
    blocks += quality_report_blocks(state)
    table = tet_stage_block(state)
    if table is not None:
        blocks.append(table)
    blocks += region_blocks(state)
    return blocks


def region_data(state: PipelineState) -> Optional[dict]:
    """The result of the region-quality stage (``None`` until it has run)."""
    return state.get_artefact(REGION_KEY)


def region_blocks(state: PipelineState) -> list[Block]:
    """Quality of the elements at the grain boundary surface, junction lines and junction points, by layer."""
    from upxo.meshing.gbconformant.pipeline.region_quality import region_tables

    result = region_data(state)
    return [] if result is None else [Block(*table) for table in region_tables(result)]


def region_flat_block(state: PipelineState) -> Optional[Block]:
    """Every statistic of every region and layer, for a spreadsheet."""
    from upxo.meshing.gbconformant.pipeline.region_quality import region_flat_table

    result = region_data(state)
    return None if result is None else Block(*region_flat_table(result))


def region_result_figures(state: PipelineState) -> list[ResultFigure]:
    from upxo.meshing.gbconformant.pipeline.region_quality import region_figures

    result = region_data(state)
    if result is None:
        return []
    return [ResultFigure(key, title, figure) for key, title, figure in region_figures(result, figsize=(6.4, 3.4))]


def quality_figures(state: PipelineState) -> list[ResultFigure]:
    from upxo.meshing.gbconformant.pipeline.quality_figures import (
        build_surface_quality_distributions, build_tet_quality_distributions)

    out = []
    for kind, build in (("surface", build_surface_quality_distributions),
                        ("tets", build_tet_quality_distributions)):
        figures, _ = build(state, figsize=(6.4, 3.4))
        for n, figure in enumerate(figures):
            title = figure.axes[0].get_title() if figure.axes else f"{kind} {n + 1}"
            out.append(ResultFigure(f"quality.{kind}.{n}", title, figure))
    return out + region_result_figures(state)


def _cached(state: PipelineState, name: str, tokens, build):
    """Build once per state and inputs (the slow measures are asked for by several views)."""
    cache = state.__dict__.setdefault("_results_cache", {})
    key = (name, tuple(id(t) for t in tokens))
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _feature_names(state: PipelineState, hierarchy=None):
    """``{meshed label: element-set name}`` of the resolved feature of each meshed grain (empty without a hierarchy)."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.export import feature_elset_name

    hierarchy = hierarchy or active_hierarchy(state)
    if hierarchy is None:
        return {}
    prefixes = deck_settings(state.config).elset_prefixes
    identity = build_identity(state, hierarchy)
    return {int(meshed): feature_elset_name(hierarchy, prefixes, hierarchy.features[raw])
            for meshed, raw in identity.winner.items() if raw in hierarchy.features}


def quality_data(state: PipelineState, top: int = WORST_GRAINS):
    """:func:`quality_report` of the verified tetrahedra (``None`` before they exist)."""
    from upxo.meshing.gbconformant.pipeline.quality_report import quality_report

    tets = state.get_artefact("verified_tets")
    if tets is None:
        return None
    target = float(state.config.get("TET_TARGET_MIN_DIHEDRAL", 30.0))
    return _cached(state, f"quality.{top}.{target}", [tets],
                   lambda: quality_report(tets, target_dihedral=target, top=top))


def _aspect_values(state: PipelineState):
    from upxo.meshing.gbconformant.pipeline.quality_report import tet_volumes_and_aspect

    tets = state.get_artefact("verified_tets")
    return _cached(state, "aspect", [tets],
                   lambda: tet_volumes_and_aspect(tets.points, tets.tetrahedra)[1])


def quality_report_blocks(state: PipelineState) -> list[Block]:
    """The mesh against the usual limits, the aspect ratio, and the worst grains."""
    data = quality_data(state)
    if data is None:
        return []
    run, grains = data["run"], data["grains"]
    n = max(run["tetrahedra"], 1)

    def share(count):
        return f"{100.0 * count / n:.3g} %"

    rows = []
    for t, count in run["dihedral_below"].items():
        rows.append([f"smallest dihedral below {t} deg", count, share(count)])
    for t, count in run["dihedral_above"].items():
        rows.append([f"largest dihedral above {t} deg", count, share(count)])
    for t, count in (run.get("sicn_below") or {}).items():
        rows.append([f"minSICN below {t}", count, share(count)])
    over = int(np.sum(np.asarray(_aspect_values(state)) > 5))
    rows.append(["aspect ratio above 5", over, share(over)])
    blocks = [Block(f"Tetrahedra against the usual limits ({run['tetrahedra']} tetrahedra, {run['grains']} grains)",
                    ["Tetrahedra with", "Count", "Share"], rows)]
    aspect = run["aspect_ratio"]
    if aspect.get("count"):
        blocks.append(_facts("Aspect ratio of the tetrahedra (1 is regular)", [
            ("Median", aspect["median"]), ("95th percentile", aspect["p95"]),
            ("99th percentile", aspect["p99"]), ("Largest", aspect["max"]),
            ("Degenerate (zero volume)", run["degenerate"] or None)]))
    names = _feature_names(state)
    index = {int(g): i for i, g in enumerate(grains["grain"])}
    worst_rows = []
    for g in data["worst"]:
        i = index[g]
        worst_rows.append([g, names.get(g), int(grains["tets"][i]), float(grains["volume"][i]),
                           float(grains["min_dihedral"][i]), float(grains["max_dihedral"][i]),
                           float(grains["min_sicn"][i]) if "min_sicn" in grains else None,
                           f"{100.0 * grains['below_target_fraction'][i]:.3g} %", float(grains["max_aspect"][i])])
    blocks.append(Block(f"The {len(worst_rows)} grains with the smallest dihedral angle",
                        ["Grain", "Feature", "Tetrahedra", "Volume", "Smallest dihedral (deg)",
                         "Largest dihedral (deg)", "Lowest minSICN",
                         f"Below {run['target_dihedral']:g} deg", "Largest aspect ratio"], worst_rows))
    return blocks


def quality_grain_table(state: PipelineState) -> Optional[Block]:
    """Every grain's quality, for a spreadsheet."""
    data = quality_data(state)
    if data is None:
        return None
    grains = data["grains"]
    names = _feature_names(state)
    with_sicn = "min_sicn" in grains
    columns = ["grain", "feature", "tets", "volume", "min_dihedral", "max_dihedral", "below_target",
               "below_target_fraction", "max_aspect"] + (["min_sicn", "mean_sicn"] if with_sicn else [])
    rows = []
    for i, g in enumerate(grains["grain"]):
        row = [int(g), names.get(int(g)), int(grains["tets"][i]), float(grains["volume"][i]),
               float(grains["min_dihedral"][i]), float(grains["max_dihedral"][i]),
               int(grains["below_target"][i]), float(grains["below_target_fraction"][i]),
               float(grains["max_aspect"][i])]
        if with_sicn:
            row += [float(grains["min_sicn"][i]), float(grains["mean_sicn"][i])]
        rows.append(row)
    return Block("Quality of every grain", columns, rows)


def write_csv(block: Block, path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(block.columns)
        writer.writerows([["" if v is None or v != v else v for v in row] for row in block.rows])
    return path


# --- fidelity ------------------------------------------------------------------------------------------
_STAT_ROWS = (("count", "Count"), ("total", "Total"), ("mean", "Mean"), ("median", "Median"),
              ("std", "Std. dev."), ("min", "Minimum"), ("p05", "5th percentile"), ("p25", "25th percentile"),
              ("p75", "75th percentile"), ("p95", "95th percentile"), ("max", "Maximum"))


def _comparison(state: PipelineState):
    return state.get_artefact(COMPARISON_KEY)


def fidelity_blocks(state: PipelineState) -> list[Block]:
    """The before/after tables (when the comparison ran), the per-feature measures and the merge log."""
    return _comparison_blocks(state) + grain_blocks(state) + merge_blocks(state)


def _comparison_blocks(state: PipelineState) -> list[Block]:
    """The tables of the chosen pair (voxel structure and tetrahedral mesh unless the settings say otherwise)."""
    from upxo.meshing.gbconformant.pipeline.comparison_view import comparison_tables

    if _comparison(state) is None:
        return []
    try:
        tables = comparison_tables(state)
    except ValueError as problem:
        return [_facts("Comparison", [("Not shown", str(problem))])]
    return [Block(*table) for table in tables]


def _raw_names(state: PipelineState, hierarchy) -> dict:
    from upxo.meshing.gbconformant.fm_steel_abaqus.export import feature_elset_name

    prefixes = deck_settings(state.config).elset_prefixes
    return {int(label): feature_elset_name(hierarchy, prefixes, feature)
            for label, feature in hierarchy.features.items()}


def grain_data(state: PipelineState):
    """Per-feature fidelity (volume, boundary area, neighbours, aspect ratio), or ``None`` without a mesh or an FM-steel input."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import grain_fidelity

    hierarchy, tets = active_hierarchy(state), state.get_artefact("verified_tets")
    if hierarchy is None or tets is None:
        return None

    def build():
        lut = build_identity(state, hierarchy).lookup()
        grains = np.asarray(tets.grain_ids)
        if grains.max() >= len(lut) or np.any(lut[grains] < 0):
            raise ValueError("A tetrahedron carries a grain with no resolved identity")
        return grain_fidelity(hierarchy, tets.points, tets.tetrahedra, lut[grains], comparison_settings(state.config),
                              shrink_limit=float(state.config.get("FIDELITY_SHRINK_LIMIT", 0.10)))

    return _cached(state, f"grains.{state.config.get('FIDELITY_SHRINK_LIMIT')}", [hierarchy, tets], build)


def grain_table(state: PipelineState) -> Optional[Block]:
    """Every feature's measures, for a spreadsheet."""
    data = grain_data(state)
    if data is None:
        return None
    names = _raw_names(state, active_hierarchy(state))
    columns = ["label", "name"] + data["columns"][1:]
    rows = [[row[0], names.get(row[0])] + row[1:] for row in data["rows"]]
    return Block(f"Fidelity of every feature ({data['units']})", columns, rows)


def grain_blocks(state: PipelineState) -> list[Block]:
    try:
        data = grain_data(state)
    except Exception as error:
        state.append_log(f"per-feature fidelity unavailable: {error}")
        return []
    if data is None:
        return []
    names = _raw_names(state, active_hierarchy(state))
    columns, rows, units, stats = data["columns"], data["rows"], data["units"], data["stats"]
    at = {name: i for i, name in enumerate(columns)}
    def percent(x):
        if x != x:
            return None
        return f"{0.0 if abs(x) < 1e-9 else 100.0 * x:.3g} %"
    change, ratio = stats["volume_change"], stats["area_ratio"]
    facts = _facts("Features compared one by one", [
        ("Features", len(rows)),
        (f"Shrunk by more than {100.0 * data['shrink_limit']:.3g} % of their volume", len(data["flagged"])),
        ("Features with no mesh", _count_or_none(data["lost"])),
        ("Volume change: median", percent(change.get("median", float("nan")))),
        ("Volume change: least", percent(change.get("min", float("nan")))),
        ("Volume change: most", percent(change.get("max", float("nan")))),
        ("Boundary area after / before: median", ratio.get("median")),
        ("Neighbours lost (all features)", int(sum(r[at["neighbours_lost"]] for r in rows if r[at["neighbours_lost"]] == r[at["neighbours_lost"]]))),
        ("Neighbours gained (all features)", int(sum(r[at["neighbours_gained"]] for r in rows if r[at["neighbours_gained"]] == r[at["neighbours_gained"]]))),
        ("Aspect ratio after / before: median", stats["aspect_change"].get("median")),
    ])
    ordered = sorted((r for r in rows if r[at["volume_change"]] == r[at["volume_change"]]),
                     key=lambda r: r[at["volume_change"]])[:WORST_GRAINS]
    table = Block(f"The {len(ordered)} features that lost the most volume (volumes in {units}^3, areas in {units}^2)",
                  ["Feature", "Volume before", "Volume after", "Volume change", "Boundary area ratio",
                   "Neighbours before", "Neighbours after", "Aspect before", "Aspect after", "Flag"],
                  [[names.get(r[0], r[0]), r[at["volume_before"]], r[at["volume_after"]], percent(r[at["volume_change"]]),
                    r[at["area_ratio"]], r[at["neighbours_before"]], r[at["neighbours_after"]],
                    r[at["aspect_before"]], r[at["aspect_after"]], r[at["flag"]] or None] for r in ordered])
    return [b for b in (facts, table) if b is not None]


def merge_data(state: PipelineState):
    """The merge log of the identity rule, ``{'unavailable': why}`` for a symmetry it cannot treat, ``None`` without a merge."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import merge_log

    hierarchy = active_hierarchy(state)
    if hierarchy is None:
        return None
    identity = build_identity(state, hierarchy)
    if not identity.report.get("merged_groups"):
        return None
    try:
        return merge_log(hierarchy, identity, symmetry=str(state.config.get("TEXTURE_CRYSTAL_SYMMETRY", "cubic")),
                         prefixes=deck_settings(state.config).elset_prefixes)
    except ValueError as error:
        return {"unavailable": str(error)}


def merge_blocks(state: PipelineState) -> list[Block]:
    try:
        log = merge_data(state)
    except Exception as error:
        state.append_log(f"merge log unavailable: {error}")
        return []
    if log is None:
        return []
    if "unavailable" in log:
        return [_facts("Merged grains", [("Misorientation", log["unavailable"])])]
    summary, rows = log["summary"], sorted(log["rows"], key=lambda r: -r[6])
    mis = summary["misorientation"]
    blocks = [_facts("Merged grains: orientation substitution", [
        ("Merged grains", summary["groups"]),
        ("Features absorbed", summary["absorbed_features"]),
        ("Voxels they hold", summary["voxels"]),
        ("Identity rule", summary["rule"]),
        ("Misorientation to the winner: median (deg)", mis.get("median")),
        ("Misorientation to the winner: largest (deg)", mis.get("max")),
        ("Misorientation, voxel-weighted mean (deg)", summary["voxel_weighted_mean_deg"]),
        ("Absorbed features above 15 deg", summary["above_15_deg"]),
    ])]
    blocks.append(Block(f"Absorbed features with the largest misorientation (of {len(rows)})",
                        ["Meshed grain", "Absorbed feature", "Voxels", "Takes the orientation of", "Misorientation (deg)"],
                        [[r[0], r[2], r[3], r[5], r[6]] for r in rows[:15]]))
    return [b for b in blocks if b is not None]


def merge_table(state: PipelineState) -> Optional[Block]:
    log = merge_data(state)
    if not log or "unavailable" in log:
        return None
    return Block("Merge log", log["columns"], log["rows"])


def fidelity_figure(state: PipelineState, which: str) -> ResultFigure:
    """One of :data:`FIDELITY_PLOTS`, drawn for the chosen pair with the plot settings of the run."""
    from upxo.meshing.gbconformant.pipeline.comparison_view import SnapshotUnavailable, comparison_figure

    if which not in PLOT_PARAMETERS:
        raise KeyError(f"{which!r} is not one of {', '.join(FIDELITY_PLOTS)}.")
    if _comparison(state) is None and which not in GRAIN_PLOTS:
        raise RuntimeError("The before/after comparison has not been run.")
    parameter, view = PLOT_PARAMETERS[which]
    try:
        picture = comparison_figure(state, parameter, view)
    except SnapshotUnavailable as problem:
        raise RuntimeError(str(problem)) from None
    return ResultFigure(f"fidelity.{which}", picture.title, picture.figure, picture.note)


def grain_data_ok(state: PipelineState) -> bool:
    try:
        return grain_data(state) is not None
    except Exception:
        return False


def fidelity_figures(state: PipelineState) -> list[ResultFigure]:
    """Every fidelity plot that can be drawn for this run."""
    if _comparison(state) is None and not grain_data_ok(state):
        return []
    out = []
    for which in FIDELITY_PLOTS:
        try:
            out.append(fidelity_figure(state, which))
        except Exception as error:                      # one plot failing must not hide the others
            state.append_log(f"fidelity plot {which} skipped: {error}")
    return out


# --- deliverables --------------------------------------------------------------------------------------
def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def deliverables(state: PipelineState) -> list[dict]:
    """The files and folders a run wrote: ``kind``, ``label``, ``path``, ``exists``, ``bytes``."""
    run = ensure_run_dir(state.config)
    rows: list[tuple[str, str, Path]] = [("folder", "Run folder", run)]

    def add(kind, label, path):
        if path:
            rows.append((kind, label, Path(str(path))))

    add("folder", "Saved stage results", run / "checkpoints")
    add("file", "Run manifest", run / "run_manifest.json")
    add("file", "Subset record", run / "subset.json")
    for stage_id in ("export_surfaces",):
        for path in _report(state, stage_id).get("written") or []:
            add("file", "Surface (VTP)", path)
    add("file", "Tetrahedral mesh (VTU)", _report(state, "export_volume").get("path")
        or _report(state, "tet_mesh").get("path"))
    export = _report(state, "export_abaqus")
    deck = (export.get("fm_deck") or {}).get("folder")
    add("folder", "Abaqus deck", deck)
    if deck:
        for name, label in (("fm_mesh.msh", "Mesh with orientations (Gmsh)"), ("fm_mesh.vtu", "Mesh with orientations (VTK)"),
                            ("features.csv", "Feature table (CSV)"), ("deck_check.json", "Deck check"),
                            ("boundary_classes.csv", "Grain boundaries by kind (CSV)"),
                            ("fm_interfaces.vtp", "Grain boundary surface (VTK)"),
                            ("provenance.json", "Record to reproduce the deck"),
                            ("submit.bat", "Job script (Windows)"), ("submit.sh", "Job script (Linux)"),
                            ("submit.slurm", "Job script (Slurm)"), ("abaqus_datacheck.json", "Abaqus data check")):
            if (Path(str(deck)) / name).is_file():
                add("file", label, Path(str(deck)) / name)
    add("folder", "Mesh-only Abaqus deck", export.get("mesh_only_folder"))
    for kind, label, folder in (("folder", "Surface exports", run / "surfaces"),
                                ("folder", "Volume exports", run / "volume"),
                                ("folder", "Abaqus exports", run / "abaqus"),
                                ("folder", "Report", run / "report")):
        add(kind, label, folder)
    add("file", "Before/after comparison", run / "comparison_summary.json")
    for name, label in (("region_quality.csv", "Quality by region (CSV)"),
                        ("region_tets.vtu", "Tetrahedra with their layers and quality (VTK)"),
                        ("region_triangles.vtp", "Surface triangles with their layers and quality (VTK)")):
        if (run / "region_quality" / name).is_file():
            add("file", label, run / "region_quality" / name)
    seen, out = set(), []
    for kind, label, path in rows:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        exists = path.exists()
        out.append({"kind": kind, "label": label, "path": key, "exists": exists,
                    "bytes": _size(path) if exists else 0})
    return out


def deliverables_block(state: PipelineState) -> Block:
    rows = [[d["label"], d["path"], human_size(d["bytes"]) if d["exists"] else "not written"]
            for d in deliverables(state)]
    return Block(SECTION_TITLES["deliverables"], ["Item", "Path", "Size"], rows)


# --- report --------------------------------------------------------------------------------------------
def report_session(state: PipelineState, folder: Optional[Path] = None):
    """The run's report (continued if it already has entries)."""
    from upxo.reporting import ReportSession

    folder = Path(folder) if folder is not None else ensure_run_dir(state.config) / "report"
    session = ReportSession.open(folder, run_title=f"Conformal meshing: {state.config.get('RUN_NAME')}",
                                 pipeline_name="Grain-conformal tetrahedral meshing")
    session.update_run_metadata(**{k: v for k, v in (
        ("run_name", state.config.get("RUN_NAME")),
        ("dataset", _report(state, "load_lgi").get("path")),
        ("hierarchy_level", _report(state, "load_lgi").get("level"))) if v})
    return session


def _drop_section(session, title: str) -> None:
    """Remove the section called ``title`` and everything in it."""
    inside, remove = False, []
    for entry in session.entries:
        if entry.kind == "section":
            inside = entry.title == title
        if inside:
            remove.append(entry.id)
    for entry_id in remove:
        session.remove_entry(entry_id)


def _add_block(session, block: Block, source: str) -> None:
    session.add_table({c: [row[i] for row in block.rows] for i, c in enumerate(block.columns)}
                      if block.rows else {c: [] for c in block.columns},
                      block.title, source_page=source)


def populate_report(session, state: PipelineState, sections=SECTIONS, *, figures: bool = True,
                    replace: bool = True) -> list[str]:
    """Put the standard sections into ``session``; returns the sections added.

    With ``replace`` a section the report already holds is rebuilt instead of repeated.
    """
    added = []
    for name in sections:
        if name not in SECTIONS:
            raise KeyError(f"{name!r} is not one of {', '.join(SECTIONS)}.")
        title = SECTION_TITLES[name]
        blocks: list[Block] = {
            "summary": lambda: summary_blocks(state),
            "settings": lambda: [settings_block(state)],
            "quality": lambda: quality_blocks(state),
            "fidelity": lambda: fidelity_blocks(state),
            "deliverables": lambda: [deliverables_block(state)],
        }[name]()
        pictures = []
        if figures and name == "quality":
            pictures = quality_figures(state)
        elif figures and name == "fidelity":
            pictures = fidelity_figures(state)
        if not blocks and not pictures:
            continue
        if replace:
            _drop_section(session, title)
        session.add_section(title)
        source = f"results:{name}"
        for block in blocks:
            _add_block(session, block, source)
        for picture in pictures:
            session.add_image(picture.figure, picture.title, caption=picture.note or None, source_page=source)
        _add_spreadsheets(session, state, name, source)
        added.append(name)
    return added


def _add_spreadsheets(session, state: PipelineState, name: str, source: str) -> None:
    """Write the per-grain tables of a section next to the report and say so in it."""
    tables = {"quality": [("quality_per_grain.csv", quality_grain_table), ("quality_by_region.csv", region_flat_block)],
              "fidelity": [("fidelity_per_feature.csv", grain_table), ("merge_log.csv", merge_table)]}.get(name, [])
    written = []
    for filename, build in tables:
        try:
            block = build(state)
        except Exception as error:
            state.append_log(f"{filename} not written: {error}")
            continue
        if block is not None and block.rows:
            write_csv(block, session.reports_dir / filename)
            written.append(f"{filename}: {len(block.rows)} rows")
    if written:
        session.add_text("\n".join(written), "Tables for a spreadsheet (in the report folder)", source_page=source)
