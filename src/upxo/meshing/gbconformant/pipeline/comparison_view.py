"""The before/after comparison of a run: which two representations are compared, the tables and the figures.

The representations are the voxel structure, the grain boundary surface before and after smoothing, the closed
remeshed surface before tetrahedral meshing and the tetrahedral mesh (:mod:`...fm_steel_abaqus.snapshots`). They are
measured when asked for, from the artefacts of the run, and kept until an artefact or a setting changes. Everything is
labelled with the name of the representation, never with "before" and "after" alone, and lengths are shown in the units
of ``COMPARE_PLOT_UNITS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.stage_input import active_hierarchy
from upxo.meshing.gbconformant.pipeline.stage_output import (
    build_identity, comparison_settings, texture_settings)
from upxo.meshing.gbconformant.pipeline.state import PipelineState

from upxo.meshing.gbconformant.pipeline.stage_volume import PRE_TET_POINTS
DEFAULT_BEFORE, DEFAULT_AFTER = "Voxel structure", "Tetrahedral mesh"
STAT_ROWS = (("count", "Count"), ("mean", "Mean"), ("std", "Standard deviation"), ("min", "Smallest"),
             ("p05", "5th percentile"), ("median", "Median"), ("p95", "95th percentile"), ("max", "Largest"))


class SnapshotUnavailable(ValueError):
    """A representation cannot be measured for this run; the message says what to run first."""


@dataclass
class Picture:
    key: str
    title: str
    figure: Any
    note: str = ""


@dataclass
class ViewSettings:
    """The display settings of the comparison, read from a run's configuration."""

    before: str = "voxels"
    after: str = "tets"
    kind: str = "histogram"
    bins: int = 40
    bin_width: Optional[float] = None
    bandwidth: Optional[float] = None
    log: bool = False
    density: bool = False
    matched_kind: str = "scatter"
    units: str = "auto"
    color_before: str = "#4878CF"
    color_after: str = "#D65F5F"
    show_mean: bool = True
    show_std: bool = True
    show_median: bool = False
    normalisation: str = "fraction of the boundary area"
    grain_weight: str = "number of grains"


def _optional_float(value):
    if value is None or str(value).strip().lower() in ("", "auto", "none"):
        return None
    number = float(value)
    return number if number > 0 else None


def view_settings(cfg: dict) -> ViewSettings:
    """The comparison's display settings of ``cfg`` (checked)."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.plots import COLORS, DISTRIBUTION_KINDS, colour
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import DISPLAY_UNITS, GRAIN_WEIGHTS, NORMALISATIONS
    from upxo.meshing.gbconformant.fm_steel_abaqus.snapshots import ORDER, key_of_title

    before = key_of_title(cfg.get("COMPARE_BEFORE") or DEFAULT_BEFORE)
    after = key_of_title(cfg.get("COMPARE_AFTER") or DEFAULT_AFTER)
    if ORDER.index(before) >= ORDER.index(after):
        raise ValueError("Choose a 'before' representation that comes earlier in the pipeline than the 'after' one: "
                         "voxel structure, surface before smoothing, surface after smoothing, remeshed surface, "
                         "tetrahedral mesh.")
    kind = str(cfg.get("COMPARE_PLOT_KIND", "histogram"))
    units = str(cfg.get("COMPARE_PLOT_UNITS", "auto"))
    norm = str(cfg.get("COMPARE_MISORIENTATION_NORM", NORMALISATIONS[0]))
    weight = str(cfg.get("COMPARE_GRAIN_WEIGHT", GRAIN_WEIGHTS[0]))
    for name, value, allowed in (("COMPARE_PLOT_KIND", kind, DISTRIBUTION_KINDS), ("COMPARE_PLOT_UNITS", units, DISPLAY_UNITS),
                                 ("COMPARE_MISORIENTATION_NORM", norm, NORMALISATIONS),
                                 ("COMPARE_GRAIN_WEIGHT", weight, GRAIN_WEIGHTS)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {', '.join(allowed)}, not {value!r}.")
    return ViewSettings(
        before=before, after=after, kind=kind, bins=int(cfg.get("COMPARE_BINS", 40)),
        bin_width=_optional_float(cfg.get("COMPARE_BIN_WIDTH")), bandwidth=_optional_float(cfg.get("COMPARE_KDE_BANDWIDTH")),
        log=bool(cfg.get("COMPARE_LOG", False)), density=bool(cfg.get("COMPARE_DENSITY", False)),
        matched_kind=str(cfg.get("COMPARE_MATCHED_KIND", "scatter")), units=units,
        color_before=colour(cfg.get("COMPARE_COLOR_BEFORE"), COLORS["before"]),
        color_after=colour(cfg.get("COMPARE_COLOR_AFTER"), COLORS["after"]),
        show_mean=bool(cfg.get("COMPARE_SHOW_MEAN", True)), show_std=bool(cfg.get("COMPARE_SHOW_STD", True)),
        show_median=bool(cfg.get("COMPARE_SHOW_MEDIAN", False)), normalisation=norm, grain_weight=weight)


# --- snapshots ---------------------------------------------------------------------------------------------------
def _cache(state: PipelineState) -> dict:
    return state.__dict__.setdefault("_comparison_cache", {})


def _resolve(lut, labels, what):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.max() >= len(lut) or labels.min() < 0 or np.any(lut[labels] < 0)):
        raise ValueError(f"A grain of the {what} has no resolved identity.")
    return lut[labels]


def _context(state: PipelineState):
    hierarchy = active_hierarchy(state)
    if hierarchy is None:
        raise SnapshotUnavailable("The comparison needs an FM-steel raw folder (or a twinned FCC folder) as the input.")
    cfg = state.config
    token = (id(hierarchy), comparison_settings(cfg).__repr__(), id(state.get_artefact("grain_id_map")),
             id(state.get_artefact("small_grains.survivor_map")), cfg.get("IDENTITY_RULE"), cfg.get("IDENTITY_SEED"))
    cache = _cache(state)
    if ("context", token) not in cache:
        cache[("context", token)] = (hierarchy, comparison_settings(cfg), build_identity(state, hierarchy).lookup())
    return cache[("context", token)]


def get_snapshot(state: PipelineState, key: str):
    """The measured representation ``key`` (a key or a title); raises :class:`SnapshotUnavailable`."""
    from upxo.meshing.gbconformant.fm_steel_abaqus import snapshots as S

    key = S.key_of_title(key)
    hierarchy, settings, lut = _context(state)
    tets = state.get_artefact("verified_tets")
    smoothed = state.get_artefact("interfaces_smoothed")
    surface = state.get_artefact("repaired_rve")
    pre = state.get_artefact(PRE_TET_POINTS)
    tokens = {S.VOXELS: (), S.TETS: (tets,), S.SURFACE_RAW: (smoothed,), S.SURFACE_SMOOTHED: (smoothed,),
              S.SURFACE_PREPARED: (tets, surface, pre)}[key]
    cache_key = (key, id(hierarchy), repr(settings), tuple(id(t) for t in tokens), id(lut))
    cache = _cache(state)
    if cache_key in cache:
        return cache[cache_key]
    if key == S.VOXELS:
        snapshot = S.voxel_snapshot(hierarchy, settings)
    elif key == S.TETS:
        if tets is None:
            raise SnapshotUnavailable("Run Tetrahedral meshing first.")
        snapshot = S.tet_snapshot(hierarchy, tets.points, tets.tetrahedra, _resolve(lut, tets.grain_ids, "mesh"), settings)
    elif key in (S.SURFACE_RAW, S.SURFACE_SMOOTHED):
        if smoothed is None:
            raise SnapshotUnavailable("Run Extract + smooth interfaces first.")
        pairs = _resolve(lut, np.asarray(smoothed.grain_pairs), "surface")
        points = smoothed.original_points if key == S.SURFACE_RAW else smoothed.points
        snapshot = S.surface_snapshot(key, hierarchy, points, smoothed.triangles, pairs[:, 0], pairs[:, 1], settings)
    else:
        if tets is None or surface is None:
            raise SnapshotUnavailable("Run Tetrahedral meshing first.")
        if pre is None or len(pre) != len(surface.points):
            raise SnapshotUnavailable("The surface before tet meshing was not kept in this run: run Tetrahedral "
                                      "meshing again.")
        pairs = _resolve(lut, np.asarray(surface.grain_pairs), "surface")
        snapshot = S.surface_snapshot(key, hierarchy, pre, surface.triangles, pairs[:, 0], pairs[:, 1], settings,
                                      tets=(tets.points, tets.tetrahedra, _resolve(lut, tets.grain_ids, "mesh")))
    cache[cache_key] = snapshot
    return snapshot


def snapshot_problems(state: PipelineState) -> dict:
    """``{key: why not (or None when it can be measured)}`` for every representation."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.snapshots import ORDER

    out = {}
    for key in ORDER:
        try:
            get_snapshot(state, key)
            out[key] = None
        except SnapshotUnavailable as problem:
            out[key] = str(problem)
    return out


def pair_snapshots(state: PipelineState, view: Optional[ViewSettings] = None):
    """``(before snapshot, after snapshot, view settings)`` of the chosen pair."""
    view = view or view_settings(state.config)
    return get_snapshot(state, view.before), get_snapshot(state, view.after), view


def pair_phrase(before, after) -> str:
    from upxo.meshing.gbconformant.fm_steel_abaqus import snapshots as S

    known = {(S.VOXELS, S.TETS): "Before and after tetrahedral meshing",
             (S.SURFACE_RAW, S.SURFACE_SMOOTHED): "Before and after smoothing the surface",
             (S.SURFACE_PREPARED, S.TETS): "Before and after tetrahedral meshing, on the same surface"}
    return known.get((before.key, after.key), "Before and after")


# --- units ---------------------------------------------------------------------------------------------------------
def display_units(view: ViewSettings, power: int, source: str, arrays) -> str:
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import choose_units

    return choose_units(power, source, arrays) if view.units == "auto" else view.units


# --- figures -------------------------------------------------------------------------------------------------------
_Y_TEXT = {
    "misorientation_boundary": {"fraction": "fraction of the boundary area per degree",
                                "area": "boundary area per degree", "per_length": "boundary area per unit volume per degree"},
    "misorientation_grain": {"fraction": "volume fraction of the grains per degree"},
}


def _legend(snapshot, series, parameter_, area=None):
    """``Voxel structure: 271 grains`` (and the total boundary area for the boundary misorientation)."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import count_noun

    text = f"{snapshot.title}: {count_noun(series.noun, series.count)}"
    if area is not None and parameter_.key == "misorientation_boundary":
        text += f", area {area[0]:.4g} {area[1]}".rstrip()
    return text


def _scaled(series, parameter_, source, units):
    """Values and weights of ``series`` in ``units``; the weights follow what they add up to."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import unit_factor, unit_power

    power = unit_power(parameter_.dimension)
    values = np.asarray(series.values, dtype=float) * unit_factor(power, source, units)
    weights = series.weights
    if weights is not None:
        factor = {"area": 2, "per_length": -1}.get(series.y_kind, 0)
        weights = np.asarray(weights, dtype=float) * unit_factor(factor, source, units)
    return values, weights


def comparison_figure(state: PipelineState, parameter_key: str, view_name: str = "distribution",
                      view: Optional[ViewSettings] = None) -> Picture:
    """One parameter of the chosen pair as a distribution, as the matched change per item, or as pole figures."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import (
        MATCHED, Series, Unavailable, count_noun, matched_items, parameter, read, unit_factor, unit_power, unit_text)
    from upxo.meshing.gbconformant.fm_steel_abaqus.plots import (
        DrawSeries, PlotStyle, compare_figure, matched_change_figure, texture_figure)

    p = parameter(parameter_key)
    before, after, view = pair_snapshots(state, view)
    if view_name not in p.views:
        raise ValueError(f"{p.label} has no {view_name} view; its views are {', '.join(p.views)}.")
    cfg = state.config
    subtitle = f"{pair_phrase(before, after)}: before = {before.title}, after = {after.title}"
    colors = (view.color_before, view.color_after)
    if p.key == "pole_figures":
        from upxo.meshing.gbconformant.fm_steel_abaqus import orientation_table

        hierarchy = active_hierarchy(state)
        _, eb = orientation_table(hierarchy, [int(x) for x in before.labels])
        _, ea = orientation_table(hierarchy, [int(x) for x in after.labels])
        figure, diff = texture_figure(eb, ea, texture_settings(cfg), titles=(before.title, after.title), noun="grains")
        figure.suptitle(f"Pole figures: {before.title} and {after.title}", fontweight="bold")
        note = f"Pole figure difference ({after.title} - {before.title}) IQR: {diff['iqr']:.6g}" if diff is not None else ""
        return Picture("fidelity.pole_figures", "Pole figures", figure, note)
    try:
        sb = read(p, before, normalisation=view.normalisation, grain_weight=view.grain_weight)
        sa = read(p, after, normalisation=view.normalisation, grain_weight=view.grain_weight)
    except Unavailable as problem:
        raise ValueError(str(problem)) from None
    power = unit_power(p.dimension)
    source = before.units
    units = display_units(view, power, source, [sb.values, sa.values])
    plain, symbol = unit_text(power, units, plain=True), unit_text(power, units)
    footnote_parts = [n for n in (sb.note, sa.note) if n]
    if p.noun == "grains" and before.key == "voxels" and len(before.labels) > len(after.labels):
        gone = len(before.labels) - len(after.labels)
        footnote_parts.append(f"{count_noun('grains', gone)} of the {before.title.lower()} {'is' if gone == 1 else 'are'} absent from "
                              f"the {after.title.lower()}: the cleaning stages removed {'it' if gone == 1 else 'them'} before "
                              "the surface was built.")
    footnote = " ".join(footnote_parts)
    if view_name == MATCHED:
        keys, vb, va = matched_items(sb, sa)
        shown_b = _scaled(Series(vb, dimension=p.dimension), p, source, units)[0]
        shown_a = _scaled(Series(va, dimension=p.dimension), p, source, units)[0]
        rows = [{"set": k, "before": float(x), "after": float(y)} for k, x, y in zip(keys, shown_b, shown_a)]
        noun = "junction sets" if p.key == "line_length" else p.noun
        quantity = p.label + (" (per set of grains)" if p.key == "line_length" else "")
        figure = matched_change_figure(rows, quantity=quantity, units=symbol, kind=view.matched_kind, bins=view.bins,
                                       log=view.log, labels=(before.title, after.title), colors=colors, subtitle=subtitle,
                                       footnote=footnote, noun=noun)
        return Picture(f"fidelity.matched.{p.key}", f"Matched change: {p.label}", figure)
    vb, wb = _scaled(sb, p, source, units)
    va, wa = _scaled(sa, p, source, units)
    area_factor, area_symbol = unit_factor(2, source, units), unit_text(2, units)
    series = [DrawSeries(_legend(before, sb, p, ((sb.total or 0.0) * area_factor, area_symbol)), vb, wb, view.color_before,
                         sb.integer),
              DrawSeries(_legend(after, sa, p, ((sa.total or 0.0) * area_factor, area_symbol)), va, wa, view.color_after,
                         sa.integer)]
    style = PlotStyle(kind=view.kind, bins=view.bins, bin_width=view.bin_width, bandwidth=view.bandwidth, log=view.log,
                      density=view.density, colors=colors, show_mean=view.show_mean, show_std=view.show_std,
                      show_median=view.show_median)
    angle = p.dimension == "angle"
    xunit = "deg" if angle else symbol
    xlabel = p.label + (f" ({xunit})" if xunit else "")
    ylabel = ""
    y_kind = sb.y_kind if sb.weights is not None else "count"
    if y_kind != "count":
        base = _Y_TEXT.get(p.key, {}).get(y_kind, "weight per unit")
        scale = {"area": f" ({unit_text(2, units)}/deg)", "per_length": f" (1/{symbol}/deg)"}.get(y_kind, "")
        ylabel = base + scale
    elif p.noun:
        ylabel = f"number of {p.noun}"
    figure = compare_figure(series, style, xlabel=xlabel, ylabel=ylabel, title=p.label, subtitle=subtitle,
                            footnote=footnote, unit="deg" if angle else symbol, note_unit="deg" if angle else plain)
    return Picture(f"fidelity.distribution.{p.key}", p.label, figure, "; ".join(figure.compare_notes))


# --- tables -------------------------------------------------------------------------------------------------------
def _describe(values):
    from upxo.meshing.gbconformant.fm_steel_abaqus.comparison import describe

    return describe(values)


def comparison_tables(state: PipelineState, view: Optional[ViewSettings] = None) -> list[tuple]:
    """``[(title, columns, rows)]`` of the chosen pair: what was compared, then one table per parameter."""
    from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import (
        PARAMETERS, Unavailable, matched_items, read, unit_factor, unit_power, unit_text)

    try:
        before, after, view = pair_snapshots(state, view)
    except SnapshotUnavailable:
        return []
    hierarchy = active_hierarchy(state)
    t0, t1 = before.title, after.title
    lost = sorted(set(int(x) for x in before.labels) - set(int(x) for x in after.labels))
    sets_b, sets_a = set(before.lines["per_set"]), set(after.lines["per_set"])
    include = bool(state.config.get("COMPARE_INCLUDE_RVE_FACES", False))
    facts = [["Before", t0], ["After", t1], ["Hierarchy level", hierarchy.level],
             [f"Grains in the {t0.lower()}", len(before.labels)], [f"Grains in the {t1.lower()}", len(after.labels)],
             ["Grains only in the first", len(lost) or None],
             ["Junction sets on both sides", len(sets_b & sets_a)],
             [f"Junction sets only in the {t0.lower()}", len(sets_b - sets_a)],
             [f"Junction sets only in the {t1.lower()}", len(sets_a - sets_b)],
             ["RVE-face traces", "included" if include else "excluded"],
             ["Length units shown", view.units if view.units != "auto" else f"chosen for each table (computed in {before.units})"]]
    tables = [("Comparison", ["Item", "Value"], [r for r in facts if r[1] is not None])]
    skipped = []
    for p in PARAMETERS:
        if p.key == "pole_figures":
            continue
        try:
            sb = read(p, before, normalisation=view.normalisation, grain_weight=view.grain_weight)
            sa = read(p, after, normalisation=view.normalisation, grain_weight=view.grain_weight)
        except Unavailable as problem:
            skipped.append(f"{p.label}: {problem}")
            continue
        power = unit_power(p.dimension)
        units = display_units(view, power, before.units, [sb.values, sa.values])
        factor = unit_factor(power, before.units, units)
        suffix = "deg" if p.dimension == "angle" else unit_text(power, units, plain=True)
        if p.key == "misorientation_boundary":
            if sb.weights is None or not len(sb.values):
                continue
            from upxo.meshing.gbconformant.fm_steel_abaqus.plots import weighted_stats

            area_b, area_a = before.pair_area[np.isfinite(before.pair_angle)], after.pair_area[np.isfinite(after.pair_angle)]
            stats_b = weighted_stats(sb.values, area_b)
            stats_a = weighted_stats(sa.values, area_a)
            tables.append((f"{p.label}: area-weighted (deg)", ["Statistic", t0, t1], [
                ["Boundaries", len(sb.values), len(sa.values)],
                ["Mean", stats_b["mean"], stats_a["mean"]], ["Standard deviation", stats_b["std"], stats_a["std"]],
                ["Median", stats_b["median"], stats_a["median"]],
                ["Share of the boundary area below 15 deg",
                 f"{100.0 * area_b[sb.values < 15].sum() / area_b.sum():.3g} %",
                 f"{100.0 * area_a[sa.values < 15].sum() / area_a.sum():.3g} %"]]))
            continue
        db, da = _describe(sb.values * factor), _describe(sa.values * factor)
        rows = [[label, db[key], da[key]] for key, label in STAT_ROWS]
        if p.dimension in ("length", "area", "volume") and p.noun == "lines":
            rows.insert(1, ["Total", db["total"], da["total"]])
        title = p.label + (f" ({suffix})" if suffix else "")
        tables.append((title, ["Statistic", t0, t1], rows))
        if "matched" in p.views and p.noun == "grains":
            keys, vb, va = matched_items(sb, sa)
            if len(keys) and p.key in ("diameter", "grain_volume"):
                ratio = _describe(va[vb > 0] / vb[vb > 0])
                tables.append((f"{p.label}: {t1} / {t0} for the {len(keys)} grains in both", ["Statistic", "Ratio"],
                               [["Median", ratio["median"]], ["Smallest", ratio["min"]], ["Largest", ratio["max"]]]))
    if skipped:
        tables.append(("Not available for this pair", ["Parameter and reason"], [[s] for s in skipped]))
    return tables


def comparison_text(state: PipelineState, view: Optional[ViewSettings] = None) -> str:
    """The tables of the chosen pair as one block of text."""
    from upxo.meshing.gbconformant.pipeline.results import Block

    tables = comparison_tables(state, view)
    return "\n\n".join(Block(*table).text() for table in tables)
