"""Quality of the tetrahedra and surface triangles at chosen places of a grain-conformal mesh.

The places are the grain boundary surface, the junction lines and the junction points
(:mod:`.mesh_regions`), each seen from the elements touching it (layer 1) out to a chosen number of layers.
For the grain boundary surface the elements that touch a junction line, or a junction point, can be left out so
that what is left shows the boundary sheet itself.

Everything here works on plain arrays. :func:`region_quality` returns the result that is kept with a run (tables
of statistics and histograms, and the layer of every element), and the per-element arrays that only the files of
:func:`write_region_files` need.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.mesh_regions import (
    FEATURE_TITLES, FEATURES, GB_SURFACE, JUNCTION_LINES, JUNCTION_POINTS, MAX_ORDER, TOUCH_CHOICES, FACE_OR_EDGE,
    feature_layers, select_layers)
from upxo.meshing.gbconformant.pipeline.quality_report import _describe, tet_measures
from upxo.meshing.gbconformant.pipeline.runfiles import cfg_float, cfg_int

MODE_UP_TO, MODE_ONLY = "layers 1 to N", "only layer N"
MODES = (MODE_UP_TO, MODE_ONLY)
BINS = 60
WHOLE, BEYOND = "all", "beyond"
REGION_NAMES = {WHOLE: "Whole mesh", BEYOND: "Beyond the selected layers", **FEATURE_TITLES}
FEATURE_FLAGS = {GB_SURFACE: "REGION_GB_SURFACE", JUNCTION_LINES: "REGION_JUNCTION_LINES",
                 JUNCTION_POINTS: "REGION_JUNCTION_POINTS"}
# key, label, (lowest, highest) of the histogram; a value outside the range is counted in the end bin
TET_MEASURES = (("smallest_dihedral", "Smallest dihedral (deg)", (0.0, 71.0)),
                ("largest_dihedral", "Largest dihedral (deg)", (70.0, 180.0)),
                ("sicn", "minSICN", (0.0, 1.0)),
                ("aspect", "Aspect ratio", (1.0, 10.0)))
TRIANGLE_MEASURES = (("smallest_angle", "Smallest angle (deg)", (0.0, 60.0)),
                     ("largest_angle", "Largest angle (deg)", (60.0, 180.0)),
                     ("shape_quality", "Shape quality", (0.0, 1.0)))
MEASURES = {"tets": TET_MEASURES, "triangles": TRIANGLE_MEASURES}
KIND_TITLES = {"tets": "Tetrahedra", "triangles": "Surface triangles"}


@dataclass(frozen=True)
class RegionSettings:
    """What to look at: which features, how many layers, how to count touching, what to leave out, the limits."""

    order: int = 3
    mode: str = MODE_UP_TO
    touch: str = FACE_OR_EDGE
    include_rve_faces: bool = False
    features: tuple = FEATURES
    exclude_lines: bool = False
    exclude_points: bool = False
    exclude_order: int = 1
    min_dihedral: float = 15.0
    target_dihedral: float = 30.0
    max_dihedral: float = 150.0
    min_angle: float = 30.0


def region_settings(cfg: dict) -> RegionSettings:
    """The region-quality settings of a run's configuration (checked)."""
    features = tuple(f for f in FEATURES if bool(cfg.get(FEATURE_FLAGS[f], True)))
    if not features:
        raise ValueError("Switch on at least one of the grain boundary surface, the junction lines and the junction points.")
    mode = str(cfg.get("REGION_ORDER_MODE", MODE_UP_TO))
    touch = str(cfg.get("REGION_TOUCH", FACE_OR_EDGE))
    if mode not in MODES:
        raise ValueError(f"REGION_ORDER_MODE must be one of {', '.join(MODES)}, not {mode!r}.")
    if touch not in TOUCH_CHOICES:
        raise ValueError(f"REGION_TOUCH must be one of {', '.join(TOUCH_CHOICES)}, not {touch!r}.")
    return RegionSettings(
        order=cfg_int(cfg, "REGION_ORDER", 3), mode=mode, touch=touch,
        include_rve_faces=bool(cfg.get("REGION_INCLUDE_RVE_FACES", False)), features=features,
        exclude_lines=bool(cfg.get("REGION_EXCLUDE_LINES", False)),
        exclude_points=bool(cfg.get("REGION_EXCLUDE_POINTS", False)),
        exclude_order=cfg_int(cfg, "REGION_EXCLUDE_ORDER", 1),
        min_dihedral=cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0),
        target_dihedral=cfg_float(cfg, "TET_TARGET_MIN_DIHEDRAL", 30.0),
        max_dihedral=cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0),
        min_angle=cfg_float(cfg, "SURFACE_MIN_ANGLE_DEGREES", 30.0))


def triangle_measures(points, triangles) -> dict:
    """Smallest angle, largest angle (degrees) and shape quality of every triangle."""
    from upxo.meshing.gbconformant.d3v2p0.surface_angles import triangle_angles
    from upxo.meshing.gbconformant.d3v2p0.surface_quality import triangle_quality

    p = np.asarray(points, dtype=float)
    t = np.asarray(triangles)
    if not len(t):
        empty = np.zeros(0)
        return {"smallest_angle": empty, "largest_angle": empty, "shape_quality": empty}
    angles = triangle_angles(p, t)
    return {"smallest_angle": angles.min(axis=1), "largest_angle": angles.max(axis=1),
            "shape_quality": np.asarray(triangle_quality(p, t), dtype=float)}


def _histogram(values, value_range):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return np.histogram(np.clip(v, *value_range), bins=BINS, range=value_range)[0].astype(np.int64).tolist()


def _below(kind, values, mask, settings) -> dict:
    if kind == "tets":
        small, large = values["smallest_dihedral"][mask], values["largest_dihedral"][mask]
        return {"min_dihedral": int(np.sum(small < settings.min_dihedral)),
                "target_dihedral": int(np.sum(small < settings.target_dihedral)),
                "max_dihedral": int(np.sum(large > settings.max_dihedral))}
    return {"min_angle": int(np.sum(values["smallest_angle"][mask] < settings.min_angle))}


def _row(kind, region, layer, text, mask, headline, values, total, settings, note="") -> dict:
    count = int(mask.sum())
    stats, hist = {}, {}
    for key, _, value_range in MEASURES[kind]:
        if values.get(key) is None:
            continue
        chosen = values[key][mask]
        stats[key] = _describe(chosen)
        hist[key] = _histogram(chosen, value_range)
    return {"kind": kind, "region": region, "layer": layer, "layers": text, "headline": bool(headline),
            "count": count, "share": count / total if total else 0.0, "stats": stats, "hist": hist,
            "below": _below(kind, values, mask, settings), "note": note}


def _exclusion(layers, settings, size) -> np.ndarray:
    """Elements to leave out of the grain boundary surface: within ``exclude_order`` layers of lines or points."""
    out = np.zeros(size, dtype=bool)
    for wanted, feature in ((settings.exclude_lines, JUNCTION_LINES), (settings.exclude_points, JUNCTION_POINTS)):
        if wanted and feature in layers:
            out |= select_layers(layers[feature], settings.exclude_order, False)
    return out


def _exclusion_note(settings) -> str:
    left_out = [name for flag, name in ((settings.exclude_lines, "junction lines"),
                                        (settings.exclude_points, "junction points")) if flag]
    if not left_out:
        return ""
    return f"without the elements within {settings.exclude_order} layer(s) of the {' and '.join(left_out)}"


def region_quality(points, tetrahedra, grains, quality=None, settings: Optional[RegionSettings] = None):
    """Quality of the elements at the chosen features: ``(result, arrays)``.

    ``result`` is kept with the run: ``settings``, ``counts``, ``rows`` (one per region and layer, each with the
    statistics, the limits it breaks and a histogram of every measure), ``limits``, the layer of every element and
    the surface triangles they refer to. ``arrays`` holds the per-element measures and selections, for the files.
    """
    settings = settings or RegionSettings()
    if settings.mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {settings.mode!r}.")
    needs_exclusion = settings.exclude_lines or settings.exclude_points
    reach = max(settings.order, settings.exclude_order if needs_exclusion else 1)
    wanted = set(settings.features)
    if needs_exclusion:
        wanted |= {JUNCTION_LINES} if settings.exclude_lines else set()
        wanted |= {JUNCTION_POINTS} if settings.exclude_points else set()
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64)
    layers = feature_layers(p, t, grains, order=reach, touch=settings.touch,
                            include_rve_faces=settings.include_rve_faces, features=tuple(f for f in FEATURES if f in wanted))
    triangles = layers.surface.nodes
    values = {"tets": tet_measures(SimpleNamespace(points=p, tetrahedra=t, quality=quality)),
              "triangles": triangle_measures(p, triangles)}
    values["tets"] = {k: v for k, v in values["tets"].items() if k != "volume"}
    sizes = {"tets": len(t), "triangles": len(triangles)}
    only = settings.mode == MODE_ONLY
    note = _exclusion_note(settings)

    rows, headline_masks, excluded = [], {}, {}
    for kind in ("tets", "triangles"):
        size = sizes[kind]
        kind_layers = layers.tet if kind == "tets" else layers.triangle
        excluded[kind] = _exclusion(kind_layers, settings, size)
        covered = np.zeros(size, dtype=bool)
        rows.append(_row(kind, WHOLE, None, "all", np.ones(size, dtype=bool), False, values[kind], size, settings))
        for feature in settings.features:
            layer = kind_layers[feature]
            skip = excluded[kind] if feature == GB_SURFACE else np.zeros(size, dtype=bool)
            feature_note = note if feature == GB_SURFACE else ""
            if kind == "triangles" and feature == GB_SURFACE:
                # every triangle of the surface lies on it, whatever the order: one row, always plotted
                mask = (layer == 1) & ~skip
                rows.append(_row(kind, feature, 1, "1", mask, True, values[kind], size, settings, feature_note))
                covered |= mask
                headline_masks.setdefault(kind, {})[feature] = mask
                continue
            for k in range(1, settings.order + 1):
                mask = (layer == k) & ~skip
                is_headline = (k == settings.order) if only else (settings.order == 1)
                if mask.any() or is_headline or k == 1:
                    rows.append(_row(kind, feature, k, str(k), mask, is_headline, values[kind], size, settings, feature_note))
                if is_headline:
                    covered |= mask
            if settings.order > 1:
                mask = select_layers(layer, settings.order, False) & ~skip
                rows.append(_row(kind, feature, None, f"1 to {settings.order}", mask, not only, values[kind], size,
                                 settings, feature_note))
                if not only:
                    covered |= mask
            headline_masks.setdefault(kind, {})[feature] = select_layers(layer, settings.order, only) & ~skip
        if (~covered).any():
            rows.append(_row(kind, BEYOND, None, "beyond", ~covered, False, values[kind], size, settings))

    result: dict[str, Any] = {
        "settings": asdict(settings),
        "counts": {**layers.counts, "excluded_tetrahedra": int(excluded["tets"].sum()),
                   "excluded_triangles": int(excluded["triangles"].sum())},
        "rows": rows,
        "limits": {"min_dihedral": settings.min_dihedral, "target_dihedral": settings.target_dihedral,
                   "max_dihedral": settings.max_dihedral, "min_angle": settings.min_angle},
        "histogram": {"bins": BINS, "ranges": {key: list(rng) for kind in MEASURES for key, _, rng in MEASURES[kind]}},
        "layers": {"tets": {f: v.astype(np.int8) for f, v in layers.tet.items()},
                   "triangles": {f: v.astype(np.int8) for f, v in layers.triangle.items()}},
        "triangles": {"nodes": triangles.astype(np.int64), "grains": layers.surface.grains.astype(np.int64),
                      "between_grains": layers.surface.between_grains},
    }
    arrays = {"values": values, "selected": headline_masks, "excluded": excluded}
    return result, arrays


# --- tables ---------------------------------------------------------------------------------------------
def _number(stats: dict, key: str):
    return stats.get(key) if stats.get("count") else None


def _percent(count, total):
    return f"{count} ({100.0 * count / total:.3g} %)" if total else "0"


def region_label(row: dict) -> str:
    return REGION_NAMES.get(row["region"], str(row["region"]))


def _plotted(row: dict):
    return "yes" if row["headline"] or row["region"] == WHOLE else None


def region_tables(result: dict) -> list[tuple]:
    """``[(title, columns, rows)]`` for the tetrahedra and for the surface triangles."""
    limits, rows = result["limits"], result["rows"]
    tables = []
    for kind in ("tets", "triangles"):
        mine = [r for r in rows if r["kind"] == kind]
        if not mine:
            continue
        total = mine[0]["count"]
        if kind == "tets":
            columns = ["Region", "Layers", "Elements", "Share", "Smallest dihedral: lowest (deg)", "5th percentile",
                       "Median", f"Below {limits['min_dihedral']:g} deg", f"Below {limits['target_dihedral']:g} deg",
                       "Largest dihedral: highest (deg)", f"Above {limits['max_dihedral']:g} deg", "Lowest minSICN",
                       "Largest aspect ratio", "Plotted"]
        else:
            columns = ["Region", "Layers", "Elements", "Share", "Smallest angle: lowest (deg)", "5th percentile",
                       "Median", f"Below {limits['min_angle']:g} deg", "Largest angle: highest (deg)",
                       "Lowest shape quality", "Plotted"]
        body = []
        notes = list(dict.fromkeys(r["note"] for r in mine if r.get("note")))
        for r in mine:
            s, b = r["stats"], r["below"]
            share = f"{100.0 * r['share']:.3g} %"
            name = region_label(r)
            if kind == "tets":
                small, large = s["smallest_dihedral"], s["largest_dihedral"]
                sicn, aspect = s.get("sicn", {}), s.get("aspect", {})
                body.append([name, r["layers"], r["count"], share, _number(small, "min"), _number(small, "p5"),
                             _number(small, "median"), _percent(b["min_dihedral"], r["count"]),
                             _percent(b["target_dihedral"], r["count"]), _number(large, "max"),
                             _percent(b["max_dihedral"], r["count"]), _number(sicn, "min"), _number(aspect, "max"),
                             _plotted(r)])
            else:
                small, large, shape = s["smallest_angle"], s["largest_angle"], s["shape_quality"]
                body.append([name, r["layers"], r["count"], share, _number(small, "min"), _number(small, "p5"),
                             _number(small, "median"), _percent(b["min_angle"], r["count"]), _number(large, "max"),
                             _number(shape, "min"), _plotted(r)])
        title = f"{KIND_TITLES[kind]} by region: {total} in the mesh"
        tables.append((title + "".join(f"; the grain boundary surface is {n}" for n in notes), columns, body))
    return tables


def region_flat_table(result: dict) -> tuple:
    """One row per region and layer with every statistic, for a spreadsheet: ``(title, columns, rows)``."""
    columns = ["element", "region", "layers", "plotted", "elements", "share"]
    measures = list(dict.fromkeys(key for r in result["rows"] for key in r["stats"]))
    spread = ("min", "p5", "median", "p95", "max", "mean")
    for key in measures:
        columns += [f"{key}_{name}" for name in spread]
    columns += ["below_min_dihedral", "below_target_dihedral", "above_max_dihedral", "below_min_angle"]
    body = []
    for r in result["rows"]:
        row = [r["kind"], region_label(r), r["layers"], "yes" if r["headline"] else "", r["count"], r["share"]]
        for key in measures:
            stats = r["stats"].get(key, {})
            row += [stats.get(name) if stats.get("count") else None for name in spread]
        b = r["below"]
        row += [b.get("min_dihedral"), b.get("target_dihedral"), b.get("max_dihedral"), b.get("min_angle")]
        body.append(row)
    return "Quality by region and layer", columns, body


def summary_numbers(result: dict) -> dict:
    """The few numbers a stage reports: what was found and the lowest smallest-dihedral of each plotted region."""
    counts = result["counts"]
    out = {"tetrahedra": counts["tetrahedra"], "triangles": counts["triangles"],
           "junction_lines": counts["junction_lines"], "junction_points": counts["junction_points"],
           "order": result["settings"]["order"], "mode": result["settings"]["mode"]}
    lowest = {}
    for r in result["rows"]:
        if r["kind"] == "tets" and r["headline"] and r["stats"]["smallest_dihedral"].get("count"):
            lowest[r["region"]] = r["stats"]["smallest_dihedral"]["min"]
    out["lowest_dihedral"] = lowest
    return out


# --- figures --------------------------------------------------------------------------------------------
_COLOURS = {WHOLE: "0.45", GB_SURFACE: "#4878CF", JUNCTION_LINES: "#D65F5F", JUNCTION_POINTS: "#6ACC65"}


def _limit_line(result, kind, key):
    limits = result["limits"]
    return {("tets", "smallest_dihedral"): limits["min_dihedral"], ("tets", "largest_dihedral"): limits["max_dihedral"],
            ("triangles", "smallest_angle"): limits["min_angle"]}.get((kind, key))


def region_histogram_figure(result: dict, kind: str, key: str, *, figsize=(6.4, 3.4)):
    """One measure of the elements of every plotted region against the whole mesh, as shares per bin; ``None`` without data."""
    from matplotlib.figure import Figure

    label, value_range = next((lab, rng) for k, lab, rng in MEASURES[kind] if k == key)
    plotted = [r for r in result["rows"] if r["kind"] == kind and key in r["hist"] and r["count"]
               and (r["headline"] or r["region"] == WHOLE)]
    if not any(r["region"] != WHOLE for r in plotted):
        return None
    edges = np.linspace(value_range[0], value_range[1], BINS + 1)
    figure = Figure(figsize=figsize, tight_layout=True)
    axes = figure.add_subplot(111)
    for r in plotted:
        share = np.asarray(r["hist"][key], dtype=float) / r["count"]
        share[share == 0] = np.nan
        text = f"{region_label(r)}, layers {r['layers']} ({r['count']})" if r["region"] != WHOLE else f"Whole mesh ({r['count']})"
        axes.stairs(share, edges, color=_COLOURS.get(r["region"]), lw=2.0 if r["region"] == WHOLE else 1.4,
                    alpha=0.9, label=text)
    limit = _limit_line(result, kind, key)
    if limit is not None:
        axes.axvline(limit, color="k", ls="--", lw=1, label=f"limit {limit:g}")
    axes.set_yscale("log")
    axes.set_xlabel(label)
    axes.set_ylabel("share of the region's elements per bin")
    axes.set_title(f"{KIND_TITLES[kind]} by region: {label.lower()}", fontweight="bold", fontsize=10)
    axes.legend(fontsize=7)
    axes.tick_params(labelsize=8)
    return figure


def region_layer_figure(result: dict, kind: str, *, figsize=(6.4, 3.4)):
    """The median and the 5th percentile of the smallest angle against the layer, per feature; ``None`` without data."""
    from matplotlib.figure import Figure

    key = "smallest_dihedral" if kind == "tets" else "smallest_angle"
    mine = [r for r in result["rows"] if r["kind"] == kind]
    whole = next((r for r in mine if r["region"] == WHOLE), None)
    series = {}
    for r in mine:
        if r["layer"] is not None and r["count"] and r["stats"][key].get("count"):
            series.setdefault(r["region"], []).append((r["layer"], r["stats"][key]["median"], r["stats"][key]["p5"]))
    series = {f: v for f, v in series.items() if len(v) >= 2}
    if not series or whole is None:
        return None
    figure = Figure(figsize=figsize, tight_layout=True)
    axes = figure.add_subplot(111)
    for feature, points in series.items():
        layer, median, p5 = zip(*sorted(points))
        axes.plot(layer, median, "o-", color=_COLOURS[feature], label=f"{FEATURE_TITLES[feature]}: median")
        axes.plot(layer, p5, "s--", color=_COLOURS[feature], label=f"{FEATURE_TITLES[feature]}: 5th percentile")
    axes.axhline(whole["stats"][key]["median"], color="0.45", lw=1.2, label="Whole mesh: median")
    axes.axhline(whole["stats"][key]["p5"], color="0.45", lw=1.2, ls="--", label="Whole mesh: 5th percentile")
    axes.set_xlabel("layers from the feature")
    axes.set_ylabel("smallest dihedral (deg)" if kind == "tets" else "smallest angle (deg)")
    axes.set_xticks(sorted({x for pts in series.values() for x, _, _ in pts}))
    axes.set_title(f"{KIND_TITLES[kind]}: quality against distance from the feature", fontweight="bold", fontsize=10)
    axes.legend(fontsize=7)
    axes.tick_params(labelsize=8)
    return figure


def region_figures(result: dict, *, figsize=(6.4, 3.4)) -> list[tuple]:
    """``[(key, title, figure)]`` of every figure the result supports."""
    out = []
    for kind in ("tets", "triangles"):
        for key, label, _ in MEASURES[kind]:
            figure = region_histogram_figure(result, kind, key, figsize=figsize)
            if figure is not None:
                out.append((f"regions.{kind}.{key}", f"{KIND_TITLES[kind]} by region: {label.lower()}", figure))
        figure = region_layer_figure(result, kind, figsize=figsize)
        if figure is not None:
            out.append((f"regions.{kind}.layers", f"{KIND_TITLES[kind]}: quality against distance", figure))
    return out


# --- files ----------------------------------------------------------------------------------------------
def write_region_files(folder, points, tetrahedra, grains, result: dict, arrays: dict) -> list[str]:
    """Write the per-region table (CSV), the tetrahedra (VTU) and the surface triangles (VTP) with their layers and measures."""
    import csv

    import pyvista as pv

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    _, columns, rows = region_flat_table(result)
    csv_path = folder / "region_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([["" if v is None else v for v in row] for row in rows])
    written.append(str(csv_path))

    features = result["settings"]["features"]
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64)
    cap = 1e6
    cell = {"grain": np.asarray(grains, dtype=np.int64)}
    for key, values in arrays["values"]["tets"].items():
        if values is not None:
            cell[key] = np.minimum(np.nan_to_num(np.asarray(values, dtype=float), posinf=cap), cap).astype(np.float32)
    for feature in features:
        cell[f"layer_{feature}"] = result["layers"]["tets"][feature]
        cell[f"selected_{feature}"] = arrays["selected"]["tets"][feature].astype(np.uint8)
    grid = pv.UnstructuredGrid(np.column_stack((np.full(len(t), 4, dtype=np.int64), t)).ravel(),
                               np.full(len(t), 10, dtype=np.uint8), p)
    for name, values in cell.items():
        grid.cell_data[name] = values
    path = folder / "region_tets.vtu"
    grid.save(str(path))
    written.append(str(path))

    tri = result["triangles"]["nodes"]
    if len(tri):
        mesh = pv.PolyData(p, np.column_stack((np.full(len(tri), 3, dtype=np.int64), tri)).ravel())
        mesh.cell_data["grain_low"] = result["triangles"]["grains"][:, 0]
        mesh.cell_data["grain_high"] = result["triangles"]["grains"][:, 1]
        mesh.cell_data["between_grains"] = result["triangles"]["between_grains"].astype(np.uint8)
        for key, values in arrays["values"]["triangles"].items():
            mesh.cell_data[key] = np.asarray(values, dtype=np.float32)
        for feature in features:
            mesh.cell_data[f"layer_{feature}"] = result["layers"]["triangles"][feature]
            mesh.cell_data[f"selected_{feature}"] = arrays["selected"]["triangles"][feature].astype(np.uint8)
        path = folder / "region_triangles.vtp"
        mesh.save(str(path))
        written.append(str(path))
    return written


__all__ = ["BEYOND", "MAX_ORDER", "MEASURES", "MODES", "MODE_ONLY", "MODE_UP_TO", "RegionSettings", "WHOLE",
           "region_figures", "region_flat_table", "region_quality", "region_settings", "region_tables",
           "summary_numbers", "triangle_measures", "write_region_files"]
