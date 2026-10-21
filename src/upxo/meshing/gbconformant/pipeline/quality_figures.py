"""Element quality distributions: surface triangle angles and shape quality, tetrahedron dihedral angles and minSICN.

Histograms with the worst band marked (``QUALITY_WORST_FRACTION`` of the elements), built from the
repaired surface and the verified tetrahedra a run holds.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.runfiles import cfg_float
from upxo.meshing.gbconformant.pipeline.state import PipelineState


def _worst_band(values, fraction, best, worst="low"):
    values = np.asarray(values, float)
    if worst == "low":
        extreme = float(values.min())
        edge = extreme + fraction * (best - extreme)
        return edge, int(np.sum(values <= edge)), extreme
    extreme = float(values.max())
    edge = extreme - fraction * (extreme - best)
    return edge, int(np.sum(values >= edge)), extreme


def _band_stats(values, fraction, best, worst="low") -> dict[str, Any]:
    edge, count, extreme = _worst_band(values, fraction, best, worst)
    lo, hi = (extreme, edge) if worst == "low" else (edge, extreme)
    return {
        "worst": float(extreme),
        "best_possible": float(best),
        "fraction": float(fraction),
        "band": [float(lo), float(hi)],
        "count": int(count),
    }


def surface_for_distributions(state: PipelineState):
    """Prefer repaired → remeshed → closed surface artefacts."""
    return (
        state.get_artefact("repaired_rve")
        or state.get_artefact("remeshed_rve")
        or state.get_artefact("closed_rve")
    )


def surface_distribution_arrays(state: PipelineState) -> tuple[Optional[dict], dict[str, Any]]:
    """Return ``({arrays...}, meta)`` or ``(None, unavailable stats)``."""
    surface = surface_for_distributions(state)
    if surface is None:
        return None, {
            "available": False,
            "reason": "No repaired/remeshed surface artefact yet",
        }
    points = getattr(surface, "points", None)
    triangles = getattr(surface, "triangles", None)
    if points is None or triangles is None:
        return None, {
            "available": False,
            "reason": "Surface artefact missing points/triangles",
        }
    cfg = state.config
    min_angle = cfg_float(cfg, "SURFACE_MIN_ANGLE_DEGREES", 30.0)
    fraction = cfg_float(cfg, "QUALITY_WORST_FRACTION", 0.05)
    if not 0 < fraction <= 1:
        return None, {"available": False, "reason": "QUALITY_WORST_FRACTION must lie in (0, 1]"}
    try:
        from upxo.meshing.gbconformant.d3v2p0.surface_angles import triangle_angles
        from upxo.meshing.gbconformant.d3v2p0.surface_quality import triangle_quality
        angles = triangle_angles(np.asarray(points, float), np.asarray(triangles))
        smallest = angles.min(axis=1)
        largest = angles.max(axis=1)
        quality = triangle_quality(np.asarray(points, float), np.asarray(triangles))
    except Exception as exc:
        return None, {"available": False, "reason": str(exc)}
    exterior = getattr(surface, "exterior", None)
    groups = None
    if exterior is not None:
        ext = np.asarray(exterior, bool)
        groups = [("grain boundaries", ~ext), ("RVE faces", ext)]
    n = int(len(triangles))
    stats = {
        "available": True,
        "triangles": n,
        "below_min_angle_limit": int(np.sum(smallest < min_angle)),
        "min_angle_limit": float(min_angle),
        "fraction": float(fraction),
        "smallest_angle": _band_stats(smallest, fraction, 60.0, "low"),
        "largest_angle": _band_stats(largest, fraction, 60.0, "high"),
        "shape_quality": _band_stats(quality, fraction, 1.0, "low"),
    }
    arrays = {
        "smallest_angle": smallest,
        "largest_angle": largest,
        "shape_quality": quality,
        "groups": groups,
        "min_angle_limit": min_angle,
        "fraction": fraction,
        "n": n,
    }
    return arrays, stats


def tet_distribution_arrays(state: PipelineState) -> tuple[Optional[dict], dict[str, Any]]:
    """Return ``({arrays...}, meta)`` or ``(None, unavailable stats)``."""
    tets = state.get_artefact("verified_tets")
    if tets is None:
        return None, {"available": False, "reason": "verified_tets missing"}
    points = getattr(tets, "points", None)
    tetrahedra = getattr(tets, "tetrahedra", None)
    if points is None or tetrahedra is None:
        return None, {
            "available": False,
            "reason": "verified_tets missing points/tetrahedra",
        }
    cfg = state.config
    min_dih = cfg_float(cfg, "MIN_TET_DIHEDRAL_DEGREES", 15.0)
    max_dih = cfg_float(cfg, "TET_MAX_DIHEDRAL", 150.0)
    fraction = cfg_float(cfg, "QUALITY_WORST_FRACTION", 0.05)
    if not 0 < fraction <= 1:
        return None, {"available": False, "reason": "QUALITY_WORST_FRACTION must lie in (0, 1]"}
    try:
        from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles
        a = dihedral_angles(points, tetrahedra)
        smallest = a.min(axis=1)
        largest = a.max(axis=1)
    except Exception as exc:
        return None, {"available": False, "reason": str(exc)}
    regular = float(np.degrees(np.arccos(1 / 3)))
    n = int(len(a))
    stats: dict[str, Any] = {
        "available": True,
        "tetrahedra": n,
        "below_min_limit": int(np.sum(smallest < min_dih)),
        "above_max_limit": int(np.sum(largest > max_dih)),
        "min_dihedral_limit": float(min_dih),
        "max_dihedral_limit": float(max_dih),
        "fraction": float(fraction),
        "smallest_dihedral": _band_stats(smallest, fraction, regular, "low"),
        "largest_dihedral": _band_stats(largest, fraction, regular, "high"),
    }
    quality = getattr(tets, "quality", None)
    q_arr = None
    if quality is not None:
        q_arr = np.asarray(quality, float)
        stats["minSICN"] = _band_stats(q_arr, fraction, 1.0, "low")
    arrays = {
        "smallest_dihedral": smallest,
        "largest_dihedral": largest,
        "minSICN": q_arr,
        "min_dihedral_limit": min_dih,
        "max_dihedral_limit": max_dih,
        "fraction": fraction,
        "regular": regular,
        "n": n,
    }
    return arrays, stats


def _draw_histogram(
    ax,
    values,
    *,
    title,
    xlabel,
    unit,
    value_range,
    fraction,
    best,
    worst,
    limit=None,
    limit_label=None,
    groups=None,
    bins=50,
    color=None,
    element="elements",
):
    if groups:
        for name, mask in groups:
            ax.hist(values[mask], bins=bins, range=value_range, alpha=0.6, label=name)
    else:
        ax.hist(values, bins=bins, range=value_range, color=color)
    edge, count, extreme = _worst_band(values, fraction, best, worst)
    lo, hi = (extreme, edge) if worst == "low" else (edge, extreme)
    ax.axvspan(
        lo, hi, color="tab:red", alpha=0.15,
        label=f"[{lo:.3g}, {hi:.3g}]{unit}: {count} {element}")
    ax.axvline(
        extreme, color="tab:red", lw=1.5,
        label=(
            f"{'lowest' if worst == 'low' else 'highest'}: "
            f"{extreme:.3g}{unit} (best possible {best:.4g}{unit})"
        ),
    )
    if limit is not None:
        ax.axvline(limit, color="k", ls="--", lw=1, label=limit_label)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(element)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)


def build_surface_quality_distributions(
    state: PipelineState,
    *,
    figsize: tuple[float, float] = (5.4, 2.55),
    bins: int = 50,
    make_figures: bool = True,
) -> tuple[list, dict[str, Any]]:
    """Return ``(figures, stats)``. Set ``make_figures=False`` for stats only."""
    arrays, stats = surface_distribution_arrays(state)
    if arrays is None or not make_figures:
        return [], stats
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib.figure import Figure
    except Exception as exc:
        stats = dict(stats)
        stats["available"] = False
        stats["reason"] = f"matplotlib unavailable: {exc}"
        return [], stats

    figures = []
    n = arrays["n"]
    fraction = arrays["fraction"]
    groups = arrays["groups"]
    specs = (
        (arrays["smallest_angle"], "Surface triangles: smallest angle",
         "smallest angle (deg)", " deg", (0, 60), 60.0, "low",
         arrays["min_angle_limit"],
         f"limit {arrays['min_angle_limit']:g} deg"),
        (arrays["largest_angle"], "Surface triangles: largest angle",
         "largest angle (deg)", " deg", (60, 180), 60.0, "high", None, None),
        (arrays["shape_quality"], "Surface triangles: shape quality",
         "shape quality (0-1)", "", (0, 1), 1.0, "low", None, None),
    )
    for values, title, xlabel, unit, rng, best, worst, limit, limit_label in specs:
        fig = Figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
        _draw_histogram(
            ax, values, title=f"{title} ({n} triangles)", xlabel=xlabel,
            unit=unit, value_range=rng, fraction=fraction, best=best,
            worst=worst, limit=limit, limit_label=limit_label, groups=groups,
            bins=bins, element="triangles")
        figures.append(fig)
    return figures, stats


def build_tet_quality_distributions(
    state: PipelineState,
    *,
    figsize: tuple[float, float] = (5.4, 2.55),
    bins: int = 50,
    make_figures: bool = True,
) -> tuple[list, dict[str, Any]]:
    """Return ``(figures, stats)``. Set ``make_figures=False`` for stats only."""
    arrays, stats = tet_distribution_arrays(state)
    if arrays is None or not make_figures:
        return [], stats
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib.figure import Figure
    except Exception as exc:
        stats = dict(stats)
        stats["available"] = False
        stats["reason"] = f"matplotlib unavailable: {exc}"
        return [], stats

    figures = []
    n = arrays["n"]
    fraction = arrays["fraction"]
    regular = arrays["regular"]
    specs = [
        (arrays["smallest_dihedral"], "Tetrahedra: smallest dihedral angle",
         "smallest dihedral (deg)", " deg", (0, 75), regular, "low",
         arrays["min_dihedral_limit"],
         f"limit {arrays['min_dihedral_limit']:g} deg", "#4878CF"),
        (arrays["largest_dihedral"], "Tetrahedra: largest dihedral angle",
         "largest dihedral (deg)", " deg", (70, 180), regular, "high",
         arrays["max_dihedral_limit"],
         f"limit {arrays['max_dihedral_limit']:g} deg", "#D65F5F"),
    ]
    if arrays["minSICN"] is not None:
        specs.append(
            (arrays["minSICN"], "Tetrahedra: Gmsh minSICN", "minSICN", "",
             (0, 1), 1.0, "low", None, None, "#6ACC65")
        )
    for values, title, xlabel, unit, rng, best, worst, limit, limit_label, color in specs:
        fig = Figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
        _draw_histogram(
            ax, values, title=f"{title} ({n} tets)", xlabel=xlabel, unit=unit,
            value_range=rng, fraction=fraction, best=best, worst=worst,
            limit=limit, limit_label=limit_label, bins=bins, color=color,
            element="tets")
        figures.append(fig)
    return figures, stats


def distribution_stats_text(stats: dict[str, Any], *, kind: str) -> str:
    """Compact monospace summary under the charts."""
    if not isinstance(stats, dict) or not stats.get("available"):
        reason = (stats or {}).get("reason") if isinstance(stats, dict) else None
        return reason or "Quality distributions unavailable."
    lines: list[str] = []
    if kind == "surface":
        n = stats.get("triangles")
        below = stats.get("below_min_angle_limit")
        lines.append(
            f"Surface triangles: {n}  |  below min-angle limit: {below}  |  "
            f"worst-band fraction={stats.get('fraction', '—')}"
        )
        for key, label in (
            ("smallest_angle", "min∠"),
            ("largest_angle", "max∠"),
            ("shape_quality", "shape q"),
        ):
            st = stats.get(key) or {}
            if not isinstance(st, dict):
                continue
            lines.append(
                f"  {label}: worst={st.get('worst', '—')}  "
                f"band={st.get('band', '—')}  count={st.get('count', '—')}"
            )
    else:
        n = stats.get("tetrahedra")
        lines.append(
            f"Tetrahedra: {n}  |  below min dihedral: {stats.get('below_min_limit')}  "
            f"|  above max dihedral: {stats.get('above_max_limit')}  |  "
            f"worst-band fraction={stats.get('fraction', '—')}"
        )
        for key, label in (
            ("smallest_dihedral", "min dih"),
            ("largest_dihedral", "max dih"),
            ("minSICN", "minSICN"),
        ):
            st = stats.get(key) or {}
            if not isinstance(st, dict):
                continue
            lines.append(
                f"  {label}: worst={st.get('worst', '—')}  "
                f"band={st.get('band', '—')}  count={st.get('count', '—')}"
            )
    return "\n".join(lines)
