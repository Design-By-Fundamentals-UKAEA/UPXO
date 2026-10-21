"""Per-grain and per-run quality of a tetrahedral mesh: dihedral angles, minSICN and aspect ratio.

The aspect ratio of a tetrahedron is its circumradius over three times its inradius: 1 for the
regular tetrahedron and growing without limit for slivers and needles.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles

DIHEDRAL_BELOW = (15.0, 20.0, 30.0)
DIHEDRAL_ABOVE = (150.0, 165.0)
SICN_BELOW = (0.1, 0.2, 0.3)
_CHUNK = 400_000


def tet_volumes_and_aspect(points, tetrahedra, chunk: int = _CHUNK):
    """Volume and aspect ratio (circumradius / (3 inradius)) of every tetrahedron."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    volume = np.empty(len(t))
    aspect = np.empty(len(t))
    for start in range(0, len(t), chunk):
        v = p[t[start:start + chunk]]
        a, b, c = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]
        bc, ca, ab = np.cross(b, c), np.cross(c, a), np.cross(a, b)
        vol = np.abs(np.einsum("ij,ij->i", a, bc)) / 6.0
        faces = (np.linalg.norm(ab, axis=1) + np.linalg.norm(ca, axis=1) + np.linalg.norm(bc, axis=1)
                 + np.linalg.norm(np.cross(b - a, c - a), axis=1)) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            inradius = 3.0 * vol / faces
            circum = np.linalg.norm(
                np.einsum("ij,ij->i", a, a)[:, None] * bc + np.einsum("ij,ij->i", b, b)[:, None] * ca
                + np.einsum("ij,ij->i", c, c)[:, None] * ab, axis=1) / (12.0 * np.abs(
                    np.einsum("ij,ij->i", a, bc)) / 6.0)
            ratio = circum / (3.0 * inradius)
        ratio[~np.isfinite(ratio)] = np.inf
        volume[start:start + chunk] = vol
        aspect[start:start + chunk] = ratio
    return volume, aspect


def _describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"count": 0}
    q = np.percentile(v, [1, 5, 50, 95, 99])
    return {"count": int(v.size), "min": float(v.min()), "p1": float(q[0]), "p5": float(q[1]),
            "median": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(v.max()),
            "mean": float(v.mean())}


def tet_measures(tets) -> dict[str, np.ndarray]:
    """Per-tetrahedron arrays: smallest and largest dihedral angle, minSICN (when held), volume, aspect ratio."""
    angles = dihedral_angles(tets.points, tets.tetrahedra)
    volume, aspect = tet_volumes_and_aspect(tets.points, tets.tetrahedra)
    quality = getattr(tets, "quality", None)
    return {
        "smallest_dihedral": angles.min(axis=1),
        "largest_dihedral": angles.max(axis=1),
        "sicn": None if quality is None else np.asarray(quality, dtype=float),
        "volume": volume,
        "aspect": aspect,
    }


def grain_table(tets, measures: Optional[dict] = None, *, target_dihedral: float = 30.0) -> dict[str, np.ndarray]:
    """Quality of each grain's tetrahedra: one array per column, one entry per grain (ascending id)."""
    measures = measures or tet_measures(tets)
    grain = np.asarray(tets.grain_ids)
    order = np.argsort(grain, kind="stable")
    sorted_grain = grain[order]
    starts = np.flatnonzero(np.r_[True, sorted_grain[1:] != sorted_grain[:-1]])
    counts = np.diff(np.r_[starts, len(sorted_grain)])

    def reduce(values, ufunc):
        return ufunc.reduceat(np.asarray(values)[order], starts)

    out = {
        "grain": sorted_grain[starts],
        "tets": counts,
        "volume": reduce(measures["volume"], np.add),
        "min_dihedral": reduce(measures["smallest_dihedral"], np.minimum),
        "max_dihedral": reduce(measures["largest_dihedral"], np.maximum),
        "below_target": reduce(measures["smallest_dihedral"] < target_dihedral, np.add),
        "max_aspect": reduce(measures["aspect"], np.maximum),
    }
    out["below_target_fraction"] = out["below_target"] / out["tets"]
    if measures["sicn"] is not None:
        out["min_sicn"] = reduce(measures["sicn"], np.minimum)
        out["mean_sicn"] = reduce(measures["sicn"], np.add) / counts
    return out


def quality_report(tets, *, target_dihedral: float = 30.0, top: int = 10) -> dict[str, Any]:
    """The run's quality against the usual limits, and per grain with the worst ones named.

    ``worst`` lists the ``top`` grains with the smallest dihedral angle, worst first.
    """
    measures = tet_measures(tets)
    grains = grain_table(tets, measures, target_dihedral=target_dihedral)
    small, large, sicn = measures["smallest_dihedral"], measures["largest_dihedral"], measures["sicn"]
    n = int(len(small))
    run: dict[str, Any] = {
        "tetrahedra": n,
        "grains": int(len(grains["grain"])),
        "target_dihedral": float(target_dihedral),
        "smallest_dihedral": _describe(small),
        "largest_dihedral": _describe(large),
        "aspect_ratio": _describe(measures["aspect"]),
        "degenerate": int(np.sum(~np.isfinite(measures["aspect"]))),
        "dihedral_below": {f"{t:g}": int(np.sum(small < t)) for t in sorted({*DIHEDRAL_BELOW, target_dihedral})},
        "dihedral_above": {f"{t:g}": int(np.sum(large > t)) for t in DIHEDRAL_ABOVE},
    }
    if sicn is not None:
        run["sicn"] = _describe(sicn)
        run["sicn_below"] = {f"{t:g}": int(np.sum(sicn < t)) for t in SICN_BELOW}
    run["fraction_below_target"] = run["dihedral_below"][f"{target_dihedral:g}"] / n if n else 0.0
    order = np.argsort(grains["min_dihedral"], kind="stable")[:max(int(top), 0)]
    return {"run": run, "grains": grains, "worst": [int(grains["grain"][i]) for i in order]}
