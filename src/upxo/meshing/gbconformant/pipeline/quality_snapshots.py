"""Summaries of the tetrahedral mesh quality, and the change between two of them."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from upxo.meshing.gbconformant.pipeline.state import PipelineState

_SNAPSHOT_SCALAR_KEYS = (
    "min", "max", "mean", "p5", "p50", "p95", "below_0_05", "below_0_1",
)


def quality_histogram(state: PipelineState, *, bins: int = 20) -> dict[str, Any]:
    tets = state.get_artefact("verified_tets")
    if tets is None:
        return {"available": False, "reason": "verified_tets missing"}
    q = np.asarray(tets.quality, dtype=float)
    if not len(q):
        return {"available": False, "reason": "empty quality array"}
    counts, edges = np.histogram(q, bins=bins, range=(0.0, 1.0))
    return {
        "available": True,
        "bins": bins,
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "min": float(q.min()),
        "max": float(q.max()),
        "mean": float(q.mean()),
        "p5": float(np.percentile(q, 5)),
        "p50": float(np.percentile(q, 50)),
        "p95": float(np.percentile(q, 95)),
        "below_0_05": int(np.count_nonzero(q < 0.05)),
        "below_0_1": int(np.count_nonzero(q < 0.1)),
        "n": int(len(q)),
    }


def dihedral_summary(state: PipelineState) -> dict[str, Any]:
    tets = state.get_artefact("verified_tets")
    if tets is None:
        return {"available": False}
    return {
        "available": True,
        "dihedral_angles": (tets.report or {}).get("dihedral_angles"),
        "ready": (tets.report or {}).get("ready"),
        "status": (tets.report or {}).get("status"),
        "minimum_quality": (tets.report or {}).get("minimum_quality"),
    }


def tet_quality_snapshot(state: PipelineState, *, bins: int = 20) -> dict[str, Any]:
    """Histogram + bounds + dihedral callouts for a before/after comparison."""
    hist = quality_histogram(state, bins=bins)
    if not hist.get("available"):
        return dict(hist)
    dih = dihedral_summary(state)
    dihedral_angles = dih.get("dihedral_angles") if dih.get("available") else None
    out = dict(hist)
    out["dihedral_angles"] = dihedral_angles
    out["minimum_quality"] = dih.get("minimum_quality") if dih.get("available") else None
    out["ready"] = dih.get("ready") if dih.get("available") else None
    out["status"] = dih.get("status") if dih.get("available") else None
    return out


def compare_snapshots(
    before: Optional[dict[str, Any]],
    after: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Numeric deltas for overlapping scalar keys (after - before)."""
    if not isinstance(before, dict) or not before.get("available"):
        return {"available": False, "reason": "before snapshot missing"}
    if not isinstance(after, dict) or not after.get("available"):
        return {"available": False, "reason": "after snapshot missing"}
    deltas: dict[str, Any] = {}
    for key in _SNAPSHOT_SCALAR_KEYS:
        b = before.get(key)
        a = after.get(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            deltas[key] = float(a) - float(b)
    return {"available": True, "deltas": deltas}
