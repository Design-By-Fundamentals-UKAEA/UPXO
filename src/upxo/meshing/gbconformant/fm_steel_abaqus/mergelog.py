"""Which raw features were absorbed into which, and how far their orientations are apart.

When whole grains merge during cleaning, the merged grain carries one raw feature's orientation. The
log lists every absorbed feature with the feature whose orientation it now has (the winner of the
identity rule) and the misorientation between the two: the size of that orientation substitution.
It reports only; nothing is decided from it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .comparison import describe
from .export import feature_elset_name

LOW_ANGLE = 15.0
COLUMNS = ("meshed_label", "absorbed", "absorbed_name", "voxels", "winner", "winner_name",
           "misorientation_deg", "winner_is_survivor")


def merge_log(hierarchy, identity, *, symmetry="cubic", prefixes=None):
    """Misorientation between every absorbed raw feature and the winner of its merged grain.

    ``identity`` is the :class:`IdentityMap`; ``prefixes`` are the element-set name prefixes.
    Only cubic symmetry is available. Returns ``{'columns', 'rows', 'summary', 'symmetry'}``.
    """
    if symmetry != "cubic":
        raise ValueError("The misorientation of merged grains is only available for cubic symmetry.")
    from upxo.xtalphy.crystal_orientation import cubic_misorientation

    prefixes = prefixes or C.ELSET_PREFIXES
    level = hierarchy.level

    def name(label):
        return feature_elset_name(hierarchy, prefixes, hierarchy.features[label])

    rows = []
    for group in identity.report.get("merged_groups", []):
        winner = int(group["winner"])
        for raw, voxels in sorted(group["members"].items()):
            raw = int(raw)
            if raw == winner:
                continue
            angle, _axis, _top = cubic_misorientation(hierarchy.features[raw].euler,
                                                      hierarchy.features[winner].euler)
            rows.append([int(group["meshed_label"]), raw, name(raw), int(voxels), winner, name(winner),
                         float(angle), not group["differs_from_survivor"]])
    angles = [r[6] for r in rows]
    weights = np.array([r[3] for r in rows], dtype=float)
    summary = {
        "groups": len(identity.report.get("merged_groups", [])),
        "absorbed_features": len(rows),
        "voxels": int(weights.sum()),
        "misorientation": describe(angles),
        "voxel_weighted_mean_deg": float(np.average(angles, weights=weights)) if rows and weights.sum() else None,
        f"above_{LOW_ANGLE:g}_deg": int(sum(a > LOW_ANGLE for a in angles)),
        "rule": identity.rule,
    }
    return {"columns": list(COLUMNS), "rows": rows, "summary": summary, "symmetry": symmetry}


def write_merge_log(folder, log):
    """Write ``merge_log.json`` and ``merge_log.csv`` into ``folder``; returns both paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = folder / "merge_log.json", folder / "merge_log.csv"
    json_path.write_text(json.dumps(log, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)),
                         encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(log["columns"])
        writer.writerows(log["rows"])
    return json_path, csv_path
