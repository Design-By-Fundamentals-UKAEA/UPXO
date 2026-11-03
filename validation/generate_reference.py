"""Regenerate the frozen mtex reference values used by tests/validation_mtex.

Needs mtex (pyMTEX). Nothing in the UPXO package imports it; the tests read the JSON this
script writes, so they run without mtex. Run from the repository root:

    python validation/generate_reference.py

Output: tests/validation_mtex/reference/ks_mtex.json
Every file carries the versions and the date of the run, so a change in a frozen value can be
traced to a change in mtex.
"""
import datetime
import json
import pathlib
import platform
import warnings
from collections import Counter

import numpy as np

warnings.simplefilter("ignore")
import mtex as mtx                                                      # noqa: E402
from mtex.geometry.standardorientations import KS_ORDER, variants       # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "validation_mtex" / "reference" / "ks_mtex.json"


def main():
    cs = mtx.crystalFrame("m-3m", [1, 1, 1])
    p2c = mtx.orientation.KurdjumovSachs(cs, cs)

    # Child orientations of a parent with the identity orientation, in mtex's own variant order.
    parent = mtx.orientation.byEuler(np.zeros(1), np.zeros(1), np.zeros(1), cs)
    kids = variants(p2c, parent)
    R_child = np.asarray(kids.matrix(), dtype=float).reshape(24, 3, 3)          # crystal -> sample

    ang = np.zeros((24, 24))
    for i in range(24):
        ang[i] = np.degrees(np.asarray(kids[i].angle(kids))).ravel()
    iu = np.triu_indices(24, 1)
    counts = Counter(np.round(ang[iu], 2).tolist())

    data = {
        "provenance": {
            "generator": "validation/generate_reference.py",
            "date": datetime.date.today().isoformat(),
            "mtex": getattr(mtx, "__version__", "unknown"),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "symmetry": "m-3m",
            "relationship": "mtex orientation.KurdjumovSachs(cs, cs)",
            "parent": "identity orientation",
        },
        "ks": {
            "variant_matrices_crystal_to_sample": R_child.tolist(),
            "mtex_variant_index_is_row_plus_one": True,
            "pair_angle_counts_deg": {f"{k:.2f}": v for k, v in sorted(counts.items(), reverse=True)},
            "n_pairs": int(len(iu[0])),
            "mtex_to_morito_permutation": [int(x) for x in KS_ORDER],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", OUT, "|", len(counts), "distinct angles over", data["ks"]["n_pairs"], "pairs")


if __name__ == "__main__":
    main()
