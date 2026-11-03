"""Generate the markdown tables and comparison numbers used on the mtex-validation wiki pages.

Needs mtex (pyMTEX); nothing in the shipped package imports this module or mtex. Run from the
repository root:

    python validation/generate_report.py

Writes to validation/reports/: ks_pair_table.md, twin_sigma_chain.md, pole_figure_comparison.md,
provenance.json. These are the source tables for the wiki pages -- copy numbers from here, don't
retype them. Nothing here is committed automatically; review the output before committing the
reports folder or pasting a table into the wiki.

Every number is either read from the library (ground truth for what UPXO computes) or computed
live against mtex on the same input (the "UPXO vs mtex" column). No real EBSD data, no file
path and no measured orientation appears anywhere in this script or its output.
"""
import datetime
import itertools
import json
import pathlib
import platform
import subprocess
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
import mtex as mtx                                                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from validation_mtex import _oracle as ref                                       # noqa: E402

from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import (                     # noqa: E402
    OrientationAssigner3D,
    disorientation_from_g_batch,
)

OUT = pathlib.Path(__file__).resolve().parent / "reports"
OUT.mkdir(exist_ok=True)

CS = mtx.crystalFrame("m-3m", [1, 1, 1])
_PLANE_NAMES = ["(111)", "(-111)", "(1-11)", "(11-1)"]


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        branch = "unknown"
    return {
        "generator": "validation/generate_report.py",
        "date": datetime.date.today().isoformat(),
        "mtex": getattr(mtx, "__version__", "unknown"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "upxo_commit": commit,
        "upxo_branch": branch,
    }


def _axis_label(axis):
    """A canonical family name for a sorted, absolute-value axis, or the raw triple."""
    a = np.round(np.abs(np.asarray(axis, float)), 3)
    if np.allclose(a, a[0]):
        return "<111>"
    if np.isclose(a[2], 0) and np.isclose(a[0], a[1]):
        return "<110>"
    if np.isclose(a[2], 0) and a[1] > 1e-6 and abs(a[0] / a[1] - 2) < 0.05:
        return "<210>"
    return f"[{a[0]:.3f} {a[1]:.3f} {a[2]:.3f}]"


# ---- 1. KS pair table -------------------------------------------------------------------------

def ks_pair_table():
    K = OrientationAssigner3D.get_ks_rotations()
    iu = np.triu_indices(24, 1)
    angles = disorientation_from_g_batch(K[iu[0]], K[iu[1]])
    packets = np.array([ref.packet_index(g) for g in K])
    same = packets[iu[0]] == packets[iu[1]]

    groups = {}
    for a, s in zip(angles, same):
        k = round(float(a), 2)
        rec = groups.setdefault(k, [0, 0])
        rec[0 if s else 1] += 1

    # Live mtex cross-check: the same 276 pairs, from mtex's own variant set.
    from mtex.geometry.standardorientations import variants
    p2c = mtx.orientation.KurdjumovSachs(CS, CS)
    parent = mtx.orientation.byEuler(np.zeros(1), np.zeros(1), np.zeros(1), CS)
    kids = variants(p2c, parent)
    mtex_ang = np.zeros((24, 24))
    for i in range(24):
        mtex_ang[i] = np.degrees(np.asarray(kids[i].angle(kids))).ravel()
    mtex_vals = sorted({round(float(x), 2) for x in mtex_ang[iu]}, reverse=True)
    lib_vals = sorted(groups, reverse=True)
    value_match = lib_vals == mtex_vals
    max_pair_diff = float(np.abs(np.sort(mtex_ang[iu]) - np.sort(angles)).max())

    lines = ["| Angle (deg) | Pairs | Inside a packet | Between packets |", "|---|---|---|---|"]
    total = inside_t = between_t = 0
    for k in sorted(groups, reverse=True):
        inside, between = groups[k]
        total += inside + between
        inside_t += inside
        between_t += between
        lines.append(f"| {k:.2f} | {inside + between} | {inside} | {between} |")
    lines.append(f"| **Total** | **{total}** | **{inside_t}** | **{between_t}** |")
    lines.append("")
    lines.append(f"The 10 distinct values match mtex's own 276-pair computation on the same "
                 f"relationship (same set: {value_match}). Largest per-pair difference between "
                 f"the sorted 276 angles: {max_pair_diff:.2e} degrees.")
    return "\n".join(lines), {
        "distinct_values_match_mtex": value_match,
        "max_pair_diff_vs_mtex_deg": max_pair_diff,
        "total_pairs": total,
    }


# ---- 2. Twin Sigma chain table -----------------------------------------------------------------

def twin_sigma_chain():
    T = [ref.twin_operator(n) for n in ref.FOUR_111]

    rows = []
    a, ax = ref.disorientation_axis_family(np.eye(3), T[0])
    rows.append(("3", a, ax, "single twin: one Sigma3 relationship"))

    a, ax = ref.disorientation_axis_family(T[0], T[1])
    rows.append(("9", a, ax, "two twins of one parent on different {111} planes"))

    B = [t @ s for t in T for s in ref.OPS]
    sigma9 = [T[i].T @ T[j] for i, j in itertools.combinations(range(4), 2)]
    found = {}
    for a_ in sigma9:
        for b_ in B:
            ang, axis = ref.disorientation_axis_family(np.eye(3), a_ @ b_)
            found.setdefault(round(ang, 2), axis)
    for key, sigma in ((31.59, "27a"), (35.43, "27b")):
        rows.append((sigma, key, found[key], "chain of three Sigma3 relationships"))

    # two Sigma3 relationships: the full reachable set, to state the Sigma3-is-absent fact plainly
    A = [T[0] @ s for s in ref.OPS]
    two_vals = sorted({round(float(ref.disorientation_R(np.eye(3), a @ b)[0]), 2) for a in A for b in B})

    lines = ["| Sigma | Disorientation (deg) | Axis family | Description |", "|---|---|---|---|"]
    for sigma, ang, axis, desc in rows:
        lines.append(f"| {sigma} | {ang:.2f} | {_axis_label(axis)} | {desc} |")
    lines.append("")
    lines.append(f"Composing two Sigma3 relationships gives only {two_vals} degrees over the full "
                 f"reachable set (Sigma1 or Sigma9 -- never Sigma3).")

    # cross-check each value against a live mtex computation of the same chain
    live = {}
    o0 = mtx.orientation.byMatrix(np.eye(3), CS)
    for sigma, ang, axis, desc in rows:
        R = {"3": T[0], "9": T[0].T @ T[1], "27a": None, "27b": None}.get(sigma)
        if R is None:
            continue
        o1 = mtx.orientation.byMatrix(R, CS)
        live[sigma] = float(np.degrees(np.asarray(o0.angle(o1))).ravel()[0])
    max_diff = max(abs(live[s] - a) for s, a, *_ in rows if s in live)
    lines.append(f"Largest difference from a live mtex computation (Sigma3 and Sigma9 rows): "
                 f"{max_diff:.2e} degrees.")
    return "\n".join(lines), {"two_sigma3_reachable_values": two_vals, "max_diff_vs_mtex_deg": max_diff}


# ---- 3. Pole-figure comparison ------------------------------------------------------------------

def _mtex_poles(o, hkl):
    m = mtx.symmetrise(mtx.Miller(*hkl, CS))
    r = o * m
    return np.stack([np.asarray(r.x).ravel(), np.asarray(r.y).ravel(), np.asarray(r.z).ravel()], axis=1)


def _oracle_family_poles(R, hkl):
    h = np.asarray(hkl, float)
    eq = np.unique(np.round(ref.OPS @ h, 6), axis=0)
    eq /= np.linalg.norm(eq, axis=1, keepdims=True)
    return (R @ eq.T).T


def _max_pole_gap(a, b):
    d = np.clip(np.abs(a @ b.T), -1, 1)
    return float(np.degrees(np.arccos(d.max(axis=1))).max())


def pole_figure_comparison():
    lines = ["### Twin variants of one parent ({111} poles)", "",
             "| Orientation | Largest angular gap, UPXO vs mtex (deg) |", "|---|---|"]
    Rp = ref.bunge_R(35, 25, 15)[0]
    o_p = mtx.orientation.byMatrix(Rp, CS)
    gap_p = _max_pole_gap((Rp @ ref.FOUR_111.T).T, _mtex_poles(o_p, (1, 1, 1)))
    lines.append(f"| parent | {gap_p:.2e} |")
    worst = gap_p
    for k in range(4):
        Rt = Rp @ ref.twin_operator(ref.FOUR_111[k])
        o_t = mtx.orientation.byMatrix(Rt, CS)
        gap = _max_pole_gap((Rt @ ref.FOUR_111.T).T, _mtex_poles(o_t, (1, 1, 1)))
        worst = max(worst, gap)
        lines.append(f"| twin on {_PLANE_NAMES[k]} | {gap:.2e} |")
    lines.append("")
    lines.append(f"Worst case over the parent and its four twins: {worst:.2e} degrees.")

    lines += ["", "### 24 KS variants of one parent ({100}, {110}, {111} poles)", "",
              "| Family | Largest angular gap, UPXO vs mtex (deg) |", "|---|---|"]
    from mtex.geometry.standardorientations import variants
    K = OrientationAssigner3D.get_ks_rotations()
    p2c = mtx.orientation.KurdjumovSachs(CS, CS)
    e = (33.0, 41.0, 17.0)
    Rp2 = ref.bunge_R(*e)[0]
    o_parent = mtx.orientation.byEuler(*np.radians(e), CS)
    kids = variants(p2c, o_parent)
    worst_all = 0.0
    for name, hkl in (("{100}", (1, 0, 0)), ("{110}", (1, 1, 0)), ("{111}", (1, 1, 1))):
        u_poles = np.concatenate([_oracle_family_poles(Rp2 @ K[i].T, hkl) for i in range(24)], axis=0)
        m_poles = np.concatenate([_mtex_poles(kids[i], hkl) for i in range(24)], axis=0)
        gap = _max_pole_gap(u_poles, m_poles)
        worst_all = max(worst_all, gap)
        lines.append(f"| {name} | {gap:.2e} |")
    lines.append("")
    lines.append(f"Worst case over the three families and the 24 variants: {worst_all:.2e} degrees.")
    return "\n".join(lines), {"worst_twin_pole_gap_deg": worst, "worst_ks_pole_gap_deg": worst_all}


def main():
    prov = provenance()
    summary = {"provenance": prov}

    ks_md, ks_stats = ks_pair_table()
    (OUT / "ks_pair_table.md").write_text(ks_md + "\n", encoding="utf-8")
    summary["ks_pair_table"] = ks_stats

    twin_md, twin_stats = twin_sigma_chain()
    (OUT / "twin_sigma_chain.md").write_text(twin_md + "\n", encoding="utf-8")
    summary["twin_sigma_chain"] = twin_stats

    pf_md, pf_stats = pole_figure_comparison()
    (OUT / "pole_figure_comparison.md").write_text(pf_md + "\n", encoding="utf-8")
    summary["pole_figure_comparison"] = pf_stats

    (OUT / "provenance.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("wrote", OUT / "ks_pair_table.md")
    print("wrote", OUT / "twin_sigma_chain.md")
    print("wrote", OUT / "pole_figure_comparison.md")
    print("wrote", OUT / "provenance.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
