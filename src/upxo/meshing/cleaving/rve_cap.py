"""
rve_cap.py
==========
Exact RVE-face capping, for uses (CPFEM foremost) where the outer box must
be EXACTLY flat -- not merely close. relax.snap_to_rve_boundary's safety
net cannot always deliver that: some boundary points structurally cannot
reach the flat plane through coordinate motion alone, because doing so
would make them exactly coincide with an already-existing vertex on that
same plane (verified directly during development -- pushing the safety
floor from 0.2 down to 0.01 left the identical subset of points stuck at
the same position, not converging toward flat, the signature of a
topological wall rather than an overly conservative parameter).

Why that's not a numerical inconvenience but the CORRECT answer: consider a
cut point on an edge from a grain vertex G (already exactly on the
boundary plane) to a wall vertex W. The true crossing of the flat plane
with segment G-W is G itself -- not a new point. Any output tet using both
G and "the cut point on G's edge" as separate corners is, in exact flat
geometry, asking for two corners at the same location: genuinely
degenerate, not approximately so. The fix is a topology change, not a
coordinate move: merge the two into one point, and remove the tet(s) that
degenerate as a result -- they were artifacts of the wobbly approximation,
never real geometry to begin with.

The first version of this used a coordinate-rounding tolerance to decide
what merges with what, which seemed reasonable but broke badly near RVE
edges/corners: a point that's independently "close enough" to two or three
different boundary planes at once got lumped into a single large cluster
with everything else near that corner, regardless of whether they were
actually the SAME point in exact geometry -- corrupting large parts of the
mesh (confirmed directly: it deleted roughly half the tets and produced
negative volumes). The fix is to stop guessing from resolved coordinates
and use the EXACT fact cleave_lattice already has and now exposes
(CleaveResult.new_point_gids): which original lattice vertices a cut/
triple/quadruple point is actually defined by. A point only ever collapses
onto ONE of its OWN defining vertices, exactly when doing so is the
provably correct answer -- never onto some other, unrelated point merely
because they end up numerically close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from upxo.meshing.cleaving.lattice import BCCLattice
from upxo.meshing.cleaving.cleave import CleaveResult
from upxo.meshing.cleaving.relax import SnapConfig, snap_to_rve_boundary

_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass
class CapConfig:
    """Controls for exact RVE-face capping."""
    # Passed through to the phase-2 ordinary (coordinate-only) snap for
    # whatever near-boundary points don't need merging.
    snap: SnapConfig = None
    # Absolute tolerance for "coordinate exactly equals a boundary plane /
    # exactly equals a defining vertex's position" -- lattice vertex
    # coordinates are exact multiples of voxel_size, so this only needs to
    # absorb floating-point noise, not approximate anything.
    exact_tolerance: float = 1e-9

    def __post_init__(self):
        if self.snap is None:
            self.snap = SnapConfig()


def strip_labels(result: CleaveResult, labels_to_remove) -> CleaveResult:
    """
    Drop every tet carrying one of ``labels_to_remove`` (e.g. the wall
    sentinel) and compact away the nodes that were only ever referenced by
    them. A natural prerequisite before capping for CPFEM use -- wall was
    never real material, only scaffolding for unambiguous labelling near
    the domain edge.
    """
    to_remove = set(int(l) for l in labels_to_remove)
    keep = np.array([int(l) not in to_remove for l in result.tet_labels])
    tets = result.tets[keep]
    tet_labels = result.tet_labels[keep]
    source_tet_index = result.source_tet_index[keep]

    used = np.unique(tets.ravel()) if len(tets) else np.array([], dtype=np.int64)
    node_coords = result.node_coords
    new_point_gids = result.new_point_gids
    if len(used) < len(node_coords):
        remap = np.full(len(node_coords), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        node_coords = node_coords[used]
        tets = remap[tets]
        new_point_gids = {int(remap[old]): g for old, g in new_point_gids.items()
                          if remap[old] >= 0}

    return CleaveResult(
        node_coords=node_coords,
        tets=tets,
        tet_labels=tet_labels,
        source_tet_index=source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=new_point_gids,
    )


def _boundary_planes(lattice: BCCLattice):
    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    return lo, hi


def _find_exact_merges(lattice: BCCLattice, result: CleaveResult, cfg: CapConfig):
    """
    For every new (cut/triple/quadruple) point, using ONLY the exact
    coordinates of its own defining original vertices: snap each axis to a
    boundary plane whenever one of those defining vertices already sits
    exactly on it, then check whether the fully-snapped position exactly
    equals one of those SAME defining vertices' own position. If so, this
    point provably collapses onto that vertex (never onto any other,
    unrelated point). Returns merge_target: {point_id: survivor_gid}.

    Points that snap on some axis WITHOUT landing on one of their own
    defining vertices are deliberately left alone here -- that's not a
    provable exact-collapse case, and moving them without a safety net
    (this function's whole reason for using EXACT facts instead of
    tolerances) would just reintroduce the risk this module exists to
    remove. relax.snap_to_rve_boundary's own tolerance-based detection
    picks those up afterward, under its bisection safety net.
    """
    tol = cfg.exact_tolerance
    lo, hi = _boundary_planes(lattice)
    base = lattice.vertices  # original vertex coordinates, indexed by gid

    merge_target = {}

    for point_id, gids in result.new_point_gids.items():
        gid_coords = base[list(gids)]  # (k, 3)
        target = result.node_coords[point_id].copy()
        any_snap = False
        for axis in range(3):
            vals = gid_coords[:, axis]
            at_lo = np.any(np.abs(vals - lo[axis]) <= tol)
            at_hi = np.any(np.abs(vals - hi[axis]) <= tol)
            if at_lo and not at_hi:
                target[axis] = lo[axis]
                any_snap = True
            elif at_hi and not at_lo:
                target[axis] = hi[axis]
                any_snap = True
            # if both at_lo and at_hi somehow (degenerate domain thickness
            # of 1), leave this axis alone rather than guess.
        if not any_snap:
            continue
        for gid, gc in zip(gids, gid_coords):
            if np.all(np.abs(target - gc) <= tol):
                merge_target[point_id] = int(gid)
                break

    return merge_target


def _tet_volumes(coords: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = coords[tets[:, 0]], coords[tets[:, 1]], coords[tets[:, 2]], coords[tets[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)


def _apply_exact_merges(result: CleaveResult, merge_target):
    n_points = len(result.node_coords)
    coords = result.node_coords  # survivors are always original vertices,
                                  # which never move, so this never needs
                                  # to change through this function

    if not merge_target:
        return CleaveResult(
            node_coords=coords.copy(), tets=result.tets, tet_labels=result.tet_labels,
            source_tet_index=result.source_tet_index,
            deferred_tet_indices=result.deferred_tet_indices,
            new_point_gids=result.new_point_gids,
        ), False

    # A merge can affect a tet in two ways: (a) the tet ALSO has the
    # survivor as a separate corner, so it collapses to a repeated vertex
    # -- correctly caught below and removed; or (b) the tet has ONLY the
    # merged point, not the survivor, as a corner -- that corner's position
    # effectively jumps from wherever the point used to sit to the
    # survivor's position, which can leave the tet non-degenerate but with
    # BAD (near-zero or negative) volume.
    #
    # A bad case-(b) tet's ONLY possible culprits are its OWN corners that
    # are themselves merging points. An earlier version instead checked
    # whether the (shared, widely-reused) SURVIVOR vertex merely appeared
    # in some bad tet ANYWHERE in the mesh -- a survivor is an original
    # lattice vertex referenced by dozens of unrelated tets, so that check
    # reverted every merge landing on a popular anchor regardless of
    # whether it actually caused the problem (confirmed directly: it
    # reverted all 421 candidates on a real test mesh, even though most
    # individually cause no conflict at all). The fix is a greedy minimum-
    # hitting-set search: repeatedly revert whichever single candidate
    # point is currently a corner of the most bad tets, re-check, repeat --
    # so one tough conflict only costs the point(s) actually responsible
    # for it, extracting the maximum subset of merges that can coexist
    # rather than an all-or-nothing outcome.
    active = dict(merge_target)
    tets = result.tets
    pid_arr = np.fromiter(active.keys(), dtype=np.int64, count=len(active))
    touches_candidate = np.isin(tets, pid_arr).any(axis=1)
    relevant_tets = tets[touches_candidate]

    for _ in range(len(merge_target) + 1):
        if not active:
            break
        remap = np.arange(n_points, dtype=np.int64)
        for pid, survivor in active.items():
            remap[pid] = survivor
        new_tets = remap[relevant_tets]

        degenerate = np.zeros(len(new_tets), dtype=bool)
        for a, b in _EDGE_PAIRS:
            degenerate |= new_tets[:, a] == new_tets[:, b]

        vol = _tet_volumes(coords, new_tets)
        bad = (~degenerate) & (vol <= 0)
        if not bad.any():
            break

        culprit_counts = {}
        for row in relevant_tets[bad]:
            for corner in row:
                c = int(corner)
                if c in active:
                    culprit_counts[c] = culprit_counts.get(c, 0) + 1
        if not culprit_counts:
            break  # safety valve; shouldn't happen
        worst = max(culprit_counts, key=culprit_counts.get)
        del active[worst]
    else:
        raise RuntimeError('exact-merge safety loop did not converge')

    remap = np.arange(n_points, dtype=np.int64)
    for pid, survivor in active.items():
        remap[pid] = survivor
    new_tets = remap[tets]
    degenerate = np.zeros(len(new_tets), dtype=bool)
    for a, b in _EDGE_PAIRS:
        degenerate |= new_tets[:, a] == new_tets[:, b]
    kept = ~degenerate

    tets = new_tets[kept]
    tet_labels = result.tet_labels[kept]
    source_tet_index = result.source_tet_index[kept]

    new_point_gids = {pid: g for pid, g in result.new_point_gids.items()
                      if pid not in active}

    # Deliberately NOT compacting node_coords/new_point_gids here: this
    # function's notion of "which node id maps to which defining gids" is
    # positional and must stay stable through phase 2 (the ordinary snap),
    # which also needs len(lattice.vertices) to mean what it always has.
    # Compaction happens exactly once, in cap_rve_boundary_exactly.
    merged = CleaveResult(
        node_coords=coords.copy(),
        tets=tets,
        tet_labels=tet_labels,
        source_tet_index=source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=new_point_gids,
    )
    return merged, True


def _drop_unreferenced_nodes(result: CleaveResult) -> CleaveResult:
    node_coords, tets = result.node_coords, result.tets
    if len(tets) == 0:
        return result
    used = np.unique(tets.ravel())
    if len(used) == len(node_coords):
        return result
    remap = np.full(len(node_coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_point_gids = {int(remap[old]): g for old, g in result.new_point_gids.items()
                      if remap[old] >= 0}
    return CleaveResult(
        node_coords=node_coords[used],
        tets=remap[tets],
        tet_labels=result.tet_labels,
        source_tet_index=result.source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=new_point_gids,
    )


def cap_rve_boundary_exactly(
        lattice: BCCLattice,
        result: CleaveResult,
        config: Optional[CapConfig] = None,
        remove_labels=None,
) -> CleaveResult:
    """
    Make the RVE's own outer faces exactly flat: merge every point that
    provably collapses onto one of its OWN defining original vertices
    (dropping the tet(s) that are genuinely degenerate in exact flat
    geometry as a result), then apply the ordinary safety-net snap
    (relax.snap_to_rve_boundary) for whatever near-boundary points remain.
    Original lattice vertices never move; topology otherwise changes only
    by removing the handful of tets identified above.

    ``remove_labels`` (e.g. ``[wall_label]``) is stripped FIRST, before any
    merging -- not as an afterthought. This matters, not just tidiness: a
    cut point sits on the shared cache, referenced by BOTH the grain-side
    output tet(s) and the wall-side one(s). Merging it onto a grain vertex
    is correct for the grain side, but would also silently redirect
    whatever wall-side tet used the same point onto that same grain
    vertex -- corrupting a tet that was never meant to be touched (this was
    caught directly during development: it produced negative volumes and
    exterior coordinates far outside the domain). Stripping wall tets
    first removes them from consideration entirely, so merges only ever
    touch genuine grain-side geometry.

    ``result`` may be cleave_lattice's own output, or anything that has
    only passed through this module's or relax.py's functions (all of
    which keep new_point_gids correctly in sync through any compaction).
    """
    cfg = config or CapConfig()
    if remove_labels:
        result = strip_labels(result, remove_labels)

    merge_target = _find_exact_merges(lattice, result, cfg)
    merged_result, _ = _apply_exact_merges(result, merge_target)
    snapped = snap_to_rve_boundary(lattice, merged_result, cfg.snap)

    return _drop_unreferenced_nodes(snapped)
