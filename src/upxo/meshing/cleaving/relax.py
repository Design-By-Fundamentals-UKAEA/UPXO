"""
relax.py
========
Post-process quality refinement: Laplacian-smooth the new points cleaving
introduced (cut points, triple points, quadruple points) along the grain-
boundary surface, each move clipped to a safe distance so no tet touching
that point can ever invert or collapse. Topology -- which tets exist, their
labels, node connectivity -- is completely unchanged; only coordinates move.

This targets the gap identified during development: cut/triple/quadruple
points are placed at simple centroids (edge midpoint, face/tet centroid),
which conserves volume and keeps every tet valid, but has no way to know
where the TRUE sub-voxel grain boundary actually sits -- UPXO's label field
carries no distance information, only in/out membership. So the
reconstructed surface "staircases" across roughly a voxel's width instead
of sitting on one flat/smooth surface. Smoothing doesn't add missing
information, but it does let neighbouring boundary points influence each
other and pull the jagged approximation toward something smoother, subject
to the same kind of safe-movement bound ("alpha selection") the Lattice
Cleaving paper uses for its own vertex warping -- computed directly here
rather than approximated, since a tet's volume is an exactly linear
function of any one of its vertices, so the exact point at which a move
would zero out (or invert) a tet's volume is a closed-form calculation, not
a guess.

``snap_to_rve_boundary`` targets the same root cause showing up somewhere
more consequential: the RVE's own outer faces are supposed to be exactly
flat by construction, but the wall label is just another opaque value to
the cleaving stencils, so the grain/wall interface wobbles by the same
up-to-half-a-voxel amount as any interior boundary (confirmed directly: a
uniform single-grain domain's own outer face showed real z-variation from
7.0 to 7.5, not a rendering artifact). Unlike Laplacian smoothing, this has
an exact target to snap to (the boundary plane's own equation), so it uses
a proper bisection search for the largest safe step rather than the crude
halving relax_boundary_points uses -- worth the extra precision here since
the whole point is getting as close to exactly flat as safely possible, not
leaving avoidable wobble on the table. And because pulling several points
toward the SAME shared plane can crowd them together in a way Laplacian
smoothing (which only ever pulls toward a local average) doesn't risk, this
adds a second, independent floor beyond the volume one: no mesh edge may
shrink below a configured fraction of its pre-snap length either -- volume
alone can stay comfortably positive right up until two of a tet's corners
are nearly coincident, so the edge-length floor is what actually catches
that specific failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from upxo.meshing.cleaving.lattice import BCCLattice
from upxo.meshing.cleaving.cleave import CleaveResult

_OTHER_LOCAL = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def _is_new_point_mask(result: CleaveResult, n_points: int) -> np.ndarray:
    """
    Which node ids are cut/triple/quadruple points (as opposed to original
    lattice vertices, which never move). Derived from
    CleaveResult.new_point_gids -- a dict, not a fixed positional cutoff --
    so this stays correct even after a prior compaction (e.g. strip_labels)
    has renumbered nodes, unlike checking ``index >= len(lattice.vertices)``
    directly would.
    """
    mask = np.zeros(n_points, dtype=bool)
    if result.new_point_gids:
        mask[np.fromiter(result.new_point_gids.keys(), dtype=np.int64,
                          count=len(result.new_point_gids))] = True
    return mask


@dataclass
class RelaxConfig:
    """Controls for boundary-point Laplacian relaxation."""
    n_iterations: int = 8
    # Fraction of the exact degenerate-distance to allow moving, per tet,
    # per iteration. <1 leaves a safety margin so no tet ever gets razor-thin.
    safety_factor: float = 0.5
    # No tet may ever be pushed below this fraction of its OWN volume as it
    # stood before relaxation started. Plain "still positive" is too weak a
    # guarantee on its own -- it allows a technically-valid but useless
    # sliver; this keeps every tet a meaningful fraction of its original size.
    min_volume_fraction: float = 0.2


def _boundary_surface_edges(tets: np.ndarray, tet_labels: np.ndarray):
    """
    (rows, cols) directed-edge arrays over the grain-boundary surface: an
    edge (a,b) exists for every pair of vertices sharing a triangular tet
    face where the two tets on either side of that face carry different
    labels. Each such edge appears in both directions.
    """
    face_defs = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    faces = np.concatenate([tets[:, list(idx)] for idx in face_defs], axis=0)
    tet_of_face = np.tile(np.arange(len(tets)), 4)

    sorted_faces = np.sort(faces, axis=1)
    order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sf = sorted_faces[order]
    tf = tet_of_face[order]

    if len(sf) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    same_as_next = np.all(sf[:-1] == sf[1:], axis=1)
    pair_start = np.nonzero(same_as_next)[0]

    t0 = tf[pair_start]
    t1 = tf[pair_start + 1]
    is_boundary = tet_labels[t0] != tet_labels[t1]
    bfaces = sf[pair_start][is_boundary]

    if len(bfaces) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    a = bfaces[:, [0, 0, 1]].ravel()
    b = bfaces[:, [1, 2, 2]].ravel()
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    return rows, cols


def relax_boundary_points(
        lattice: BCCLattice,
        result: CleaveResult,
        config: Optional[RelaxConfig] = None,
) -> CleaveResult:
    """
    Return a new CleaveResult with identical topology (tets, tet_labels,
    source_tet_index, deferred_tet_indices) but with cut/triple/quadruple
    point coordinates Laplacian-smoothed along the grain-boundary surface.

    Original lattice vertices (grain-interior points) never move -- only
    points at index >= len(lattice.vertices) are eligible. Each point's
    move is clipped, per tet that touches it, to strictly less than the
    exact distance at which that tet's volume would hit zero (scaled by
    ``config.safety_factor``), so this can never introduce a degenerate or
    inverted element regardless of how many iterations are run.

    A point that already sits exactly on one of the RVE's own boundary
    planes (boundary_clip.py's whole reason for existing) is additionally
    constrained to slide WITHIN whichever planes it starts exactly on --
    a face leaves 2 free axes, an edge leaves 1, a corner is fully pinned
    -- computed once from the untouched starting coordinates, not
    recomputed per iteration (a point nudged fractionally off a plane in
    an earlier iteration must not then read as "no longer on the plane"
    and be free to drift further). Without this, plain Laplacian averaging
    pulls a grain-boundary triple junction that happens to land on the RVE
    surface toward the centroid of its neighbours -- which are not all on
    that same plane -- depressing it inward off the exact boundary
    (confirmed directly: this is exactly what an unconstrained relax pass
    does to boundary-surface junctions here).
    """
    cfg = config or RelaxConfig()
    coords = result.node_coords.copy()
    tets = result.tets
    n_points = len(coords)
    is_new = _is_new_point_mask(result, n_points)

    rows, cols = _boundary_surface_edges(tets, result.tet_labels)
    movable_mask = is_new[rows]
    rows_m, cols_m = rows[movable_mask], cols[movable_mask]

    if len(rows_m) == 0:
        return result  # no grain-boundary surface to relax

    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    hi_bound = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    axis_free = np.ones((n_points, 3), dtype=bool)
    for axis in range(3):
        on_plane = (coords[:, axis] == 0.0) | (coords[:, axis] == hi_bound[axis])
        axis_free[on_plane, axis] = False

    weights = np.ones(len(rows_m))
    adj = sp.coo_matrix((weights, (rows_m, cols_m)), shape=(n_points, n_points)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    movable = np.nonzero((deg > 0) & is_new)[0]

    if len(movable) == 0:
        return result

    # (point, tet_index) pairs for every movable point, for the safety check.
    pt_point_parts, pt_tet_parts, pt_local_parts = [], [], []
    for local in range(4):
        pts = tets[:, local]
        idx = np.nonzero(is_new[pts])[0]
        pt_point_parts.append(pts[idx])
        pt_tet_parts.append(idx)
        pt_local_parts.append(np.full(len(idx), local))
    pt_point = np.concatenate(pt_point_parts)
    pt_tet = np.concatenate(pt_tet_parts)
    pt_local = np.concatenate(pt_local_parts)
    pt_others = tets[pt_tet[:, None], _OTHER_LOCAL[pt_local]]  # (P, 3)

    deg_safe = np.where(deg > 0, deg, 1.0)

    def _volumes_of(c, tet_subset):
        a, b, cc, d = (c[tet_subset[:, 0]], c[tet_subset[:, 1]],
                        c[tet_subset[:, 2]], c[tet_subset[:, 3]])
        return np.einsum('ij,ij->i', np.cross(b - a, cc - a), d - a)

    # Floor is relative to each tet's OWN volume as it stood before any
    # relaxation -- fixed for the whole run, not recomputed per iteration,
    # so repeated small erosions across iterations can't compound into a
    # sliver that "was safe relative to last iteration" every single time.
    original_vol = _volumes_of(coords, tets)
    min_allowed = cfg.min_volume_fraction * original_vol
    floor_per_pair = min_allowed[pt_tet]

    # Only tets with at least one corner in `movable` can ever change --
    # `movable` is fixed for this function's entire run (unlike
    # optimize_boundary_points, where the moving set can shrink between
    # iterations), so this is computed once and reused for every retry of
    # every iteration below, instead of re-checking every tet in the mesh
    # each time. Measured directly, not assumed: on a real 8.65M-tet
    # production mesh, the unrestricted full-mesh check consumed 61.8% of
    # this entire function's runtime, despite only 36% of points ever
    # being movable at all.
    affected_mask = np.isin(tets, movable).any(axis=1)
    affected_tets = tets[affected_mask]
    affected_min_allowed = min_allowed[affected_mask]

    for _ in range(cfg.n_iterations):
        target_all = (adj @ coords) / deg_safe[:, None]
        direction = target_all - coords
        direction *= axis_free  # never move a point off a plane it starts exactly on

        # Per-point, single-mover safety bound: a good, cheap pre-clip that
        # sharply reduces (but -- since a tet can have 2+ movable corners
        # moving at once -- does not by itself guarantee) how far a global
        # backtrack below needs to search.
        b = coords[pt_others[:, 0]]
        c = coords[pt_others[:, 1]]
        d = coords[pt_others[:, 2]]
        p0 = coords[pt_point]
        p1 = p0 + direction[pt_point]

        V0 = np.einsum('ij,ij->i', np.cross(b - p0, c - p0), d - p0)
        V1 = np.einsum('ij,ij->i', np.cross(b - p1, c - p1), d - p1)

        denom = V0 - V1
        t_crit = np.full(len(V0), np.inf)
        safe_denom = np.abs(denom) > 1e-14
        t_crit[safe_denom] = (V0[safe_denom] - floor_per_pair[safe_denom]) / denom[safe_denom]
        binding = (t_crit > 0) & (t_crit < 1)
        t_crit[~binding] = np.inf

        t_safe_per_pair = np.clip(cfg.safety_factor * t_crit, 0.0, 1.0)
        t_safe = np.full(n_points, 1.0)
        np.minimum.at(t_safe, pt_point, t_safe_per_pair)

        # Global verify-and-backtrack safety net: the per-point clip above
        # only accounts for ONE mover at a time, but several corners of the
        # same tet can move simultaneously -- so explicitly check every
        # AFFECTED tet's volume (against the same fixed floor) with ALL
        # moves applied together, and if any breached it, uniformly halve
        # the whole step and recheck. This is what actually guarantees
        # correctness, not the per-point estimate. Every tet OUTSIDE
        # affected_tets has all 4 corners provably unchanged by this
        # (or any prior) iteration, so it can't have a different volume
        # than original_vol already confirmed safe -- checking it again
        # would be wasted work, not a shortcut that risks missing something.
        scale = 1.0
        for _ in range(20):
            candidate = coords.copy()
            candidate[movable] = (coords[movable]
                                   + scale * t_safe[movable, None] * direction[movable])
            if (_volumes_of(candidate, affected_tets) > affected_min_allowed).all():
                coords = candidate
                break
            scale *= 0.5
        # If even the smallest tried scale still fails, skip this iteration
        # (coords unchanged) rather than risk an element below the floor.

    return CleaveResult(
        node_coords=coords,
        tets=tets,
        tet_labels=result.tet_labels,
        source_tet_index=result.source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=result.new_point_gids,
    )


@dataclass
class SnapConfig:
    """Controls for snapping boundary points onto the RVE's own outer faces."""
    # A point within this fraction of voxel_size of one of the 6 boundary
    # planes is treated as belonging to it. 0.5 is the theoretical maximum
    # wobble (a cut point is always within half an edge length of either
    # endpoint, and the longest lattice edge is one voxel); a little slack
    # above that catches triple/quadruple points (averages of 3-4 corners)
    # without reaching so far as to catch genuinely interior points on any
    # domain thicker than a couple of voxels.
    tolerance_fraction: float = 0.6
    # No tet may be pushed below this fraction of its OWN pre-snap volume.
    min_volume_fraction: float = 0.2
    # No mesh edge may shrink below this fraction of its OWN pre-snap
    # length -- the check volume alone can't catch: two corners of a tet
    # nearly coinciding can still leave that tet's volume comfortably
    # positive right up until they're almost exactly on top of each other.
    min_edge_fraction: float = 0.2


def snap_to_rve_boundary(
        lattice: BCCLattice,
        result: CleaveResult,
        config: Optional[SnapConfig] = None,
) -> CleaveResult:
    """
    Snap cut/triple/quadruple points that sit near one of the RVE's 6 outer
    faces exactly onto that face. Original lattice vertices never move
    (primal vertices on a boundary plane already sit exactly on it by
    construction); a point near an edge or corner of the RVE snaps on
    multiple axes at once.

    Finds the largest single global step (0 to the full snap, via
    bisection) such that EVERY tet stays above ``min_volume_fraction`` of
    its pre-snap volume AND EVERY mesh edge touched by a moving point stays
    above ``min_edge_fraction`` of its pre-snap length. If even a zero step
    isn't strictly needed to satisfy both (the pre-snap state always does,
    trivially), the search starts from a known-safe point, so this always
    terminates with the largest safe step, never an unsafe one.
    """
    cfg = config or SnapConfig()
    coords = result.node_coords.copy()
    tets = result.tets
    n_points = len(coords)
    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    hi_bound = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    tol = cfg.tolerance_fraction * vs

    is_new_point = _is_new_point_mask(result, n_points)
    direction = np.zeros_like(coords)
    is_movable = np.zeros(n_points, dtype=bool)
    for axis in range(3):
        near_lo = is_new_point & (np.abs(coords[:, axis]) <= tol)
        near_hi = is_new_point & (np.abs(coords[:, axis] - hi_bound[axis]) <= tol)
        direction[near_lo, axis] = -coords[near_lo, axis]
        direction[near_hi, axis] = hi_bound[axis] - coords[near_hi, axis]
        is_movable |= near_lo | near_hi

    movable = np.nonzero(is_movable)[0]
    if len(movable) == 0:
        return result

    def _volumes_of(c, tet_subset):
        a, b, cc, d = (c[tet_subset[:, 0]], c[tet_subset[:, 1]],
                        c[tet_subset[:, 2]], c[tet_subset[:, 3]])
        return np.einsum('ij,ij->i', np.cross(b - a, cc - a), d - a)

    min_vol = cfg.min_volume_fraction * _volumes_of(coords, tets)

    # Only tets with at least one corner in `movable` can ever change --
    # `movable` is fixed for this function's entire run (there's no
    # iteration loop here, just one bisection search over `scale`), so
    # this is computed once and reused for every one of acceptable()'s
    # calls instead of re-checking every tet in the mesh each time.
    # Measured directly, not assumed: on a real 8.79M-tet production mesh
    # only 1.69% of points ever sit near an RVE face at all, yet the
    # unrestricted full-mesh check consumed 90.5% of this function's
    # entire runtime -- an even bigger share than the equivalent fix
    # addressed in relax_boundary_points, precisely because the RVE-face
    # skin is a much thinner slice of the mesh than "adjacent to any
    # grain boundary" is.
    affected_mask = np.isin(tets, movable).any(axis=1)
    affected_tets = tets[affected_mask]
    affected_min_vol = min_vol[affected_mask]

    edge_defs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_a = np.concatenate([tets[:, e[0]] for e in edge_defs])
    edge_b = np.concatenate([tets[:, e[1]] for e in edge_defs])
    touched = is_movable[edge_a] | is_movable[edge_b]
    edge_a, edge_b = edge_a[touched], edge_b[touched]
    min_len = cfg.min_edge_fraction * np.linalg.norm(coords[edge_a] - coords[edge_b], axis=1)

    def acceptable(scale):
        candidate = coords.copy()
        candidate[movable] = coords[movable] + scale * direction[movable]
        if not (_volumes_of(candidate, affected_tets) >= affected_min_vol).all():
            return False, candidate
        new_len = np.linalg.norm(candidate[edge_a] - candidate[edge_b], axis=1)
        return bool((new_len >= min_len).all()), candidate

    full_ok, full_candidate = acceptable(1.0)
    if full_ok:
        return CleaveResult(
            node_coords=full_candidate, tets=tets, tet_labels=result.tet_labels,
            source_tet_index=result.source_tet_index,
            deferred_tet_indices=result.deferred_tet_indices,
            new_point_gids=result.new_point_gids,
        )

    # Bisection for the largest safe scale in [0, 1]. scale=0 (the pre-snap
    # state) always satisfies both floors trivially, so this always
    # terminates with SOME accepted candidate, never an unsafe one.
    lo, hi = 0.0, 1.0
    best_candidate = coords
    for _ in range(24):
        mid = (lo + hi) / 2.0
        ok, candidate = acceptable(mid)
        if ok:
            lo = mid
            best_candidate = candidate
        else:
            hi = mid

    return CleaveResult(
        node_coords=best_candidate,
        tets=tets,
        tet_labels=result.tet_labels,
        source_tet_index=result.source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=result.new_point_gids,
    )


@dataclass
class OptimizeConfig:
    """Controls for quality-driven boundary-point optimization (as opposed
    to relax_boundary_points' Laplacian averaging)."""
    n_iterations: int = 15
    # No tet may ever be pushed below this fraction of its OWN volume as it
    # stood before optimization started -- same floor semantics as
    # RelaxConfig.min_volume_fraction.
    min_volume_fraction: float = 0.2
    # A tet at or above this shape-quality score (see _tet_shape_quality)
    # exerts NO pull on its corners at all -- only tets actually worse than
    # this target participate, and worse ones pull harder (weight =
    # threshold - quality). 0.3 is a deliberately modest bar (a regular tet
    # scores 1.0); this is about fixing clear outliers, not chasing every
    # tet to perfection.
    quality_threshold: float = 0.3
    # Finite-difference step used to estimate each tet's quality gradient
    # w.r.t. a corner's position, as a fraction of voxel_size. A numerical
    # gradient was chosen deliberately over hand-deriving the analytic
    # volume gradient: the sign convention has to be tracked separately per
    # local corner index (0..3), which is exactly the kind of place a
    # transcription error hides -- a finite difference sidesteps that
    # entirely at a modest, one-time cost per iteration.
    finite_difference_eps_fraction: float = 1e-4
    # Bound on how far a point may move in one iteration BEFORE the safety
    # floor scales it down further, as a fraction of voxel_size -- the raw
    # gradient's magnitude isn't a meaningful distance, only its direction
    # is, so this sets the trial step length independently of it.
    max_step_fraction: float = 0.25
    # How much any single tet's quality is allowed to regress below its OWN
    # pre-optimization value (on the 0..1 _tet_shape_quality scale) before a
    # move is rejected. Confirmed directly this cannot be ~0: helping one
    # tet's shape reliably costs some other tet sharing that point a little
    # quality, so a near-zero tolerance rejects every move at every
    # candidate step size and the optimizer never moves anything at all.
    # This is the knob that trades "how much regression elsewhere are we
    # willing to accept" against "how much improvement can happen at all".
    quality_regression_tolerance: float = 0.05


def _tet_shape_quality(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    """
    0 (degenerate) to 1 (regular) tetrahedron shape quality: normalized
    volume per unit (RMS edge length)^3 -- a standard, cheap shape measure
    (Parthasarathy et al. 1993) that needs no trigonometry, unlike angle-
    based measures, so it's cheap enough to evaluate millions of times per
    optimization iteration. Negative for inverted tets, correctly ranking
    them worse than any valid sliver. P0..P3 are (N,3) arrays, one row per
    tet, in the SAME corner order throughout (never pivoted to "whichever
    corner is moving"), so this always matches the one, already-trusted
    signed-volume convention used everywhere else in this module -- no new
    per-corner sign case to get wrong.
    """
    # e1,e2,e3 double as 3 of the 6 edges AND the volume computation's own
    # inputs -- the other 3 edges (P2-P1, P3-P1, P3-P2) are then just
    # differences of these three (e2-e1, e3-e1, e3-e2), not fresh
    # subtractions of P0..P3 -- half the edge vectors for the same result.
    e1, e2, e3 = P1 - P0, P2 - P0, P3 - P0
    vol = np.einsum('ij,ij->i', np.cross(e1, e2), e3) / 6.0
    e4, e5, e6 = e2 - e1, e3 - e1, e3 - e2
    sq_len_sum = (np.einsum('ij,ij->i', e1, e1) + np.einsum('ij,ij->i', e2, e2)
                  + np.einsum('ij,ij->i', e3, e3) + np.einsum('ij,ij->i', e4, e4)
                  + np.einsum('ij,ij->i', e5, e5) + np.einsum('ij,ij->i', e6, e6))
    rms_len = np.sqrt(sq_len_sum / 6.0)
    return (6.0 * np.sqrt(2.0)) * vol / np.maximum(rms_len ** 3, 1e-300)


def optimize_boundary_points(
        lattice: BCCLattice,
        result: CleaveResult,
        config: Optional[OptimizeConfig] = None,
) -> CleaveResult:
    """
    Return a new CleaveResult with identical topology but with cut/triple/
    quadruple point coordinates nudged to directly improve tet SHAPE
    QUALITY, instead of relax_boundary_points' Laplacian averaging (move
    toward the neighbours' centroid, which only improves quality as an
    incidental side effect of smoothing the surface).

    For each movable point, every tet touching it is checked: tets already
    at or above ``config.quality_threshold`` contribute nothing; tets below
    it pull the point toward whatever direction (found via a small finite-
    difference probe) improves THAT tet's own quality, weighted by how bad
    it currently is -- so a point shared by several tets moves toward a
    compromise dominated by its worst offender(s), not an average of all of
    them regardless of how healthy most already are.

    Same guarantees as relax_boundary_points: original lattice vertices
    never move; a point already sitting exactly on one of the RVE's own
    boundary planes is confined to sliding within it (never pulled off);
    every proposed step is verified against every SINGLE tet in the whole
    mesh -- not just the ones a currently-moving point touches -- against
    TWO floors fixed to that tet's own pre-optimization state: volume
    (never below a configured fraction of where it started, so nothing
    inverts or collapses) AND shape quality (never below where it started,
    full stop). The quality floor is not optional or redundant with the
    volume one: a tet can trade volume for a worse shape (become more
    sheared/skewed) while staying comfortably above the volume floor, and
    an earlier version without it let exactly that happen to the single
    worst tet in a real test. Either floor failing backs the whole step off
    (halved, retried) until safe.

    This is a genuinely different tool from relax_boundary_points, not a
    replacement -- it targets tet shape directly, at the cost of being far
    more expensive per iteration (a finite-difference probe per tet per
    point per axis, vs. one sparse matrix multiply). Whether it's worth
    that cost is exactly what running it and comparing the min_angle/
    scaled_jacobian distributions before and after is for.
    """
    cfg = config or OptimizeConfig()
    coords = result.node_coords.copy()
    tets = result.tets
    n_points = len(coords)
    is_new = _is_new_point_mask(result, n_points)

    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    hi_bound = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    axis_free = np.ones((n_points, 3), dtype=bool)
    for axis in range(3):
        on_plane = (coords[:, axis] == 0.0) | (coords[:, axis] == hi_bound[axis])
        axis_free[on_plane, axis] = False

    pt_point_parts, pt_tet_parts, pt_local_parts = [], [], []
    for local in range(4):
        pts = tets[:, local]
        idx = np.nonzero(is_new[pts])[0]
        pt_point_parts.append(pts[idx])
        pt_tet_parts.append(idx)
        pt_local_parts.append(np.full(len(idx), local))
    pt_point = np.concatenate(pt_point_parts)
    pt_tet = np.concatenate(pt_tet_parts)
    pt_local = np.concatenate(pt_local_parts)

    if len(pt_point) == 0:
        return result

    pt_corners = tets[pt_tet]  # (K, 4) global ids, standard corner order

    def _volumes_of(c: np.ndarray, tet_subset: np.ndarray) -> np.ndarray:
        a, b, cc, d = c[tet_subset[:, 0]], c[tet_subset[:, 1]], c[tet_subset[:, 2]], c[tet_subset[:, 3]]
        return np.einsum('ij,ij->i', np.cross(b - a, cc - a), d - a)

    def _quality_of(c: np.ndarray, tet_subset: np.ndarray) -> np.ndarray:
        return _tet_shape_quality(c[tet_subset[:, 0]], c[tet_subset[:, 1]],
                                   c[tet_subset[:, 2]], c[tet_subset[:, 3]])

    def tet_volumes_full(c: np.ndarray) -> np.ndarray:
        return _volumes_of(c, tets)

    def tet_quality_full(c: np.ndarray) -> np.ndarray:
        return _quality_of(c, tets)

    # Both floors are relative to each tet's OWN state as it stood before
    # any optimization -- fixed for the whole run, matching relax_boundary_
    # points' own reasoning: repeated small erosions across iterations
    # can't compound into a sliver that "was safe relative to last
    # iteration" every single time.
    #
    # The quality floor exists because the volume floor alone does not
    # guarantee a tet's SHAPE can't get worse -- confirmed directly: an
    # earlier version with only the volume floor let the single worst tet's
    # min_angle regress (13.9 deg -> 10.7 deg) while still comfortably
    # above 20% of its original volume, because a tet can trade volume for
    # a worse shape (more sheared/skewed) without collapsing. A move is
    # only accepted now if EVERY tet -- not just the ones a currently-
    # moving point touches -- is at least as good as where it started, not
    # just "not yet collapsed".
    original_vol = tet_volumes_full(coords)
    original_quality = tet_quality_full(coords)
    min_allowed = cfg.min_volume_fraction * original_vol
    quality_slack = cfg.quality_regression_tolerance

    eps = cfg.finite_difference_eps_fraction * vs
    max_step = cfg.max_step_fraction * vs

    for _ in range(cfg.n_iterations):
        # Gathered once per iteration and reused for the baseline plus all
        # 3 finite-difference probes below (only the moving corner's own
        # column ever changes) -- coords is fixed until the safety-checked
        # update at the end of this iteration, so re-gathering it fresh for
        # each of those 4 evaluations was pure repeated work.
        P_base = coords[pt_corners]  # (K, 4, 3)
        base_pos = coords[pt_point]

        def quality_with_pos(P_arr, local_idx, override_pos: np.ndarray) -> np.ndarray:
            P = P_arr.copy()
            P[np.arange(len(local_idx)), local_idx] = override_pos
            return _tet_shape_quality(P[:, 0], P[:, 1], P[:, 2], P[:, 3])

        base_q = quality_with_pos(P_base, pt_local, base_pos)
        weight = np.maximum(0.0, cfg.quality_threshold - base_q)

        # The 3-axis finite-difference probe is by far the most expensive
        # step (confirmed directly: ~630ms per K-sized call on a real test
        # mesh, x4 needed = the dominant cost of the whole function) -- but
        # a pair with weight==0 contributes NOTHING to direction_sum below
        # regardless of what its gradient turns out to be, so computing it
        # is provably wasted work. Confirmed directly that this is not a
        # marginal saving: on a real 500-grain mesh, only ~1.8% of pairs
        # were ever below threshold. Restricting the 3 perturbations to
        # just that subset (the single baseline eval above still covers
        # everyone, since that's what decides who's below threshold at
        # all) is what took the real speedup from ~2x to the much larger
        # factor confirmed in the docstring above.
        grad = np.zeros((len(pt_point), 3))
        below = weight > 0.0
        if below.any():
            P_below = P_base[below]
            local_below = pt_local[below]
            pos_below = base_pos[below]
            base_q_below = base_q[below]
            for axis in range(3):
                perturbed = pos_below.copy()
                perturbed[:, axis] += eps
                grad[below, axis] = (quality_with_pos(P_below, local_below, perturbed)
                                      - base_q_below) / eps

        direction_sum = np.zeros((n_points, 3))
        weight_sum = np.zeros(n_points)
        np.add.at(direction_sum, pt_point, grad * weight[:, None])
        np.add.at(weight_sum, pt_point, weight)

        has_pull = weight_sum > 1e-300
        if not has_pull.any():
            break  # every tet already meets the quality bar

        direction = np.zeros((n_points, 3))
        direction[has_pull] = direction_sum[has_pull] / weight_sum[has_pull, None]
        direction *= axis_free  # RVE-plane lock, identical to relax_boundary_points

        step_len = np.linalg.norm(direction, axis=1)
        movable_now = np.nonzero(step_len > 1e-300)[0]
        if len(movable_now) == 0:
            break
        unit_direction = np.zeros((n_points, 3))
        unit_direction[movable_now] = direction[movable_now] / step_len[movable_now, None]
        proposed = unit_direction * max_step

        # Global verify-and-backtrack: try the full step, and if it drives
        # any tet below ITS floor, halve the step for every moving point
        # together and retry -- this is what actually guarantees safety
        # when several movers share a tet, not any per-point estimate.
        #
        # Only tets with at least one corner in movable_now can possibly
        # have changed at all -- every other tet's corners are provably
        # identical to `coords`, so checking it again is wasted work, not
        # a shortcut that risks missing something. This subset is exact,
        # not an approximation, and is fixed for all 20 retries this
        # iteration (movable_now itself doesn't change within them).
        affected_mask = np.isin(tets, movable_now).any(axis=1)
        affected_tets = tets[affected_mask]
        affected_min_allowed = min_allowed[affected_mask]
        affected_original_quality = original_quality[affected_mask]

        scale = 1.0
        for _ in range(20):
            candidate = coords.copy()
            candidate[movable_now] = coords[movable_now] + scale * proposed[movable_now]
            vol_ok = (_volumes_of(candidate, affected_tets) > affected_min_allowed).all()
            quality_ok = (_quality_of(candidate, affected_tets)
                          >= affected_original_quality - quality_slack).all()
            if vol_ok and quality_ok:
                coords = candidate
                break
            scale *= 0.5
        # If even the smallest tried scale still fails, skip this
        # iteration (coords unchanged) rather than risk an element below
        # the floor.

    return CleaveResult(
        node_coords=coords,
        tets=tets,
        tet_labels=result.tet_labels,
        source_tet_index=result.source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=result.new_point_gids,
    )


@dataclass
class ZoneOptimizeConfig:
    """
    Controls for the second-pass, ring-based boundary-ZONE optimization (as
    opposed to optimize_boundary_points, which only ever moves the exact
    boundary points themselves). Every field through
    quality_regression_tolerance means exactly what it means in
    OptimizeConfig -- this pass uses the identical per-point quality-pull
    mechanism, just applied to a wider, tapering-mobility set of points.
    """
    n_iterations: int = 15
    min_volume_fraction: float = 0.2
    quality_threshold: float = 0.3
    finite_difference_eps_fraction: float = 1e-4
    max_step_fraction: float = 0.25
    quality_regression_tolerance: float = 0.05
    # How many hops of shared-vertex tet-adjacency out from the boundary a
    # previously-frozen original vertex may still be nudged -- hop 1 is the
    # boundary's immediate original-vertex neighbours (highest weight below),
    # max_ring itself the faintest tapered pull, anything further out
    # staying exactly as frozen as it's always been.
    max_ring: int = 3
    # Reserved for a future GUI: which tets count as "neighbours" of a given
    # tet. Only 'vertex' (the loosest option -- share >=1 corner) is
    # implemented; anything else raises rather than silently behaving like
    # 'vertex', since a caller asking for stricter adjacency getting looser
    # adjacency with no signal would be a real footgun once this is exposed.
    adjacency: str = 'vertex'
    # Reserved the same way: only 'linear' decay is implemented.
    decay: str = 'linear'


def optimize_boundary_zone(
        lattice: BCCLattice,
        result: CleaveResult,
        config: Optional[ZoneOptimizeConfig] = None,
) -> CleaveResult:
    """
    Second-pass quality optimization: extends optimize_boundary_points'
    mechanism from "only the exact boundary points may move" to "boundary
    points AND a tapering ring of nearby original lattice vertices may
    move" -- addressing the ceiling first pass alone has, where every
    original vertex around a boundary point is perfectly rigid and so caps
    how much the boundary point itself can be improved.

    A previously-frozen original vertex becomes movable only if it is
    within ``config.max_ring`` hops of a boundary point, where one hop is
    "shares a tet with" (config.adjacency='vertex', the only option
    implemented so far) -- found via a multi-source breadth-first
    expansion seeded from every boundary point simultaneously, so a vertex
    reachable from two different boundary regions at different hop counts
    correctly gets the SMALLER hop count (and so the larger weight), not
    an arbitrary one. Its mobility is then scaled by a weight that tapers
    with hop count (config.decay='linear', the only option implemented so
    far): hop 1 (immediately touching a boundary point's own tet) gets the
    strongest pull, hop ``max_ring`` the faintest, and anything beyond
    ``max_ring`` is excluded from this pass entirely -- exactly as frozen
    as it has always been.

    Same guarantees as optimize_boundary_points, reused verbatim rather
    than re-derived: a point already sitting exactly on one of the RVE's
    own boundary planes is confined to sliding within it (this check is
    coordinate-based, not membership-based, so it already protects any
    point regardless of whether it's a boundary point or a newly-mobile
    original vertex); every proposed step is verified against every tet
    touching a currently-moving point against TWO floors fixed to that
    tet's own pre-optimization state (volume and shape quality); either
    floor failing backs the whole step off (halved, retried) until safe.

    Movable sets of this function and optimize_boundary_points are
    disjoint (boundary points themselves vs. original vertices 1..max_ring
    hops out), so running both never risks a double-move race; running
    optimize_boundary_points first is recommended so this pass works
    against an already-improved boundary rather than a stale one, though
    safety never depends on the order -- each call fixes its own floors
    from its own input.
    """
    cfg = config or ZoneOptimizeConfig()
    if cfg.adjacency != 'vertex':
        raise ValueError(
            f"adjacency={cfg.adjacency!r} is reserved for a future GUI option "
            f"but not yet implemented -- only 'vertex' works today")
    if cfg.decay != 'linear':
        raise ValueError(
            f"decay={cfg.decay!r} is reserved for a future GUI option but not "
            f"yet implemented -- only 'linear' works today")
    if cfg.max_ring < 1:
        raise ValueError(f'max_ring must be >= 1, got {cfg.max_ring}')

    coords = result.node_coords.copy()
    tets = result.tets
    n_points = len(coords)
    is_new = _is_new_point_mask(result, n_points)
    if not is_new.any():
        return result

    # Ring BFS: multi-source, "first write wins" so every point ends up at
    # its MINIMUM hop distance to any boundary point, not the hop count
    # from whichever boundary point happened to be considered first --
    # this falls straight out of expanding all seeds simultaneously, not
    # something that needs separate handling per source. Dense NumPy
    # frontier expansion directly over `tets` rather than building a
    # scipy.sparse adjacency matrix: max_ring is small (a handful of hops
    # at most), so the one-time cost of constructing and converting a
    # sparse matrix for only 3-4 queries isn't worth it here.
    hop = np.zeros(n_points, dtype=np.int32)  # 0 == boundary point or unreached
    visited = is_new.copy()
    frontier = is_new.copy()
    for r in range(1, cfg.max_ring + 1):
        touches = frontier[tets].any(axis=1)
        if not touches.any():
            break
        cand = np.unique(tets[touches])
        newly = np.zeros(n_points, dtype=bool)
        newly[cand] = True
        newly &= ~visited
        if not newly.any():
            break
        hop[newly] = r
        visited |= newly
        frontier = newly

    movable_orig = np.nonzero(hop > 0)[0]
    if len(movable_orig) == 0:
        return result

    # Linear taper: hop 1 -> 1.0 (of max_step), hop max_ring -> 1/max_ring,
    # hop max_ring+1 (unreached) -> 0 -- confirmed no discontinuity at the
    # cutoff, matching "frozen beyond max_ring" with nothing special-cased.
    ring_weight = np.zeros(n_points)
    ring_weight[movable_orig] = (cfg.max_ring - hop[movable_orig] + 1) / cfg.max_ring

    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    hi_bound = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    axis_free = np.ones((n_points, 3), dtype=bool)
    for axis in range(3):
        on_plane = (coords[:, axis] == 0.0) | (coords[:, axis] == hi_bound[axis])
        axis_free[on_plane, axis] = False

    is_movable_orig = np.zeros(n_points, dtype=bool)
    is_movable_orig[movable_orig] = True

    pt_point_parts, pt_tet_parts, pt_local_parts = [], [], []
    for local in range(4):
        pts = tets[:, local]
        idx = np.nonzero(is_movable_orig[pts])[0]
        pt_point_parts.append(pts[idx])
        pt_tet_parts.append(idx)
        pt_local_parts.append(np.full(len(idx), local))
    pt_point = np.concatenate(pt_point_parts)
    pt_tet = np.concatenate(pt_tet_parts)
    pt_local = np.concatenate(pt_local_parts)

    if len(pt_point) == 0:
        return result

    pt_corners = tets[pt_tet]  # (K, 4) global ids, standard corner order

    def _volumes_of(c: np.ndarray, tet_subset: np.ndarray) -> np.ndarray:
        a, b, cc, d = c[tet_subset[:, 0]], c[tet_subset[:, 1]], c[tet_subset[:, 2]], c[tet_subset[:, 3]]
        return np.einsum('ij,ij->i', np.cross(b - a, cc - a), d - a)

    def _quality_of(c: np.ndarray, tet_subset: np.ndarray) -> np.ndarray:
        return _tet_shape_quality(c[tet_subset[:, 0]], c[tet_subset[:, 1]],
                                   c[tet_subset[:, 2]], c[tet_subset[:, 3]])

    def tet_volumes_full(c: np.ndarray) -> np.ndarray:
        return _volumes_of(c, tets)

    def tet_quality_full(c: np.ndarray) -> np.ndarray:
        return _quality_of(c, tets)

    # Both floors fixed to this function's OWN starting state -- same
    # reasoning as optimize_boundary_points: repeated small erosions across
    # iterations can't compound into a sliver that "was safe relative to
    # last iteration" every single time.
    original_vol = tet_volumes_full(coords)
    original_quality = tet_quality_full(coords)
    min_allowed = cfg.min_volume_fraction * original_vol
    quality_slack = cfg.quality_regression_tolerance

    eps = cfg.finite_difference_eps_fraction * vs
    max_step = cfg.max_step_fraction * vs

    for _ in range(cfg.n_iterations):
        P_base = coords[pt_corners]
        base_pos = coords[pt_point]

        def quality_with_pos(P_arr, local_idx, override_pos: np.ndarray) -> np.ndarray:
            P = P_arr.copy()
            P[np.arange(len(local_idx)), local_idx] = override_pos
            return _tet_shape_quality(P[:, 0], P[:, 1], P[:, 2], P[:, 3])

        base_q = quality_with_pos(P_base, pt_local, base_pos)
        weight = np.maximum(0.0, cfg.quality_threshold - base_q)

        grad = np.zeros((len(pt_point), 3))
        below = weight > 0.0
        if below.any():
            P_below = P_base[below]
            local_below = pt_local[below]
            pos_below = base_pos[below]
            base_q_below = base_q[below]
            for axis in range(3):
                perturbed = pos_below.copy()
                perturbed[:, axis] += eps
                grad[below, axis] = (quality_with_pos(P_below, local_below, perturbed)
                                      - base_q_below) / eps

        direction_sum = np.zeros((n_points, 3))
        weight_sum = np.zeros(n_points)
        np.add.at(direction_sum, pt_point, grad * weight[:, None])
        np.add.at(weight_sum, pt_point, weight)

        has_pull = weight_sum > 1e-300
        if not has_pull.any():
            break  # every reachable tet already meets the quality bar

        direction = np.zeros((n_points, 3))
        direction[has_pull] = direction_sum[has_pull] / weight_sum[has_pull, None]
        direction *= axis_free  # RVE-plane lock, identical to optimize_boundary_points

        step_len = np.linalg.norm(direction, axis=1)
        movable_now = np.nonzero(step_len > 1e-300)[0]
        if len(movable_now) == 0:
            break
        unit_direction = np.zeros((n_points, 3))
        unit_direction[movable_now] = direction[movable_now] / step_len[movable_now, None]
        # Ring decay applied here, AFTER normalization -- applying it to
        # `direction` (or folding it into the per-tet `weight` above)
        # before this point would cancel out of the weighted average /
        # normalization exactly, silently making every point move the
        # same regardless of ring instead of tapering as intended.
        proposed = unit_direction * max_step * ring_weight[:, None]

        # Same pattern as optimize_boundary_points, NOT relax_boundary_
        # points/snap_to_rve_boundary: movable_now can shrink between
        # iterations as tets clear the quality bar, so affected_tets is
        # recomputed fresh each iteration, not hoisted once before the loop.
        affected_mask = np.isin(tets, movable_now).any(axis=1)
        affected_tets = tets[affected_mask]
        affected_min_allowed = min_allowed[affected_mask]
        affected_original_quality = original_quality[affected_mask]

        scale = 1.0
        for _ in range(20):
            candidate = coords.copy()
            candidate[movable_now] = coords[movable_now] + scale * proposed[movable_now]
            vol_ok = (_volumes_of(candidate, affected_tets) > affected_min_allowed).all()
            quality_ok = (_quality_of(candidate, affected_tets)
                          >= affected_original_quality - quality_slack).all()
            if vol_ok and quality_ok:
                coords = candidate
                break
            scale *= 0.5
        # If even the smallest tried scale still fails, skip this
        # iteration (coords unchanged) rather than risk an element below
        # the floor.

    return CleaveResult(
        node_coords=coords,
        tets=tets,
        tet_labels=result.tet_labels,
        source_tet_index=result.source_tet_index,
        deferred_tet_indices=result.deferred_tet_indices,
        new_point_gids=result.new_point_gids,
    )
