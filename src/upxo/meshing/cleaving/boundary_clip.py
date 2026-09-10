"""
boundary_clip.py
=================
Exact-geometry replacement for relax.py/rve_cap.py's approximate-then-patch
strategy for RVE outer-face flatness.

Why a replacement, not another patch: rve_cap.py tried to fix wobble AFTER
cleaving, by merging a wobbly cut point onto an exact anchor and deleting
whatever tet became degenerate as a result. Direct testing showed every
single candidate merge -- not most, all of them, on every grain structure
tried -- leaves at least one NEIGHBOURING tet with bad (inverted, not just
near-zero) volume, even completely alone with no other merge active. That
tet was fan/cone-triangulated around the cut point's WOBBLY position; once
that corner reaches its true flat position, the tet's shape isn't merely
imprecise, it's the wrong tet entirely. No amount of smarter merge
selection fixes that, because there is no conflict to resolve -- the local
topology itself needs to change, not just one corner's coordinate.

The fix here sidesteps that failure mode by never creating the wobble in
the first place. The RVE's outer box is not like an interior grain
boundary: its position isn't inferred from noisy voxel labels at all, it's
exact by definition -- the box IS the domain. So instead of cleaving
"grain label" vs "wall label" through the same generic multi-label stencil
used for grain-vs-grain (which only ever KNOWS about labels, not
geometry, and so can only place a cut point at a centroid), this module
clips each boundary-straddling lattice tet directly against the domain's
known-exact planes, computing the true line-plane intersection. Grain
labels never enter into it.

This reuses stencils.cleave_tet completely unchanged: with only 2 distinct
"labels" (inside the real domain vs outside it), cleave_tet's existing
fan-triangulation already produces exactly the marching-tetrahedra-style
1-vs-3 / 2-vs-2 sub-tet decomposition this needs (verified independently
for the grain case; the topology doesn't care what the 2 labels MEAN).
The only new piece is _BoxClipCache, a drop-in analogue of cleave.py's
_CutPointCache that places a cut point at the exact segment-plane
intersection instead of a centroid -- and forces the clipped axis to the
plane's exact value afterward, rather than trusting interpolation
arithmetic to reproduce it bit-for-bit.

Once every boundary-straddling tet is clipped this way, the result exactly
fills the real domain box (verified via volume conservation against the
box's own analytic volume) with brand new points sitting exactly on
whichever face(s) they cross -- multiple planes at once near an RVE edge
or corner are handled by clipping sequentially, one plane at a time, which
is valid because half-space intersection doesn't depend on the order the
half-spaces are applied in.

Downstream, grain labels are sampled onto this already-exactly-flat mesh
(including its new boundary points -- sample_labels' ordinary nearest-
voxel lookup works on them unchanged, since they're just points with real
coordinates) and cleaved by the ordinary grain-label path exactly as
before. That second pass can still place its own cut/triple/quadruple
points at plain centroids -- the ordinary, accepted interior-boundary
approximation -- because whenever a grain boundary happens to fall AT the
RVE surface, every point it's a centroid of is already exactly on that
plane, so the centroid is too. Flatness therefore never depends on the
grain-cleave step at all; it's guaranteed entirely by the geometric clip
that runs before it.

One consequence worth being explicit about: there is no more "wall" region
to strip after this. Every vertex fed to the grain-cleave pass is either
an original in-domain lattice vertex or a new point exactly on the
domain's own boundary -- nothing is ever outside it -- so wall_label
should never be reachable downstream of clip_lattice_to_domain, and
relax.snap_to_rve_boundary / rve_cap.py's merge-and-prune step are not
needed for RVE-flatness purposes at all when this module is used.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from upxo.meshing.cleaving.config import LabelingConfig
from upxo.meshing.cleaving.lattice import BCCLattice, PRIMAL, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.stencils import cleave_tet
from upxo.meshing.cleaving.cleave import cleave_lattice

_TOL = 1e-9
_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# (axis, direction, is_lo) -- processed one plane at a time; order doesn't
# affect the final result, only the intermediate decomposition, since a
# half-space intersection is order-independent.
_PLANE_AXES_DIRECTIONS = ((0, True), (0, False), (1, True), (1, False),
                          (2, True), (2, False))


def _domain_bounds(lattice: BCCLattice) -> Tuple[np.ndarray, np.ndarray]:
    vs = lattice.voxel_size
    nx, ny, nz = lattice.grid_shape
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([(nx - 1) * vs, (ny - 1) * vs, (nz - 1) * vs])
    return lo, hi


def _inside_mask(coords: np.ndarray, axis: int, is_lo: bool, plane_value: float) -> np.ndarray:
    """Inclusive: a coordinate exactly ON the plane counts as inside -- the
    plane itself belongs to the real domain, matching sample_labels' own
    inclusive idx-in-range convention."""
    if is_lo:
        return coords[:, axis] >= plane_value - _TOL
    return coords[:, axis] <= plane_value + _TOL


class _BoxClipCache:
    """
    Persists new boundary points across the WHOLE clip pass (all
    straddling tets, all planes) so two original tets sharing an edge that
    straddles the same plane produce the identical new point -- the same
    cross-tet-consistency concern cleave.py's _CutPointCache exists for,
    just with a different placement rule: the exact line-plane
    intersection, not a centroid.
    """

    def __init__(self, base_vertices: np.ndarray):
        self._base = base_vertices
        self._n_base = len(base_vertices)
        self._coords: List[np.ndarray] = []
        self._id_of: Dict[Tuple, int] = {}
        self._provenance: List[Tuple[int, int]] = []

    def coord_of(self, gid: int) -> np.ndarray:
        if gid < self._n_base:
            return self._base[gid]
        return self._coords[gid - self._n_base]

    def resolve(self, point, axis: int, is_lo: bool, plane_value: float) -> int:
        if point[0] == 'v':
            return point[1]
        cached = self._id_of.get(point)
        if cached is not None:
            return cached

        a, b = point[1], point[2]
        ca, cb = self.coord_of(a), self.coord_of(b)
        a_inside = _inside_mask(ca[None, :], axis, is_lo, plane_value)[0]
        gid_in, v_in, gid_out, v_out = (a, ca, b, cb) if a_inside else (b, cb, a, ca)

        if abs(v_in[axis] - plane_value) <= _TOL:
            # v_in is already exactly on this plane -- the true crossing
            # of segment(v_in, v_out) with it IS v_in, not a new point.
            # (This is precisely the "topological wall" rve_cap.py's
            # docstring identified: merging onto v_in after the fact
            # broke neighbouring tets built around a DIFFERENT wobbly
            # position; resolving it correctly here, before any tet
            # referencing this edge is even built, avoids that entirely.)
            self._id_of[point] = int(gid_in)
            return int(gid_in)

        t = (plane_value - v_in[axis]) / (v_out[axis] - v_in[axis])
        new_coord = v_in + t * (v_out - v_in)
        new_coord[axis] = plane_value  # exact, not left to interpolation roundoff
        new_gid = self._n_base + len(self._coords)
        self._id_of[point] = new_gid
        self._coords.append(new_coord)
        self._provenance.append((int(gid_in), int(gid_out)))
        return new_gid

    @property
    def all_coords(self) -> np.ndarray:
        if not self._coords:
            return self._base.copy()
        return np.vstack([self._base, np.array(self._coords, dtype=np.float64)])

    @property
    def provenance(self) -> Dict[int, Tuple[int, int]]:
        """{new_gid: (inside_gid, outside_gid)} for every new boundary point."""
        return {self._n_base + i: g for i, g in enumerate(self._provenance)}


def _drop_degenerate(tets: np.ndarray) -> np.ndarray:
    """
    Drop tets with a repeated vertex -- always exactly zero volume, never
    a symptom of lost geometry. Arises whenever a clipped tet's ONLY
    inside corner is itself already exactly on the plane being clipped:
    the other 3 corners are then all strictly beyond it, so the true
    inside portion is that single point, correctly zero volume (confirmed
    directly: dropping these tets never changes the total volume, to
    float precision). _BoxClipCache.resolve's early return for a cut
    point whose "inside" endpoint already sits on the plane is what
    produces them -- collapsing 1, 2, or 3 of a sub-tet's 4 corners onto
    that same vertex, depending on how many of its edges hit this case.
    """
    if len(tets) == 0:
        return tets
    degenerate = np.zeros(len(tets), dtype=bool)
    for a, b in _EDGE_PAIRS:
        degenerate |= tets[:, a] == tets[:, b]
    return tets[~degenerate]


def _fix_winding(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    if len(tets) == 0:
        return tets
    vol = signed_volumes(vertices, tets)
    bad = vol <= 0
    if np.any(bad):
        tets = tets.copy()
        tets[bad, 2], tets[bad, 3] = tets[bad, 3], tets[bad, 2]
    return tets


def clip_lattice_to_domain(lattice: BCCLattice) -> Tuple[BCCLattice, Dict[int, Tuple[int, int]]]:
    """
    Discard everything outside ``lattice``'s real (unpadded) domain,
    clipping straddling tets EXACTLY against the domain's 6 axis-aligned
    planes -- never approximated from label data, since the box shape is
    exact by definition. Returns a new BCCLattice whose tets exactly fill
    the real domain box, with no wobble on any outer face, plus a
    provenance dict for the new boundary points it had to create
    ({new_gid: (inside_gid, outside_gid)}).

    Grain labels are not considered here -- this is pure box geometry, run
    BEFORE labeling.sample_labels / cleave.cleave_lattice. Feed the
    returned lattice through those exactly as usual for the grain-vs-grain
    interior boundaries; those stay approximate, which is expected and
    fine (see relax.py) -- only the RVE's own outer shape is a hard
    requirement.
    """
    lo, hi = _domain_bounds(lattice)
    base_vertices = lattice.vertices
    tets = lattice.tets

    all_coords = base_vertices[tets]  # (T, 4, 3)
    outside = (all_coords < lo - _TOL) | (all_coords > hi + _TOL)  # (T, 4, 3)
    needs_clip = outside.any(axis=(1, 2))

    kept_tets = [tuple(int(g) for g in row) for row in tets[~needs_clip]]
    cache = _BoxClipCache(base_vertices)

    for tet in tets[needs_clip]:
        current = [tuple(int(g) for g in tet)]
        for axis, is_lo in _PLANE_AXES_DIRECTIONS:
            plane_value = lo[axis] if is_lo else hi[axis]
            next_current = []
            for t in current:
                coords = np.array([cache.coord_of(g) for g in t])
                inside = _inside_mask(coords, axis, is_lo, plane_value)
                if inside.all():
                    next_current.append(t)
                    continue
                if not inside.any():
                    continue  # entirely beyond this plane -- discard
                by_label = cleave_tet(t, inside.astype(np.int64))
                for pts in by_label.get(1, []):
                    resolved = tuple(cache.resolve(p, axis, is_lo, plane_value) for p in pts)
                    next_current.append(resolved)
            current = next_current
        kept_tets.extend(current)

    out_tets = np.array(kept_tets, dtype=np.int64).reshape(-1, 4)
    out_vertices = cache.all_coords
    out_tets = _drop_degenerate(out_tets)
    out_tets = _fix_winding(out_vertices, out_tets)

    n_new = len(out_vertices) - len(lattice.vertex_kind)
    vertex_kind = np.concatenate([
        lattice.vertex_kind,
        np.full(n_new, PRIMAL, dtype=np.int8),
    ])

    clipped = BCCLattice(
        vertices=out_vertices,
        vertex_kind=vertex_kind,
        tets=out_tets,
        voxel_size=lattice.voxel_size,
        grid_shape=lattice.grid_shape,
        pad=lattice.pad,
    )
    return clipped, cache.provenance


def cleave_lattice_exact_boundary(lattice: BCCLattice, lgi: np.ndarray, labeling_config=None):
    """
    Convenience one-shot: clip to the exact domain box, sample grain labels
    onto the result (including its new boundary points), then cleave by
    grain label. Equivalent to calling clip_lattice_to_domain,
    labeling.sample_labels, and cleave.cleave_lattice in sequence -- kept
    as 3 separately callable/testable pieces; this just chains them for the
    common case. Unlike the old build_bcc_lattice+sample_labels+
    cleave_lattice+cap_rve_boundary_exactly pipeline, there is no wall
    label to strip afterward -- see module docstring.
    """
    clipped, provenance = clip_lattice_to_domain(lattice)
    cfg = labeling_config or LabelingConfig()
    labels = sample_labels(clipped, lgi, cfg)
    result = cleave_lattice(clipped, labels)
    return result, clipped, provenance
