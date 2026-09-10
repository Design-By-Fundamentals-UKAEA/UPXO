"""
cleave.py
=========
Orchestrates stencils.cleave_tet across a whole BCC lattice: maintains the
shared cut/triple/quadruple-point cache (so two tets sharing a cut edge or
a triple-labelled face always get the exact same new vertex, not two
independently-computed near-duplicates -- a quadruple point is tet-private
and never needs this, since two tets never share all 4 corners), and
resolves the tagged Points from stencils.py into a final concrete mesh.

stencils.cleave_tet handles every possible corner-label pattern a real
tetrahedron can have (1 to 4 distinct labels), so ``deferred_tet_indices``
should always come back empty in normal operation; it exists only as a
defensive guard against stencils.TooManyLabelsError, which is otherwise
unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from upxo.meshing.cleaving.lattice import BCCLattice, signed_volumes
from upxo.meshing.cleaving.stencils import (
    Point, TooManyLabelsError, cleave_tet,
)


@dataclass
class CleaveResult:
    """
    node_coords         : (N, 3) float64 -- original lattice vertices
                          followed by newly-created cut points
    tets                : (T, 4) int64   -- 0-based indices into node_coords
    tet_labels          : (T,) int64     -- grain label of each output tet
    source_tet_index    : (T,) int64    -- which original lattice tet each
                          output tet came from (many-to-one for cut tets)
    deferred_tet_indices: should always be empty in normal operation -- see
                          module docstring
    new_point_gids      : {node_id: (original_vertex_gid, ...)} -- for every
                          node NOT an original lattice vertex, the 2, 3, or
                          4 original vertex ids it was derived from (an
                          edge midpoint, face centroid, or tet centroid
                          respectively). Lets downstream code (rve_cap.py)
                          reason about EXACTLY which original vertices a
                          new point sits between, rather than guessing from
                          its resolved coordinate -- e.g. "is one of this
                          cut point's two defining vertices exactly on the
                          RVE boundary" is an exact fact, not a tolerance.
    """
    node_coords: np.ndarray
    tets: np.ndarray
    tet_labels: np.ndarray
    source_tet_index: np.ndarray
    deferred_tet_indices: np.ndarray
    new_point_gids: Dict[int, Tuple[int, ...]] = field(default_factory=dict)


class _CutPointCache:
    """
    Maps a new point (a cut point's defining edge, a face triple point's
    defining face, or a tet's quadruple point's defining 4 corners) to a
    stable new output vertex id.

    The dict key is the point tuple itself (e.g. ('c', 3, 7) or
    ('t', 2, 5, 9)) rather than a stripped-down (id_a, id_b) pair -- using
    only a fixed prefix of numeric fields would truncate a longer point's
    trailing ids and could collide two different points (or a longer point
    with an unrelated shorter one) that happen to share a common prefix.
    Placement is generic for the same reason: the centroid of however many
    ids the point carries (edge midpoint for 2, face centroid for 3, tet
    centroid for 4) -- adding a still-longer point type later needs no
    change here, unlike a per-type-length branch would.
    """

    def __init__(self, base_vertices: np.ndarray):
        self._base = base_vertices
        # Plain Python floats for resolve()'s hot path -- confirmed
        # directly that numpy's per-call dispatch overhead dominates
        # self._base[list(defining_ids)].mean(axis=0) for an array this
        # tiny (2-4 rows): ~8.4 microseconds per call, almost none of it
        # the actual arithmetic. A one-time conversion here trades a
        # single O(len(base_vertices)) cost for removing that overhead
        # from what can be millions of calls.
        self._base_list = base_vertices.tolist()
        self._n_base = len(base_vertices)
        self._id_of: Dict[Point, int] = {}
        self._coords: List[Tuple[float, float, float]] = []
        self._defining_gids: List[Tuple[int, ...]] = []

    def resolve(self, point: Point) -> int:
        if point[0] == 'v':
            return point[1]
        vid = self._id_of.get(point)
        if vid is None:
            vid = self._n_base + len(self._coords)
            self._id_of[point] = vid
            # Cut/triple/quadruple-point placement: centroid of the ids the
            # point is defined by. Topological-correctness milestone --
            # warping toward the true sub-voxel boundary location is a
            # later quality refinement, not a change to which tets result.
            defining_ids = point[1:]
            n = len(defining_ids)
            x = y = z = 0.0
            for gid in defining_ids:
                row = self._base_list[gid]
                x += row[0]
                y += row[1]
                z += row[2]
            self._coords.append((x / n, y / n, z / n))
            self._defining_gids.append(tuple(defining_ids))
        return vid

    @property
    def all_coords(self) -> np.ndarray:
        if not self._coords:
            return self._base.copy()
        return np.vstack([self._base, np.array(self._coords, dtype=np.float64)])

    @property
    def new_point_gids(self) -> Dict[int, Tuple[int, ...]]:
        return {self._n_base + i: g for i, g in enumerate(self._defining_gids)}


def cleave_lattice(lattice: BCCLattice, labels: np.ndarray) -> CleaveResult:
    """
    Cleave every tet in ``lattice`` according to ``labels`` (one label per
    lattice vertex, as produced by labeling.sample_labels). Every tet is
    cleaved via stencils.cleave_tet, which handles all 4 possible corner-
    label-count cases; ``deferred_tet_indices`` is a defensive fallback,
    not an expected occurrence -- see module docstring.
    """
    cache = _CutPointCache(lattice.vertices)

    out_tets: List[Tuple[int, int, int, int]] = []
    out_labels: List[int] = []
    out_source: List[int] = []
    deferred: List[int] = []

    # Plain Python lists for cleave_tet's hot path -- same reasoning as
    # _CutPointCache._base_list above: numpy fancy-indexing (labels[tet])
    # and per-element numpy-scalar handling both carry per-call dispatch
    # overhead that a one-time .tolist() avoids for every one of the
    # lattice's tets.
    tets_list = lattice.tets.tolist()
    labels_list = labels.tolist()

    for t_idx, tet in enumerate(tets_list):
        corner_labels = [labels_list[g] for g in tet]
        try:
            by_label = cleave_tet(tet, corner_labels)
        except TooManyLabelsError:
            deferred.append(t_idx)
            continue

        for lab, sub_tets in by_label.items():
            for pts in sub_tets:
                out_tets.append(tuple(cache.resolve(p) for p in pts))
                out_labels.append(lab)
                out_source.append(t_idx)

    node_coords = cache.all_coords
    tets = np.array(out_tets, dtype=np.int64).reshape(-1, 4)
    tets = _fix_winding(node_coords, tets)
    node_coords, tets, new_point_gids = _drop_unreferenced_nodes(
        node_coords, tets, cache.new_point_gids)

    return CleaveResult(
        node_coords=node_coords,
        tets=tets,
        tet_labels=np.array(out_labels, dtype=np.int64),
        source_tet_index=np.array(out_source, dtype=np.int64),
        deferred_tet_indices=np.array(deferred, dtype=np.int64),
        new_point_gids=new_point_gids,
    )


def _drop_unreferenced_nodes(node_coords: np.ndarray, tets: np.ndarray,
                              new_point_gids: Dict[int, Tuple[int, ...]]):
    """
    Compact away any node no tet references. In practice this is exactly
    the padded lattice's own 8 outer corners: a true 3-way corner of the
    padding compounds the (already-expected -- see lattice.py) boundary
    support shortfall in all 3 axes simultaneously, so those 8 vertices get
    zero tet support -- confirmed by direct inspection, not just a count.
    They carry no geometry either way (nothing ever pointed at them); this
    just keeps them from shipping in ``node_coords`` for no reason.
    """
    if len(tets) == 0:
        return node_coords, tets, new_point_gids
    used = np.unique(tets.ravel())
    if len(used) == len(node_coords):
        return node_coords, tets, new_point_gids
    remap = np.full(len(node_coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    remapped_gids = {int(remap[old_id]): gids
                     for old_id, gids in new_point_gids.items()
                     if remap[old_id] >= 0}
    return node_coords[used], remap[tets], remapped_gids


def _fix_winding(node_coords: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Swap the last two vertices of any tet with non-positive signed volume."""
    if len(tets) == 0:
        return tets
    vol = signed_volumes(node_coords, tets)
    bad = vol <= 0
    if np.any(bad):
        tets = tets.copy()
        tets[bad, 2], tets[bad, 3] = tets[bad, 3], tets[bad, 2]
    return tets
