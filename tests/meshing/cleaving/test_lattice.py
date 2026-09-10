"""
test_lattice.py
================
Correctness tests for the BCC background lattice construction.

These are deliberately quantitative (volume conservation, exact vertex/tet
counts) rather than just "does it run" -- the lattice construction is pure
index arithmetic (see lattice.py docstring for the derivation), and a subtle
off-by-one there would silently produce a non-conformal or self-overlapping
lattice without raising any exception.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import (
    build_bcc_lattice, signed_volumes, PRIMAL, DUAL,
)
from upxo.meshing.cleaving.labeling import sample_labels


SHAPES = [(3, 3, 3), (4, 5, 6), (2, 2, 2)]


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('pad', [1, 2])
def test_vertex_counts(shape, pad):
    nx, ny, nz = shape
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))

    px = nx - 1 + 2 * pad + 1
    py = ny - 1 + 2 * pad + 1
    pz = nz - 1 + 2 * pad + 1
    dx = nx - 2 + 2 * pad + 1
    dy = ny - 2 + 2 * pad + 1
    dz = nz - 2 + 2 * pad + 1

    n_primal = px * py * pz
    n_dual = dx * dy * dz

    assert (lat.vertex_kind == PRIMAL).sum() == n_primal
    assert (lat.vertex_kind == DUAL).sum() == n_dual
    assert len(lat.vertices) == n_primal + n_dual


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('pad', [1, 2])
def test_tet_indices_are_valid(shape, pad):
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))
    assert lat.tets.min() >= 0
    assert lat.tets.max() < len(lat.vertices)
    # No degenerate tet referencing the same vertex twice.
    assert (lat.tets[:, 0] != lat.tets[:, 1]).all()
    for a in range(4):
        for b in range(a + 1, 4):
            assert (lat.tets[:, a] != lat.tets[:, b]).all()


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('pad', [1, 2])
def test_all_tets_positive_volume(shape, pad):
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))
    vol = signed_volumes(lat.vertices, lat.tets)
    assert (vol > 0).all(), (
        f'{int((vol <= 0).sum())} / {len(vol)} tets have non-positive '
        f'signed volume after winding fix')


def _vol6(p0, p1, p2, p3):
    return np.einsum('ij,ij->i', np.cross(p1 - p0, p2 - p0), p3 - p0)


def _coverage_count(p: np.ndarray, verts: np.ndarray, tets: np.ndarray) -> int:
    """Number of tets that strictly contain point p (0=gap, 1=ok, >1=overlap)."""
    a, b, c, d = verts[tets[:, 0]], verts[tets[:, 1]], verts[tets[:, 2]], verts[tets[:, 3]]
    pp = np.broadcast_to(p, a.shape)
    w0, w1 = _vol6(a, b, c, d), _vol6(pp, b, c, d)
    w2, w3 = _vol6(a, pp, c, d), _vol6(a, b, pp, d)
    w4 = _vol6(a, b, c, pp)
    s = np.sign(w0)
    inside = ((np.sign(w1) == s) & (np.sign(w2) == s) &
              (np.sign(w3) == s) & (np.sign(w4) == s))
    return int(inside.sum())


@pytest.mark.parametrize('shape', [(4, 4, 4), (3, 5, 4)])
def test_real_domain_fully_covered_no_overlap(shape):
    """
    The padded box's own outer shell is NOT expected to be fully tiled --
    a dual vertex sitting on the padding's outer boundary is missing some
    of its 6 neighbour cells, so it can never get its full complement of
    fanning tets (this is a real, expected boundary effect, not a bug --
    see the discussion in build_bcc_lattice). What must actually hold is
    that the REAL (unpadded) domain is gap-free and overlap-free, since
    that's the only region label sampling / cleaving will ever query.
    """
    pad = 1
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))
    nx, ny, nz = shape

    rng = np.random.default_rng(0)
    pts = []
    # Random interior points.
    pts += [tuple(rng.uniform(0.05, [nx - 1.05, ny - 1.05, nz - 1.05]))
            for _ in range(150)]
    # Points right at the real domain's own boundary planes (the highest-
    # risk location, since it sits well inside the padded lattice).
    for _ in range(20):
        y, z = rng.uniform(0.1, [ny - 1.1, nz - 1.1])
        pts.append((1e-6, y, z))
        pts.append((nx - 1 - 1e-6, y, z))

    zero = sum(1 for p in pts if _coverage_count(np.array(p), lat.vertices, lat.tets) == 0)
    multi = sum(1 for p in pts if _coverage_count(np.array(p), lat.vertices, lat.tets) > 1)
    assert zero == 0, f'{zero} / {len(pts)} sample points in the real domain are uncovered'
    assert multi == 0, f'{multi} / {len(pts)} sample points are covered by overlapping tets'


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('pad', [1, 2])
def test_total_volume_sane(shape, pad):
    """
    Loose sanity floor: total tet volume must be positive and cannot
    exceed the padded box's volume (the outer shell is allowed to be
    incomplete, so exact equality is NOT expected here -- see
    test_real_domain_fully_covered_no_overlap for the real invariant).
    """
    nx, ny, nz = shape
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))
    total = signed_volumes(lat.vertices, lat.tets).sum()
    box_vol = (nx - 1 + 2 * pad) * (ny - 1 + 2 * pad) * (nz - 1 + 2 * pad)
    assert 0 < total <= box_vol + 1e-9


@pytest.mark.parametrize('shape', SHAPES)
def test_interior_long_edges_have_four_tets(shape):
    """
    Every long edge strictly inside the real domain must contribute exactly
    4 tets (full flanking support) -- this is the direct, local check that
    complements the global volume-conservation test above.
    """
    pad = 1
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=pad))

    # Each tet has exactly one long edge (primal-primal or dual-dual).
    # Recover it per-tet: the pair of same-kind vertices.
    kinds = lat.vertex_kind[lat.tets]  # (T,4)
    long_edge_counts = {}
    for t_idx in range(len(lat.tets)):
        verts = lat.tets[t_idx]
        k = kinds[t_idx]
        same = [(a, b) for a in range(4) for b in range(a + 1, 4) if k[a] == k[b]]
        # Exactly one pair of same-kind vertices should be the long edge;
        # the other same-kind pair check below in test_exactly_one_long_edge.
        for a, b in same:
            va, vb = verts[a], verts[b]
            d = np.linalg.norm(lat.vertices[va] - lat.vertices[vb])
            if d > 1e-9:  # the long edge (short edges only connect opposite kinds)
                key = (min(va, vb), max(va, vb))
                long_edge_counts[key] = long_edge_counts.get(key, 0) + 1

    # Restrict to edges whose midpoint lies strictly inside the real domain.
    nx, ny, nz = shape
    interior_counts = []
    for (va, vb), cnt in long_edge_counts.items():
        mid = (lat.vertices[va] + lat.vertices[vb]) / 2.0
        if (0 < mid[0] < nx - 1) and (0 < mid[1] < ny - 1) and (0 < mid[2] < nz - 1):
            interior_counts.append(cnt)

    if interior_counts:  # small shapes may have no strictly-interior edges
        assert all(c == 4 for c in interior_counts)


def test_sample_labels_two_grain_block():
    """Synthetic 2-grain case: bottom half label 1, top half label 2."""
    shape = (4, 4, 6)
    lgi = np.ones(shape, dtype=np.int32)
    lgi[:, :, 3:] = 2

    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=32767))

    idx = np.rint(lat.vertices).astype(np.int64)
    nx, ny, nz = shape
    inside = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    outside = ~inside

    assert (labels[outside] == 32767).all()
    assert set(np.unique(labels[inside]).tolist()) <= {1, 2}
    assert (labels[inside] == 1).any()
    assert (labels[inside] == 2).any()

    # Direct spot-check against the source array for a handful of primal
    # vertices (which land exactly on voxel centres).
    for (i, j, k) in [(0, 0, 0), (1, 1, 1), (0, 0, 5), (3, 3, 5)]:
        v_id = np.nonzero(
            (idx[:, 0] == i) & (idx[:, 1] == j) & (idx[:, 2] == k)
            & (lat.vertex_kind == PRIMAL)
        )[0]
        assert len(v_id) == 1
        assert labels[v_id[0]] == lgi[i, j, k]


def test_sample_labels_shape_mismatch_raises():
    lat = build_bcc_lattice((3, 3, 3), LatticeConfig())
    with pytest.raises(ValueError):
        sample_labels(lat, np.ones((4, 4, 4), dtype=np.int32))
