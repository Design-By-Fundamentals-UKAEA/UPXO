"""
test_boundary_clip.py
======================
Correctness tests for boundary_clip.py, the exact-geometry replacement for
relax.py/rve_cap.py's approximate-then-patch strategy for RVE outer-face
flatness (see boundary_clip.py's module docstring for why the patch
approach was abandoned: every one of 1359 candidate merges tried across 3
grain structures broke a neighbouring tet, individually, with nothing else
active -- not a conflict a smarter merge selection could resolve).

Two layers are tested separately, mirroring the module's own 2-pass split:

1. clip_lattice_to_domain alone -- pure box geometry, no grain labels.
   Exact volume conservation against the domain's own analytic volume,
   all-positive tets, nothing ever left outside the domain, and cross-tet
   consistency of the new boundary points it creates.

2. cleave_lattice_exact_boundary (box-clip + grain-label cleave) -- the
   actual CPFEM guarantee: every exterior face of the final mesh is
   EXACTLY (bit-exact, not tolerance-based) flat on the RVE's own 6 faces,
   checked across ordinary grain structures AND the specific edge cases
   the project flagged as needing explicit support: a grain boundary
   running exactly along an RVE edge, and a single voxel occupying
   exactly one RVE corner.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.boundary_clip import (
    clip_lattice_to_domain, cleave_lattice_exact_boundary,
)

WALL = 32767
_FACE_DEFS = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _grouped_faces(tets):
    faces = np.concatenate([tets[:, list(idx)] for idx in _FACE_DEFS], axis=0)
    tet_of_face = np.tile(np.arange(len(tets)), 4)
    sorted_faces = np.sort(faces, axis=1)
    order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sf = sorted_faces[order]
    tf = tet_of_face[order]
    boundaries = np.nonzero(np.any(sf[1:] != sf[:-1], axis=1))[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(sf)]])
    return sf, tf, starts, ends


def _n_connected_components(tets, mask=None):
    if mask is not None:
        tets = tets[mask]
    if len(tets) == 0:
        return 0
    sf, tf, starts, ends = _grouped_faces(tets)
    shared = np.nonzero((ends - starts) == 2)[0]
    rows = tf[starts[shared]]
    cols = tf[starts[shared] + 1]
    n = len(tets)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, _ = connected_components(graph, directed=False)
    return n_components


def _exterior_faces(tets):
    sf, tf, starts, ends = _grouped_faces(tets)
    counts = ends - starts
    assert (counts <= 2).all(), f'a face shared by {counts.max()} tets -- non-manifold'
    ext_idx = np.nonzero(counts == 1)[0]
    return sf[starts[ext_idx]]


def _domain_bounds(shape, vs=1.0):
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([(s - 1) * vs for s in shape])
    return lo, hi


def _uniform_lgi(shape):
    return np.ones(shape, dtype=np.int32)


def _two_grain_lgi(shape, split_at=None):
    lgi = np.ones(shape, dtype=np.int32)
    split_at = split_at if split_at is not None else shape[2] // 2
    lgi[:, :, split_at:] = 2
    return lgi


def _four_grain_octant_lgi(shape):
    nx, ny, nz = shape
    hx, hy = nx // 2, ny // 2
    lgi = np.ones(shape, dtype=np.int32)
    lgi[hx:, :hy, :] = 2
    lgi[:hx, hy:, :] = 3
    lgi[hx:, hy:, :] = 4
    return lgi


def _edge_boundary_lgi(shape):
    """A grain boundary running EXACTLY along one RVE edge (x=hi, y=hi)."""
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    lgi[nx - 1, ny - 1, :] = 2
    return lgi


def _corner_voxel_lgi(shape):
    """A single voxel occupying exactly one RVE corner is its own grain."""
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    lgi[nx - 1, ny - 1, nz - 1] = 2
    return lgi


def _voronoi_lgi(shape, n_seeds, seed=0):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, np.array(shape) - 1, size=(n_seeds, 3))
    idx = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')
    pts = np.stack(idx, axis=-1).reshape(-1, 3).astype(np.float64)
    d2 = ((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    return (np.argmin(d2, axis=1) + 1).reshape(shape).astype(np.int32)


def _corner_junction_lgi(shape):
    """
    5 grains, one of them occupying the octant whose OWN corner is the
    domain's own corner voxel -- a multi-grain junction pinned exactly at
    an RVE corner, not just a single grain boundary happening to touch it.
    """
    nx, ny, nz = shape
    hx, hy, hz = nx // 2, ny // 2, nz // 2
    lgi = np.ones(shape, dtype=np.int32)
    lgi[hx:, hy:, :hz - 1] = 2
    lgi[hx:, :hy, hz - 1:] = 3
    lgi[:hx, hy:, hz - 1:] = 4
    lgi[hx:, hy:, hz - 1:] = 5
    return lgi


def _nested_shell_corner_lgi(shape):
    """Concentric grains collapsing onto the domain's own corner point."""
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    lgi[nx - 2:, ny - 2:, nz - 2:] = 2
    lgi[nx - 1, ny - 1, nz - 1] = 3
    return lgi


def _thin_domain_lgi(shape):
    """An ordinary two-grain split, but the domain itself is extremely
    thin along one axis -- stresses whether padding/clipping degenerates
    when the real domain barely has any thickness at all."""
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    lgi[nx // 2:, :, :] = 2
    return lgi


def _combined_stress_lgi(shape):
    """
    Three stresses in the SAME structure at once, so nothing about
    handling one interferes with the others: a grain boundary running
    exactly along one RVE edge, a single-voxel grain at the OPPOSITE
    corner, and an unrelated small grain nowhere near any boundary.
    """
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    lgi[nx - 1, ny - 1, :] = 2
    lgi[0, 0, 0] = 3
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    lgi[cx:cx + 2, cy:cy + 2, cz:cz + 2] = 4
    return lgi


CASES = [
    ('uniform_single_grain', _uniform_lgi((8, 8, 8))),
    ('two_grain', _two_grain_lgi((6, 6, 8))),
    ('four_grain_octant', _four_grain_octant_lgi((10, 10, 10))),
    ('edge_boundary', _edge_boundary_lgi((8, 8, 8))),
    ('corner_voxel', _corner_voxel_lgi((8, 8, 8))),
    ('voronoi_16grain', _voronoi_lgi((12, 12, 12), 16, seed=1)),
    # Adversarial combinations, not just single edge cases in isolation --
    # see boundary_clip.py's "numerically foolproof?" verification.
    ('corner_junction_5grain', _corner_junction_lgi((8, 8, 8))),
    ('nested_shells_at_corner', _nested_shell_corner_lgi((8, 8, 8))),
    ('extremely_thin_domain', _thin_domain_lgi((8, 8, 2))),
    ('minimal_domain_2x2x2', _corner_voxel_lgi((2, 2, 2))),
    ('combined_edge_corner_interior', _combined_stress_lgi((10, 10, 10))),
]


def _run(lgi, pad=1, vs=1.0):
    lat = build_bcc_lattice(lgi.shape, LatticeConfig(voxel_size=vs, pad=pad))
    result, clipped, provenance = cleave_lattice_exact_boundary(
        lat, lgi, LabelingConfig(wall_label=WALL))
    return lat, result


# ---------------------------------------------------------------------------
# clip_lattice_to_domain alone -- pure geometry, no grain labels involved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('shape,pad,vs', [
    ((8, 8, 8), 1, 1.0),
    ((4, 4, 4), 1, 1.0),
    ((6, 10, 5), 1, 1.0),
    ((8, 8, 8), 1, 0.37),
    ((8, 8, 8), 2, 1.0),
])
def test_clip_conserves_exact_domain_volume(shape, pad, vs):
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=vs, pad=pad))
    clipped, _ = clip_lattice_to_domain(lat)
    vol = signed_volumes(clipped.vertices, clipped.tets)
    nx, ny, nz = shape
    expected = (nx - 1) * (ny - 1) * (nz - 1) * vs ** 3
    assert abs(vol.sum() - expected) < 1e-9, (
        f'clipped volume {vol.sum()} != exact domain volume {expected}')


@pytest.mark.parametrize('shape', [(8, 8, 8), (4, 4, 4), (6, 10, 5)])
def test_clip_all_tets_positive_volume(shape):
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    clipped, _ = clip_lattice_to_domain(lat)
    vol = signed_volumes(clipped.vertices, clipped.tets)
    assert (vol > 0).all(), f'{int((vol <= 0).sum())} tet(s) with non-positive volume after clipping'


@pytest.mark.parametrize('shape', [(8, 8, 8), (4, 4, 4), (6, 10, 5)])
def test_clip_no_vertex_beyond_domain(shape):
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    clipped, _ = clip_lattice_to_domain(lat)
    used = np.unique(clipped.tets.ravel())
    coords = clipped.vertices[used]
    lo, hi = _domain_bounds(shape)
    beyond = (coords < lo - 1e-9) | (coords > hi + 1e-9)
    assert not beyond.any(), 'a referenced vertex sits outside the real domain after clipping'


PRECISION_CASES = [
    ('vs=1.0_(8,8,8)', (8, 8, 8), 1.0),
    ('vs=1.0_(33,33,33)_odd_prime', (33, 33, 33), 1.0),
    ('vs=0.37_(8,8,8)_ugly_fraction', (8, 8, 8), 0.37),
    ('vs=1-3_(10,10,10)_repeating_decimal', (10, 10, 10), 1.0 / 3.0),
    ('vs=1e-4_(8,8,8)_tiny', (8, 8, 8), 1e-4),
    ('vs=1e4_(8,8,8)_huge', (8, 8, 8), 1e4),
    pytest.param('vs=1.0_(128,128,128)_large_domain', (128, 128, 128), 1.0,
                 marks=pytest.mark.slow),
    pytest.param('vs=0.1_(200,200,200)_large_and_fractional', (200, 200, 200), 0.1,
                 marks=pytest.mark.slow),
]


@pytest.mark.parametrize('name,shape,vs', PRECISION_CASES)
def test_clip_boundary_bit_exact_across_scales(name, shape, vs):
    """
    Every point in clip_lattice_to_domain's output that sits on the domain
    boundary must be BIT-EXACT to the domain's own bound, not merely
    close, regardless of voxel size or domain scale. New points get this
    by explicit assignment (see boundary_clip.py's _BoxClipCache.resolve),
    but ORIGINAL vertices already on the plane rely on two independently-
    written expressions -- (n-1)*vs here and inside lattice.py's own
    vertex construction -- landing on the identical float. IEEE 754
    multiplication is deterministic, so this should always hold, but is
    checked directly (not assumed) across scales spanning tiny to huge
    and round to ugly-looking voxel sizes -- e.g. vs=0.1 at (200,200,200)
    produces 19.900000000000002, not a clean 19.9, and it must still
    match bit-for-bit.
    """
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=vs, pad=1))
    clipped, _ = clip_lattice_to_domain(lat)
    used = np.unique(clipped.tets.ravel())
    coords = clipped.vertices[used]
    lo, hi = _domain_bounds(shape, vs)
    for axis in range(3):
        near_hi = np.abs(coords[:, axis] - hi[axis]) < 1e-6
        near_lo = np.abs(coords[:, axis] - lo[axis]) < 1e-6
        assert np.all(coords[near_hi, axis] == hi[axis]), (
            f'{name}: axis {axis} hi-boundary point(s) not bit-exact to {hi[axis]}')
        assert np.all(coords[near_lo, axis] == lo[axis]), (
            f'{name}: axis {axis} lo-boundary point(s) not bit-exact to {lo[axis]}')


def test_clip_cross_tet_consistency():
    """
    Two original lattice tets sharing an edge that straddles the same
    plane must produce the IDENTICAL new boundary point -- otherwise the
    output has a crack exactly at the RVE surface. Every new point's
    provenance (inside_gid, outside_gid) is a pure function of that edge,
    so grouping all new points by their provenance pair and requiring
    each group to resolve to a single, unique coordinate is an exhaustive
    check across the whole lattice, not a hand-picked pair of tets.
    """
    lat = build_bcc_lattice((8, 8, 8), LatticeConfig(voxel_size=1.0, pad=1))
    clipped, provenance = clip_lattice_to_domain(lat)

    pair_to_positions = {}
    for new_gid, pair in provenance.items():
        pair_to_positions.setdefault(pair, set()).add(tuple(clipped.vertices[new_gid]))
    for pair, positions in pair_to_positions.items():
        assert len(positions) == 1, f'edge {pair} produced {len(positions)} different boundary points'


# ---------------------------------------------------------------------------
# Full pipeline (box-clip + grain-label cleave): the actual CPFEM guarantee
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name,lgi', CASES)
def test_exact_flatness_on_every_rve_face(name, lgi):
    """
    The actual point of this module: every exterior triangle of the final,
    grain-cleaved mesh must have all 3 vertices sharing one EXACT axis
    value equal to the domain's own lo/hi bound on that axis -- not
    approximately, bit-exactly. Checked directly per triangle (a "does
    this face have a shared exact plane" test), not via a "mean position
    near a plane" tolerance heuristic -- the latter can misclassify a
    legitimate interior grain-boundary point that happens to sit close to
    (but not on) a face as though it were meant to be flat there, which is
    exactly the false positive this module's own development first hit.
    """
    _, result = _run(lgi)
    ext = _exterior_faces(result.tets)
    ext_coords = result.node_coords[ext]
    lo, hi = _domain_bounds(lgi.shape)
    on_lo = (ext_coords == lo[None, None, :]).all(axis=1)
    on_hi = (ext_coords == hi[None, None, :]).all(axis=1)
    flat = (on_lo | on_hi).any(axis=1)
    n_bad = int((~flat).sum())
    assert n_bad == 0, f'{name}: {n_bad}/{len(ext)} exterior face(s) not exactly flat'


@pytest.mark.parametrize('name,lgi', CASES)
def test_all_positive_volume(name, lgi):
    _, result = _run(lgi)
    vol = signed_volumes(result.node_coords, result.tets)
    assert (vol > 0).all(), f'{name}: {int((vol <= 0).sum())} tet(s) with non-positive volume'


@pytest.mark.parametrize('name,lgi', CASES)
def test_no_wall_labels_reach_output(name, lgi):
    """
    clip_lattice_to_domain removes everything outside the real domain
    before grain labels are even sampled, so the wall sentinel should
    never be reachable -- unlike the old cleave_lattice(padded lattice)
    path, which always produced wall-labelled tets that had to be
    stripped afterward.
    """
    _, result = _run(lgi)
    assert WALL not in set(result.tet_labels.tolist())


@pytest.mark.parametrize('name,lgi', CASES)
def test_no_dangling_nodes(name, lgi):
    _, result = _run(lgi)
    used = np.unique(result.tets.ravel())
    assert len(used) == len(result.node_coords), (
        f'{name}: {len(result.node_coords) - len(used)} node(s) never referenced by any tet')


@pytest.mark.parametrize('name,lgi', CASES)
def test_whole_mesh_and_every_grain_single_component(name, lgi):
    _, result = _run(lgi)
    n_whole = _n_connected_components(result.tets)
    assert n_whole == 1, f'{name}: whole mesh split into {n_whole} components'
    for lab in sorted(set(result.tet_labels.tolist())):
        mask = result.tet_labels == lab
        n_comp = _n_connected_components(result.tets, mask)
        assert n_comp == 1, f'{name}: grain {lab} split into {n_comp} disconnected components'
