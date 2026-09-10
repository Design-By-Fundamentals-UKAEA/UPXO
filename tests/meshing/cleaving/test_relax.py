"""
test_relax.py
=============
Correctness tests for relax.relax_boundary_points.

The property that matters most: relaxation must NEVER be able to push any
tet below its configured volume floor, no matter how many of a tet's
corners are movable and moving simultaneously in the same iteration (the
bug this module's design was built to rule out -- see relax.py's docstring
on the per-point pre-clip vs. the global verify-and-backtrack safety net).
Everything else (topology unchanged, boundary points actually move,
original vertices never move) is tested too, but the volume floor is the
one that would silently corrupt a mesh if it ever regressed.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice
from upxo.meshing.cleaving.relax import relax_boundary_points, RelaxConfig

WALL = 32767


def _two_grain_lgi(shape, split_at=None):
    lgi = np.ones(shape, dtype=np.int32)
    split_at = split_at if split_at is not None else shape[2] // 2
    lgi[:, :, split_at:] = 2
    return lgi


def _four_grain_quadrant_lgi(shape):
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    hx, hy = nx // 2, ny // 2
    lgi[:hx, :hy, :] = 1
    lgi[hx:, :hy, :] = 2
    lgi[:hx, hy:, :] = 3
    lgi[hx:, hy:, :] = 4
    return lgi


def _cleave(lgi, pad=1):
    lat = build_bcc_lattice(lgi.shape, LatticeConfig(voxel_size=1.0, pad=pad))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=WALL))
    return lat, cleave_lattice(lat, labels)


def _original_vertex_mask(result):
    """
    Which node ids are ORIGINAL lattice vertices, i.e. NOT in
    new_point_gids. A plain ``index < len(lat.vertices)`` slice looks
    equivalent but silently breaks the moment ANY original vertex gets
    dropped by cleave_lattice's own dangling-node compaction (always
    exactly the padded lattice's 8 outer corners) -- that shifts later
    rows down into what used to be original-vertex index territory, so
    the tail of a "first n_base rows" slice quietly starts containing new
    points instead. new_point_gids is the compaction-robust source of
    truth (see cleave.py / relax.py's own _is_new_point_mask).
    """
    n_points = len(result.node_coords)
    is_new = np.zeros(n_points, dtype=bool)
    if result.new_point_gids:
        is_new[np.fromiter(result.new_point_gids.keys(), dtype=np.int64,
                            count=len(result.new_point_gids))] = True
    return ~is_new


CASES = [
    ('two_grain', _two_grain_lgi((5, 5, 6))),
    ('four_grain_quadrant', _four_grain_quadrant_lgi((6, 6, 6))),
]


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('min_volume_fraction', [0.1, 0.2, 0.5])
def test_volume_floor_never_breached(name, lgi, min_volume_fraction):
    lat, result = _cleave(lgi)
    original_vol = signed_volumes(result.node_coords, result.tets)

    cfg = RelaxConfig(n_iterations=10, min_volume_fraction=min_volume_fraction)
    relaxed = relax_boundary_points(lat, result, cfg)
    relaxed_vol = signed_volumes(relaxed.node_coords, relaxed.tets)

    assert (relaxed_vol > 0).all()
    # Small float slack below the exact floor -- the safety net compares
    # against the floor with a plain `>`, so a value can land arbitrarily
    # close to (but strictly above) it.
    slack = 1e-6
    assert (relaxed_vol >= min_volume_fraction * original_vol - slack).all(), (
        f'{name}: some tet dropped below its {min_volume_fraction} volume floor')


@pytest.mark.parametrize('name,lgi', CASES)
def test_topology_unchanged(name, lgi):
    """Relaxation must only move coordinates -- never touch connectivity,
    labels, or the source/deferred bookkeeping."""
    lat, result = _cleave(lgi)
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=5))

    assert np.array_equal(result.tets, relaxed.tets)
    assert np.array_equal(result.tet_labels, relaxed.tet_labels)
    assert np.array_equal(result.source_tet_index, relaxed.source_tet_index)
    assert np.array_equal(result.deferred_tet_indices, relaxed.deferred_tet_indices)
    assert relaxed.node_coords.shape == result.node_coords.shape


@pytest.mark.parametrize('name,lgi', CASES)
def test_original_vertices_never_move(name, lgi):
    lat, result = _cleave(lgi)
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=5))
    original = _original_vertex_mask(result)
    assert np.array_equal(result.node_coords[original], relaxed.node_coords[original])


def test_boundary_staircase_tightens():
    """
    The actual point of this module: the grain-boundary cut points near a
    nominal flat interior boundary should cluster tighter after relaxation
    than before (fewer/closer distinct z-values), not just "not break
    anything". Uses the same construction that originally exposed the
    staircase effect during development.
    """
    shape = (5, 5, 6)
    split_at = 3
    lat, result = _cleave(_two_grain_lgi(shape, split_at=split_at))
    is_new = ~_original_vertex_mask(result)

    def interior_boundary_zspread(node_coords):
        cut = node_coords[is_new]
        mask = (cut[:, 0] > 1) & (cut[:, 0] < 3) & (cut[:, 1] > 1) & (cut[:, 1] < 3)
        z = cut[mask, 2]
        # restrict to the grain1/grain2 boundary cluster, not the wall ones
        z = z[(z > split_at - 1.5) & (z < split_at + 0.5)]
        return z

    z_before = interior_boundary_zspread(result.node_coords)
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=10))
    z_after = interior_boundary_zspread(relaxed.node_coords)

    assert len(z_before) > 0 and len(z_after) > 0
    assert z_after.std() < z_before.std(), (
        f'expected the boundary cluster to tighten: '
        f'std before={z_before.std():.4f} after={z_after.std():.4f}')


def test_zero_iterations_is_a_noop():
    lat, result = _cleave(_two_grain_lgi((5, 5, 6)))
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=0))
    assert np.array_equal(result.node_coords, relaxed.node_coords)


def test_points_actually_move_by_default():
    lat, result = _cleave(_two_grain_lgi((5, 5, 6)))
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=5))
    assert not np.array_equal(result.node_coords, relaxed.node_coords)


def test_rve_plane_points_stay_exactly_on_their_plane():
    """
    A grain-boundary point that already sits exactly on one of the RVE's
    own outer faces (boundary_clip.cleave_lattice_exact_boundary's whole
    reason for existing) must never be pulled off it by plain Laplacian
    averaging with neighbours that are not all on that same plane --
    confirmed directly as a real defect before this test existed: an
    unconstrained relax pass depresses grain-boundary triple junctions
    inward off the RVE surface wherever one happens to land on it.
    _four_grain_quadrant_lgi's boundaries run the full z-extent, so they
    genuinely reach the z=0 and z=hi faces, giving real on-plane junction
    points to check, not just face-interior points.
    """
    from upxo.meshing.cleaving.boundary_clip import cleave_lattice_exact_boundary

    shape = (8, 8, 8)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    result = cleave_lattice_exact_boundary(lat, _four_grain_quadrant_lgi(shape))[0]

    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([s - 1.0 for s in shape])
    before = result.node_coords
    assert ((before == lo) | (before == hi)).any(), (
        'test setup problem: no boundary point landed exactly on an RVE plane')

    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=10))
    after = relaxed.node_coords

    # Per axis, not per whole point: a point on a FACE is still free to
    # slide within that face (2 of its 3 coordinates legitimately change),
    # so the invariant is "whichever axis was locked to a plane stays
    # locked to that same plane" -- not "the point never moves at all".
    for axis in range(3):
        was_lo = before[:, axis] == lo[axis]
        was_hi = before[:, axis] == hi[axis]
        assert np.all(after[was_lo, axis] == lo[axis]), (
            f'axis {axis}: a point on the lo-plane moved off it during relaxation')
        assert np.all(after[was_hi, axis] == hi[axis]), (
            f'axis {axis}: a point on the hi-plane moved off it during relaxation')
