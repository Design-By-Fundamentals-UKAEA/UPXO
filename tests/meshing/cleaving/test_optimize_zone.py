"""
test_optimize_zone.py
======================
Correctness tests for relax.optimize_boundary_zone -- the second-pass,
ring-based quality optimization (see cleaving/__init__.py's Milestone 7).

Reuses test_optimize.py's fixtures/helpers directly (same domains, same
_cleave/_mesh_quality plumbing) since this pass shares optimize_boundary_
points' exact movement mechanism and safety floors -- only the SET of
movable points (a tapering ring of original vertices instead of just the
boundary points) and the ring-decay weighting are new.

Two properties get the same never-breached treatment as test_optimize.py's
volume/quality floors (shared reasoning, not re-derived here). Beyond
those, this module's own concern is the ring mechanism itself: points
strictly beyond ``max_ring`` hops from any boundary point must never move,
checked against an INDEPENDENTLY recomputed reachability set (not by
peeking at the function's own internal hop bookkeeping) -- and the
boundary points themselves, which this pass deliberately excludes, must
stay exactly as frozen as optimize_boundary_points left them.

A numeric ring-weight decay ratio (e.g. "a hop-1 point moves exactly 3x a
hop-3 point") is deliberately NOT tested via full optimization dynamics:
actual displacement is also driven by each point's local tet quality
deficit, which varies point to point independently of ring weight, so a
strict magnitude-ratio assertion would be testing mesh-quality noise as
much as the decay formula and would be fragile. The cutoff tests below
(a point either can or cannot move at all under a given max_ring) test the
same underlying mechanism without that confound.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice
from upxo.meshing.cleaving.relax import (
    optimize_boundary_zone, ZoneOptimizeConfig, _tet_shape_quality,
)

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


def _mesh_quality(node_coords, tets):
    return _tet_shape_quality(node_coords[tets[:, 0]], node_coords[tets[:, 1]],
                               node_coords[tets[:, 2]], node_coords[tets[:, 3]])


def _is_new_and_hop1_mask(result):
    """Independently recomputes (a) which points are boundary points and
    (b) which points share a tet with one -- the exact set
    optimize_boundary_zone(max_ring=1) is allowed to move -- without
    relying on any of that function's own internals."""
    n_points = len(result.node_coords)
    is_new = np.zeros(n_points, dtype=bool)
    if result.new_point_gids:
        is_new[np.fromiter(result.new_point_gids.keys(), dtype=np.int64,
                            count=len(result.new_point_gids))] = True
    tets = result.tets
    touches_boundary = is_new[tets].any(axis=1)
    hop1 = np.unique(tets[touches_boundary])
    hop1_mask = np.zeros(n_points, dtype=bool)
    hop1_mask[hop1] = True
    return is_new, hop1_mask


CASES = [
    ('two_grain', _two_grain_lgi((8, 8, 12))),
    ('four_grain_quadrant', _four_grain_quadrant_lgi((10, 10, 10))),
]


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('min_volume_fraction', [0.1, 0.2, 0.5])
def test_volume_floor_never_breached(name, lgi, min_volume_fraction):
    lat, result = _cleave(lgi)
    original_vol = signed_volumes(result.node_coords, result.tets)

    cfg = ZoneOptimizeConfig(n_iterations=5, min_volume_fraction=min_volume_fraction)
    optimized = optimize_boundary_zone(lat, result, cfg)
    new_vol = signed_volumes(optimized.node_coords, optimized.tets)

    assert (new_vol > 0).all()
    slack = 1e-6
    assert (new_vol >= min_volume_fraction * original_vol - slack).all(), (
        f'{name} min_volume_fraction={min_volume_fraction}: some tet dropped below its volume floor')


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('quality_regression_tolerance', [0.0, 0.02, 0.05, 0.2])
def test_quality_floor_never_breached(name, lgi, quality_regression_tolerance):
    lat, result = _cleave(lgi)
    original_quality = _mesh_quality(result.node_coords, result.tets)

    cfg = ZoneOptimizeConfig(n_iterations=5, quality_regression_tolerance=quality_regression_tolerance)
    optimized = optimize_boundary_zone(lat, result, cfg)
    new_quality = _mesh_quality(optimized.node_coords, optimized.tets)

    slack = 1e-6
    assert (new_quality >= original_quality - quality_regression_tolerance - slack).all(), (
        f'{name} tolerance={quality_regression_tolerance}: some tet regressed past its quality floor')


@pytest.mark.parametrize('name,lgi', CASES)
def test_topology_unchanged(name, lgi):
    lat, result = _cleave(lgi)
    optimized = optimize_boundary_zone(lat, result, ZoneOptimizeConfig(n_iterations=5))

    assert np.array_equal(result.tets, optimized.tets)
    assert np.array_equal(result.tet_labels, optimized.tet_labels)
    assert np.array_equal(result.source_tet_index, optimized.source_tet_index)
    assert np.array_equal(result.deferred_tet_indices, optimized.deferred_tet_indices)
    assert optimized.node_coords.shape == result.node_coords.shape


@pytest.mark.parametrize('name,lgi', CASES)
def test_boundary_points_never_move(name, lgi):
    """optimize_boundary_zone's whole purpose is moving a DIFFERENT
    category of point (nearby original vertices) -- the exact boundary
    points it deliberately excludes (optimize_boundary_points' own
    territory) must stay exactly as frozen as they started here."""
    lat, result = _cleave(lgi)
    is_new, _ = _is_new_and_hop1_mask(result)

    optimized = optimize_boundary_zone(lat, result, ZoneOptimizeConfig(n_iterations=8))

    assert np.array_equal(result.node_coords[is_new], optimized.node_coords[is_new])


@pytest.mark.parametrize('name,lgi', CASES)
def test_points_beyond_ring_never_move(name, lgi):
    """With max_ring=1, a point that doesn't even share a tet with a
    boundary point (recomputed independently, not via the function's own
    hop bookkeeping) must be completely untouched."""
    lat, result = _cleave(lgi)
    is_new, hop1_mask = _is_new_and_hop1_mask(result)
    never_movable = ~hop1_mask & ~is_new

    optimized = optimize_boundary_zone(
        lat, result, ZoneOptimizeConfig(n_iterations=8, max_ring=1))

    assert np.array_equal(result.node_coords[never_movable],
                           optimized.node_coords[never_movable]), (
        f'{name}: a point strictly beyond max_ring=1 moved')


def test_rve_plane_points_stay_exactly_on_their_plane():
    """Same invariant as optimize_boundary_points' own version of this
    test, now covering the newly-mobile original vertices too: a point
    already sitting exactly on an RVE face must slide only within it."""
    from upxo.meshing.cleaving.boundary_clip import cleave_lattice_exact_boundary

    shape = (8, 8, 8)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    result = cleave_lattice_exact_boundary(lat, _four_grain_quadrant_lgi(shape))[0]

    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([s - 1.0 for s in shape])
    before = result.node_coords
    assert ((before == lo) | (before == hi)).any(), (
        'test setup problem: no point landed exactly on an RVE plane')

    optimized = optimize_boundary_zone(lat, result, ZoneOptimizeConfig(n_iterations=5))
    after = optimized.node_coords

    for axis in range(3):
        was_lo = before[:, axis] == lo[axis]
        was_hi = before[:, axis] == hi[axis]
        assert np.all(after[was_lo, axis] == lo[axis]), (
            f'axis {axis}: a point on the lo-plane moved off it during optimization')
        assert np.all(after[was_hi, axis] == hi[axis]), (
            f'axis {axis}: a point on the hi-plane moved off it during optimization')


def test_zero_iterations_is_a_noop():
    lat, result = _cleave(_two_grain_lgi((8, 8, 12)))
    optimized = optimize_boundary_zone(lat, result, ZoneOptimizeConfig(n_iterations=0))
    assert np.array_equal(result.node_coords, optimized.node_coords)


def test_invalid_adjacency_raises():
    lat, result = _cleave(_two_grain_lgi((8, 8, 12)))
    with pytest.raises(ValueError):
        optimize_boundary_zone(lat, result, ZoneOptimizeConfig(adjacency='face'))


def test_invalid_decay_raises():
    lat, result = _cleave(_two_grain_lgi((8, 8, 12)))
    with pytest.raises(ValueError):
        optimize_boundary_zone(lat, result, ZoneOptimizeConfig(decay='exponential'))


def test_invalid_max_ring_raises():
    lat, result = _cleave(_two_grain_lgi((8, 8, 12)))
    with pytest.raises(ValueError):
        optimize_boundary_zone(lat, result, ZoneOptimizeConfig(max_ring=0))
