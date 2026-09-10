"""
test_optimize.py
=================
Correctness tests for relax.optimize_boundary_points -- the standard first
quality-improvement pass (see cleaving/__init__.py's Milestone 6).

Two properties matter most, mirroring test_relax.py's and test_snap.py's
discipline, but this module needs BOTH a volume floor AND a quality floor
checked, not just one:

1. Neither floor can ever be breached -- volume (shared reasoning with
   relax_boundary_points) AND shape quality itself. The quality floor is
   not redundant with the volume one: a tet can trade volume for a worse
   shape (more sheared/skewed) while staying comfortably above the volume
   floor, confirmed directly during development (an early version without
   it let the single worst tet's min_angle regress, 13.9 deg -> 10.7 deg).

2. The quality floor's tolerance cannot be near-zero either -- confirmed
   directly that literal non-regression everywhere freezes the optimizer
   completely (0 tets change, at any candidate step size), since helping
   one tet reliably costs some other tet sharing that point a little
   quality. test_default_tolerance_actually_moves_points guards against
   silently regressing back to that frozen state.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice
from upxo.meshing.cleaving.boundary_clip import cleave_lattice_exact_boundary
from upxo.meshing.cleaving.relax import (
    optimize_boundary_points, OptimizeConfig, _tet_shape_quality,
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


CASES = [
    ('two_grain', _two_grain_lgi((8, 8, 10))),
    ('four_grain_quadrant', _four_grain_quadrant_lgi((10, 10, 10))),
]


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('min_volume_fraction', [0.1, 0.2, 0.5])
def test_volume_floor_never_breached(name, lgi, min_volume_fraction):
    lat, result = _cleave(lgi)
    original_vol = signed_volumes(result.node_coords, result.tets)

    cfg = OptimizeConfig(n_iterations=5, min_volume_fraction=min_volume_fraction)
    optimized = optimize_boundary_points(lat, result, cfg)
    new_vol = signed_volumes(optimized.node_coords, optimized.tets)

    assert (new_vol > 0).all()
    slack = 1e-6
    assert (new_vol >= min_volume_fraction * original_vol - slack).all(), (
        f'{name} min_volume_fraction={min_volume_fraction}: some tet dropped below its volume floor')


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('quality_regression_tolerance', [0.0, 0.02, 0.05, 0.2])
def test_quality_floor_never_breached(name, lgi, quality_regression_tolerance):
    """No tet's shape quality may drop below its OWN pre-optimization value
    by more than the configured tolerance -- checked at tolerance=0.0 too
    (the exact scenario that was directly confirmed to freeze the
    optimizer entirely, i.e. this must hold trivially by not moving
    anything, not by quietly exceeding the floor)."""
    lat, result = _cleave(lgi)
    original_quality = _mesh_quality(result.node_coords, result.tets)

    cfg = OptimizeConfig(n_iterations=5, quality_regression_tolerance=quality_regression_tolerance)
    optimized = optimize_boundary_points(lat, result, cfg)
    new_quality = _mesh_quality(optimized.node_coords, optimized.tets)

    slack = 1e-6
    assert (new_quality >= original_quality - quality_regression_tolerance - slack).all(), (
        f'{name} tolerance={quality_regression_tolerance}: some tet regressed past its quality floor')


@pytest.mark.parametrize('name,lgi', CASES)
def test_topology_unchanged(name, lgi):
    lat, result = _cleave(lgi)
    optimized = optimize_boundary_points(lat, result, OptimizeConfig(n_iterations=5))

    assert np.array_equal(result.tets, optimized.tets)
    assert np.array_equal(result.tet_labels, optimized.tet_labels)
    assert np.array_equal(result.source_tet_index, optimized.source_tet_index)
    assert np.array_equal(result.deferred_tet_indices, optimized.deferred_tet_indices)
    assert optimized.node_coords.shape == result.node_coords.shape


@pytest.mark.parametrize('name,lgi', CASES)
def test_original_vertices_never_move(name, lgi):
    lat, result = _cleave(lgi)
    n_points = len(result.node_coords)
    is_new = np.zeros(n_points, dtype=bool)
    if result.new_point_gids:
        is_new[np.fromiter(result.new_point_gids.keys(), dtype=np.int64,
                            count=len(result.new_point_gids))] = True
    original = ~is_new

    optimized = optimize_boundary_points(lat, result, OptimizeConfig(n_iterations=5))
    assert np.array_equal(result.node_coords[original], optimized.node_coords[original])


def test_rve_plane_points_stay_exactly_on_their_plane():
    """Same invariant as relax_boundary_points' own version of this test --
    a point already sitting exactly on an RVE face must slide only within
    it (other axes may legitimately move), never be pulled off it."""
    shape = (8, 8, 8)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    result = cleave_lattice_exact_boundary(lat, _four_grain_quadrant_lgi(shape))[0]

    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([s - 1.0 for s in shape])
    before = result.node_coords
    assert ((before == lo) | (before == hi)).any(), (
        'test setup problem: no boundary point landed exactly on an RVE plane')

    optimized = optimize_boundary_points(lat, result, OptimizeConfig(n_iterations=5))
    after = optimized.node_coords

    for axis in range(3):
        was_lo = before[:, axis] == lo[axis]
        was_hi = before[:, axis] == hi[axis]
        assert np.all(after[was_lo, axis] == lo[axis]), (
            f'axis {axis}: a point on the lo-plane moved off it during optimization')
        assert np.all(after[was_hi, axis] == hi[axis]), (
            f'axis {axis}: a point on the hi-plane moved off it during optimization')


def test_zero_iterations_is_a_noop():
    lat, result = _cleave(_two_grain_lgi((8, 8, 10)))
    optimized = optimize_boundary_points(lat, result, OptimizeConfig(n_iterations=0))
    assert np.array_equal(result.node_coords, optimized.node_coords)


def test_zero_tolerance_is_a_noop():
    """Confirmed directly during development: a near-zero quality
    regression tolerance rejects every candidate move at every step size,
    so nothing moves at all -- this is the honest, expected behaviour of
    that setting, not a bug, and this test is what would catch it silently
    changing (e.g. someone loosening the check without meaning to)."""
    lat, result = _cleave(_four_grain_quadrant_lgi((10, 10, 10)))
    optimized = optimize_boundary_points(
        lat, result, OptimizeConfig(n_iterations=5, quality_regression_tolerance=0.0))
    assert np.array_equal(result.node_coords, optimized.node_coords)


def test_default_tolerance_actually_moves_points():
    """
    Confirmed directly: this fixture's worst tet sits at quality 0.3062 --
    just above OptimizeConfig's default 0.3 threshold, so at the DEFAULT
    threshold there's nothing to fix and nothing should move (that's
    correct behaviour, not the frozen-optimizer bug this test guards
    against). Lowering just the threshold to 0.4 for this one test brings
    that same tet into scope, so this checks what it's meant to: that the
    default TOLERANCE (0.05) actually permits movement once there IS
    something below the bar, not that every mesh has something to fix.
    """
    lat, result = _cleave(_four_grain_quadrant_lgi((10, 10, 10)))
    optimized = optimize_boundary_points(
        lat, result, OptimizeConfig(n_iterations=5, quality_threshold=0.4))
    assert not np.array_equal(result.node_coords, optimized.node_coords)


def test_worst_tet_quality_does_not_regress():
    """The actual point of this module, on a structure with several real
    grain-boundary junctions to work with: the single worst tet's quality
    must not be lower after optimization than before it -- the specific
    failure an earlier version (volume floor only, no quality floor) had,
    confirmed directly (13.9 deg -> 10.7 deg min_angle regression)."""
    lat, result = _cleave(_four_grain_quadrant_lgi((10, 10, 10)))
    quality_before = _mesh_quality(result.node_coords, result.tets)

    optimized = optimize_boundary_points(lat, result, OptimizeConfig(n_iterations=8))
    quality_after = _mesh_quality(optimized.node_coords, optimized.tets)

    assert quality_after.min() >= quality_before.min() - 1e-6, (
        f'worst tet regressed: before={quality_before.min():.4f} after={quality_after.min():.4f}')
