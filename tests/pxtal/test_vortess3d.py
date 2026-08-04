"""Tests for upxo.pxtal.vortess3d Voronoi constructors.

Note: the real class is ``gtess3d`` (not ``Voronoi3D`` -- no such class
exists in this codebase). ``bounds`` must coerce to shape (3, 2), e.g.
``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` -- a flat 6-tuple does not.
"""
import numpy as np
import pytest
from upxo.pxtal.vortess3d import gtess3d


def test_gtess3d_from_seed_points_basic():
    """Create gtess3d from seed points."""
    seed_points = np.array([
        [10, 10, 10],
        [30, 30, 30],
        [50, 50, 50]
    ], dtype=float)
    bounds = [[0, 60], [0, 60], [0, 60]]

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None
    assert 'coords' in vor.sp


def test_gtess3d_from_mpoint3d():
    """Create gtess3d from MPoint3d seed points."""
    from upxo.geoEntities.mulpoint3d import MPoint3d

    mpts = MPoint3d.from_custom_seeds(np.array([
        [10, 10, 10],
        [30, 30, 30],
        [50, 50, 50]
    ], dtype=float))
    bounds = [[0, 60], [0, 60], [0, 60]]

    vor = gtess3d.from_mpoint3d(mpts, bounds=bounds)
    assert vor is not None


def test_gtess3d_from_seed_points_with_periodic():
    """Create gtess3d with periodic boundary conditions."""
    seed_points = np.array([
        [5, 5, 5],
        [15, 15, 15],
        [25, 25, 25]
    ], dtype=float)
    bounds = [[0, 30], [0, 30], [0, 30]]
    periodic = (True, True, True)

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds, periodic=periodic)
    assert vor is not None
    assert vor.info['uinputs']['periodic'] == [True, True, True]


def test_gtess3d_from_seed_points_no_bounds():
    """Create gtess3d without explicit bounds (auto-inferred from extrema)."""
    seed_points = np.array([
        [0, 0, 0],
        [10, 10, 10],
        [20, 20, 20]
    ], dtype=float)

    vor = gtess3d.from_seed_points(seed_points, bounds=None)
    assert vor is not None
    assert vor.bounds['xbound'][0] <= 0.0
    assert vor.bounds['xbound'][1] >= 20.0


def test_gtess3d_from_seed_points_rejects_flat_bounds():
    """A flat 6-tuple bounds does not coerce to the required (3, 2) shape."""
    seed_points = np.array([[10, 10, 10], [30, 30, 30]], dtype=float)

    with pytest.raises(ValueError):
        gtess3d.from_seed_points(seed_points, bounds=(0, 60, 0, 60, 0, 60))


def test_gtess3d_from_seed_points_single_seed():
    """Create gtess3d with a single seed point."""
    seed_points = np.array([[15, 15, 15]], dtype=float)
    bounds = [[0, 30], [0, 30], [0, 30]]

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None
    assert vor.ninst == 1


def test_gtess3d_from_seed_points_many_seeds():
    """Create gtess3d with many seed points."""
    np.random.seed(42)
    seed_points = np.random.uniform(0, 100, size=(50, 3))
    bounds = [[0, 100], [0, 100], [0, 100]]

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None
    assert len(vor.pxtals[1]) == 50


def test_gtess3d_from_seed_points_partial_periodic():
    """Create gtess3d with periodic BCs on some axes only."""
    seed_points = np.array([
        [10, 10, 10],
        [30, 30, 30]
    ], dtype=float)
    bounds = [[0, 40], [0, 40], [0, 40]]
    periodic = (True, False, True)  # Periodic in x and z, not y

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds, periodic=periodic)
    assert vor is not None
    assert vor.info['uinputs']['periodic'] == [True, False, True]


def test_gtess3d_from_seed_points_negative_coords():
    """Create gtess3d with negative coordinate seeds."""
    seed_points = np.array([
        [-10, -10, -10],
        [0, 0, 0],
        [10, 10, 10]
    ], dtype=float)
    bounds = [[-20, 20], [-20, 20], [-20, 20]]

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None


def test_gtess3d_from_seed_points_returns_gtess3d():
    """from_seed_points returns a gtess3d instance."""
    seed_points = np.array([[15, 15, 15]], dtype=float)
    bounds = [[0, 30], [0, 30], [0, 30]]

    vor = gtess3d.from_seed_points(seed_points, bounds=bounds)
    assert isinstance(vor, gtess3d)
