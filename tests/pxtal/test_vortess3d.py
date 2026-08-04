"""Tests for upxo.pxtal.vortess3d Voronoi constructors."""
import numpy as np
import pytest
from upxo.pxtal.vortess3d import Voronoi3D


def test_voronoi3d_from_seed_points_basic():
    """Create Voronoi3D from seed points."""
    seed_points = np.array([
        [10, 10, 10],
        [30, 30, 30],
        [50, 50, 50]
    ], dtype=float)
    bounds = (0, 60, 0, 60, 0, 60)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None
    assert hasattr(vor, 'seed_points')


def test_voronoi3d_from_mpoint3d():
    """Create Voronoi3D from MPoint3D seed points."""
    try:
        from upxo.geoEntities.mpoint3d import MPoint3D

        mpts = [
            MPoint3D(10, 10, 10),
            MPoint3D(30, 30, 30),
            MPoint3D(50, 50, 50)
        ]
        bounds = (0, 60, 0, 60, 0, 60)

        vor = Voronoi3D.from_mpoint3d(mpts, bounds=bounds)
        assert vor is not None
    except ImportError:
        pytest.skip("MPoint3D not available")


def test_voronoi3d_from_seed_points_with_periodic():
    """Create Voronoi3D with periodic boundary conditions."""
    seed_points = np.array([
        [5, 5, 5],
        [15, 15, 15],
        [25, 25, 25]
    ], dtype=float)
    bounds = (0, 30, 0, 30, 0, 30)
    periodic = (True, True, True)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds, periodic=periodic)
    assert vor is not None


def test_voronoi3d_from_seed_points_no_bounds():
    """Create Voronoi3D without explicit bounds (auto-inferred)."""
    seed_points = np.array([
        [0, 0, 0],
        [10, 10, 10],
        [20, 20, 20]
    ], dtype=float)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=None)
    assert vor is not None


def test_voronoi3d_from_seed_points_single_seed():
    """Create Voronoi3D with single seed point."""
    seed_points = np.array([[15, 15, 15]], dtype=float)
    bounds = (0, 30, 0, 30, 0, 30)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None


def test_voronoi3d_from_seed_points_many_seeds():
    """Create Voronoi3D with many seed points."""
    np.random.seed(42)
    seed_points = np.random.uniform(0, 100, size=(50, 3))
    bounds = (0, 100, 0, 100, 0, 100)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None


def test_voronoi3d_from_seed_points_partial_periodic():
    """Create Voronoi3D with periodic BCs on some axes only."""
    seed_points = np.array([
        [10, 10, 10],
        [30, 30, 30]
    ], dtype=float)
    bounds = (0, 40, 0, 40, 0, 40)
    periodic = (True, False, True)  # Periodic in x and z, not y

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds, periodic=periodic)
    assert vor is not None


def test_voronoi3d_from_seed_points_negative_coords():
    """Create Voronoi3D with negative coordinate seeds."""
    seed_points = np.array([
        [-10, -10, -10],
        [0, 0, 0],
        [10, 10, 10]
    ], dtype=float)
    bounds = (-20, 20, -20, 20, -20, 20)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds)
    assert vor is not None


def test_voronoi3d_from_seed_points_returns_voronoi3d():
    """from_seed_points returns a Voronoi3D instance."""
    seed_points = np.array([[15, 15, 15]], dtype=float)
    bounds = (0, 30, 0, 30, 0, 30)

    vor = Voronoi3D.from_seed_points(seed_points, bounds=bounds)
    assert isinstance(vor, Voronoi3D)
