"""Tests for upxo.viz.ebsdviz grain-role distribution functions."""
import numpy as np
import pytest
from upxo.viz.ebsdviz import (
    compute_pct_interior_grains,
    compute_grain_role_property_distributions,
    plot_grain_role_property_distributions
)


def test_compute_pct_interior_grains_empty():
    """Compute interior grain percentage with empty grain list."""
    lfi = np.array([[1, 1], [2, 2]], dtype=int)
    pct = compute_pct_interior_grains(lfi, [])
    assert pct is None or pct == 0


def test_compute_pct_interior_grains_all_interior():
    """All grains are interior (none touch boundary)."""
    lfi = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)
    pct = compute_pct_interior_grains(lfi, [1])
    assert pct == 100.0 or pct == 1.0


def test_compute_pct_interior_grains_boundary_grains():
    """Compute percentage with boundary-touching grains."""
    lfi = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 3]
    ], dtype=int)
    pct = compute_pct_interior_grains(lfi, [1, 2, 3])
    assert isinstance(pct, (int, float))
    assert 0 <= pct <= 100


def test_compute_grain_role_property_distributions_basic():
    """Compute property distributions by grain role."""
    lfi = np.ones((5, 5), dtype=int)
    grain_ids = np.arange(1, 6)
    grain_roles = {i: ('host' if i % 2 == 0 else 'twin') for i in grain_ids}
    properties = {i: np.random.rand() for i in grain_ids}

    dists = compute_grain_role_property_distributions(
        lfi, grain_ids, grain_roles, 'test_prop', properties
    )
    assert isinstance(dists, dict)


def test_compute_grain_role_property_distributions_multiple_roles():
    """Distributions across multiple grain roles."""
    lfi = np.ones((10, 10), dtype=int)
    grain_ids = range(1, 11)
    grain_roles = {
        i: 'host' if i < 4 else 'primary_twin' if i < 7 else 'secondary_twin'
        for i in grain_ids
    }
    properties = {i: float(i) for i in grain_ids}

    dists = compute_grain_role_property_distributions(
        lfi, grain_ids, grain_roles, 'size', properties
    )
    assert len(dists) == 3  # Three roles


def test_compute_grain_role_property_distributions_empty_role():
    """Handle a role with no grains."""
    lfi = np.ones((5, 5), dtype=int)
    grain_ids = [1, 2, 3]
    grain_roles = {1: 'host', 2: 'host', 3: 'host'}  # No twins
    properties = {1: 1.0, 2: 2.0, 3: 3.0}

    dists = compute_grain_role_property_distributions(
        lfi, grain_ids, grain_roles, 'test', properties
    )
    assert isinstance(dists, dict)


def test_plot_grain_role_property_distributions_smoke():
    """Smoke test for plotting function."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        lfi = np.ones((5, 5), dtype=int)
        grain_ids = [1, 2, 3]
        grain_roles = {1: 'host', 2: 'twin', 3: 'host'}
        properties = {1: 1.0, 2: 2.0, 3: 3.0}

        fig = plot_grain_role_property_distributions(
            lfi, grain_ids, grain_roles, 'test_prop', properties
        )
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_compute_grain_role_property_distributions_single_grain():
    """Single grain distribution."""
    lfi = np.ones((5, 5), dtype=int)
    grain_ids = [1]
    grain_roles = {1: 'host'}
    properties = {1: 42.0}

    dists = compute_grain_role_property_distributions(
        lfi, grain_ids, grain_roles, 'prop', properties
    )
    assert isinstance(dists, dict)
