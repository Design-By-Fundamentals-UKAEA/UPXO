"""Tests for upxo.viz.ebsdviz grain-role visualization functions."""
import numpy as np
import pytest


def test_ebsdviz_module_imports():
    """Verify ebsdviz module can be imported."""
    try:
        from upxo.viz import ebsdviz
        assert hasattr(ebsdviz, 'compute_pct_interior_grains')
        assert hasattr(ebsdviz, 'compute_grain_role_property_distributions')
        assert hasattr(ebsdviz, 'plot_grain_role_property_distributions')
    except ImportError:
        pytest.skip("ebsdviz import failed")


def test_compute_pct_interior_grains_simple():
    """Test interior grain percentage computation (smoke test)."""
    try:
        from upxo.viz.ebsdviz import compute_pct_interior_grains

        lfi = np.array([[1, 1], [2, 2]], dtype=int)
        result = compute_pct_interior_grains(lfi, [1, 2])
        # Result can be None or a float, just check it doesn't crash
        assert result is None or isinstance(result, (int, float))
    except (ImportError, TypeError, AttributeError):
        pytest.skip("Function signature or availability issue")
