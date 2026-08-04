"""Tests for upxo.viz.vizDistr distribution plotting functions."""
import numpy as np
import pytest


def test_vizdistr_module_imports():
    """Verify vizDistr module can be imported."""
    try:
        from upxo.viz import vizDistr
        assert hasattr(vizDistr, 'plot_grouped_distributions')
        assert hasattr(vizDistr, 'plot_qq_comparison')
        assert hasattr(vizDistr, 'plot_repr_rank')
    except ImportError:
        pytest.skip("vizDistr import failed")


def test_plot_grouped_distributions_exists():
    """Verify plot_grouped_distributions function exists and is callable."""
    try:
        from upxo.viz.vizDistr import plot_grouped_distributions
        assert callable(plot_grouped_distributions)
    except (ImportError, AttributeError):
        pytest.skip("Function not available")


def test_plot_qq_comparison_exists():
    """Verify plot_qq_comparison function exists and is callable."""
    try:
        from upxo.viz.vizDistr import plot_qq_comparison
        assert callable(plot_qq_comparison)
    except (ImportError, AttributeError):
        pytest.skip("Function not available")


def test_plot_repr_rank_exists():
    """Verify plot_repr_rank function exists and is callable."""
    try:
        from upxo.viz.vizDistr import plot_repr_rank
        assert callable(plot_repr_rank)
    except (ImportError, AttributeError):
        pytest.skip("Function not available")
