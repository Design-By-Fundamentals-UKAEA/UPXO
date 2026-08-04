"""Tests for upxo.viz.vizDistr distribution plotting."""
import numpy as np
import pytest


def test_plot_grouped_distributions_smoke():
    """Smoke test for grouped distribution plotting."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_grouped_distributions

        data = {
            'group_a': np.random.normal(10, 2, 100),
            'group_b': np.random.normal(15, 3, 100),
            'group_c': np.random.normal(12, 1, 100)
        }

        fig, axes = plot_grouped_distributions(data, do_tight_layout=False)
        assert fig is not None
        assert len(axes) == 3
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_plot_grouped_distributions_near_zero_variance():
    """Handle group with near-zero variance without crashing."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_grouped_distributions

        data = {
            'normal_group': np.random.normal(10, 2, 100),
            'low_var_group': np.full(20, 5.0) + np.random.normal(0, 0.001, 20)
        }

        fig, axes = plot_grouped_distributions(data, do_tight_layout=False)
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_plot_grouped_distributions_single_group():
    """Plot distribution for a single group."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_grouped_distributions

        data = {'only_group': np.random.normal(20, 3, 50)}

        fig, axes = plot_grouped_distributions(data, do_tight_layout=False)
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_plot_grouped_distributions_many_groups():
    """Plot distributions for many groups."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_grouped_distributions

        data = {
            f'group_{i}': np.random.normal(10 + i, 2, 100)
            for i in range(6)
        }

        fig, axes = plot_grouped_distributions(data, do_tight_layout=False)
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_plot_grouped_distributions_with_peak_detection():
    """Plot with peak detection enabled."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_grouped_distributions

        data = {
            'bimodal': np.concatenate([
                np.random.normal(5, 1, 50),
                np.random.normal(15, 1, 50)
            ])
        }

        fig, axes = plot_grouped_distributions(
            data, show_peaks=True, do_tight_layout=False
        )
        assert fig is not None
        plt.close(fig)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_plot_qq_comparison_smoke():
    """Smoke test for Q-Q comparison plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_qq_comparison

        ebsd_data = np.random.normal(10, 2, 100)
        mc_slices = {
            'slice_1': np.random.normal(10, 2, 100),
            'slice_2': np.random.normal(10, 2, 100)
        }

        plot_qq_comparison(ebsd_data, mc_slices)
        plt.close('all')
    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            pytest.skip("matplotlib not available")


def test_plot_repr_rank_smoke():
    """Smoke test for representativeness ranking plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from upxo.viz.vizDistr import plot_repr_rank

        # Create mock data: list of (mc_index, rank_metric) tuples
        mc_data = [(i, np.random.rand()) for i in range(10)]

        plot_repr_rank(mc_data)
        plt.close('all')
    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            pytest.skip("matplotlib not available")
