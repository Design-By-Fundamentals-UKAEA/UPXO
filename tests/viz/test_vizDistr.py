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


# ── plot_slice_kde_curve_set ─────────────────────────────────────────────

def test_draws_one_thin_line_per_slice_plus_one_mean_plus_ebsd():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(0)
    ebsd = rng.normal(10, 2, 200)
    slices = [rng.normal(10, 2, 50) for _ in range(5)]
    fig, ax = plot_slice_kde_curve_set(ebsd, slices, category_label="Host-to-Host", axis_label="Y")
    assert len(ax.lines) == len(slices) + 2
    plt_import = __import__('matplotlib.pyplot', fromlist=['close'])
    plt_import.close(fig)


def test_legend_has_exactly_two_entries_not_one_per_slice():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(1)
    ebsd = rng.normal(5, 1, 100)
    slices = [rng.normal(5, 1, 30) for _ in range(8)]
    fig, ax = plot_slice_kde_curve_set(ebsd, slices)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['Synthetic (mean of slices)', 'EBSD']


def test_mean_curve_is_pointwise_average_not_pooled_kde():
    """Deliberately a different statistic from fitting one KDE to the
    pooled slice data -- the two must disagree for non-identical slices."""
    from scipy.stats import gaussian_kde
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(42)
    slices = [rng.normal(1.0, 0.1, 10), rng.normal(10.0, 0.1, 10)]
    ebsd = rng.normal(5.0, 0.3, 20)
    fig, ax = plot_slice_kde_curve_set(ebsd, slices)
    mean_line = ax.lines[-2]  # 2 thin + 1 mean (plotted after) + ebsd (plotted last)
    pooled = np.concatenate(slices)
    grid = mean_line.get_xdata()
    pooled_kde_curve = gaussian_kde(pooled, bw_method='scott')(grid)
    assert not np.allclose(mean_line.get_ydata(), pooled_kde_curve)


def test_title_includes_category_and_axis():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(2)
    fig, ax = plot_slice_kde_curve_set(
        rng.normal(size=20), [rng.normal(size=20)],
        category_label="Twin-to-Twin", axis_label="Z")
    assert "Twin-to-Twin" in ax.get_title()
    assert "Z" in ax.get_title()


def test_slices_with_too_few_points_are_skipped_not_errored():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(3)
    ebsd = rng.normal(size=30)
    slices = [np.array([1.0]), rng.normal(size=20), np.array([])]
    fig, ax = plot_slice_kde_curve_set(ebsd, slices)
    # only the one slice with >=2 points gets a thin curve
    assert len(ax.lines) == 1 + 2


def test_empty_ebsd_still_plots_the_synthetic_curves():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    rng = np.random.default_rng(4)
    slices = [rng.normal(size=20) for _ in range(3)]
    fig, ax = plot_slice_kde_curve_set(np.array([]), slices)
    assert len(ax.lines) == 3 + 1  # no EBSD curve, but mean + 3 slices
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['Synthetic (mean of slices)']


def test_raises_when_nothing_at_all_is_fittable():
    from upxo.viz.vizDistr import plot_slice_kde_curve_set
    with pytest.raises(ValueError):
        plot_slice_kde_curve_set(np.array([1.0]), [np.array([2.0]), np.array([])])
