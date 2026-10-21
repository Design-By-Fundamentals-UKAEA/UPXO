"""Density plots of one or several samples."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from upxo.viz import distribution_plots as dp


@pytest.fixture()
def ax():
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


def _sample(n=200, seed=0, scale=1.0):
    return np.random.default_rng(seed).normal(scale=scale, size=n)


# ── bin count ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('n, expected', [(3, 5), (25, 5), (100, 20), (1000, 30)])
def test_automatic_bin_count(n, expected):
    assert dp.histogram_bin_count(np.arange(float(n))) == expected


@pytest.mark.parametrize('values, width, expected', [
    (np.linspace(0, 10, 50), 2.0, 5),
    (np.linspace(0, 10, 50), 100.0, 3),            # never fewer than 3
    (np.linspace(0, 1000, 50), 1.0, 200),          # never more than 200
    (np.full(10, 4.0), 1.0, 5),                    # all values equal
])
def test_bin_count_from_a_bin_width(values, width, expected):
    assert dp.histogram_bin_count(values, width) == expected


# ── one sample ────────────────────────────────────────────────────────────────

def test_kde_draws_a_line_and_a_shaded_area(ax):
    values = _sample()
    shown = dp.plot_kde(ax, values, 'size', 'distribution')
    assert len(ax.lines) == 1 and len(ax.collections) == 1
    xs = ax.lines[0].get_xdata()
    assert xs[0] == pytest.approx(values.min()) and xs[-1] == pytest.approx(values.max()) and len(xs) == 200
    np.testing.assert_array_equal(shown, values)
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ('size', 'Density', 'distribution')


def test_non_finite_values_are_dropped(ax):
    shown = dp.plot_kde(ax, [1.0, np.nan, 2.0, np.inf, 5.0, 3.0], 'x', 't')
    np.testing.assert_array_equal(shown, [1.0, 2.0, 5.0, 3.0])


def test_a_single_value_is_a_vertical_line(ax):
    shown = dp.plot_kde(ax, [3.0], 'x', 't')
    assert len(ax.lines) == 1 and list(ax.lines[0].get_xdata()) == [3.0, 3.0]
    np.testing.assert_array_equal(shown, [3.0])


def test_no_values(ax):
    assert dp.plot_kde(ax, [], 'x', 't') is None
    assert dp.plot_kde(ax, [np.nan], 'x', 't') is None
    assert [t.get_text() for t in ax.texts] == ['No data available']


def test_the_axes_are_cleared_first(ax):
    dp.plot_kde(ax, _sample(), 'x', 't')
    dp.plot_kde(ax, _sample(seed=1), 'x', 't')
    assert len(ax.lines) == 1 and len(ax.collections) == 1


def test_a_bin_width_sets_the_bandwidth(ax):
    values = _sample()
    dp.plot_kde(ax, values, 'x', 't')
    default_peak = ax.lines[0].get_ydata().max()
    dp.plot_kde(ax, values, 'x', 't', bin_width=0.05)
    assert ax.lines[0].get_ydata().max() > default_peak              # narrower kernel, taller peak


def test_a_histogram_is_drawn_if_the_estimate_fails(ax, monkeypatch):
    import scipy.stats

    def broken(*args, **kwargs):
        raise RuntimeError('no estimate')

    monkeypatch.setattr(scipy.stats, 'gaussian_kde', broken)
    dp.plot_kde(ax, _sample(100), 'x', 't', bin_width=0.5)
    assert len(ax.lines) == 0 and len(ax.patches) == dp.histogram_bin_count(_sample(100), 0.5)


def test_limits_and_font_sizes(ax):
    dp.plot_kde(ax, _sample(), 'x', 't', xlim=(-5, 5), ylim=(0, 2), font_sizes={"tick": 5, "title": 12, "axis": 10, "legend": 4})
    assert ax.get_xlim() == (-5, 5) and ax.get_ylim() == (0, 2)
    assert ax.title.get_fontsize() == 12 and ax.xaxis.label.get_fontsize() == 10
    assert ax.get_xticklabels()[0].get_fontsize() == 5


def test_default_font_sizes(ax):
    dp.plot_kde(ax, _sample(), 'x', 't')
    assert ax.title.get_fontsize() == dp.DEFAULT_FONT_SIZES['title']


# ── several samples ───────────────────────────────────────────────────────────

def test_overlay_of_two_samples(ax):
    a, b = _sample(100, 0), _sample(80, 1, 2.0)
    pooled = dp.plot_kde_overlay(ax, [('A', a, '#3182CE'), ('B', b, '#DD6B20')], 'x', 't')
    assert len(ax.lines) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['A (n=100)', 'B (n=80)']
    assert len(ax.patches) == dp.histogram_bin_count(a) + dp.histogram_bin_count(b)
    np.testing.assert_array_equal(pooled, np.concatenate([a, b]))


def test_overlay_skips_series_without_values(ax):
    pooled = dp.plot_kde_overlay(ax, [('A', _sample(50), 'k'), ('B', [np.nan], 'r'), ('C', [], 'g')], 'x', 't')
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['A (n=50)']
    assert len(pooled) == 50


def test_overlay_without_any_values(ax):
    assert dp.plot_kde_overlay(ax, [('A', [], 'k')], 'x', 't') is None
    assert ax.get_legend() is None and [t.get_text() for t in ax.texts] == ['No data available']


def test_overlay_draws_a_line_only_for_samples_with_more_than_three_distinct_values(ax):
    dp.plot_kde_overlay(ax, [('few', [1.0, 2.0, 3.0, 3.0, 3.0], 'k'), ('many', _sample(30), 'b')], 'x', 't')
    assert len(ax.lines) == 1


def test_overlay_legend_is_never_smaller_than_six_points(ax):
    dp.plot_kde_overlay(ax, [('A', _sample(30), 'k')], 'x', 't', font_sizes={"tick": 3, "title": 3, "axis": 3, "legend": 2})
    assert ax.get_legend().get_texts()[0].get_fontsize() == 6


def test_overlay_limits_and_labels(ax):
    dp.plot_kde_overlay(ax, [('A', _sample(30), 'k')], 'size', 'title', ylabel='Count', xlim=(-4, 4), ylim=(0, 1))
    assert ax.get_xlim() == (-4, 4) and ax.get_ylim() == (0, 1)
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ('size', 'Count', 'title')
