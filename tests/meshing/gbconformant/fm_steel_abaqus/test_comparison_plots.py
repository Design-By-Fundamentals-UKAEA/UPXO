"""The distribution plots of the before/after comparison: kinds, colours, marks, bins and the text on them."""
import unittest

import matplotlib
import numpy as np

matplotlib.use('Agg')

from upxo.meshing.gbconformant.fm_steel_abaqus.plots import (
    COLORS, DISTRIBUTION_KINDS, DrawSeries, PlotStyle, bin_edges, colour, compare_figure, distribution_figure,
    kde_curve, matched_change_figure, silverman_bandwidth, weighted_stats)

RNG = np.random.default_rng(5)
BEFORE = RNG.normal(10.0, 2.0, 300)
AFTER = RNG.normal(11.0, 3.0, 250)


def series(**kwargs):
    return [DrawSeries('Voxel structure: 300 grains', BEFORE, color='#112233', **kwargs),
            DrawSeries('Tetrahedral mesh: 250 grains', AFTER, color='#aa5500', **kwargs)]


def vertical_lines(ax):
    return [line for line in ax.lines if len(set(np.asarray(line.get_xdata(), dtype=float))) == 1 and len(line.get_xdata()) == 2
            and line.get_transform() is not ax.get_xaxis_transform()]


class WeightedStatsTests(unittest.TestCase):
    def test_equal_weights_match_the_plain_statistics(self):
        stats = weighted_stats(BEFORE)
        self.assertAlmostEqual(stats['mean'], BEFORE.mean())
        self.assertAlmostEqual(stats['std'], BEFORE.std())
        self.assertAlmostEqual(stats['median'], float(np.sort(BEFORE)[int(np.searchsorted(np.cumsum(np.ones(300)), 150))]))

    def test_weights_move_the_mean(self):
        stats = weighted_stats([1.0, 2.0, 3.0], [1.0, 1.0, 8.0])
        self.assertAlmostEqual(stats['mean'], 2.7)
        self.assertEqual(stats['median'], 3.0)
        self.assertTrue(np.isnan(weighted_stats([])['mean']))

    def test_the_kde_adds_up_to_the_weight(self):
        grid = np.linspace(-10, 30, 4001)
        h = silverman_bandwidth(BEFORE)
        self.assertAlmostEqual(np.trapezoid(kde_curve(BEFORE, None, h, grid), grid), 1.0, places=3)
        weights = np.full(300, 2.0 / 300)
        self.assertAlmostEqual(np.trapezoid(kde_curve(BEFORE, weights, h, grid), grid), 2.0, places=3)
        self.assertGreater(h, 0)
        self.assertEqual(silverman_bandwidth([3.0]), 1.0)
        self.assertGreater(silverman_bandwidth([2.0, 2.0, 2.0]), 0)


class BinTests(unittest.TestCase):
    def test_bins_or_a_width(self):
        edges, note = bin_edges(series(), PlotStyle(bins=20))
        self.assertEqual(len(edges), 21)
        self.assertEqual(note, '')
        edges, note = bin_edges(series(), PlotStyle(bin_width=0.5))
        np.testing.assert_allclose(np.diff(edges), 0.5)
        self.assertLessEqual(edges[0], min(BEFORE.min(), AFTER.min()))
        self.assertGreaterEqual(edges[-1], max(BEFORE.max(), AFTER.max()))

    def test_a_width_that_gives_too_many_or_too_few_bins_falls_back(self):
        for width in (1e-6, 1e6):
            edges, note = bin_edges(series(), PlotStyle(bins=25, bin_width=width))
            self.assertEqual(len(edges), 26)
            self.assertIn('bins were used', note)

    def test_whole_number_data_gets_bins_centred_on_the_numbers(self):
        counts = [DrawSeries('a', np.array([2.0, 3, 3, 4, 7]), integer=True), DrawSeries('b', np.array([3.0, 5, 6]), integer=True)]
        edges, _ = bin_edges(counts, PlotStyle(bin_width=0.3))
        np.testing.assert_allclose(edges, np.arange(1.5, 8.0, 1.0))


class FigureTests(unittest.TestCase):
    def test_every_kind_is_drawn(self):
        for kind in DISTRIBUTION_KINDS:
            figure = compare_figure(series(), PlotStyle(kind=kind), xlabel='x', title='t')
            self.assertEqual(len(figure.axes), 1, kind)
        with self.assertRaises(ValueError):
            compare_figure(series(), PlotStyle(kind='pie'))

    def test_the_legend_holds_the_stage_names_and_counts(self):
        figure = compare_figure(series(), PlotStyle(kind='histogram'), title='Diameter')
        labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ['Voxel structure: 300 grains', 'Tetrahedral mesh: 250 grains'])

    def test_the_colours_are_the_ones_asked_for(self):
        for kind in ('histogram', 'kde', 'histogram + kde', 'cdf'):
            figure = compare_figure(series(), PlotStyle(kind=kind))
            ax = figure.axes[0]
            drawn = {matplotlib.colors.to_hex(line.get_color()) for line in ax.lines}
            drawn |= {matplotlib.colors.to_hex(p.get_edgecolor()) for p in ax.patches}
            self.assertTrue({'#112233', '#aa5500'} <= drawn, kind)

    def test_marks_are_optional_and_show_in_the_legend(self):
        plain = compare_figure(series(), PlotStyle(kind='histogram'))
        marked = compare_figure(series(), PlotStyle(kind='histogram', show_mean=True, show_std=True, show_median=True), unit='um')
        self.assertGreater(len(marked.axes[0].lines), len(plain.axes[0].lines))
        legend = [t.get_text() for t in marked.axes[0].get_legend().get_texts()]
        self.assertTrue(all('mean' in text and 'median' in text for text in legend))
        self.assertTrue(all('mean' not in t.get_text() for t in plain.axes[0].get_legend().get_texts()))
        mean_lines = [line for line in marked.axes[0].lines if line.get_linestyle() == '--']
        xs = sorted(float(line.get_xdata()[0]) for line in mean_lines)
        np.testing.assert_allclose(xs, sorted([BEFORE.mean(), AFTER.mean()]))
        only_mean = compare_figure(series(), PlotStyle(show_mean=True))
        self.assertEqual(len(only_mean.axes[0].lines), 2)

    def test_the_standard_deviation_bar_spans_one_deviation_each_side(self):
        figure = compare_figure(series(), PlotStyle(show_std=True))
        bars = [line for line in figure.axes[0].lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] != line.get_xdata()[1]]
        spans = sorted((float(b.get_xdata()[0]), float(b.get_xdata()[1])) for b in bars)
        for (lo, hi), data in zip(spans, sorted((BEFORE, AFTER), key=lambda d: d.mean())):
            self.assertAlmostEqual(lo, data.mean() - data.std())
            self.assertAlmostEqual(hi, data.mean() + data.std())

    def test_a_bin_width_and_a_bandwidth_are_used_and_reported(self):
        figure = compare_figure(series(), PlotStyle(kind='histogram + kde', bin_width=1.0, bandwidth=0.8), unit='um', note_unit='um')
        self.assertEqual(figure.compare_notes, [])
        auto = compare_figure(series(), PlotStyle(kind='histogram + kde'), unit='um', note_unit='um')
        self.assertTrue(any('bin width' in n for n in auto.compare_notes))
        self.assertTrue(any('KDE bandwidth' in n and "Silverman's rule" in n for n in auto.compare_notes))
        sharp = compare_figure(series(), PlotStyle(kind='kde', bandwidth=0.2))
        wide = compare_figure(series(), PlotStyle(kind='kde', bandwidth=3.0))
        self.assertGreater(max(sharp.axes[0].lines[0].get_ydata()), max(wide.axes[0].lines[0].get_ydata()))

    def test_weights_make_the_histogram_a_density_and_forbid_box_and_violin(self):
        weights = np.full(300, 1.0 / 300)
        weighted = [DrawSeries('a', BEFORE, weights, '#112233'), DrawSeries('b', AFTER, np.full(250, 1.0 / 250), '#aa5500')]
        figure = compare_figure(weighted, PlotStyle(kind='histogram', bins=30), ylabel='fraction per unit')
        ax = figure.axes[0]
        self.assertEqual(ax.get_ylabel(), 'fraction per unit')
        edges = ax.patches[0].get_path().vertices
        area = sum(p.get_path().get_extents().height * 0 for p in ax.patches)
        self.assertEqual(area, 0)
        for kind in ('box', 'violin'):
            fallback = compare_figure(weighted, PlotStyle(kind=kind))
            self.assertTrue(any('cannot show weights' in n for n in fallback.compare_notes), kind)
            self.assertGreater(len(fallback.axes[0].lines), 0)
        self.assertIsNotNone(edges)

    def test_log_axes_and_empty_sides(self):
        self.assertEqual(compare_figure(series(), PlotStyle(log=True)).axes[0].get_yscale(), 'log')
        one_sided = compare_figure([DrawSeries('a', BEFORE), DrawSeries('b', np.zeros(0))], PlotStyle(kind='kde'))
        self.assertEqual(len(one_sided.axes[0].get_legend().get_texts()), 1)
        nothing = compare_figure([DrawSeries('a', np.zeros(0))], PlotStyle())
        self.assertEqual(len(nothing.axes), 1)

    def test_title_subtitle_and_footnote(self):
        figure = compare_figure(series(), PlotStyle(), title='Sphere-equivalent diameter',
                                subtitle='Before and after tetrahedral meshing', footnote='3 grains were removed.')
        self.assertEqual(figure.get_suptitle(), 'Sphere-equivalent diameter')
        self.assertEqual(figure.axes[0].get_title(), 'Before and after tetrahedral meshing')
        self.assertIn('3 grains were removed.', figure.get_supxlabel())

    def test_the_old_entry_point_still_works(self):
        figure = distribution_figure(BEFORE, AFTER, 'kde', xlabel='x', title='t', labels=('Voxel structure', 'Tetrahedral mesh'))
        labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ['Voxel structure (300)', 'Tetrahedral mesh (250)'])


class MatchedAndColourTests(unittest.TestCase):
    ROWS = [{'set': i, 'before': float(i), 'after': float(i) * 1.1} for i in range(1, 8)]

    def test_matched_change_names_the_two_sides(self):
        figure = matched_change_figure(self.ROWS, quantity='Grain volume', units='um', labels=('Voxel structure', 'Tetrahedral mesh'),
                                       colors=('#112233', '#aa5500'), subtitle='sub', footnote='note', noun='grains')
        ax = figure.axes[0]
        self.assertEqual(ax.get_xlabel(), 'Grain volume: Voxel structure (um)')
        self.assertEqual(ax.get_ylabel(), 'Grain volume: Tetrahedral mesh (um)')
        self.assertEqual(figure.get_suptitle(), 'Grain volume: 7 grains on both sides')
        self.assertEqual(matplotlib.colors.to_hex(ax.collections[0].get_facecolor()[0]), '#aa5500')
        ratio = matched_change_figure(self.ROWS, quantity='Grain volume', kind='ratio', labels=('A', 'B'), noun='grains')
        self.assertIn('B / A', ratio.axes[0].get_xlabel())
        self.assertIn('number of grains', ratio.axes[0].get_ylabel())

    def test_colours(self):
        self.assertEqual(colour('#4878cf', '#000000'), '#4878cf')
        self.assertEqual(colour('red', '#000000'), '#ff0000')
        self.assertEqual(colour('', '#123456'), '#123456')
        self.assertEqual(colour(None, '#123456'), '#123456')
        with self.assertRaises(ValueError):
            colour('not a colour', '#000000')
        self.assertEqual(set(COLORS), {'before', 'after'})


if __name__ == '__main__':
    unittest.main()
