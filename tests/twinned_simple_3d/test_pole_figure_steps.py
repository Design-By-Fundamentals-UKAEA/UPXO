import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upxo.pxtal.twinned_simple_3d.steps import steps_visualization_export as sve
from upxo.pxtal.twinned_simple_3d.steps import steps_distribution_viewer as sdv


def stage(n=12, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    q[q[:, 0] < 0] *= -1
    return {'gids': np.arange(1, n + 1), 'quats': q}


class PoleFigureStepTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_sample_frame_conversion_and_symmetry(self):
        s = stage()
        q, g = sve.sample_frame_quats(s, apply_sample_symmetry=False)
        np.testing.assert_allclose(q[:, 0], s['quats'][:, 0])
        np.testing.assert_allclose(q[:, 1:], -s['quats'][:, 1:])
        np.testing.assert_array_equal(g, s['gids'])
        q, g = sve.sample_frame_quats(s)                       # default: RD, TD, ND -> 4 operations
        self.assertEqual(len(q), 4 * len(s['quats']))
        self.assertEqual(len(g), len(q))
        q, _ = sve.sample_frame_quats(s, use_td=False, use_nd=False)
        self.assertEqual(len(q), 2 * len(s['quats']))
        self.assertEqual(len(s['quats']), 12)                  # input untouched

    def test_plot_modes_and_validation(self):
        for mode in ('scatter', 'density', 'hybrid'):
            with self.subTest(mode=mode):
                fig, ax = sve.plot_pole_figure(stage(), pole_family='111', plot_mode=mode, grid_points=40)
                self.assertIsNotNone(ax)
        with self.assertRaises(ValueError):
            sve.plot_pole_figure(stage(), plot_mode='other')

    def test_comparison_both_modes(self):
        for mode in ('scatter', 'density'):
            with self.subTest(mode=mode):
                fig, axes = sve.plot_pole_figure_comparison(stage(seed=1), stage(9, seed=2), plot_mode=mode,
                                                            grid_points=40)
                self.assertEqual(len(axes), 3)
                self.assertIn('(12, 48)', axes[0].get_title())          # grains, plotted orientations
        with self.assertRaises(ValueError):
            sve.plot_pole_figure_comparison(stage(), stage(), plot_mode='hybrid')

    def test_overlay_and_residual_accept_symmetry_options(self):
        sve.plot_pole_figure_overlay(stage(seed=3), 'A', stage(seed=4), 'B', apply_sample_symmetry=False)
        fig, axes, iqr = sdv.plot_texture_residual(stage(seed=5), stage(seed=6), grid_points=40)
        self.assertTrue(np.isfinite(iqr))

    def test_pretwin_populations(self):
        orientations = {g: q for g, q in zip(range(1, 6), stage(5)['quats'])}
        assigner = SimpleNamespace(all_grain_orientations=orientations)
        base = SimpleNamespace(host_grain_ids={2, 4, 99})
        ebsd = stage(3, seed=7)
        active = ebsd['quats'].copy(); active[:, 1:] *= -1
        with patch('upxo.pxtal.twinned_simple_3d.orientation_3d.compute_ebsd_pure_parents_for_csl',
                   return_value=(active, ebsd['gids'])):
            pops = sve.pretwin_pole_figure_populations(None, None, base, assigner, 'S3')
        np.testing.assert_allclose(pops['pure_parents']['quats'], ebsd['quats'])
        np.testing.assert_array_equal(pops['hosts']['gids'], [2, 4])
        self.assertEqual(len(pops['all']['gids']), 5)


if __name__ == '__main__':
    unittest.main()
