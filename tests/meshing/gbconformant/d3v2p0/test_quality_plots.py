import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upxo.meshing.gbconformant.d3v2p0 import quality_plots as qp

MODULE = 'upxo.meshing.gbconformant.d3v2p0.quality_plots'


class QualityPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_triangle_distribution(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .01]])
        f = np.array([[0, 1, 2], [1, 3, 2]])
        figs, stats = qp.triangle_quality_distribution(p, f, exterior=[False, True])
        self.assertEqual(len(figs), 3)
        self.assertEqual(stats['triangles'], 2)
        self.assertAlmostEqual(stats['smallest_angle']['worst'], 45., places=0)
        self.assertEqual(stats['smallest_angle']['best_possible'], 60.)
        self.assertEqual(stats['below_min_angle_limit'], 0)
        with self.assertRaises(ValueError):
            qp.triangle_quality_distribution(p, f, fraction=0)

    def test_tet_distribution(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
        figs, stats = qp.tet_quality_distribution(p, [[0, 1, 2, 3]], quality=[.5])
        self.assertEqual(len(figs), 3)
        self.assertAlmostEqual(stats['largest_dihedral']['worst'], 90.)
        self.assertAlmostEqual(stats['largest_dihedral']['best_possible'], np.degrees(np.arccos(1 / 3)))
        self.assertEqual(stats['minSICN']['count'], 1)
        figs, stats = qp.tet_quality_distribution(p, [[0, 1, 2, 3]])
        self.assertEqual(len(figs), 2)
        self.assertNotIn('minSICN', stats)

    def test_worst_band(self):
        v = np.array([10., 12., 14., 30., 50.])
        edge, count, worst = qp._worst_band(v, .1, 60., 'low')          # [10, 10 + 0.1*50] = [10, 15]
        self.assertEqual((edge, count, worst), (15., 3, 10.))
        v = np.array([170., 168., 150., 100.])
        edge, count, worst = qp._worst_band(v, .1, 60., 'high')         # [170 - 0.1*110, 170] = [159, 170]
        self.assertEqual((edge, count, worst), (159., 2, 170.))

    def test_surface_view_launch_and_options(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 's.vtp'); open(path, 'w').close()
            with patch(MODULE + '.subprocess.Popen') as popen:
                qp.launch_surface_view(path, show_edges=False, color_by='min_angle')
            cmd = popen.call_args[0][0]
            self.assertIn('--no-edges', cmd)
            self.assertEqual(cmd[cmd.index('--color-by') + 1], 'min_angle')
        with self.assertRaises(FileNotFoundError):
            qp.launch_surface_view('missing.vtp')
        with patch(MODULE + '.show_surface') as show:
            qp.main(['x.vtp', '--no-edges'])
        self.assertFalse(show.call_args.kwargs['show_edges'])
        with self.assertRaises(ValueError):
            qp.show_surface('x.vtp', color_by='other')


if __name__ == '__main__':
    unittest.main()
