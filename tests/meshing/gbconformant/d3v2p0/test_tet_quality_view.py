import unittest
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0 import tet_quality_view as tqv

MODULE = 'upxo.meshing.gbconformant.d3v2p0.tet_quality_view'


class TetQualityViewTests(unittest.TestCase):
    def test_mask_combines_plane_and_both_filters(self):
        centres = np.array([[-1, 0, 0], [-1, 0, 0], [1, 0, 0], [-2, 0, 0.]])
        smallest = np.array([10., 40., 10., 10.])
        largest = np.array([160., 100., 160., 120.])
        everything = tqv.shown_cells(centres, smallest, largest, [0, 0, 0], [1, 0, 0])
        np.testing.assert_array_equal(everything, [True, True, False, True])     # plane keeps x <= 0
        np.testing.assert_array_equal(tqv.shown_cells(centres, smallest, largest, [0, 0, 0], [1, 0, 0], min_upper=15.),
                                      [True, False, False, True])
        np.testing.assert_array_equal(tqv.shown_cells(centres, smallest, largest, [0, 0, 0], [1, 0, 0], max_lower=150.),
                                      [True, False, False, False])

    def test_angle_arrays(self):
        import pyvista as pv
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
        grid = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([pv.CellType.TETRA], np.uint8), p)
        mn, mx = tqv.tet_angle_arrays(grid)
        self.assertAlmostEqual(mn[0], np.degrees(np.arccos(1 / np.sqrt(3))))
        self.assertAlmostEqual(mx[0], 90.)

    def test_launch_and_command_line(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 't.vtu'); open(path, 'w').close()
            with patch(MODULE + '.subprocess.Popen') as popen:
                tqv.launch_tet_quality_view(path, title='T')
            self.assertEqual(popen.call_args[0][0][1:4], ['-m', MODULE, path])
        with self.assertRaises(FileNotFoundError):
            tqv.launch_tet_quality_view('missing.vtu')
        with patch(MODULE + '.show_tet_quality') as show:
            tqv.main(['x.vtu', '--title', 'Y'])
        show.assert_called_once_with('x.vtu', title='Y')


if __name__ == '__main__':
    unittest.main()
