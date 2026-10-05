import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.tet_angles import dihedral_angles, dihedral_summary


class TetAngleTests(unittest.TestCase):
    def test_regular_tetrahedron(self):
        p = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        np.testing.assert_allclose(dihedral_angles(p, [[0, 1, 2, 3]]), np.degrees(np.arccos(1 / 3)))

    def test_corner_tetrahedron_and_orientation(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        expected = np.degrees([np.pi / 2, np.pi / 2, np.pi / 2, np.arccos(1 / np.sqrt(3))] * 1)
        for tet in ([0, 1, 2, 3], [1, 0, 2, 3]):
            a = np.sort(dihedral_angles(p, [tet])[0])
            np.testing.assert_allclose(a, np.sort([90, 90, 90] + [np.degrees(np.arccos(1 / np.sqrt(3)))] * 3))

    def test_sliver_is_small(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .01]], float)
        self.assertLess(dihedral_angles(p, [[0, 1, 2, 3]]).min(), 2)

    def test_summary_and_batches(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, .01]], float)
        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]] * 3)
        np.testing.assert_allclose(dihedral_angles(p, tets, batch_size=4), dihedral_angles(p, tets))
        s = dihedral_summary(p, tets)
        self.assertEqual(s['tets_below_5_deg'], 3)
        self.assertEqual(s['tets_below_30_deg'], 3)
        self.assertAlmostEqual(s['maximum'], dihedral_angles(p, tets).max())


if __name__ == '__main__':
    unittest.main()
