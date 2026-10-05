import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.surface_metrics import voxel_like_fraction


class VoxelLikeTests(unittest.TestCase):
    def test_flat_tilted_and_excluded(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.], [1, 1, 1.]])
        tris = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 4]])        # axis-aligned, tilted, tilted
        r = voxel_like_fraction(p, tris)
        a = [.5, np.sqrt(3) / 2, np.linalg.norm(np.cross([1, 0, 0], [1, 1, 1])) / 2]
        self.assertAlmostEqual(r['voxel_like'], a[0] / sum(a))
        self.assertAlmostEqual(r['smooth'], 1 - a[0] / sum(a))
        r = voxel_like_fraction(p, tris, exterior=[True, False, False])
        self.assertEqual(r['voxel_like'], 0.)
        r = voxel_like_fraction(p, tris, grain_pairs=np.array([[1, 2], [3, 4], [3, 4]]), grain_ids=[2])
        self.assertAlmostEqual(r['on_given_grains'], r['voxel_like'])
        with self.assertRaises(ValueError):
            voxel_like_fraction(p, tris, tolerance_deg=0)


if __name__ == '__main__':
    unittest.main()
