import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.small_grains import remove_small_grains
from upxo.meshing.gbconformant.d3v2p0.voxel_topology import detect_voxel_topology


class SmallGrainTests(unittest.TestCase):
    def test_island_and_disabled(self):
        a = np.ones((5, 5, 5), int); a[2, 2, 2] = 900
        b, r = remove_small_grains(a, enabled=False)
        np.testing.assert_array_equal(a, b)
        b, r = remove_small_grains(a)
        self.assertEqual(r['removed_ids'], [900])
        self.assertEqual(r['original_to_survivor'][900], 1)
        self.assertEqual(a[2, 2, 2], 900)
        self.assertFalse(detect_voxel_topology(b))

    def test_nested_island_protected(self):
        a = np.ones((7, 7, 7), int); a[2:5, 2:5, 2:5] = 2; a[3, 3, 3] = 3
        b, r = remove_small_grains(a, min_voxels=27, protected_ids=[3])
        self.assertIn(3, np.unique(b))
        self.assertTrue(r['total_voxels_preserved'])
        self.assertFalse(detect_voxel_topology(b))

    def test_mutual_choices_and_threshold(self):
        a = np.array([0, 9]).reshape(2, 1, 1)
        b, r = remove_small_grains(a, min_voxels=2)
        self.assertEqual(len(np.unique(b)), 1)
        self.assertEqual(len(r['removed_ids']), 1)
        b, r = remove_small_grains(a, min_voxels=1)
        np.testing.assert_array_equal(a, b)

    def test_boundary_protection(self):
        a = np.ones((4, 4, 4), int); a[0, 0, 0] = 2
        b, r = remove_small_grains(a, protect_rve=True)
        np.testing.assert_array_equal(a, b)
        b, r = remove_small_grains(a)
        self.assertEqual(r['removed_ids'], [2])

    def test_disconnected_donor_not_attached_to_recipient(self):
        a = np.ones((7, 7, 7), int); a[4:] = 2
        a[1, 1, 1] = 3; a[5, 5, 5] = 3
        b, r = remove_small_grains(a, min_voxels=3)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(r['rejected_proposals'])

    def test_random_merges_never_add_defects(self):
        for seed in range(5):
            a = np.random.default_rng(seed).integers(1, 7, (3, 3, 3))
            before = detect_voxel_topology(a)
            b, r = remove_small_grains(a, min_voxels=5)
            after = detect_voxel_topology(b)
            self.assertTrue(all(gs <= before.get(p, set()) for p, gs in after.items()))
            self.assertEqual(sum(r['voxel_changes'].values()), 0)

    def test_bad_controls(self):
        for kw in ({'min_voxels': True}, {'spacing': (0, 1, 1)}, {'protected_ids': [99]}):
            with self.assertRaises(ValueError):
                remove_small_grains(np.ones((2, 2, 2), int), **kw)

    def test_bridge_cannot_join_recipient_islands(self):
        a = np.ones((5, 3, 3), int)
        a[1, 1, 1] = 2; a[3, 1, 1] = 2; a[2, 1, 1] = 3
        b, r = remove_small_grains(a, min_voxels=2, protected_ids=[2], selection='shared_area')
        from scipy.ndimage import label
        self.assertEqual(label(b == 2)[1], 2)

    def test_point_neighbour_is_never_a_recipient(self):
        a = np.ones((3, 3, 3), int)
        a[0, 0, 0] = 3; a[1:, 1:, 1:] = 2
        b, r = remove_small_grains(a, min_voxels=2)
        self.assertEqual(r['original_to_survivor'][3], 1)
