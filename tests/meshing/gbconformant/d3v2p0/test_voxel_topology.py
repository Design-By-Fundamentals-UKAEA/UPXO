import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.voxel_topology import detect_voxel_topology,clean_voxel_topology,_VALID


class VoxelTopologyTests(unittest.TestCase):
    def test_link_test_is_complement_symmetric(self):
        np.testing.assert_array_equal(_VALID,_VALID[::-1])
        self.assertTrue(_VALID[1])
        self.assertFalse(_VALID[129])  # opposite octants meet only at a point
        self.assertFalse(_VALID[126])  # connected grain, disconnected complement

    def test_box_and_planar_partition_have_no_defects(self):
        a=np.ones((3,3,3),int)
        self.assertFalse(detect_voxel_topology(a))
        a[1:]=2
        self.assertFalse(detect_voxel_topology(a))

    def test_point_contact_is_cleaned_without_losing_ids(self):
        a=np.ones((3,3,3),int);a[0,0,0]=2;a[1,1,1]=2
        self.assertTrue(detect_voxel_topology(a))
        b,r,edits=clean_voxel_topology(a,max_volume_fraction=1)
        self.assertEqual(r['remaining_defects'],0)
        np.testing.assert_array_equal(np.unique(a),np.unique(b))
        self.assertEqual(a[0,0,0],2)
        self.assertTrue(edits)

    def test_disabled_and_zero_budget_are_honest(self):
        a=np.ones((3,3,3),int);a[0,0,0]=2;a[1,1,1]=2
        for options in [dict(enabled=False),dict(max_changes=0)]:
            b,r,_=clean_voxel_topology(a,**options)
            np.testing.assert_array_equal(a,b)
            self.assertGreater(r['remaining_defects'],0)

    def test_pair_repair_preserves_zero_net_volume_budget(self):
        a=np.ones((3,3,3),int);a[0,0,0]=2;a[1,1,1]=2
        b,r,edits=clean_voxel_topology(a,max_volume_fraction=0)
        self.assertEqual(r['remaining_defects'],0)
        np.testing.assert_array_equal(np.unique(a,return_counts=True),np.unique(b,return_counts=True))
        self.assertTrue(any('pair_group' in e for e in edits))
