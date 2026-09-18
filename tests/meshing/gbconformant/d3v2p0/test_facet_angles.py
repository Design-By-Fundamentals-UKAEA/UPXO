import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles


class FacetAngleTests(unittest.TestCase):
    def test_near_closed_junction_detected_but_flat_neighbours_allowed(self):
        t=np.radians(.05)
        p=np.array([[0.,0,0],[1.,0,0],[0,1,0],[0,np.cos(t),np.sin(t)],[0,-1,0]])
        f=np.array([[0,1,2],[1,0,3],[0,1,4]])
        pairs,angles=small_facet_angles(p,f,.1)
        np.testing.assert_array_equal(pairs,[[0,1]])
        self.assertAlmostEqual(angles[0],.05)
        self.assertEqual(len(small_facet_angles(p,f,0)[0]),0)

    def test_rigid_rotation_does_not_change_opening(self):
        p=np.array([[0.,0,0],[1.,0,0],[0,1,0],[0,1,.0001]])
        f=np.array([[0,1,2],[1,0,3]])
        _,a=small_facet_angles(p,f,1)
        _,b=small_facet_angles(p[:,[2,0,1]]+123,f,1)
        np.testing.assert_allclose(a,b,rtol=1e-8)
