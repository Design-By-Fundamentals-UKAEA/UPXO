import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections


class IntersectionTests(unittest.TestCase):
    def test_mixed_size_search_matches_all_pairs(self):
        from upxo.meshing.gbconformant.d3v2p0.surface_intersections import _intersecting_pairs
        rng=np.random.default_rng(16)
        p=rng.normal(size=(90,3));p[:3]*=30;p[3:6]*=.01
        f=np.arange(90).reshape(-1,3)
        pairs=np.array([(a,b) for a in range(30) for b in range(a+1,30)])
        tol=1e-9*np.ptp(p,axis=0).max()
        expected=pairs[_intersecting_pairs(p,f,pairs,tol)]
        np.testing.assert_array_equal(find_surface_intersections(p,f),expected)
        selected=np.array([0,1,4,8])
        expected=expected[np.any(np.isin(expected,selected),axis=1)]
        np.testing.assert_array_equal(find_surface_intersections(p,f,triangle_ids=selected),expected)

    def test_shared_edge_is_allowed(self):
        p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
        self.assertEqual(len(find_surface_intersections(p,np.array([[0,1,2],[0,2,3]]))),0)

    def test_crossing_facets(self):
        p=np.array([[0,0,0],[2,0,0],[0,2,0],[.5,.5,-1],[.5,.5,1],[1,.5,0]],float)
        np.testing.assert_array_equal(find_surface_intersections(p,np.array([[0,1,2],[3,4,5]])),[[0,1]])

    def test_coplanar_overlap(self):
        p=np.array([[0,0,0],[2,0,0],[0,2,0],[.2,.2,0],[1,.2,0],[.2,1,0]],float)
        self.assertEqual(len(find_surface_intersections(p,np.array([[0,1,2],[3,4,5]]))),1)

    def test_shared_node_does_not_hide_crossing(self):
        p=np.array([[0,0,0],[2,0,0],[0,2,0],[.5,.5,-1],[.5,.5,1]],float)
        self.assertEqual(len(find_surface_intersections(p,np.array([[0,1,2],[0,3,4]]))),1)

    def test_separated_triangles(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.1],[1,0,.1],[0,1,.1]],float)
        self.assertEqual(len(find_surface_intersections(p,np.array([[0,1,2],[3,4,5]]))),0)
