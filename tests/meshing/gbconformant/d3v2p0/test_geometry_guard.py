import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.geometry_guard import stabilize_interfaces


class GeometryGuardTests(unittest.TestCase):
    def test_nearly_closed_wedge_is_opened_with_endpoints_frozen(self):
        from dataclasses import make_dataclass
        Surface=make_dataclass('Surface',['points','original_points','triangles','node_kind','fixed_axes'])
        original=np.array([[1.,1,1],[2.,1,1],[1.,2,1],[1.,1,2]])
        points=original.copy();points[3]=[1.,2.,1.0001]
        source=Surface(points,original,np.array([[0,1,2],[1,0,3]]),
                       np.array([3,3,1,1]),np.zeros((4,3),bool))
        result,report=stabilize_interfaces(source,[4,4,4],minimum_facet_angle=1.)
        self.assertGreater(report['history'][0]['small_facet_angles'],0)
        self.assertEqual(report['remaining_small_facet_angles'],0)
        np.testing.assert_array_equal(result.points[:2],points[:2])

    def test_cap_tangency_is_retracted_with_trace_fixed(self):
        a=np.ones((4,4,4),int);a[2:]=2
        s=smooth_interfaces(a,iterations=0)
        movable=(s.original_points[:,1]==1)&~s.fixed_axes[:,1]
        s.points[movable,1]=0
        before=s.points.copy()
        fixed=s.fixed_axes.copy()
        r,report=stabilize_interfaces(s,[4,4,4],boundary_clearance=.1)
        self.assertTrue(report['verified'])
        self.assertGreater(report['moved_nodes'],0)
        self.assertTrue(np.all(r.points[movable,1]>=.1))
        np.testing.assert_array_equal(r.points[fixed],before[fixed])
        np.testing.assert_array_equal(s.points,before)

    def test_small_correction_budget_rejects(self):
        a=np.ones((4,4,4),int);a[2:]=2
        s=smooth_interfaces(a,iterations=0)
        s.points[s.original_points[:,1]==1,1]=0
        with self.assertRaises(RuntimeError):
            stabilize_interfaces(s,[4,4,4],boundary_clearance=.1,max_correction=.001)
