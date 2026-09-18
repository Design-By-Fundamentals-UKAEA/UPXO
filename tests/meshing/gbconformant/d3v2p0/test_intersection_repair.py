import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE
from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections
from upxo.meshing.gbconformant.d3v2p0.intersection_repair import repair_interface_crossings


class CrossingRepairTests(unittest.TestCase):
    def test_junction_fixed_while_crossing_flaps_separate(self):
        p=np.array([[18,13.5,20],[18,13,20],[17.74752589,13.59320828,20.12928175],
                    [17.75872795,13.40905732,20.35063656],[17.85495074,13.32559319,20.15184337],
                    [17.60260551,13.53186343,20.10729973],[17.63693785,13.22328002,20.38742964]])
        f=np.array([[0,2,1],[2,3,1],[0,4,5],[6,4,1],[5,4,6]])
        pairs=np.array([[16,18],[16,18],[16,77],[16,77],[16,77]])
        s=ClosedRVE(p,f,pairs,np.zeros(5,bool),np.full(5,-1),{'rve_dimensions':[30,35,40]})
        self.assertGreater(len(find_surface_intersections(p,f)),0)
        with self.assertRaisesRegex(RuntimeError,'above limit'):
            repair_interface_crossings(s,max_displacement=.001)
        r=repair_interface_crossings(s,max_displacement=.35)
        self.assertEqual(len(find_surface_intersections(r.points,r.triangles)),0)
        np.testing.assert_array_equal(r.points[:2],s.points[:2])
        np.testing.assert_array_equal(s.points,p)
        self.assertLessEqual(r.report['intersection_repair']['maximum_displacement'],.35)

    def test_clean_input_stays_unchanged(self):
        p=np.array([[1,1,1],[2,1,1],[1,2,1]],float)
        s=ClosedRVE(p,np.array([[0,1,2]]),np.array([[1,2]]),np.array([False]),np.array([-1]),{'rve_dimensions':[3,3,3]})
        r=repair_interface_crossings(s,max_displacement=1e-9)
        np.testing.assert_array_equal(r.points,p)
        self.assertEqual(r.report['intersection_repair']['initial_intersections'],0)
