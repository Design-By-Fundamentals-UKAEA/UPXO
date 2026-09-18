import unittest
from types import SimpleNamespace
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.tet_validation import validate_tet_surfaces


def box():
    p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    f=np.array([[0,4,7],[0,7,3],[1,2,6],[1,6,5],[0,1,5],[0,5,4],
                [3,7,6],[3,6,2],[0,3,2],[0,2,1],[4,5,6],[4,6,7]])
    return SimpleNamespace(points=p,triangles=f,grain_pairs=np.ones((12,2),int),
                           exterior=np.ones(12,bool),rve_face=np.repeat(np.arange(6),2),
                           report={'rve_dimensions':[1,1,1]})


class ValidationTests(unittest.TestCase):
    def test_valid_box(self):
        r=validate_tet_surfaces(box(),[1])
        self.assertEqual(r['blockers'],[])
        self.assertEqual(r['total_signed_volume'],1.)
        self.assertFalse(r['tetrahedralisation_certified'])

    def test_open_shell(self):
        s=box()
        for key in ('triangles','grain_pairs','exterior','rve_face'):
            setattr(s,key,getattr(s,key)[:-1])
        r=validate_tet_surfaces(s)
        self.assertEqual(r['status'],'BLOCKED')
        self.assertGreater(r['per_grain']['1']['open_edges'],0)

    def test_flipped_triangle(self):
        s=box();s.triangles[0]=s.triangles[0,::-1]
        r=validate_tet_surfaces(s)
        self.assertGreater(r['per_grain']['1']['inconsistent_oriented_edges'],0)

    def test_duplicate_and_missing_grain(self):
        s=box();s.triangles[-1]=s.triangles[-2]
        r=validate_tet_surfaces(s,[1,2])
        self.assertTrue(any('duplicate' in v for v in r['blockers']))
        self.assertTrue(any('Grain IDs' in v for v in r['blockers']))

    def test_point_pinch(self):
        s=box()
        # Two otherwise closed tetrahedral shells meet only at node zero.
        s.points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],float)
        t=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        s.triangles=np.vstack((t,np.array([0,4,5,6])[t][:,::-1]))
        s.grain_pairs=np.ones((8,2),int);s.exterior=np.ones(8,bool);s.rve_face=np.zeros(8,int)
        r=validate_tet_surfaces(s)
        self.assertEqual(r['per_grain']['1']['nonmanifold_edges'],0)
        self.assertEqual(r['per_grain']['1']['nonmanifold_vertices'],1)
