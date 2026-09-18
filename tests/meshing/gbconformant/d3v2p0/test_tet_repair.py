import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE,close_rve_faces
from upxo.meshing.gbconformant.d3v2p0.tet_repair import repair_tet_contacts
from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces


class TetRepairTests(unittest.TestCase):
    def test_point_contact_separated_without_mutating_input(self):
        p=np.array([[2,2,2],[3,2,2],[2,3,2],[2,2,3],[1,2,2],[2,1,2],[2,2,1]],float)
        t=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        f=np.vstack((t,np.array([0,4,5,6])[t][:,::-1]))
        s=ClosedRVE(p,f,np.repeat([[1,2],[1,3]],4,axis=0),np.zeros(8,bool),np.full(8,-1),{'rve_dimensions':[4,4,4]})
        r=repair_tet_contacts(s,.1)
        self.assertEqual(len(r.points),8)
        self.assertEqual(len(set(r.triangles[:4].ravel())&set(r.triangles[4:].ravel())),0)
        self.assertGreater(np.linalg.norm(r.points[0]-r.points[-1]),0)
        np.testing.assert_array_equal(s.points,p)
        np.testing.assert_array_equal(r.points[1:7],p[1:])

    def test_nested_and_disconnected_grain_pieces(self):
        labels=np.ones((5,5,5),int)
        labels[1,1,1]=2;labels[3,3,3]=2
        s=smooth_interfaces(labels,iterations=0)
        xyz=s.original_points[s.triangles]
        normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        flip=(normal.sum(axis=1)<0)!=(s.grain_pairs[:,0]>s.grain_pairs[:,1])
        s.triangles[flip]=s.triangles[flip][:,[0,2,1]]
        s.grain_pairs=np.sort(s.grain_pairs,axis=1)
        closed=close_rve_faces(s,labels,mesh_size=1.)
        result=mesh_repaired_rve_gmsh(closed,mesh_size=1.2,optimize_netgen=True,minimum_quality=.9)
        self.assertEqual(result.report['grains'],2)
        self.assertEqual(result.report['volume_entities'],3)
        self.assertTrue(result.report['source_surface_conformity_verified'])
        self.assertAlmostEqual(result.report['total_volume'],125.)
        self.assertEqual(set(result.grain_ids),{1,2})

    def test_edge_contact_separated(self):
        p=np.array([[2,2,1],[2,2,3],[3,2,2],[2,3,2],[1,2,2],[2,1,2]],float)
        t=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        f=np.vstack((t,np.array([0,1,4,5])[t]))
        s=ClosedRVE(p,f,np.repeat([[1,2],[1,3]],4,axis=0),np.zeros(8,bool),np.full(8,-1),{'rve_dimensions':[4,4,4]})
        r=repair_tet_contacts(s,.1)
        self.assertEqual(r.report['contact_repair']['separable_contact_edges'],1)
        self.assertEqual(r.report['contact_repair']['added_nodes'],2)
        self.assertFalse(set(r.triangles[:4].ravel())&set(r.triangles[4:].ravel()))
