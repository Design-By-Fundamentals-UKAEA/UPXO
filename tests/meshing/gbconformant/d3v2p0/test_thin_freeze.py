import unittest
from copy import deepcopy
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.thin_freeze import detect_thin_grains, FrozenGeometry
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces, upscale_labels


class FreezeTests(unittest.TestCase):
    def test_detection(self):
        a = np.ones((9,9,9), int)
        a[2:7,2:7,2:7] = 2
        self.assertEqual(detect_thin_grains(a)[0], [])
        a[4,4,4] = 3
        self.assertIn(3, detect_thin_grains(a)[0])
        self.assertEqual(detect_thin_grains(a, enabled=False)[0], [])
        self.assertEqual(detect_thin_grains(a, automatic=False, explicit_ids=[2])[0], [2])
        a = np.ones((6,6,6), int); a[2,1:5,1:5] = 2
        self.assertIn(2, detect_thin_grains(a)[0])
        a = np.ones((6,6,6), int); a[2,2,1:5] = 2
        self.assertIn(2, detect_thin_grains(a)[0])

    def test_upscale_and_shared_nodes(self):
        a = np.ones((4,4,4), int); a[1,1,1] = 2; a[2:] = 3
        for factor in (1,2,3):
            b,s = upscale_labels(a, factor, (1,2,3))
            surf = smooth_interfaces(b, spacing=s, iterations=8, frozen_grain_ids=[2])
            nodes = np.unique(surf.triangles[np.any(surf.grain_pairs == 2, axis=1)])
            np.testing.assert_array_equal(surf.points[nodes], surf.original_points[nodes])
            self.assertEqual(FrozenGeometry(a,[2],(1,2,3)).check(surf)['frozen_grains'],1)

    def test_reject_and_retriangulation(self):
        a = np.ones((3,3,3), int); a[1,1,1] = 2
        surf = smooth_interfaces(a, iterations=0)
        guard = FrozenGeometry(a,[2]); guard.check(surf)
        split = deepcopy(surf)
        # Refine every triangle without changing geometry.
        centers = surf.points[surf.triangles].mean(axis=1)
        split.points = np.vstack((surf.points, centers))
        ci = np.arange(len(centers))+len(surf.points)
        split.triangles = np.array([[u,v,c] for tri,c in zip(surf.triangles,ci)
                                   for u,v in zip(tri,np.roll(tri,-1))])
        split.grain_pairs = np.repeat(surf.grain_pairs,3,axis=0)
        guard.check(split)
        def move(x):
            x.points[0] += .1
            return x
        original = surf.points.copy()
        with self.assertRaises(RuntimeError): guard.run(move,surf)
        np.testing.assert_array_equal(original,surf.points)

    def test_rve_and_gmsh(self):
        from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
        a = np.ones((3,3,3), int); a[0,1,1] = 2
        surf = smooth_interfaces(a, iterations=4, frozen_grain_ids=[2])
        guard = FrozenGeometry(a,[2])
        result = guard.run(remesh_interfaces_gmsh,surf,mesh_size=.6)
        self.assertEqual(result.grain_pairs.shape[1],2)
        guard.check(result)
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p0.gmsh_closed import remesh_closed_rve_gmsh
        closed = guard.run(close_rve_faces, result, a, mesh_size=.6)
        joint = guard.run(remesh_closed_rve_gmsh, closed, mesh_size=.6)
        guard.check(joint)
        from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import mesh_repaired_rve_gmsh
        tets = mesh_repaired_rve_gmsh(joint, mesh_size=.8, minimum_quality=0.)
        p = tets.points[tets.tetrahedra[tets.grain_ids == 2]]
        volume = np.einsum('ij,ij->i', p[:,1]-p[:,0], np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0])).sum()/6
        self.assertAlmostEqual(volume, 1., places=8)
        self.assertTrue(np.all(tets.quality > 0))
