import unittest
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import _align_chart_orientation, remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
from upxo.meshing.gbconformant.d3v2p0.gmsh_closed import remesh_closed_rve_gmsh


class ChartOrientationTests(unittest.TestCase):
    def test_reversed_open_chart(self):
        p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
        f=np.array([[0,1,2],[0,2,3]])
        fixed,flipped=_align_chart_orientation(f,p,f[:,::-1],p)
        self.assertTrue(flipped)
        normal=np.cross(p[fixed[:,1]]-p[fixed[:,0]],p[fixed[:,2]]-p[fixed[:,0]])
        self.assertTrue(np.all(normal[:,2]>0))

    def test_reversed_closed_chart(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        _,flipped=_align_chart_orientation(f,p,f[:,::-1],p)
        self.assertTrue(flipped)

    def test_mixed_winding_rejected(self):
        p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
        f=np.array([[0,1,2],[0,2,3]])
        broken=f.copy();broken[0]=broken[0,::-1]
        with self.assertRaisesRegex(RuntimeError,'winding'):
            _align_chart_orientation(f,p,broken,p)

    def test_gmsh_reversed_charts_keep_positive_grain_volumes(self):
        import gmsh
        labels=np.ones((4,4,4),int);labels[2:]=8
        internal=remesh_interfaces_gmsh(smooth_interfaces(labels,iterations=2),.8)
        closed=close_rve_faces(internal,labels,mesh_size=.8)
        original=gmsh.model.mesh.generate
        def reversed_generate(dim):
            original(dim)
            gmsh.model.mesh.reverse(gmsh.model.getEntities(2))
        with patch.object(gmsh.model.mesh,'generate',side_effect=reversed_generate):
            result=remesh_closed_rve_gmsh(closed,mesh_size=1.)
        self.assertTrue(result.report['reoriented_surface_charts'])
        self.assertGreater(result.report['enclosed_grain_volumes']['8'],0)
        self.assertAlmostEqual(sum(result.report['enclosed_grain_volumes'].values()),64.)
