import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
from upxo.meshing.gbconformant.d3v2p0.gmsh_closed import remesh_closed_rve_gmsh


class JointRemeshTests(unittest.TestCase):
    def test_bounded_charts_keep_cap_ownership(self):
        labels = np.ones((3, 3, 3), dtype=int)
        labels[1:] = 2
        internal = remesh_interfaces_gmsh(smooth_interfaces(labels, iterations=2), .8)
        closed = close_rve_faces(internal, labels, mesh_size=.8)
        result = remesh_closed_rve_gmsh(closed, mesh_size=.9,
                    max_chart_triangles=8, check_intersections=True)
        self.assertEqual(result.grain_pairs.shape[1], 2)
        self.assertEqual(result.report['remaining_intersections'], 0)
        self.assertEqual(result.report['grains_with_nonmanifold_edges'], [])
        self.assertAlmostEqual(sum(result.report['enclosed_grain_volumes'].values()), 27.)
        np.testing.assert_array_equal(np.unique(result.rve_face), np.arange(-1, 6))

    def test_caps_and_internal_interfaces_regenerated_together(self):
        labels = np.ones((5, 5, 5), dtype=int)
        labels[2:] = 2
        labels[:2, 2:] = 3
        internal = remesh_interfaces_gmsh(smooth_interfaces(labels, iterations=3), .6)
        closed = close_rve_faces(internal, labels, mesh_size=.6)
        result = remesh_closed_rve_gmsh(closed, mesh_size=1.1)
        self.assertNotEqual(int(closed.exterior.sum()), int(result.exterior.sum()))
        self.assertNotEqual(int((~closed.exterior).sum()), int((~result.exterior).sum()))
        self.assertEqual(result.report['grains_with_nonmanifold_edges'], [])
        self.assertTrue(result.report['grain_surface_edge_closure_verified'])
        self.assertEqual(result.report['maximum_curve_coordinate_error'], 0.)
        self.assertGreater(result.report['rve_trace_nodes_preserved'], 0)
        self.assertAlmostEqual(sum(result.report['enclosed_grain_volumes'].values()), 125.)
        for face in range(6):
            axis, side = divmod(face, 2)
            np.testing.assert_allclose(result.points[result.triangles[result.rve_face == face], axis],
                                       side*5, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
