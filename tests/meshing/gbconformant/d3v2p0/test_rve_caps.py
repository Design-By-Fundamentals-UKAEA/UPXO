import unittest
from types import SimpleNamespace
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces


class CapTests(unittest.TestCase):
    def test_three_grains_and_shared_edges(self):
        labels = np.ones((5, 5, 5), dtype=int)
        labels[2:] = 2
        labels[:2, 2:] = 3
        source = remesh_interfaces_gmsh(smooth_interfaces(labels, iterations=3), .6)
        closed = close_rve_faces(source, labels, mesh_size=.7)
        self.assertTrue(closed.report['all_six_faces_closed'])
        self.assertTrue(closed.report['grain_surface_edge_closure_verified'])
        self.assertEqual(closed.report['grains_with_nonmanifold_edges'], [])
        self.assertAlmostEqual(sum(closed.report['enclosed_grain_volumes'].values()), 125.)
        np.testing.assert_array_equal(source.points, closed.points[:len(source.points)])
        np.testing.assert_array_equal(source.triangles, closed.triangles[:len(source.triangles)])
        self.assertEqual(set(closed.rve_face[closed.exterior]), set(range(6)))

    def test_single_grain_rectangular_rve(self):
        empty = SimpleNamespace(points=np.empty((0, 3)), triangles=np.empty((0, 3), dtype=int),
                                grain_pairs=np.empty((0, 2), dtype=int))
        labels = np.full((2, 3, 4), -7, dtype=int)
        closed = close_rve_faces(empty, labels, spacing=(1., 2., 3.), mesh_size=2.)
        self.assertEqual(closed.report['cap_regions'], 6)
        self.assertAlmostEqual(closed.report['enclosed_grain_volumes']['-7'], 144.)
        self.assertTrue(np.all(closed.exterior))
        self.assertGreater(closed.report['added_rve_edge_nodes'], 0)
        edges = np.unique(np.sort(closed.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
        xyz = closed.points[edges]
        extent = np.array([2., 6., 12.])
        on_box_edge = ((np.all(np.isclose(xyz, 0.), axis=1) |
                        np.all(np.isclose(xyz, extent), axis=1)).sum(axis=1) >= 2)
        lengths = np.linalg.norm(xyz[on_box_edge, 1]-xyz[on_box_edge, 0], axis=1)
        self.assertLessEqual(lengths.max(), 2.+1e-8)
        self.assertGreater(len(lengths), 12)


if __name__ == '__main__':
    unittest.main()
