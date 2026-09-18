"""Integration checks; run in a Python environment with Gmsh installed."""
import unittest
import numpy as np
try:
    import gmsh
except ImportError:
    gmsh = None
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import remesh_interfaces_gmsh


@unittest.skipIf(gmsh is None, 'Gmsh is not installed')
class GmshInterfaceTests(unittest.TestCase):
    def test_bounded_input_charts_preserve_interfaces(self):
        labels = np.ones((4, 4, 4), dtype=int)
        labels[2:] = 2
        labels[:2, 2:] = 3
        source = smooth_interfaces(labels, iterations=3)
        result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8,
                                       check_intersections=True)
        np.testing.assert_array_equal(np.unique(np.sort(source.grain_pairs, axis=1), axis=0),
                                      np.unique(result.grain_pairs, axis=0))
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['remaining_intersections'], 0)
        self.assertEqual(result.report['max_input_chart_triangles'], 8)
        self.assertGreater(result.report['explicit_input_charts'], 3)

    def test_algorithm_selection(self):
        labels = np.ones((4, 4, 4), dtype=int)
        labels[2:] = 2
        labels[:2, 2:] = 3
        source = smooth_interfaces(labels, iterations=3)
        for algorithm in (1, 2, 5, 6):
            with self.subTest(algorithm=algorithm):
                result = remesh_interfaces_gmsh(source, .6, algorithm=algorithm)
                self.assertEqual(result.report['requested_algorithm'], algorithm)
                self.assertTrue(result.report['shared_boundaries_verified'])
                self.assertEqual(result.report['maximum_curve_coordinate_error'], 0.)
        with self.assertRaises(ValueError):
            remesh_interfaces_gmsh(source, algorithm=99)

    def test_remesh_with_shared_junction_and_rve_traces(self):
        labels = np.ones((5, 5, 5), dtype=int)
        labels[2:] = 2
        labels[:2, 2:] = 3
        source = smooth_interfaces(labels, iterations=3)
        result = remesh_interfaces_gmsh(source, mesh_size=.6)
        self.assertNotEqual(len(source.triangles), len(result.triangles))
        np.testing.assert_array_equal(np.unique(np.sort(source.grain_pairs, axis=1), axis=0),
                                      np.unique(result.grain_pairs, axis=0))
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['maximum_curve_coordinate_error'], 0.)
        self.assertGreater(result.report['rve_trace_nodes_preserved'], 0)
        self.assertFalse(gmsh.isInitialized())

    def test_preserves_existing_gmsh_session(self):
        gmsh.initialize()
        try:
            gmsh.model.add('existing_user_model')
            gmsh.option.setNumber('Mesh.MeshSizeMax', 123.)
            labels = np.ones((3, 3, 3), dtype=int)
            labels[1:] = 2
            remesh_interfaces_gmsh(smooth_interfaces(labels), .7)
            self.assertEqual(gmsh.model.getCurrent(), 'existing_user_model')
            self.assertEqual(gmsh.option.getNumber('Mesh.MeshSizeMax'), 123.)
            self.assertEqual(gmsh.model.list(), ['', 'existing_user_model'])
        finally:
            gmsh.finalize()

    def test_frozen_point_inside_surface_is_retained(self):
        labels = np.ones((5, 5, 5), dtype=int)
        labels[2:] = 2
        source = smooth_interfaces(labels, iterations=0)
        node = np.flatnonzero(np.all(source.points == [2, 2, 2], axis=1))[0]
        source.node_kind[node] = 3
        result = remesh_interfaces_gmsh(source, .6)
        retained = np.flatnonzero(np.all(result.points == [2, 2, 2], axis=1))
        self.assertEqual(len(retained), 1)
        self.assertIn(retained[0], result.triangles)


if __name__ == '__main__':
    unittest.main()
