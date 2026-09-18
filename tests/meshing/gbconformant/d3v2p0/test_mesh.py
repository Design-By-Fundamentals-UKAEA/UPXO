"""Run with python -m unittest upxo.meshing.gbconformant.d3v2p0.test_mesh."""
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.mesh import mesh_voxels, quality, faces_of
from upxo.meshing.gbconformant.d3v2p0.optimization import optimize_quality, _energy_gradient


class MeshTests(unittest.TestCase):
    def test_single_voxel(self):
        m = mesh_voxels(np.array([[[-7]]]))
        self.assertEqual(len(m.tetrahedra), 6)
        self.assertAlmostEqual(m.validate()['volume'], 1)

    def test_junctions_and_smoothing(self):
        labels = np.ones((6, 6, 6), dtype=int)
        labels[3:, :, :] = 2
        labels[:3, 3:, :] = 3
        labels[3:, 3:, 3:] = 4
        labels[2, :3, 1:4] = 2
        m = mesh_voxels(labels, iterations=8)
        self.assertEqual(set(m.grain_ids), {1, 2, 3, 4})
        self.assertTrue(np.any(m.node_kind == 3))
        self.assertTrue(np.any(m.node_kind == 2))
        self.assertTrue(np.any(m.points[m.node_kind == 1] != m.original_points[m.node_kind == 1]))
        self.assertTrue(np.any(m.points[m.node_kind == 2] != m.original_points[m.node_kind == 2]))
        self.assertGreaterEqual(quality(m.points, m.tetrahedra)[1].min(), 0.3)
        m.validate()
        f, count, a, b = faces_of(m.tetrahedra)
        self.assertTrue(np.all(np.isin(count, [1, 2])))
        self.assertEqual(np.sum((count == 2) & (m.grain_ids[a] != m.grain_ids[b])), np.sum(~m.exterior))

    def test_zero_disconnected_and_anisotropic(self):
        labels = np.zeros((3, 3, 3), dtype=int)
        labels[0, 0, 0] = labels[2, 2, 2] = 9
        m = mesh_voxels(labels, spacing=(1, 2, 1), iterations=2)
        self.assertEqual(set(m.grain_ids), {0, 9})
        self.assertAlmostEqual(m.validate()['volume'], 54)

    def test_invalid_input(self):
        for labels in [np.ones((2, 2)), np.ones((2, 2, 2)), np.zeros((0, 2, 2), dtype=int)]:
            with self.assertRaises(ValueError):
                mesh_voxels(labels)

    def test_quality_gradient(self):
        points = np.array([[0., 0., 0.], [1.2, 0., 0.], [.2, .8, 0.], [.1, .2, .7]])
        tets = np.array([[0, 1, 2, 3]])
        weights = np.array([3.])
        _, gradient = _energy_gradient(points, tets, weights)
        numerical = np.zeros_like(points)
        for i in range(4):
            for j in range(3):
                plus, minus = points.copy(), points.copy()
                plus[i, j] += 1e-6
                minus[i, j] -= 1e-6
                numerical[i, j] = (_energy_gradient(plus, tets, weights)[0]
                                    - _energy_gradient(minus, tets, weights)[0])/2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-7)

    def test_optimization_improves_boundary_quality(self):
        labels = np.ones((6, 6, 6), dtype=int)
        labels[3:, :, :] = 2
        labels[2, :3, 1:4] = 2
        labels[:3, 3:, :] = 3
        labels[3:, 3:, 3:] = 4
        m = mesh_voxels(labels, min_quality=.3, quality_iterations=0)
        before = m.boundary_quality()['boundary_node_tetrahedra']
        anchor = m.points.copy()
        connectivity = m.tetrahedra.copy()
        history = optimize_quality(m, iterations=20, max_displacement=.25)
        after = m.boundary_quality()['boundary_node_tetrahedra']
        self.assertGreater(len(history), 0)
        self.assertGreater(after['mean'], before['mean'])
        self.assertGreaterEqual(after['minimum'], before['minimum'])
        self.assertTrue(all(a['energy'] > b['energy'] for a, b in zip(history, history[1:])))
        self.assertLessEqual(np.linalg.norm(m.points-anchor, axis=1).max(), .25+1e-12)
        np.testing.assert_array_equal(m.tetrahedra, connectivity)
        m.validate()


if __name__ == '__main__':
    unittest.main()
