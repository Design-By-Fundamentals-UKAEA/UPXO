"""Checks for label-preserving resolution changes before surface smoothing."""
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import upscale_labels, smooth_interfaces


class UpscaleTests(unittest.TestCase):
    def test_identity(self):
        labels = np.arange(8).reshape(2, 2, 2)
        refined, spacing = upscale_labels(labels, 1, (1, 2, 3))
        self.assertIs(refined, labels)
        np.testing.assert_array_equal(spacing, [1, 2, 3])

    def test_geometry_and_labels(self):
        labels = np.array([[[-3, 0]], [[7, 42]]], dtype=np.int32)
        refined, spacing = upscale_labels(labels, 3, (1, 2, 4))
        self.assertEqual(refined.dtype, labels.dtype)
        np.testing.assert_array_equal(refined[::3, ::3, ::3], labels)
        np.testing.assert_array_equal(np.unique(refined), np.unique(labels))
        np.testing.assert_allclose(np.array(refined.shape)*spacing, np.array(labels.shape)*[1, 2, 4])
        for gid in np.unique(labels):
            self.assertEqual(np.count_nonzero(refined == gid), 27*np.count_nonzero(labels == gid))

    def test_refined_interfaces_and_constraints(self):
        labels = np.ones((3, 3, 3), dtype=int)
        labels[1:, :, :] = 2
        labels[:1, 1:, :] = 3
        base = smooth_interfaces(labels, iterations=0)
        refined, spacing = upscale_labels(labels, 2)
        result = smooth_interfaces(refined, spacing=spacing, iterations=3)
        self.assertEqual(len(result.triangles), 4*len(base.triangles))
        np.testing.assert_allclose(result.original_points.min(axis=0), base.original_points.min(axis=0))
        np.testing.assert_allclose(result.original_points.max(axis=0), base.original_points.max(axis=0))
        np.testing.assert_array_equal(result.points[result.fixed_axes], result.original_points[result.fixed_axes])
        np.testing.assert_array_equal(result.points[result.node_kind == 3], result.original_points[result.node_kind == 3])

    def test_invalid_factor(self):
        for factor in [0, -1, 1.5, np.nan, np.inf, True, '2']:
            with self.assertRaises(ValueError):
                upscale_labels(np.ones((2, 2, 2), dtype=int), factor)


class JunctionTests(unittest.TestCase):
    @staticmethod
    def staircase():
        labels = np.ones((10, 8, 8), dtype=int)
        for z in range(8):
            x = 3+z//2
            labels[x:, :, z] = 2
            labels[:x, 4:, z] = 3
        return labels

    def test_true_edges_and_controlled_smoothing(self):
        result = smooth_interfaces(self.staircase(), iterations=40,
                                   junction_iterations=40, junction_relaxation=.25,
                                   junction_max_displacement=.6)
        # Voxel junctions follow cube edges, never triangulation diagonals.
        vectors = result.original_points[result.junction_edges[:, 1]]-result.original_points[result.junction_edges[:, 0]]
        self.assertTrue(np.all(np.count_nonzero(vectors, axis=1) == 1))
        metrics = result.junction_metrics()
        self.assertGreater(metrics['movable_nodes'], 0)
        self.assertLess(metrics['mean_turn_after_deg'], metrics['mean_turn_before_deg'])
        self.assertGreater(metrics['maximum_line_displacement'], 0)
        self.assertLessEqual(metrics['maximum_line_displacement'], .6+1e-12)
        np.testing.assert_array_equal(result.points[result.node_kind == 3], result.original_points[result.node_kind == 3])
        np.testing.assert_array_equal(result.points[result.fixed_axes], result.original_points[result.fixed_axes])

    def test_junction_smoothing_can_be_disabled(self):
        result = smooth_interfaces(self.staircase(), iterations=10, junction_iterations=0)
        line = result.node_kind == 2
        self.assertTrue(np.any(line))
        np.testing.assert_array_equal(result.points[line], result.original_points[line])

    def test_invalid_junction_controls(self):
        for options in [dict(junction_iterations=-1), dict(junction_relaxation=1.1),
                        dict(junction_max_displacement=-.1)]:
            with self.assertRaises(ValueError):
                smooth_interfaces(self.staircase(), **options)


if __name__ == '__main__':
    unittest.main()
