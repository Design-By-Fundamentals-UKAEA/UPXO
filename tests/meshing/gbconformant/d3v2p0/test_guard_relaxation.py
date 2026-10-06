import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.guard_relaxation import relax_guarded_interfaces, fold_counts
from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections

MODULE = 'upxo.meshing.gbconformant.d3v2p0.guard_relaxation'


def surfaces():
    """A smoothed two-grain block and a 'guarded' copy whose middle nodes were
    pulled fully back to their voxel coordinates, as a rollback would."""
    labels = np.ones((6, 6, 6), int)
    labels[1:5, 1:5, 1:4] = 2
    smoothed = smooth_interfaces(labels, iterations=10, relaxation=.5)
    p = smoothed.points.copy()
    rolled = np.flatnonzero((smoothed.original_points[:, 2] == 4.) & (smoothed.node_kind == 1))
    p[rolled] = smoothed.original_points[rolled]
    return replace(smoothed, points=p), smoothed, rolled


class GuardRelaxationTests(unittest.TestCase):
    def test_relaxation_smooths_only_the_rollback_region(self):
        guarded, smoothed, rolled = surfaces()
        out, report = relax_guarded_interfaces(guarded, smoothed, [6., 6., 6.], margin_rings=1)
        moved = np.flatnonzero(np.linalg.norm(out.points - guarded.points, axis=1) > 0)
        self.assertGreater(len(moved), 0)
        region = set(rolled.tolist())
        edges = np.sort(guarded.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        for a, b in edges:
            if a in rolled or b in rolled:
                region.update((int(a), int(b)))
        self.assertTrue(set(moved.tolist()) <= region)
        k3 = guarded.node_kind == 3
        np.testing.assert_array_equal(out.points[k3], guarded.points[k3])
        np.testing.assert_array_equal(out.points[guarded.fixed_axes], guarded.points[guarded.fixed_axes])
        self.assertEqual(len(find_surface_intersections(out.points, out.triangles)), 0)
        self.assertTrue(report['verified'])
        self.assertEqual(report['rolled_back_nodes'], len(rolled))

    def test_rejected_moves_are_reverted(self):
        guarded, smoothed, _ = surfaces()
        hit_all = lambda p, f, triangle_ids=None: (np.column_stack((triangle_ids, triangle_ids))
                                                   if triangle_ids is not None else np.empty((0, 2), int))
        with patch(MODULE + '.find_surface_intersections', side_effect=hit_all):
            out, report = relax_guarded_interfaces(guarded, smoothed, [6., 6., 6.])
        np.testing.assert_array_equal(out.points, guarded.points)
        self.assertEqual(report['history'][0]['accepted'], 0)

    def test_no_rollback_means_no_change(self):
        _, smoothed, _ = surfaces()
        out, report = relax_guarded_interfaces(smoothed, smoothed, [6., 6., 6.])
        np.testing.assert_array_equal(out.points, smoothed.points)
        self.assertEqual(report['moved_nodes'], 0)

    def test_disabled(self):
        guarded, smoothed, _ = surfaces()
        out, report = relax_guarded_interfaces(guarded, smoothed, [6., 6., 6.], enabled=False)
        np.testing.assert_array_equal(out.points, guarded.points)
        self.assertFalse(report['enabled'])

    def test_bad_arguments(self):
        guarded, smoothed, _ = surfaces()
        other = replace(smoothed, triangles=smoothed.triangles[:, [0, 2, 1]])
        with self.assertRaises(ValueError):
            relax_guarded_interfaces(guarded, other, [6., 6., 6.])
        for kwargs in (dict(iterations=-1), dict(relaxation=0.), dict(junction_relaxation=1.5),
                       dict(margin_rings=-1), dict(minimum_facet_angle=180.), dict(boundary_clearance=0.),
                       dict(junction_max_displacement=-1.), dict(enabled=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                relax_guarded_interfaces(guarded, smoothed, [6., 6., 6.], **kwargs)
        with self.assertRaises(ValueError):
            relax_guarded_interfaces(guarded, smoothed, [6., 6.])

    def test_fold_counts(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        flat = np.array([[0, 1, 2], [1, 3, 2]])
        self.assertEqual(fold_counts(points, flat, np.ones((2, 2), int)), {'below_150': 0, 'below_120': 0})
        folded = np.array([[0, 1, 2], [0, 4, 1]])                # 90 degree fold along the x axis
        self.assertEqual(fold_counts(points, folded, np.ones((2, 2), int)), {'below_150': 1, 'below_120': 1})
        self.assertEqual(fold_counts(points, folded, np.ones((2, 2), int), exclude_grain_ids=[1]),
                         {'below_150': 0, 'below_120': 0})
        self.assertEqual(fold_counts(points, folded, np.array([[1, 2], [1, 3]])), {'below_150': 0, 'below_120': 0})


if __name__ == '__main__':
    unittest.main()
