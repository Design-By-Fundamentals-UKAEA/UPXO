import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.local_thickness import local_thickness, node_local_thickness
from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles


def dumbbell():
    """Grain 2: two 5^3 lobes joined by a 1x1x9 bar, inside grain 1."""
    labels = np.ones((23, 9, 9), int)
    labels[2:7, 2:7, 2:7] = 2
    labels[16:21, 2:7, 2:7] = 2
    labels[7:16, 4, 4] = 2
    return labels


def voronoi(seed, n=10, grains=6):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, n, size=(grains, 3))
    grid = np.stack(np.meshgrid(*[np.arange(n) + .5] * 3, indexing='ij'), -1)
    return np.argmin(np.linalg.norm(grid[..., None, :] - centres, axis=-1), axis=-1) + 1


class LocalThicknessTests(unittest.TestCase):
    def test_slabs(self):
        for width, expected in ((1, 1), (2, 1), (3, 3), (4, 3), (5, 5)):
            labels = np.ones((12, 12, 12), int)
            labels[3:3 + width] = 2
            with self.subTest(width=width):
                self.assertTrue(np.all(local_thickness(labels)[labels == 2] == expected))

    def test_neck_is_thin_and_lobes_are_thick(self):
        labels = dumbbell()
        lt = local_thickness(labels)
        self.assertTrue(np.all(lt[8:15, 4, 4] == 1))           # bar ends are reached by lobe balls
        self.assertEqual(lt[4, 4, 4], 5)
        self.assertEqual(lt[18, 4, 4], 5)

    def test_node_values_use_the_given_grains(self):
        labels = dumbbell()
        corner = np.array([[11, 4, 5]])             # on the bar surface, mid-length
        self.assertEqual(node_local_thickness(labels, corner, grain_ids=[{2}])[0], 1)
        self.assertGreater(node_local_thickness(labels, corner, grain_ids=[{1}])[0], 1)

    def test_rve_faces_do_not_thin_grains(self):
        labels = np.ones((12, 12, 12), int)
        labels[:, :, :3] = 2                                   # 3-voxel slab on the z = 0 face
        self.assertTrue(np.all(local_thickness(labels)[labels == 2] >= 3))

    def test_thick_grain_corners_read_thick(self):
        labels = np.ones((14, 14, 14), int)
        labels[3:11, 3:11, 3:11] = 2                           # 8-voxel cube: its corners must not read thin
        corner = np.array([[3, 3, 3]])
        self.assertGreaterEqual(node_local_thickness(labels, corner, grain_ids=[{2}])[0], 5)

    def test_rejects_bad_labels(self):
        with self.assertRaises(ValueError):
            local_thickness(np.ones((3, 3), int))


class DifferentialSmoothingTests(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        labels = voronoi(0)
        a = smooth_interfaces(labels, iterations=10, relaxation=.5)
        b = smooth_interfaces(labels, iterations=10, relaxation=.5, iterations_by_pair=None,
                              thickness_cap_fraction=None, minimum_wedge_angle=None)
        np.testing.assert_array_equal(a.points, b.points)
        self.assertEqual(a.smoothing_report, {})

    def test_zero_iterations_for_a_pair_keeps_its_own_nodes(self):
        labels = voronoi(1)
        s0 = smooth_interfaces(labels, iterations=10, relaxation=.5)
        pair = tuple(int(x) for x in s0.grain_pairs[0])
        s = smooth_interfaces(labels, iterations=10, relaxation=.5, iterations_by_pair={pair: 0}, ramp_rings=0)
        on_pair = np.zeros(len(s.points), bool)
        on_pair[np.unique(s.triangles[np.all(np.sort(s.grain_pairs, axis=1) == sorted(pair), axis=1)])] = True
        np.testing.assert_array_equal(s.points[on_pair], s.original_points[on_pair])
        self.assertGreater(np.abs(s.points[~on_pair] - s.original_points[~on_pair]).max(), 0)
        self.assertGreater(s.smoothing_report['nodes_with_reduced_iterations'], 0)

    def test_ramp_limits_the_jump_between_neighbours(self):
        labels = voronoi(2)
        s0 = smooth_interfaces(labels, iterations=12, relaxation=.5)
        pair = tuple(int(x) for x in s0.grain_pairs[0])
        flat = smooth_interfaces(labels, iterations=12, relaxation=.5, iterations_by_pair={pair: 0}, ramp_rings=0)
        ramped = smooth_interfaces(labels, iterations=12, relaxation=.5, iterations_by_pair={pair: 0}, ramp_rings=3)
        self.assertGreater(ramped.smoothing_report['nodes_with_reduced_iterations'],
                           flat.smoothing_report['nodes_with_reduced_iterations'])

    def test_thickness_cap_protects_the_neck(self):
        labels = dumbbell()
        free = smooth_interfaces(labels, iterations=40, relaxation=.75)
        capped = smooth_interfaces(labels, iterations=40, relaxation=.75, thickness_cap_fraction=.5)
        thick = node_local_thickness(labels, np.round(capped.original_points).astype(int),
                                     grain_ids=[{1, 2}] * len(capped.points))
        limit = .5 * .5 * thick
        move = np.linalg.norm(capped.points - capped.original_points, axis=1)
        self.assertTrue(np.all(move <= limit + 1e-9))
        o = capped.original_points
        bar = (o[:, 0] >= 10) & (o[:, 0] <= 13) & np.isin(o[:, 1], [4, 5]) & np.isin(o[:, 2], [4, 5])
        free_move = np.linalg.norm(free.points - free.original_points, axis=1)
        self.assertLess(move[bar].max(), free_move[bar].max())
        self.assertGreater(capped.smoothing_report['nodes_at_thickness_cap'], 0)

    def test_wedge_guard_keeps_openings(self):
        for seed in range(4):
            labels = voronoi(seed, grains=8)
            with self.subTest(seed=seed):
                s = smooth_interfaces(labels, iterations=40, relaxation=.75, minimum_wedge_angle=30.)
                folds, _ = small_facet_angles(s.points, s.triangles, 30.)
                self.assertEqual(len(folds), 0)
                self.assertIn('wedge_stopped_nodes', s.smoothing_report)

    def test_bad_arguments(self):
        labels = voronoi(0)
        for kwargs in (dict(ramp_rings=-1), dict(thickness_cap_fraction=0.), dict(minimum_wedge_angle=0.),
                       dict(minimum_wedge_angle=90.), dict(iterations_by_pair={(1, 2): -1}),
                       dict(iterations_by_pair={(1, 2): 1.5})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                smooth_interfaces(labels, iterations=2, **kwargs)


if __name__ == '__main__':
    unittest.main()


class CornerGuardTests(unittest.TestCase):
    def test_patch_corners_stay_open(self):
        for seed in range(4):
            labels = voronoi(seed, grains=8)
            with self.subTest(seed=seed):
                s = smooth_interfaces(labels, iterations=40, relaxation=.75, junction_iterations=40,
                                      junction_relaxation=.5, minimum_corner_angle=25.)
                self.assertGreaterEqual(s.smoothing_report['smallest_patch_corner'], 25. - 1e-6)
                plain = smooth_interfaces(labels, iterations=40, relaxation=.75, junction_iterations=40,
                                          junction_relaxation=.5, minimum_corner_angle=89.)
                self.assertIn('smallest_patch_corner', plain.smoothing_report)

    def test_corner_angle_validation(self):
        for bad in (0., 90., float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                smooth_interfaces(voronoi(0), iterations=2, minimum_corner_angle=bad)
