import unittest
from unittest.mock import patch
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE
from upxo.meshing.gbconformant.d3v2p0.surface_angles import improve_surface_angles, triangle_angles

MODULE = 'upxo.meshing.gbconformant.d3v2p0.surface_angles'


def grid(n=5):
    """Planar n x n node grid in z=0, two counter-clockwise triangles per cell."""
    xy = np.array([(i, j) for j in range(n) for i in range(n)], float)
    points = np.column_stack((xy, np.zeros(len(xy))))
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            faces += [[a, b, c], [a, c, d]]
    return points, np.array(faces)


def patch_surface(points, faces, pairs=(1, 2), report=None):
    k = len(faces)
    return ClosedRVE(points, faces, np.tile(pairs, (k, 1)), np.zeros(k, bool),
                     np.full(k, -1), {} if report is None else report)


def min_angle(s):
    return triangle_angles(s.points, s.triangles).min()


def area(s):
    x = s.points[s.triangles]
    return np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1).sum() / 2


def boundary_edges(f):
    e, n = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
    return e[n == 1]


class SurfaceAngleTests(unittest.TestCase):
    def displaced(self):
        points, faces = grid()
        points[12, :2] = [2.93, 2.02]                 # interior node pushed next to its neighbour
        return patch_surface(points, faces)

    def test_interior_node_relocation_raises_angles_in_plane(self):
        source = self.displaced()
        result = improve_surface_angles(source, max_deviation=0.)
        self.assertLess(min_angle(source), 5)
        self.assertGreater(min_angle(result), 30)
        np.testing.assert_array_equal(result.points[:, 2], 0.)      # planar: zero deviation suffices
        self.assertAlmostEqual(area(result), area(source))
        np.testing.assert_allclose(result.points.min(axis=0), source.points.min(axis=0))
        np.testing.assert_allclose(result.points.max(axis=0), source.points.max(axis=0))

    def test_sliver_flip_keeps_coordinates(self):
        points = np.array([[0., 0, 0], [2., 0, 0], [1., .001, 0], [0., -1, 0]])
        source = patch_surface(points, np.array([[0, 1, 2], [1, 0, 3]]))
        result = improve_surface_angles(source, relocation=False, collapses=False)
        np.testing.assert_array_equal(result.points, source.points)
        self.assertEqual(result.report['surface_angle_repair']['flips'], 1)
        self.assertGreater(min_angle(result), min_angle(source))
        np.testing.assert_array_equal(boundary_edges(source.triangles), boundary_edges(result.triangles))

    def test_short_edge_collapse_removes_node(self):
        points, faces = grid()
        points = np.vstack((points, [2.02, 2.01, 0.]))  # extra node next to node 12
        t = int(np.flatnonzero((faces == 12).any(1) & (faces == 13).any(1) & (faces == 18).any(1))[0])
        a, b, c = faces[t]
        faces = np.vstack((np.delete(faces, t, axis=0), [[a, b, 25], [b, c, 25], [c, a, 25]]))
        source = patch_surface(points, faces)
        result = improve_surface_angles(source, relocation=False, flips=False)
        self.assertEqual(len(result.points), 25)
        self.assertGreater(result.report['surface_angle_repair']['collapses'], 0)
        self.assertGreater(min_angle(result), min_angle(source))
        self.assertAlmostEqual(area(result), area(source))

    def test_frozen_grain_is_untouched(self):
        source = self.displaced()
        frozen = patch_surface(source.points, source.triangles, pairs=(1, 7))
        result = improve_surface_angles(frozen, frozen_grain_ids=[7])
        np.testing.assert_array_equal(result.points, frozen.points)
        np.testing.assert_array_equal(result.triangles, frozen.triangles)

    def test_rve_plane_coordinates_are_exact(self):
        points, faces = grid()
        points[12, :2] = [2.93, 2.02]
        points[:, 2] = 3.                           # cap on the upper z face of a 4 x 4 x 3 box
        k = len(faces)
        source = ClosedRVE(points, faces, np.tile([1, -1], (k, 1)), np.ones(k, bool), np.full(k, 5),
                           {'rve_dimensions': [4., 4., 3.]})
        result = improve_surface_angles(source)
        self.assertGreater(min_angle(result), 30)
        self.assertTrue(np.all(result.points[:, 2] == 3.))

    def test_rve_plane_nodes_stay_fixed_without_dimensions(self):
        points, faces = grid()
        points[12, :2] = [2.93, 2.02]
        k = len(faces)
        source = ClosedRVE(points, faces, np.tile([1, -1], (k, 1)), np.ones(k, bool), np.full(k, 4), {})
        result = improve_surface_angles(source, flips=False, collapses=False)
        np.testing.assert_array_equal(result.points, source.points)

    def test_deviation_limit_blocks_nonplanar_relocation(self):
        source = self.displaced()
        source.points[12, 2] = .5                   # a bump: moving the node flattens it
        result = improve_surface_angles(source, max_deviation=1e-6, flips=False, collapses=False)
        np.testing.assert_array_equal(result.points[12], source.points[12])

    def test_intersection_rolls_back(self):
        source = self.displaced()
        with patch(MODULE + '.find_surface_intersections', side_effect=lambda p, f, triangle_ids: np.column_stack((triangle_ids, triangle_ids))):
            result = improve_surface_angles(source)
        np.testing.assert_array_equal(result.points, source.points)
        np.testing.assert_array_equal(result.triangles, source.triangles)
        self.assertEqual(result.report['surface_angle_repair']['history'][0]['accepted'], 0)

    def test_small_facet_opening_rolls_back(self):
        source = self.displaced()
        with patch(MODULE + '.small_facet_angles', side_effect=lambda p, f, m: (np.column_stack((np.arange(len(f)), np.arange(len(f)))), np.zeros(len(f)))):
            result = improve_surface_angles(source)
        np.testing.assert_array_equal(result.points, source.points)

    def test_disabled_and_already_good_are_identity(self):
        source = self.displaced()
        result = improve_surface_angles(source, enabled=False)
        np.testing.assert_array_equal(result.points, source.points)
        np.testing.assert_array_equal(result.triangles, source.triangles)
        good = patch_surface(*grid())
        result = improve_surface_angles(good)
        np.testing.assert_array_equal(result.triangles, good.triangles)
        self.assertEqual(result.report['surface_angle_repair']['history'], [])

    def test_volume_report_updates(self):
        # closed tetrahedral shell, one grain; a node on one face is displaced to make slivers
        points = np.array([[0., 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [3.9, .05, 0]])
        faces = np.array([[0, 2, 4], [4, 2, 1], [0, 4, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        shell = ClosedRVE(points, faces, np.ones((6, 2), int), np.ones(6, bool), np.array([4, 4, 4, 2, 1, 0]),
                          {'enclosed_grain_volumes': {'1': -999.}, 'rve_dimensions': [4., 4., 4.]})
        result = improve_surface_angles(shell)
        self.assertGreater(min_angle(result), min_angle(shell))
        x = result.points[result.triangles]
        actual = np.einsum('ij,ij->i', x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6
        self.assertAlmostEqual(result.report['enclosed_grain_volumes']['1'], actual)

    def test_bad_arguments(self):
        source = self.displaced()
        for kwargs in (dict(minimum_angle=0), dict(minimum_angle=60), dict(max_passes=-1),
                       dict(max_deviation=-1.), dict(minimum_facet_angle=180), dict(max_normal_change=90),
                       dict(enabled=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                improve_surface_angles(source, **kwargs)


if __name__ == '__main__':
    unittest.main()


class GrainFanTests(unittest.TestCase):
    def test_fan_check_detects_a_pinched_grain(self):
        from upxo.meshing.gbconformant.d3v2p0.surface_angles import _grain_fans_ok
        ring = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])              # one closed fan around node 0
        pairs = np.tile([5, 6], (4, 1)); ext = np.zeros(4, bool)
        self.assertTrue(_grain_fans_ok(0, ring, pairs, ext))
        two = np.vstack((ring, ring + [0, 4, 4]))                                  # a second fan of grain 5 at node 0
        two[4:, 0] = 0
        self.assertFalse(_grain_fans_ok(0, two, np.tile([5, 6], (8, 1)), np.zeros(8, bool)))
        self.assertFalse(_grain_fans_ok(0, ring[:3], pairs[:3], ext[:3]))       # open fan


class WedgeRuleTests(unittest.TestCase):
    def test_sharpens_rule(self):
        from upxo.meshing.gbconformant.d3v2p0.facet_angles import sharpens
        np.testing.assert_array_equal(sharpens([20, 20, 40, 40], [19, 21, 31, 29], 30.), [True, False, False, True])

    def test_edge_openings(self):
        from upxo.meshing.gbconformant.d3v2p0.facet_angles import edge_openings
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1.]])
        tris = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        m = {(0, 1): [0, 1, 2]}
        self.assertAlmostEqual(edge_openings(p, tris, [(0, 1)], m)[0], 90.)
        self.assertEqual(edge_openings(p, tris, [(1, 2)], {})[0], 180.)

    def test_repair_with_wedge_limit_never_sharpens(self):
        from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles
        points, faces = grid()
        points[12, :2] = [2.93, 2.02]
        points[[6, 7, 8], 2] = .3                                    # a gentle fold through the patch
        src = patch_surface(points, faces)
        out = improve_surface_angles(src, wedge_limit=170.)
        before = small_facet_angles(src.points, src.triangles, 170.)[1]
        after = small_facet_angles(out.points, out.triangles, 170.)[1]
        self.assertGreaterEqual(after.min() if len(after) else 180., (before.min() if len(before) else 180.) - 1e-6)
        with self.assertRaises(ValueError):
            improve_surface_angles(src, wedge_limit=0.)
