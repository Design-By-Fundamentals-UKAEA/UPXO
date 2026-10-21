"""d3v2p1.facet_angles.small_facet_angles: d3v2p0's result, element for element."""
import unittest
import numpy as np
from ..d3v2p0 import test_facet_angles as base
from upxo.meshing.gbconformant.d3v2p0.facet_angles import small_facet_angles as reference
from upxo.meshing.gbconformant.d3v2p1.facet_angles import small_facet_angles
from upxo.meshing.gbconformant.d3v2p1.mesh_tables import EdgeIncidence


class Facet1OwnTests(base.FacetAngleTests):
    """d3v2p0's facet-angle tests on the d3v2p1 function."""

    def setUp(self):
        self._saved = base.small_facet_angles
        base.small_facet_angles = small_facet_angles

    def tearDown(self):
        base.small_facet_angles = self._saved


def soup(seed, dtype=np.int64, n_nodes=80, n_triangles=500):
    """Random triangles over few points: shared edges, junction edges (3+ triangles), coincident points."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 3, size=(n_nodes, 3))
    points[5] = points[4]                                         # a zero-length edge
    points[10:14, 2] = 0.                                         # coplanar neighbours
    triangles = np.array([rng.choice(n_nodes, 3, replace=False) for _ in range(n_triangles)]).astype(dtype)
    return points, triangles


class EquivalenceTests(unittest.TestCase):
    def assertSame(self, points, triangles, minimum, **kw):
        want = reference(points, triangles, minimum)
        got = small_facet_angles(points, triangles, minimum, **kw)
        for g, w in zip(got, want):
            self.assertEqual((g.shape, g.dtype), (w.shape, w.dtype))
            np.testing.assert_array_equal(g, w)
        return want

    def test_random_soups_every_threshold(self):
        for seed in range(4):
            for dtype in (np.int64, np.int32):
                points, triangles = soup(seed, dtype)
                for minimum in (0, .1, 5., 30., 90., 179.):
                    with self.subTest(seed=seed, dtype=dtype.__name__, minimum=minimum):
                        hits, _ = self.assertSame(points, triangles, minimum)
                        if minimum == 30.:
                            self.assertGreater(len(hits), 0)

    def test_precomputed_incidence_is_reused_across_coordinate_sets(self):
        points, triangles = soup(7)
        incidence = EdgeIncidence(triangles)
        rng = np.random.default_rng(8)
        for step in range(3):
            moved = points + rng.normal(0, .05, size=points.shape)
            self.assertSame(moved, triangles, 30., incidence=incidence)

    def test_mismatched_incidence_and_bad_threshold_raise(self):
        points, triangles = soup(0)
        with self.assertRaises(ValueError):
            small_facet_angles(points, triangles[:-1], 10., incidence=EdgeIncidence(triangles))
        for module in (reference, small_facet_angles):
            for bad in (-1., 180., np.nan):
                with self.assertRaises(ValueError):
                    module(points, triangles, bad)

    def test_empty_triangle_list(self):
        points, _ = soup(0)
        self.assertSame(points, np.empty((0, 3), int), 10.)

    def test_real_surface(self):
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        rng = np.random.default_rng(0)
        seeds = rng.uniform(0, 14, size=(10, 3))
        x = np.stack(np.meshgrid(*[np.arange(14) + .5] * 3, indexing='ij'), -1)
        labels = (np.argmin(((x[..., None, :] - seeds) ** 2).sum(-1), -1) + 1).astype(int)
        s = smooth_interfaces(labels, iterations=30, relaxation=.9, junction_iterations=10, junction_relaxation=.5)
        for minimum in (1., 15., 30.):
            self.assertSame(s.points, s.triangles, minimum)
            self.assertSame(s.points, s.triangles, minimum, incidence=EdgeIncidence(s.triangles))


if __name__ == '__main__':
    unittest.main()
