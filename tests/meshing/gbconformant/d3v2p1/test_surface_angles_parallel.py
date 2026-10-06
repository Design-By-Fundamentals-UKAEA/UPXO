"""d3v2p0's surface-angle tests run against d3v2p1, serially and with workers,
plus worker-count independence and the vectorised topology."""
import functools
import unittest
import numpy as np
from ..d3v2p0 import test_surface_angles as base
from upxo.meshing.gbconformant.d3v2p0.surface_angles import _Topology
from upxo.meshing.gbconformant.d3v2p1 import surface_angles as fast

FAST_MODULE = 'upxo.meshing.gbconformant.d3v2p1.surface_angles'


class _Swap:
    n_workers = 1

    def setUp(self):
        self._saved = base.improve_surface_angles, base.MODULE
        base.improve_surface_angles = functools.partial(fast.improve_surface_angles, n_workers=self.n_workers)
        base.MODULE = FAST_MODULE

    def tearDown(self):
        base.improve_surface_angles, base.MODULE = self._saved


class SerialSurfaceAngleTests(_Swap, base.SurfaceAngleTests):
    pass


class SerialWedgeRuleTests(_Swap, base.WedgeRuleTests):
    pass


class ParallelSurfaceAngleTests(_Swap, base.SurfaceAngleTests):
    n_workers = 2


class WorkerCountTests(unittest.TestCase):
    def rough_grid(self, seed):
        points, faces = base.grid(9)
        rng = np.random.default_rng(seed)
        inner = np.all((points[:, :2] > 0) & (points[:, :2] < 8), axis=1)
        points[inner, :2] += rng.uniform(-.45, .45, size=(inner.sum(), 2))
        return base.patch_surface(points, faces)

    def test_worker_count_does_not_change_the_result(self):
        for seed in range(2):
            with self.subTest(seed=seed):
                s = self.rough_grid(seed)
                one = fast.improve_surface_angles(s, n_workers=1, wedge_limit=30.)
                two = fast.improve_surface_angles(s, n_workers=3, wedge_limit=30.)
                np.testing.assert_array_equal(one.points, two.points)
                np.testing.assert_array_equal(one.triangles, two.triangles)
                self.assertGreater(base.min_angle(one), base.min_angle(s))
                self.assertAlmostEqual(base.area(one), base.area(s))
                self.assertEqual(one.report['surface_angle_repair']['n_workers'], 1)

    def test_bad_max_rounds(self):
        s = self.rough_grid(0)
        for value in (0, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fast.improve_surface_angles(s, max_rounds=value)

    def test_fast_topology_matches_reference(self):
        s = self.rough_grid(0)
        f = s.triangles
        alive = np.ones(len(f), bool)
        alive[::7] = False
        frozen = np.zeros(len(f), bool)
        frozen[:4] = True
        patch = np.arange(len(f)) % 3
        args = (s.points, f, alive, patch, frozen, np.zeros(len(f), bool), np.full(len(f), -1))
        a, b = _Topology(*args), fast._FastTopology(*args)
        np.testing.assert_array_equal(a.node_tri, b.node_tri)
        np.testing.assert_array_equal(a.offsets, b.offsets)
        np.testing.assert_array_equal(a.fixed, b.fixed)
        self.assertEqual(list(a.edge_tri.items()), list(b.edge_tri.items()))
        self.assertEqual(list(a.curve.items()), list(b.curve.items()))


if __name__ == '__main__':
    unittest.main()
