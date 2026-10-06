"""numba pair-test kernel: same pairs as d3v2p0 on every tier and thread count."""
import unittest
import numpy as np
from tests.meshing.gbconformant.d3v2p0 import test_surface_intersections as base
from upxo.meshing.gbconformant.d3v2p0.surface_intersections import (find_surface_intersections as reference_find,
                                                                     _intersecting_pairs)
from upxo.meshing.gbconformant.d3v2p1 import backend
from upxo.meshing.gbconformant.d3v2p1.surface_intersections import find_surface_intersections

HAVE_NUMBA = backend.numba_available()


def numba_find(*args, **kwargs):
    kwargs.setdefault('backend', 'numba')
    return find_surface_intersections(*args, **kwargs)


def soup(seed, n=6000):
    """Random triangles with shared vertices, shared edges and coplanar groups."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 12, size=(n, 3))
    points = (centres[:, None] + rng.normal(0, .4, size=(n, 3, 3))).reshape(-1, 3)
    triangles = np.arange(len(points)).reshape(-1, 3)
    triangles[1::7, 0] = triangles[0::7, 0][:len(triangles[1::7])]
    triangles[2::7, :2] = triangles[0::7, :2][:len(triangles[2::7])]
    points[::11, 2] = np.round(points[::11, 2])
    return points, triangles


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class NumbaIntersectionSuite(base.IntersectionTests):
    """d3v2p0's intersection tests on the numba tier."""

    def setUp(self):
        self._saved = base.find_surface_intersections
        base.find_surface_intersections = staticmethod(numba_find)

    def tearDown(self):
        base.find_surface_intersections = self._saved


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class NumbaKernelTests(unittest.TestCase):
    def test_kernel_matches_reference_pair_test(self):
        from upxo.meshing.gbconformant.d3v2p1.numba_intersections import pairs_intersect
        import numba
        points, triangles = soup(0, n=3000)
        rng = np.random.default_rng(1)
        pairs = np.sort(rng.integers(0, len(triangles), size=(40000, 2)), axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        expected = _intersecting_pairs(points, triangles, pairs, 1e-9)
        previous = numba.get_num_threads()
        try:
            for threads in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
                numba.set_num_threads(threads)
                got = pairs_intersect(points, triangles.astype(np.int64), pairs.astype(np.int64), 1e-9)
                np.testing.assert_array_equal(got, expected)
        finally:
            numba.set_num_threads(previous)
        self.assertGreater(expected.sum(), 0)

    def test_search_matches_reference_on_every_tier(self):
        for seed in range(2):
            points, triangles = soup(seed)
            ref = reference_find(points, triangles)
            ids = np.arange(0, len(triangles), 5)
            ref_subset = reference_find(points, triangles, triangle_ids=ids)
            for kwargs in (dict(backend='numba'), dict(backend='numba', n_workers=1), dict(backend='auto'),
                           dict(backend='parallel', n_workers=2), dict(backend='numpy')):
                with self.subTest(seed=seed, **kwargs):
                    np.testing.assert_array_equal(find_surface_intersections(points, triangles, **kwargs), ref)
                    np.testing.assert_array_equal(
                        find_surface_intersections(points, triangles, triangle_ids=ids, **kwargs), ref_subset)

    def test_auto_prefers_numba(self):
        self.assertEqual(backend.plan(numba_kernels=True).used, 'numba')
        self.assertEqual(backend.plan('parallel', n_workers=2, numba_kernels=True).used,
                         'parallel' if backend.usable_cores() > 1 else 'numpy')


class NumbaUnavailableTests(unittest.TestCase):
    def test_falls_back_without_numba(self):
        saved = dict(backend._CACHE)
        try:
            backend._CACHE['numba'] = False
            p = backend.plan('numba', n_workers=2, numba_kernels=True)
            self.assertIn('numba not available', p.fallback)
            self.assertNotEqual(p.used, 'numba')
            self.assertNotEqual(backend.plan(numba_kernels=True).used, 'numba')
            points, triangles = soup(0, n=1500)
            np.testing.assert_array_equal(find_surface_intersections(points, triangles, backend='numba'),
                                          reference_find(points, triangles))
        finally:
            backend._CACHE.clear()
            backend._CACHE.update(saved)


if __name__ == '__main__':
    unittest.main()
