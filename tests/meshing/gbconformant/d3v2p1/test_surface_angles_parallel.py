"""d3v2p0's surface-angle tests run against d3v2p1, serially and with workers,
plus worker-count independence and the vectorised topology."""
import functools
import os
import sys
import unittest
from unittest import mock
import numpy as np
from ..d3v2p0 import test_surface_angles as base
from upxo.meshing.gbconformant.d3v2p0.surface_angles import _Topology, triangle_angles
from upxo.meshing.gbconformant.d3v2p1 import backend, surface_angles as fast

FAST_MODULE = 'upxo.meshing.gbconformant.d3v2p1.surface_angles'
COMPILED_MODULE = 'upxo.meshing.gbconformant.d3v2p1.numba_surface_angles'


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


HAVE_NUMBA = backend.numba_available()


class CompiledTierSelectionTests(unittest.TestCase):
    """The compiled tier runs only when asked for (backend='numba' or UPXO_BACKEND=numba), never from 'auto',
    and every way it can be unavailable ends in the numpy / parallel tiers with the reason in the report."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ('UPXO_BACKEND', 'UPXO_N_WORKERS'):
            os.environ.pop(key, None)
        points, faces = base.grid(9)
        rng = np.random.default_rng(3)
        inner = np.all((points[:, :2] > 0) & (points[:, :2] < 8), axis=1)
        points[inner, :2] += rng.uniform(-.45, .45, size=(inner.sum(), 2))
        self.source = base.patch_surface(points, faces)
        self.kwargs = dict(wedge_limit=30., n_workers=1)

    def tearDown(self):
        self._env.stop()

    def tier(self, out):
        return out.report['surface_angle_repair']['backend']

    def same_surface(self, a, b):
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.triangles, b.triangles)

    def test_auto_does_not_select_the_compiled_tier(self):
        out = fast.improve_surface_angles(self.source, **self.kwargs)
        self.assertNotEqual(self.tier(out)['used'], 'numba')
        self.assertEqual(self.tier(out)['requested'], 'auto')

    def test_explicit_numpy_wins_over_the_environment(self):
        os.environ['UPXO_BACKEND'] = 'numba'
        out = fast.improve_surface_angles(self.source, backend='numpy', **self.kwargs)
        self.assertEqual(self.tier(out)['used'], 'numpy')

    def test_nothing_to_do_never_loads_the_compiled_tier(self):
        with mock.patch.dict(sys.modules, {COMPILED_MODULE: None}):
            for kwargs in (dict(enabled=False), dict(max_passes=0)):
                out = fast.improve_surface_angles(self.source, backend='numba', **kwargs, **self.kwargs)
                self.same_surface(out, self.source)
                self.assertEqual(self.tier(out)['used'], 'numpy')

    def test_without_numba_the_other_tiers_run(self):
        with mock.patch.dict(backend._CACHE, {'numba': False}):
            out = fast.improve_surface_angles(self.source, backend='numba', **self.kwargs)
        reference = fast.improve_surface_angles(self.source, backend='numpy', **self.kwargs)
        self.same_surface(out, reference)
        self.assertNotEqual(self.tier(out)['used'], 'numba')
        self.assertIn('numba not available', self.tier(out)['fallback'])

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_explicit_request_runs_the_compiled_tier(self):
        from upxo.meshing.gbconformant.d3v2p1.numba_surface_angles import improve_surface_angles_compiled
        out = fast.improve_surface_angles(self.source, backend='numba', **self.kwargs)
        self.assertEqual({k: self.tier(out)[k] for k in ('requested', 'used', 'fallback')},
                         dict(requested='numba', used='numba', fallback=None))
        self.same_surface(out, improve_surface_angles_compiled(self.source, **self.kwargs))
        self.assertGreater(base.min_angle(out), base.min_angle(self.source))

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_environment_request_runs_the_compiled_tier(self):
        os.environ['UPXO_BACKEND'] = 'numba'
        out = fast.improve_surface_angles(self.source, **self.kwargs)
        self.assertEqual(self.tier(out)['used'], 'numba')

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_failed_self_test_falls_back(self):
        from upxo.meshing.gbconformant.d3v2p1 import numba_surface_angles
        with mock.patch.dict(numba_surface_angles._READY, {'result': (False, 'boom')}):
            out = fast.improve_surface_angles(self.source, backend='numba', **self.kwargs)
        self.same_surface(out, fast.improve_surface_angles(self.source, backend='numpy', **self.kwargs))
        self.assertEqual(self.tier(out)['used'], 'numpy')
        self.assertIn('compiled angle repair not usable (boom)', self.tier(out)['fallback'])

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_unimportable_compiled_module_falls_back(self):
        with mock.patch.dict(sys.modules, {COMPILED_MODULE: None}):
            out = fast.improve_surface_angles(self.source, backend='numba', **self.kwargs)
        self.assertEqual(self.tier(out)['used'], 'numpy')
        self.assertIn('numba_surface_angles', self.tier(out)['fallback'])

    @unittest.skipUnless(HAVE_NUMBA, 'numba not available')
    def test_self_test_passes_where_numba_works(self):
        from upxo.meshing.gbconformant.d3v2p1.numba_surface_angles import compiled_ready
        self.assertEqual(compiled_ready(), (True, None))


class WedgeRuleAgreementTests(unittest.TestCase):
    """d3v2p1's no-sharpening check answers as d3v2p0's does for every candidate operation. For a relocation
    the opening before the operation is measured with the node at its old position."""

    def surface(self, seed):
        points, faces = base.grid(9)
        rng = np.random.default_rng(seed)
        inner = np.all((points[:, :2] > 0) & (points[:, :2] < 8), axis=1)
        points[inner, :2] += rng.uniform(-.45, .45, size=(inner.sum(), 2))
        points[inner, 2] += rng.uniform(-.25, .25, size=inner.sum())
        return base.patch_surface(points, faces)

    def candidates(self, s):
        p, f = s.points, s.triangles
        k = len(f)
        alive, frozen = np.ones(k, bool), np.zeros(k, bool)
        patch = np.zeros(k, np.int64)
        ang = triangle_angles(p, f).min(axis=1)
        topo = fast._FastTopology(p, f, alive, patch, frozen, np.zeros(k, bool), np.full(k, -1))
        topo.pairs = s.grain_pairs
        cos_normal = float(np.cos(np.radians(60.)))
        for t in np.flatnonzero(ang < 30.):
            for op in fast._candidates(int(t), p, f, topo, ang, True, True, True, 1., cos_normal, None):
                yield op, p, f, topo

    def test_same_answer_as_d3v2p0(self):
        from upxo.meshing.gbconformant.d3v2p0.surface_angles import _keeps_openings as reference
        checked = rejected_relocations = 0
        for seed in range(4):
            for op, p, f, topo in self.candidates(self.surface(seed)):
                for limit in (30., 60., 100.):
                    expected = reference(op, p, f, topo, limit)
                    self.assertEqual(fast._keeps_openings(op, p, f, topo, limit), expected,
                                     f'seed {seed}, {op["kind"]}, limit {limit}')
                    checked += 1
                    rejected_relocations += op['kind'] == 'relocation' and not expected
        self.assertGreater(checked, 300)
        self.assertGreater(rejected_relocations, 0)               # the rule was exercised on moved nodes


if __name__ == '__main__':
    unittest.main()
