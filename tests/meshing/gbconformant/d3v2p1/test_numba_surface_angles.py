"""Compiled surface-angle repair: d3v2p0's tests run against it, and it makes the same choices as d3v2p0."""
import functools
import unittest

import numpy as np

from ..d3v2p0 import test_surface_angles as base
from upxo.meshing.gbconformant.d3v2p1 import backend

HAVE_NUMBA = backend.numba_available()
if HAVE_NUMBA:
    from upxo.meshing.gbconformant.d3v2p0 import surface_angles as ref
    from upxo.meshing.gbconformant.d3v2p1 import numba_surface_angles as compiled

COMPILED_MODULE = 'upxo.meshing.gbconformant.d3v2p1.numba_surface_angles'
NAMES = {1: 'flip', 2: 'relocation', 3: 'collapse'}


class _Swap:
    def setUp(self):
        self._saved = base.improve_surface_angles, base.MODULE
        base.improve_surface_angles = compiled.improve_surface_angles_compiled
        base.MODULE = COMPILED_MODULE

    def tearDown(self):
        base.improve_surface_angles, base.MODULE = self._saved


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class CompiledSurfaceAngleTests(_Swap, base.SurfaceAngleTests):
    pass


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class CompiledWedgeRuleTests(_Swap, base.WedgeRuleTests):
    pass


def rough_grid(seed, n=9):
    points, faces = base.grid(n)
    rng = np.random.default_rng(seed)
    inner = np.all((points[:, :2] > 0) & (points[:, :2] < n - 1), axis=1)
    points[inner, :2] += rng.uniform(-.45, .45, size=(inner.sum(), 2))
    points[inner, 2] += rng.uniform(-.05, .05, size=inner.sum())
    return points, faces


def same_choice(op, result):
    kind, rows, new, node, x, y, z, gain, keep, gone = result
    if op is None:
        return kind == 0
    if kind == 0 or NAMES[kind] != op['kind']:
        return False
    if not (np.array_equal(rows, np.asarray(op['rows'])) and np.array_equal(new, np.asarray(op['new']))):
        return False
    if abs(gain - op['gain']) > 1e-9:
        return False
    if op['kind'] == 'relocation' and (node != op['node'] or np.abs(np.array([x, y, z]) - op['point']).max() > 1e-9):
        return False
    if op['kind'] == 'collapse' and not (np.array_equal(keep, op['keep']) and np.array_equal(gone, np.asarray(op['gone']))):
        return False
    return True


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class AdjacencyTests(unittest.TestCase):
    def test_arrays_match_the_reference_topology(self):
        points, faces = rough_grid(0)
        alive = np.ones(len(faces), bool)
        alive[::7] = False
        frozen = np.zeros(len(faces), bool)
        frozen[:4] = True
        patch = np.arange(len(faces)) % 3
        topo = ref._Topology(points, faces, alive, patch, frozen, np.zeros(len(faces), bool), np.full(len(faces), -1))
        adj = compiled.Adjacency(len(points), faces, alive, patch, frozen)
        np.testing.assert_array_equal(adj.node_rows, topo.node_tri)
        np.testing.assert_array_equal(adj.node_off, topo.offsets)
        np.testing.assert_array_equal(adj.fixed, topo.fixed)
        edges = list(topo.edge_tri.items())
        self.assertEqual(len(edges), len(adj.edge_u))
        for i, ((a, b), rows) in enumerate(edges):
            self.assertEqual((adj.edge_u[i], adj.edge_v[i]), (a, b))
            self.assertEqual(adj.erows[adj.eoff[i]:adj.eoff[i + 1]].tolist(), [int(r) for r in rows])
        for node in range(len(points)):
            kind = topo.kind(node)
            expected = (0, -1, -1) if kind is None else ((1, -1, -1) if kind == 'interior' else (2, kind[1], kind[2]))
            got = compiled._node_kind(node, adj.fixed, adj.curve_off, adj.curve_edge, adj.curve_key, adj.edge_u, adj.edge_v)
            self.assertEqual(tuple(got), expected, f'node {node}')

    def test_edge_rows_match_the_reference(self):
        points, faces = rough_grid(1)
        alive = np.ones(len(faces), bool)
        patch = np.zeros(len(faces), int)
        frozen = np.zeros(len(faces), bool)
        topo = ref._Topology(points, faces, alive, patch, frozen, np.zeros(len(faces), bool), np.full(len(faces), -1))
        adj = compiled.Adjacency(len(points), faces, alive, patch, frozen)
        for (a, b), rows in topo.edge_tri.items():
            self.assertEqual(adj.edge_rows(a, b), [int(r) for r in rows])
            self.assertEqual(adj.edge_rows(b, a), [int(r) for r in rows])
        self.assertEqual(adj.edge_rows(0, len(points) - 1), [])


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class SameChoiceTests(unittest.TestCase):
    """For every poor triangle the compiled search picks what d3v2p0's picks."""

    def reference_choice(self, t, p, f, topo, ang, wedge, max_dev, cos_normal):
        best = None
        for op in ref._candidates(t, p, f, topo, ang, True, True, True, max_dev, cos_normal, None):
            if wedge is not None and not ref._keeps_openings(op, p, f, topo, wedge):
                continue
            if best is None or op['gain'] > best['gain']:
                best = op
        return best

    def test_every_poor_triangle(self):
        checked = 0
        for seed in range(3):
            points, faces = rough_grid(seed)
            k = len(faces)
            alive, frozen = np.ones(k, bool), np.zeros(k, bool)
            pairs, exterior, face = np.tile([1, 2], (k, 1)), np.zeros(k, bool), np.full(k, -1)
            patch = np.zeros(k, np.int64)
            ang = ref.triangle_angles(points, faces).min(axis=1)
            topo = ref._Topology(points, faces, alive, patch, frozen, exterior, face)
            topo.pairs = pairs
            adj = compiled.Adjacency(len(points), faces, alive, patch, frozen)
            max_dev = .1 * float(np.median(np.linalg.norm(points[faces[:, 1]] - points[faces[:, 0]], axis=1)))
            cos_normal = float(np.cos(np.radians(60.)))
            args = (points, faces.astype(np.int64), ang, adj.node_off, adj.node_rows, adj.ekey_sorted, adj.esort, adj.width,
                    adj.eoff, adj.erows, adj.edge_u, adj.edge_v, adj.fixed, adj.curve_off, adj.curve_edge, adj.curve_key,
                    patch, pairs, exterior, face, np.zeros(3), False)
            for wedge in (None, 30.):
                for t in np.flatnonzero(ang < 30.):
                    t = int(t)
                    op = self.reference_choice(t, points, faces, topo, ang, wedge, max_dev, cos_normal)
                    res = compiled.evaluate(t, *args, True, True, True, max_dev, cos_normal,
                                            -1. if wedge is None else wedge, adj.max_owner)
                    self.assertTrue(same_choice(op, res), f'seed {seed}, wedge {wedge}, triangle {t}')
                    checked += 1
        self.assertGreater(checked, 40)


@unittest.skipUnless(HAVE_NUMBA, 'numba not available')
class WholeRepairTests(unittest.TestCase):
    def test_same_surface_as_d3v2p0(self):
        for seed in range(3):
            for wedge in (None, 30.):
                with self.subTest(seed=seed, wedge=wedge):
                    source = base.patch_surface(*rough_grid(seed))
                    a = ref.improve_surface_angles(source, wedge_limit=wedge)
                    b = compiled.improve_surface_angles_compiled(source, wedge_limit=wedge)
                    np.testing.assert_allclose(a.points, b.points, rtol=0, atol=1e-9)    # last-digit float differences
                    np.testing.assert_array_equal(a.triangles, b.triangles)
                    np.testing.assert_array_equal(a.grain_pairs, b.grain_pairs)
                    self.assertGreater(base.min_angle(b), base.min_angle(source))
                    ra, rb = a.report['surface_angle_repair'], b.report['surface_angle_repair']
                    self.assertEqual((ra['flips'], ra['relocations'], ra['collapses']),
                                     (rb['flips'], rb['relocations'], rb['collapses']))

    def test_report_names_the_tier(self):
        out = compiled.improve_surface_angles_compiled(base.patch_surface(*rough_grid(0)))
        r = out.report['surface_angle_repair']
        self.assertEqual(r['n_workers'], 1)
        self.assertEqual((r['backend']['requested'], r['backend']['used'], r['backend']['fallback']), ('numba', 'numba', None))
        self.assertTrue(r['full_validation_required'])

    def test_intersection_search_tier_does_not_change_the_result(self):
        source = base.patch_surface(*rough_grid(1))
        auto = compiled.improve_surface_angles_compiled(source, wedge_limit=30.)
        for kwargs in (dict(backend='numpy'), dict(backend='numba', n_workers=1), dict(backend='parallel', n_workers=2)):
            with self.subTest(kwargs=kwargs):
                other = compiled.improve_surface_angles_compiled(source, wedge_limit=30., **kwargs)
                np.testing.assert_array_equal(auto.points, other.points)
                np.testing.assert_array_equal(auto.triangles, other.triangles)

    def test_bad_backend_arguments(self):
        source = base.patch_surface(*rough_grid(0))
        for kwargs in (dict(backend='gpu'), dict(n_workers=-1), dict(n_workers=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compiled.improve_surface_angles_compiled(source, **kwargs)

    def test_plan_report_is_stored(self):
        report = dict(requested='numba', used='numba', n_workers=3, fallback=None, cores={})
        out = compiled.improve_surface_angles_compiled(base.patch_surface(*rough_grid(0)), plan_report=report)
        self.assertEqual(out.report['surface_angle_repair']['backend'], report)

    def test_frozen_grain_surface_is_untouched(self):
        points, faces = rough_grid(2)
        source = base.patch_surface(points, faces, pairs=(7, 2))
        out = compiled.improve_surface_angles_compiled(source, frozen_grain_ids=[7])
        np.testing.assert_array_equal(out.points, source.points)
        np.testing.assert_array_equal(out.triangles, source.triangles)

    def test_progress_is_reported_and_may_fail(self):
        calls = []

        def progress(message, step, total):
            calls.append((step, total))
            raise RuntimeError('a display that fails must not stop the repair')

        out = compiled.improve_surface_angles_compiled(base.patch_surface(*rough_grid(0)), progress=progress)
        self.assertTrue(calls)
        self.assertGreater(base.min_angle(out), 0)


if __name__ == '__main__':
    unittest.main()
