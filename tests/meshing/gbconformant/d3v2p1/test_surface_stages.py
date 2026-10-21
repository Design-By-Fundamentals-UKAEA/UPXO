"""geometry_guard, guard_relaxation, intersection_repair and surface_quality on d3v2p1:
the same result as d3v2p0 on every tier, with and without numba."""
import unittest
from unittest import mock
import numpy as np
from ..d3v2p0 import test_geometry_guard as guard_base
from ..d3v2p0 import test_intersection_repair as crossing_base
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE
from upxo.meshing.gbconformant.d3v2p0 import (geometry_guard as guard0, guard_relaxation as relax0,
                                              intersection_repair as crossing0, surface_quality as quality0)
from upxo.meshing.gbconformant.d3v2p1 import (backend, geometry_guard as guard1, guard_relaxation as relax1,
                                              intersection_repair as crossing1, surface_quality as quality1)

HAVE_NUMBA = backend.numba_available()
TIERS = [dict(backend='numpy'), dict(backend='parallel', n_workers=2), dict(backend='auto')]
if HAVE_NUMBA:
    TIERS.append(dict(backend='numba'))


def labels(seed, n=14, grains=10):
    """Voronoi-like grains on a voxel grid."""
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, n, size=(grains, 3))
    x = np.stack(np.meshgrid(*[np.arange(n) + .5] * 3, indexing='ij'), -1)
    return (np.argmin(((x[..., None, :] - seeds) ** 2).sum(-1), -1) + 1).astype(int)


def smoothed(seed, n=14):
    return smooth_interfaces(labels(seed, n), iterations=30, relaxation=.9, junction_iterations=10,
                             junction_relaxation=.5), [float(n)] * 3


def interfaces_equal(testcase, a, b):
    for name in ('points', 'triangles', 'grain_pairs', 'original_points', 'node_kind', 'fixed_axes',
                 'junction_edges'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def tiers_including_numba_off():
    """Each tier, and 'auto' with numba reported as unavailable."""
    for kwargs in TIERS:
        yield kwargs, mock.patch.dict(backend._CACHE, {}, clear=False)
    yield dict(backend='auto'), mock.patch.dict(backend._CACHE, {'numba': False})


class GuardAndRelaxationTests(unittest.TestCase):
    def test_guard_and_relaxation_match_d3v2p0(self):
        for seed in (0, 1):
            surface, dims = smoothed(seed)
            kw = dict(boundary_clearance=.05, max_passes=100, minimum_facet_angle=15.)
            ref, ref_report = guard0.stabilize_interfaces(surface, dims, **kw)
            self.assertGreater(ref_report['moved_nodes'], 0)                    # the guard has work to do
            relax_kw = dict(iterations=3, margin_rings=2, minimum_facet_angle=15., boundary_clearance=.05)
            ref_relaxed, ref_relax_report = relax0.relax_guarded_interfaces(ref, surface, dims, **relax_kw)
            for tier, context in tiers_including_numba_off():
                with self.subTest(seed=seed, **tier), context:
                    out, report = guard1.stabilize_interfaces(surface, dims, **kw, **tier)
                    interfaces_equal(self, out, ref)
                    self.assertEqual(report, ref_report)
                    relaxed, relax_report = relax1.relax_guarded_interfaces(out, surface, dims, **relax_kw, **tier)
                    interfaces_equal(self, relaxed, ref_relaxed)
                    self.assertEqual(relax_report, ref_relax_report)

    def test_disabled_and_failure_behaviour_match(self):
        surface, dims = smoothed(0)
        out, report = guard1.stabilize_interfaces(surface, dims, enabled=False)
        ref, ref_report = guard0.stabilize_interfaces(surface, dims, enabled=False)
        interfaces_equal(self, out, ref)
        self.assertEqual(report, ref_report)
        a = np.ones((4, 4, 4), int)
        a[2:] = 2
        s = smooth_interfaces(a, iterations=0)
        s.points[s.original_points[:, 1] == 1, 1] = 0
        for module in (guard0, guard1):
            with self.assertRaises(RuntimeError):
                module.stabilize_interfaces(s, [4, 4, 4], boundary_clearance=.1, max_correction=.001)

    def test_progress_messages_match(self):
        surface, dims = smoothed(1)
        calls = {}
        for name, module in (('p0', guard0), ('p1', guard1)):
            calls[name] = []
            module.stabilize_interfaces(surface, dims, boundary_clearance=.05, max_passes=100, minimum_facet_angle=15.,
                                        progress=lambda *args, name=name: calls[name].append(args))
        self.assertEqual(calls['p0'], calls['p1'])
        self.assertGreater(len(calls['p0']), 1)

    def test_bad_backend_arguments_fail_before_work(self):
        surface, dims = smoothed(0)
        for call in (lambda **k: guard1.stabilize_interfaces(surface, dims, **k),
                     lambda **k: relax1.relax_guarded_interfaces(surface, surface, dims, **k)):
            with self.assertRaises(ValueError):
                call(backend='gpu')
            with self.assertRaises(ValueError):
                call(n_workers=-1)

    def test_fold_counts_match_d3v2p0(self):
        from upxo.meshing.gbconformant.d3v2p1.mesh_tables import EdgeIncidence
        for seed in (0, 1, 2):
            surface, _ = smoothed(seed)
            for exclude in ((), (2,), (1, 3, 7)):
                for thresholds in ((150., 120.), (179., 90., 30.)):
                    kw = dict(thresholds=thresholds, exclude_grain_ids=exclude)
                    want = relax0.fold_counts(surface.points, surface.triangles, surface.grain_pairs, **kw)
                    got = relax1.fold_counts(surface.points, surface.triangles, surface.grain_pairs, **kw)
                    shared = relax1.fold_counts(surface.points, surface.triangles, surface.grain_pairs,
                                                incidence=EdgeIncidence(surface.triangles), **kw)
                    self.assertEqual(got, want)
                    self.assertEqual(shared, want)
                    self.assertGreater(sum(want.values()), 0)

    def test_check_nodes_outside_is_d3v2p0s(self):
        self.assertIs(relax1.check_nodes_outside, relax0.check_nodes_outside)


class GuardOwnTests(guard_base.GeometryGuardTests):
    """d3v2p0's guard tests on the d3v2p1 function."""

    def setUp(self):
        self._saved = guard_base.stabilize_interfaces
        guard_base.stabilize_interfaces = guard1.stabilize_interfaces

    def tearDown(self):
        guard_base.stabilize_interfaces = self._saved


class CrossingRepairOwnTests(crossing_base.CrossingRepairTests):
    """d3v2p0's crossing-repair tests on the d3v2p1 function."""

    def setUp(self):
        self._saved = crossing_base.repair_interface_crossings
        crossing_base.repair_interface_crossings = crossing1.repair_interface_crossings

    def tearDown(self):
        crossing_base.repair_interface_crossings = self._saved


def flaps():
    p = np.array([[18, 13.5, 20], [18, 13, 20], [17.74752589, 13.59320828, 20.12928175],
                  [17.75872795, 13.40905732, 20.35063656], [17.85495074, 13.32559319, 20.15184337],
                  [17.60260551, 13.53186343, 20.10729973], [17.63693785, 13.22328002, 20.38742964]])
    f = np.array([[0, 2, 1], [2, 3, 1], [0, 4, 5], [6, 4, 1], [5, 4, 6]])
    pairs = np.array([[16, 18], [16, 18], [16, 77], [16, 77], [16, 77]])
    return ClosedRVE(p, f, pairs, np.zeros(5, bool), np.full(5, -1), {'rve_dimensions': [30, 35, 40]})


def surface_of(seed):
    """The interfaces of a smoothed volume as a ClosedRVE, with slivers from a rough smoothing."""
    s, dims = smoothed(seed)
    n = len(s.triangles)
    return ClosedRVE(s.points.copy(), s.triangles.copy(), s.grain_pairs.copy(), np.zeros(n, bool),
                     np.full(n, -1), {'rve_dimensions': dims})


class CrossingAndQualityTests(unittest.TestCase):
    def test_crossing_repair_matches_d3v2p0(self):
        ref = crossing0.repair_interface_crossings(flaps(), max_displacement=.35)
        for tier, context in tiers_including_numba_off():
            with self.subTest(**tier), context:
                out = crossing1.repair_interface_crossings(flaps(), max_displacement=.35, **tier)
                np.testing.assert_array_equal(out.points, ref.points)
                np.testing.assert_array_equal(out.triangles, ref.triangles)
                self.assertEqual(out.report, ref.report)
                with self.assertRaisesRegex(RuntimeError, 'above limit'):
                    crossing1.repair_interface_crossings(flaps(), max_displacement=.001, **tier)

    def test_quality_flips_match_d3v2p0(self):
        for seed in (0, 1):
            surface = surface_of(seed)
            kw = dict(minimum_quality=.5, max_passes=10, minimum_facet_angle=15.)
            ref = quality0.improve_surface_quality(surface, **kw)
            self.assertGreater(ref.report['surface_quality_repair']['accepted_flips'], 0)
            for tier, context in tiers_including_numba_off():
                with self.subTest(seed=seed, **tier), context:
                    out = quality1.improve_surface_quality(surface, **kw, **tier)
                    np.testing.assert_array_equal(out.triangles, ref.triangles)
                    np.testing.assert_array_equal(out.points, ref.points)
                    self.assertEqual(out.report, ref.report)

    def test_quality_disabled_and_rollback(self):
        source = surface_of(0)
        np.testing.assert_array_equal(quality1.improve_surface_quality(source, enabled=False).triangles,
                                      source.triangles)
        points = np.array([[0., 0, 0], [2., 0, 0], [1., .001, 0], [0., -1, 0]])
        sliver = ClosedRVE(points, np.array([[0, 1, 2], [1, 0, 3]]), np.array([[1, 2], [1, 2]]),
                           np.zeros(2, bool), np.full(2, -1), {})
        with mock.patch('upxo.meshing.gbconformant.d3v2p1.surface_quality.find_surface_intersections',
                        return_value=np.array([[0, 1]])):
            result = quality1.improve_surface_quality(sliver)
        np.testing.assert_array_equal(result.triangles, sliver.triangles)
        self.assertEqual(result.report['surface_quality_repair']['history'][0]['rejected_geometry'], 1)

    def test_triangle_quality_is_d3v2p0s(self):
        self.assertIs(quality1.triangle_quality, quality0.triangle_quality)

    def test_bad_backend_arguments_fail_before_work(self):
        for call in (lambda **k: crossing1.repair_interface_crossings(flaps(), **k),
                     lambda **k: quality1.improve_surface_quality(surface_of(0), **k)):
            with self.assertRaises(ValueError):
                call(backend='gpu')
            with self.assertRaises(ValueError):
                call(n_workers=-1)


class SearchEdgeCaseTests(unittest.TestCase):
    """Queries the guard stages can issue: none, one, all; same pairs and dtype as d3v2p0."""

    def test_empty_single_and_full_queries_match_d3v2p0(self):
        from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections as ref
        from upxo.meshing.gbconformant.d3v2p1.surface_intersections import find_surface_intersections as fast
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
        t = np.array([[0, 1, 2], [0, 1, 3]])
        for tier in TIERS:
            for ids in (np.array([], int), None, np.array([1])):
                with self.subTest(ids=None if ids is None else ids.tolist(), **tier):
                    a, b = ref(p, t, triangle_ids=ids), fast(p, t, triangle_ids=ids, **tier)
                    self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
                    np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
