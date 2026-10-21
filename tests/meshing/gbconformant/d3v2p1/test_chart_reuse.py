"""Chart reuse in the Gmsh retry passes: the default is unchanged; with retry_failed_only a retry pass meshes only
the charts that failed or changed and the result passes the same checks."""
import unittest
from unittest import mock

import numpy as np

try:
    import gmsh
except ImportError:
    gmsh = None

from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p1.chart_reuse import ChartCache, chart_key, warned_surfaces
from upxo.meshing.gbconformant.d3v2p1.discrete_topology import add_chart_topology
from upxo.meshing.gbconformant.d3v2p1.gmsh_closed import remesh_closed_rve_gmsh
from upxo.meshing.gbconformant.d3v2p1.gmsh_interfaces import remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p1 import surface_charts, surface_intersections
from .test_validation_parallel import closed_block_rve

REAL_IS_DISK = surface_charts.is_disk_chart
REAL_SEARCH = surface_intersections.find_surface_intersections


class KeyAndMessageTests(unittest.TestCase):
    def test_chart_key_is_order_free_and_set_specific(self):
        self.assertEqual(chart_key([3, 1, 2]), chart_key(np.array([1, 2, 3])))
        self.assertNotEqual(chart_key([1, 2, 3]), chart_key([1, 2, 4]))
        self.assertNotEqual(chart_key([1, 2, 3]), chart_key([1, 2]))

    def test_warned_surfaces_reads_warnings_and_errors_only(self):
        messages = ['Info: Meshing surface 4 (Discrete surface)', 'Warning: 3 elements remain invalid in surface 7',
                    'error: no elements in surface 9', 'Warning: something else']
        self.assertEqual(warned_surfaces(messages), {7, 9})
        self.assertEqual(warned_surfaces([]), set())


def grid(n):
    xy = np.array([(i, j) for j in range(n) for i in range(n)], float)
    points = np.column_stack((xy, np.zeros(len(xy))))
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            faces += [[a, b, c], [a, c, d]]
    return points, np.array(faces)


@unittest.skipIf(gmsh is None, 'Gmsh is not installed')
class CacheRoundTripTests(unittest.TestCase):
    """A chart's mesh put back into a rebuilt model after generate(2) is the mesh it had."""

    def setUp(self):
        self.points, self.faces = grid(9)
        centre = self.points[self.faces].mean(axis=1)[:, 0]
        self.groups = [np.flatnonzero(centre < 4), np.flatnonzero(centre >= 4)]
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)

    def tearDown(self):
        gmsh.finalize()

    def build(self):
        gmsh.model.add('m' + str(len(gmsh.model.list())))
        for key, value in {'Mesh.MeshOnlyEmpty': 1, 'Mesh.MeshOnlyVisible': 0, 'Mesh.MeshSizeMin': .6,
                           'Mesh.MeshSizeMax': .6, 'Mesh.Algorithm': 6, 'Mesh.Renumber': 0}.items():
            gmsh.option.setNumber(key, value)
        add_chart_topology(gmsh, self.points, self.faces, self.groups)
        gmsh.model.mesh.createGeometry(gmsh.model.getEntities(1))
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.createGeometry([(2, tag)])
        charts = {tag: np.asarray(gmsh.model.mesh.getElementsByType(2, tag)[0], dtype=int) - 1
                  for _, tag in gmsh.model.getEntities(2)}
        gmsh.model.mesh.clear(gmsh.model.getEntities(2))
        return charts

    def mesh_of(self, tag):
        _, conn = gmsh.model.mesh.getElementsByType(2, tag)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        position = {int(t): c for t, c in zip(node_tags, np.asarray(coords).reshape(-1, 3))}
        triangles = np.asarray(conn, dtype=int).reshape(-1, 3)
        return np.sort(np.round(np.array([[position[int(n)] for n in t] for t in triangles]), 9).reshape(len(triangles), 9), axis=0)

    def test_put_back_mesh_equals_the_kept_mesh(self):
        charts = self.build()
        gmsh.model.mesh.generate(2)
        first = {tag: self.mesh_of(tag) for tag in charts}
        cache = ChartCache()
        self.assertEqual(cache.harvest(gmsh, charts), 2)
        self.assertEqual(cache.harvest(gmsh, charts), 0)                     # already kept
        self.assertEqual(len(cache), 2)
        charts = self.build()
        self.assertEqual(cache.tags_with_meshes(charts), {1, 2})
        gmsh.model.setVisibility([(2, 1)], 0)
        gmsh.option.setNumber('Mesh.MeshOnlyVisible', 1)
        gmsh.model.mesh.generate(2)                                          # meshes chart 2 only
        self.assertEqual(len(gmsh.model.mesh.getElementsByType(2, 1)[0]), 0)
        self.assertGreater(len(gmsh.model.mesh.getElementsByType(2, 2)[0]), 0)
        largest = int(gmsh.model.mesh.getMaxNodeTag())
        connections, new_tags, new_coords = cache.meshes(charts, {1}, largest + 1)
        self.assertEqual(set(connections), {1})
        self.assertEqual(new_tags.tolist(), list(range(largest + 1, largest + 1 + len(new_tags))))
        self.assertEqual(new_coords.shape, (len(new_tags), 3))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        position = {int(t): c for t, c in zip(node_tags, np.asarray(coords).reshape(-1, 3))}
        position.update({int(t): c for t, c in zip(new_tags, new_coords)})
        triangles = np.array([[position[int(n)] for n in t] for t in connections[1]])
        np.testing.assert_array_equal(np.sort(np.round(triangles, 9).reshape(len(triangles), 9), axis=0), first[1])

    def test_excluded_and_unmeshed_charts_are_not_kept(self):
        charts = self.build()
        gmsh.model.setVisibility([(2, 2)], 0)
        gmsh.option.setNumber('Mesh.MeshOnlyVisible', 1)
        gmsh.model.mesh.generate(2)
        cache = ChartCache()
        self.assertEqual(cache.harvest(gmsh, charts), 1)                     # chart 2 has no mesh
        self.assertEqual(cache.tags_with_meshes(charts), {1})
        cache = ChartCache()
        self.assertEqual(cache.harvest(gmsh, charts, exclude={1}), 0)


def labels_three_grains(n=5):
    labels = np.ones((n, n, n), dtype=int)
    labels[2:] = 2
    labels[:2, 2:] = 3
    return labels


@unittest.skipIf(gmsh is None, 'Gmsh is not installed')
class RetryTests(unittest.TestCase):
    def source(self):
        return smooth_interfaces(labels_three_grains(), iterations=3)

    def one_bad_chart(self):
        """is_disk_chart that rejects the first chart it is asked about, once (a parametrization failure)."""
        calls = []

        def predicate(triangles):
            calls.append(1)
            return False if len(calls) == 1 else REAL_IS_DISK(triangles)
        return predicate

    def one_bad_geometry(self):
        """Intersection search that reports one crossing on its first call, then searches for real."""
        calls = []

        def search(points, triangles, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return np.array([[0, len(triangles) - 1]])
            return REAL_SEARCH(points, triangles, *args, **kwargs)
        return search

    def check_valid(self, result, source):
        np.testing.assert_array_equal(np.unique(np.sort(source.grain_pairs, axis=1), axis=0),
                                      np.unique(result.grain_pairs, axis=0))
        self.assertTrue(result.report['shared_boundaries_verified'])
        self.assertEqual(result.report['maximum_curve_coordinate_error'], 0.)

    def test_default_run_does_not_touch_the_retry_machinery(self):
        source = self.source()
        with mock.patch('upxo.meshing.gbconformant.d3v2p1.chart_reuse.ChartCache') as cache:
            result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True)
        cache.assert_not_called()
        self.assertNotIn('chart_reuse', result.report)
        self.check_valid(result, source)

    def test_flag_is_checked(self):
        for value in ('yes', 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                remesh_interfaces_gmsh(self.source(), .6, retry_failed_only=value)

    def test_without_a_retry_nothing_is_reused(self):
        source = self.source()
        result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, retry_failed_only=True)
        summary = result.report['chart_reuse']
        self.assertEqual(summary['charts_reused_total'], 0)
        self.assertEqual(len(summary['passes']), 1)
        self.check_valid(result, source)

    def test_parametrization_retry_meshes_only_the_split_charts(self):
        source = self.source()
        with mock.patch.object(surface_charts, 'is_disk_chart', self.one_bad_chart()):
            full = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True)
        with mock.patch.object(surface_charts, 'is_disk_chart', self.one_bad_chart()):
            result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True,
                                            retry_failed_only=True)
        self.assertEqual(len(full.report['parametrization_repairs']), 1)
        self.assertEqual(len(result.report['parametrization_repairs']), 1)
        self.check_valid(result, source)
        self.assertEqual(result.report['remaining_intersections'], 0)
        # the failed chart was rejected before any meshing, so the retry is the first pass that meshes anything
        passes = result.report['chart_reuse']['passes']
        self.assertEqual(passes[0]['reused'], 0)
        self.assertEqual(result.report['surface_charts'], full.report['surface_charts'])
        self.assertAlmostEqual(len(result.triangles) / len(full.triangles), 1., delta=.15)

    def one_invalid_chart(self, tag):
        """gmsh generate that meshes normally and, the first time, logs invalid elements in surface ``tag``."""
        real = gmsh.model.mesh.generate
        calls = []

        def generate(dim=-1):
            real(dim)
            calls.append(1)
            if len(calls) == 1:
                gmsh.logger.write(f'3 elements remain invalid in surface {tag}', 'warning')
        return generate

    def test_invalid_elements_retry_keeps_every_other_chart(self):
        source = self.source()
        with mock.patch.object(gmsh.model.mesh, 'generate', self.one_invalid_chart(1)):
            default = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True)
        with mock.patch.object(gmsh.model.mesh, 'generate', self.one_invalid_chart(1)):
            result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True,
                                            retry_failed_only=True)
        for r in (default, result):
            self.assertEqual([x['stage'] for x in r.report['parametrization_repairs']], ['surface_generation'])
            self.check_valid(r, source)
            self.assertEqual(r.report['remaining_intersections'], 0)
        passes = result.report['chart_reuse']['passes']
        self.assertEqual(len(passes), 2)
        self.assertEqual(passes[0]['reused'], 0)
        self.assertEqual(passes[1]['reused'], passes[0]['charts'] - 1)           # only the invalid chart is meshed again
        self.assertGreater(passes[1]['meshed'], 0)
        self.assertAlmostEqual(len(result.triangles) / len(default.triangles), 1., delta=.15)

    def test_geometric_retry_reuses_the_unaffected_charts(self):
        source = self.source()
        with mock.patch.object(surface_intersections, 'find_surface_intersections', self.one_bad_geometry()):
            result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True,
                                            intersection_retries=2, retry_failed_only=True)
        self.assertEqual(len(result.report['geometric_repairs']), 1)
        self.check_valid(result, source)
        self.assertEqual(result.report['remaining_intersections'], 0)
        passes = result.report['chart_reuse']['passes']
        self.assertEqual(len(passes), 2)
        self.assertEqual(passes[0]['reused'], 0)
        self.assertGreater(passes[1]['reused'], 0)                                 # unaffected charts came from the first pass
        self.assertLess(passes[1]['meshed'], passes[1]['charts'])
        self.assertEqual(result.report['chart_reuse']['charts_reused_total'], passes[1]['reused'])

    def test_geometry_is_not_degraded_by_reuse(self):
        source = self.source()
        with mock.patch.object(surface_intersections, 'find_surface_intersections', self.one_bad_geometry()):
            full = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True,
                                          intersection_retries=2)
        with mock.patch.object(surface_intersections, 'find_surface_intersections', self.one_bad_geometry()):
            result = remesh_interfaces_gmsh(source, .6, max_chart_triangles=8, check_intersections=True,
                                            intersection_retries=2, retry_failed_only=True)
        from upxo.meshing.gbconformant.d3v2p0.surface_angles import triangle_angles
        for name, r in (('default', full), ('reuse', result)):
            self.assertEqual(r.report['remaining_intersections'], 0, name)
        a = triangle_angles(full.points, full.triangles).min(axis=1)
        b = triangle_angles(result.points, result.triangles).min(axis=1)
        self.assertAlmostEqual(float(b.mean()), float(a.mean()), delta=1.5)
        self.assertAlmostEqual(len(result.triangles) / len(full.triangles), 1., delta=.15)


@unittest.skipIf(gmsh is None, 'Gmsh is not installed')
class JointRemeshTests(unittest.TestCase):
    def test_flag_reaches_the_joint_remesh(self):
        rve = closed_block_rve()
        kw = dict(mesh_size=.8, check_intersections=True, intersection_retries=2, minimum_facet_angle=15.,
                  minimum_remesh_opening=30.)
        result = remesh_closed_rve_gmsh(rve, retry_failed_only=True, **kw)
        self.assertEqual(result.report['grains'], 64)
        self.assertAlmostEqual(sum(result.report['enclosed_grain_volumes'].values()), 512.)
        self.assertIn('chart_reuse', result.report)
        default = remesh_closed_rve_gmsh(rve, **kw)
        self.assertNotIn('chart_reuse', default.report)


if __name__ == '__main__':
    unittest.main()
