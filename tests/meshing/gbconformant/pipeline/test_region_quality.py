"""Layers of elements around the grain boundary surface, junction lines and junction points; their quality."""
import unittest
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use('Agg')

from upxo.meshing.gbconformant.fm_steel_abaqus.junctions import (
    box_face_bits, junction_point_nodes, tet_junction_edges)
from upxo.meshing.gbconformant.pipeline import results
from upxo.meshing.gbconformant.pipeline.mesh_regions import (
    FEATURES, GB_SURFACE, JUNCTION_LINES, JUNCTION_POINTS, feature_layers, mesh_surface, select_layers, tet_faces)
from upxo.meshing.gbconformant.pipeline.region_quality import (
    BEYOND, MODE_ONLY, MODE_UP_TO, WHOLE, RegionSettings, region_figures, region_flat_table, region_quality,
    region_settings, region_tables, summary_numbers, write_region_files)
from upxo.meshing.gbconformant.pipeline.runner import PipelineRunner
from upxo.meshing.gbconformant.pipeline.serialize import read_value, write_value
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config
from upxo.meshing.gbconformant.pipeline.stage_quality import FOLDER, REGION_KEY
from upxo.meshing.gbconformant.pipeline.stages import stage_def

from ..fm_steel_abaqus.helpers import voxel_tets, work_dir
from .test_results import meshed_state

N = 6


def octant_mesh(n=N):
    """Eight grain blocks meeting at the centre: three lines of four grains cross there, one junction point."""
    x, y, z = np.indices((n, n, n))
    half = n // 2
    labels = (1 + (x >= half) + 2 * (y >= half) + 4 * (z >= half)).astype(np.int64)
    return voxel_tets(labels)


def brute_layers(tets, seed, order):
    """Layer of every element by rings of shared nodes, with sets and loops."""
    layer = {int(i): 1 for i in np.flatnonzero(seed)}
    for step in range(1, order):
        nodes = {int(v) for i, k in layer.items() if k == step for v in tets[i]}
        for i in range(len(tets)):
            if i not in layer and nodes.intersection(int(v) for v in tets[i]):
                layer[i] = step + 1
    out = np.zeros(len(tets), dtype=np.int8)
    for i, k in layer.items():
        out[i] = k
    return out


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.points, self.tets, self.grains = octant_mesh()

    def test_every_face_is_found_once(self):
        nodes, first, second, count = tet_faces(self.tets, len(self.points))
        self.assertEqual(len(nodes), len({tuple(r) for r in nodes.tolist()}))
        self.assertEqual(int(count.sum()), 4 * len(self.tets))
        self.assertTrue(set(count.tolist()) <= {1, 2})
        self.assertTrue(np.all(second[count == 1] == -1))

    def test_grain_boundary_triangles_lie_in_the_three_mid_planes(self):
        surface = mesh_surface(self.tets, self.grains, len(self.points))
        self.assertEqual(len(surface.nodes), 3 * 2 * N * N)
        self.assertTrue(surface.between_grains.all())
        mid = N // 2
        flat = [(self.points[surface.nodes][:, :, axis] == mid).all(axis=1) for axis in range(3)]
        self.assertTrue(np.all(np.sum(flat, axis=0) == 1))
        self.assertTrue(np.all(surface.grains[:, 0] < surface.grains[:, 1]))
        self.assertEqual(surface.nonmanifold, 0)

    def test_the_box_exterior_adds_its_triangles_when_asked(self):
        surface = mesh_surface(self.tets, self.grains, len(self.points), include_rve_faces=True)
        self.assertEqual(len(surface.nodes), 3 * 2 * N * N + 6 * 2 * N * N)
        outside = ~surface.between_grains
        self.assertTrue(np.all(surface.grains[outside, 0] == 0))
        self.assertTrue(np.all(surface.tets[outside, 1] == -1))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.points, self.tets, self.grains = octant_mesh()
        self.mid = N // 2
        c = self.points
        on_line = ((c[:, 1] == self.mid) & (c[:, 2] == self.mid)) | ((c[:, 0] == self.mid) & (c[:, 2] == self.mid)) \
            | ((c[:, 0] == self.mid) & (c[:, 1] == self.mid))
        self.line_nodes = on_line
        self.centre = int(np.flatnonzero((c == self.mid).all(axis=1))[0])

    def test_counts_of_the_features(self):
        found = feature_layers(self.points, self.tets, self.grains)
        self.assertEqual(found.counts['junction_edges'], 3 * N)
        self.assertEqual(found.counts['junction_lines'], 6)           # each axis is cut in two at the centre
        self.assertEqual(found.counts['junction_points'], 1)
        self.assertEqual(found.counts['grain_boundary_triangles'], 3 * 2 * N * N)
        self.assertTrue(found.point_nodes[self.centre])
        self.assertTrue(np.array_equal(found.line_nodes, self.line_nodes))

    def test_junction_points_include_the_box_face_ends_with_the_exterior(self):
        inside = feature_layers(self.points, self.tets, self.grains)
        outside = feature_layers(self.points, self.tets, self.grains, include_rve_faces=True)
        self.assertEqual(outside.counts['junction_points'], 1 + 6)    # the centre and the six face centres
        self.assertGreater(outside.counts['junction_edges'], inside.counts['junction_edges'])

    def test_a_line_that_continues_unchanged_has_no_point_on_it(self):
        edges = tet_junction_edges(self.points, self.tets, self.grains)
        on_box = box_face_bits(self.points) != 0
        every_end = junction_point_nodes(edges, len(self.points))
        self.assertEqual(int(every_end.sum()), 1 + 6)                    # the centre and the six ends on the box
        self.assertEqual(int(np.sum(every_end & on_box)), 6)
        ignoring = junction_point_nodes(edges, len(self.points), ignore_ends=on_box)
        self.assertEqual(int(ignoring.sum()), 1)                          # nodes along the lines are never points
        self.assertTrue(ignoring[self.centre])

    def test_layers_of_the_junction_points_match_a_brute_force_search(self):
        found = feature_layers(self.points, self.tets, self.grains, order=3)
        seed = (self.tets == self.centre).any(axis=1)
        self.assertTrue(np.array_equal(found.tet[JUNCTION_POINTS], brute_layers(self.tets, seed, 3)))

    def test_layers_of_the_junction_lines_by_node_match_a_brute_force_search(self):
        found = feature_layers(self.points, self.tets, self.grains, order=3, touch='node')
        seed = self.line_nodes[self.tets].any(axis=1)
        self.assertTrue(np.array_equal(found.tet[JUNCTION_LINES], brute_layers(self.tets, seed, 3)))

    def test_layers_of_the_grain_boundary_surface_by_face_match_a_brute_force_search(self):
        found = feature_layers(self.points, self.tets, self.grains, order=3)
        on_plane = np.zeros(len(self.tets), dtype=bool)
        for axis in range(3):
            planar = (self.points[self.tets][:, :, axis] == self.mid)
            on_plane |= planar.sum(axis=1) >= 3
        self.assertTrue(np.array_equal(found.tet[GB_SURFACE], brute_layers(self.tets, on_plane, 3)))

    def test_touching_by_a_face_or_edge_is_inside_touching_by_a_node(self):
        tight = feature_layers(self.points, self.tets, self.grains, order=1)
        wide = feature_layers(self.points, self.tets, self.grains, order=1, touch='node')
        for feature in (GB_SURFACE, JUNCTION_LINES):
            self.assertTrue(np.all(wide.tet[feature][tight.tet[feature] == 1] == 1), feature)
            self.assertGreater(int((wide.tet[feature] == 1).sum()), int((tight.tet[feature] == 1).sum()), feature)

    def test_a_lower_order_gives_the_same_layers_where_it_reaches(self):
        far = feature_layers(self.points, self.tets, self.grains, order=4)
        near = feature_layers(self.points, self.tets, self.grains, order=2)
        for feature in FEATURES:
            expected = np.where(far.tet[feature] <= 2, far.tet[feature], 0)
            self.assertTrue(np.array_equal(near.tet[feature], expected), feature)
            expected = np.where(far.triangle[feature] <= 2, far.triangle[feature], 0)
            self.assertTrue(np.array_equal(near.triangle[feature], expected), feature)

    def test_triangle_layers_grow_along_the_surface_only(self):
        found = feature_layers(self.points, self.tets, self.grains, order=2, touch='node')
        triangles = found.surface.nodes
        seed = found.line_nodes[triangles].any(axis=1)
        self.assertTrue(np.array_equal(found.triangle[JUNCTION_LINES], brute_layers(triangles, seed, 2)))
        self.assertTrue(np.all(found.triangle[GB_SURFACE] == 1))

    def test_select_layers(self):
        layer = np.array([0, 1, 2, 3, 1])
        self.assertEqual(select_layers(layer, 2, False).tolist(), [False, True, True, False, True])
        self.assertEqual(select_layers(layer, 2, True).tolist(), [False, False, True, False, False])

    def test_bad_arguments(self):
        for kwargs in ({'order': 0}, {'order': 51}, {'order': 1.5}, {'touch': 'edge'}, {'features': ('lines',)}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                feature_layers(self.points, self.tets, self.grains, **kwargs)

    def test_a_mesh_without_boundaries_has_no_features(self):
        points, tets, _ = octant_mesh(2)
        found = feature_layers(points, tets, np.ones(len(tets), dtype=np.int64), order=2)
        self.assertEqual(found.counts['grain_boundary_triangles'], 0)
        for feature in FEATURES:
            self.assertFalse(found.tet[feature].any())

    def test_two_grains_have_a_surface_but_no_lines_or_points(self):
        x = np.indices((4, 4, 4))[0]
        points, tets, grains = voxel_tets((1 + (x >= 2)).astype(np.int64))
        found = feature_layers(points, tets, grains, order=2)
        self.assertGreater(found.counts['grain_boundary_triangles'], 0)
        self.assertEqual(found.counts['junction_lines'], 0)
        self.assertEqual(found.counts['junction_points'], 0)
        self.assertTrue(found.tet[GB_SURFACE].any())
        self.assertFalse(found.tet[JUNCTION_LINES].any())


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.points, self.tets, self.grains = octant_mesh()
        self.quality = np.linspace(0.2, 0.9, len(self.tets))

    def run_quality(self, **kwargs):
        return region_quality(self.points, self.tets, self.grains, self.quality, RegionSettings(**kwargs))

    @staticmethod
    def rows(result, kind='tets', region=None):
        return [r for r in result['rows'] if r['kind'] == kind and (region is None or r['region'] == region)]

    def test_a_layer_row_counts_match_the_layers(self):
        result, _ = self.run_quality(order=3)
        layers = result['layers']['tets'][JUNCTION_LINES]
        for row in self.rows(result, region=JUNCTION_LINES):
            if row['layer'] is not None:
                self.assertEqual(row['count'], int((layers == row['layer']).sum()), row['layers'])
        combined = [r for r in self.rows(result, region=JUNCTION_LINES) if r['layers'] == '1 to 3'][0]
        self.assertEqual(combined['count'], int(((layers >= 1) & (layers <= 3)).sum()))
        self.assertEqual(combined['count'], sum(r['count'] for r in self.rows(result, region=JUNCTION_LINES)
                                                if r['layer'] is not None))

    def test_the_whole_mesh_and_the_rest_are_in_the_table(self):
        result, _ = self.run_quality(order=1)
        whole = self.rows(result, region=WHOLE)[0]
        self.assertEqual(whole['count'], len(self.tets))
        beyond = self.rows(result, region=BEYOND)[0]
        union = np.zeros(len(self.tets), dtype=bool)
        for feature in FEATURES:
            union |= result['layers']['tets'][feature] == 1
        self.assertEqual(beyond['count'], int((~union).sum()))

    def test_the_plotted_row_follows_the_mode(self):
        up_to, _ = self.run_quality(order=2)
        only, _ = self.run_quality(order=2, mode=MODE_ONLY)
        single, _ = self.run_quality(order=1)
        for feature in FEATURES:
            self.assertEqual([r['layers'] for r in self.rows(up_to, region=feature) if r['headline']], ['1 to 2'])
            self.assertEqual([r['layers'] for r in self.rows(only, region=feature) if r['headline']], ['2'])
            self.assertEqual([r['layers'] for r in self.rows(single, region=feature) if r['headline']], ['1'])

    def test_the_selections_agree_with_the_plotted_rows(self):
        for mode in (MODE_UP_TO, MODE_ONLY):
            result, arrays = self.run_quality(order=2, mode=mode)
            for feature in FEATURES:
                row = [r for r in self.rows(result, region=feature) if r['headline']][0]
                self.assertEqual(int(arrays['selected']['tets'][feature].sum()), row['count'])

    def test_the_grain_boundary_surface_triangles_are_one_row(self):
        result, _ = self.run_quality(order=3)
        rows = self.rows(result, 'triangles', GB_SURFACE)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['headline'])
        self.assertEqual(rows[0]['count'], 3 * 2 * N * N)
        only, _ = self.run_quality(order=3, mode=MODE_ONLY)
        self.assertTrue(self.rows(only, 'triangles', GB_SURFACE)[0]['headline'])

    def test_leaving_out_the_elements_near_lines_and_points(self):
        plain, _ = self.run_quality(order=2)
        no_lines, arrays = self.run_quality(order=2, exclude_lines=True)
        no_points, _ = self.run_quality(order=2, exclude_points=True)
        both, _ = self.run_quality(order=2, exclude_lines=True, exclude_points=True)
        wide, _ = self.run_quality(order=2, exclude_lines=True, exclude_points=True, exclude_order=2)

        def count(result, kind='tets'):
            return [r for r in self.rows(result, kind, GB_SURFACE) if r['headline']][0]['count']

        self.assertGreater(count(plain), count(no_points))
        self.assertGreater(count(no_points), count(both))
        self.assertGreater(count(plain), count(no_lines))
        self.assertGreater(count(both), count(wide))
        self.assertGreater(count(plain, 'triangles'), count(no_lines, 'triangles'))
        lines = no_lines['layers']['tets'][JUNCTION_LINES]
        self.assertFalse((arrays['selected']['tets'][GB_SURFACE] & (lines == 1)).any())
        self.assertEqual(no_lines['counts']['excluded_tetrahedra'], int((lines == 1).sum()))
        self.assertEqual(plain['counts']['excluded_tetrahedra'], 0)
        for feature in (JUNCTION_LINES, JUNCTION_POINTS):                # only the surface loses elements
            self.assertEqual(self.rows(plain, region=feature)[0]['count'], self.rows(both, region=feature)[0]['count'])

    def test_exclusion_works_when_the_excluded_feature_is_not_reported(self):
        result, arrays = self.run_quality(order=1, features=(GB_SURFACE,), exclude_lines=True)
        self.assertEqual([r['region'] for r in result['rows'] if r['kind'] == 'tets'], [WHOLE, GB_SURFACE, BEYOND])
        self.assertGreater(result['counts']['excluded_tetrahedra'], 0)

    def test_the_statistics_describe_the_selected_elements(self):
        result, arrays = self.run_quality(order=1)
        row = self.rows(result, region=JUNCTION_POINTS)[0]
        mask = result['layers']['tets'][JUNCTION_POINTS] == 1
        smallest = arrays['values']['tets']['smallest_dihedral'][mask]
        self.assertAlmostEqual(row['stats']['smallest_dihedral']['min'], float(smallest.min()))
        self.assertAlmostEqual(row['stats']['sicn']['mean'], float(self.quality[mask].mean()))
        self.assertEqual(sum(row['hist']['smallest_dihedral']), row['count'])
        self.assertEqual(row['below']['target_dihedral'], int((smallest < 30.0).sum()))

    def test_limits_come_from_the_settings(self):
        result, _ = self.run_quality(order=1, target_dihedral=50.0, min_angle=50.0)
        whole = self.rows(result, region=WHOLE)[0]
        self.assertEqual(result['limits']['target_dihedral'], 50.0)
        self.assertGreater(whole['below']['target_dihedral'], 0)
        self.assertGreater(self.rows(result, 'triangles', WHOLE)[0]['below']['min_angle'], 0)

    def test_without_a_quality_array_the_sicn_is_left_out(self):
        result, _ = region_quality(self.points, self.tets, self.grains, None, RegionSettings(order=1))
        self.assertNotIn('sicn', self.rows(result, region=WHOLE)[0]['stats'])
        self.assertTrue(region_tables(result))

    def test_tables_and_the_flat_table_are_rectangular(self):
        result, _ = self.run_quality(order=2, exclude_lines=True)
        tables = region_tables(result)
        self.assertEqual(len(tables), 2)
        for title, columns, rows in tables:
            self.assertTrue(rows)
            self.assertTrue(all(len(r) == len(columns) for r in rows))
        self.assertIn('without the elements within 1 layer(s) of the junction lines', tables[0][0])
        _, columns, rows = region_flat_table(result)
        self.assertEqual(len(rows), len(result['rows']))
        self.assertTrue(all(len(r) == len(columns) for r in rows))
        self.assertIn('smallest_angle_median', columns)
        self.assertIn('smallest_dihedral_median', columns)

    def test_figures_are_drawn_for_every_measure_and_for_the_layers(self):
        result, _ = self.run_quality(order=3)
        keys = [key for key, _, _ in region_figures(result)]
        for expected in ('regions.tets.smallest_dihedral', 'regions.tets.sicn', 'regions.tets.layers',
                         'regions.triangles.smallest_angle', 'regions.triangles.layers'):
            self.assertIn(expected, keys)
        one, _ = self.run_quality(order=1)
        self.assertNotIn('regions.tets.layers', [k for k, _, _ in region_figures(one)])

    def test_the_summary_numbers(self):
        result, _ = self.run_quality(order=2)
        numbers = summary_numbers(result)
        self.assertEqual(numbers['junction_points'], 1)
        self.assertEqual(numbers['order'], 2)
        self.assertEqual(set(numbers['lowest_dihedral']), set(FEATURES))

    def test_the_result_survives_the_checkpoint_format(self):
        result, _ = self.run_quality(order=2, exclude_points=True)
        with work_dir() as root:
            write_value(root / 'region', result)
            back = read_value(root / 'region')
        self.assertEqual(back['rows'], result['rows'])
        self.assertEqual(back['counts'], result['counts'])
        for kind in ('tets', 'triangles'):
            for feature in result['layers'][kind]:
                self.assertTrue(np.array_equal(back['layers'][kind][feature], result['layers'][kind][feature]))
        self.assertEqual(region_tables(back), region_tables(result))

    def test_the_files(self):
        import pyvista as pv

        result, arrays = self.run_quality(order=2, features=(GB_SURFACE, JUNCTION_LINES))
        with work_dir() as root:
            written = write_region_files(root / 'out', self.points, self.tets, self.grains, result, arrays)
            self.assertEqual([p.rsplit('.', 1)[1] for p in map(str, written)], ['csv', 'vtu', 'vtp'])
            text = (root / 'out' / 'region_quality.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(text), 1 + len(result['rows']))
            grid = pv.read(str(root / 'out' / 'region_tets.vtu'))
            self.assertEqual(grid.n_cells, len(self.tets))
            for name in ('grain', 'smallest_dihedral', 'layer_gb_surface', 'selected_junction_lines', 'aspect'):
                self.assertIn(name, grid.cell_data)
            self.assertNotIn('layer_junction_points', grid.cell_data)
            self.assertTrue(np.array_equal(grid.cell_data['layer_junction_lines'],
                                           result['layers']['tets'][JUNCTION_LINES]))
            surface = pv.read(str(root / 'out' / 'region_triangles.vtp'))
            self.assertEqual(surface.n_cells, len(result['triangles']['nodes']))
            for name in ('grain_low', 'grain_high', 'smallest_angle', 'layer_junction_lines'):
                self.assertIn(name, surface.cell_data)


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        found = region_settings(default_config())
        self.assertEqual(found, RegionSettings())
        self.assertEqual(found.features, FEATURES)
        self.assertEqual(found.mode, MODE_UP_TO)

    def test_settings_are_read_from_the_configuration(self):
        found = region_settings(dict(REGION_ORDER=2, REGION_ORDER_MODE='only layer N', REGION_TOUCH='node',
                                     REGION_JUNCTION_POINTS=False, REGION_EXCLUDE_LINES=True, REGION_EXCLUDE_ORDER=2,
                                     MIN_TET_DIHEDRAL_DEGREES=12.0))
        self.assertEqual((found.order, found.mode, found.touch), (2, MODE_ONLY, 'node'))
        self.assertEqual(found.features, (GB_SURFACE, JUNCTION_LINES))
        self.assertTrue(found.exclude_lines)
        self.assertEqual((found.exclude_order, found.min_dihedral), (2, 12.0))

    def test_a_configuration_without_a_feature_or_with_a_bad_choice_is_refused(self):
        off = dict(REGION_GB_SURFACE=False, REGION_JUNCTION_LINES=False, REGION_JUNCTION_POINTS=False)
        with self.assertRaises(ValueError):
            region_settings(off)
        with self.assertRaises(ValueError):
            region_settings(dict(REGION_ORDER_MODE='all'))
        with self.assertRaises(ValueError):
            region_settings(dict(REGION_TOUCH='corner'))

    def test_the_settings_are_in_the_table_for_the_stage(self):
        keys = [k for k, s in SCHEMA.items() if 'region_quality' in s.stages]
        self.assertEqual(len(keys), 11)
        self.assertTrue(all(k.startswith('REGION_') for k in keys))
        self.assertEqual(SCHEMA['REGION_ORDER'].minimum, 1)
        self.assertEqual(SCHEMA['REGION_ORDER'].maximum, 50)

    def test_the_stage_sits_before_the_comparison_and_the_exports(self):
        from upxo.meshing.gbconformant.pipeline.stages import stage_ids
        order = stage_ids()
        self.assertEqual(stage_def('region_quality').optional_flag, 'STAGE_REGION_QUALITY')
        self.assertLess(order.index('insert_nodes'), order.index('region_quality'))
        self.assertLess(order.index('region_quality'), order.index('compare_before_after'))
        self.assertLess(order.index('region_quality'), order.index('export_surfaces'))


class StageTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        import os
        from unittest import mock
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state = meshed_state(self.root, 'regions')

    def test_the_stage_runs_and_reports(self):
        self.state.config.update(REGION_ORDER=2)
        PipelineRunner(self.state).run_stage('region_quality')
        report = self.state.reports['region_quality']
        self.assertEqual(report['order'], 2)
        self.assertIn('junction lines', report['text'])
        self.assertIn('Tetrahedra by region', report['text'])
        result = self.state.get_artefact(REGION_KEY)
        self.assertEqual(result['counts']['tetrahedra'], len(self.state.get_artefact('verified_tets').tetrahedra))
        files = [p.name for p in (results.ensure_run_dir(self.state.config) / FOLDER).iterdir()]
        self.assertEqual(sorted(files), ['region_quality.csv', 'region_tets.vtu', 'region_triangles.vtp'])

    def test_files_can_be_switched_off(self):
        self.state.config.update(REGION_WRITE_FILES=False)
        PipelineRunner(self.state).run_stage('region_quality')
        self.assertEqual(self.state.reports['region_quality']['files'], [])
        self.assertFalse((results.ensure_run_dir(self.state.config) / FOLDER).exists())

    def test_it_needs_the_tetrahedra(self):
        self.state.artefacts.pop('verified_tets')
        with self.assertRaises(Exception):
            PipelineRunner(self.state).run_stage('region_quality')

    def test_the_results_show_it(self):
        PipelineRunner(self.state).run_stage('region_quality')
        blocks = results.region_blocks(self.state)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([b.title.split(' by region')[0] for b in blocks], ['Tetrahedra', 'Surface triangles'])
        titles = [q.title for q in results.quality_blocks(self.state)]
        for block in blocks:
            self.assertIn(block.title, titles)
        keys = [f.key for f in results.quality_figures(self.state)]
        self.assertIn('regions.tets.smallest_dihedral', keys)
        self.assertIn('junction', results.stage_note(self.state, 'region_quality'))
        self.assertIsNotNone(results.region_flat_block(self.state))
        labels = [d['label'] for d in results.deliverables(self.state) if d['exists']]
        self.assertIn('Quality by region (CSV)', labels)

    def test_before_it_has_run_the_results_have_nothing(self):
        self.assertEqual(results.region_blocks(self.state), [])
        self.assertIsNone(results.region_flat_block(self.state))
        self.assertEqual(results.region_result_figures(self.state), [])


if __name__ == '__main__':
    unittest.main()
