import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    ComparisonSettings, TextureSettings, compare_before_after, density_difference,
    feature_diameters, format_summary, load_hierarchy, orientation_table, report_summary,
    resolve_identities)
from upxo.meshing.gbconformant.fm_steel_abaqus.comparison import describe, sphere_equivalent_diameter
from upxo.meshing.gbconformant.fm_steel_abaqus.junctions import (
    JunctionEdges, SET_WIDTH, junction_lines, regions_of, tet_junction_edges,
    totals_per_grain, totals_per_set, voxel_junction_edges)
from upxo.meshing.gbconformant.fm_steel_abaqus.plots import (
    distribution_figure, matched_change_figure, texture_figure)
from .helpers import work_dir, voxel_tets, write_raw_folder


def triple_line_labels():
    labels = np.empty((2, 2, 3), dtype=np.int32)
    labels[0, 0], labels[0, 1], labels[1, 0], labels[1, 1] = 1, 2, 3, 3
    return labels


def random_labels(seed=0, shape=(7, 6, 5), n=5):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(1, n + 1, size=(4, 3, 3))
    return np.kron(coarse, np.ones((2, 2, 2), dtype=int))[:shape[0], :shape[1], :shape[2]].astype(np.int32)


class JunctionTests(unittest.TestCase):
    def test_triple_line_in_voxels(self):
        edges = voxel_junction_edges(triple_line_labels())
        self.assertEqual(len(edges.nodes), 3)
        lines = junction_lines(edges)
        self.assertEqual(len(lines.length), 1)
        self.assertEqual(lines.length[0], 3.0)
        self.assertEqual(regions_of(lines.sets[0]), (1, 2, 3))
        self.assertEqual(totals_per_grain(lines), {1: 3.0, 2: 3.0, 3: 3.0})
        self.assertEqual(totals_per_set(lines), {(1, 2, 3): 3.0})

    def test_rve_faces_are_optional_and_add_exterior_traces(self):
        labels = triple_line_labels()
        inside = voxel_junction_edges(labels)
        with_faces = voxel_junction_edges(labels, include_rve_faces=True)
        self.assertGreater(len(with_faces.nodes), len(inside.nodes))
        sets = {tuple(int(v) for v in row if v >= 0) for row in with_faces.sets}
        self.assertIn((1, 2, 3), sets)
        self.assertTrue(any(0 in s for s in sets))

    def test_tets_and_voxels_agree_on_a_structured_mesh(self):
        for seed in range(3):
            labels = random_labels(seed)
            pts, tets, grains = voxel_tets(labels)
            for rve in (False, True):
                v = totals_per_set(junction_lines(voxel_junction_edges(labels, rve)))
                t = totals_per_set(junction_lines(tet_junction_edges(pts, tets, grains, rve)))
                self.assertEqual(v, t, (seed, rve))

    def test_tet_lengths_follow_the_coordinates(self):
        labels = triple_line_labels()
        pts, tets, grains = voxel_tets(labels, spacing=0.5)
        lines = junction_lines(tet_junction_edges(pts, tets, grains))
        self.assertAlmostEqual(float(lines.length[0]), 1.5)

    def test_lines_split_by_region_set_and_by_connectivity(self):
        sets = np.full((4, SET_WIDTH), -1, dtype=np.int64)
        sets[:, -3:] = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 4]]
        nodes = np.array([[0, 1], [1, 2], [10, 11], [2, 3]])
        lines = junction_lines(JunctionEdges(nodes, sets, np.array([1.0, 1.0, 2.0, 5.0])))
        self.assertEqual(sorted(lines.length.tolist()), [2.0, 2.0, 5.0])
        self.assertEqual(sorted(lines.n_edges.tolist()), [1, 1, 2])

    def test_empty_inputs(self):
        flat = np.ones((3, 3, 3), dtype=np.int32)
        edges = voxel_junction_edges(flat)
        self.assertEqual(len(edges.nodes), 0)
        lines = junction_lines(edges)
        self.assertEqual(len(lines.length), 0)
        self.assertEqual(totals_per_grain(lines), {})
        with self.assertRaises(ValueError):
            voxel_junction_edges(-flat)

    def test_describe(self):
        self.assertEqual(describe([1, 2, 3, 4])['median'], 2.5)
        self.assertEqual(describe([])['count'], 0)
        self.assertTrue(np.isnan(describe([])['mean']))


def folder_pieces(root, merge=None, level='blk'):
    (root / 'raw').mkdir()
    write_raw_folder(root / 'raw')
    hier = load_hierarchy(root / 'raw', level)
    meshed = hier.lgi.copy()
    survivor = None
    if merge:
        survivor = {label: label for label in hier.features}
        for absorbed, target in merge.items():
            survivor[absorbed] = target
            meshed[hier.lgi == absorbed] = target
    ident = resolve_identities(hier.voxel_counts, survivor_map=survivor)
    pts, tets, grains = voxel_tets(meshed)
    return hier, ident, pts, tets, grains


class ComparisonTests(unittest.TestCase):
    def test_unchanged_mesh_reports_no_change(self):
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root)
            result = compare_before_after(hier, pts, tets, grains, ident,
                                          ComparisonSettings(output_units='microns'))
        lines = result['junction_lines']
        self.assertEqual(lines['units'], 'microns')
        self.assertEqual(lines['before']['lines']['total'], lines['after']['lines']['total'])
        self.assertEqual(lines['by_region_set']['only_before'], {})
        self.assertEqual(lines['by_region_set']['only_after'], {})
        self.assertTrue(all(abs(r['ratio'] - 1) < 1e-12 for r in lines['by_region_set']['matched']))
        diam = result['diameters']
        self.assertEqual(diam['before'].keys(), diam['after'].keys())
        for row in diam['matched']:
            self.assertAlmostEqual(row['ratio'], 1.0)
        self.assertEqual(diam['lost'], [])
        self.assertEqual((result['features_before'], result['features_after']), (5, 5))
        # one voxel is 0.001 mm = 1 micron; label 4 has 24 voxels
        self.assertAlmostEqual(diam['before'][4], float(sphere_equivalent_diameter(24.0)))

    def test_merge_shows_up_as_change(self):
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root, merge={4: 3})
            result = compare_before_after(hier, pts, tets, grains, ident)
        diam = result['diameters']
        self.assertEqual(diam['lost'], [3])
        self.assertEqual((result['features_before'], result['features_after']), (5, 4))
        self.assertGreater(diam['after'][4], diam['before'][4])
        self.assertAlmostEqual(diam['after'][4], float(sphere_equivalent_diameter(40 * 0.001 ** 3)))
        lines = result['junction_lines']
        self.assertEqual(lines['units'], 'mm')
        self.assertTrue(lines['by_region_set']['only_before'] or lines['by_region_set']['only_after']
                        or lines['before']['lines']['total'] != lines['after']['lines']['total'])

    def test_mesh_units_are_scaled_to_output_units(self):
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root)
            base = compare_before_after(hier, pts, tets, grains, ident, ComparisonSettings())
            half = compare_before_after(hier, pts * 0.5, tets, grains, ident,
                                        ComparisonSettings(mesh_voxel_length=0.5))
        self.assertAlmostEqual(base['junction_lines']['after']['lines']['total'],
                               half['junction_lines']['after']['lines']['total'])
        self.assertAlmostEqual(base['diameters']['after'][1], half['diameters']['after'][1])

    def test_summary_is_json_ready(self):
        import json
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root, merge={4: 3})
            summary = report_summary(compare_before_after(hier, pts, tets, grains, ident))
        text = json.dumps(summary)
        self.assertIn('diameter_ratio_matched', text)
        self.assertEqual(summary['features_lost'], [3])

    def test_text_summary_and_label_lists(self):
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root, merge={4: 3})
            result = compare_before_after(hier, pts, tets, grains, ident)
        self.assertEqual(result['labels_before'], [1, 2, 3, 4, 5])
        self.assertEqual(result['labels_after'], [1, 2, 4, 5])
        text = format_summary(result)
        self.assertIn('Sphere-equivalent diameter (mm)', text)
        self.assertIn('Grains with no mesh: [3]', text)
        self.assertIn('RVE-face traces: excluded', text)

    def test_feature_diameters_direct(self):
        with work_dir() as root:
            hier, ident, pts, tets, grains = folder_pieces(root)
            out = feature_diameters(hier, pts, tets, grains, ComparisonSettings())
        self.assertEqual(out['ratio_stats']['count'], 5)


class TextureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.before = rng.uniform(0, 360, (40, 3))
        self.before[:, 1] = rng.uniform(0, 90, 40)

    def test_orientation_table(self):
        with work_dir() as root:
            (root / 'raw').mkdir()
            _, orientations, _ = write_raw_folder(root / 'raw')
            hier = load_hierarchy(root / 'raw', 'blk')
        labels, eulers = orientation_table(hier)
        self.assertEqual(labels.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(eulers[2].tolist(), list(orientations['blk']['B_1_2_1']))
        labels, eulers = orientation_table(hier, [4, 2])
        self.assertEqual(labels.tolist(), [2, 4])
        with self.assertRaises(ValueError):
            orientation_table(hier, [99])

    def test_identical_populations_have_zero_difference(self):
        diff = density_difference(self.before, self.before, TextureSettings(half_width_deg=10))
        self.assertAlmostEqual(float(np.nanmax(np.abs(diff['zi_diff']))), 0.0)
        self.assertEqual(diff['iqr'], 0.0)

    def test_difference_is_after_minus_before(self):
        extra = np.tile([[10.0, 20.0, 30.0]], (30, 1))
        after = np.vstack((self.before, extra))
        settings = TextureSettings(half_width_deg=10, unit_normalize=False)
        forward = density_difference(self.before, after, settings)
        backward = density_difference(after, self.before, settings)
        self.assertGreater(float(np.nanmax(forward['zi_diff'])), 0.5)
        np.testing.assert_allclose(forward['zi_diff'], -backward['zi_diff'], atol=1e-9)

    def test_unit_normalisation_changes_the_scale_not_the_shape_of_equal_fields(self):
        twice = np.vstack((self.before, self.before))
        raw = density_difference(self.before, twice, TextureSettings(half_width_deg=10, unit_normalize=False))
        norm = density_difference(self.before, twice, TextureSettings(half_width_deg=10, unit_normalize=True))
        self.assertAlmostEqual(float(np.nanmax(np.abs(norm['zi_diff']))), 0.0, places=6)
        self.assertLessEqual(float(np.nanmax(np.abs(raw['zi_diff']))), 1.0)

    def test_pole_families_symmetries_and_validation(self):
        for family in ('100', '110', '111', [1, 2, 3]):
            diff = density_difference(self.before, self.before[:20],
                                      TextureSettings(pole_family=family, half_width_deg=10))
            self.assertEqual(diff['zi_diff'].shape, diff['Xi'].shape)
        diff = density_difference(self.before, self.before[:20], TextureSettings(
            crystal_symmetry='hexagonal', apply_sample_symmetry=False, half_width_deg=10))
        self.assertTrue(np.isfinite(diff['iqr']))
        with self.assertRaises(ValueError):
            density_difference(self.before, self.before, TextureSettings(crystal_symmetry='x'))
        with self.assertRaises(ValueError):
            density_difference(self.before, np.zeros((0, 3)), TextureSettings())
        with self.assertRaises(ValueError):
            TextureSettings(plot_mode='surface').validate()


class PlotTests(unittest.TestCase):
    def test_distribution_kinds(self):
        rng = np.random.default_rng(1)
        a, b = rng.gamma(3, 2, 200), rng.gamma(3, 2.2, 150)
        for kind in ('histogram', 'cdf', 'box', 'violin'):
            fig = distribution_figure(a, b, kind, bins=20, log=kind == 'cdf', xlabel='length (mm)')
            self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(distribution_figure(a, [], 'histogram').axes), 1)
        with self.assertRaises(ValueError):
            distribution_figure(a, b, 'pie')

    def test_matched_change_figures(self):
        rows = [{'before': 1.0, 'after': 1.2}, {'before': 2.0, 'after': 1.8}]
        for kind in ('scatter', 'ratio'):
            self.assertEqual(len(matched_change_figure(rows, quantity='length', units='mm', kind=kind).axes), 1)
        self.assertEqual(len(matched_change_figure([]).axes), 1)

    def test_texture_figures(self):
        rng = np.random.default_rng(2)
        before = rng.uniform(0, 360, (30, 3))
        after = before[:25]
        fig, diff = texture_figure(before, after, TextureSettings(half_width_deg=10))
        self.assertIsNotNone(diff)
        self.assertGreaterEqual(len(fig.axes), 3)
        fig, diff = texture_figure(before, after, TextureSettings(plot_mode='scatter'))
        self.assertIsNone(diff)
        fig, diff = texture_figure(before, after, TextureSettings(plot_mode='hybrid', half_width_deg=10))
        self.assertIsNotNone(diff)


if __name__ == '__main__':
    unittest.main()
