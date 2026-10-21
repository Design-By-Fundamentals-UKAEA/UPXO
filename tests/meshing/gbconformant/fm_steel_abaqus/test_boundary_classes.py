"""Grain boundaries by kind: interface triangles, classes, misorientation, node sets and files."""
import csv
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, resolve_identities, write_deck
from upxo.meshing.gbconformant.fm_steel_abaqus.boundary_classes import (
    class_nodes, class_order, classify, pair_class)
from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import check_deck
from upxo.meshing.gbconformant.fm_steel_abaqus.fidelity import mesh_boundaries
from upxo.meshing.gbconformant.fm_steel_abaqus.interfaces import interface_faces
from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import load_twinned
from upxo.xtalphy.crystal_orientation import cubic_misorientation
from .helpers import voxel_tets, work_dir, write_raw_folder, write_twinned_folder
from .test_deck import build
from .test_outputs import read_nsets

CLASSES = DeckSettings(boundary_sets=True, boundary_classes=True, deck_check=True)


def read_table(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


class InterfaceTests(unittest.TestCase):
    def mesh(self):
        with work_dir() as root:
            hier, _, (pts, tets, grains), res = build(root)
            return hier, pts, tets, grains

    def test_the_pair_areas_equal_those_of_the_fidelity_measure(self):
        hier, pts, tets, grains = self.mesh()
        found = interface_faces(pts, tets, grains)
        pairs, inverse = found.pairs()
        area = np.bincount(inverse, weights=found.area)
        _per_label, reference = mesh_boundaries(pts, tets, grains)
        self.assertEqual({tuple(int(v) for v in p) for p in pairs}, set(reference))
        for pair, value in zip(pairs, area):
            self.assertAlmostEqual(value, reference[tuple(int(v) for v in pair)])

    def test_the_known_interface_areas_of_the_test_structure(self):
        hier, pts, tets, grains = self.mesh()
        found = interface_faces(pts, tets, grains)
        pairs, inverse = found.pairs()
        area = dict(zip(map(tuple, pairs.tolist()), np.bincount(inverse, weights=found.area)))
        self.assertEqual(area, {(1, 2): 8.0, (1, 3): 8.0, (2, 3): 8.0, (3, 4): 8.0, (3, 5): 8.0, (4, 5): 12.0})

    def test_box_faces_are_not_interfaces(self):
        hier, pts, tets, grains = self.mesh()
        one_feature = np.ones_like(grains)
        self.assertEqual(len(interface_faces(pts, tets, one_feature).faces), 0)

    def test_a_label_per_tetrahedron_is_required(self):
        hier, pts, tets, grains = self.mesh()
        with self.assertRaises(ValueError):
            interface_faces(pts, tets, grains[:-1])


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.hier, _, (self.pts, self.tets, self.grains), self.res = build(self.root, settings=CLASSES)
        self.found = interface_faces(self.pts, self.tets, self.grains)

    def test_the_kinds_follow_the_hierarchy(self):
        kinds = {(a, b): pair_class(self.hier, a, b) for a, b in ((1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5))}
        self.assertEqual(kinds, {(1, 2): 'block', (1, 3): 'packet', (2, 3): 'packet',
                                 (3, 4): 'pag', (3, 5): 'pag', (4, 5): 'block'})

    def test_the_kinds_a_level_can_have(self):
        self.assertEqual(class_order(self.hier), ('block', 'packet', 'pag'))

    def test_area_by_kind(self):
        classes = classify(self.hier, self.found)
        by_kind = classes.summary['by_kind']
        self.assertEqual({k: v['area'] for k, v in by_kind.items()}, {'block': 20.0, 'packet': 16.0, 'pag': 16.0})
        self.assertAlmostEqual(by_kind['block']['area_fraction'], 20 / 52)
        self.assertEqual({k: v['pairs'] for k, v in by_kind.items()}, {'block': 2, 'packet': 2, 'pag': 2})
        self.assertEqual(classes.summary['total_area'], 52.0)

    def test_areas_are_in_output_units(self):
        classes = classify(self.hier, self.found, scale=0.5)
        self.assertEqual(classes.summary['total_area'], 52.0 * 0.25)

    def test_the_misorientation_is_the_cubic_angle_of_the_pair(self):
        classes = classify(self.hier, self.found)
        for (a, b), angle in zip(classes.pairs.tolist(), classes.misorientation):
            expected = cubic_misorientation(self.hier.features[a].euler, self.hier.features[b].euler)[0]
            self.assertAlmostEqual(angle, expected)

    def test_low_angle_follows_the_limit(self):
        everything = classify(self.hier, self.found, low_angle_degrees=90.0)
        self.assertTrue(everything.low_angle().all())
        self.assertAlmostEqual(everything.summary['low_angle_area_fraction'], 1.0)
        nothing = classify(self.hier, self.found, low_angle_degrees=0.0)
        self.assertFalse(nothing.low_angle().any())
        self.assertAlmostEqual(nothing.summary['low_angle_area_fraction'], 0.0)

    def test_the_mean_misorientation_is_weighted_by_area(self):
        classes = classify(self.hier, self.found)
        self.assertAlmostEqual(classes.summary['mean_misorientation_deg'],
                               float(np.average(classes.misorientation, weights=classes.area)))

    def test_without_cubic_symmetry_there_is_no_misorientation(self):
        classes = classify(self.hier, self.found, symmetry='hexagonal')
        self.assertFalse(classes.has_misorientation)
        self.assertIsNone(classes.low_angle())
        self.assertNotIn('ns_gb_lowangle', class_nodes(self.found, classes))
        self.assertIsNone(classes.summary['by_kind']['block']['mean_misorientation_deg'])

    def test_node_sets_are_the_nodes_of_the_boundary_planes(self):
        classes = classify(self.hier, self.found)
        sets = class_nodes(self.found, classes)
        x, y, z = self.pts.T
        expected = {'ns_gb_block': ((y == 2) & (x <= 2)) | ((z == 2) & (x >= 3)),
                    'ns_gb_packet': (x == 2) & (y <= 4),
                    'ns_gb_pag': x == 3}
        for name, mask in expected.items():
            self.assertEqual(sorted(sets[name].tolist()), np.flatnonzero(mask).tolist(), name)
        every = sorted(set(sets['ns_gb_lowangle'].tolist()) | set(sets['ns_gb_highangle'].tolist()))
        self.assertEqual(every, sorted(np.unique(self.found.faces).tolist()))

    def test_the_interface_nodes_are_the_nodes_shared_by_two_features(self):
        deck = self.res['directory']
        shared = set(read_nsets(deck / '04b_nsets_boundaries.inp')['ns_gb_all'])
        classes = read_nsets(deck / '04c_nsets_boundary_classes.inp')
        joined = set()
        for name in ('ns_gb_block', 'ns_gb_packet', 'ns_gb_pag'):
            joined |= set(classes[name])
        self.assertEqual(joined, shared)


class DeckFilesTests(unittest.TestCase):
    def test_the_deck_has_the_table_the_node_sets_and_the_surface(self):
        import pyvista as pv
        with work_dir() as root:
            hier, _, (pts, tets, grains), res = build(root, settings=CLASSES)
            deck = res['directory']
            rows = read_table(deck / 'boundary_classes.csv')
            self.assertEqual(len(rows), 6)
            row = next(r for r in rows if (r['label_a'], r['label_b']) == ('4', '5'))
            self.assertEqual((row['kind'], row['elset_a'], row['elset_b']), ('block', 'es_blk_2_1_1', 'es_blk_2_1_2'))
            scale = res['info']['coordinate_scale']
            self.assertAlmostEqual(float(row['area']), 12.0 * scale ** 2, places=15)
            self.assertIn('04c_nsets_boundary_classes.inp', (deck / 'model_master.inp').read_text())
            surface = pv.read(str(deck / 'fm_interfaces.vtp'))
            self.assertEqual(surface.n_cells, int(sum(int(r['triangles']) for r in rows)))
            self.assertAlmostEqual(float(surface.cell_data['area'].sum()), 52.0 * scale ** 2, places=12)
            self.assertEqual(set(surface.cell_data['boundary_kind'].tolist()), {0, 1, 2})
            self.assertTrue(check_deck(deck)['ok'], check_deck(deck)['failed'])
            info = res['info']['boundary_classes']
            self.assertEqual(info['by_kind']['pag']['pairs'], 2)
            self.assertEqual(info['low_angle_degrees'], 15.0)

    def test_the_limit_in_the_settings_decides_the_low_angle_sets(self):
        with work_dir() as root:
            settings = DeckSettings(boundary_classes=True, low_angle_degrees=90.0)
            _, _, _, res = build(root, settings=settings)
            sets = read_nsets(res['directory'] / '04c_nsets_boundary_classes.inp')
            self.assertIn('ns_gb_lowangle', sets)
            self.assertNotIn('ns_gb_highangle', sets)
            self.assertEqual(res['info']['boundary_classes']['low_angle_degrees'], 90.0)

    def test_nothing_is_written_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            names = {p.name for p in res['directory'].iterdir()}
            self.assertFalse(names & {'boundary_classes.csv', 'fm_interfaces.vtp', '04c_nsets_boundary_classes.inp'})
            self.assertIsNone(res['info']['boundary_classes'])

    def test_one_feature_has_no_boundaries(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=CLASSES, merge={2: 1, 3: 1, 4: 1, 5: 1})
            self.assertFalse((res['directory'] / '04c_nsets_boundary_classes.inp').exists())
            self.assertEqual(read_table(res['directory'] / 'boundary_classes.csv'), [])
            self.assertEqual(res['info']['boundary_classes']['total_area'], 0.0)

    def test_a_twinned_structure_has_twin_and_grain_boundaries(self):
        with work_dir() as root:
            folder = root / 'twinned'
            folder.mkdir()
            labels, *_ = write_twinned_folder(folder)
            hier = load_twinned(folder)
            points, tets, grains = voxel_tets(labels)
            res = write_deck(root / 'deck', points, tets, grains, hier, resolve_identities(hier.voxel_counts),
                             CLASSES, meshed_labels=labels)
            summary = res['info']['boundary_classes']['by_kind']
            self.assertEqual(list(summary), ['twin', 'grain'])
            self.assertEqual(summary['twin']['area'] * 1.0, 16.0 * res['info']['coordinate_scale'] ** 2)
            self.assertEqual(summary['twin']['pairs'], 1)
            self.assertGreater(summary['grain']['pairs'], 1)
            sets = read_nsets(res['directory'] / '04c_nsets_boundary_classes.inp')
            self.assertIn('ns_gb_twin', sets)
            self.assertIn('ns_gb_grain', sets)


if __name__ == '__main__':
    unittest.main()
