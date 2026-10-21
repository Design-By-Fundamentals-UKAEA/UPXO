import json
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    BoxSpec, DeckSettings, applied_strain, domain_lengths, cut_statistics, describe_box, extract_labels, extract_subset, load_hierarchy,
    new_indexed_directory, next_index, resolve_box, resolve_identities, subset_record,
    write_deck, write_subset_record)
from upxo.meshing.gbconformant.fm_steel_abaqus.plots import subset_preview_figure
from .helpers import SHAPE, read_elsets, voxel_tets, work_dir, write_raw_folder


class BoxTests(unittest.TestCase):
    def test_index_percent_and_physical_entry(self):
        shape = (50, 75, 100)
        self.assertEqual(resolve_box(BoxSpec('index', (0, 0, 0), (20, 20, 20)), shape),
                         ((0, 0, 0), (20, 20, 20)))
        self.assertEqual(resolve_box(BoxSpec('index'), shape), ((0, 0, 0), shape))
        self.assertEqual(resolve_box(BoxSpec('percent', (0, 0, 0), (40, 20, 10)), shape),
                         ((0, 0, 0), (20, 15, 10)))
        # 0.001 mm voxels: 0.02 mm is 20 voxels, also when entered in microns
        mm = BoxSpec('physical', (0.01, 0.0, 0.0), (0.03, 0.02, 0.02), 'mm')
        self.assertEqual(resolve_box(mm, shape, 0.001, 'mm'), ((10, 0, 0), (30, 20, 20)))
        microns = BoxSpec('physical', (10.0, 0, 0), (30.0, 20.0, 20.0), 'microns')
        self.assertEqual(resolve_box(microns, shape, 0.001, 'mm'), ((10, 0, 0), (30, 20, 20)))
        self.assertEqual(resolve_box(BoxSpec('physical', (0, 0, 0)), shape, 0.001, 'mm'),
                         ((0, 0, 0), shape))

    def test_invalid_boxes(self):
        shape = (10, 10, 10)
        for spec in (BoxSpec('index', (0, 0, 0), (11, 5, 5)), BoxSpec('index', (-1, 0, 0), (5, 5, 5)),
                     BoxSpec('index', (5, 0, 0), (5, 5, 5)), BoxSpec('index', (6, 0, 0), (5, 5, 5)),
                     BoxSpec('percent', (0, 0, 0), (100.0, 100.0, 120.0)),
                     BoxSpec('index', (0, 0), (5, 5, 5)), BoxSpec('volume')):
            with self.assertRaises(ValueError, msg=str(spec)):
                resolve_box(spec, shape)
        with self.assertRaises(ValueError):
            resolve_box(BoxSpec('physical', (0, 0, 0), (1, 1, 1), 'km'), shape, 1.0, 'mm')

    def test_description_round_trips_through_every_mode(self):
        shape, voxel, units = (50, 75, 100), 0.001, 'mm'
        box = ((5, 10, 20), (30, 60, 90))
        info = describe_box(*box, shape, voxel, units, 'microns')
        self.assertEqual(info['shape'], [25, 50, 70])
        self.assertEqual(info['voxels'], 25 * 50 * 70)
        self.assertAlmostEqual(info['size'][0], 25.0)               # microns
        for mode, spec in (('index', BoxSpec('index', info['index']['lo'], info['index']['hi'])),
                           ('percent', BoxSpec('percent', info['percent']['lo'], info['percent']['hi'])),
                           ('physical', BoxSpec('physical', info['physical']['lo'],
                                                info['physical']['hi'], 'microns'))):
            self.assertEqual(resolve_box(spec, shape, voxel, units), box, mode)


class CropTests(unittest.TestCase):
    def setUp(self):
        self.ctx = work_dir()
        self.root = self.ctx.__enter__()
        (self.root / 'raw').mkdir()
        write_raw_folder(self.root / 'raw')
        self.hier = load_hierarchy(self.root / 'raw', 'blk')

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_labels_crop_is_a_contiguous_copy(self):
        crop = extract_labels(self.hier.lgi, (1, 0, 2), (5, 3, 4))
        self.assertEqual(crop.shape, (4, 3, 2))
        self.assertTrue(crop.flags['C_CONTIGUOUS'])
        crop[:] = 0
        self.assertGreater(self.hier.lgi.max(), 0)
        self.assertTrue(np.array_equal(extract_labels(self.hier.lgi, (0, 0, 0), SHAPE), self.hier.lgi))

    def test_features_ids_and_orientations_are_those_of_the_parent(self):
        sub = extract_subset(self.hier, (3, 0, 0), (6, 4, 2))   # only grain 3, z < 2: block 4
        self.assertEqual(list(sub.features), [4])
        self.assertIs(sub.features[4], self.hier.features[4])
        self.assertEqual(sub.voxel_counts, {4: 3 * 4 * 2})
        self.assertEqual((sub.origin, sub.parent_shape), ((3, 0, 0), SHAPE))
        self.assertEqual(sub.pag_order, [2])
        self.assertEqual(sub.pck_order, [(2, 1)])
        self.assertEqual(sub.blk_order, [(2, 1, 1)])
        self.assertEqual((sub.units, sub.voxel_size), (self.hier.units, self.hier.voxel_size))
        self.assertEqual(self.hier.origin, None)               # the parent is untouched

    def test_larger_crop_keeps_parent_order(self):
        sub = extract_subset(self.hier, (1, 0, 0), (5, 4, 4))
        self.assertEqual(sorted(sub.features), [1, 2, 3, 4, 5])
        self.assertEqual(sub.pag_order, [1, 2])
        self.assertEqual(sub.pck_order, [(1, 1), (1, 2), (2, 1)])
        self.assertEqual(sub.voxel_counts[1], self.hier.voxel_counts[1] // 2)

    def test_cut_statistics(self):
        sub = extract_subset(self.hier, (1, 0, 0), (3, 4, 4))
        stats = cut_statistics(self.hier.voxel_counts, sub.voxel_counts, small_threshold=20)
        self.assertEqual(stats['features_in_parent'], 5)
        self.assertEqual(stats['features_in_crop'], 3)
        self.assertEqual(stats['features_dropped'], 2)
        self.assertEqual(stats['cut_features'], 2)           # blocks 1, 2 lose their x = 0 slab
        self.assertEqual(stats['whole_features'], 1)         # block 3 (grain 2) is whole
        self.assertEqual(stats['fragments_below_threshold'], 3)   # 8, 8 and 16 voxels, threshold 20
        self.assertEqual(stats['smallest_in_crop'], 8)

    def test_deck_of_a_crop_keeps_names_and_global_ids(self):
        full_ident = resolve_identities(self.hier.voxel_counts)
        pts, tets, grains = voxel_tets(self.hier.lgi)
        full = write_deck(self.root / 'full', pts, tets, grains, self.hier, full_ident)
        sub = extract_subset(self.hier, (1, 0, 0), (5, 4, 4))
        pts, tets, grains = voxel_tets(sub.lgi)
        part = write_deck(self.root / 'part', pts, tets, grains, sub,
                          resolve_identities(sub.voxel_counts))
        text = lambda r, name: (r['directory'] / name).read_text()
        for line in text(full, '05_materials.inp').split('*Material, name=')[1:]:
            if line.splitlines()[0] in text(part, '05_materials.inp'):
                self.assertIn(line.rstrip(), text(part, '05_materials.inp'))
        self.assertEqual(set(read_elsets(part['directory'] / '03c_elsets_blk.inp')),
                         set(read_elsets(full['directory'] / '03c_elsets_blk.inp')))
        info = part['info']
        self.assertEqual(info['subset_of_parent'], {'origin_voxels': [1, 0, 0], 'parent_shape': list(SHAPE)})
        self.assertIsNone(full['info']['subset_of_parent'])
        self.assertIn('Subset of a parent structure: first voxel (1, 0, 0)', text(part, 'model_master.inp'))
        self.assertNotIn('Subset of a parent', text(full, 'model_master.inp'))

    def test_deck_of_a_crop_with_fewer_blocks(self):
        sub = extract_subset(self.hier, (3, 0, 0), (6, 4, 2))
        pts, tets, grains = voxel_tets(sub.lgi)
        res = write_deck(self.root / 'one', pts, tets, grains, sub, resolve_identities(sub.voxel_counts))
        self.assertEqual(sorted(read_elsets(res['directory'] / '03c_elsets_blk.inp')), ['es_blk_2_1_1'])
        self.assertEqual(res['info']['features_in_structure'], 1)
        material = (res['directory'] / '05_materials.inp').read_text()
        self.assertIn('MAT_BLK_2_1_1', material)
        self.assertIn(', 8., 15, 0', material)                  # global ID of the full structure

    def test_preview_figure(self):
        fig = subset_preview_figure(self.hier.lgi, (1, 0, 0), (5, 4, 4))
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].patches), 1)


class StrainTests(unittest.TestCase):
    def test_domain_lengths_and_strain(self):
        with work_dir() as root:
            (root / 'raw').mkdir()
            write_raw_folder(root / 'raw')
            hier = load_hierarchy(root / 'raw', 'blk')
        self.assertEqual(domain_lengths(hier), tuple(n * 0.001 for n in SHAPE))
        self.assertAlmostEqual(domain_lengths(hier, 'microns')[2], 4.0)
        # the displacement and the coordinates share the length scale, so it cancels
        settings = DeckSettings(applied_displacement=0.002, load_axis='z')
        self.assertAlmostEqual(applied_strain(settings, hier), 0.5)
        self.assertAlmostEqual(applied_strain(DeckSettings(applied_displacement=0.002, load_axis='z',
                                                           length_scale=3.0), hier), 0.5)
        self.assertAlmostEqual(applied_strain(DeckSettings(applied_displacement=3.0, load_axis='x',
                                                           output_units='microns'), hier), 0.5)
        sub = extract_subset(hier, (0, 0, 0), (6, 4, 2))
        self.assertAlmostEqual(applied_strain(DeckSettings(applied_displacement=0.002, load_axis='z'),
                                              sub), 1.0)


class RecordAndFolderTests(unittest.TestCase):
    def test_record_is_json_ready(self):
        spec = BoxSpec('percent', (0, 0, 0), (50, 50, 50))
        lo, hi = resolve_box(spec, (10, 10, 10))
        record = subset_record(spec, lo, hi, (10, 10, 10), 0.001, 'mm', 'mm',
                               statistics={'cut_features': 3})
        with work_dir() as root:
            path = write_subset_record(root / 'run', record)
            loaded = json.loads(path.read_text())
        self.assertEqual(loaded['origin_voxels'], [0, 0, 0])
        self.assertEqual(loaded['entered']['mode'], 'percent')
        self.assertEqual(loaded['box']['index']['hi'], [5, 5, 5])
        self.assertEqual(loaded['statistics']['cut_features'], 3)

    def test_indexed_folders(self):
        with work_dir() as root:
            self.assertEqual(next_index(root / 'none'), 1)
            first = new_indexed_directory(root, 'run_subset')
            second = new_indexed_directory(root, 'run_subset')
            other = new_indexed_directory(root, 'other')
            self.assertEqual((first.name, second.name, other.name),
                             ('run_subset_001', 'run_subset_002', 'other_001'))
            (root / 'run_subset_007').mkdir()
            self.assertEqual(new_indexed_directory(root, 'run_subset').name, 'run_subset_008')
            exports = root / 'abaqus'
            exports.mkdir()
            self.assertEqual(next_index(exports), 1)
            (exports / '001_fm').mkdir()
            (exports / '001_mesh').mkdir()
            (exports / '002_fm').mkdir()
            (exports / 'notes').mkdir()
            self.assertEqual(next_index(exports), 3)


if __name__ == '__main__':
    unittest.main()
