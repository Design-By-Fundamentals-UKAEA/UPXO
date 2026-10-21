import pickle
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import detect_levels, load_hierarchy
from upxo.meshing.gbconformant.fm_steel_abaqus.hierarchy import parent_tables
from .helpers import work_dir, write_raw_folder


class HierarchyTests(unittest.TestCase):
    def test_detects_levels_present(self):
        with work_dir() as root:
            write_raw_folder(root)
            self.assertEqual(detect_levels(root), ['pag', 'pck', 'blk'])
        with work_dir() as root:
            write_raw_folder(root, with_sblk=True)
            self.assertEqual(detect_levels(root), ['pag', 'pck', 'blk', 'sblk'])

    def test_level_needs_both_files(self):
        with work_dir() as root:
            write_raw_folder(root)
            (root / 'blk_orientations.pkl').unlink()
            self.assertEqual(detect_levels(root), ['pag', 'pck'])

    def test_prefix_override(self):
        with work_dir() as root:
            write_raw_folder(root)
            (root / 'blk_lgi.npy').rename(root / 'lath_lgi.npy')
            (root / 'blk_orientations.pkl').rename(root / 'lath_orientations.pkl')
            self.assertEqual(detect_levels(root), ['pag', 'pck'])
            self.assertEqual(detect_levels(root, {'blk': 'lath'}), ['pag', 'pck', 'blk'])
            self.assertEqual(len(load_hierarchy(root, 'blk', {'blk': 'lath'}).features), 5)

    def test_block_features_names_orientations_and_global_ids(self):
        with work_dir() as root:
            _, orientations, _ = write_raw_folder(root)
            h = load_hierarchy(root, 'blk')
        self.assertEqual((h.units, h.voxel_size), ('mm', 0.001))
        keys = {f.label: f.key for f in h.features.values()}
        self.assertEqual(keys, {1: (1, 1, 1), 2: (1, 1, 2), 3: (1, 2, 1),
                                4: (2, 1, 1), 5: (2, 1, 2)})
        self.assertEqual({l: f.global_id for l, f in h.features.items()},
                         {1: 5, 2: 6, 3: 7, 4: 8, 5: 9})
        self.assertEqual(h.features[3].euler, orientations['blk']['B_1_2_1'])
        self.assertEqual(h.features[3].source_id, 'B_1_2_1')
        self.assertEqual(h.pag_order, [1, 2])
        self.assertEqual(h.pck_order, [(1, 1), (1, 2), (2, 1)])
        self.assertEqual(h.voxel_counts[1], 16)

    def test_packet_and_pag_levels(self):
        with work_dir() as root:
            _, orientations, _ = write_raw_folder(root)
            pck = load_hierarchy(root, 'pck')
            pag = load_hierarchy(root, 'pag')
        self.assertEqual({f.key for f in pck.features.values()}, {(1, 1), (1, 2), (2, 1)})
        self.assertEqual({f.label: f.global_id for f in pck.features.values()}, {1: 2, 2: 3, 3: 4})
        self.assertEqual(pck.features[2].euler, orientations['pck'][2])
        self.assertEqual({f.label: f.global_id for f in pag.features.values()}, {1: 0, 2: 1})

    def test_subblock_features(self):
        with work_dir() as root:
            _, orientations, _ = write_raw_folder(root, with_sblk=True)
            h = load_hierarchy(root, 'sblk')
        self.assertEqual(len(h.features), 10)
        first = h.features[1]
        self.assertEqual(first.key, (1, 1, 1, 1))
        self.assertEqual(first.euler, orientations['sblk']['SB_B_1_1_1_1'])
        self.assertEqual(first.global_id, 10)
        self.assertEqual(h.features[10].key, (2, 1, 2, 2))
        self.assertEqual(len(h.sblk_order), 10)

    def test_parent_tables(self):
        with work_dir() as root:
            write_raw_folder(root)
            h = load_hierarchy(root, 'blk')
        pag, pck, pck_keys, blk, blk_keys = parent_tables(h)
        self.assertEqual(pag[1:].tolist(), [1, 1, 1, 2, 2])
        self.assertEqual([pck_keys[i] for i in pck[1:]], [(1, 1), (1, 1), (1, 2), (2, 1), (2, 1)])
        self.assertEqual(len(blk_keys), 5)

    def test_unsupported_inputs_fail_loudly(self):
        with work_dir() as root:
            write_raw_folder(root, retained=True)
            with self.assertRaises(NotImplementedError):
                load_hierarchy(root, 'blk')
        with work_dir() as root:
            write_raw_folder(root, zero_label=True)
            with self.assertRaises(NotImplementedError):
                load_hierarchy(root, 'blk')

    def test_missing_pieces(self):
        with work_dir() as root:
            write_raw_folder(root)
            with self.assertRaises(FileNotFoundError):
                load_hierarchy(root, 'sblk')
            with self.assertRaises(ValueError):
                load_hierarchy(root, 'lath')
            (root / 'raw_export.pkl').unlink()
            with self.assertRaises(FileNotFoundError):
                load_hierarchy(root, 'blk')

    def test_label_without_orientation_is_rejected(self):
        with work_dir() as root:
            write_raw_folder(root)
            with (root / 'blk_orientations.pkl').open('rb') as handle:
                data = pickle.load(handle)
            data.pop('B_1_1_2')
            with (root / 'blk_orientations.pkl').open('wb') as handle:
                pickle.dump(data, handle)
            with self.assertRaises(ValueError):
                load_hierarchy(root, 'blk')


if __name__ == '__main__':
    unittest.main()
