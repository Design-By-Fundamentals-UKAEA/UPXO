import json
import unittest

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus import (
    load_hierarchy, merge_log, resolve_identities, write_merge_log)
from upxo.xtalphy.crystal_orientation import cubic_misorientation

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder


class MergeLogTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        write_raw_folder(self.root)
        self.hierarchy = load_hierarchy(self.root, 'blk')
        self.sizes = self.hierarchy.voxel_counts

    def identity(self, survivor, rule='larger'):
        return resolve_identities(self.sizes, survivor_map=survivor, rule=rule, seed=1)

    def test_no_merge_gives_an_empty_log(self):
        log = merge_log(self.hierarchy, self.identity({k: k for k in self.sizes}))
        self.assertEqual(log['rows'], [])
        self.assertEqual(log['summary']['absorbed_features'], 0)
        self.assertIsNone(log['summary']['voxel_weighted_mean_deg'])

    def test_each_absorbed_feature_is_compared_with_the_winner(self):
        survivor = {k: k for k in self.sizes}
        survivor[4] = 3                                   # feature 4 merged into 3
        identity = self.identity(survivor)
        log = merge_log(self.hierarchy, identity)
        self.assertEqual(len(log['rows']), 1)
        row = dict(zip(log['columns'], log['rows'][0]))
        winner = identity.winner[3]
        absorbed = 4 if winner == 3 else 3
        self.assertEqual((row['meshed_label'], row['absorbed'], row['winner']), (3, absorbed, winner))
        expected = cubic_misorientation(self.hierarchy.features[absorbed].euler,
                                        self.hierarchy.features[winner].euler)[0]
        self.assertAlmostEqual(row['misorientation_deg'], expected)
        self.assertEqual(row['voxels'], self.sizes[absorbed])
        self.assertTrue(row['absorbed_name'].startswith('es_blk_') and row['winner_name'].startswith('es_blk_'))
        self.assertEqual(log['summary']['voxels'], self.sizes[absorbed])
        self.assertAlmostEqual(log['summary']['voxel_weighted_mean_deg'], expected)

    def test_the_survivor_rule_compares_with_the_grain_that_absorbed(self):
        survivor = {k: k for k in self.sizes}
        survivor[5] = 3
        log = merge_log(self.hierarchy, self.identity(survivor, rule='survivor'))
        row = dict(zip(log['columns'], log['rows'][0]))
        self.assertEqual((row['absorbed'], row['winner']), (5, 3))
        self.assertTrue(row['winner_is_survivor'])

    def test_a_chain_of_merges_lists_every_absorbed_feature(self):
        survivor = {k: k for k in self.sizes}
        survivor.update({2: 1, 4: 3, 5: 3})
        log = merge_log(self.hierarchy, self.identity(survivor))
        self.assertEqual(log['summary']['groups'], 2)
        self.assertEqual(log['summary']['absorbed_features'], 3)
        self.assertEqual(len({r[1] for r in log['rows']}), 3)

    def test_only_cubic_symmetry_is_available(self):
        with self.assertRaises(ValueError):
            merge_log(self.hierarchy, self.identity({k: k for k in self.sizes}), symmetry='hexagonal')

    def test_the_log_is_written_as_json_and_csv(self):
        survivor = {k: k for k in self.sizes}
        survivor[4] = 3
        log = merge_log(self.hierarchy, self.identity(survivor))
        json_path, csv_path = write_merge_log(self.root / 'out', log)
        self.assertEqual(json.loads(json_path.read_text())['summary']['absorbed_features'], 1)
        lines = csv_path.read_text().splitlines()
        self.assertEqual(lines[0].split(',')[:3], ['meshed_label', 'absorbed', 'absorbed_name'])
        self.assertEqual(len(lines), 2)


if __name__ == '__main__':
    unittest.main()
