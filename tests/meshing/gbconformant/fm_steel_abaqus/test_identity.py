import unittest
from upxo.meshing.gbconformant.fm_steel_abaqus import resolve_identities

SIZES = {1: 50, 2: 3, 3: 3, 4: 40}


class IdentityTests(unittest.TestCase):
    def test_no_merges_keeps_every_label(self):
        ident = resolve_identities(SIZES)
        self.assertEqual(ident.winner, {k: k for k in SIZES})
        self.assertEqual(ident.report['lost_raw_labels'], [])
        self.assertEqual(ident.lookup().tolist(), [-1, 1, 2, 3, 4])

    def test_larger_member_wins_even_when_it_was_absorbed(self):
        # 1 (50 voxels) was merged into 4 (40 voxels): 4 survived but 1 is larger.
        ident = resolve_identities(SIZES, survivor_map={1: 4, 2: 2, 3: 3, 4: 4})
        self.assertEqual(ident.winner, {2: 2, 3: 3, 4: 1})
        self.assertEqual(ident.members[4], (1, 4))
        self.assertEqual(ident.report['lost_raw_labels'], [4])
        self.assertEqual(ident.report['winner_differs_from_survivor'], 1)

    def test_survivor_rule_keeps_the_absorbing_grain(self):
        ident = resolve_identities(SIZES, survivor_map={1: 4, 2: 2, 3: 3, 4: 4}, rule='survivor')
        self.assertEqual(ident.winner[4], 4)
        self.assertEqual(ident.report['lost_raw_labels'], [1])

    def test_chain_of_merges_picks_the_largest_member(self):
        ident = resolve_identities(SIZES, survivor_map={1: 4, 2: 4, 3: 4, 4: 4})
        self.assertEqual(ident.winner, {4: 1})
        self.assertEqual(sorted(ident.report['lost_raw_labels']), [2, 3, 4])

    def test_ties_are_seeded_and_independent_of_order(self):
        merge = {1: 1, 2: 3, 3: 3, 4: 4}
        first = resolve_identities(SIZES, survivor_map=merge, seed=7).winner[3]
        again = resolve_identities(dict(reversed(list(SIZES.items()))),
                                   survivor_map=dict(reversed(list(merge.items()))), seed=7).winner[3]
        self.assertEqual(first, again)
        self.assertIn(first, (2, 3))
        picks = {resolve_identities(SIZES, survivor_map=merge, seed=s).winner[3] for s in range(40)}
        self.assertEqual(picks, {2, 3})

    def test_shuffle_is_inverted(self):
        shuffle = {1: 40, 2: 30, 3: 20, 4: 10}
        survivor = {40: 10, 30: 30, 20: 20, 10: 10}
        ident = resolve_identities(SIZES, shuffle_map=shuffle, survivor_map=survivor)
        self.assertEqual(ident.winner, {10: 1, 20: 3, 30: 2})
        self.assertEqual(ident.survivor_raw, {10: 4, 20: 3, 30: 2})

    def test_inconsistent_maps_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_identities(SIZES, shuffle_map={1: 1, 2: 2, 3: 3})
        with self.assertRaises(ValueError):
            resolve_identities(SIZES, shuffle_map={1: 1, 2: 1, 3: 3, 4: 4})
        with self.assertRaises(ValueError):
            resolve_identities(SIZES, survivor_map={1: 2, 2: 3, 3: 3, 4: 4})
        with self.assertRaises(ValueError):
            resolve_identities(SIZES, rule='smaller')


if __name__ == '__main__':
    unittest.main()
