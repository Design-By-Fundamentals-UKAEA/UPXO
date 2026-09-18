import unittest
import numpy as np
from scipy.ndimage import label
from upxo.meshing.gbconformant.d3v2p0.pinch_cleaning import detect_pinch_candidates, clean_pinch_contacts


class PinchTests(unittest.TestCase):
    def test_corner_contact_and_rotation(self):
        a = np.ones((2, 2, 2), dtype=int)
        a[0, 0, 0] = a[1, 1, 1] = 2
        self.assertEqual(detect_pinch_candidates(a), {(0, 0, 0): {2}})
        self.assertEqual(detect_pinch_candidates(np.rot90(a)), {(0, 0, 0): {2}})
        a[0] = 2
        a[1] = 1
        self.assertEqual(detect_pinch_candidates(a), {})

    def test_repair_invariants_and_relabeling(self):
        a = np.random.default_rng(10).integers(1, 4, (5, 5, 5))
        original = a.copy()
        before = detect_pinch_candidates(a)
        options = dict(max_volume_fraction=.2, protect_grains_up_to=0, max_changes=10)
        b, report, edits = clean_pinch_contacts(a, **options)
        self.assertGreater(len(edits), 0)
        np.testing.assert_array_equal(a, original)
        np.testing.assert_array_equal(np.unique(b), np.unique(a))
        after = detect_pinch_candidates(b)
        self.assertLess(sum(map(len, after.values())), sum(map(len, before.values())))
        self.assertTrue({(o, g) for o, gs in after.items() for g in gs}.issubset(
                        {(o, g) for o, gs in before.items() for g in gs}))
        for g in np.unique(a):
            self.assertEqual(label(a == g)[1], label(b == g)[1])
            self.assertLessEqual(abs(np.sum(a == g)-np.sum(b == g)), max(1, int(np.sum(a == g)*.2)))
        replay = a.copy()
        for edit in edits:
            pos = tuple(edit['voxel'])
            self.assertEqual(replay[pos], edit['from_grain'])
            replay[pos] = edit['to_grain']
        np.testing.assert_array_equal(replay, b)
        # Grain ID ordering must not drive selection or tie-breaking.
        mapping = np.array([0, 30, 10, 20])
        renamed, _, _ = clean_pinch_contacts(mapping[a], **options)
        np.testing.assert_array_equal(renamed, mapping[b])

    def test_disabled_and_zero_budget(self):
        a = np.random.default_rng(1).integers(1, 4, (4, 4, 4))
        for kwargs in [dict(enabled=False), dict(max_changes=0), dict(max_volume_fraction=0)]:
            b, report, edits = clean_pinch_contacts(a, **kwargs)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(edits, [])

    def test_small_grain_protection(self):
        a = np.ones((3, 3, 3), dtype=int)
        a[0, 0, 0] = a[1, 1, 1] = 2
        b, _, edits = clean_pinch_contacts(a)
        self.assertTrue(all(e['from_grain'] != 2 for e in edits))
        self.assertGreaterEqual(np.sum(b == 2), 2)


if __name__ == '__main__':
    unittest.main()
