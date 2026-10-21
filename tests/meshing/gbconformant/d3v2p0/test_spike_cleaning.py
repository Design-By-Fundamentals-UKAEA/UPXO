import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.spike_cleaning import clean_spikes


def two_grains_with_a_spike():
    labels = np.ones((8, 8, 8), dtype=np.int32)
    labels[4:] = 2
    labels[4, 3, 3] = 1          # a one-voxel spike of grain 1 inside grain 2
    return labels


class SpikeCleaningTests(unittest.TestCase):
    def test_spike_is_removed_and_input_is_untouched(self):
        labels = two_grains_with_a_spike()
        before = labels.copy()
        cleaned, report = clean_spikes(labels, max_passes=10)
        self.assertTrue(np.array_equal(labels, before))
        self.assertEqual(int(cleaned[4, 3, 3]), 2)
        self.assertEqual(report['net_changed_voxels'], 1)
        self.assertEqual(report['stop_reason'], 'no further changes')
        self.assertEqual(report['history'][-1]['changed_voxels'], 0)
        self.assertEqual(sorted(np.unique(cleaned).tolist()), [1, 2])

    def test_clean_array_is_unchanged(self):
        labels = np.ones((5, 5, 5), dtype=np.int32)
        labels[2:] = 2
        cleaned, report = clean_spikes(labels)
        self.assertTrue(np.array_equal(cleaned, labels))
        self.assertEqual(report['net_changed_voxels'], 0)
        self.assertEqual(len(report['history']), 1)

    def test_pass_limit_and_progress(self):
        calls = []
        _, report = clean_spikes(two_grains_with_a_spike(), max_passes=1,
                                 progress=lambda *a: calls.append(a))
        self.assertEqual(report['stop_reason'], 'maximum passes reached')
        self.assertEqual(calls, [('Spike removal pass 1/1', 1, 1)])

    def test_invalid_inputs(self):
        labels = two_grains_with_a_spike()
        for bad in (0, -3, True, 2.5):
            with self.assertRaises(ValueError):
                clean_spikes(labels, max_passes=bad)
        with self.assertRaises(ValueError):
            clean_spikes(-labels)
        with self.assertRaises(ValueError):
            clean_spikes(labels[0])


if __name__ == '__main__':
    unittest.main()
