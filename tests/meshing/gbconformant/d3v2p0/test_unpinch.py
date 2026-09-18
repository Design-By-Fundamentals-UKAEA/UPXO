import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
from upxo.meshing.gbconformant.d3v2p0.unpinch import separate_pinch_fans, _fans


class UnpinchTests(unittest.TestCase):
    def source(self):
        a = np.ones((4, 4, 4), dtype=int)
        a[1, 1, 1] = a[2, 2, 2] = 2
        return smooth_interfaces(a, iterations=0)

    def test_split_and_press_point_contact(self):
        source = self.source()
        before = source.points.copy()
        result, report, audit = separate_pinch_fans(source, max_displacement=.3)
        self.assertEqual(report['split_contacts'], 1)
        self.assertEqual(len(result.points), len(source.points)+1)
        self.assertFalse(_fans(result.triangles, len(result.points)))
        self.assertGreater(report['minimum_tip_separation'], .02)
        self.assertLessEqual(report['maximum_displacement'], .3+1e-12)
        self.assertTrue(report['edge_incidence_preserved'])
        np.testing.assert_array_equal(source.points, before)
        np.testing.assert_array_equal(source.grain_pairs, result.grain_pairs)
        self.assertEqual(len(audit[0]['fan_nodes']), 2)

    def test_disabled_and_unsuccessful_opening(self):
        source = self.source()
        for options in [dict(enabled=False), dict(min_separation=10.)]:
            result, report, _ = separate_pinch_fans(source, **options)
            np.testing.assert_array_equal(result.points, source.points)
            np.testing.assert_array_equal(result.triangles, source.triangles)
            self.assertEqual(report['split_contacts'], 0)

    def test_shared_junction_is_not_cut(self):
        a = np.ones((4, 4, 4), dtype=int)
        a[2:] = 2
        a[:2, 2:] = 3
        source = smooth_interfaces(a, iterations=0)
        result, report, _ = separate_pinch_fans(source)
        self.assertEqual(report['candidate_contacts'], 0)
        np.testing.assert_array_equal(source.points, result.points)
        np.testing.assert_array_equal(source.triangles, result.triangles)


if __name__ == '__main__':
    unittest.main()
