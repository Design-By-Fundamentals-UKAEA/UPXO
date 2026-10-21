import unittest
from types import SimpleNamespace

import numpy as np

from upxo.meshing.gbconformant.pipeline.quality_report import (
    grain_table, quality_report, tet_measures, tet_volumes_and_aspect)

from ..fm_steel_abaqus.helpers import voxel_tets

REGULAR = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def mesh(labels, with_quality=True):
    points, tets, grains = voxel_tets(labels)
    return SimpleNamespace(points=points, tetrahedra=tets, grain_ids=grains,
                           quality=np.linspace(0.2, 0.9, len(tets)) if with_quality else None)


class AspectTests(unittest.TestCase):
    def test_regular_tetrahedron(self):
        volume, aspect = tet_volumes_and_aspect(REGULAR, [[0, 1, 2, 3]])
        self.assertAlmostEqual(volume[0], 8.0 / 3.0)
        self.assertAlmostEqual(aspect[0], 1.0)

    def test_scale_and_translation_do_not_change_it(self):
        _, a = tet_volumes_and_aspect(REGULAR * 7 + 3, [[0, 1, 2, 3]])
        self.assertAlmostEqual(a[0], 1.0)

    def test_a_flattened_tetrahedron_has_a_large_ratio_and_a_flat_one_is_infinite(self):
        flat = REGULAR.copy()
        flat[:, 2] *= 0.01
        _, aspect = tet_volumes_and_aspect(flat, [[0, 1, 2, 3]])
        self.assertGreater(aspect[0], 20)
        degenerate = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        _, aspect = tet_volumes_and_aspect(degenerate, [[0, 1, 2, 3]])
        self.assertTrue(np.isinf(aspect[0]))


class GrainTests(unittest.TestCase):
    def setUp(self):
        labels = np.ones((4, 3, 3), dtype=int)
        labels[2:] = 2
        labels[:, :, 2] = np.where(labels[:, :, 2] == 2, 3, 3)      # a third grain on one side
        self.labels = labels
        self.tets = mesh(labels)

    def test_grain_columns_match_a_direct_computation(self):
        measures = tet_measures(self.tets)
        table = grain_table(self.tets, measures, target_dihedral=30.0)
        grain = np.asarray(self.tets.grain_ids)
        self.assertEqual(list(table['grain']), sorted(set(grain)))
        for i, g in enumerate(table['grain']):
            mask = grain == g
            self.assertEqual(table['tets'][i], mask.sum())
            self.assertAlmostEqual(table['volume'][i], measures['volume'][mask].sum())
            self.assertAlmostEqual(table['min_dihedral'][i], measures['smallest_dihedral'][mask].min())
            self.assertAlmostEqual(table['max_dihedral'][i], measures['largest_dihedral'][mask].max())
            self.assertAlmostEqual(table['min_sicn'][i], self.tets.quality[mask].min())
            self.assertAlmostEqual(table['mean_sicn'][i], self.tets.quality[mask].mean())
            self.assertEqual(table['below_target'][i], (measures['smallest_dihedral'][mask] < 30).sum())
        self.assertAlmostEqual(table['volume'].sum(), self.labels.size)        # unit voxels

    def test_report_counts_against_the_limits_and_names_the_worst_grains(self):
        report = quality_report(self.tets, target_dihedral=30.0, top=2)
        run, grains = report['run'], report['grains']
        small = tet_measures(self.tets)['smallest_dihedral']
        self.assertEqual(run['tetrahedra'], len(small))
        self.assertEqual(run['grains'], 3)
        self.assertEqual(run['dihedral_below']['30'], int(np.sum(small < 30)))
        self.assertEqual(run['dihedral_below']['15'], int(np.sum(small < 15)))
        self.assertAlmostEqual(run['fraction_below_target'], np.sum(small < 30) / len(small))
        self.assertEqual(run['sicn_below']['0.3'], int(np.sum(self.tets.quality < 0.3)))
        self.assertEqual(run['degenerate'], 0)
        self.assertEqual(len(report['worst']), 2)
        order = np.argsort(grains['min_dihedral'], kind='stable')
        self.assertEqual(report['worst'], [int(grains['grain'][i]) for i in order[:2]])

    def test_without_minsicn_those_columns_are_absent(self):
        report = quality_report(mesh(self.labels, with_quality=False))
        self.assertNotIn('sicn', report['run'])
        self.assertNotIn('min_sicn', report['grains'])


if __name__ == '__main__':
    unittest.main()
