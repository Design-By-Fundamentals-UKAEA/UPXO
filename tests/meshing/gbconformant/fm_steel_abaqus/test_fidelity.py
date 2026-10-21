import unittest

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus import ComparisonSettings, Hierarchy, grain_fidelity
from upxo.meshing.gbconformant.fm_steel_abaqus.fidelity import (
    mesh_aspect, mesh_boundaries, voxel_aspect, voxel_boundaries)

from ..fm_steel_abaqus.helpers import voxel_tets


def hierarchy_of(labels, voxel_size=1.0):
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    features = {int(i): object() for i in ids if i > 0}
    return Hierarchy(level='blk', lgi=labels, features=features,
                     voxel_counts={int(i): int(c) for i, c in zip(ids, counts) if i > 0},
                     units='mm', voxel_size=voxel_size)


def structure():
    labels = np.zeros((4, 3, 3), dtype=int)
    labels[:2] = 1
    labels[2:, :2] = 2
    labels[2:, 2:] = 3
    return labels


class VoxelSideTests(unittest.TestCase):
    def test_two_cubes(self):
        faces, pairs = voxel_boundaries(np.array([[[1, 2]]]))
        self.assertEqual(list(faces[1:3]), [6, 6])
        self.assertEqual(pairs, {(1, 2): 1})

    def test_background_counts_as_exposed_but_not_as_a_neighbour(self):
        faces, pairs = voxel_boundaries(np.array([[[1, 0, 2]]]))
        self.assertEqual(list(faces[1:3]), [6, 6])
        self.assertEqual(pairs, {})

    def test_aspect_of_a_bar_is_its_length_ratio(self):
        labels = np.ones((1, 1, 4), dtype=int)
        self.assertAlmostEqual(voxel_aspect(labels, 2)[1], 4.0)
        self.assertAlmostEqual(voxel_aspect(np.ones((3, 3, 3), dtype=int), 2)[1], 1.0)


class MeshSideTests(unittest.TestCase):
    def test_a_structured_mesh_has_the_voxel_areas_and_pairs(self):
        labels = structure()
        points, tets, grains = voxel_tets(labels)
        area, pairs = mesh_boundaries(points, tets, grains)
        faces, vpairs = voxel_boundaries(labels)
        for label in (1, 2, 3):
            self.assertAlmostEqual(area[label], faces[label])
        self.assertEqual(set(pairs), set(vpairs))
        for key, shared in pairs.items():
            self.assertAlmostEqual(shared, vpairs[key])
        aspect = mesh_aspect(points, tets, grains, 4)
        np.testing.assert_allclose(aspect[1:], voxel_aspect(labels, 4)[1:], rtol=1e-9)

    def test_a_triangle_in_three_tetrahedra_is_refused(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1]], dtype=float)
        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])
        with self.assertRaises(ValueError):
            mesh_boundaries(points, tets, np.array([1, 2, 3]))


class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.labels = structure()
        self.hierarchy = hierarchy_of(self.labels, voxel_size=0.001)
        self.points, self.tets, self.grains = voxel_tets(self.labels)

    def run_it(self, tets=None, grains=None, **kwargs):
        return grain_fidelity(self.hierarchy, self.points, self.tets if tets is None else tets,
                              self.grains if grains is None else grains,
                              ComparisonSettings(), **kwargs)

    def test_a_faithful_mesh_changes_nothing(self):
        result = self.run_it()
        at = {name: i for i, name in enumerate(result['columns'])}
        self.assertEqual([r[0] for r in result['rows']], [1, 2, 3])
        for row in result['rows']:
            self.assertAlmostEqual(row[at['volume_change']], 0.0, places=12)
            self.assertAlmostEqual(row[at['area_ratio']], 1.0)
            self.assertEqual(row[at['neighbours_before']], row[at['neighbours_after']])
            self.assertEqual((row[at['neighbours_lost']], row[at['neighbours_gained']]), (0, 0))
            self.assertAlmostEqual(row[at['aspect_after']], row[at['aspect_before']], places=9)
            self.assertEqual(row[at['flag']], '')
        self.assertEqual(result['units'], 'mm')
        self.assertAlmostEqual(result['rows'][0][at['volume_before']], 18 * 1e-9)       # 18 voxels of (1e-3 mm)^3
        self.assertEqual(result['flagged'], [])
        self.assertEqual(result['stats']['volume_change']['count'], 3)

    def test_dropping_half_a_grain_flags_it_and_a_missing_grain_is_lost(self):
        grain_2 = np.flatnonzero(self.grains == 2)
        keep = np.ones(len(self.tets), dtype=bool)
        keep[grain_2[: len(grain_2) // 2]] = False
        half = self.run_it(self.tets[keep], self.grains[keep])
        at = {name: i for i, name in enumerate(half['columns'])}
        row = half['rows'][1]
        self.assertAlmostEqual(row[at['volume_change']], -0.5, places=6)
        self.assertEqual(row[at['flag']], 'shrunk')
        self.assertEqual(half['flagged'], [2])
        self.assertEqual(self.run_it(self.tets[keep], self.grains[keep], shrink_limit=0.6)['flagged'], [])
        gone = self.grains != 3
        lost = self.run_it(self.tets[gone], self.grains[gone])
        self.assertEqual(lost['lost'], [3])
        row3 = lost['rows'][2]
        self.assertEqual(row3[at['flag']], 'lost')
        self.assertTrue(np.isnan(row3[at['volume_after']]))
        self.assertGreater(lost['rows'][1][at['neighbours_lost']], 0)         # grain 2 lost grain 3 as a neighbour

    def test_a_merged_grain_takes_its_neighbours_label(self):
        merged = np.where(self.grains == 3, 2, self.grains)
        result = self.run_it(grains=merged)
        at = {name: i for i, name in enumerate(result['columns'])}
        rows = {r[0]: r for r in result['rows']}
        self.assertAlmostEqual(rows[2][at['volume_change']], 0.0 + (6 / 12), places=6)   # grain 2 now holds grain 3 too
        self.assertEqual(rows[3][at['flag']], 'lost')

    def test_wrong_label_count_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_it(grains=self.grains[:-1])


if __name__ == '__main__':
    unittest.main()
