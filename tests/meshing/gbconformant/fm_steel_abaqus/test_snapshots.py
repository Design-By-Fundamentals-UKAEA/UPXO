"""The grain structure measured at five points of the pipeline, and the parameters read from those measurements."""
import unittest

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus.boundary_classes import _angles
from upxo.meshing.gbconformant.fm_steel_abaqus.comparison import ComparisonSettings
from upxo.meshing.gbconformant.fm_steel_abaqus.junctions import junction_lines, tet_junction_edges
from upxo.meshing.gbconformant.fm_steel_abaqus.parameters import (
    AREA_ABSOLUTE, AREA_FRACTION, AREA_PER_VOLUME, BY_VOLUME, DISPLAY_UNITS, PARAMETERS, Unavailable, available,
    choose_units, count_noun, matched_items, parameter, read, unit_factor, unit_power, unit_text)
from upxo.meshing.gbconformant.fm_steel_abaqus.snapshots import (
    ORDER, SURFACE_PREPARED, SURFACE_RAW, SURFACE_SMOOTHED, TETS, TITLES, VOXELS, closed_volumes, key_of_title,
    pair_misorientation, surface_junction_edges, surface_snapshot, tet_snapshot, voxel_snapshot)
from upxo.meshing.gbconformant.pipeline.mesh_regions import mesh_surface

from .helpers import voxel_tets, work_dir
from .test_comparison import folder_pieces

SETTINGS = ComparisonSettings()


def pieces(merge=None):
    with work_dir() as root:
        hier, ident, pts, tets, grains = folder_pieces(root, merge=merge)
    return hier, ident, pts, tets, grains


def boundary_triangles(points, tets, grains, include_rve_faces=True):
    """The surface triangles of a mesh as (nodes, region_a, region_b): a box triangle has one grain on both sides."""
    surface = mesh_surface(tets, grains, len(points), include_rve_faces)
    a, b = surface.grains[:, 0].copy(), surface.grains[:, 1].copy()
    box = a == 0
    a[box] = b[box]
    return surface.nodes, a, b


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.hier, self.ident, self.pts, self.tets, self.grains = pieces()
        self.voxel = voxel_snapshot(self.hier, SETTINGS)
        self.tet = tet_snapshot(self.hier, self.pts, self.tets, self.grains, SETTINGS)

    def test_the_voxel_structure(self):
        self.assertEqual((self.voxel.key, self.voxel.title), (VOXELS, 'Voxel structure'))
        self.assertEqual(self.voxel.labels.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(self.voxel.units, 'mm')
        box = 6 * 4 * 4 * 0.001 ** 3
        self.assertAlmostEqual(self.voxel.volume.sum(), box)
        self.assertAlmostEqual(self.voxel.rve_volume, box)
        self.assertTrue(np.all(self.voxel.pair_labels[:, 0] < self.voxel.pair_labels[:, 1]))
        self.assertTrue(self.voxel.has_volume)

    def test_a_mesh_that_follows_the_voxels_is_measured_the_same(self):
        self.assertTrue(np.array_equal(self.voxel.labels, self.tet.labels))
        np.testing.assert_allclose(self.tet.volume, self.voxel.volume)
        np.testing.assert_allclose(self.tet.aspect, self.voxel.aspect)
        self.assertTrue(np.array_equal(self.tet.pair_labels, self.voxel.pair_labels))
        np.testing.assert_allclose(self.tet.pair_area, self.voxel.pair_area)
        np.testing.assert_allclose(np.sort(self.tet.lines['length']), np.sort(self.voxel.lines['length']))
        self.assertTrue(np.array_equal(self.tet.neighbour_count(), self.voxel.neighbour_count()))
        np.testing.assert_allclose(self.tet.grain_area(), self.voxel.grain_area())

    def test_areas_count_interior_boundaries_only(self):
        both = self.voxel.pair_area.sum()
        total_faces = (6 * 4 + 6 * 4 + 4 * 4) * 2 * 0.001 ** 2
        self.assertLess(both, 6 * 4 * 4 * 3 * 0.001 ** 2)
        self.assertGreater(both, 0)
        self.assertLess(self.voxel.grain_area().sum() / 2, 10 * total_faces)

    def test_neighbours_and_areas_per_grain(self):
        count = self.voxel.neighbour_count()
        for i, label in enumerate(self.voxel.labels):
            partners = {int(b) if int(a) == label else int(a) for a, b in self.voxel.pair_labels if label in (a, b)}
            self.assertEqual(count[i], len(partners))
        np.testing.assert_allclose(self.voxel.grain_area().sum(), 2 * self.voxel.pair_area.sum())

    def test_mean_misorientation_is_the_area_weighted_mean(self):
        mean = self.voxel.mean_misorientation()
        label = int(self.voxel.labels[0])
        rows = [i for i, (a, b) in enumerate(self.voxel.pair_labels) if label in (a, b)]
        expected = np.average(self.voxel.pair_angle[rows], weights=self.voxel.pair_area[rows])
        self.assertAlmostEqual(mean[0], expected)

    def test_pair_misorientation(self):
        pairs = self.voxel.pair_labels
        np.testing.assert_allclose(pair_misorientation(self.hier, pairs, 'cubic'), _angles(self.hier, pairs, 'cubic'))
        self.assertTrue(np.all(np.isnan(pair_misorientation(self.hier, pairs, 'hexagonal'))))
        self.assertTrue(np.isnan(pair_misorientation(self.hier, [[1, 999]], 'cubic')[0]))
        self.assertEqual(pair_misorientation(self.hier, [[2, 2]], 'cubic')[0], 0.0)
        self.assertEqual(len(pair_misorientation(self.hier, np.zeros((0, 2)), 'cubic')), 0)

    def test_the_two_surfaces_of_a_structure_keep_its_boundaries(self):
        nodes, a, b = boundary_triangles(self.pts, self.tets, self.grains, include_rve_faces=False)
        for key in (SURFACE_RAW, SURFACE_SMOOTHED):
            surface = surface_snapshot(key, self.hier, self.pts, nodes, a, b, SETTINGS)
            self.assertFalse(surface.has_volume)
            self.assertEqual(surface.title, TITLES[key])
            np.testing.assert_allclose(surface.pair_area, self.voxel.pair_area)
            np.testing.assert_allclose(np.sort(surface.lines['length']), np.sort(self.voxel.lines['length']))
            self.assertAlmostEqual(surface.rve_volume, self.voxel.rve_volume)

    def test_a_scaled_surface_scales_its_areas_and_lengths(self):
        nodes, a, b = boundary_triangles(self.pts, self.tets, self.grains, include_rve_faces=False)
        small = surface_snapshot(SURFACE_SMOOTHED, self.hier, self.pts * 0.5, nodes, a, b, SETTINGS)
        np.testing.assert_allclose(small.pair_area, self.voxel.pair_area * 0.25)
        np.testing.assert_allclose(np.sort(small.lines['length']), np.sort(self.voxel.lines['length']) * 0.5)


class ClosedSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.hier, self.ident, self.pts, self.tets, self.grains = pieces()
        self.tet = tet_snapshot(self.hier, self.pts, self.tets, self.grains, SETTINGS)
        self.nodes, self.a, self.b = boundary_triangles(self.pts, self.tets, self.grains)

    def closed(self, points=None, nodes=None, **kwargs):
        return surface_snapshot(SURFACE_PREPARED, self.hier, self.pts if points is None else points,
                                self.nodes if nodes is None else nodes, self.a, self.b, SETTINGS,
                                tets=(self.pts, self.tets, self.grains), **kwargs)

    def test_a_closed_surface_gives_the_grain_volumes_of_the_mesh(self):
        snap = self.closed()
        self.assertTrue(snap.has_volume)
        self.assertEqual(snap.problems['unmatched'], 0)
        np.testing.assert_allclose(snap.volume, self.tet.volume)
        np.testing.assert_allclose(snap.aspect, self.tet.aspect)
        np.testing.assert_allclose(snap.rve_volume, self.tet.rve_volume)
        np.testing.assert_allclose(snap.pair_area, self.tet.pair_area)

    def test_the_winding_of_the_stored_triangles_does_not_matter(self):
        rng = np.random.default_rng(3)
        flipped = self.nodes.copy()
        turn = rng.random(len(flipped)) < 0.5
        flipped[turn] = flipped[turn][:, ::-1]
        np.testing.assert_allclose(self.closed(nodes=flipped).volume, self.tet.volume)

    def test_the_volume_follows_the_coordinates_of_the_surface(self):
        squeezed = self.pts * np.array([0.5, 1.0, 1.0])
        np.testing.assert_allclose(self.closed(points=squeezed).volume, self.tet.volume * 0.5)

    def test_a_surface_that_is_not_made_of_faces_of_the_mesh_gives_no_volume(self):
        broken = self.nodes.copy()
        broken[0] = [0, 1, len(self.pts) - 1]
        snap = self.closed(nodes=broken)
        self.assertGreater(snap.problems['unmatched'], 0)
        self.assertFalse(snap.has_volume)

    def test_closed_volumes_directly(self):
        labels, volume, aspect, problems = closed_volumes(self.pts, self.nodes, self.pts, self.tets, self.grains)
        self.assertEqual(problems['unmatched'], 0)
        self.assertEqual(labels.tolist(), self.tet.labels.tolist())
        np.testing.assert_allclose(volume * 0.001 ** 3, self.tet.volume)

    def test_surface_junctions_match_the_mesh_junctions(self):
        for include in (False, True):
            nodes, a, b = boundary_triangles(self.pts, self.tets, self.grains, include_rve_faces=True)
            from_surface = surface_junction_edges(self.pts, nodes, a, b, include)
            from_mesh = tet_junction_edges(self.pts, self.tets, self.grains, include)
            self.assertEqual(len(from_surface.nodes), len(from_mesh.nodes), include)
            order_s = np.lexsort(np.vstack((from_surface.nodes.T, from_surface.sets.T)))
            order_m = np.lexsort(np.vstack((from_mesh.nodes.T, from_mesh.sets.T)))
            self.assertTrue(np.array_equal(from_surface.nodes[order_s], from_mesh.nodes[order_m]))
            self.assertTrue(np.array_equal(from_surface.sets[order_s], from_mesh.sets[order_m]))
            np.testing.assert_allclose(from_surface.lengths[order_s], from_mesh.lengths[order_m])
            self.assertEqual(len(junction_lines(from_surface).length), len(junction_lines(from_mesh).length))

    def test_a_surface_without_junctions(self):
        x = np.indices((4, 4, 4))[0]
        pts, tets, grains = voxel_tets((1 + (x >= 2)).astype(np.int64))
        nodes, a, b = boundary_triangles(pts, tets, grains, include_rve_faces=False)
        edges = surface_junction_edges(pts, nodes, a, b)
        self.assertEqual(len(edges.nodes), 0)


class KeysTests(unittest.TestCase):
    def test_titles_and_keys(self):
        self.assertEqual(ORDER, (VOXELS, SURFACE_RAW, SURFACE_SMOOTHED, SURFACE_PREPARED, TETS))
        for key in ORDER:
            self.assertEqual(key_of_title(TITLES[key]), key)
            self.assertEqual(key_of_title(key), key)
        with self.assertRaises(ValueError):
            key_of_title('The mesh')


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.hier, self.ident, self.pts, self.tets, self.grains = pieces()
        self.voxel = voxel_snapshot(self.hier, SETTINGS)
        self.tet = tet_snapshot(self.hier, self.pts, self.tets, self.grains, SETTINGS)
        nodes, a, b = boundary_triangles(self.pts, self.tets, self.grains, include_rve_faces=False)
        self.open = surface_snapshot(SURFACE_SMOOTHED, self.hier, self.pts, nodes, a, b, SETTINGS)

    def test_every_parameter_is_read_from_a_full_snapshot(self):
        for p in PARAMETERS:
            if p.key == 'pole_figures':
                continue
            series = read(p, self.voxel)
            self.assertGreater(series.count, 0, p.key)
            self.assertEqual(series.noun, p.noun)
        self.assertEqual(parameter('Grain volume').key, 'grain_volume')
        with self.assertRaises(KeyError):
            parameter('mass')

    def test_the_values(self):
        np.testing.assert_allclose(read('grain_volume', self.voxel).values, self.voxel.volume)
        self.assertTrue(read('coordination', self.voxel).integer)
        np.testing.assert_allclose(read('coordination', self.voxel).values, self.voxel.neighbour_count())
        np.testing.assert_allclose(read('grain_area', self.voxel).values, self.voxel.grain_area())
        diameter = read('diameter', self.voxel).values
        np.testing.assert_allclose(np.pi / 6 * diameter ** 3, self.voxel.volume)
        self.assertEqual(read('line_length', self.voxel).count, len(self.voxel.lines['length']))
        self.assertEqual(read('line_length_per_grain', self.voxel).count, len(self.voxel.lines['per_grain']))

    def test_an_open_surface_has_no_volume_but_has_everything_else(self):
        for key in ('grain_volume', 'diameter', 'aspect'):
            with self.assertRaises(Unavailable):
                read(key, self.open)
            self.assertIn('no grain volume', available(key, self.open))
        for key in ('coordination', 'grain_area', 'line_length', 'misorientation_boundary', 'misorientation_grain'):
            self.assertIsNone(available(key, self.open), key)

    def test_the_boundary_misorientation_weights(self):
        fraction = read('misorientation_boundary', self.voxel, normalisation=AREA_FRACTION)
        self.assertAlmostEqual(fraction.weights.sum(), 1.0)
        self.assertEqual(fraction.y_kind, 'fraction')
        per_volume = read('misorientation_boundary', self.voxel, normalisation=AREA_PER_VOLUME)
        np.testing.assert_allclose(per_volume.weights, fraction.weights * fraction.total / self.voxel.rve_volume)
        self.assertEqual(per_volume.y_kind, 'per_length')
        absolute = read('misorientation_boundary', self.voxel, normalisation=AREA_ABSOLUTE)
        self.assertAlmostEqual(absolute.weights.sum(), fraction.total)
        with self.assertRaises(ValueError):
            read('misorientation_boundary', self.voxel, normalisation='per grain')

    def test_misorientation_needs_cubic_symmetry(self):
        hexagonal = voxel_snapshot(self.hier, ComparisonSettings(symmetry='hexagonal'))
        for key in ('misorientation_boundary', 'misorientation_grain'):
            with self.assertRaises(Unavailable):
                read(key, hexagonal)

    def test_the_grain_misorientation_can_be_weighted_by_volume(self):
        counted = read('misorientation_grain', self.voxel)
        self.assertIsNone(counted.weights)
        by_volume = read('misorientation_grain', self.voxel, grain_weight=BY_VOLUME)
        self.assertAlmostEqual(by_volume.weights.sum(), 1.0)
        self.assertEqual(by_volume.y_kind, 'fraction')
        without = read('misorientation_grain', self.open, grain_weight=BY_VOLUME)
        self.assertIsNone(without.weights)
        self.assertIn('no grain volume', without.note)
        with self.assertRaises(ValueError):
            read('misorientation_grain', self.voxel, grain_weight='size')

    def test_matched_items(self):
        before, after = read('grain_volume', self.voxel), read('grain_volume', self.tet)
        keys, b, a = matched_items(before, after)
        self.assertEqual(keys, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(b, a)
        del after.items[3]
        self.assertEqual(matched_items(before, after)[0], [1, 2, 4, 5])
        sets = matched_items(read('line_length', self.voxel), read('line_length', self.tet))[0]
        self.assertTrue(all(isinstance(k, tuple) for k in sets))
        self.assertEqual(matched_items(read('line_length', self.voxel), read('grain_volume', self.tet))[0], [])

    def test_units(self):
        self.assertEqual(unit_power('length'), 1)
        self.assertEqual(unit_power('volume'), 3)
        self.assertEqual(unit_power('angle'), 0)
        self.assertAlmostEqual(unit_factor(1, 'mm', 'microns'), 1000.0)
        self.assertAlmostEqual(unit_factor(2, 'mm', 'microns'), 1e6)
        self.assertAlmostEqual(unit_factor(3, 'microns', 'mm'), 1e-9)
        self.assertAlmostEqual(unit_factor(-1, 'mm', 'microns'), 1e-3)
        self.assertEqual(unit_factor(0, 'mm', 'm'), 1.0)
        self.assertEqual(DISPLAY_UNITS, ('auto', 'microns', 'mm', 'm'))
        self.assertEqual(unit_text(2, 'mm', plain=True), 'mm^2')
        self.assertEqual(unit_text(1, 'microns', plain=True), 'microns')
        self.assertEqual(unit_text(2, 'microns'), r'$\mu$m$^2$')
        self.assertEqual(unit_text(0, 'mm'), '')
        self.assertEqual(count_noun('grains', 1), '1 grain')
        self.assertEqual(count_noun('grains', 271), '271 grains')

    def test_auto_units_choose_the_fewest_digits(self):
        self.assertEqual(choose_units(1, 'mm', [np.array([0.0075, 0.009, 0.006])]), 'microns')
        self.assertEqual(choose_units(1, 'mm', [np.array([5.0, 8.0])]), 'mm')
        self.assertEqual(choose_units(1, 'microns', [np.array([5000.0, 8000.0])]), 'mm')
        self.assertEqual(choose_units(2, 'mm', [np.array([0.00004])]), 'microns')
        self.assertEqual(choose_units(0, 'mm', [np.array([3.0])]), 'mm')
        self.assertEqual(choose_units(1, 'mm', [np.zeros(0)]), 'mm')


if __name__ == '__main__':
    unittest.main()
