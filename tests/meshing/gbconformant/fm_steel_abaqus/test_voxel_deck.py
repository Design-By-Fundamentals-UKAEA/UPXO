"""The voxel (C3D8) deck: hexahedron geometry, label resampling, tet voxelization and the deck itself."""
import json
import unittest

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, load_hierarchy, resolve_identities, write_deck
from upxo.meshing.gbconformant.fm_steel_abaqus.boundary_sets import boundary_nodes
from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import read_sets
from upxo.meshing.gbconformant.fm_steel_abaqus.elements import (
    HEX8, element_faces, element_kind, element_volumes, face_areas, grid_hexahedra)
from upxo.meshing.gbconformant.fm_steel_abaqus.interfaces import interface_faces
from upxo.meshing.gbconformant.fm_steel_abaqus.junctions import tet_junction_edges, voxel_junction_edges
from upxo.meshing.gbconformant.fm_steel_abaqus.voxel_deck import (
    label_hexahedra, resample_labels, voxelize_tetrahedra, write_voxel_deck)
from upxo.meshing.gbconformant.pipeline.mesh_regions import (
    GB_SURFACE, JUNCTION_LINES, JUNCTION_POINTS, feature_layers)
from .helpers import voxel_tets, work_dir, write_raw_folder

ALL_SETS = dict(boundary_sets=True, layer_sets=True, boundary_classes=True, feature_table=True, deck_check=True,
                write_msh=True, write_vtu=True)


def _labels():
    a = np.ones((4, 5, 3), dtype=np.int64)
    a[2:, :, :] = 2
    a[:, 3:, :] = 3
    a[2:, 3:, 2] = 4
    return a


def _set_sizes(folder):
    out = {}
    for path in sorted(folder.glob('0[34]*.inp')):
        out.update({name: len(ids) for name, ids in read_sets(path, 'Elset').items()})
        out.update({name: len(ids) for name, ids in read_sets(path, 'Nset').items()})
    return out


class HexahedronGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points, self.hexes = grid_hexahedra((2, 3, 4), (1.0, 2.0, 0.5))

    def test_every_voxel_has_its_volume_and_c3d8_numbering(self):
        np.testing.assert_allclose(element_volumes(self.points, self.hexes), 1.0)
        self.assertIs(element_kind(self.hexes), HEX8)
        corner = self.points[self.hexes[:, 0]]
        np.testing.assert_allclose(corner[1], (0.0, 0.0, 0.5))     # element 1 is voxel (0, 0, 1): C order

    def test_faces_point_out_and_are_shared_once(self):
        centre = self.points[self.hexes].mean(axis=1)
        for corners in HEX8.faces:
            quad = self.points[self.hexes[:, corners]]
            normal = np.cross(quad[:, 1] - quad[:, 0], quad[:, 2] - quad[:, 0])
            self.assertTrue((np.einsum('ij,ij->i', normal, quad.mean(axis=1) - centre) > 0).all())
        nodes, first, second, count = element_faces(self.hexes, len(self.points))
        self.assertEqual(len(nodes), 3 * 3 * 4 + 2 * 4 * 4 + 2 * 3 * 5)
        self.assertEqual(int(count.max()), 2)
        # faces across x (area 2.0 * 0.5), across y (1.0 * 0.5) and across z (1.0 * 2.0)
        np.testing.assert_allclose(face_areas(self.points, nodes).sum(), 36 * 1.0 + 32 * 0.5 + 30 * 2.0)

    def test_inverted_hexahedra_have_negative_volume(self):
        flipped = self.hexes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        self.assertTrue((element_volumes(self.points, flipped) < 0).all())


class SharedRegionCodeTests(unittest.TestCase):
    """The region and junction code used for tetrahedra gives the voxel answer on hexahedra."""

    def test_junction_edges_match_the_voxel_algorithm(self):
        labels = _labels()
        points, hexes, grains = label_hexahedra(labels)
        for include in (False, True):
            on_hexes = tet_junction_edges(points, hexes, grains, include)
            on_voxels = voxel_junction_edges(labels, include)
            self.assertEqual(len(on_hexes.nodes), len(on_voxels.nodes))
            self.assertEqual(sorted(map(tuple, on_hexes.sets)), sorted(map(tuple, on_voxels.sets)))

    def test_interfaces_are_the_voxel_faces_between_grains(self):
        labels = _labels()
        points, hexes, grains = label_hexahedra(labels)
        found = interface_faces(points, hexes, grains)
        expected = sum(int((np.diff(labels, axis=ax) != 0).sum()) for ax in range(3))
        self.assertEqual(found.faces.shape, (expected, 4))
        np.testing.assert_allclose(found.area, 1.0)

    def test_layers_grow_from_the_voxel_boundaries(self):
        labels = _labels()
        points, hexes, grains = label_hexahedra(labels)
        layers = feature_layers(points, hexes, grains, order=2, touch='face or edge')
        on_boundary = np.zeros(labels.shape, dtype=bool)
        for ax in range(3):
            step = np.diff(labels, axis=ax) != 0
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(None, -1), slice(1, None)
            on_boundary[tuple(lo)] |= step
            on_boundary[tuple(hi)] |= step
        np.testing.assert_array_equal(layers.tet[GB_SURFACE].reshape(labels.shape) == 1, on_boundary)
        self.assertTrue((layers.tet[GB_SURFACE] == 2).any())
        for feature in (JUNCTION_LINES, JUNCTION_POINTS):
            self.assertTrue((layers.tet[feature] == 1).any())
        self.assertEqual(layers.surface.nodes.shape[1], 4)

    def test_boundary_nodes_accept_hexahedra(self):
        points, hexes, grains = label_hexahedra(_labels())
        nodes = boundary_nodes(hexes, grains, len(points))
        self.assertGreater(len(nodes.grain_boundary), len(nodes.junction))


class ResampleTests(unittest.TestCase):
    def test_factor_one_keeps_the_labels(self):
        labels = _labels()
        out, report = resample_labels(labels, 1)
        np.testing.assert_array_equal(out, labels)
        self.assertEqual(report['labels_lost'], [])

    def test_refining_repeats_each_voxel(self):
        labels = _labels()
        out, report = resample_labels(labels, 2)
        self.assertEqual(out.shape, (8, 10, 6))
        np.testing.assert_array_equal(out[::2, ::2, ::2], labels)
        np.testing.assert_array_equal(out[1::2, 1::2, 1::2], labels)

    def test_coarsening_takes_the_majority_and_reports_lost_labels(self):
        labels = np.full((4, 4, 4), 1, dtype=np.int64)
        labels[0, 0, 0] = 3                          # one voxel of eight: outvoted
        labels[2:, 2:, 2:] = 2
        labels[2, 2, 2] = 5
        out, report = resample_labels(labels, 0.5)
        self.assertEqual(out.shape, (2, 2, 2))
        self.assertEqual(int(out[0, 0, 0]), 1)
        self.assertEqual(int(out[1, 1, 1]), 2)
        self.assertEqual(report['labels_lost'], [3, 5])

    def test_a_tie_goes_to_the_smaller_label(self):
        labels = np.full((2, 2, 2), 7, dtype=np.int64)
        labels[:, :, 1] = 4
        out, _ = resample_labels(labels, 0.5)
        self.assertEqual(int(out[0, 0, 0]), 4)


class VoxelizeTests(unittest.TestCase):
    def test_the_tet_mesh_of_a_voxel_grid_voxelizes_back_to_it(self):
        labels = _labels()
        points, tets, grains = voxel_tets(labels)
        out, report = voxelize_tetrahedra(points, tets, grains, 1.0)
        np.testing.assert_array_equal(out, labels)
        self.assertEqual(report['voxels_filled_by_nearest'], 0)
        self.assertEqual(report['grains_lost'], [])

    def test_a_finer_grid_keeps_every_grain(self):
        labels = _labels()
        points, tets, grains = voxel_tets(labels)
        out, report = voxelize_tetrahedra(points, tets, grains, 0.5)
        self.assertEqual(out.shape, (8, 10, 6))
        np.testing.assert_array_equal(out[::2, ::2, ::2], labels)


class VoxelDeckTests(unittest.TestCase):
    def build(self, root):
        raw = root / 'raw'
        raw.mkdir()
        write_raw_folder(raw, with_sblk=True)
        hier = load_hierarchy(raw, 'sblk')
        ident = resolve_identities(hier.voxel_counts)
        return hier, ident, hier.lgi.copy()

    def test_the_voxel_deck_passes_its_check_and_has_the_tet_decks_sets(self):
        with work_dir() as root:
            hier, ident, labels = self.build(root)
            voxel = write_voxel_deck(root / 'voxel', labels, hier, ident, DeckSettings(**ALL_SETS),
                                     meshed_labels=labels, source={'kind': 'cleaned labels'})
            points, tets, grains = voxel_tets(labels)
            tet = write_deck(root / 'tet', points, tets, grains, hier, ident, DeckSettings(**ALL_SETS),
                             meshed_labels=labels)
            check = json.loads((root / 'voxel' / 'deck_check.json').read_text())
            self.assertTrue(check['ok'], check['failed'])
            self.assertEqual(check['counts']['element_type'], 'C3D8')
            self.assertEqual(sorted(voxel['files']), sorted(tet['files']))
            elements = (root / 'voxel' / '02_elements.inp').read_text().splitlines()
            self.assertEqual(elements[0], '*Element, type=C3D8')
            self.assertEqual(len(elements) - 1, labels.size)
            sizes_voxel, sizes_tet = _set_sizes(root / 'voxel'), _set_sizes(root / 'tet')
            self.assertEqual(set(sizes_voxel), set(sizes_tet))
            # the tet mesh splits each voxel into six on the same nodes: grain sets and node sets agree
            for name, size in sizes_voxel.items():
                if name.startswith('ns_'):
                    self.assertEqual(size, sizes_tet[name], name)
                elif name.startswith(('es_pag_', 'es_pck_', 'es_blk_', 'es_subblk_')) \
                        and name[-1].isdigit():
                    self.assertEqual(6 * size, sizes_tet[name], name)
            info = json.loads((root / 'voxel' / 'deck_info.json').read_text())
            self.assertEqual(info['element_type'], 'C3D8')
            self.assertEqual(info['hexahedra'], labels.size)
            self.assertEqual(info['voxel_source'], {'kind': 'cleaned labels'})
            self.assertEqual(info['voxel_grid']['shape'], list(labels.shape))
            header = (root / 'voxel' / 'features.csv').read_text().splitlines()[0]
            self.assertIn('hexahedra', header)
            self.assertIn('C3D8 (voxel hexahedra)', (root / 'voxel' / 'model_master.inp').read_text())

    def test_a_resampled_grid_keeps_the_box_and_passes_its_check(self):
        with work_dir() as root:
            hier, ident, labels = self.build(root)
            fine, _ = resample_labels(labels, 2)
            write_voxel_deck(root / 'fine', fine, hier, ident, DeckSettings(deck_check=True), voxel_size=0.5,
                             meshed_labels=labels)
            check = json.loads((root / 'fine' / 'deck_check.json').read_text())
            self.assertTrue(check['ok'], check['failed'])
            info = json.loads((root / 'fine' / 'deck_info.json').read_text())
            self.assertEqual(info['elements'], fine.size)
            np.testing.assert_allclose(info['total_volume_mesh_units'], labels.size)

    def test_a_twinned_structure_gets_the_same_sets_as_its_tet_deck(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import load_twinned
        from .helpers import write_twinned_folder

        with work_dir() as root:
            raw = root / 'raw'
            raw.mkdir()
            write_twinned_folder(raw)
            hier = load_twinned(raw)
            ident = resolve_identities(hier.voxel_counts)
            labels = hier.lgi.copy()
            write_voxel_deck(root / 'voxel', labels, hier, ident, DeckSettings(**ALL_SETS), meshed_labels=labels)
            points, tets, grains = voxel_tets(labels)
            write_deck(root / 'tet', points, tets, grains, hier, ident, DeckSettings(**ALL_SETS), meshed_labels=labels)
            check = json.loads((root / 'voxel' / 'deck_check.json').read_text())
            self.assertTrue(check['ok'], check['failed'])
            self.assertEqual(set(_set_sizes(root / 'voxel')), set(_set_sizes(root / 'tet')))
            table = (root / 'voxel' / 'boundary_classes.csv').read_text().splitlines()
            self.assertIn('quadrilaterals', table[0])
            self.assertTrue(any(',twin,' in row for row in table[1:]))


if __name__ == '__main__':
    unittest.main()
