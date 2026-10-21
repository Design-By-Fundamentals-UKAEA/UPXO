"""Twinned FCC folders read into the deck writer."""
import pickle
import re
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, resolve_identities, write_deck
from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import (
    host_family, is_twinned_folder, load_twinned, quaternion_to_bunge, role_keys)
from .helpers import TWINNED_SHAPE, read_elsets, voxel_tets, work_dir, write_twinned_folder


def loaded(root, **kwargs):
    folder = root / 'twinned'
    folder.mkdir()
    labels, roles, parents, quats = write_twinned_folder(folder)
    return folder, load_twinned(folder, **kwargs), labels, roles, parents, quats


class LoadTests(unittest.TestCase):
    def test_the_structure_is_read_as_x_y_z_with_one_feature_per_grain(self):
        with work_dir() as root:
            folder, hierarchy, labels, *_ = loaded(root, voxel_size=2.0, units='mm')
            self.assertTrue(is_twinned_folder(folder))
            self.assertEqual(hierarchy.lgi.shape, TWINNED_SHAPE)
            np.testing.assert_array_equal(hierarchy.lgi, labels)
            self.assertEqual(sorted(hierarchy.features), [2, 3, 4, 5, 6])
            self.assertEqual((hierarchy.voxel_size, hierarchy.units, hierarchy.level), (2.0, 'mm', 'pag'))
            self.assertEqual(hierarchy.system, 'twinned FCC')
            self.assertEqual(hierarchy.voxel_counts[3], int((labels == 3).sum()))

    def test_grains_are_named_as_the_voxel_deck_names_them(self):
        with work_dir() as root:
            _, hierarchy, *_ = loaded(root)
            self.assertEqual(hierarchy.names, {2: 'es_cell_host_2', 3: 'es_cell_ptwin_3', 4: 'es_cell_host_4',
                                               5: 'es_cell_stwin_b_5', 6: 'es_cell_nonpart_6'})
            self.assertEqual(hierarchy.material_names[5], 'MAT_CELL_STWIN_B_5')
            self.assertEqual([f.global_id for f in hierarchy.ordered_features()], [1, 2, 3, 4, 5])

    def test_orientations_are_bunge_angles_in_the_positive_range(self):
        with work_dir() as root:
            _, hierarchy, _labels, _roles, _parents, quats = loaded(root)
            for label, feature in hierarchy.features.items():
                self.assertTrue(all(0.0 <= a < 360.0 for a in feature.euler))
                expected = [a % 360.0 for a in quaternion_to_bunge(quats[label])]
                np.testing.assert_allclose(feature.euler, expected)

    def test_the_conversion_agrees_with_the_voxel_exporter(self):
        from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import _quat_to_bunge
        rng = np.random.default_rng(1)
        for _ in range(40):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            np.testing.assert_allclose(quaternion_to_bunge(q), _quat_to_bunge(q), atol=1e-9)
        for q in ([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]):
            np.testing.assert_allclose(quaternion_to_bunge(q), _quat_to_bunge(np.array(q, float)), atol=1e-9)

    def test_roles_and_families_are_joined_into_groups(self):
        with work_dir() as root:
            _, hierarchy, *_ = loaded(root)
            groups = hierarchy.groups
            self.assertEqual(groups['es_role_host'], (2, 4))
            self.assertEqual(groups['es_role_ptwin'], (3,))
            self.assertEqual(groups['es_role_stwin_b'], (5,))
            self.assertEqual(groups['es_role_stwin'], (5,))
            self.assertEqual(groups['es_role_twin'], (3, 5))
            self.assertEqual(groups['es_role_nonpart'], (6,))
            self.assertEqual(groups['es_family_2'], (2, 3, 5))
            self.assertEqual(groups['es_family_4'], (4,))
            self.assertNotIn('es_family_6', groups)

    def test_a_secondary_twin_of_a_host_is_the_outward_kind(self):
        roles = {1: 'host', 2: 'secondary_twin', 3: 'primary_twin', 4: 'secondary_twin'}
        keys = role_keys(roles, {2: 1, 4: 3})
        self.assertEqual((keys[2], keys[4]), ('stwin_a', 'stwin_b'))

    def test_a_twin_whose_host_is_missing_has_no_family(self):
        self.assertIsNone(host_family(9, {9: 8}, hosts={1}))
        self.assertEqual(host_family(1, {}, hosts={1}), 1)
        self.assertEqual(host_family(3, {3: 2, 2: 1}, hosts={1}), 1)
        self.assertIsNone(host_family(3, {3: 2, 2: 3}, hosts={1}))              # a loop ends

    def test_bad_folders_are_refused(self):
        with work_dir() as root:
            empty = root / 'empty'
            empty.mkdir()
            self.assertFalse(is_twinned_folder(empty))
            with self.assertRaises(FileNotFoundError):
                load_twinned(empty)
            folder, *_ = loaded(root)
            with self.assertRaises(ValueError):
                load_twinned(folder, voxel_size=0)
            with (folder / 'twin_role.pkl').open('wb') as handle:
                pickle.dump({2: 'host', 3: 'unknown_role', 4: 'host', 5: 'host', 6: 'host'}, handle)
            with self.assertRaises(ValueError):
                load_twinned(folder)

    def test_a_grain_without_an_orientation_is_refused(self):
        with work_dir() as root:
            folder, *_ = loaded(root)
            with (folder / 'orientations_all.pkl').open('wb') as handle:
                pickle.dump({2: np.array([1., 0, 0, 0])}, handle)
            with self.assertRaisesRegex(ValueError, 'no orientation'):
                load_twinned(folder)

    def test_a_void_label_is_refused(self):
        with work_dir() as root:
            folder, *_ = loaded(root)
            array = np.load(folder / 'lgi_twinned_clean.npy')
            array[0, 0, 0] = 0
            np.save(folder / 'lgi_twinned_clean.npy', array)
            with self.assertRaises(NotImplementedError):
                load_twinned(folder)


class DeckTests(unittest.TestCase):
    def deck(self, root, **settings):
        _, hierarchy, labels, roles, parents, quats = loaded(root)
        identity = resolve_identities(hierarchy.voxel_counts)
        points, tets, grains = voxel_tets(labels)
        result = write_deck(root / 'deck', points, tets, grains, hierarchy, identity, DeckSettings(**settings),
                            meshed_labels=labels)
        return result, hierarchy, tets, grains

    def test_element_sets_follow_the_voxel_deck_naming(self):
        with work_dir() as root:
            result, hierarchy, tets, grains = self.deck(root)
            deck = result['directory']
            self.assertTrue((deck / '03a_elsets_pag.inp').is_file())
            self.assertFalse((deck / '03b_elsets_pck.inp').exists())
            cells = read_elsets(deck / '03a_elsets_pag.inp')
            self.assertEqual(set(cells), set(hierarchy.names.values()))
            for label, name in hierarchy.names.items():
                self.assertEqual(cells[name], (np.flatnonzero(grains == label) + 1).tolist())

    def test_group_sets_join_the_member_grains(self):
        with work_dir() as root:
            result, _hierarchy, _tets, grains = self.deck(root)
            groups = read_elsets(result['directory'] / '03e_elsets_groups.inp')
            expect = lambda labels: np.flatnonzero(np.isin(grains, labels)) + 1
            self.assertEqual(groups['es_family_2'], expect([2, 3, 5]).tolist())
            self.assertEqual(groups['es_role_twin'], expect([3, 5]).tolist())
            self.assertEqual(groups['es_role_host'], expect([2, 4]).tolist())
            self.assertIn('03e_elsets_groups.inp', (result['directory'] / 'model_master.inp').read_text())

    def test_sections_link_each_grain_to_its_material(self):
        with work_dir() as root:
            result, hierarchy, *_ = self.deck(root)
            text = (result['directory'] / '06_sections.inp').read_text()
            links = dict(re.findall(r'elset=(\S+), material=(\S+)', text))
            self.assertEqual(links['es_cell_ptwin_3'], 'MAT_CELL_PTWIN_3')
            self.assertEqual(len(links), 5)
            materials = (result['directory'] / '05_materials.inp').read_text()
            for material in links.values():
                self.assertIn(f'*Material, name={material}', materials)

    def test_materials_carry_the_orientations_and_grain_numbers(self):
        with work_dir() as root:
            result, hierarchy, *_ = self.deck(root)
            text = (result['directory'] / '05_materials.inp').read_text()
            block = text.split('*Material, name=MAT_CELL_STWIN_B_5')[1].split('*Material')[0]
            numbers = [float(v) for v in re.search(r'constants=6\n(.*)\n', block).group(1).split(',')]
            np.testing.assert_allclose(numbers[:3], hierarchy.features[5].euler, atol=1e-5)
            self.assertEqual(numbers[3], 4.0)                          # the fourth grain in label order
            self.assertNotIn('phase_id', text)

    def test_headings_name_the_structure(self):
        with work_dir() as root:
            result, *_ = self.deck(root)
            self.assertIn('twinned FCC microstructure', (result['directory'] / 'model_master.inp').read_text())
            self.assertEqual(result['info']['system'], 'twinned FCC')

    def test_boundary_sets_and_mesh_files_use_the_grain_names(self):
        with work_dir() as root:
            result, *_ = self.deck(root, boundary_sets=True, write_msh=True, write_vtu=True)
            text = (result['directory'] / '04b_nsets_boundaries.inp').read_text()
            self.assertIn('nset=ns_gb_es_cell_host_2,', text)
            msh = (result['directory'] / 'fm_mesh.msh').read_text()
            self.assertIn('"es_cell_stwin_b_5"', msh)


if __name__ == '__main__':
    unittest.main()
