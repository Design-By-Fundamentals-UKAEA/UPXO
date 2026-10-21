"""The feature table, hierarchy ids in the mesh files, boundary-layer element sets and the deck check."""
import csv
import json
import shutil
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    DeckSettings, load_hierarchy, resolve_identities, write_deck)
from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import check_deck, format_check, read_sets
from upxo.meshing.gbconformant.fm_steel_abaqus.features import id_columns, tet_volumes
from .helpers import read_elsets, voxel_tets, work_dir, write_raw_folder, write_twinned_folder
from .test_deck import build
from .test_outputs import read_nsets

EXTRAS = DeckSettings(boundary_sets=True, layer_sets=True, feature_table=True, deck_check=True,
                      write_msh=True, write_vtu=True)


def read_table(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


class FeatureTableTests(unittest.TestCase):
    def test_one_row_per_feature_with_names_ids_and_mesh_numbers(self):
        with work_dir() as root:
            hier, _, (pts, tets, grains), res = build(root, settings=EXTRAS)
            rows = read_table(res['directory'] / 'features.csv')
            self.assertEqual(len(rows), 5)
            row = next(r for r in rows if r['elset'] == 'es_blk_1_2_1')
            self.assertEqual((row['pag_id'], row['packet_id'], row['block_id'], row['subblock_id']), ('1', '2', '3', ''))
            self.assertEqual((row['pag_elset'], row['packet_elset'], row['block_elset']),
                             ('es_pag_1', 'es_pck_1_2', 'es_blk_1_2_1'))
            self.assertEqual(row['material'], 'MAT_BLK_1_2_1')
            label = next(l for l, f in hier.features.items() if f.global_id == int(row['umat_grain_number']))
            self.assertEqual(int(row['tetrahedra']), int((grains == label).sum()))
            self.assertEqual(int(row['voxels_raw']), hier.voxel_counts[label])
            np.testing.assert_allclose([float(row[c]) for c in ('phi1_deg', 'Phi_deg', 'phi2_deg')],
                                       hier.features[label].euler, atol=1e-5)

    def test_volumes_add_up_to_the_mesh_in_output_units(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=EXTRAS)
            rows = read_table(res['directory'] / 'features.csv')
            scale = res['info']['coordinate_scale']
            total = float(tet_volumes(pts, tets).sum()) * scale ** 3
            self.assertAlmostEqual(sum(float(r['volume']) for r in rows) / total, 1.0, places=6)
            self.assertAlmostEqual(sum(float(r['volume_fraction']) for r in rows), 1.0, places=6)
            self.assertEqual(sum(int(r['tetrahedra']) for r in rows), len(tets))

    def test_the_table_is_not_written_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            self.assertFalse((res['directory'] / 'features.csv').exists())
            self.assertIsNone(res['info']['feature_table'])

    def test_hierarchy_ids_number_the_levels_from_one(self):
        with work_dir() as root:
            (root / 'raw').mkdir()
            write_raw_folder(root / 'raw')
            hier = load_hierarchy(root / 'raw', 'blk')
            ids = id_columns(hier)
            self.assertEqual(sorted(set(ids['packet_id'].values())), [1, 2, 3])
            self.assertEqual(sorted(set(ids['block_id'].values())), [1, 2, 3, 4, 5])
            self.assertEqual(set(ids['subblock_id'].values()), {-1})
            self.assertEqual(sorted(set(ids['pag_id'].values())), [1, 2])

    def test_a_pag_level_structure_has_no_packet_or_block_ids(self):
        with work_dir() as root:
            (root / 'raw').mkdir()
            write_raw_folder(root / 'raw')
            ids = id_columns(load_hierarchy(root / 'raw', 'pag'))
            self.assertEqual(set(ids['packet_id'].values()), {-1})
            self.assertEqual(set(ids['block_id'].values()), {-1})

    def test_the_mesh_files_carry_the_ids(self):
        import pyvista as pv
        with work_dir() as root:
            hier, _, (pts, tets, grains), res = build(root, settings=EXTRAS)
            grid = pv.read(str(res['directory'] / 'fm_mesh.vtu'))
            ids = id_columns(hier)
            for name in ('pag_id', 'packet_id', 'block_id', 'subblock_id'):
                expected = np.array([ids[name][int(l)] for l in grains])
                np.testing.assert_array_equal(grid.cell_data[name], expected)
            msh = (res['directory'] / 'fm_mesh.msh').read_text()
            for name in ('pag_id', 'packet_id', 'block_id', 'subblock_id'):
                self.assertIn(f'"{name}"', msh)

    def test_a_twinned_structure_lists_role_and_family(self):
        with work_dir() as root:
            folder = root / 'twinned'
            folder.mkdir()
            labels, roles, parents, quats = write_twinned_folder(folder)
            from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import load_twinned
            hier = load_twinned(folder)
            points, tets, grains = voxel_tets(labels)
            res = write_deck(root / 'deck', points, tets, grains, hier, resolve_identities(hier.voxel_counts),
                             EXTRAS, meshed_labels=labels)
            rows = {r['elset']: r for r in read_table(res['directory'] / 'features.csv')}
            self.assertEqual((rows['es_cell_stwin_b_5']['role'], rows['es_cell_stwin_b_5']['family_id'],
                              rows['es_cell_stwin_b_5']['family_elset']), ('stwin_b', '2', 'es_family_2'))
            self.assertEqual(rows['es_cell_nonpart_6']['family_id'], '-1')
            self.assertEqual(rows['es_cell_nonpart_6']['family_elset'], '')
            self.assertNotIn('pag_id', rows['es_cell_host_2'])
            import pyvista as pv
            grid = pv.read(str(res['directory'] / 'fm_mesh.vtu'))
            self.assertEqual(set(grid.cell_data['role_code'].tolist()), {0, 1, 2, 4})
            self.assertEqual(set(grid.cell_data['family_id'].tolist()), {-1, 2, 4})


class LayerSetTests(unittest.TestCase):
    def test_the_layer_is_every_element_with_a_node_on_a_grain_boundary(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=EXTRAS)
            deck = res['directory']
            gb = set(read_nsets(deck / '04b_nsets_boundaries.inp')['ns_gb_all'])
            sets = read_elsets(deck / '03f_elsets_layers.inp')
            expected = [i + 1 for i, tet in enumerate(tets) if any(n + 1 in gb for n in tet)]
            self.assertEqual(sets['es_gb_layer'], expected)
            self.assertEqual(sorted(sets['es_gb_layer'] + sets['es_interior']), list(range(1, len(tets) + 1)))
            info = res['info']['layer_sets']
            self.assertEqual(info['gb_layer_elements'], len(expected))
            self.assertAlmostEqual(info['gb_layer_fraction'], len(expected) / len(tets))

    def test_layer_sets_work_without_the_node_sets_file(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=DeckSettings(layer_sets=True))
            deck = res['directory']
            self.assertFalse((deck / '04b_nsets_boundaries.inp').exists())
            self.assertTrue((deck / '03f_elsets_layers.inp').is_file())
            self.assertNotIn('04b_nsets', (deck / 'model_master.inp').read_text())
            self.assertIn('03f_elsets_layers.inp', (deck / 'model_master.inp').read_text())

    def test_one_feature_has_no_layer(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=EXTRAS, merge={2: 1, 3: 1, 4: 1, 5: 1})
            sets = read_elsets(res['directory'] / '03f_elsets_layers.inp')
            self.assertEqual(list(sets), ['es_interior'])
            self.assertEqual(res['info']['layer_sets']['gb_layer_elements'], 0)

    def test_nothing_is_written_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            self.assertFalse((res['directory'] / '03f_elsets_layers.inp').exists())
            self.assertIsNone(res['info']['layer_sets'])


class DeckCheckTests(unittest.TestCase):
    def deck(self, root, **kwargs):
        _, _, _, res = build(root, settings=EXTRAS, **kwargs)
        return res['directory'], res

    def altered(self, deck, root, filename, change):
        copy = root / ('copy_' + filename.replace('.', '_'))
        shutil.copytree(deck, copy)
        path = copy / filename
        path.write_text(change(path.read_text(encoding='ascii')), encoding='ascii')
        return check_deck(copy)

    @staticmethod
    def failed(result):
        return set(result['failed'])

    def test_a_fresh_deck_passes_every_check(self):
        with work_dir() as root:
            deck, res = self.deck(root)
            result = check_deck(deck)
            self.assertTrue(result['ok'], format_check(result))
            self.assertEqual(result['failed'], [])
            self.assertEqual(res['info']['deck_check'], {'ok': True, 'failed': []})
            saved = json.loads((deck / 'deck_check.json').read_text())
            self.assertTrue(saved['ok'])
            self.assertEqual(saved['counts']['elements'], res['info']['tetrahedra'])
            self.assertEqual(saved['counts']['materials'], 5)

    def test_the_check_is_not_run_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            self.assertFalse((res['directory'] / 'deck_check.json').exists())
            self.assertIsNone(res['info']['deck_check'])

    def test_twinned_and_merged_decks_pass(self):
        with work_dir() as root:
            _, res = self.deck(root, merge={2: 1})
            self.assertTrue(check_deck(res['directory'])['ok'])

    def test_an_element_missing_from_its_set_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)

            def drop_first_id(text):
                lines = text.splitlines()
                at = next(i for i, l in enumerate(lines) if l.startswith('*Elset')) + 1
                lines[at] = ', '.join(lines[at].split(', ')[1:])
                return '\n'.join(lines) + '\n'

            result = self.altered(deck, root, '03c_elsets_blk.inp', drop_first_id)
            self.assertIn('each element is in exactly one section set', self.failed(result))
            self.assertFalse(result['ok'])

    def test_an_element_in_two_sets_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            extra = '*Elset, elset=es_blk_2_1_2\n1, \n'
            result = self.altered(deck, root, '03c_elsets_blk.inp', lambda t: t + extra.replace('es_blk_2_1_2', 'es_blk_1_2_1'))
            self.assertIn('each element is in exactly one section set', self.failed(result))

    def test_a_section_with_no_material_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            result = self.altered(deck, root, '05_materials.inp',
                                  lambda t: t.replace('*Material, name=MAT_BLK_1_1_1', '*Material, name=MAT_OTHER', 1))
            self.assertIn('sections name existing materials', self.failed(result))
            self.assertIn('every material is used', self.failed(result))

    def test_an_empty_node_set_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)

            def empty_first(text):
                out, skipping, done = [], False, False
                for line in text.splitlines():
                    if line.startswith('*Nset') and not done:
                        out.append(line)
                        skipping = done = True
                    elif line.startswith('*'):
                        skipping = False
                        out.append(line)
                    elif not skipping:
                        out.append(line)
                return '\n'.join(out) + '\n'

            result = self.altered(deck, root, '04_nsets_bc.inp', empty_first)
            self.assertIn('no empty sets', self.failed(result))

    def test_a_flipped_tetrahedron_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)

            def flip_first(text):
                lines = text.splitlines()
                parts = lines[1].split(',')
                parts[2], parts[3] = parts[3], parts[2]
                lines[1] = ','.join(parts)
                return '\n'.join(lines) + '\n'

            result = self.altered(deck, root, '02_elements.inp', flip_first)
            self.assertIn('every tetrahedron has a positive volume', self.failed(result))

    def test_a_node_number_gap_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            result = self.altered(deck, root, '01_nodes.inp', lambda t: t.replace('         2,', '        99,', 1))
            self.assertIn('nodes numbered 1..N', self.failed(result))

    def test_a_missing_include_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            copy = root / 'missing_include'
            shutil.copytree(deck, copy)
            (copy / '03f_elsets_layers.inp').unlink()
            result = check_deck(copy)
            self.assertIn('included files exist', self.failed(result))

    def test_a_step_that_names_a_missing_node_set_is_found(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            result = self.altered(deck, root, '08_steps_output.inp', lambda t: t.replace('ns_zplus', 'ns_nowhere'))
            self.assertIn('step boundary conditions use existing node sets', self.failed(result))

    def test_a_missing_file_is_reported_not_raised(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            copy = root / 'no_nodes'
            shutil.copytree(deck, copy)
            (copy / '01_nodes.inp').unlink()
            result = check_deck(copy)
            self.assertFalse(result['ok'])
            self.assertIn('01_nodes.inp present', self.failed(result))

    def test_the_text_form_has_a_line_per_check(self):
        with work_dir() as root:
            deck, _ = self.deck(root)
            result = check_deck(deck)
            self.assertEqual(len(format_check(result).splitlines()), len(result['checks']))
            self.assertTrue(all(line.startswith('[ok]') for line in format_check(result).splitlines()))

    def test_read_sets_reads_blocks_and_ignores_comments(self):
        with work_dir() as root:
            path = root / 'sets.inp'
            path.write_text('** comment\n*Elset, elset=A\n1, 2, 3,\n4\n*Elset, elset=B\n9\n*Nset, nset=N, instance=X\n7, 8\n')
            self.assertEqual({k: v.tolist() for k, v in read_sets(path, 'Elset').items()}, {'A': [1, 2, 3, 4], 'B': [9]})
            self.assertEqual({k: v.tolist() for k, v in read_sets(path, 'Nset').items()}, {'N': [7, 8]})


if __name__ == '__main__':
    unittest.main()
