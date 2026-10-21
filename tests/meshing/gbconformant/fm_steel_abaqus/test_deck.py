import json
import re
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    DeckSettings, IdentityError, load_hierarchy, resolve_identities, write_deck)
from .helpers import SHAPE, read_elsets, voxel_tets, work_dir, write_raw_folder


def build(root, level='blk', with_sblk=False, merge=None, settings=None, rule='larger',
          mutate=None, **kwargs):
    """Write a raw folder and a deck of the structured mesh; return the pieces."""
    (root / 'raw').mkdir()
    write_raw_folder(root / 'raw', with_sblk=with_sblk)
    hier = load_hierarchy(root / 'raw', level)
    meshed = hier.lgi.copy()
    survivor = None
    if merge:
        survivor = {label: label for label in hier.features}
        for absorbed, target in merge.items():
            survivor[absorbed] = target
            meshed[hier.lgi == absorbed] = target
    ident = resolve_identities(hier.voxel_counts, survivor_map=survivor, rule=rule)
    pts, tets, grains = voxel_tets(meshed)
    if mutate:
        pts, tets, grains = mutate(pts, tets, grains)
    result = write_deck(root / 'deck', pts, tets, grains, hier, ident, settings,
                        meshed_labels=meshed, **kwargs)
    return hier, ident, (pts, tets, grains), result


def sections(path):
    return re.findall(r'\*Solid Section, elset=(\S+), material=(\S+)', path.read_text())


def materials(path):
    cards = {}
    for block in path.read_text().split('*Material, name=')[1:]:
        name = block.splitlines()[0]
        match = re.search(r'\*User Material, constants=6\n(.*)\n', block)
        cards[name] = [float(v) for v in match.group(1).split(',')] if match else None
    return cards


class DeckTests(unittest.TestCase):
    def test_block_level_structure(self):
        with work_dir() as root:
            hier, ident, (pts, tets, grains), res = build(root)
            deck = res['directory']
            self.assertEqual(sorted(p.name for p in deck.iterdir()), sorted(res['files']))
            for name in ('01_nodes.inp', '02_elements.inp', '03a_elsets_pag.inp',
                         '03b_elsets_pck.inp', '03c_elsets_blk.inp', '04_nsets_bc.inp',
                         '05_materials.inp', '06_sections.inp', '07_interactions.inp',
                         '08_steps_output.inp', 'model_master.inp'):
                self.assertTrue((deck / name).is_file(), name)
            self.assertFalse((deck / '03d_elsets_subblk.inp').exists())

            blk = read_elsets(deck / '03c_elsets_blk.inp')
            self.assertEqual(set(blk), {'es_blk_1_1_1', 'es_blk_1_1_2', 'es_blk_1_2_1',
                                        'es_blk_2_1_1', 'es_blk_2_1_2'})
            names = {1: 'es_blk_1_1_1', 2: 'es_blk_1_1_2', 3: 'es_blk_1_2_1',
                     4: 'es_blk_2_1_1', 5: 'es_blk_2_1_2'}
            for label, name in names.items():
                self.assertEqual(blk[name], (np.flatnonzero(grains == label) + 1).tolist())
            self.assertEqual(sorted(sum(blk.values(), [])), list(range(1, len(tets) + 1)))

            pck = read_elsets(deck / '03b_elsets_pck.inp')
            self.assertEqual(sorted(pck['es_pck_1_1']), sorted(blk['es_blk_1_1_1'] + blk['es_blk_1_1_2']))
            self.assertEqual(pck['es_pck_1_2'], blk['es_blk_1_2_1'])
            pag = read_elsets(deck / '03a_elsets_pag.inp')
            self.assertEqual(len(pag['es_pag_1']) + len(pag['es_pag_2']), len(tets))
            self.assertEqual(sorted(pag['es_pag_2']), sorted(blk['es_blk_2_1_1'] + blk['es_blk_2_1_2']))

    def test_sections_materials_and_global_ids(self):
        with work_dir() as root:
            hier, _, _, res = build(root)
            deck = res['directory']
            blk = read_elsets(deck / '03c_elsets_blk.inp')
            cards = materials(deck / '05_materials.inp')
            pairs = sections(deck / '06_sections.inp')
            self.assertEqual(len(pairs), 5)
            for elset, material in pairs:
                self.assertIn(elset, blk)
                self.assertIn(material, cards)
                self.assertEqual(material, 'MAT_BLK_' + elset[len('es_blk_'):])
            by_key = {'MAT_BLK_' + '_'.join(map(str, f.key)): f for f in hier.features.values()}
            for name, constants in cards.items():
                f = by_key[name]
                self.assertEqual(constants[:3], [round(a, 6) for a in f.euler])
                self.assertEqual(constants[3:], [float(f.global_id), 15.0, 0.0])
            self.assertIn('*Depvar\n1,\n', (deck / '05_materials.inp').read_text())

    def test_face_node_sets(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            text = (res['directory'] / '04_nsets_bc.inp').read_text()
        counts = res['info']['face_node_counts']
        nx, ny, nz = SHAPE
        self.assertEqual(counts, {'XMIN': (ny + 1) * (nz + 1), 'XMAX': (ny + 1) * (nz + 1),
                                  'YMIN': (nx + 1) * (nz + 1), 'YMAX': (nx + 1) * (nz + 1),
                                  'ZMIN': (nx + 1) * (ny + 1), 'ZMAX': (nx + 1) * (ny + 1)})
        for name in ('ns_xminus', 'ns_xplus', 'ns_yminus', 'ns_yplus', 'ns_zminus', 'ns_zplus'):
            self.assertIn(f'*Nset, nset={name}, instance=FM_STEEL_RVE-1', text)

    def test_step_and_master(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            steps = (res['directory'] / '08_steps_output.inp').read_text()
            master = (res['directory'] / 'model_master.inp').read_text()
        self.assertIn('*Static\n0.01, 80, 1e-08, 1\n', steps)
        self.assertIn('*Boundary\nns_zplus, 3, 3, 0.2\n', steps)
        for line in ('ns_xminus, 1, 1', 'ns_yminus, 2, 2', 'ns_zminus, 3, 3'):
            self.assertIn(line, steps)
        self.assertIn('*Element Output, directions=YES\nLE, NE, S, SDV\n', steps)
        self.assertIn('*Node Output\nU,\n', steps)
        self.assertIn('*Part, name=FM_STEEL_RVE', master)
        self.assertLess(master.index('01_nodes.inp'), master.index('*End Part'))
        self.assertLess(master.index('*End Assembly'), master.index('05_materials.inp'))

    def test_units_and_length_scale(self):
        nx = SHAPE[0]
        with work_dir() as root:
            _, _, _, res = build(root)
            last = (res['directory'] / '01_nodes.inp').read_text().splitlines()[-1]
        self.assertAlmostEqual(float(last.split(',')[1]), nx * 0.001)
        self.assertAlmostEqual(res['info']['coordinate_scale'], 0.001)
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(output_units='microns', length_scale=2.0))
            last = (res['directory'] / '01_nodes.inp').read_text().splitlines()[-1]
            steps = (res['directory'] / '08_steps_output.inp').read_text()
        self.assertAlmostEqual(float(last.split(',')[1]), nx * 1.0 * 2.0)
        self.assertIn('ns_zplus, 3, 3, 0.4\n', steps)

    def test_mesh_voxel_length_is_divided_out(self):
        with work_dir() as root:
            (root / 'raw').mkdir()
            write_raw_folder(root / 'raw')
            hier = load_hierarchy(root / 'raw', 'blk')
            ident = resolve_identities(hier.voxel_counts)
            pts, tets, grains = voxel_tets(hier.lgi, spacing=0.5)
            res = write_deck(root / 'deck', pts, tets, grains, hier, ident, mesh_voxel_length=0.5)
            last = (root / 'deck' / '01_nodes.inp').read_text().splitlines()[-1]
        self.assertAlmostEqual(float(last.split(',')[1]), SHAPE[0] * 0.001)

    def test_packet_and_pag_levels(self):
        with work_dir() as root:
            _, _, (_, _, grains), res = build(root, level='pck')
            deck = res['directory']
            self.assertTrue((deck / '03b_elsets_pck.inp').is_file())
            self.assertFalse((deck / '03c_elsets_blk.inp').exists())
            pairs = sections(deck / '06_sections.inp')
            self.assertEqual(sorted(pairs), [('es_pck_1_1', 'MAT_PCK_1_1'), ('es_pck_1_2', 'MAT_PCK_1_2'),
                                             ('es_pck_2_1', 'MAT_PCK_2_1')])
            self.assertEqual(len(read_elsets(deck / '03b_elsets_pck.inp')['es_pck_1_2']),
                             int(np.sum(grains == 2)))
        with work_dir() as root:
            _, _, _, res = build(root, level='pag')
            deck = res['directory']
            self.assertFalse((deck / '03b_elsets_pck.inp').exists())
            self.assertEqual(sorted(sections(deck / '06_sections.inp')),
                             [('es_pag_1', 'MAT_PAG_1'), ('es_pag_2', 'MAT_PAG_2')])
            self.assertEqual(materials(deck / '05_materials.inp')['MAT_PAG_2'][3], 1.0)

    def test_subblock_level(self):
        with work_dir() as root:
            hier, _, _, res = build(root, level='sblk', with_sblk=True)
            deck = res['directory']
            sub = read_elsets(deck / '03d_elsets_subblk.inp')
            self.assertEqual(len(sub), 10)
            self.assertIn('es_subblk_1_1_1_1', sub)
            self.assertEqual(len(sections(deck / '06_sections.inp')), 10)
            self.assertIn('MAT_SUBBLK_2_1_2_2', materials(deck / '05_materials.inp'))
            self.assertIn('03d_elsets_subblk.inp', (deck / 'model_master.inp').read_text())

    def test_larger_grain_keeps_identity_when_merged(self):
        # raw label 4 (24 voxels) was absorbed by raw label 3 (16 voxels)
        with work_dir() as root:
            hier, ident, (_, _, grains), res = build(root, merge={4: 3})
            deck = res['directory']
            self.assertEqual(ident.winner[3], 4)
            blk = read_elsets(deck / '03c_elsets_blk.inp')
            self.assertNotIn('es_blk_1_2_1', blk)
            self.assertEqual(len(blk['es_blk_2_1_1']), int(np.sum(grains == 3)) )
            cards = materials(deck / '05_materials.inp')
            self.assertEqual(len(cards), 4)
            self.assertNotIn('MAT_BLK_1_2_1', cards)
            self.assertEqual(cards['MAT_BLK_2_1_1'][3], 8.0)
            self.assertEqual(sorted(cards['MAT_BLK_2_1_2'][3:4]), [9.0])
            pag = read_elsets(deck / '03a_elsets_pag.inp')
            self.assertEqual(len(pag['es_pag_2']), int(np.sum(np.isin(grains, [3, 5]))))
            self.assertEqual(res['info']['features_without_elements'], [3])
            self.assertEqual(len(sections(deck / '06_sections.inp')), 4)

    def test_survivor_rule_keeps_absorbing_grain(self):
        with work_dir() as root:
            _, _, _, res = build(root, merge={4: 3}, rule='survivor')
            blk = read_elsets(res['directory'] / '03c_elsets_blk.inp')
            self.assertIn('es_blk_1_2_1', blk)
            self.assertNotIn('es_blk_2_1_1', blk)

    def test_reports(self):
        with work_dir() as root:
            _, _, _, res = build(root, merge={4: 3})
            deck = res['directory']
            realloc = json.loads((deck / 'reallocation_report.json').read_text())
            identity = json.loads((deck / 'identity_report.json').read_text())
            audit = json.loads((deck / 'location_audit.json').read_text())
        self.assertEqual(realloc['voxels_moved'], 16)
        self.assertEqual(realloc['moved_across_pag'], 16)
        self.assertEqual(realloc['absorbed_raw_labels'], [3])
        self.assertEqual(identity['merged_groups'][0]['winner'], 4)
        self.assertEqual(audit['violations'], [])
        self.assertGreater(audit['volume_over_own_raw_label'], 0.6)

    def test_audit_rejects_a_wrong_identity(self):
        def swap(pts, tets, grains):
            swapped = grains.copy()
            swapped[grains == 1], swapped[grains == 5] = 5, 1
            return pts, tets, swapped
        with work_dir() as root:
            with self.assertRaises(IdentityError):
                build(root, mutate=swap)
        with work_dir() as root:
            _, _, _, res = build(root, mutate=swap, audit=False)
            self.assertIsNone(res['audit'])

    def test_unused_nodes_are_dropped(self):
        def stray(pts, tets, grains):
            return np.vstack((pts, [[99., 99., 99.]])), tets, grains
        with work_dir() as root:
            _, _, (pts, _, _), res = build(root, mutate=stray)
            nodes = (res['directory'] / '01_nodes.inp').read_text().splitlines()
        self.assertEqual(res['info']['unused_nodes_dropped'], 1)
        self.assertEqual(len(nodes) - 1, len(pts) - 1)

    def test_invalid_meshes_are_rejected(self):
        def inverted(pts, tets, grains):
            flipped = tets.copy()
            flipped[0, [1, 2]] = flipped[0, [2, 1]]
            return pts, flipped, grains
        def unknown(pts, tets, grains):
            other = grains.copy()
            other[0] = 99
            return pts, tets, other
        with work_dir() as root:
            with self.assertRaises(ValueError):
                build(root, mutate=inverted)
        with work_dir() as root:
            with self.assertRaises(ValueError):
                build(root, mutate=unknown)
        with work_dir() as root:
            build(root)
            hier = load_hierarchy(root / 'raw', 'blk')
            pts, tets, grains = voxel_tets(hier.lgi)
            with self.assertRaises(FileExistsError):
                write_deck(root / 'deck', pts, tets, grains, hier,
                           resolve_identities(hier.voxel_counts))

    def test_material_formats(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(material_format='bunge_euler', n_depvar=4))
            text = (res['directory'] / '05_materials.inp').read_text()
        self.assertIn('*User Material, constants=3\n', text)
        self.assertIn('*Depvar\n4\n', text)
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(material_format='orientation'))
            self.assertIn('*Elastic\n210000., 0.3\n', (res['directory'] / '05_materials.inp').read_text())

    def test_no_step_and_custom_header(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(write_step=False, header_message='run 7'))
            self.assertIn('No step written', (res['directory'] / '08_steps_output.inp').read_text())
            self.assertIn('** run 7\n', (res['directory'] / 'model_master.inp').read_text())

    def test_invalid_settings(self):
        bad = [dict(material_format='x'), dict(load_axis='w'), dict(n_depvar=0),
               dict(length_scale=0), dict(output_units='km'), dict(step_time=0),
               dict(min_inc=1.0, initial_inc=0.5), dict(max_inc=100.0),
               dict(output_interval=500.0), dict(element_outputs=(), node_outputs=()),
               dict(nset_names={'XMIN': 'a'})]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                DeckSettings(**kwargs).validate()
        self.assertEqual(DeckSettings(element_outputs='le, s').validate()[0], ('LE', 'S'))


if __name__ == '__main__':
    unittest.main()
