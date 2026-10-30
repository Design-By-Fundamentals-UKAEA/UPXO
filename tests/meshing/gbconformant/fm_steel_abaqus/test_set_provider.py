"""Sets from a set provider, the naming rules and the deck check of names and output sets."""
import json
import unittest

import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, load_hierarchy, resolve_identities, write_deck
from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import check_deck, read_sets, read_surfaces
from upxo.meshing.gbconformant.fm_steel_abaqus.setnames import SetNameError, SetRegistry, name_problem
from upxo.meshing.gbconformant.pipeline.provenance import build_provenance
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, build_config
from upxo.meshing.gbconformant.pipeline.stage_output import deck_settings
from .helpers import voxel_tets, work_dir, write_raw_folder

FAKE = 'meshing.gbconformant.fm_steel_abaqus.fake_set_provider'


class Fixture:
    def __init__(self, root):
        raw = root / 'raw'
        raw.mkdir()
        write_raw_folder(raw, with_sblk=True)
        self.hier = load_hierarchy(raw, 'sblk')
        self.ident = resolve_identities(self.hier.voxel_counts)
        self.points, self.elements, self.grains = voxel_tets(self.hier.lgi.copy())

    def deck(self, root, name='deck', **settings):
        settings.setdefault('deck_check', True)
        return write_deck(root / name, self.points, self.elements, self.grains, self.hier, self.ident,
                          DeckSettings(**settings))


class ProviderTests(unittest.TestCase):
    def test_without_a_provider_the_deck_holds_only_its_own_sets(self):
        with work_dir() as root:
            res = Fixture(root).deck(root, layer_sets=True, boundary_sets=True)
            names = sorted(p.name for p in res['directory'].glob('0[34]*.inp'))
            self.assertEqual(names, ['03a_elsets_pag.inp', '03b_elsets_pck.inp', '03c_elsets_blk.inp',
                                     '03d_elsets_subblk.inp', '03f_elsets_layers.inp', '04_nsets_bc.inp',
                                     '04b_nsets_boundaries.inp'])
            self.assertIsNone(res['info']['extra_sets'])

    def test_a_provider_adds_sets_surfaces_and_node_sets_and_the_deck_says_so(self):
        with work_dir() as root:
            res = Fixture(root).deck(root, set_provider=FAKE, provider_options={'name': 'es_extra'})
            deck = res['directory']
            self.assertEqual(read_sets(deck / '03x_elsets_extra.inp', 'Elset')['es_extra'].tolist(), [1, 2, 3])
            self.assertIn('surf_extra', read_surfaces(deck / '03y_surfaces_extra.inp'))
            self.assertIn('ns_extra', read_sets(deck / '04x_nsets_extra.inp', 'Nset'))
            master = (deck / 'model_master.inp').read_text()
            part, assembly = master.split('*End Part')
            self.assertIn('03x_elsets_extra.inp', part)
            self.assertIn('03y_surfaces_extra.inp', part)
            self.assertIn('04x_nsets_extra.inp', assembly)
            info = json.loads((deck / 'deck_info.json').read_text())
            self.assertEqual(info['extra_sets']['provider'], FAKE)
            self.assertEqual(info['extra_sets']['skipped'][0]['name'], 'es_never')
            self.assertEqual(info['extra_sets']['summary'], {'calls': 1, 'identity_given': True})
            self.assertTrue(json.loads((deck / 'deck_check.json').read_text())['ok'])

    def test_provider_names_are_checked_like_the_decks_own(self):
        with work_dir() as root:
            fx = Fixture(root)
            with self.assertRaisesRegex(SetNameError, 'es_pag_1'):
                fx.deck(root, 'clash', set_provider=FAKE, provider_options={'name': 'ES_PAG_1'})
            self.assertFalse((root / 'clash').exists())
            with self.assertRaisesRegex(SetNameError, 'refuses'):
                fx.deck(root, 'refused', set_provider=FAKE, provider_options={'name': 'refuse'})
            with self.assertRaisesRegex(ValueError, 'files named 03'):
                fx.deck(root, 'file', set_provider=FAKE, provider_options={'name': 'es_extra', 'file': 'extra.inp'})
            with self.assertRaisesRegex(ValueError, 'already has'):
                fx.deck(root, 'taken', set_provider=FAKE,
                        provider_options={'name': 'es_extra', 'file': '03a_elsets_pag.inp'})
            with self.assertRaisesRegex(ValueError, 'cannot be imported'):
                fx.deck(root, 'missing', set_provider='no.such.provider')

    def test_naming_the_provider_in_the_settings_makes_its_settings_known(self):
        config = build_config({'SET_PROVIDER': FAKE, 'FAKE_EXTRA_NAME': 'es_other'})
        self.assertIn('FAKE_EXTRA_NAME', SCHEMA)
        settings = deck_settings(config)
        self.assertEqual(settings.set_provider, FAKE)
        self.assertEqual(settings.provider_options['name'], 'es_other')
        record = build_provenance(config)['set_provider']
        self.assertEqual(record['module'], FAKE)
        self.assertIsNone(build_provenance(build_config({}))['set_provider'])


class NameRuleTests(unittest.TestCase):
    def test_rules(self):
        self.assertIsNone(name_problem('es_pag_1'))
        self.assertIsNone(name_problem('FM-steel_set'))
        for bad in ('', ' es', 'es pag', 'es,pag', '1es', 'es.pag', 'e' * 81):
            self.assertIsNotNone(name_problem(bad), bad)

    def test_registry_catches_repeats_ignoring_case_and_keeps_scopes_apart(self):
        registry = SetRegistry()
        registry.add('Elset', 'es_a', 'first')
        registry.add('Elset', 'ES_A', 'second')
        registry.add('Nset', 'es_a', 'a node set may share an element set name')
        self.assertEqual(len(registry.problems), 1)
        self.assertIn('Abaqus ignores case', registry.problems[0])

    def test_the_export_stops_on_bad_or_clashing_names_and_writes_nothing(self):
        with work_dir() as root:
            fx = Fixture(root)
            for pattern, settings in (
                    ('already used', dict(layer_sets=True, prefix_gb_layer='es_x', prefix_interior='ES_X')),
                    ('empty', dict(layer_sets=True, prefix_gb_layer='')),
                    ('prefix', dict(elset_prefixes={'pag': 'es pag', 'pck': 'es_pck_', 'blk': 'es_blk_',
                                                    'sblk': 'es_subblk_'})),
                    ('es_pag_1', dict(layer_sets=True, prefix_gb_layer='ES_PAG_1')),
                    ('ns_xminus', dict(boundary_sets=True, nset_names={'XMIN': 'ns_xminus', 'XMAX': 'NS_XMINUS',
                                                                       'YMIN': 'a', 'YMAX': 'b', 'ZMIN': 'c',
                                                                       'ZMAX': 'd'}))):
                with self.subTest(pattern=pattern):
                    with self.assertRaisesRegex(SetNameError, pattern):
                        fx.deck(root, 'stopped', **settings)
                    self.assertFalse((root / 'stopped').exists())


class DeckCheckTests(unittest.TestCase):
    def test_output_sets_are_named_through_the_instance(self):
        with work_dir() as root:
            res = Fixture(root).deck(root, layer_sets=True, output_elset='es_gb_layer')
            text = (res['directory'] / '08_steps_output.inp').read_text()
            self.assertIn('*Element Output, elset=FM_STEEL_RVE-1.es_gb_layer, directions=YES', text)

    def test_the_check_finds_repeated_names_and_unqualified_output_sets(self):
        with work_dir() as root:
            res = Fixture(root).deck(root, layer_sets=True, output_elset='es_gb_layer')
            deck = res['directory']
            self.assertTrue(check_deck(deck)['ok'])
            text = (deck / '08_steps_output.inp').read_text()
            (deck / '08_steps_output.inp').write_text(text.replace('FM_STEEL_RVE-1.es_gb_layer', 'es_gb_layer'))
            path = deck / '03f_elsets_layers.inp'
            path.write_text(path.read_text() + '*Elset, elset=ES_GB_LAYER\n1\n')
            failed = check_deck(deck)['failed']
            self.assertIn('no elset name defined twice', failed)
            self.assertIn('element output sets exist (named through the instance)', failed)


if __name__ == '__main__':
    unittest.main()
