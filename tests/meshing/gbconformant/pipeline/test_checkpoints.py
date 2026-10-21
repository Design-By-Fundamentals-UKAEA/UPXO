import unittest
from types import SimpleNamespace

import numpy as np

from upxo.meshing.gbconformant.pipeline import (
    CheckpointStore, canonical_signature, chain_signatures)

from ..fm_steel_abaqus.helpers import work_dir

STAGES = ['load', 'clean', 'mesh', 'export']


def make_state():
    return SimpleNamespace(artefacts={}, artefact_writer={}, reports={})


def produce(state, stage, report=None, **artefacts):
    for key, value in artefacts.items():
        state.artefacts[key] = value
        state.artefact_writer[key] = stage
    state.reports[stage] = report or {'duration_s': 1.5}


def own(stage):
    return {'stage': stage, 'setting': SETTINGS.get(stage, 0)}


SETTINGS = {}


def signatures():
    return chain_signatures(STAGES, own)


class Recorder:
    def __init__(self):
        self.status = {}

    def __call__(self, stage, text):
        self.status[stage] = text


class SignatureTests(unittest.TestCase):
    def test_signatures_ignore_order_and_see_changes(self):
        self.assertEqual(canonical_signature({'a': 1, 'b': [1, 2]}), canonical_signature({'b': [1, 2], 'a': 1}))
        self.assertNotEqual(canonical_signature({'a': 1}), canonical_signature({'a': 2}))
        self.assertEqual(canonical_signature({'v': np.array([1, 2])}), canonical_signature({'v': [1, 2]}))
        self.assertEqual(canonical_signature({'v': (1.0, 2)}), canonical_signature({'v': [1, 2.0]}))
        self.assertNotEqual(canonical_signature({'v': 1.5}), canonical_signature({'v': 1}))

    def test_chain_depends_on_everything_before(self):
        SETTINGS.clear()
        base = signatures()
        SETTINGS['clean'] = 5
        changed = signatures()
        SETTINGS.clear()
        self.assertEqual(base['load'], changed['load'])
        self.assertTrue(all(base[s] != changed[s] for s in ('clean', 'mesh', 'export')))


class StoreTests(unittest.TestCase):
    def setUp(self):
        SETTINGS.clear()
        self.ctx = work_dir()
        self.root = self.ctx.__enter__()
        self.store = CheckpointStore(self.root)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def run_pipeline(self, state=None):
        state = state or make_state()
        sig = signatures()
        produce(state, 'load', labels=np.arange(4), hierarchy={'levels': (1, 2)})
        self.store.save_stage(state, 'load', sig['load'])
        produce(state, 'clean', {'duration_s': 2.0, 'text': 'hello'}, labels=np.arange(4) * 2)
        self.store.save_stage(state, 'clean', sig['clean'])
        produce(state, 'mesh', tets=np.ones((3, 4), dtype=np.int32), table={1: {'q': 0.5}})
        self.store.save_stage(state, 'mesh', sig['mesh'])
        return state

    def restore(self):
        fresh, status = make_state(), Recorder()
        summary = self.store.restore(fresh, STAGES, signatures(), status)
        return fresh, status.status, summary

    def test_round_trip(self):
        original = self.run_pipeline()
        fresh, status, summary = self.restore()
        self.assertEqual(summary['restored'], ['load', 'clean', 'mesh'])
        self.assertEqual(status, {'load': 'OK', 'clean': 'OK', 'mesh': 'OK'})
        np.testing.assert_array_equal(fresh.artefacts['labels'], np.arange(4) * 2)
        np.testing.assert_array_equal(fresh.artefacts['tets'], original.artefacts['tets'])
        self.assertEqual(fresh.artefacts['table'], {1: {'q': 0.5}})
        self.assertEqual(fresh.artefacts['hierarchy'], {'levels': (1, 2)})
        self.assertEqual(fresh.artefact_writer['labels'], 'clean')
        self.assertEqual(fresh.reports['clean'], {'duration_s': 2.0, 'text': 'hello'})
        self.assertGreater(summary['bytes'], 0)

    def test_every_stage_keeps_its_own_copy_by_default(self):
        self.run_pipeline()
        for stage in ('load', 'clean'):
            self.assertIn('labels', self.store.meta(stage)['artefacts'])
        self.assertTrue((self.store.stage_dir('load') / 'labels.json').exists())
        self.assertNotIn('superseded', self.store.meta('load'))

    def test_without_history_a_newer_result_replaces_the_older_copy(self):
        self.store = CheckpointStore(self.root, keep_history=False)
        self.run_pipeline()
        self.assertNotIn('labels', self.store.meta('load')['artefacts'])
        self.assertFalse((self.store.stage_dir('load') / 'labels.json').exists())
        self.assertIn('hierarchy', self.store.meta('load')['artefacts'])
        self.assertIn('labels', self.store.meta('clean')['artefacts'])
        self.assertEqual(self.store.meta('load')['superseded'], {'labels': 'clean'})
        fresh, status, summary = self.restore()
        self.assertEqual(summary['restored'], ['load', 'clean', 'mesh'])
        np.testing.assert_array_equal(fresh.artefacts['labels'], np.arange(4) * 2)

    def test_without_history_only_complete_prefixes_are_restored(self):
        self.store = CheckpointStore(self.root, keep_history=False)
        self.run_pipeline()
        SETTINGS['mesh'] = 3                  # the last stage changed: the earlier ones are complete
        fresh, status, summary = self.restore()
        self.assertEqual(summary['restored'], ['load', 'clean'])
        self.assertEqual(summary['stale'], ['mesh'])
        SETTINGS.clear()
        SETTINGS['clean'] = 3                 # load's labels were replaced by the stale clean stage
        fresh, status, summary = self.restore()
        self.assertEqual(summary['restored'], [])
        self.assertEqual(summary['stale'], ['load', 'clean', 'mesh'])
        self.assertEqual(status, {'load': 'Stale', 'clean': 'Stale', 'mesh': 'Stale'})
        self.assertNotIn('labels', fresh.artefacts)
        self.assertIn('replaced by a later stage', summary['problems'][-1])

    def test_changed_setting_stales_that_stage_and_later_ones(self):
        self.run_pipeline()
        SETTINGS['clean'] = 3
        fresh, status, summary = self.restore()
        self.assertEqual(status, {'load': 'OK', 'clean': 'Stale', 'mesh': 'Stale'})
        self.assertEqual(summary['restored'], ['load'])
        self.assertEqual(summary['stale'], ['clean', 'mesh'])
        self.assertIn('settings or inputs changed', summary['problems'][0])
        self.assertNotIn('tets', fresh.artefacts)
        np.testing.assert_array_equal(fresh.artefacts['labels'], np.arange(4))   # as the load stage left them

    def test_unknown_version_and_damaged_files(self):
        self.run_pipeline()
        meta_path = self.store.stage_dir('mesh') / 'meta.json'
        meta_path.write_text(meta_path.read_text().replace('"version": 1', '"version": 9'))
        self.assertEqual(self.restore()[1]['mesh'], 'Stale')
        self.run_pipeline()
        (self.store.stage_dir('clean') / 'labels.npz').write_bytes(b'not an archive')
        fresh, status, summary = self.restore()
        self.assertEqual(status['clean'], 'Stale')
        self.assertEqual(status['mesh'], 'Stale')
        self.assertIn('could not be read', ' '.join(summary['problems']))

    def test_a_missing_stage_stops_the_restore(self):
        self.run_pipeline()
        self.store.discard(['clean'])
        fresh, status, summary = self.restore()
        self.assertEqual(summary['restored'], ['load'])
        self.assertEqual(status, {'load': 'OK', 'mesh': 'Stale'})     # built on the missing stage
        self.assertIn('an earlier stage is not valid', summary['problems'][0])
        self.assertNotIn('tets', fresh.artefacts)

    def test_markers_and_unstorable_results(self):
        sig = signatures()
        self.store.save_marker('load', sig['load'], 'Skipped', {'skipped': True})
        state = make_state()
        produce(state, 'clean', fine=np.zeros(2), bad=object())
        meta = self.store.save_stage(state, 'clean', sig['clean'])
        self.assertIn('bad', meta['skipped'])
        fresh, status, summary = self.restore()
        self.assertEqual(status, {'load': 'Skipped', 'clean': 'OK'})
        self.assertIn('fine', fresh.artefacts)
        self.assertNotIn('bad', fresh.artefacts)
        self.assertEqual(fresh.reports['load'], {'skipped': True})

    def test_unfinished_writes_are_ignored_and_cleaned(self):
        self.run_pipeline()
        (self.store.root / '.mesh.writing').mkdir()
        (self.store.root / '.mesh.writing' / 'junk.json').write_text('{}')
        self.assertEqual(self.store.stages(), ['clean', 'load', 'mesh'])
        state = make_state()
        produce(state, 'mesh', tets=np.zeros(1))
        self.store.save_stage(state, 'mesh', signatures()['mesh'])
        self.assertFalse((self.store.root / '.mesh.writing').exists())

    def test_discard_and_overwrite(self):
        self.run_pipeline()
        self.store.discard(['mesh', 'export'])
        self.assertEqual(self.store.stages(), ['clean', 'load'])
        state = make_state()
        produce(state, 'clean', labels=np.zeros(2))
        self.store.save_stage(state, 'clean', signatures()['clean'])
        self.assertEqual(self.store.meta('clean')['artefacts'].keys(), {'labels'})
        self.assertEqual(self.store.meta('clean')['status'], 'OK')


if __name__ == '__main__':
    unittest.main()
