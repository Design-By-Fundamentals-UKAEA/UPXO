import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from upxo.meshing.gbconformant.pipeline import SettingsError, run, stage_ids
from upxo.meshing.gbconformant.pipeline.checkpoints import CheckpointStore
from upxo.meshing.gbconformant.pipeline.manifest import MANIFEST_NAME

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2,
    GRAIN_ID_SEED=3)


class RunTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        write_raw_folder(self.raw)
        self.events = []

    def go(self, through='spikes', resume=True, write_manifest=True, **settings):
        merged = dict(FAST, RUN_NAME='headless', RAW_FOLDER=str(self.raw))
        merged.update(settings)
        return run(merged, through=through, resume=resume, write_manifest=write_manifest,
                   on_event=self.events.append)

    def kinds(self, kind):
        return [e['stage'] for e in self.events if e['event'] == kind]

    def test_runs_the_stages_in_order_and_saves_each(self):
        result = self.go()
        self.assertTrue(result.ok, result.error)
        self.assertEqual([s for s in stage_ids() if s in result.ran],
                         result.ran)                       # in pipeline order
        self.assertIn('spikes', result.ran)
        self.assertEqual(self.kinds('start'), [s for s in result.ran])
        self.assertEqual(self.kinds('done'), [s for s in result.ran])
        self.assertIn('subset_extract', result.skipped)      # its switch is off
        saved = CheckpointStore(result.run_folder).stages()
        for stage in ('load_lgi', 'grain_id_map', 'small_grains', 'spikes'):
            self.assertIn(stage, saved)
        self.assertIsNotNone(result.state.get_artefact('labels'))
        self.assertEqual(self.kinds('failed'), [])

    def test_a_second_run_restores_instead_of_running(self):
        first = self.go()
        self.events.clear()
        second = self.go()
        self.assertTrue(second.ok, second.error)
        self.assertEqual(second.ran, [])
        self.assertIn('spikes', second.restored)
        self.assertEqual(self.kinds('start'), [])
        self.assertEqual(self.kinds('restored'), second.restored)
        np.testing.assert_array_equal(second.state.get_artefact('labels'),
                                      first.state.get_artefact('labels'))

    def test_a_changed_setting_runs_that_stage_and_the_ones_after(self):
        self.go(through='pinches')
        self.events.clear()
        result = self.go(through='pinches', SPIKE_MAX_PASSES=7)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.ran, ['spikes', 'pinches'])
        self.assertIn('small_grains', result.restored)
        self.assertIn('spikes', self.kinds('stale'))

    def test_without_history_a_changed_early_setting_runs_everything_again(self):
        self.go(through='pinches', CHECKPOINT_KEEP_HISTORY=False)
        self.events.clear()
        result = self.go(through='pinches', CHECKPOINT_KEEP_HISTORY=False, SPIKE_MAX_PASSES=7)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.restored, [])
        self.assertEqual(result.ran[0], 'load_lgi')
        self.assertIn('pinches', result.ran)

    def test_resume_can_be_turned_off(self):
        self.go()
        self.events.clear()
        result = self.go(resume=False)
        self.assertEqual(result.restored, [])
        self.assertIn('spikes', result.ran)

    def test_a_switched_off_stage_is_skipped_and_reported(self):
        result = self.go(STAGE_SPIKES=False, through='pinches')
        self.assertTrue(result.ok, result.error)
        self.assertIn('spikes', result.skipped)
        self.assertNotIn('spikes', result.ran)
        skipped = [e for e in self.events if e['event'] == 'skipped' and e['stage'] == 'spikes']
        self.assertEqual(len(skipped), 1)
        self.assertIn('STAGE_SPIKES', skipped[0]['reason'])

    def test_a_failing_stage_is_reported_not_raised(self):
        result = self.go(RAW_FOLDER=str(self.root / 'missing'))
        self.assertFalse(result.ok)
        self.assertEqual(result.failed_stage, 'load_lgi')
        self.assertIn('RAW_FOLDER', result.error)
        self.assertEqual(self.kinds('failed'), ['load_lgi'])
        with self.assertRaises(RuntimeError):
            result.raise_for_failure()

    def test_unusable_settings_and_stages_are_refused_before_any_work(self):
        with self.assertRaises(SettingsError):
            run(NOT_A_SETTING=1)
        with self.assertRaises(SettingsError):
            run(UPSCALE_FACTOR=9)
        with self.assertRaises(KeyError):
            run(RAW_FOLDER=str(self.raw), through='not_a_stage')
        self.assertFalse((self.root / 'data').exists())

    def test_the_manifest_is_written_with_the_run(self):
        result = self.go()
        self.assertEqual(result.manifest, result.run_folder / MANIFEST_NAME)
        manifest = json.loads(result.manifest.read_text(encoding='utf-8'))
        self.assertEqual(manifest['format'], 'upxo.gbconformant.run_manifest')
        self.assertEqual(manifest['seeds']['grain_id_seed'], 3)
        self.assertEqual(manifest['input']['hierarchy_level'], 'blk')
        self.assertIn('spikes', manifest['saved_results']['stages'])
        self.assertEqual(manifest['stages'], stage_ids())
        changed = {row['key'] for row in manifest['config_diff_vs_defaults']}
        self.assertIn('SPIKE_MAX_PASSES', changed)
        self.assertIsNone(self.go(write_manifest=False).manifest)

    def test_a_subset_run_keeps_its_results_in_its_own_folder_and_resumes_there(self):
        subset = dict(STAGE_SUBSET=True, SUBSET_MODE='index', SUBSET_LO=[1, 0, 0],
                      SUBSET_HI=[5, 4, 4])
        first = self.go(**subset)
        self.assertTrue(first.ok, first.error)
        self.assertEqual(first.run_folder.name, 'headless_subset_001')
        self.assertEqual(first.state.get_artefact('labels').shape, (4, 4, 4))
        self.events.clear()
        resumed = self.go(_RUN_FOLDER=str(first.run_folder), **subset)
        self.assertTrue(resumed.ok, resumed.error)
        self.assertEqual(resumed.run_folder, first.run_folder)
        self.assertEqual(resumed.ran, [])
        self.assertIn('subset_extract', resumed.restored)
        self.assertFalse((first.run_folder.parent / 'headless_subset_002').exists())


if __name__ == '__main__':
    unittest.main()
