import os
import shutil
import unittest
from unittest import mock

import numpy as np

from upxo.meshing.gbconformant.pipeline import PipelineState, StageStatus, build_config, stage_ids
from upxo.meshing.gbconformant.pipeline import saved_results
from upxo.meshing.gbconformant.pipeline.background import (
    NotReady, StageProcess, run_stage_separately)
from upxo.meshing.gbconformant.pipeline.run import new_state, run
from upxo.meshing.gbconformant.pipeline.runner import PipelineRunner

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2,
    GRAIN_ID_SEED=3)


class SeparateProcessTests(unittest.TestCase):
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

    def state(self, name='separate', **settings):
        merged = dict(FAST, RUN_NAME=name, RAW_FOLDER=str(self.raw))
        merged.update(settings)
        return new_state(build_config(merged))

    def test_stages_run_in_other_processes_give_the_results_of_running_here(self):
        state = self.state('separate')
        events = []
        for stage in ('load_lgi', 'grain_id_map', 'small_grains', 'spikes'):
            job = run_stage_separately(state, stage, on_event=events.append)
            self.assertEqual(job.outcome, 'finished', job.error)
            self.assertEqual(state.status[stage], StageStatus.OK)
        reference = run(dict(FAST, RUN_NAME='local', RAW_FOLDER=str(self.raw)), through='spikes')
        self.assertTrue(reference.ok, reference.error)
        np.testing.assert_array_equal(state.get_artefact('labels'),
                                      reference.state.get_artefact('labels'))
        self.assertEqual(state.get_artefact('grain_id_map'), reference.state.get_artefact('grain_id_map'))
        self.assertEqual(state.reports['spikes']['stop_reason'], reference.state.reports['spikes']['stop_reason'])
        self.assertEqual(state.artefact_writer['labels'], 'spikes')
        self.assertTrue(any(e['event'] == 'progress' and e['stage'] == 'spikes' for e in events))
        self.assertTrue(any('ok: spikes' in line for line in state.log_lines))
        self.assertFalse((self.root / 'data' / 'runs' / 'separate' / '.worker').exists()
                         and any((self.root / 'data' / 'runs' / 'separate' / '.worker').iterdir()))

    def test_optional_stages_skipped_on_the_way_are_taken_over(self):
        state = self.state('skipped', STAGE_SMALL_GRAINS=False)
        for stage in ('load_lgi', 'grain_id_map', 'spikes'):
            self.assertEqual(run_stage_separately(state, stage).outcome, 'finished')
        self.assertEqual(state.status['small_grains'], StageStatus.SKIPPED)
        self.assertEqual(state.status['spikes'], StageStatus.OK)

    def test_settings_a_stage_changes_come_back_and_survive_a_restore(self):
        state = self.state('upscaled', UPSCALE_FACTOR=2)
        for stage in stage_ids()[:stage_ids().index('upscale') + 1]:
            if stage in ('subset_extract',):
                continue
            job = run_stage_separately(state, stage)
            self.assertIn(job.outcome, ('finished',), (stage, job.error))
        self.assertEqual(state.config['EFFECTIVE_SPACING'], (0.5, 0.5, 0.5))
        self.assertEqual(state.get_artefact('labels_upscaled').shape,
                         tuple(2 * n for n in state.get_artefact('labels_meshing').shape))
        fresh = self.state('upscaled', UPSCALE_FACTOR=2)
        summary = saved_results.restore(fresh)
        self.assertIn('upscale', summary['restored'])
        self.assertEqual(fresh.config['EFFECTIVE_SPACING'], (0.5, 0.5, 0.5))
        self.assertEqual(fresh.config['HIERARCHY_LEVEL'], 'blk')

    def test_a_subset_stage_moves_the_run_to_its_own_folder(self):
        state = self.state('crop', STAGE_SUBSET=True, SUBSET_MODE='index', SUBSET_LO=[1, 0, 0],
                           SUBSET_HI=[5, 4, 4])
        self.assertEqual(run_stage_separately(state, 'load_lgi').outcome, 'finished')
        self.assertEqual(run_stage_separately(state, 'subset_extract').outcome, 'finished')
        self.assertTrue(state.config['_RUN_FOLDER'].endswith('crop_subset_001'))
        self.assertEqual(state.get_artefact('labels').shape, (4, 4, 4))
        self.assertEqual(run_stage_separately(state, 'grain_id_map').outcome, 'finished')
        self.assertIn('grain_id_map', saved_results.store_for(state).stages())
        self.assertEqual(state.get_artefact('labels').shape, (4, 4, 4))

    def test_cancel_stops_the_process_and_leaves_the_state_alone(self):
        state = self.state('cancelled')
        job = StageProcess(state, 'load_lgi')
        job.start()
        job.cancel()
        self.assertTrue(job.done)
        self.assertEqual(job.outcome, 'cancelled')
        self.assertIsNotNone(job._process.returncode)
        self.assertEqual(state.status['load_lgi'], StageStatus.NOT_RUN)
        self.assertIsNone(state.get_artefact('labels'))
        self.assertNotIn('load_lgi', saved_results.store_for(state).stages())
        with self.assertRaises(RuntimeError):
            job.apply()
        job.cleanup()

    def test_cancel_through_the_wait_helper(self):
        state = self.state('cancel_helper')
        job = run_stage_separately(state, 'load_lgi', should_cancel=lambda: True)
        self.assertEqual(job.outcome, 'cancelled')
        self.assertEqual(state.status['load_lgi'], StageStatus.NOT_RUN)

    def test_a_failing_stage_reports_why_and_leaves_the_state_alone(self):
        state = self.state('failing', RAW_FOLDER=str(self.root / 'missing'))
        job = run_stage_separately(state, 'load_lgi')
        self.assertEqual(job.outcome, 'failed')
        self.assertIn('RAW_FOLDER', job.error)
        self.assertIn('Traceback', job.traceback)
        self.assertEqual(state.status['load_lgi'], StageStatus.NOT_RUN)
        self.assertTrue(job.folder.exists())                  # kept for diagnosis

    def test_a_stage_whose_earlier_results_are_not_saved_is_refused(self):
        state = self.state('unsaved')
        PipelineRunner(state).run_stage('load_lgi')
        for path in (self.root / 'data' / 'runs' / 'unsaved' / 'checkpoints').glob('*'):
            shutil.rmtree(path)
        with self.assertRaises(NotReady) as caught:
            StageProcess(state, 'grain_id_map')
        self.assertIn('load_lgi', str(caught.exception))
        off = PipelineState(build_config(dict(FAST, RUN_NAME='off', RAW_FOLDER=str(self.raw),
                                              CHECKPOINT_RESULTS=False)), stage_ids())
        with self.assertRaises(NotReady):
            StageProcess(off, 'load_lgi')
        with self.assertRaises(KeyError):
            StageProcess(self.state('typo'), 'not_a_stage')

    def test_a_changed_setting_makes_the_saved_results_unusable_for_a_process(self):
        state = self.state('edited')
        self.assertEqual(run_stage_separately(state, 'load_lgi').outcome, 'finished')
        state.config['GRAIN_ID_SEED'] = 99                      # grain_id_map's own setting
        self.assertEqual(run_stage_separately(state, 'grain_id_map').outcome, 'finished')
        state.config['HIERARCHY_LEVEL'] = 'pck'                 # load_lgi changed after it ran
        with self.assertRaises(NotReady):
            StageProcess(state, 'small_grains')


if __name__ == '__main__':
    unittest.main()
