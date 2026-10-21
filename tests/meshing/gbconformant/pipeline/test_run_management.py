import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from upxo.meshing.gbconformant.pipeline import PipelineState, SCHEMA, StageStatus, build_config, run, stage_ids
from upxo.meshing.gbconformant.pipeline import results
from upxo.meshing.gbconformant.pipeline.run import new_state
from upxo.meshing.gbconformant.pipeline.runs import describe_run, list_runs
from upxo.meshing.gbconformant.pipeline.stage_input import HIERARCHY_KEY
from upxo.meshing.gbconformant.pipeline.units import format_length, physical_text, voxel_edge

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2,
    GRAIN_ID_SEED=3)


class StaleReasonTests(unittest.TestCase):
    def state(self):
        state = PipelineState({}, ['a', 'b', 'c', 'd'])
        for sid, seconds in (('a', 1.5), ('b', 20.0), ('c', 30.0)):
            state.set_status(sid, StageStatus.OK)
            state.reports[sid] = {'duration_s': seconds}
        return state

    def test_invalidation_records_why_for_every_stage_that_held_results(self):
        state = self.state()
        hit = state.invalidate_from('b', 'MIN_GRAIN_VOXELS changed')
        self.assertEqual(hit, ['b', 'c', 'd'])
        self.assertEqual(state.status['b'], StageStatus.STALE)
        self.assertEqual(state.status['d'], StageStatus.NOT_RUN)
        self.assertIn('b', state.stale_reasons)
        self.assertEqual(state.stale_reasons['c']['reason'], 'MIN_GRAIN_VOXELS changed')
        self.assertEqual(state.stale_reasons['c']['source'], 'b')
        self.assertNotIn('d', state.stale_reasons)                  # it never ran: nothing is lost
        self.assertNotIn('a', state.stale_reasons)
        self.assertTrue(state.stale_reason_text('b').startswith('MIN_GRAIN_VOXELS changed ('))
        self.assertEqual(state.stale_reason_text('a'), '')

    def test_running_the_stage_again_clears_the_reason(self):
        state = self.state()
        state.invalidate_from('b', 'because')
        state.set_status('b', StageStatus.OK)
        self.assertNotIn('b', state.stale_reasons)
        self.assertEqual(state.stale_reason_text('b'), '')
        self.assertIn('c', state.stale_reasons)

    def test_a_second_edit_keeps_the_first_reason_of_a_stale_stage(self):
        state = self.state()
        state.invalidate_from('b', 'first edit')
        state.invalidate_from('c', 'second edit')
        self.assertEqual(state.stale_reasons['c']['reason'], 'first edit')

    def test_the_preview_counts_finished_stages_and_their_time(self):
        state = self.state()
        preview = state.invalidation_preview('b')
        self.assertEqual(preview['stages'], ['b', 'c'])
        self.assertAlmostEqual(preview['seconds'], 50.0)
        self.assertEqual(state.invalidation_preview('d'), {'stages': [], 'seconds': 0.0})

    def test_invalidate_after_passes_the_reason_on(self):
        state = self.state()
        state.invalidate_after('a', 'the subset changed')
        self.assertEqual(state.stale_reasons['b']['reason'], 'the subset changed')

    def test_a_restore_that_finds_other_settings_says_so(self):
        with work_dir() as root:
            raw = root / 'raw'
            raw.mkdir()
            write_raw_folder(raw)
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
                config = dict(FAST, RUN_NAME='stale', RAW_FOLDER=str(raw))
                run(config, through='spikes')
                fresh = new_state(build_config(dict(config, SPIKE_MAX_PASSES=9)))
                from upxo.meshing.gbconformant.pipeline import saved_results
                saved_results.restore(fresh)
                self.assertEqual(fresh.status['spikes'], StageStatus.STALE)
                self.assertIn('settings or inputs changed', fresh.stale_reasons['spikes']['reason'])
                self.assertIn('settings or inputs changed', fresh.stale_reason_text('spikes'))


class RewindTests(unittest.TestCase):
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

    def finished_stages(self, **settings):
        import numpy as np
        from upxo.meshing.gbconformant.pipeline import saved_results
        state = new_state(build_config(dict(FAST, RUN_NAME='rewind', RAW_FOLDER=str(self.raw), **settings)))

        def finish(stage, **artefacts):
            state.current_stage = stage
            for key, value in artefacts.items():
                state.store_artefact(key, value)
            state.reports[stage] = {'duration_s': 1.0}
            state.set_status(stage, StageStatus.OK)
            state.current_stage = None
            saved_results.save_after_run(state, stage)

        finish('load_lgi', labels=np.arange(4))
        finish('subset_extract')
        finish('grain_id_map')
        finish('small_grains', labels=np.arange(4) * 2, kept=1)
        finish('spikes', labels=np.arange(4) * 3)
        return state

    def test_dropping_a_stage_brings_back_the_copy_the_stage_before_it_saved(self):
        import numpy as np
        state = self.finished_stages()
        state.invalidate_from('spikes', 'edited')
        np.testing.assert_array_equal(state.get_artefact('labels'), np.arange(4) * 2)
        self.assertEqual(state.artefact_writer['labels'], 'small_grains')
        self.assertEqual(state.get_artefact('kept'), 1)
        state.invalidate_from('small_grains', 'edited')
        np.testing.assert_array_equal(state.get_artefact('labels'), np.arange(4))
        self.assertNotIn('kept', state.artefacts)                 # saved by a stage that was dropped

    def test_only_the_missing_results_are_read_back(self):
        from upxo.meshing.gbconformant.pipeline import saved_results
        state = self.finished_stages()
        before = state.get_artefact('kept')
        state.invalidate_from('spikes', 'edited')
        self.assertIs(state.get_artefact('kept'), before)           # still in memory: not read again
        self.assertEqual(saved_results.rewind(state, 'spikes'), [])  # nothing left to bring back

    def test_without_history_the_replaced_copies_are_gone_and_nothing_breaks(self):
        state = self.finished_stages(CHECKPOINT_KEEP_HISTORY=False)
        state.invalidate_from('spikes', 'edited')
        self.assertNotIn('labels', state.artefacts)
        self.assertEqual(state.status['spikes'], StageStatus.STALE)

    def test_a_changed_earlier_setting_stops_the_rewind(self):
        import numpy as np
        state = self.finished_stages()
        state.config['GRAIN_ID_SEED'] = 99            # grain_id_map's setting: everything after it is out of date
        state.invalidate_from('spikes', 'edited')
        np.testing.assert_array_equal(state.get_artefact('labels'), np.arange(4))   # only as far as the load stage


class UnitsTests(unittest.TestCase):
    def state_with_voxels(self, size, units):
        from types import SimpleNamespace
        state = PipelineState({}, stage_ids())
        state.store_artefact(HIERARCHY_KEY, SimpleNamespace(voxel_size=size, units=units))
        return state

    def test_settings_that_are_sizes_carry_a_unit_of_measure(self):
        self.assertEqual(SCHEMA['GMSH_SIZE'].unit, 'length')
        self.assertEqual(SCHEMA['GMSH_SIZE_FACTOR'].unit, 'factor')
        self.assertEqual(SCHEMA['MIN_GRAIN_VOXELS'].unit, 'voxels')
        self.assertEqual(SCHEMA['THIN_THICKNESS_VOXELS'].unit, 'thickness')
        self.assertEqual(SCHEMA['LAPLACE_RELAXATION'].unit, '')

    def test_lengths_factors_counts_and_thickness_read_in_the_units_of_the_voxel(self):
        state = self.state_with_voxels(0.001, 'mm')                   # 1 um voxels
        self.assertEqual(voxel_edge(state), (0.001, 'mm'))
        self.assertEqual(physical_text(state, 'GMSH_SIZE', 0.75), '0.75 x voxel edge = 0.75 um')
        self.assertEqual(physical_text(state, 'TET_MESH_SIZE_FACTOR', 2), '2 x voxel edge = 2 um')
        self.assertEqual(physical_text(state, 'MIN_GRAIN_VOXELS', 8), '8 voxels = 8 um^3')
        self.assertEqual(physical_text(state, 'THIN_THICKNESS_VOXELS', 3), '3 voxels across = 3 um')
        big = self.state_with_voxels(0.5, 'mm')
        self.assertEqual(physical_text(big, 'GMSH_SIZE', 2), '2 x voxel edge = 1 mm')
        self.assertEqual(physical_text(big, 'MIN_GRAIN_VOXELS', 8), '8 voxels = 1 mm^3')

    def test_nothing_is_said_without_a_voxel_size_or_a_value_or_a_unit(self):
        self.assertEqual(physical_text(PipelineState({}, stage_ids()), 'GMSH_SIZE', 1.0), '')
        state = self.state_with_voxels(0.001, 'mm')
        self.assertEqual(physical_text(state, 'GMSH_SIZE', None), '')
        self.assertEqual(physical_text(state, 'LAPLACE_RELAXATION', 0.75), '')
        self.assertEqual(physical_text(state, 'GMSH_SIZE', 'abc'), '')
        self.assertEqual(physical_text(self.state_with_voxels(1.0, 'furlongs'), 'GMSH_SIZE', 1), '')

    def test_format_length_picks_a_readable_unit(self):
        self.assertEqual(format_length(2.5e-6), '2.5 um')
        self.assertEqual(format_length(0.002), '2 mm')
        self.assertEqual(format_length(3.0), '3 m')


class RunListTests(unittest.TestCase):
    def test_runs_are_listed_newest_first_with_how_far_they_got(self):
        with work_dir() as root:
            raw = root / 'raw'
            raw.mkdir()
            write_raw_folder(raw)
            data = root / 'data'
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(data)}):
                run(dict(FAST, RUN_NAME='first', RAW_FOLDER=str(raw)), through='spikes')
                time.sleep(0.05)
                run(dict(FAST, RUN_NAME='second', RAW_FOLDER=str(raw), STAGE_SUBSET=True, SUBSET_MODE='index',
                         SUBSET_LO=[1, 0, 0], SUBSET_HI=[5, 4, 4]), through='small_grains')
                (data / 'runs' / 'unrelated').mkdir()
                runs = list_runs()
        self.assertEqual([r.name for r in runs], ['second_subset_001', 'second', 'first'])
        second, parent, first = runs
        self.assertEqual((parent.last_stage, parent.stages_saved), ('load_lgi', 1))       # saved before the crop
        self.assertEqual((second.parent, second.subset_index), ('second', 1))
        self.assertEqual(second.subset_box, 'x1-5  y0-4  z0-4')
        self.assertEqual(second.dataset, 'fm_steel')
        self.assertEqual(second.last_stage, 'small_grains')
        self.assertEqual(second.progress_text, 'stopped after Small-grain removal')
        self.assertFalse(second.finished)
        self.assertTrue(first.has_manifest)
        self.assertEqual(first.last_stage, 'spikes')
        self.assertGreater(first.saved_bytes, 0)
        self.assertRegex(first.modified_text, r'^\d{4}-\d\d-\d\d \d\d:\d\d$')

    def test_a_missing_directory_gives_an_empty_list_and_a_bare_run_folder_is_described(self):
        with work_dir() as root:
            self.assertEqual(list_runs(root / 'nope'), [])
            folder = root / 'runs' / 'bare'
            (folder / 'checkpoints' / 'load_lgi').mkdir(parents=True)
            (folder / 'checkpoints' / 'load_lgi' / 'meta.json').write_text('{}')
            info = describe_run(folder)
            self.assertEqual((info.stages_saved, info.last_stage, info.has_manifest), (1, 'load_lgi', False))
            self.assertEqual(info.progress_text, 'stopped after Load labelled grain image')
            self.assertEqual([r.name for r in list_runs(root / 'runs')], ['bare'])
            (folder / 'run_manifest.json').write_text('not json')
            self.assertFalse(describe_run(folder).has_manifest)


class StageSummaryTests(unittest.TestCase):
    def test_every_stage_that_has_a_status_gets_a_row_with_its_key_numbers(self):
        with work_dir() as root:
            raw = root / 'raw'
            raw.mkdir()
            write_raw_folder(raw)
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
                made = run(dict(FAST, RUN_NAME='notes', RAW_FOLDER=str(raw)), through='spikes')
            block = results.stage_summary_block(made.state)
            rows = {r[0]: r for r in block.rows}
            self.assertEqual(rows['Load labelled grain image'][1], 'OK')
            self.assertIn('5 grains', rows['Load labelled grain image'][3])
            self.assertIn('voxels changed', rows['Spike removal'][3])
            self.assertEqual(rows['Subset extract'][1], 'Skipped')
            self.assertEqual(rows['Subset extract'][3], 'switched off')
            self.assertNotIn('Pinch cleaning', rows)                  # never reached
            self.assertEqual(block.rows[-1][0], 'Total')
            self.assertIsNone(results.stage_summary_block(new_state(build_config())))
            self.assertIn('Stages', [b.title for b in results.summary_blocks(made.state)])

    def test_a_failed_stage_shows_its_error(self):
        with work_dir() as root:
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
                made = run(dict(FAST, RUN_NAME='bad', RAW_FOLDER=str(root / 'missing')), through='spikes')
            row = next(r for r in results.stage_summary_block(made.state).rows if r[0] == 'Load labelled grain image')
            self.assertEqual(row[1], 'Failed')
            self.assertIn('RAW_FOLDER', row[3])


if __name__ == '__main__':
    unittest.main()
