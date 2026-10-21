"""The twinned FCC dataset type through the pipeline: load, early stages, export, command line."""
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from upxo.meshing.gbconformant.pipeline import build_config, run
from upxo.meshing.gbconformant.pipeline.preflight import check_stage_preflight
from upxo.meshing.gbconformant.pipeline.run import new_state
from upxo.meshing.gbconformant.pipeline.runner import PipelineRunner
from upxo.meshing.gbconformant.pipeline.saved_results import input_signature
from upxo.meshing.gbconformant.pipeline.stage_output import handle_export_abaqus
from upxo.meshing.gbconformant.pipeline.stage_input import active_hierarchy

from .test_cli import call
from ..fm_steel_abaqus.helpers import voxel_tets, work_dir, write_twinned_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2, GRAIN_ID_SEED=3)


class TwinnedPipelineTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = self.root / 'twinned'
        self.folder.mkdir()
        self.labels, *_ = write_twinned_folder(self.folder)

    def config(self, **settings):
        merged = dict(FAST, RUN_NAME='twin', DATASET_TYPE='twinned_fcc', TWINNED_FOLDER=str(self.folder),
                      TWINNED_VOXEL_SIZE=0.5, TWINNED_UNITS='mm')
        merged.update(settings)
        return merged

    def test_the_load_stage_reads_the_folder(self):
        state = new_state(build_config(self.config(STAGE_GRAIN_ID_SHUFFLE=False)))
        PipelineRunner(state).run_stage('load_lgi')
        report = state.reports['load_lgi']
        self.assertEqual(report['dataset_type'], 'twinned_fcc')
        self.assertEqual((report['source_units'], report['voxel_size']), ('mm', 0.5))
        self.assertEqual(report['grains_by_role']['es_role_host'], 2)
        self.assertEqual(report['host_families'], 2)
        np.testing.assert_array_equal(state.get_artefact('labels'), self.labels)
        self.assertEqual(active_hierarchy(state).system, 'twinned FCC')

    def test_the_early_stages_run_and_a_second_run_restores(self):
        first = run(self.config(), through='spikes')
        self.assertTrue(first.ok, first.error)
        second = run(self.config(), through='spikes')
        self.assertEqual(second.ran, [])
        self.assertIn('load_lgi', second.restored)

    def test_the_folder_is_part_of_the_input_signature(self):
        names = [row[0] for row in input_signature(self.config())]
        self.assertIn('lgi_twinned_clean.npy', names)
        self.assertIn('twin_role.pkl', names)

    def test_a_changed_input_file_makes_the_load_stage_stale(self):
        run(self.config(), through='small_grains')
        np.save(self.folder / 'lgi_twinned_clean.npy', np.transpose(self.labels, (2, 1, 0))[:, :, :5])
        again = run(self.config(), through='small_grains')
        self.assertIn('load_lgi', again.ran)

    def test_loading_without_a_folder_says_what_to_set(self):
        state = new_state(build_config(self.config(TWINNED_FOLDER='')))
        issue = check_stage_preflight(state, 'load_lgi')
        self.assertIsNotNone(issue)
        self.assertIn('TWINNED_FOLDER', issue.message)

    def test_a_missing_folder_fails_the_load_stage_clearly(self):
        result = run(self.config(TWINNED_FOLDER=str(self.root / 'nowhere')), through='load_lgi')
        self.assertFalse(result.ok)
        self.assertIn('TWINNED_FOLDER', result.error)

    def test_the_export_writes_the_deck_with_the_default_extras(self):
        state = new_state(build_config(self.config(STAGE_GRAIN_ID_SHUFFLE=False)))
        PipelineRunner(state).run_stage('load_lgi')
        labels = np.asarray(state.get_artefact('labels'))
        points, tets, grains = voxel_tets(labels)
        state.store_artefact('labels_meshing', labels.copy())
        state.store_artefact('verified_tets', SimpleNamespace(
            points=points, tetrahedra=tets, grain_ids=grains, quality=np.full(len(tets), 0.5),
            report={'ready': True, 'status': 'TET_MESH_VERIFIED'}))
        report = handle_export_abaqus(state)
        deck = Path(report['fm_deck']['folder'])
        names = sorted(p.name for p in deck.iterdir())
        for expected in ('03e_elsets_groups.inp', '04b_nsets_boundaries.inp', 'fm_mesh.msh', 'fm_mesh.vtu'):
            self.assertIn(expected, names)
        info = json.loads((deck / 'deck_info.json').read_text())
        self.assertEqual(info['system'], 'twinned FCC')
        self.assertEqual(info['coordinate_scale'], 0.5)
        self.assertIn('es_cell_host_2', (deck / '03a_elsets_pag.inp').read_text())

    def test_the_export_switches_turn_the_extras_off(self):
        state = new_state(build_config(self.config(STAGE_GRAIN_ID_SHUFFLE=False, EXPORT_MSH=False, EXPORT_VTU=False,
                                                   EXPORT_BOUNDARY_SETS=False)))
        PipelineRunner(state).run_stage('load_lgi')
        labels = np.asarray(state.get_artefact('labels'))
        points, tets, grains = voxel_tets(labels)
        state.store_artefact('labels_meshing', labels.copy())
        state.store_artefact('verified_tets', SimpleNamespace(
            points=points, tetrahedra=tets, grain_ids=grains, quality=np.full(len(tets), 0.5),
            report={'ready': True, 'status': 'TET_MESH_VERIFIED'}))
        deck = Path(handle_export_abaqus(state)['fm_deck']['folder'])
        names = {p.name for p in deck.iterdir()}
        self.assertFalse(names & {'fm_mesh.msh', 'fm_mesh.vtu', '04b_nsets_boundaries.inp'})

    def test_the_command_line_takes_the_folder(self):
        code, out, _err = call('run', '--twinned-folder', str(self.folder), '--through', 'load_lgi',
                               '--set', 'TWINNED_UNITS=m', '--set', 'TWINNED_VOXEL_SIZE=1e-6',
                               '--data-dir', str(self.root / 'cli'), '--json')
        self.assertEqual(code, 0, out)
        manifest = json.loads(Path(json.loads(out)['manifest']).read_text())
        self.assertEqual(manifest['config']['DATASET_TYPE'], 'twinned_fcc')
        self.assertEqual(manifest['config']['TWINNED_UNITS'], 'm')


if __name__ == '__main__':
    unittest.main()
