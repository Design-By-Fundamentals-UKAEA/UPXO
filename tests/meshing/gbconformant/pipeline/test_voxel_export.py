"""The voxel deck stage: its sources, where it writes and which settings it shares with the tetrahedral deck."""
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config
from upxo.meshing.gbconformant.pipeline.stage_output import (
    handle_export_abaqus, handle_export_abaqus_voxel, voxel_deck_folder)
from upxo.meshing.gbconformant.pipeline.stages import stage_ids

from .test_results import meshed_state
from ..fm_steel_abaqus.helpers import work_dir


class VoxelStageRegistryTests(unittest.TestCase):
    def test_the_stage_follows_the_tetrahedral_export_and_is_off_by_default(self):
        ids = stage_ids()
        self.assertEqual(ids.index('export_abaqus_voxel'), ids.index('export_abaqus') + 1)
        config = default_config()
        self.assertIs(config['STAGE_EXPORT_VOXEL'], False)
        self.assertEqual(config['VOXEL_SOURCE'], 'cleaned labels')
        self.assertIs(config['VOXEL_RESCALE'], False)
        self.assertEqual(config['VOXEL_SCALE_FACTOR'], 1.0)

    def test_element_set_settings_decide_both_decks(self):
        for key in ('FM_PREFIX_PAG', 'EXPORT_BOUNDARY_LAYER', 'SET_PROVIDER', 'FM_LOAD_AXIS'):
            self.assertEqual(SCHEMA[key].stages[-2:], ('export_abaqus', 'export_abaqus_voxel'), key)
        for key in ('EXPORT_FM_DECK', 'EXPORT_MESH_ONLY', 'REQUIRE_READY_TET_MESH'):
            self.assertNotIn('export_abaqus_voxel', SCHEMA[key].stages, key)


class VoxelFolderTests(unittest.TestCase):
    def test_the_voxel_deck_takes_the_number_of_the_latest_conformal_export(self):
        with work_dir() as root:
            base = root / 'abaqus'
            for name in ('001_fm', '002_fm'):
                (base / name).mkdir(parents=True)
            self.assertEqual(voxel_deck_folder(base, base / '002_fm', True), (base / '002_voxel', '002'))
            self.assertEqual(voxel_deck_folder(base, None, True), (base / '002_voxel', '002'))
            (base / '002_voxel').mkdir()
            self.assertEqual(voxel_deck_folder(base, base / '002_fm', True), (base / '003_voxel', None))

    def test_time_stamped_runs_reuse_the_stamp(self):
        with work_dir() as root:
            base = root / 'abaqus'
            (base / '20261029_101500_000001').mkdir(parents=True)
            folder, stamp = voxel_deck_folder(base, base / '20261029_101500_000001', False)
            self.assertEqual(folder.name, '20261029_101500_000001_voxel')
            self.assertEqual(stamp, '20261029_101500_000001')


class VoxelStageTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict('os.environ', {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cleaned_labels_give_one_element_per_voxel_beside_the_conformal_deck(self):
        state = meshed_state(self.root, 'voxel')
        tet = handle_export_abaqus(state)
        state.store_artefact('abaqus_folder', Path(tet['folder']))
        report = handle_export_abaqus_voxel(state)
        labels = np.asarray(state.get_artefact('labels_meshing'))
        folder = Path(report['folder'])
        self.assertEqual(folder.parent, Path(tet['folder']).parent)
        self.assertEqual(folder.name, Path(tet['folder']).name.replace('_fm', '_voxel'))
        self.assertEqual(report['paired_with'], Path(tet['folder']).name)
        self.assertEqual(report['elements'], labels.size)
        self.assertTrue(report['deck_check']['ok'], report['deck_check']['failed'])
        self.assertEqual(report['voxel_source']['kind'], 'cleaned labels')
        self.assertIsNone(report['voxel_source']['resampled'])
        self.assertEqual(json.loads((folder / 'voxel_source.json').read_text())['kind'], 'cleaned labels')
        self.assertEqual(state.get_artefact('abaqus_voxel_folder'), folder)

    def test_the_scale_factor_applies_only_when_switched_on(self):
        state = meshed_state(self.root, 'rescale')
        labels = np.asarray(state.get_artefact('labels_meshing'))
        state.config['VOXEL_SCALE_FACTOR'] = 2.0
        self.assertEqual(handle_export_abaqus_voxel(state)['elements'], labels.size)
        state.config['VOXEL_RESCALE'] = True
        report = handle_export_abaqus_voxel(state)
        self.assertEqual(report['elements'], labels.size * 8)
        self.assertEqual(report['voxel_source']['resampled']['shape_after'], [2 * n for n in labels.shape])
        self.assertTrue(report['deck_check']['ok'], report['deck_check']['failed'])

    def test_the_tet_mesh_can_be_voxelized_instead(self):
        state = meshed_state(self.root, 'from_tets')
        state.config['VOXEL_SOURCE'] = 'voxelized tet mesh'
        report = handle_export_abaqus_voxel(state)
        labels = np.asarray(state.get_artefact('labels_meshing'))
        self.assertEqual(report['grid_shape'], list(labels.shape))
        self.assertEqual(report['voxel_source']['grains_lost'], [])
        self.assertTrue(report['deck_check']['ok'], report['deck_check']['failed'])

    def test_missing_inputs_are_explained(self):
        state = meshed_state(self.root, 'missing')
        state.config['VOXEL_SOURCE'] = 'voxelized tet mesh'
        state.store_artefact('verified_tets', None)
        with self.assertRaisesRegex(RuntimeError, 'verified_tets missing'):
            handle_export_abaqus_voxel(state)
        state.config['VOXEL_SOURCE'] = 'something else'
        with self.assertRaisesRegex(ValueError, 'VOXEL_SOURCE'):
            handle_export_abaqus_voxel(state)


if __name__ == '__main__':
    unittest.main()
