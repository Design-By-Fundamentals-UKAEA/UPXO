"""The tet stages write verified_tets.vtu and .json to the run folder only when WRITE_STAGE_TET_FILES is on."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from upxo.meshing.gbconformant.pipeline import stage_volume
from upxo.meshing.gbconformant.pipeline.settings import SETTINGS, default_config


class StageTetFileTests(unittest.TestCase):
    def test_setting_is_off_by_default_and_belongs_to_no_stage(self):
        self.assertIs(default_config()['WRITE_STAGE_TET_FILES'], False)
        setting = next(s for s in SETTINGS if s.key == 'WRITE_STAGE_TET_FILES')
        self.assertEqual(setting.stages, ())          # a convenience copy: toggling it never makes a stage stale

    def test_nothing_is_written_by_default(self):
        state = SimpleNamespace(config={})
        with mock.patch('upxo.meshing.gbconformant.d3v2p1.gmsh_tets.save_grain_tetrahedra') as save, \
                mock.patch.object(stage_volume, 'ensure_run_dir') as run_dir:
            self.assertIsNone(stage_volume._save_tets(state, SimpleNamespace(report={})))
        save.assert_not_called()
        run_dir.assert_not_called()

    def test_written_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            state = SimpleNamespace(config={'WRITE_STAGE_TET_FILES': True})
            tets = SimpleNamespace(report={'status': 'ok'})
            with mock.patch('upxo.meshing.gbconformant.d3v2p1.gmsh_tets.save_grain_tetrahedra') as save, \
                    mock.patch.object(stage_volume, 'ensure_run_dir', return_value=Path(folder)):
                path = stage_volume._save_tets(state, tets)
            self.assertEqual(path, str(Path(folder) / 'verified_tets.vtu'))
            save.assert_called_once_with(tets, Path(folder) / 'verified_tets.vtu')
            self.assertTrue((Path(folder) / 'verified_tets.json').is_file())


if __name__ == '__main__':
    unittest.main()
