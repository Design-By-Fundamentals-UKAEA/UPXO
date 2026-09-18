"""Compatibility and external-data path contracts after package relocation."""
import importlib
import os
from pathlib import Path
from unittest.mock import patch
import unittest


class LayoutTests(unittest.TestCase):
    @unittest.skip(
        "confMesh3D_01/ and cleaving/ are gitignored (stray duplicate / "
        "superseded, migration WIP) and don't exist in a fresh checkout; "
        "re-enable once the cleaving3dV1P0 migration is committed."
    )
    def test_old_imports_share_module_identity(self):
        for old, new, child in (
            ('confMesh3D_01', 'd3v2p0', 'interfaces'),
            ('cleaving', 'cleaving3dV1P0', 'config'),
        ):
            alias = importlib.import_module('upxo.meshing.'+old+'.'+child)
            canonical = importlib.import_module('upxo.meshing.gbconformant.'+new+'.'+child)
            self.assertIs(alias, canonical)

    def test_data_override(self):
        from upxo.meshing.gbconformant.d3v2p0.paths import sample_path, run_directory
        with patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(Path.cwd()/'custom_mesh_data')}):
            base = (Path.cwd()/'custom_mesh_data').resolve()
            self.assertEqual(sample_path('input.npy'), base/'samples/input.npy')
            self.assertEqual(run_directory('demo'), base/'runs/demo')
            with self.assertRaises(ValueError): run_directory('../escape')
