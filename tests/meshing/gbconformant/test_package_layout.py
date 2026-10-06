"""Compatibility and external-data path contracts after package relocation."""
import os
from pathlib import Path
from unittest.mock import patch
import unittest


class LayoutTests(unittest.TestCase):
    def test_data_override(self):
        from upxo.meshing.gbconformant.d3v2p0.paths import sample_path, run_directory
        with patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(Path.cwd()/'custom_mesh_data')}):
            base = (Path.cwd()/'custom_mesh_data').resolve()
            self.assertEqual(sample_path('input.npy'), base/'samples/input.npy')
            self.assertEqual(run_directory('demo'), base/'runs/demo')
            with self.assertRaises(ValueError): run_directory('../escape')
