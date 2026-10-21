"""Pasted paths with quotes are accepted everywhere; output files carry neutral names."""
import os
import unittest
from unittest import mock

from upxo.meshing.gbconformant.pipeline import build_config, run
from upxo.meshing.gbconformant.pipeline.saved_results import effective_level, input_signature
from upxo.meshing.gbconformant.pipeline.settings import clean_path_text, parse_assignment

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder, write_twinned_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2, GRAIN_ID_SEED=3)


class CleanPathTests(unittest.TestCase):
    def test_windows_copy_as_path_loses_its_quotes(self):
        self.assertEqual(clean_path_text('"C:\\Data\\my folder\\raw"'), 'C:\\Data\\my folder\\raw')

    def test_blanks_line_ends_and_single_quotes_go_too(self):
        self.assertEqual(clean_path_text('  "C:\\a b"\r\n'), 'C:\\a b')
        self.assertEqual(clean_path_text("'C:\\a'"), 'C:\\a')
        self.assertEqual(clean_path_text('\u201cC:\\a\u201d'), 'C:\\a')
        self.assertEqual(clean_path_text(' " "C:\\a" " '), 'C:\\a')

    def test_a_path_without_quotes_is_unchanged_and_inner_blanks_stay(self):
        self.assertEqual(clean_path_text('C:\\Program Files\\x'), 'C:\\Program Files\\x')
        self.assertEqual(clean_path_text("C:\\Bob's data"), "C:\\Bob's data")
        self.assertEqual(clean_path_text(''), '')
        self.assertEqual(clean_path_text(None), '')

    def test_a_stray_double_quote_at_one_end_goes(self):
        self.assertEqual(clean_path_text('"C:\\a'), 'C:\\a')
        self.assertEqual(clean_path_text('C:\\a"'), 'C:\\a')


class SettingsTests(unittest.TestCase):
    def test_the_path_settings_are_cleaned_when_a_config_is_built(self):
        config = build_config(RAW_FOLDER='"C:\\raw"', TWINNED_FOLDER=" 'C:\\twin' ", SAMPLE_PATH='"C:\\x.npy"')
        self.assertEqual((config['RAW_FOLDER'], config['TWINNED_FOLDER'], config['SAMPLE_PATH']),
                         ('C:\\raw', 'C:\\twin', 'C:\\x.npy'))

    def test_a_set_option_on_the_command_line_is_cleaned(self):
        self.assertEqual(parse_assignment('RAW_FOLDER="C:\\raw"'), ('RAW_FOLDER', 'C:\\raw'))

    def test_other_text_settings_keep_their_quotes(self):
        self.assertEqual(build_config(FM_HEADER_MESSAGE='"quoted"')['FM_HEADER_MESSAGE'], '"quoted"')


class RunWithQuotedPathsTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_raw_folder_pasted_with_quotes_loads_and_is_part_of_the_signature(self):
        raw = self.root / 'raw folder'
        raw.mkdir()
        write_raw_folder(raw)
        quoted = f'"{raw}"'
        names = [row[0] for row in input_signature({'DATASET_TYPE': 'fm_steel', 'RAW_FOLDER': quoted})]
        self.assertIn('raw_export.pkl', names)
        self.assertEqual(effective_level({'RAW_FOLDER': quoted}), 'blk')
        result = run(dict(FAST, RUN_NAME='quoted', RAW_FOLDER=quoted), through='small_grains')
        self.assertTrue(result.ok, result.error)

    def test_a_twinned_folder_pasted_with_quotes_loads(self):
        folder = self.root / 'twinned'
        folder.mkdir()
        write_twinned_folder(folder)
        result = run(dict(FAST, RUN_NAME='twin', DATASET_TYPE='twinned_fcc', TWINNED_FOLDER=f'"{folder}"'),
                     through='load_lgi')
        self.assertTrue(result.ok, result.error)


class NeutralNameTests(unittest.TestCase):
    def test_the_files_a_run_writes_carry_no_notebook_names(self):
        with work_dir() as root:
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
                raw = root / 'raw'
                raw.mkdir()
                write_raw_folder(raw)
                result = run(dict(FAST, RUN_NAME='names', RAW_FOLDER=str(raw)), through='pinches')
                self.assertTrue(result.ok, result.error)
                names = [p.name for p in result.run_folder.iterdir()]
                self.assertTrue(names)
                self.assertFalse([n for n in names if 'CM0' in n], names)
                self.assertIn('small_grains.json', names)
                self.assertIn('pinch_cleaning.json', names)

    def test_no_stage_message_names_a_notebook(self):
        import inspect
        from upxo.meshing.gbconformant.pipeline import (
            stage_cleaning, stage_output, stage_surface, stage_volume)
        for module in (stage_cleaning, stage_output, stage_surface, stage_volume):
            self.assertNotIn('CM04', inspect.getsource(module), module.__name__)


if __name__ == '__main__':
    unittest.main()
