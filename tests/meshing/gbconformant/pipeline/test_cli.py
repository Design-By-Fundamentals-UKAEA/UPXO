import contextlib
import io
import json
import os
import unittest
from unittest import mock

from upxo.meshing.gbconformant.pipeline import stage_ids
from upxo.meshing.gbconformant.pipeline.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder

FAST = ['--set', 'STAGE_FREEZE_THIN=no', '--set', 'REQUIRE_CLEAN_TOPOLOGY=no',
        '--set', 'LAPLACE_ITERATIONS=2', '--set', 'JUNCTION_ITERATIONS=1',
        '--set', 'SPIKE_MAX_PASSES=5', '--set', 'PINCH_MAX_PASSES=2',
        '--set', 'TOPOLOGY_MAX_PASSES=2']


def call(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ListingTests(unittest.TestCase):
    def test_stages_are_listed_in_order(self):
        code, out, _ = call('stages')
        self.assertEqual(code, EXIT_OK)
        positions = [out.index(stage) for stage in stage_ids()]
        self.assertEqual(positions, sorted(positions))

    def test_settings_are_listed_for_one_stage_or_all(self):
        code, out, _ = call('settings', '--stage', 'small_grains')
        self.assertEqual(code, EXIT_OK)
        self.assertIn('MIN_GRAIN_VOXELS', out)
        self.assertNotIn('SPIKE_MAX_PASSES', out)
        code, out, _ = call('settings', '--json')
        rows = json.loads(out)
        names = {row['name'] for row in rows}
        self.assertIn('SPIKE_MAX_PASSES', names)
        row = next(r for r in rows if r['name'] == 'UPSCALE_FACTOR')
        self.assertEqual((row['minimum'], row['maximum'], row['default']), (1, 3, 1))
        self.assertEqual(call('settings', '--stage', 'nope')[0], EXIT_USAGE)


class RunTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        write_raw_folder(self.raw)
        self.data = self.root / 'data'
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.data)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *extra, raw=None):
        return call('run', '--raw-folder', str(raw or self.raw), '--run-name', 'cli_run',
                    '--through', 'spikes', *FAST, *extra)

    def test_run_prints_each_stage_and_the_folder(self):
        code, out, err = self.run_cli()
        self.assertEqual(code, EXIT_OK, err + out)
        self.assertIn('Load labelled grain image', out)
        self.assertIn('Spike removal: done', out)
        self.assertIn('run folder:', out)
        self.assertIn('manifest:', out)
        self.assertTrue((self.data / 'runs' / 'cli_run' / 'run_manifest.json').is_file())

    def test_second_run_restores_and_json_output_says_so(self):
        self.run_cli()
        code, out, err = self.run_cli('--json')
        self.assertEqual(code, EXIT_OK, err)
        summary = json.loads(out)
        self.assertTrue(summary['ok'])
        self.assertEqual(summary['ran'], [])
        self.assertIn('spikes', summary['restored'])
        code, out, _ = self.run_cli('--json', '--no-resume')
        self.assertEqual(json.loads(out)['restored'], [])

    def test_quiet_prints_nothing_on_success(self):
        code, out, _ = self.run_cli('-q')
        self.assertEqual((code, out), (EXIT_OK, ''))

    def test_verbose_prints_progress_messages(self):
        code, out, _ = self.run_cli('-v', '--no-resume')
        self.assertEqual(code, EXIT_OK)
        self.assertIn('Spike', out)

    def test_a_failing_stage_gives_the_failure_exit_code(self):
        code, out, _ = self.run_cli(raw=self.root / 'missing')
        self.assertEqual(code, EXIT_FAILED)
        self.assertIn('FAILED', out)
        self.assertIn('stopped at load_lgi', out)

    def test_bad_settings_give_the_usage_exit_code(self):
        for extra in (('--set', 'NOT_A_SETTING=1'), ('--set', 'MIN_GRAIN_VOXELS=0'),
                      ('--set', 'oops')):
            with self.subTest(extra=extra):
                code, _, err = self.run_cli(*extra)
                self.assertEqual(code, EXIT_USAGE, err)
        code, _, err = call('run', '--raw-folder', str(self.raw), '--through', 'nope')
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn('nope', err)

    def test_a_subset_box_needs_its_seven_values(self):
        with self.assertRaises(SystemExit) as caught:
            call('run', '--raw-folder', str(self.raw), '--subset-box', 'index', '1', '0', '0')
        self.assertEqual(caught.exception.code, 2)

    def test_settings_file_and_command_line_values_combine(self):
        path = self.root / 'settings.json'
        path.write_text(json.dumps({'config_version': 2, 'config': {'SPIKE_MAX_PASSES': 3,
                                                                      'MIN_GRAIN_VOXELS': 2}}))
        code, out, err = self.run_cli('--settings', str(path), '--set', 'MIN_GRAIN_VOXELS=3', '--json')
        self.assertEqual(code, EXIT_OK, err)
        manifest = json.loads((self.data / 'runs' / 'cli_run' / 'run_manifest.json').read_text())
        self.assertEqual(manifest['config']['MIN_GRAIN_VOXELS'], 3)        # --set wins
        self.assertEqual(manifest['config']['SPIKE_MAX_PASSES'], 5)        # FAST comes after the file

    def test_subset_box_crops_the_input(self):
        code, out, err = self.run_cli('--subset-box', 'index', '1', '0', '0', '5', '4', '4', '--json')
        self.assertEqual(code, EXIT_OK, err)
        summary = json.loads(out)
        self.assertTrue(summary['run_folder'].endswith('cli_run_subset_001'))

    def test_status_lists_the_saved_stages(self):
        self.run_cli()
        code, out, _ = call('status', str(self.data / 'runs' / 'cli_run'))
        self.assertEqual(code, EXIT_OK)
        self.assertIn('load_lgi', out)
        self.assertIn('spikes', out)
        self.assertIn('total', out)
        self.assertEqual(call('status', str(self.root / 'nowhere'))[0], EXIT_USAGE)
        empty = self.root / 'empty'
        empty.mkdir()
        self.assertIn('no saved stage results', call('status', str(empty))[1])


if __name__ == '__main__':
    unittest.main()
