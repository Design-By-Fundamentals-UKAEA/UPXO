"""The reproduce command, the record beside a deck, and the job scripts and data check through the pipeline."""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from upxo.meshing.gbconformant.pipeline import build_config, provenance
from upxo.meshing.gbconformant.pipeline.cli import _collect_settings, build_parser
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config, differences
from upxo.meshing.gbconformant.pipeline import results
from upxo.meshing.gbconformant.pipeline.stage_output import handle_export_abaqus

from .test_cli import call
from .test_results import meshed_state
from ..fm_steel_abaqus.helpers import work_dir
from ..fm_steel_abaqus.test_jobs import make_fake_abaqus


def rebuilt_from(config):
    """The settings the reproduce command gives when it is run."""
    args = build_parser().parse_args(provenance.reproduce_arguments(config))
    return build_config(_collect_settings(args))


def assert_same_settings(case, config):
    original = build_config({k: v for k, v in config.items() if k in SCHEMA})
    rebuilt = rebuilt_from(config)
    derived = set(provenance.DERIVED) if config.get('DATASET_TYPE') != 'labels' else {'EFFECTIVE_SPACING'}
    left = {k: v for k, v in original.items() if k not in derived and not k.startswith('_')}
    right = {k: v for k, v in rebuilt.items() if k not in derived and not k.startswith('_')}
    case.assertEqual(differences(left, right), [])


class ArgumentTests(unittest.TestCase):
    def config(self, **settings):
        base = dict(DATASET_TYPE='fm_steel', RAW_FOLDER='C:\\data\\raw', HIERARCHY_LEVEL='blk')
        base.update(settings)
        return build_config(base)

    def test_the_input_the_level_and_the_changed_settings_are_arguments(self):
        tokens = provenance.reproduce_arguments(self.config(GRAIN_ID_SEED=42, LAPLACE_ITERATIONS=10))
        self.assertEqual(tokens[:5], ['run', '--raw-folder', 'C:\\data\\raw', '--level', 'blk'])
        self.assertIn('GRAIN_ID_SEED=42', tokens)
        self.assertIn('LAPLACE_ITERATIONS=10', tokens)

    def test_nothing_that_is_at_its_default_is_listed(self):
        tokens = provenance.reproduce_arguments(self.config())
        self.assertNotIn('--set', tokens)

    def test_the_crop_is_a_subset_box(self):
        tokens = provenance.reproduce_arguments(self.config(
            STAGE_SUBSET=True, SUBSET_MODE='index', SUBSET_LO=[10, 10, 10], SUBSET_HI=[32, 30, 30]))
        at = tokens.index('--subset-box')
        self.assertEqual(tokens[at + 1:at + 8], ['index', '10.0', '10.0', '10.0', '32.0', '30.0', '30.0'])
        self.assertFalse([t for t in tokens if t.startswith(('STAGE_SUBSET', 'SUBSET_'))])

    def test_a_crop_without_an_end_is_set_by_name(self):
        tokens = provenance.reproduce_arguments(self.config(STAGE_SUBSET=True, SUBSET_LO=[1, 2, 3]))
        self.assertNotIn('--subset-box', tokens)
        self.assertIn('STAGE_SUBSET=true', tokens)

    def test_the_other_input_types(self):
        twinned = provenance.reproduce_arguments(build_config(
            dict(DATASET_TYPE='twinned_fcc', TWINNED_FOLDER='C:\\twin', TWINNED_VOXEL_SIZE=0.5)))
        self.assertEqual(twinned[1:3], ['--twinned-folder', 'C:\\twin'])
        self.assertIn('TWINNED_VOXEL_SIZE=0.5', twinned)
        labels = provenance.reproduce_arguments(build_config(dict(DATASET_TYPE='labels', SAMPLE_PATH='C:\\x.npy',
                                                                  SPACING=[2.0, 2.0, 2.0])))
        self.assertEqual(labels[1:3], ['--labels', 'C:\\x.npy'])
        self.assertIn('SPACING=2.0,2.0,2.0', labels)

    def test_values_that_stages_fill_in_are_not_arguments(self):
        config = self.config(EFFECTIVE_SPACING=[0.5, 0.5, 0.5], SPACING=[1.0, 1.0, 1.0])
        config['_RUN_FOLDER'] = 'C:\\x'
        config['SOME_UNLISTED_SETTING'] = 1
        text = ' '.join(provenance.reproduce_arguments(config))
        self.assertNotIn('SPACING', text)
        self.assertNotIn('_RUN_FOLDER', text)
        self.assertNotIn('SOME_UNLISTED_SETTING', text)

    def test_the_command_gives_the_settings_it_came_from(self):
        cases = {
            'fm steel': dict(DATASET_TYPE='fm_steel', RAW_FOLDER='C:\\data\\raw', HIERARCHY_LEVEL='blk',
                             GRAIN_ID_SEED=42, STAGE_SUBSET=True, SUBSET_MODE='index', SUBSET_LO=[10, 10, 10],
                             SUBSET_HI=[32, 30, 30], REQUIRE_CLEAN_TOPOLOGY=True, GMSH_SIZE_FACTOR=0.5,
                             N_WORKERS=3, TET_MESH_SIZE=0.8, STAGE_FREEZE_THIN=False, UPSCALE_FACTOR=2,
                             FM_LOAD_MODE='strain', FM_LOAD_CASE='cyclic', FM_CYCLES=7, FM_R_RATIO=0.1,
                             FM_ELEMENT_OUTPUTS='LE, S', FM_HEADER_MESSAGE='my run', RUN_NAME='steel_a'),
            'twinned': dict(DATASET_TYPE='twinned_fcc', TWINNED_FOLDER='C:\\a b\\twin', TWINNED_UNITS='mm',
                            TWINNED_VOXEL_SIZE=0.25, EXPORT_MSH=False, UMAT_FILE='C:\\u s\\umat.f'),
            'labels': dict(DATASET_TYPE='labels', SAMPLE_PATH='C:\\x.npy', SPACING=[2.0, 2.0, 3.0]),
            'a list of numbers and a json setting': dict(DATASET_TYPE='fm_steel', RAW_FOLDER='C:\\r',
                                                         SMOOTH_ITERATIONS_BY_PAIR={'1-2': 5}),
        }
        for name, settings in cases.items():
            with self.subTest(name):
                assert_same_settings(self, build_config(settings))

    def test_paths_are_written_clean_and_absolute(self):
        folder = Path('relative_raw_for_test')
        folder.mkdir(exist_ok=True)
        try:
            config = self.config(RAW_FOLDER=str(folder))
            config['UMAT_FILE'] = '"C:\\users\\umat.f"'                  # pasted with its quotes, set without checks
            tokens = provenance.reproduce_arguments(config)
            self.assertEqual(tokens[2], str(folder.resolve()))
            self.assertIn('UMAT_FILE=C:\\users\\umat.f', tokens)
        finally:
            folder.rmdir()

    def test_the_command_line_quotes_what_needs_it(self):
        command = provenance.reproduce_command(self.config(RAW_FOLDER='C:\\my data\\raw', FM_HEADER_MESSAGE='a "b" c'))
        self.assertTrue(command.startswith('python -m upxo.meshing.gbconformant.pipeline run --raw-folder '))
        self.assertIn('"C:\\my data\\raw"', command)
        self.assertIn('"FM_HEADER_MESSAGE=a \\"b\\" c"', command)

    def test_a_value_is_written_as_it_is_read(self):
        self.assertEqual(provenance.format_value(SCHEMA['REQUIRE_CLEAN_TOPOLOGY'], True), 'true')
        self.assertEqual(provenance.format_value(SCHEMA['N_WORKERS'], None), 'auto')
        self.assertEqual(provenance.format_value(SCHEMA['SUBSET_LO'], [1, 2.5, 3]), '1,2.5,3')


class RecordTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def export(self, name='prov', **settings):
        state = meshed_state(self.root, name)
        state.config.update(settings)
        return state, handle_export_abaqus(state)

    def test_the_deck_has_a_record_with_the_command_the_inputs_and_the_versions(self):
        state, report = self.export(GRAIN_ID_SEED=42)
        deck = Path(report['fm_deck']['folder'])
        record = json.loads((deck / 'provenance.json').read_text())
        self.assertEqual(record['format'], provenance.FORMAT)
        self.assertEqual(record['command'], provenance.reproduce_command(state.config))
        self.assertEqual(record['seeds']['grain_id_seed'], 42)
        self.assertIn('raw_export.pkl', [row[0] for row in record['input']['files']])
        self.assertIn('numpy', record['environment'])
        keys = [row['key'] for row in record['settings_changed_from_defaults']]
        self.assertIn('GRAIN_ID_SEED', keys)
        self.assertNotIn('SPACING', keys)

    def test_the_record_can_be_switched_off(self):
        _, report = self.export('noprov', EXPORT_PROVENANCE=False)
        self.assertFalse((Path(report['fm_deck']['folder']) / 'provenance.json').exists())

    def test_the_deck_has_job_scripts_that_follow_the_settings(self):
        _, report = self.export('jobs', JOB_NAME='steel1', JOB_CPUS=12, UMAT_FILE='"C:\\u\\umat.f"', JOB_MEMORY_GB=32.0,
                                SLURM_PARTITION='cpu')
        deck = Path(report['fm_deck']['folder'])
        text = (deck / 'submit.slurm').read_text()
        self.assertIn('--job-name=steel1', text)
        self.assertIn('user=C:\\u\\umat.f', text)                       # the quotes pasted with the path are gone
        self.assertIn('--partition=cpu', text)
        self.assertEqual(report['fm_deck']['job_scripts'][0], 'submit.bat')

    def test_the_scripts_can_be_switched_off(self):
        _, report = self.export('nojobs', JOB_SCRIPTS=False)
        self.assertFalse((Path(report['fm_deck']['folder']) / 'submit.sh').exists())

    def test_a_job_name_that_cannot_be_written_stops_the_export(self):
        with self.assertRaisesRegex(ValueError, 'job_name'):
            self.export('badjob', JOB_NAME='bad name')

    def test_the_deliverables_list_the_record_and_the_scripts(self):
        state, report = self.export('deliv')
        state.reports['export_abaqus'] = report
        labels = [item['label'] for item in results.deliverables(state)]
        for label in ('Record to reproduce the deck', 'Job script (Windows)', 'Job script (Linux)', 'Job script (Slurm)'):
            self.assertIn(label, labels)


class CommandTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        state = meshed_state(self.root, 'cmd')
        state.config.update(GRAIN_ID_SEED=5)
        self.state = state
        self.deck = Path(handle_export_abaqus(state)['fm_deck']['folder'])

    def test_reproduce_prints_the_stored_command(self):
        code, out, _err = call('reproduce', str(self.deck))
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), json.loads((self.deck / 'provenance.json').read_text())['command'])
        self.assertIn('GRAIN_ID_SEED=5', out)

    def test_reproduce_rebuilds_the_command_from_a_manifest(self):
        from upxo.meshing.gbconformant.pipeline.manifest import write_run_manifest
        run_folder = Path(write_run_manifest(self.state)).parent
        code, out, _err = call('reproduce', str(run_folder))
        self.assertEqual(code, 0)
        self.assertIn('--raw-folder', out)
        self.assertIn('GRAIN_ID_SEED=5', out)

    def test_reproduce_needs_a_folder_with_a_record(self):
        empty = self.root / 'empty'
        empty.mkdir()
        code, _out, err = call('reproduce', str(empty))
        self.assertEqual(code, 2)
        self.assertIn('neither', err)

    def test_the_data_check_command_runs_abaqus_and_reports(self):
        fake = make_fake_abaqus(self.root)
        with mock.patch.dict(os.environ, {'FAKE_ABAQUS': 'ok'}):
            code, out, _err = call('datacheck', str(self.deck), '--abaqus', str(fake))
        self.assertEqual(code, 0, out)
        self.assertIn('data check: passed', out)
        self.assertIn('1 warning(s)', out)
        with mock.patch.dict(os.environ, {'FAKE_ABAQUS': 'error'}):
            code, out, _err = call('datacheck', str(self.deck), '--abaqus', str(fake))
        self.assertEqual(code, 1)
        self.assertIn('data check: FAILED', out)
        self.assertIn('ES_NOWHERE', out)

    def test_the_data_check_without_abaqus_says_so(self):
        code, out, _err = call('datacheck', str(self.deck), '--abaqus', str(self.root / 'nowhere'))
        self.assertEqual(code, 1)
        self.assertIn('data check not run: Abaqus was not found', out)

    def test_the_data_check_needs_a_deck(self):
        code, _out, err = call('datacheck', str(self.root))
        self.assertEqual(code, 2)
        self.assertIn('model_master.inp', err)


if __name__ == '__main__':
    unittest.main()
