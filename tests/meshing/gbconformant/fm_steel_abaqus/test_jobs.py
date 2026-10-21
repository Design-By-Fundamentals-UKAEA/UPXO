"""Job scripts and the Abaqus data check (against a stand-in for Abaqus)."""
import json
import os
import stat
import sys
import unittest
from pathlib import Path
from unittest import mock

from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, jobs
from .helpers import work_dir
from .test_deck import build

FAKE = r'''
import os, sys
args = sys.argv[1:]
job = next(a.split('=', 1)[1] for a in args if a.startswith('job='))
mode = os.environ.get('FAKE_ABAQUS', 'ok')
if mode == 'sleep':
    import time
    time.sleep(30)
with open(job + '.dat', 'w') as handle:
    handle.write('Abaqus fake data file\n')
    handle.write('***WARNING: a warning about the deck\n')
    if mode == 'error':
        handle.write('***ERROR: THE ELEMENT SET ES_NOWHERE HAS NOT BEEN DEFINED\n')
with open('fake_args.txt', 'w') as handle:
    handle.write(' '.join(args))
sys.exit(3 if mode == 'crash' else 0)
'''


def make_fake_abaqus(folder):
    """A command that behaves like Abaqus for the data check; returns its path."""
    folder = Path(folder)
    (folder / 'fake_abaqus.py').write_text(FAKE, encoding='utf8')
    if os.name == 'nt':
        path = folder / 'abaqus.bat'
        path.write_text(f'@echo off\r\n"{sys.executable}" "{folder / "fake_abaqus.py"}" %*\r\n', encoding='ascii')
    else:
        path = folder / 'abaqus'
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{folder / "fake_abaqus.py"}" "$@"\n', encoding='ascii')
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def settings(**kwargs):
    return DeckSettings(job_scripts=True, **kwargs)


class ScriptTests(unittest.TestCase):
    def test_the_local_script_runs_the_job(self):
        text = jobs.script_texts(settings())['submit.sh']
        self.assertIn('abaqus job=fm_job input=model_master.inp cpus=4 interactive', text)
        self.assertTrue(text.startswith('#!/bin/bash\n'))
        self.assertIn('cd "$(dirname "$0")"', text)

    def test_the_user_subroutine_and_memory_are_passed_on(self):
        text = jobs.script_texts(settings(umat_file='/home/me/umat.f', job_cpus=8, job_memory_gb=16.0))['submit.sh']
        self.assertIn('user=/home/me/umat.f cpus=8 memory="16 gb" interactive', text)
        self.assertNotIn('No UMAT file is set', text)

    def test_a_path_with_a_blank_is_quoted(self):
        text = jobs.script_texts(settings(umat_file='C:\\My Folder\\umat.f', abaqus_command='C:\\Program Files\\abq\\abaqus.bat'))
        self.assertIn('"C:\\Program Files\\abq\\abaqus.bat"', text['submit.bat'])
        self.assertIn('user="C:\\My Folder\\umat.f"', text['submit.bat'])

    def test_without_a_user_subroutine_the_script_says_so(self):
        for name, text in jobs.script_texts(settings()).items():
            self.assertIn('No UMAT file is set (UMAT_FILE)', text, name)

    def test_the_windows_scripts_use_call_and_dos_line_ends(self):
        texts = jobs.script_texts(settings())
        self.assertIn('call abaqus job=fm_job', texts['submit.bat'])
        self.assertIn('\r\n', texts['submit.bat'])
        self.assertTrue(texts['submit.bat'].startswith('@echo off\r\n'))
        self.assertNotIn('\r', texts['submit.sh'])

    def test_the_data_check_script_runs_only_the_check(self):
        text = jobs.script_texts(settings(umat_file='/u.f', job_cpus=8, job_memory_gb=16.0))['datacheck.sh']
        self.assertIn('abaqus job=fm_job_datacheck input=model_master.inp user=/u.f datacheck cpus=1 interactive', text)
        self.assertNotIn('memory', text)

    def test_the_slurm_script_takes_its_options_from_the_settings(self):
        text = jobs.script_texts(settings(job_name='steel1', job_cpus=16, job_memory_gb=64.0, job_time_hours=36.0,
                                          slurm_partition='cpu,long', umat_file='/u.f'))['submit.slurm']
        for line in ('#SBATCH --job-name=steel1', '#SBATCH --cpus-per-task=16', '#SBATCH --mem=64G',
                     '#SBATCH --time=1-12:00:00', '#SBATCH --partition=cpu,long'):
            self.assertIn(line, text)
        self.assertIn('abaqus job=steel1 input=model_master.inp user=/u.f cpus=${SLURM_CPUS_PER_TASK} memory="64 gb" interactive',
                      text)

    def test_the_slurm_script_leaves_out_what_is_not_set(self):
        text = jobs.script_texts(settings())['submit.slurm']
        self.assertNotIn('--mem', text)
        self.assertNotIn('--partition', text)
        self.assertIn('#SBATCH --time=1-00:00:00', text)

    def test_slurm_times(self):
        self.assertEqual(jobs.slurm_time(5.5), '05:30:00')
        self.assertEqual(jobs.slurm_time(24), '1-00:00:00')
        self.assertEqual(jobs.slurm_time(0.01), '00:00:36')
        self.assertEqual(jobs.slurm_time(100), '4-04:00:00')

    def test_unusable_settings_are_refused(self):
        for bad in (dict(job_name='bad name'), dict(job_name='-x'), dict(job_name=''), dict(job_cpus=0),
                    dict(job_memory_gb=-1.0), dict(job_time_hours=0), dict(slurm_partition='a b'),
                    dict(umat_file='a"b'), dict(umat_file='a&b'), dict(umat_file='$(x)'), dict(abaqus_command='ab;aqus'),
                    dict(abaqus_command='  ')):
            with self.assertRaises(ValueError, msg=str(bad)):
                settings(**bad).validate()


class DeckScriptTests(unittest.TestCase):
    def test_the_deck_writes_the_scripts_beside_itself(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=settings(umat_file='/u.f'))
            deck = res['directory']
            for name in jobs.SCRIPT_NAMES:
                self.assertTrue((deck / name).is_file(), name)
                self.assertIn(name, res['files'])
            self.assertEqual(res['info']['job_scripts'], list(jobs.SCRIPT_NAMES))
            self.assertEqual(sorted(p.name for p in deck.iterdir()), sorted(res['files']))
            self.assertIn(b'\r\n', (deck / 'submit.bat').read_bytes())
            self.assertNotIn(b'\r', (deck / 'submit.sh').read_bytes())

    def test_no_scripts_unless_asked_for(self):
        with work_dir() as root:
            _, _, _, res = build(root)
            self.assertFalse(any((res['directory'] / n).exists() for n in jobs.SCRIPT_NAMES))
            self.assertEqual(res['info']['job_scripts'], [])


class DataCheckTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.fake = make_fake_abaqus(self.root)
        _, _, _, res = build(self.root, settings=settings(umat_file='/u.f'))
        self.deck = res['directory']

    def run_check(self, mode='ok', **kwargs):
        with mock.patch.dict(os.environ, {'FAKE_ABAQUS': mode}):
            return jobs.run_datacheck(self.deck, settings(umat_file='/u.f'), command=str(self.fake), **kwargs)

    def test_a_clean_check_passes_and_records_the_warnings(self):
        result = self.run_check()
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['warnings'], ['***WARNING: a warning about the deck'])
        self.assertEqual(result['job'], 'fm_job_datacheck')
        self.assertEqual(result['returncode'], 0)
        self.assertTrue((self.deck / 'fm_job_datacheck.dat').is_file())          # run in the deck folder
        self.assertEqual(json.loads((self.deck / jobs.DATACHECK_REPORT).read_text())['ok'], True)

    def test_abaqus_is_called_with_the_data_check_arguments(self):
        self.run_check()
        arguments = (self.deck / 'fake_args.txt').read_text()
        self.assertEqual(arguments, 'job=fm_job_datacheck input=model_master.inp user=/u.f datacheck cpus=1 interactive')

    def test_an_error_line_fails_the_check(self):
        result = self.run_check('error')
        self.assertFalse(result['ok'])
        self.assertEqual(result['errors'], ['***ERROR: THE ELEMENT SET ES_NOWHERE HAS NOT BEEN DEFINED'])

    def test_a_failing_exit_code_fails_the_check(self):
        result = self.run_check('crash')
        self.assertFalse(result['ok'])
        self.assertEqual(result['returncode'], 3)

    def test_no_abaqus_is_a_stated_problem_not_an_exception(self):
        result = jobs.run_datacheck(self.deck, settings(), command=str(self.root / 'nowhere'))
        self.assertFalse(result['ok'])
        self.assertIn('Abaqus was not found', result['problem'])
        self.assertIsNone(result['returncode'])
        self.assertTrue((self.deck / jobs.DATACHECK_REPORT).is_file())

    def test_a_check_that_takes_too_long_is_stopped(self):
        result = self.run_check('sleep', timeout=1)
        self.assertFalse(result['ok'])
        self.assertIn('did not finish in 1 s', result['problem'])

    def test_find_abaqus(self):
        self.assertEqual(Path(jobs.find_abaqus(str(self.fake))).resolve(), self.fake.resolve())
        self.assertIsNone(jobs.find_abaqus(''))
        self.assertIsNone(jobs.find_abaqus(str(self.root / 'nowhere')))

    def test_messages_are_read_from_the_data_message_and_log_files(self):
        (self.deck / 'j.dat').write_text('x\n***ERROR: one\n***WARNING: w1\n')
        (self.deck / 'j.msg').write_text('***ERROR: one\n***ERROR: two\nfine\n')
        (self.deck / 'j.log').write_text('Abaqus Error: three\n')
        errors, warnings = jobs.read_messages(self.deck, 'j')
        self.assertEqual(errors, ['***ERROR: one', '***ERROR: two', 'Abaqus Error: three'])
        self.assertEqual(warnings, ['***WARNING: w1'])


if __name__ == '__main__':
    unittest.main()
