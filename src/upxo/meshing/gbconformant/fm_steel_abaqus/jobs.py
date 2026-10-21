"""Scripts that run the deck with Abaqus, and Abaqus's own data check of it.

``write_job_scripts`` puts these beside the deck (the values come from the deck settings):

* ``submit.bat`` and ``submit.sh``: run the job on this machine;
* ``datacheck.bat`` and ``datacheck.sh``: the same with the Abaqus data check only;
* ``submit.slurm``: a Slurm batch script with the job name, CPUs, memory and time limit.

Every script runs ``abaqus job=<name> input=model_master.inp`` with the user subroutine file (``user=``) when one is
set. ``run_datacheck`` runs the data check where Abaqus is installed and reads the error and warning lines Abaqus
writes into the job's files. None of this has been run against a particular Abaqus installation by UPXO: the
scripts are the commands, written out.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess
import time

INPUT_NAME = 'model_master.inp'
DATACHECK_REPORT = 'abaqus_datacheck.json'
SCRIPT_NAMES = ('submit.bat', 'submit.sh', 'datacheck.bat', 'datacheck.sh', 'submit.slurm')
_JOB_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,79}$')
_PARTITION = re.compile(r'^[A-Za-z0-9_\-,]+$')
_FORBIDDEN = set('"\r\n`$%;&|<>')


def validate_job(settings):
    """Raise ``ValueError`` when a job setting cannot be written into a script safely."""
    if not _JOB_NAME.match(str(settings.job_name)):
        raise ValueError('job_name may hold letters, digits, underscore and hyphen, and must start with a letter or digit.')
    if int(settings.job_cpus) < 1:
        raise ValueError('job_cpus must be >= 1.')
    if not float(settings.job_memory_gb) >= 0:
        raise ValueError('job_memory_gb must be >= 0 (0: not set).')
    if not float(settings.job_time_hours) > 0:
        raise ValueError('job_time_hours must be > 0.')
    if settings.slurm_partition and not _PARTITION.match(str(settings.slurm_partition)):
        raise ValueError('slurm_partition may hold letters, digits, underscore, hyphen and comma.')
    for name in ('abaqus_command', 'umat_file'):
        value = str(getattr(settings, name) or '')
        if set(value) & _FORBIDDEN:
            raise ValueError(f'{name} must not contain quotes, line ends or any of ` $ % ; & | < >.')
    if not str(settings.abaqus_command).strip():
        raise ValueError('abaqus_command must not be empty.')


def slurm_time(hours):
    """Slurm's ``days-hours:minutes:seconds`` for a time in hours."""
    seconds = int(round(float(hours) * 3600))
    days, rest = divmod(seconds, 86400)
    h, rest = divmod(rest, 3600)
    m, s = divmod(rest, 60)
    return f'{days}-{h:02d}:{m:02d}:{s:02d}' if days else f'{h:02d}:{m:02d}:{s:02d}'


def abaqus_arguments(settings, *, datacheck=False, cpus=None, memory=True):
    """The arguments after the Abaqus command, as ``key=value`` tokens (a data check runs on one CPU, without memory)."""
    job = f'{settings.job_name}_datacheck' if datacheck else str(settings.job_name)
    tokens = [f'job={job}', f'input={INPUT_NAME}']
    if settings.umat_file:
        tokens.append(f'user={settings.umat_file}')
    if datacheck:
        tokens += ['datacheck', 'cpus=1']
    else:
        tokens.append(f'cpus={settings.job_cpus if cpus is None else cpus}')
        if memory and float(settings.job_memory_gb) > 0:
            tokens.append(f'memory={settings.job_memory_gb:g} gb')
    tokens.append('interactive')
    return tokens


def _quote(token):
    """``key=value`` with the value in double quotes when it holds a blank."""
    key, sep, value = token.partition('=')
    if sep and (' ' in value or '\t' in value):
        return f'{key}="{value}"'
    if not sep and ' ' in token:
        return f'"{token}"'
    return token


def _command_line(settings, *, datacheck, cpus=None, windows=False):
    command = str(settings.abaqus_command).strip()
    command = f'"{command}"' if ' ' in command else command
    parts = [('call ' if windows else '') + command] + [_quote(t) for t in abaqus_arguments(settings, datacheck=datacheck, cpus=cpus)]
    return ' '.join(parts)


def _umat_note(settings, comment):
    if settings.umat_file:
        return []
    return [f'{comment} No UMAT file is set (UMAT_FILE): add user=<your UMAT file> to the command if the deck uses *User Material.']


def script_texts(settings):
    """``{file name: text}`` of the job scripts."""
    validate_job(settings)
    job = settings.job_name
    texts = {}
    for datacheck in (False, True):
        stem = 'datacheck' if datacheck else 'submit'
        what = 'the Abaqus data check of' if datacheck else 'the job'
        bat = ['@echo off', f'rem Runs {what} {job} on this machine. Written with the deck; edit as needed.',
               *_umat_note(settings, 'rem'), 'cd /d "%~dp0"', _command_line(settings, datacheck=datacheck, windows=True), '']
        sh = ['#!/bin/bash', f'# Runs {what} {job} on this machine. Written with the deck; edit as needed.',
              *_umat_note(settings, '#'), 'cd "$(dirname "$0")" || exit 1', _command_line(settings, datacheck=datacheck), '']
        texts[f'{stem}.bat'] = '\r\n'.join(bat)
        texts[f'{stem}.sh'] = '\n'.join(sh)
    slurm = ['#!/bin/bash', f'#SBATCH --job-name={job}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             f'#SBATCH --cpus-per-task={int(settings.job_cpus)}']
    if float(settings.job_memory_gb) > 0:
        slurm.append(f'#SBATCH --mem={settings.job_memory_gb:g}G')
    slurm.append(f'#SBATCH --time={slurm_time(settings.job_time_hours)}')
    if settings.slurm_partition:
        slurm.append(f'#SBATCH --partition={settings.slurm_partition}')
    slurm += ['', '# Submit with: sbatch submit.slurm', *_umat_note(settings, '#'),
              'cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1',
              _command_line(settings, datacheck=False, cpus='${SLURM_CPUS_PER_TASK}'), '']
    texts['submit.slurm'] = '\n'.join(slurm)
    return texts


def write_job_scripts(folder, settings):
    """Write the job scripts into ``folder``; returns their file names."""
    folder = Path(folder)
    written = []
    for name, text in script_texts(settings).items():
        path = folder / name
        with path.open('w', encoding='ascii', newline='') as handle:
            handle.write(text)
        if name.endswith(('.sh', '.slurm')):
            try:
                path.chmod(0o755)
            except OSError:
                pass
        written.append(name)
    return written


# --- the data check ---------------------------------------------------------------------------------------
def find_abaqus(command):
    """Full path of the Abaqus command (``None`` when it is not found on the PATH or at that path)."""
    command = str(command or '').strip()
    if not command:
        return None
    return shutil.which(command)


def read_messages(folder, job):
    """``(errors, warnings)``: the ``***ERROR`` and ``***WARNING`` lines in the job's data, message and log files."""
    errors, warnings = [], []
    for extension in ('.dat', '.msg', '.log'):
        path = Path(folder) / f'{job}{extension}'
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            text = line.strip()
            if text.startswith('***ERROR') or text.startswith('Abaqus Error'):
                errors.append(text)
            elif text.startswith('***WARNING'):
                warnings.append(text)
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def run_datacheck(folder, settings, *, command=None, timeout=3600):
    """Run the Abaqus data check on the deck in ``folder``; returns (and writes ``abaqus_datacheck.json``) the outcome.

    ``ok`` means Abaqus exited with 0 and its files hold no ``***ERROR`` line. When Abaqus cannot be started, or
    runs out of time, ``ok`` is false and ``problem`` says why.
    """
    validate_job(settings)
    folder = Path(folder)
    job = f'{settings.job_name}_datacheck'
    executable = find_abaqus(command or settings.abaqus_command)
    result = {'ok': False, 'job': job, 'command': None, 'returncode': None, 'seconds': None, 'errors': [],
              'warnings': [], 'problem': None}
    if executable is None:
        result['problem'] = f'Abaqus was not found (command {command or settings.abaqus_command!r}).'
    else:
        args = [executable, *abaqus_arguments(settings, datacheck=True)]
        result['command'] = ' '.join(_quote(a) for a in args)
        start = time.monotonic()
        try:
            done = subprocess.run(args, cwd=folder, capture_output=True, text=True, timeout=timeout)
            result['returncode'] = done.returncode
            for stream in (done.stdout, done.stderr):
                for line in (stream or '').splitlines():
                    if line.strip().startswith('Abaqus Error'):
                        result['errors'].append(line.strip())
        except subprocess.TimeoutExpired:
            result['problem'] = f'The data check did not finish in {timeout} s.'
        except OSError as error:
            result['problem'] = f'Abaqus could not be started: {error}'
        result['seconds'] = round(time.monotonic() - start, 2)
        errors, warnings = read_messages(folder, job)
        result['errors'] = list(dict.fromkeys(result['errors'] + errors))
        result['warnings'] = warnings
        result['ok'] = result['problem'] is None and result['returncode'] == 0 and not result['errors']
    (folder / DATACHECK_REPORT).write_text(json.dumps(result, indent=2), encoding='utf8')
    return result
