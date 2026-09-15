"""External sample/run paths; no generated data is written into the package."""
import os
from pathlib import Path


def data_directory():
    """Use UPXO_CONFORMAL_DATA, or the checkout's data directory.

    Installed distributions without a checkout default to a data directory
    beneath the working directory. Samples must be supplied separately there.
    """
    configured = os.environ.get('UPXO_CONFORMAL_DATA')
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent/'pyproject.toml').is_file() and (parent/'src/upxo').is_dir():
            return parent/'data/conformalMeshing3DData'
    return Path.cwd()/'data/conformalMeshing3DData'


def sample_path(name='blk_lgi.npy'):
    if Path(name).name != name: raise ValueError('Sample name must be a filename')
    return data_directory()/'samples'/name


def run_directory(name):
    if not name or Path(name).name != name or name in ('.','..'):
        raise ValueError('Run name must be a single directory name')
    return data_directory()/'runs'/name
