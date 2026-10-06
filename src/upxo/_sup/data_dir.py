"""Default folder for files that UPXO pipelines write.

A source checkout writes to <checkout>/data. An installed package writes to
./data under the working directory, never inside the Python environment.
"""
from pathlib import Path


def default_data_dir():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'src' / 'upxo').is_dir():
            return parent / 'data'
    return Path.cwd() / 'data'
