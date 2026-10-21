"""Ensemble Seed Config -- Part K of the FM Steel 3D walkthrough.

Writes a run's parameters to a JSON "seed" file for a later ensemble tool to
consume; this step records the parameters, it does not generate an ensemble.
The file format, its loader and its writer are in
``upxo.pxtal.fm_steel_3d.ensemble_seed_config``; this step passes the
notebook's own already-built parameter dict to the writer.
"""
from upxo.pxtal.fm_steel_3d.ensemble_seed_config import (
    CURRENT_SCHEMA_VERSION, write_ensemble_seed_config)


def write_seed_config(params, out_dir, seed_name="seed", notes="", pipeline_mode="block"):
    """Writes `params` (a plain dict of this run's parameters, however the
    notebook has organized it) to `<out_dir>/<seed_name>_<idx>.json`
    (auto-incrementing suffix, exclusive-create -- never overwrites an
    existing file). The file can be read back with
    ``ensemble_seed_config.load_ensemble_seed_config``.

    Returns
    -------
    pathlib.Path : the file written.
    """
    return write_ensemble_seed_config(
        params, out_dir, seed_name=seed_name, notes=notes, pipeline_mode=pipeline_mode)
