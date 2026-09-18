"""Ensemble Seed Config -- Part K of the FM Steel 3D walkthrough.

FM Steel's GUI has no reusable core ensemble generator -- EnsembleSeedConfigPage
itself is, per its own docstring, "a pedestal, not the ensemble generator
itself": it just dumps the run's parameters to a JSON "seed" file for a
future ensemble tool to consume. This module replicates that JSON schema
at notebook scope (schema_version 1, matching
gui/ensemble_seed_config.py's CURRENT_SCHEMA_VERSION) using the notebook's
own already-built parameter dict, rather than a GUI shared_state.
"""
import datetime
import json
import re
from pathlib import Path

CURRENT_SCHEMA_VERSION = 1


def write_seed_config(params, out_dir, seed_name="seed", notes="", pipeline_mode="block"):
    """Writes `params` (a plain dict of this run's parameters, however the
    notebook has organized it) to `<out_dir>/<seed_name>_<idx>.json`
    (auto-incrementing suffix, exclusive-create -- never overwrites an
    existing file).

    Returns
    -------
    pathlib.Path : the file written.
    """
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", seed_name.strip() or "seed")
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed_name": name,
        "notes": notes,
        "pipeline_mode": pipeline_mode,
        "config": params,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in out_dir.glob(f"{name}_*.json"):
        m = re.search(r"_(\d+)\.json$", f.name)
        if m:
            existing.add(int(m.group(1)))
    idx = 0
    while idx in existing:
        idx += 1
    path = out_dir / f"{name}_{idx}.json"
    with open(path, "x") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
