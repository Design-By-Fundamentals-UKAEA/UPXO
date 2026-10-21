"""Ensemble seed-config files: a pipeline run's configuration, written and read back.

A seed-config file records the full configuration of a completed pipeline run
(``config``) and a first-pass grouping of its keys by pipeline stage
(``injection_points``), so that a later tool can vary the parameters of one
stage while holding the others fixed, e.g. re-roll the PAG texture and keep
the base-grain morphology. The groupings in ``INJECTION_POINT_KEYS`` are not
exhaustive: a key that is not listed still appears in ``config``.

Schema evolution: whenever a field is added, renamed or removed, bump
``CURRENT_SCHEMA_VERSION`` and add a ``_migrate_v{N}_to_v{N+1}(payload)``
function to ``_MIGRATIONS``, so that files written by an older version keep
loading.
"""
import datetime
import json
import re
from pathlib import Path

__all__ = ['CURRENT_SCHEMA_VERSION', 'REQUIRED_KEYS', 'INJECTION_POINT_KEYS',
           'EnsembleSeedConfigError', 'load_ensemble_seed_config',
           'list_ensemble_seed_configs', 'write_ensemble_seed_config']

CURRENT_SCHEMA_VERSION = 1

REQUIRED_KEYS = ("config", "injection_points", "pipeline_mode")

INJECTION_POINT_KEYS = {
    "base_grain": [
        "pipeline_mode", "base_grain_method",
        "LX", "LY", "LZ", "NX", "NY", "NZ", "force_isotropic",
        "N_SEEDS", "RANDOM_SEED", "VERBOSITY", "CONNECTIVITY",
        "MIN_GRAIN_NVOXELS", "GS_CLEAN_MERGE_MIN_NVOXELS", "GRAIN_CLEANUP_MAX_PASSES",
        "voronoi_type_left", "voronoi_type_right",
        "gs_clean_do_merge", "gs_clean_do_spikes", "gs_clean_iterations",
        "MC_NX", "MC_NY", "MC_NZ", "MC_NUM_STATES_Q", "MC_ALGORITHM", "MC_MCSTEPS",
        "MC_SAVE_INTERVAL", "MC_PRINT_INTERVAL", "MC_CONSIDER_BOLTZMANN",
        "MC_BOLTZMANN_Q_RELATED", "MC_BOLTZMANN_TEMP_FACTOR", "MC_BASE_RNG_SEED",
        "MC_RANDOMIZE_GRAIN_IDS", "mc_tslice",
        "ebsd_file_path", "ebsd_spec_path", "rep_metric",
        "ref_mode", "ref_ebsd_file_path", "ref_ebsd_spec_path",
        "ref_prop_area", "ref_prop_aspect_ratio", "ref_prop_perimeter",
        "ref_prop_solidity", "ref_prop_n_neighbours",
        "ref_group_pag", "ref_group_packets", "ref_group_blocks", "ref_group_subblocks",
    ],
    "pag_orientation": [
        "pag_algorithm", "pag_min_size", "pag_max_size", "pag_min_prob", "pag_max_prob",
        "PAG_GRAIN_FRACTION", "USE_NON_NEIGH_PAG",
        "ISOLATED_GRAIN_STRATEGY", "ISOLATED_SIZE_TOL",
        "PAG_ORI_MODE", "HAGB_THRESHOLD", "HAGB_MAX_ATTEMPTS",
        "FIXED_PAG_PHI1", "FIXED_PAG_PHI", "FIXED_PAG_PHI2",
        "PAG_ORI_POOL_SOURCE", "PAG_ORI_POOL_SIZE", "PAG_TC_SPREAD", "PAG_TEX_HAGB_CHK",
        "PAG_TC_COPPER", "PAG_TC_COPPER_FRAC", "PAG_TC_BRASS", "PAG_TC_BRASS_FRAC",
        "PAG_TC_S", "PAG_TC_S_FRAC", "PAG_TC_GOSS", "PAG_TC_GOSS_FRAC",
        "PAG_TC_CUBE", "PAG_TC_CUBE_FRAC",
        "PAG_SAMPLE_SYM", "PAG_SAMPLE_SYM_RD", "PAG_SAMPLE_SYM_TD", "PAG_SAMPLE_SYM_ND",
        "ISOLATED_ORI_MODE", "ISOLATED_HAGB_THRESHOLD", "ISOLATED_HAGB_MAX_ATTEMPTS",
        "ISOLATED_ORI_POOL_SIZE", "ISOLATED_USER_TCS",
        "ISOLATED_SAMPLE_SYM", "ISOLATED_SAMPLE_SYM_RD",
        "ISOLATED_SAMPLE_SYM_TD", "ISOLATED_SAMPLE_SYM_ND",
    ],
    "block_orientation": [
        "blk_thk_dist", "blk_thk_dist_file",
        # block_thickness_min/max start from a default that depends on the
        # pipeline mode, but the value that matters is the one set at the
        # block-generation stage, so they are grouped here.
        "block_thickness_min", "block_thickness_max", "BLOCK_SLAB_CONNECTIVITY",
        "KS_VARIANT_SELECTION", "KS_ORI_SEED", "BLK_CUSTOM_ORI_FILE",
    ],
    "subblock": [
        "subblock_thickness_min", "subblock_thickness_max",
        "INTRABLOCK_ORI_SPREAD_DEG", "THIN_BLOCK_STRATEGY",
        "SUBBLOCK_SLAB_CONNECTIVITY", "SB_ORI_SEED",
    ],
    "mesh": [
        "EXPORT_MESH", "MESH_ELEM_TYPE", "MESH_OUTPUT_UNIT", "folder_prefix",
        "MESH_CUSTOM_MESSAGE", "MESH_FORCE", "MESH_IGNORE_IDS",
        "RESCALE_FACTOR", "RESCALE_METHOD", "MESH_MAX_VOXELS",
        "output_base_directory", "material_association_level",
        "elset_pag", "elset_pck", "elset_blk", "elset_subblk", "elset_iso",
        "elset_pag_gbsegs", "elset_pag_jlsegs", "elset_pag_jp",
        "elset_pck_gbsegs", "elset_pck_jlsegs", "elset_pck_jp",
        "elset_blk_gbsegs", "elset_blk_jlsegs", "elset_blk_jp",
        "elset_sblk_gbsegs", "elset_sblk_jlsegs", "elset_sblk_jp",
        "prefix_pag", "prefix_pck", "prefix_blk", "prefix_subblk", "prefix_iso",
        "prefix_pag_gbsegs", "prefix_pag_jlsegs", "prefix_pag_jp",
        "prefix_pck_gbsegs", "prefix_pck_jlsegs", "prefix_pck_jp",
        "prefix_blk_gbsegs", "prefix_blk_jlsegs", "prefix_blk_jp",
        "prefix_sblk_gbsegs", "prefix_sblk_jlsegs", "prefix_sblk_jp",
        "pag_gb_thick_vox", "pag_gbjl_thick_vox", "pag_gbjp_thick_vox",
        "pck_gb_thick_vox", "pck_gbjl_thick_vox", "pck_gbjp_thick_vox",
        "blk_gb_thick_vox", "blk_gbjl_thick_vox", "blk_gbjp_thick_vox",
        "sblk_gb_thick_vox", "sblk_gbjl_thick_vox", "sblk_gbjp_thick_vox",
        "nset_x_min", "nset_x_max", "nset_y_min", "nset_y_max",
        "nset_z_min", "nset_z_max", "nset_origins", "nset_rve_edges",
        "prefix_x_min", "prefix_x_max", "prefix_y_min", "prefix_y_max",
        "prefix_z_min", "prefix_z_max", "prefix_origins", "prefix_rve_edges",
    ],
}


class EnsembleSeedConfigError(Exception):
    """Raised when a seed-config file can't be loaded or understood."""


# Registry of migrations: {from_version: fn(payload_at_from_version) -> payload_at_from_version+1}
# Empty for now: schema_version 1 is the only version that has ever existed.
# Example once schema_version 2 exists:
#     def _migrate_v1_to_v2(payload):
#         payload = dict(payload)
#         payload["injection_points"].setdefault("void", {})
#         payload["schema_version"] = 2
#         return payload
#     _MIGRATIONS = {1: _migrate_v1_to_v2}
_MIGRATIONS = {}


def load_ensemble_seed_config(path):
    """Load, validate, and (if needed) migrate an ensemble seed-config JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        The payload, guaranteed to be at CURRENT_SCHEMA_VERSION and to contain
        at least: schema_version, generated_at, seed_name, notes,
        pipeline_mode, config, injection_points.

    Raises
    ------
    EnsembleSeedConfigError
        If the file is missing, malformed, missing required keys, was written
        by a newer schema version than this code understands, or no
        migration path exists from its schema_version to the current one.
    """
    p = Path(path)
    if not p.exists():
        raise EnsembleSeedConfigError(f"Seed-config file not found: {p}")

    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise EnsembleSeedConfigError(
            f"Could not read/parse seed-config file {p}: {exc}") from exc

    if not isinstance(payload, dict):
        raise EnsembleSeedConfigError(
            f"Seed-config file {p} does not contain a JSON object.")

    version = payload.get("schema_version")
    if version is None:
        raise EnsembleSeedConfigError(
            f"Seed-config file {p} has no 'schema_version' — not a recognised "
            f"ensemble seed-config file.")

    if version > CURRENT_SCHEMA_VERSION:
        raise EnsembleSeedConfigError(
            f"Seed-config file {p} was written by schema_version {version}, but "
            f"this code only understands up to {CURRENT_SCHEMA_VERSION}. Update "
            f"UPXO before loading this file.")

    while version < CURRENT_SCHEMA_VERSION:
        migrate = _MIGRATIONS.get(version)
        if migrate is None:
            raise EnsembleSeedConfigError(
                f"No migration registered from schema_version {version} to "
                f"{version + 1} — cannot upgrade {p}.")
        payload = migrate(payload)
        new_version = payload.get("schema_version", version + 1)
        if new_version <= version:
            raise EnsembleSeedConfigError(
                f"Migration from schema_version {version} did not advance the "
                f"version (still {new_version}) — refusing to loop forever on {p}.")
        version = new_version

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise EnsembleSeedConfigError(
            f"Seed-config file {p} is missing required key(s): {', '.join(missing)}.")

    return payload


def list_ensemble_seed_configs(directory):
    """Return sorted Path objects for every *.json file in `directory` that
    looks like an ensemble seed-config (has a readable 'schema_version').
    Unreadable/unrelated .json files are silently skipped, not raised.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    found = []
    for f in sorted(d.glob("*.json")):
        try:
            load_ensemble_seed_config(f)
            found.append(f)
        except EnsembleSeedConfigError:
            continue
    return found


def write_ensemble_seed_config(config, out_dir, seed_name="seed", notes="",
                               pipeline_mode=None, injection_points=None):
    """Write a seed-config file ``<out_dir>/<seed_name>_<n>.json``.

    ``n`` is the lowest index not yet used for this name, and the file is
    created exclusively, so an existing file is never overwritten. Characters
    of ``seed_name`` other than letters, digits, ``_`` and ``-`` become ``_``.

    Parameters
    ----------
    config : dict
        The full configuration of the run (written as ``config``).
    out_dir : str or Path
        Created if it does not exist.
    pipeline_mode : str, optional
        Taken from ``config['pipeline_mode']`` (else 'block') when not given.
    injection_points : dict, optional
        ``{stage: {key: value}}``. By default the keys of ``config`` listed in
        ``INJECTION_POINT_KEYS`` for each stage.

    Returns
    -------
    pathlib.Path : the file written.

    Raises
    ------
    OSError
        If the directory or file can't be created.
    """
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", str(seed_name).strip() or "seed")
    if pipeline_mode is None:
        pipeline_mode = config.get("pipeline_mode", "block")
    if injection_points is None:
        injection_points = {
            stage: {k: config.get(k) for k in keys if k in config}
            for stage, keys in INJECTION_POINT_KEYS.items()
        }
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed_name": name,
        "notes": notes,
        "pipeline_mode": pipeline_mode,
        "config": dict(config),
        "injection_points": injection_points,
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
    # Exclusive creation: if another process takes this index between the scan
    # above and this open(), FileExistsError is raised instead of overwriting.
    with open(path, "x") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
