"""Raw pipeline-state export.

Gathers whatever grain-structure / hierarchy data currently exists across
the pipeline's stage objects -- fm_base, fm_with_pags, fm_with_blocks,
fm_with_orientations, fm_with_subblocks (see gui/pipeline_state.py's
STAGE_ORDER) -- and writes it out as plain files. fm_base's own data (the
grain-ID array and its immediate companions) is always included. On top
of that, the caller selects which of the four hierarchy LEVELS to include
-- PAG, Packet, Block, Sub-block (mirroring the PAG/PCK/BLK/SBLK elset
levels on the Abaqus mesh-export page) -- each with its own filename
prefix. Every selected level contributes its own 3D labeled array
("<prefix>_lgi", the same shape as the base structure, built via
feature_props_3d's per-level LGI builders -- the same ones the
hierarchy visualizations already use) alongside its other hierarchy
metadata, and, when it has orientation data, that gets written as its
OWN dedicated pickle file (<prefix>_orientations.pkl) rather than folded
into the shared bundle, since a level's orientations are the one thing
about it most likely to be consumed on its own downstream.

Three output formats are supported: .npy (one file per array), pickle
(one raw_export.pkl holding every non-array/array field together, plus
one dedicated *_orientations.pkl per level that has orientation data),
and .npz (one compressed archive bundling every array together). Every
non-underscore __slots__ field of the relevant stage object is collected
automatically rather than hand-listed per class, so this stays correct if
a stage class's own fields ever change.
"""
import pickle
from pathlib import Path

import numpy as np

# (level key, display label) -- mirrors gui/pages_mesh.py's PAG/PCK/BLK/SBLK
# elset levels exactly, so a user already familiar with that naming
# recognizes these immediately.
LEVELS = [
    ("pag",  "PAG"),
    ("pck",  "Packet"),
    ("blk",  "Block"),
    ("sblk", "Sub-block"),
]
DEFAULT_LEVEL_PREFIXES = {"pag": "pag", "pck": "pck", "blk": "blk", "sblk": "sblk"}


def _public_fields(obj):
    """{field_name: value} for every non-underscore __slots__ field an
    object exposes (skips '_parent' and other private/internal fields --
    each stage object already wraps the one before it, so following
    '_parent' would just re-collect the same upstream data again)."""
    field_names = getattr(obj, "__slots__", None) or list(vars(obj).keys())
    out = {}
    for field in field_names:
        if field.startswith("_"):
            continue
        try:
            out[field] = getattr(obj, field)
        except AttributeError:
            continue
    return out


def level_availability(app):
    """{level_key: bool} -- whether each of the 4 hierarchy levels has
    been reached yet this session. Packet and Block both key off
    fm_with_blocks (a packet, in this pipeline, IS a grain viewed at the
    block-hierarchy scale -- grain_to_blocks_map is packet membership;
    it and block_orientations/all_blocks appear together the moment
    blocks exist)."""
    fm_pag  = getattr(app, "_fm_with_pags", None)
    fm_blk  = getattr(app, "_fm_with_blocks", None)
    fm_sblk = getattr(app, "_fm_with_subblocks", None)
    return {
        "pag":  fm_pag is not None,
        "pck":  fm_blk is not None,
        "blk":  fm_blk is not None,
        "sblk": fm_sblk is not None,
    }


def collect_base_data(app):
    """{"base_<field>": value} for fm_base's own fields -- the foundation
    every export includes unconditionally, independent of level
    selection. Empty if fm_base doesn't exist yet."""
    fm_base = getattr(app, "_fm_base", None)
    if fm_base is None:
        return {}
    return {f"base_{k}": v for k, v in _public_fields(fm_base).items()}


def collect_level_data(app, level_key):
    """Returns (orientations, fields) for one hierarchy level:
    `orientations` is that level's own {id: (phi1, Phi, phi2)} dict (or
    None if not available/derivable yet), `fields` is
    {field_name: value} for its other, non-orientation hierarchy data --
    always including "lgi", a 3D array the same shape as the base grain
    structure with every voxel stamped by that level's own ID (built via
    feature_props_3d's per-level LGI builders -- the same ones the 2D/3D
    hierarchy visualizations already use), plus, for Block and Sub-block
    (whose IDs are strings, not plain integers), "lgi_id_to_int" /
    "lgi_int_to_id" so the array's integer labels can be mapped back to
    the real block/sub-block IDs used elsewhere (all_blocks, ...).
    Both return values are empty/None if the level hasn't been reached
    yet."""
    from upxo.pxtal.fm_steel_3d.feature_props_3d import (
        _build_pag_lgi, _build_packet_lgi, _build_block_lgi, _build_subblock_lgi)

    fm_pag = getattr(app, "_fm_with_pags", None)
    fm_blk = getattr(app, "_fm_with_blocks", None)
    fm_ori = getattr(app, "_fm_with_orientations", None)
    fm_sblk = getattr(app, "_fm_with_subblocks", None)

    if level_key == "pag":
        if fm_pag is None:
            return None, {}
        fields = _public_fields(fm_pag)
        orientations = fields.pop("pag_orientations", None)
        fields["lgi"], _, _ = _build_pag_lgi(fm_pag)
        return orientations, fields

    if level_key == "pck":
        # Packet membership (grain_to_blocks_map, grain_to_plane_idx) lives
        # on fm_with_blocks (packet_id == grain_id in this model -- see
        # orientation_mean_3d.compute_packet_mean_orientations). A packet's
        # own orientation isn't stored anywhere -- it's the
        # crystallographically-correct mean of its constituent blocks'
        # orientations, so it needs fm_with_orientations too. Its "lgi" is
        # literally the base grain LGI (packets ARE grains, viewed at the
        # block hierarchy's own scale) -- _build_packet_lgi documents this.
        if fm_blk is None:
            return None, {}
        all_fields = _public_fields(fm_blk)
        fields = {k: v for k, v in all_fields.items()
                 if k in ("grain_to_blocks_map", "grain_to_plane_idx")}
        fields["lgi"], _, _ = _build_packet_lgi(fm_blk)
        orientations = None
        g2b = fields.get("grain_to_blocks_map")
        block_ori = getattr(fm_ori, "block_orientations", None) if fm_ori is not None else None
        if g2b and block_ori:
            from upxo.pxtal.fm_steel_3d.orientation_mean_3d import (
                compute_packet_mean_orientations)
            orientations = compute_packet_mean_orientations(g2b, block_ori)
        return orientations, fields

    if level_key == "blk":
        if fm_blk is None:
            return None, {}
        all_fields = _public_fields(fm_blk)
        fields = {k: v for k, v in all_fields.items()
                 if k in ("all_blocks", "block_slicing_normals", "slicing_planes")}
        lgi, s2i, i2s = _build_block_lgi(fm_blk)
        fields["lgi"] = lgi
        fields["lgi_id_to_int"] = s2i
        fields["lgi_int_to_id"] = i2s
        orientations = None
        if fm_ori is not None:
            ori_fields = _public_fields(fm_ori)
            orientations = ori_fields.get("block_orientations") or None
            if ori_fields.get("block_to_variant_idx") is not None:
                fields["block_to_variant_idx"] = ori_fields["block_to_variant_idx"]
        return orientations, fields

    if level_key == "sblk":
        if fm_sblk is None:
            return None, {}
        fields = _public_fields(fm_sblk)
        orientations = fields.pop("subblock_orientations", None)
        lgi, s2i, i2s = _build_subblock_lgi(fm_sblk)
        fields["lgi"] = lgi
        fields["lgi_id_to_int"] = s2i
        fields["lgi_int_to_id"] = i2s
        return orientations, fields

    raise ValueError(f"Unknown level key: {level_key!r}")


def export_raw_data(app, out_dir, formats, levels=None, prefixes=None):
    """Writes fm_base's data plus the requested hierarchy `levels`
    (any subset of {"pag", "pck", "blk", "sblk"}; default: every level
    currently available, see level_availability) to `out_dir`, in the
    requested `formats` (any non-empty subset of {"npy", "pkl", "npz"}):

    - "npy": one <name>.npy per collected entry that is a real numpy
      ndarray.
    - "npz": a single raw_export.npz bundling every ndarray entry
      together via numpy.savez_compressed.
    - "pkl": a single raw_export.pkl holding every collected entry
      (arrays and non-array hierarchy metadata alike), PLUS one
      dedicated "<prefix>_orientations.pkl" per requested level that has
      orientation data -- pickle is the only format here that can hold
      a plain {id: euler} dict at all, so this is gated on "pkl" being
      selected, same as the merged bundle.

    `prefixes`: {level_key: filename_prefix}: overrides
    DEFAULT_LEVEL_PREFIXES for the given level(s); every field this
    level contributes to the merged bundle is named "<prefix>_<field>".

    Returns {"out_dir": str, "files": {filename: size_bytes},
    "n_entries": int, "levels_exported": [level_key, ...],
    "levels_skipped": {level_key: reason}}.
    Raises ValueError if fm_base doesn't exist yet.
    """
    out_dir = Path(out_dir)
    base_data = collect_base_data(app)
    if not base_data:
        raise ValueError(
            "Nothing to export yet -- generate the base grain structure first.")

    avail = level_availability(app)
    if levels is None:
        levels = [k for k, ok in avail.items() if ok]
    resolved_prefixes = dict(DEFAULT_LEVEL_PREFIXES)
    resolved_prefixes.update(prefixes or {})

    data = dict(base_data)
    orientation_payloads = {}  # prefix -> {id: euler} dict
    levels_exported = []
    levels_skipped = {}
    for level_key in levels:
        if not avail.get(level_key, False):
            levels_skipped[level_key] = "not generated yet"
            continue
        prefix = resolved_prefixes.get(level_key, level_key)
        orientations, fields = collect_level_data(app, level_key)
        for field_name, value in fields.items():
            data[f"{prefix}_{field_name}"] = value
        if orientations:
            orientation_payloads[prefix] = orientations
        levels_exported.append(level_key)

    array_entries = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    written = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    if "npy" in formats:
        for name, arr in array_entries.items():
            fp = out_dir / f"{name}.npy"
            np.save(fp, arr)
            written[fp.name] = fp.stat().st_size

    if "npz" in formats and array_entries:
        fp = out_dir / "raw_export.npz"
        np.savez_compressed(fp, **array_entries)
        written[fp.name] = fp.stat().st_size

    if "pkl" in formats:
        fp = out_dir / "raw_export.pkl"
        with open(fp, "wb") as f:
            pickle.dump(data, f)
        written[fp.name] = fp.stat().st_size

        for prefix, ori_dict in orientation_payloads.items():
            fp = out_dir / f"{prefix}_orientations.pkl"
            with open(fp, "wb") as f:
                pickle.dump(ori_dict, f)
            written[fp.name] = fp.stat().st_size

    return {
        "out_dir": str(out_dir),
        "files": written,
        "n_entries": len(data) + len(orientation_payloads),
        "levels_exported": levels_exported,
        "levels_skipped": levels_skipped,
    }
