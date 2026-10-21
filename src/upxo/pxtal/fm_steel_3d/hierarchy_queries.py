"""Lookups over the PAG / packet / block / sub-block hierarchy of a pipeline state.

The functions take any pipeline state that has the attributes they read;
attributes that are missing count as empty. The levels are ``'pag'``,
``'block'``, ``'subblock'`` and ``'grain'``. A packet is identified by the
id of its grain.
"""
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

from .phases_3d import PHASE_MARTENSITE, PHASE_NAMES, PHASE_RETAINED_AUSTENITE

__all__ = ['LEVELS', 'pag_of_items', 'phase_of_items', 'voxels_for_level',
           'orientations_for_level', 'collect_ids_and_orientations',
           'packet_ids_for_pags', 'pag_ids_for_blocks', 'packet_ids_for_blocks',
           'block_ids_for_subblocks', 'pag_ids_for_subblocks',
           'packet_ids_for_subblocks', 'resolve_packet_grain_id']

LEVELS = ('pag', 'block', 'subblock', 'grain')

_EMPTY_VOXELS = (0, 3)


def pag_of_items(fm_state) -> Tuple[dict, dict, dict]:
    """``(grain_to_pag, block_to_pag, subblock_to_pag)``: the PAG each grain,
    block and sub-block belongs to. Blocks of grains without a PAG, and
    sub-blocks of such blocks, are left out."""
    g2pag = getattr(fm_state, "grain_to_pag_id", {}) or {}
    blk_map = {}
    if hasattr(fm_state, "grain_to_blocks_map"):
        for gid, blk_list in fm_state.grain_to_blocks_map.items():
            pag_id = g2pag.get(gid)
            if pag_id is None:
                continue
            for bid in blk_list:
                blk_map[bid] = pag_id
    sb_map = {}
    if hasattr(fm_state, "block_to_subblocks_map"):
        for bid, sb_list in fm_state.block_to_subblocks_map.items():
            pag_id = blk_map.get(bid)
            if pag_id is None:
                continue
            for sid in sb_list:
                sb_map[sid] = pag_id
    return g2pag, blk_map, sb_map


def phase_of_items(fm_state, level: str, ids: Iterable) -> Dict:
    """``{id: phase name}`` for the items of one level.

    A PAG listed in ``retained_austenite_pag_ids`` is retained austenite and
    every other PAG is martensite. A block or sub-block takes the phase of its
    PAG (martensite when it has none). A grain takes the phase of its PAG;
    without a PAG it is retained austenite if it is an isolated grain and
    martensite otherwise.
    """
    retained_pag_ids = getattr(fm_state, "retained_austenite_pag_ids", set()) or set()
    g2pag, blk_map, sb_map = pag_of_items(fm_state)
    isolated = getattr(fm_state, "isolated_grains", set()) or set()
    retained = PHASE_NAMES[PHASE_RETAINED_AUSTENITE]
    martensite = PHASE_NAMES[PHASE_MARTENSITE]

    def phase_of_pag(pag_id):
        return retained if pag_id in retained_pag_ids else martensite

    out = {}
    if level == "pag":
        for pid in ids:
            out[pid] = phase_of_pag(pid)
    elif level == "block":
        for bid in ids:
            out[bid] = phase_of_pag(blk_map[bid]) if bid in blk_map else martensite
    elif level == "subblock":
        for sid in ids:
            out[sid] = phase_of_pag(sb_map[sid]) if sid in sb_map else martensite
    else:
        for gid in ids:
            if gid in g2pag:
                out[gid] = phase_of_pag(g2pag[gid])
            else:
                out[gid] = retained if gid in isolated else martensite
    return out


def voxels_for_level(fm_state, level: str, item_id) -> np.ndarray:
    """Voxel coordinates (n, 3) of one item; empty if the item is unknown.

    A PAG is the union of its grains.
    """
    if level == "pag":
        grain_locs = getattr(fm_state, "grain_locs", {})
        grain_ids = getattr(fm_state, "clusters_dict", {}).get(item_id, [])
        parts = [grain_locs[g] for g in grain_ids if g in grain_locs and len(grain_locs[g])]
        return np.concatenate(parts, axis=0) if parts else np.empty(_EMPTY_VOXELS, dtype=int)
    if level == "block":
        return getattr(fm_state, "all_blocks", {}).get(item_id, np.empty(_EMPTY_VOXELS, dtype=int))
    if level == "subblock":
        return getattr(fm_state, "all_subblocks", {}).get(item_id, np.empty(_EMPTY_VOXELS, dtype=int))
    return getattr(fm_state, "grain_locs", {}).get(item_id, np.empty(_EMPTY_VOXELS, dtype=int))


def orientations_for_level(fm_state, level: str) -> dict:
    """``{id: (phi1, Phi, phi2)}`` stored on the state for a level
    (``'pag'``, ``'block'``, ``'subblock'`` or ``'grain'``); empty for any other."""
    return {
        "pag": getattr(fm_state, "pag_orientations", {}),
        "block": getattr(fm_state, "block_orientations", {}),
        "subblock": getattr(fm_state, "subblock_orientations", {}),
        "grain": getattr(fm_state, "grain_orientations", {}),
    }.get(level, {})


def collect_ids_and_orientations(fm_state, level: str, orientations: dict,
                                 restrict_axis: Optional[str] = None,
                                 restrict_idx: Optional[int] = None,
                                 scope_ids: Optional[Iterable] = None):
    """Ids with an orientation, and their Euler angles as an (n, 3) array.

    Parameters
    ----------
    orientations : dict
        ``{id: (phi1, Phi, phi2)}`` of the level.
    restrict_axis : {'X', 'Y', 'Z'}, optional
    restrict_idx : int, optional
        With ``restrict_axis``, keep only the items that have a voxel in the
        slice ``restrict_idx`` along that axis (ignored when negative).
    scope_ids : iterable, optional
        Keep only these ids.

    Returns
    -------
    (ids, eulers) : list and float64 array, in the order of ``orientations``.
    """
    ids = list(orientations.keys())
    if scope_ids is not None:
        scope_set = set(scope_ids)
        ids = [i for i in ids if i in scope_set]
    if restrict_axis is not None and restrict_idx is not None and restrict_idx >= 0:
        axis_i = {"X": 0, "Y": 1, "Z": 2}[restrict_axis]
        kept = []
        for item_id in ids:
            vox = voxels_for_level(fm_state, level, item_id)
            if vox.size and np.any(vox[:, axis_i] == restrict_idx):
                kept.append(item_id)
        ids = kept
    eulers = (np.array([orientations[i] for i in ids], dtype=np.float64) if ids
              else np.empty((0, 3), dtype=np.float64))
    return ids, eulers


def packet_ids_for_pags(fm_state, pag_ids: Iterable) -> Set[int]:
    """Packets (grain ids) of the given PAGs."""
    clusters = getattr(fm_state, "clusters_dict", {}) or {}
    out = set()
    for pid in pag_ids:
        out.update(int(g) for g in clusters.get(pid, []))
    return out


def pag_ids_for_blocks(fm_state, block_ids: Iterable) -> Set:
    """PAGs the given blocks belong to."""
    _, blk_map, _ = pag_of_items(fm_state)
    return {blk_map[b] for b in block_ids if b in blk_map}


def packet_ids_for_blocks(fm_state, block_ids: Iterable) -> Set:
    """Packets (grain ids) the given blocks belong to."""
    g2b = getattr(fm_state, "grain_to_blocks_map", {}) or {}
    blk_to_packet = {b: gid for gid, blist in g2b.items() for b in blist}
    return {blk_to_packet[b] for b in block_ids if b in blk_to_packet}


def block_ids_for_subblocks(fm_state, subblock_ids: Iterable) -> Set:
    """Blocks the given sub-blocks belong to."""
    b2s = getattr(fm_state, "block_to_subblocks_map", {}) or {}
    sb_to_blk = {s: bid for bid, slist in b2s.items() for s in slist}
    return {sb_to_blk[s] for s in subblock_ids if s in sb_to_blk}


def pag_ids_for_subblocks(fm_state, subblock_ids: Iterable) -> Set:
    """PAGs the given sub-blocks belong to."""
    _, _, sb_map = pag_of_items(fm_state)
    return {sb_map[s] for s in subblock_ids if s in sb_map}


def packet_ids_for_subblocks(fm_state, subblock_ids: Iterable) -> Set:
    """Packets (grain ids) the given sub-blocks belong to."""
    return packet_ids_for_blocks(fm_state, block_ids_for_subblocks(fm_state, subblock_ids))


def resolve_packet_grain_id(fm_state, pag_id, local_idx) -> Optional[int]:
    """Grain id (= packet id) whose local ordinal within ``pag_id`` is
    ``local_idx``, or None if there is none."""
    g2pkt = getattr(fm_state, "grain_to_local_pkt_idx", {})
    for gid in getattr(fm_state, "clusters_dict", {}).get(pag_id, []):
        if g2pkt.get(gid) == local_idx:
            return gid
    return None
