"""Label maps of the FM steel hierarchy for visualisation, at full and rescaled resolution.

``build_visualization_data`` returns, for a structure that has PAGs, blocks
and orientations (and optionally sub-blocks):

* the grain, PAG, block-index and block-id maps and, with sub-blocks, the
  sub-block id map;
* the same maps resampled by a scale factor (``*_rs``);
* the packet-index map of the blocks (with sub-blocks);
* for sub-blocks, the colour of each sub-block id, with one hue per parent
  block and the lightness varying between the sub-blocks of a block.

Block and sub-block ids are assigned in a random order fixed by the seed, so
that neighbouring ids differ visibly.
"""
from typing import Callable, Dict, Optional

import numpy as np

import upxo.gsdataops.grid_ops as gridOps

from .slice_region_stats import pag_label_grid

__all__ = ['label_groups_in_random_order', 'subblock_family_colours',
           'packet_index_grid', 'build_visualization_data']


def label_groups_in_random_order(template: np.ndarray, groups: dict, seed: int):
    """Grid shaped like ``template`` in which the voxels of each group carry one label.

    ``groups`` maps a group key to its voxel coordinates (n, 3). The groups
    receive the labels 1..n in an order drawn from ``default_rng(seed)``;
    voxels in no group are 0.

    Returns
    -------
    (grid, labels) : the grid and the list of labels, in the order of ``groups``.
    """
    labels = np.random.default_rng(seed=seed).permutation(len(groups)) + 1
    grid = np.zeros_like(template)
    for label, voxels in zip(labels, groups.values()):
        grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = int(label)
    return grid, [int(label) for label in labels]


def subblock_family_colours(label_to_subblock: Dict[int, str], block_to_subblocks: dict,
                            n_subblocks: int, seed: int, saturation: float = 0.82) -> np.ndarray:
    """RGB colour per sub-block label; row 0 and unknown labels stay black.

    Sub-blocks of one block share a hue (the hues of the blocks are shuffled
    with ``default_rng(seed)``); within a block the value rises from 0.30 to
    0.90 with the position of the sub-block, and is 0.60 for a block with a
    single sub-block.
    """
    import matplotlib.colors as mcolors

    subblock_to_block = {sb: bid for bid, sbs in block_to_subblocks.items() for sb in sbs}
    block_list = list(block_to_subblocks.keys())
    n_blocks = len(block_list)
    hue_order = np.random.default_rng(seed=seed).permutation(n_blocks)
    block_hue = {bid: int(hue_order[i]) / n_blocks for i, bid in enumerate(block_list)}
    colours = np.zeros((n_subblocks + 2, 3), dtype=np.float32)
    for label, sb_id in label_to_subblock.items():
        block_id = subblock_to_block.get(sb_id)
        if block_id is None:
            continue
        siblings = block_to_subblocks[block_id]
        n_siblings = len(siblings)
        position = siblings.index(sb_id)
        value = 0.30 + 0.60 * (position / max(1, n_siblings - 1)) if n_siblings > 1 else 0.60
        colours[label] = mcolors.hsv_to_rgb([block_hue[block_id], saturation, value])
    return colours


def packet_index_grid(template: np.ndarray, blocks: dict, grain_to_plane_idx: dict) -> np.ndarray:
    """Grid in which each block carries the packet index (plane index + 1) of its grain.

    The grain id is the third underscore-separated field of the block id.
    """
    grid = np.zeros_like(template)
    for block_id, voxels in blocks.items():
        if len(voxels):
            grain_id = int(block_id.split('_')[2])
            grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = grain_to_plane_idx.get(grain_id, 0) + 1
    return grid


def build_visualization_data(fm_pag, fm_blocks, fm_orientations, fm_final, *, seed: int = 42,
                             rescale_factor: float = 1.0, rescale_method: str = 'nearest',
                             domain_length: float = 100.0, subblock_mode: bool = False,
                             family_colours: bool = False,
                             log: Optional[Callable[[str], None]] = None) -> dict:
    """Label maps of the hierarchy for visualisation.

    Parameters
    ----------
    fm_pag :
        State with PAGs: ``clusters_dict``, ``n_pags`` and, as ``_parent``,
        the base structure (``lgi``, ``n_grains``).
    fm_blocks :
        State with ``all_blocks`` (block id -> voxel coordinates).
    fm_orientations :
        State with ``all_blocks`` and the base structure's
        ``grain_to_plane_idx`` as ``_parent`` (used for the packet map in
        sub-block mode).
    fm_final :
        The state the maps are made for: the sub-block state in sub-block
        mode (``all_subblocks``, ``block_to_subblocks_map``), otherwise the
        orientation state. It is returned as ``fm_state``.
    seed : int
        Block ids use ``seed`` and ``seed + 35``, sub-block ids ``seed + 70``
        and the sub-block hues ``seed + 71``.
    rescale_factor, rescale_method :
        Passed to ``grid_ops.rescale_grid_3d``.
    domain_length : float
        Length of the domain along x; ``voxel_size_rs`` is this divided by
        the rescaled number of voxels along x.
    log : callable(str), optional
        Receives progress messages.

    Returns
    -------
    dict with the keys fm_state, grain_id_map, pag_id_map, block_idx_map,
    block_fid_map, subblock_fid_map, grain_id_map_rs, pag_id_map_rs,
    block_fid_map_rs, subblock_fid_map_rs, fid_to_rgb, fid_to_sb_id,
    packet_id_map_rs, voxel_size_rs, n_blocks and
    isolated_grain_orientations. Entries that do not apply (sub-block
    entries without sub-blocks) are None.
    """
    log = log or (lambda message: None)
    if subblock_mode and fm_final is None:
        raise RuntimeError("Sub-block data not found — generate sub-blocks first.")

    log("Starting vis-data preparation...")
    log(f"Mode: {'subblock' if subblock_mode else 'block'}, "
        f"rescale_factor={rescale_factor}, method={rescale_method!r}")

    base = fm_pag._parent
    log(f"Grid shape: {base.lgi.shape}  n_grains={base.n_grains}  "
        f"n_PAGs={fm_pag.n_pags}  n_blocks={len(fm_blocks.all_blocks)}")

    log("Building grain ID map...")
    grain_id_map = base.lgi.copy()

    # One lookup table remaps the whole grid in a single pass; a full-grid
    # comparison per grain would grow with n_grains * n_voxels.
    log("Building PAG ID map...")
    pag_id_map = pag_label_grid(grain_id_map, fm_pag.clusters_dict, dtype=grain_id_map.dtype)
    log(f"PAG map done — {len(fm_pag.clusters_dict)} PAGs")

    log("Building block index and FID maps...")
    n_blocks = len(fm_blocks.all_blocks)
    block_idx_map, _ = label_groups_in_random_order(grain_id_map, fm_blocks.all_blocks, seed)
    block_fid_map, _ = label_groups_in_random_order(grain_id_map, fm_blocks.all_blocks, seed + 35)
    log(f"Block maps done — {n_blocks} blocks")

    log(f"Sub-block map: {'building...' if subblock_mode else 'skipped (block mode)'}")
    subblock_fid_map = None
    fid_to_rgb = None
    fid_to_sb_id: Dict[int, str] = {}
    if subblock_mode:
        n_subblocks = len(fm_final.all_subblocks)
        subblock_fid_map, labels = label_groups_in_random_order(
            grain_id_map, fm_final.all_subblocks, seed + 70)
        fid_to_sb_id = dict(zip(labels, fm_final.all_subblocks.keys()))
        if family_colours:
            fid_to_rgb = subblock_family_colours(
                fid_to_sb_id, fm_final.block_to_subblocks_map, n_subblocks, seed + 71)

    log(f"Rescaling maps (factor={rescale_factor}, method={rescale_method!r})...")

    def rescale(grid):
        return gridOps.rescale_grid_3d(grid, rescale_factor, method=rescale_method)

    grain_id_map_rs = rescale(grain_id_map)
    pag_id_map_rs = rescale(pag_id_map)
    block_fid_map_rs = rescale(block_fid_map)

    subblock_fid_map_rs = None
    packet_id_map_rs = None
    if subblock_fid_map is not None:
        subblock_fid_map_rs = rescale(subblock_fid_map)
        packet_id_map_rs = rescale(packet_index_grid(
            grain_id_map, fm_orientations.all_blocks, fm_orientations._parent.grain_to_plane_idx))

    voxel_size_rs = domain_length / grain_id_map_rs.shape[0]
    log(f"Rescaled grid shape: {grain_id_map_rs.shape}  voxel_size_rs={voxel_size_rs:.4f}")

    return {
        "fm_state": fm_final,
        "grain_id_map": grain_id_map,
        "pag_id_map": pag_id_map,
        "block_idx_map": block_idx_map,
        "block_fid_map": block_fid_map,
        "subblock_fid_map": subblock_fid_map,
        "grain_id_map_rs": grain_id_map_rs,
        "pag_id_map_rs": pag_id_map_rs,
        "block_fid_map_rs": block_fid_map_rs,
        "subblock_fid_map_rs": subblock_fid_map_rs,
        "fid_to_rgb": fid_to_rgb,
        "fid_to_sb_id": fid_to_sb_id,
        "packet_id_map_rs": packet_id_map_rs,
        "voxel_size_rs": voxel_size_rs,
        "n_blocks": n_blocks,
        "isolated_grain_orientations": dict(getattr(fm_pag, 'isolated_grain_orientations', {})),
    }
