"""Colour composites of the FM steel hierarchy.

A composite is a float32 RGB array shaped like the grain label grid plus a
trailing axis of 3, painted from coarse to fine so that the finest level
present wins:

1. isolated grains (grains in no PAG), coloured by grain id;
2. PAGs, one colour each;
3. blocks, or packets, in the hue of their packet's habit plane;
4. optionally sub-blocks, in the same hue.

The hue of a habit plane is taken from the packet colour map. Within a grain
the lightness rises with the position of a block among the blocks of its
grain, and within a block with the position of a sub-block among the
sub-blocks of its block.

A block id has the grain id of its packet as its third underscore-separated
field.
"""
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

__all__ = ['N_HABIT_PLANES', 'packet_hues', 'order_lightness', 'base_composite',
           'paint_blocks', 'paint_subblocks', 'subblock_hue_table', 'block_level_composite',
           'subblock_level_composite', 'flatten_rgb']

N_HABIT_PLANES = 4

_SATURATION = 0.85
_PACKET_LIGHTNESS = 0.60


def _colormap(cmap):
    return matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap


def packet_hues(packet_cmap: Union[str, object] = "tab10") -> List[float]:
    """Hue (0..1) of each habit plane: the packet colour map sampled at 0, 1/3, 2/3 and 1."""
    cmap = _colormap(packet_cmap)
    return [float(mcolors.rgb_to_hsv(np.array(cmap(i / 3.0)[:3]))[0]) for i in range(N_HABIT_PLANES)]


def order_lightness(position: int, count: int) -> float:
    """Lightness of the ``position``-th of ``count`` siblings: 0.35 for the
    first up to 0.90 for the last, and 0.60 when there is only one."""
    return 0.35 + 0.55 * (position / max(1, count - 1)) if count > 1 else _PACKET_LIGHTNESS


def base_composite(grain_id_map: np.ndarray, pag_id_map: np.ndarray, pag_cmap, isolated_cmap
                   ) -> np.ndarray:
    """Composite of isolated grains and PAGs.

    Voxels of a grain in no PAG take ``isolated_cmap`` at the grain id
    scaled to 0..1; PAG ``k`` of ``n`` takes ``pag_cmap`` at ``(k - 1) / (n - 1)``.
    Void stays black.
    """
    pag_cmap, isolated_cmap = _colormap(pag_cmap), _colormap(isolated_cmap)
    rgb = np.zeros(grain_id_map.shape + (3,), dtype=np.float32)

    n_grains = int(grain_id_map.max())
    isolated = (pag_id_map == 0) & (grain_id_map > 0)
    if isolated.any():
        scaled = np.clip((grain_id_map[isolated].astype(float) - 1) / max(1, n_grains - 1), 0, 1)
        rgb[isolated] = np.array(isolated_cmap(scaled))[:, :3].astype(np.float32)

    n_pags = int(pag_id_map.max())
    for pag in range(1, n_pags + 1):
        mask = pag_id_map == pag
        if mask.any():
            rgb[mask] = pag_cmap((pag - 1) / max(1, n_pags - 1))[:3]
    return rgb


def paint_blocks(rgb: np.ndarray, blocks: dict, grain_to_plane_idx: dict, hues: Sequence[float],
                 grain_to_blocks: Optional[dict] = None) -> None:
    """Paint each block, in place, in the hue of its packet's habit plane.

    The lightness is 0.60 for every block, or, when ``grain_to_blocks`` is
    given, graded by the position of the block among the blocks of its grain.
    """
    for block_id, voxels in blocks.items():
        if not len(voxels):
            continue
        grain_id = int(block_id.split('_')[2])
        hue = hues[grain_to_plane_idx.get(grain_id, 0) % N_HABIT_PLANES]
        lightness = _PACKET_LIGHTNESS
        if grain_to_blocks is not None:
            siblings = grain_to_blocks.get(grain_id, [block_id])
            position = siblings.index(block_id) if block_id in siblings else 0
            lightness = order_lightness(position, len(siblings))
        rgb[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = mcolors.hsv_to_rgb([hue, _SATURATION, lightness])


def paint_subblocks(rgb: np.ndarray, subblocks: dict, block_to_subblocks: dict,
                    grain_to_plane_idx: dict, hues: Sequence[float]) -> None:
    """Paint each sub-block, in place, in the hue of its packet's habit plane
    with a lightness graded by its position among the sub-blocks of its block."""
    subblock_to_block = {sb: bid for bid, sbs in block_to_subblocks.items() for sb in sbs}
    for sb_id, voxels in subblocks.items():
        if not len(voxels):
            continue
        block_id = subblock_to_block.get(sb_id)
        if block_id is None:
            continue
        hue = hues[grain_to_plane_idx.get(int(block_id.split('_')[2]), 0) % N_HABIT_PLANES]
        siblings = block_to_subblocks[block_id]
        position = siblings.index(sb_id) if sb_id in siblings else 0
        rgb[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = mcolors.hsv_to_rgb(
            [hue, _SATURATION, order_lightness(position, len(siblings))])


def subblock_hue_table(label_to_subblock: Dict[int, str], block_to_subblocks: dict,
                       grain_to_plane_idx: dict, hues: Sequence[float]) -> np.ndarray:
    """RGB per sub-block label, as ``paint_subblocks`` colours them.

    Row ``k`` is the colour of label ``k``; rows of labels that are unknown or
    whose sub-block is in no block are black. The table has
    ``max(label) + 2`` rows (2 for no labels).
    """
    subblock_to_block = {sb: bid for bid, sbs in block_to_subblocks.items() for sb in sbs}
    n_rows = max(label_to_subblock.keys()) + 2 if label_to_subblock else 2
    table = np.zeros((n_rows, 3), dtype=np.float32)
    for label, sb_id in label_to_subblock.items():
        block_id = subblock_to_block.get(sb_id)
        if block_id is None:
            continue
        hue = hues[grain_to_plane_idx.get(int(block_id.split('_')[2]), 0) % N_HABIT_PLANES]
        siblings = block_to_subblocks[block_id]
        position = siblings.index(sb_id) if sb_id in siblings else 0
        table[label] = mcolors.hsv_to_rgb([hue, _SATURATION, order_lightness(position, len(siblings))])
    return table


def block_level_composite(grain_id_map, pag_id_map, blocks, grain_to_plane_idx, grain_to_blocks,
                          pag_cmap, isolated_cmap, packet_cmap="tab10") -> np.ndarray:
    """PAG -> block composite: ``base_composite`` with the blocks painted over it,
    their lightness graded within each grain."""
    rgb = base_composite(grain_id_map, pag_id_map, pag_cmap, isolated_cmap)
    paint_blocks(rgb, blocks, grain_to_plane_idx, packet_hues(packet_cmap), grain_to_blocks)
    return rgb


def subblock_level_composite(grain_id_map, pag_id_map, blocks, subblocks, block_to_subblocks,
                             grain_to_plane_idx, pag_cmap, isolated_cmap,
                             packet_cmap="tab10") -> np.ndarray:
    """PAG -> packet -> sub-block composite: ``base_composite``, the blocks in
    the plain hue of their packet, and the sub-blocks over them."""
    hues = packet_hues(packet_cmap)
    rgb = base_composite(grain_id_map, pag_id_map, pag_cmap, isolated_cmap)
    paint_blocks(rgb, blocks, grain_to_plane_idx, hues)
    paint_subblocks(rgb, subblocks, block_to_subblocks, grain_to_plane_idx, hues)
    return rgb


def flatten_rgb(rgb: np.ndarray) -> np.ndarray:
    """(n_voxels, 3) colours, voxels in Fortran order as VTK image data expects."""
    return np.column_stack([rgb[..., c].flatten(order='F') for c in range(3)])
