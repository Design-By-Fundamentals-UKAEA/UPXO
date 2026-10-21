"""Small synthetic pipeline states for deck-export tests.

The 6 x 6 x 6 grid holds four grains in a 2 x 2 arrangement of columns, one
isolated grain and a void corner. Grains 1 and 2 form PAG 1, grains 3 and 4
form PAG 2; each grain is split into two blocks and each block into two
sub-blocks.
"""
from types import SimpleNamespace

import numpy as np


def _halves(voxels):
    middle = len(voxels) // 2
    return voxels[:middle], voxels[middle:]


def build_state(level='subblock', voxel_size=2.0):
    """A state with the hierarchy down to ``level`` ('pag', 'block', 'subblock')."""
    lgi = np.zeros((6, 6, 6), dtype=int)
    x, y = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    column_label = 1 + (x >= 3).astype(int) + 2 * (y >= 3).astype(int)
    lgi[:] = column_label[:, :, None]
    lgi[0:2, 0:2, 4:6] = 5                 # isolated grain
    lgi[5, 5, 0:2] = 0                     # void

    grain_locs = {gid: np.argwhere(lgi == gid) for gid in (1, 2, 3, 4, 5)}
    state = SimpleNamespace(
        lgi=lgi, voxel_size=voxel_size, units='microns',
        grain_locs=grain_locs,
        clusters_dict={1: [1, 2], 2: [3, 4]},
        grain_to_pag_id={1: 1, 2: 1, 3: 2, 4: 2},
        grain_to_local_pkt_idx={1: 1, 2: 2, 3: 1, 4: 1},
        pag_orientations={1: (10.0, 20.0, 30.0), 2: (40.0, 50.0, 60.0)},
        isolated_grains={5},
        isolated_grain_orientations={5: (1.0, 2.0, 3.0)},
        retained_austenite_pag_ids=set(),
    )
    if level == 'pag':
        return state

    all_blocks, grain_to_blocks, block_orientations = {}, {}, {}
    next_block = 101
    for gid in (1, 2, 3, 4):
        grain_to_blocks[gid] = []
        for part in _halves(grain_locs[gid]):
            all_blocks[next_block] = part
            grain_to_blocks[gid].append(next_block)
            block_orientations[next_block] = (next_block * 0.1, 5.0, 7.0)
            next_block += 1
    state.all_blocks = all_blocks
    state.grain_to_blocks_map = grain_to_blocks
    state.block_orientations = block_orientations
    if level == 'block':
        return state

    all_subblocks, block_to_subblocks, subblock_orientations = {}, {}, {}
    next_sub = 1001
    for block_id, voxels in all_blocks.items():
        block_to_subblocks[block_id] = []
        for part in _halves(voxels):
            all_subblocks[next_sub] = part
            block_to_subblocks[block_id].append(next_sub)
            subblock_orientations[next_sub] = (next_sub * 0.01, 9.0, 8.0)
            next_sub += 1
    state.all_subblocks = all_subblocks
    state.block_to_subblocks_map = block_to_subblocks
    state.subblock_orientations = subblock_orientations
    return state
