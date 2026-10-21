"""Synthetic PAG / block / sub-block hierarchy with orientations, for statistics tests.

Blocks are Voronoi cells of random seeds on a small grid, grouped into PAGs
with non-contiguous ids. Every second block is split into two sub-blocks.
A column of voxels is void.
"""
from types import SimpleNamespace

import numpy as np

PAG_IDS = (3, 7, 11, 20)


def _random_euler(rng, n):
    return np.column_stack([rng.uniform(0, 360, n), rng.uniform(0, 90, n), rng.uniform(0, 90, n)])


def build_hierarchy(seed=0, shape=(14, 14, 14), n_blocks=12):
    """Return ``(pag_id_map, fm_blocks, fm_subblocks)``.

    ``fm_blocks`` carries blocks, block and PAG orientations, PAG neighbours
    two parents up and a ``compute_misorientation_distribution`` that draws
    its pairs from ``default_rng(random_seed)``. ``fm_subblocks`` carries the
    sub-blocks and has ``fm_blocks`` as its parent.
    """
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1).astype(float)
    seeds = rng.uniform(0, 1, (n_blocks, 3)) * np.array(shape)
    nearest = np.argmin(((grid[..., None, :] - seeds) ** 2).sum(-1), axis=-1)       # block index 0..n-1
    void = np.zeros(shape, dtype=bool)
    void[-1, -1, :] = True

    block_pag = {b: PAG_IDS[b % len(PAG_IDS)] for b in range(n_blocks)}
    pag_id_map = np.zeros(shape, dtype=int)
    all_blocks, block_orientations, counters = {}, {}, {}
    block_names = []
    for b in range(n_blocks):
        voxels = np.argwhere((nearest == b) & ~void)
        pag = block_pag[b]
        counters[pag] = counters.get(pag, 0) + 1
        name = f'{pag}_1_{counters[pag]}'
        block_names.append(name)
        if len(voxels):
            all_blocks[name] = voxels
            pag_id_map[tuple(voxels.T)] = pag
    euler = _random_euler(rng, len(all_blocks))
    block_orientations = {name: tuple(e) for name, e in zip(all_blocks, euler)}

    pag_orientations = {pag: tuple(e) for pag, e in zip(PAG_IDS, _random_euler(rng, len(PAG_IDS)))}
    neigh_clid = {3: [7, 11], 7: [3, 20], 11: [3, 20], 20: [7, 11]}

    all_subblocks, block_to_subblocks = {}, {}
    for number, (name, voxels) in enumerate(all_blocks.items()):
        order = voxels[np.argsort(voxels[:, 0], kind='stable')]
        parts = [order] if number % 2 else [order[:len(order) // 2], order[len(order) // 2:]]
        block_to_subblocks[name] = []
        for local, part in enumerate(parts, start=1):
            if len(part):
                sb_id = f'SB_{name}_{local}'
                all_subblocks[sb_id] = part
                block_to_subblocks[name].append(sb_id)
    subblock_orientations = {sb: tuple(e) for sb, e in zip(all_subblocks, _random_euler(rng, len(all_subblocks)))}

    def compute_misorientation_distribution(n_sample_pairs, random_seed):
        r = np.random.default_rng(random_seed)
        return {'within_pag': r.uniform(0, 62, n_sample_pairs).tolist(),
                'across_pag': r.uniform(0, 62, n_sample_pairs).tolist()}

    pag_state = SimpleNamespace(neigh_clid=neigh_clid)
    blocks_state = SimpleNamespace(_parent=pag_state)
    fm_blocks = SimpleNamespace(
        all_blocks=all_blocks, block_orientations=block_orientations,
        pag_orientations=pag_orientations, _parent=blocks_state,
        compute_misorientation_distribution=compute_misorientation_distribution)
    fm_subblocks = SimpleNamespace(
        all_subblocks=all_subblocks, subblock_orientations=subblock_orientations,
        block_to_subblocks_map=block_to_subblocks, _parent=fm_blocks)
    return pag_id_map, fm_blocks, fm_subblocks
