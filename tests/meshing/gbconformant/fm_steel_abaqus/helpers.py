"""Synthetic raw export folders and structured tetrahedral meshes for the tests."""
from contextlib import contextmanager
from pathlib import Path
import pickle
import shutil
import uuid
import numpy as np

# Six tetrahedra per voxel sharing the body diagonal, positive Jacobian.
KUHN = (((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)))
SHAPE = (6, 4, 4)


@contextmanager
def work_dir():
    root = Path.cwd() / ('fm_abaqus_test_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        assert root.parent == Path.cwd() and root.name.startswith('fm_abaqus_test_')
        shutil.rmtree(root)


def _euler(index):
    rng = np.random.default_rng(1000 + index)
    return tuple(float(v) for v in rng.uniform(0, 360, 3))


def write_raw_folder(folder, with_sblk=False, retained=False, zero_label=False):
    """Write a small hierarchical export: 2 PAGs, 3 packets, 5 blocks (+ sub-blocks)."""
    folder = Path(folder)
    nx, ny, nz = SHAPE
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    base = np.where(x < 2, 1, np.where(x < 3, 2, 3)).astype(np.int32)
    pag = np.where(x < 3, 1, 2).astype(np.int32)
    block_ids = ['B_1_1_1', 'B_1_1_2', 'B_1_2_1', 'B_2_3_1', 'B_2_3_2']
    block_of = np.empty(SHAPE, dtype=object)
    block_of[(base == 1) & (y < 2)] = 'B_1_1_1'
    block_of[(base == 1) & (y >= 2)] = 'B_1_1_2'
    block_of[base == 2] = 'B_1_2_1'
    block_of[(base == 3) & (z < 2)] = 'B_2_3_1'
    block_of[(base == 3) & (z >= 2)] = 'B_2_3_2'
    blk_int = {bid: i for i, bid in enumerate(block_ids, 1)}
    blk = np.vectorize(blk_int.get)(block_of).astype(np.int32)
    if zero_label:
        blk[0, 0, 0] = 0
    arrays = {'base': base, 'pag': pag, 'pck': base, 'blk': blk}
    orientations = {
        'pag': {1: _euler(1), 2: _euler(2)},
        'pck': {1: _euler(11), 2: _euler(12), 3: _euler(13)},
        'blk': {bid: _euler(20 + i) for i, bid in enumerate(block_ids)},
    }
    bundle = {
        'base_units': 'mm', 'base_voxel_size': 0.001,
        'base_grain_locs': {1: None, 2: None, 3: None},
        'pag_clusters_dict': {1: [1, 2], 2: [3]},
        'pag_grain_to_pag_id': {1: 1, 2: 1, 3: 2},
        'pag_grain_to_local_pkt_idx': {1: 1, 2: 2, 3: 1},
        'pck_grain_to_blocks_map': {1: ['B_1_1_1', 'B_1_1_2'], 2: ['B_1_2_1'],
                                    3: ['B_2_3_1', 'B_2_3_2']},
        'blk_lgi_int_to_id': {v: k for k, v in blk_int.items()},
        'blk_lgi_id_to_int': dict(blk_int),
        'pag_retained_austenite_pag_ids': {2} if retained else set(),
        'pag_isolated_grains': set(),
    }
    if with_sblk:
        sb_ids, sb_map = [], {}
        sblk = np.zeros(SHAPE, dtype=np.int32)
        for bid in block_ids:
            for half in (0, 1):
                sid = f'SB_{bid}_{half + 1}'
                sb_ids.append(sid)
                sblk[(block_of == bid) & (z % 2 == half)] = len(sb_ids)
            sb_map[bid] = [f'SB_{bid}_1', f'SB_{bid}_2']
        arrays['sblk'] = sblk
        orientations['sblk'] = {sid: _euler(40 + i) for i, sid in enumerate(sb_ids)}
        bundle['sblk_lgi_int_to_id'] = {i: sid for i, sid in enumerate(sb_ids, 1)}
        bundle['sblk_block_to_subblocks_map'] = sb_map
    for name, array in arrays.items():
        np.save(folder / f'{name}_lgi.npy', array)
    for name, data in orientations.items():
        with (folder / f'{name}_orientations.pkl').open('wb') as handle:
            pickle.dump(data, handle)
    with (folder / 'raw_export.pkl').open('wb') as handle:
        pickle.dump(bundle, handle)
    return arrays, orientations, bundle


def voxel_tets(labels, spacing=1.0):
    """Structured tetrahedral mesh of a label array; returns points, tets, grains.

    Tetrahedron IDs run voxel by voxel (i, j, k order), six per voxel.
    """
    nx, ny, nz = labels.shape
    stride_i, stride_j = (ny + 1) * (nz + 1), nz + 1
    gi, gj, gk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1),
                             indexing='ij')
    pts = np.column_stack((gi.ravel(), gj.ravel(), gk.ravel())).astype(float) * spacing
    vi, vj, vk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    base = (vi * stride_i + vj * stride_j + vk).ravel()
    per_voxel = np.stack([
        np.stack([base + a * stride_i + b * stride_j + c for a, b, c in corners], axis=1)
        for corners in KUHN], axis=1)
    return pts, per_voxel.reshape(-1, 4), np.repeat(labels.ravel(), 6)


def read_elsets(path):
    """{name: [element ids]} of an element-set include file."""
    sets, current = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('*Elset'):
            current = line.split('elset=')[1].strip()
            sets[current] = []
        elif current and line and not line.startswith('*') and not line.startswith('**'):
            sets[current] += [int(v) for v in line.split(',') if v.strip()]
    return sets


TWINNED_SHAPE = (6, 4, 4)    # x, y, z of the structure; the folder stores it as z, y, x


def write_twinned_folder(folder):
    """A small twinned structure: two hosts, a primary twin and a secondary twin of it, and a non-host grain.

    Returns ``(labels in x, y, z order, roles, parent map, quaternions)``.
    """
    folder = Path(folder)
    nx, ny, nz = TWINNED_SHAPE
    x, _y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    labels = np.where(x < 2, 2, np.where(x < 3, 3, np.where(x < 5, np.where(z < 2, 4, 6), 5))).astype(np.int32)
    roles = {2: 'host', 3: 'primary_twin', 4: 'host', 5: 'secondary_twin', 6: 'non_host'}
    parents = {3: 2, 5: 3}
    rng = np.random.default_rng(7)
    quats = {}
    for label in roles:
        q = rng.normal(size=4)
        quats[label] = q / np.linalg.norm(q)
    np.save(folder / 'lgi_twinned_clean.npy', np.transpose(labels, (2, 1, 0)))
    for name, data in (('orientations_all', quats), ('twin_role', roles), ('twin_parent_of', parents)):
        with (folder / f'{name}.pkl').open('wb') as handle:
            pickle.dump(data, handle)
    return labels, roles, parents, quats
