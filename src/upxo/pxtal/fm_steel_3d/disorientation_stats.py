"""Disorientation statistics across the FM steel hierarchy.

Two kinds of statistics are provided.

3-D pair sampling (``hierarchical_disorientations``)
    Disorientation angles of PAG pairs (adjacent or random), block pairs
    (within one PAG or across PAGs) and sub-block pairs (within one block or
    across blocks).

2-D slice histograms (``slice_disorientation_histograms``)
    Density histograms of the disorientation across the boundaries seen on
    evenly spaced slices normal to x, y and z.

Angles are in degrees. Disorientations of orientation triples come from
``OrientationAssigner3D.disorientation_deg``.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from upxo.xtalphy.crystal_orientation import euler_bunge_to_matrix_batch

from .orientation_assigner_3d import OrientationAssigner3D

__all__ = [
    'SAMPLED_TYPES', 'SLICE_TYPES', 'DEFAULT_ANGLE_BINS', 'KS_CHARACTERISTIC_ANGLES',
    'parent_block_id', 'adjacent_pag_disorientations', 'random_pag_disorientations',
    'subblock_disorientations', 'hierarchical_disorientations',
    'slice_disorientation_histograms',
]

SAMPLED_TYPES = ('blk_within', 'blk_across', 'pag_adj', 'pag_rand', 'sb_within', 'sb_across')
SLICE_TYPES = ('blk_within', 'blk_across', 'pag_adj', 'sb_within', 'sb_across')

# Histogram bins of the slice statistics: 1.3 degree wide, 0 to 65 degrees.
DEFAULT_ANGLE_BINS = np.linspace(0.0, 65.0, 51)

# Characteristic disorientation angles (degrees) of the Kurdjumov-Sachs
# orientation relationship between parent austenite and martensite.
KS_CHARACTERISTIC_ANGLES = (10.53, 14.88, 20.61, 21.06, 47.11, 49.47, 57.21, 60.00)


def parent_block_id(subblock_id: str) -> str:
    """Id of the block a sub-block belongs to: ``'SB_<block id>_<n>'`` -> block id."""
    return "_".join(subblock_id[3:].split("_")[:-1])


# ── 3-D pair sampling ────────────────────────────────────────────────────────

def adjacent_pag_disorientations(pag_orientations: dict, neighbours: dict,
                                 assigner: Optional[OrientationAssigner3D] = None) -> List[float]:
    """Disorientation of every pair of neighbouring PAGs, each pair once.

    ``neighbours`` maps a PAG id to the ids of its neighbours. Pairs where a
    PAG has no orientation are skipped.
    """
    assigner = assigner or OrientationAssigner3D()
    angles, seen = [], set()
    for pid, neighbour_ids in neighbours.items():
        for npid in neighbour_ids:
            key = (min(pid, npid), max(pid, npid))
            if key in seen:
                continue
            seen.add(key)
            if pid in pag_orientations and npid in pag_orientations:
                angles.append(assigner.disorientation_deg(
                    pag_orientations[pid], pag_orientations[npid]))
    return angles


def random_pag_disorientations(pag_orientations: dict, n_pairs: int, rng,
                               assigner: Optional[OrientationAssigner3D] = None) -> List[float]:
    """Disorientation of ``n_pairs`` random pairs of distinct PAGs.

    Empty if there are fewer than two PAGs.
    """
    assigner = assigner or OrientationAssigner3D()
    pag_ids = list(pag_orientations.keys())
    n_pags = len(pag_ids)
    angles: List[float] = []
    if n_pags >= 2:
        for _ in range(n_pairs):
            i, j = rng.choice(n_pags, 2, replace=False)
            angles.append(assigner.disorientation_deg(
                pag_orientations[pag_ids[i]], pag_orientations[pag_ids[j]]))
    return angles


def subblock_disorientations(subblock_orientations: dict, block_to_subblocks: dict,
                             n_pairs: int, rng, within: bool = True, across: bool = True,
                             assigner: Optional[OrientationAssigner3D] = None) -> dict:
    """Sub-block pair disorientations, ``{'within': [...], 'across': [...]}``.

    *within*: two sub-blocks of the same block (blocks with fewer than two
    sub-blocks are not used). *across*: two sub-blocks of different blocks;
    at most ``n_pairs`` pairs, from up to ``4 * n_pairs`` draws. The within
    pairs are drawn first, so the two lists share one random stream.
    """
    assigner = assigner or OrientationAssigner3D()
    out: Dict[str, List[float]] = {}

    if within:
        multi = {bid: sbs for bid, sbs in block_to_subblocks.items() if len(sbs) >= 2}
        block_ids = list(multi.keys())
        angles = []
        for _ in range(n_pairs):
            if not block_ids:
                break
            bid = block_ids[int(rng.integers(len(block_ids)))]
            sbs = multi[bid]
            si, sj = rng.choice(len(sbs), 2, replace=False)
            angles.append(assigner.disorientation_deg(
                subblock_orientations[sbs[int(si)]], subblock_orientations[sbs[int(sj)]]))
        out['within'] = angles

    if across:
        sb_ids = list(subblock_orientations.keys())
        angles = []
        for _ in range(n_pairs * 4):
            if len(angles) >= n_pairs:
                break
            i, j = rng.choice(len(sb_ids), 2, replace=False)
            if parent_block_id(sb_ids[i]) != parent_block_id(sb_ids[j]):
                angles.append(assigner.disorientation_deg(
                    subblock_orientations[sb_ids[i]], subblock_orientations[sb_ids[j]]))
        out['across'] = angles

    return out


def hierarchical_disorientations(fm_orientations, selected: Iterable[str], n_pairs: int,
                                 seed: int, fm_subblocks=None,
                                 assigner: Optional[OrientationAssigner3D] = None) -> dict:
    """Sampled disorientation angles of the selected pair types.

    Parameters
    ----------
    fm_orientations :
        State with block and PAG orientations (``pag_orientations`` and
        ``compute_misorientation_distribution``); its parents give the PAG
        neighbours.
    selected : iterable of str
        Types from ``SAMPLED_TYPES``.
    n_pairs : int
        Pairs sampled per type.
    seed : int
        ``compute_misorientation_distribution`` uses ``seed``; the PAG and
        sub-block pairs are drawn from ``default_rng(seed + 100)``.
    fm_subblocks :
        State with ``subblock_orientations`` and ``block_to_subblocks_map``;
        the sub-block types are skipped when it is None.

    Returns
    -------
    dict
        One list of angles per selected type that could be computed.
    """
    selected = set(selected)
    unknown = selected - set(SAMPLED_TYPES)
    if unknown:
        raise ValueError(f"unknown disorientation types {sorted(unknown)}; choose from {SAMPLED_TYPES}")
    assigner = assigner or OrientationAssigner3D()
    rng = np.random.default_rng(seed + 100)
    out: dict = {}

    if selected & {'blk_within', 'blk_across'}:
        distribution = fm_orientations.compute_misorientation_distribution(
            n_sample_pairs=n_pairs, random_seed=seed)
        if 'blk_within' in selected:
            out['blk_within'] = distribution.get('within_pag', [])
        if 'blk_across' in selected:
            out['blk_across'] = distribution.get('across_pag', [])

    if 'pag_adj' in selected:
        try:
            neighbours = fm_orientations._parent._parent.neigh_clid
        except AttributeError:
            out['pag_adj'] = []
        else:
            out['pag_adj'] = adjacent_pag_disorientations(
                fm_orientations.pag_orientations, neighbours, assigner)

    if 'pag_rand' in selected:
        out['pag_rand'] = random_pag_disorientations(
            fm_orientations.pag_orientations, n_pairs, rng, assigner)

    if fm_subblocks is not None and selected & {'sb_within', 'sb_across'}:
        sampled = subblock_disorientations(
            fm_subblocks.subblock_orientations, fm_subblocks.block_to_subblocks_map,
            n_pairs, rng, within='sb_within' in selected, across='sb_across' in selected,
            assigner=assigner)
        if 'sb_within' in selected:
            out['sb_within'] = sampled['within']
        if 'sb_across' in selected:
            out['sb_across'] = sampled['across']

    return out


# ── 2-D slice histograms ─────────────────────────────────────────────────────

def _rotation_matrices(n_items: int, orientations: Sequence) -> np.ndarray:
    """Matrices indexed from 1 (row 0 stays zero) for ``(index, euler angles)`` pairs."""
    matrices = np.zeros((n_items + 1, 3, 3), dtype=np.float64)
    if len(orientations):
        index = np.array([i for i, _ in orientations], dtype=np.intp)
        angles = np.array([a for _, a in orientations], dtype=np.float64)
        matrices[index] = euler_bunge_to_matrix_batch(angles[:, 0], angles[:, 1], angles[:, 2])
    return matrices


def _pair_angles(ia, ib, rotation_matrices) -> np.ndarray:
    """Rotation angle between the orientations of index arrays ``ia`` and ``ib``."""
    trace = np.einsum('nij,nij->n', rotation_matrices[ia], rotation_matrices[ib])
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))


def _boundary_pairs(slice_2d) -> list:
    """Label pairs across 4-connected boundaries of two positive labels."""
    pairs = []
    for dr, dc in [(1, 0), (0, 1)]:
        a = slice_2d[:-1, :] if dr else slice_2d[:, :-1]
        b = slice_2d[1:, :] if dr else slice_2d[:, 1:]
        mask = (a != b) & (a > 0) & (b > 0)
        if mask.any():
            pairs.append((a[mask], b[mask]))
    return pairs


def _density_histogram(angles, bins):
    """Density histogram of the angles, or None when there are none."""
    if not len(angles):
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        h, _ = np.histogram(angles, bins=bins, density=True)
    return np.nan_to_num(h)


def _histograms_on_slices(label_grid, n_slices, histogram_of_slice) -> dict:
    """Histograms on evenly spaced slices normal to x, y and z.

    Slices with no boundary pairs (None) are dropped so that they do not
    bias a mean over slices.
    """
    nx, ny, nz = label_grid.shape
    n_x, n_y, n_z = n_slices
    results = {'x': [], 'y': [], 'z': []}
    for i in np.linspace(1, nx - 2, max(1, n_x), endpoint=True).astype(int):
        results['x'].append(histogram_of_slice(label_grid[int(i), :, :]))
    for j in np.linspace(1, ny - 2, max(1, n_y), endpoint=True).astype(int):
        results['y'].append(histogram_of_slice(label_grid[:, int(j), :]))
    for k in np.linspace(1, nz - 2, max(1, n_z), endpoint=True).astype(int):
        results['z'].append(histogram_of_slice(label_grid[:, :, int(k)]))
    return {axis: [h for h in hists if h is not None] for axis, hists in results.items()}


def slice_disorientation_histograms(pag_id_map: np.ndarray, fm_blocks, selected: Iterable[str],
                                    n_slices: Sequence[int], fm_subblocks=None,
                                    bins: Optional[np.ndarray] = None,
                                    assigner: Optional[OrientationAssigner3D] = None) -> dict:
    """Disorientation histograms on slices, one list of histograms per axis.

    Parameters
    ----------
    pag_id_map : ndarray
        PAG id of every voxel (0 or less for void).
    fm_blocks :
        State with ``all_blocks`` (block id -> voxel coordinates),
        ``block_orientations`` and ``pag_orientations``.
    selected : iterable of str
        Types from ``SLICE_TYPES``: ``blk_within`` / ``blk_across`` (block
        pairs in the same / different PAGs), ``pag_adj`` (PAG pairs) and
        ``sb_within`` / ``sb_across`` (sub-block pairs in the same / different
        blocks).
    n_slices : (int, int, int)
        Number of slices normal to x, y and z.
    fm_subblocks :
        State with ``all_subblocks`` and ``subblock_orientations``; the
        sub-block types are skipped when it is None.
    bins : ndarray, optional
        Histogram bin edges (degrees); ``DEFAULT_ANGLE_BINS`` by default.

    Returns
    -------
    dict
        ``{type: {'x': [hist, ...], 'y': [...], 'z': [...]}}``; each histogram
        is the density over ``bins`` for one slice.
    """
    selected = set(selected)
    unknown = selected - set(SLICE_TYPES)
    if unknown:
        raise ValueError(f"unknown disorientation types {sorted(unknown)}; choose from {SLICE_TYPES}")
    bins = DEFAULT_ANGLE_BINS if bins is None else bins
    shape = pag_id_map.shape
    n_slices = tuple(n_slices)

    def run(label_grid, histogram_of_slice):
        return _histograms_on_slices(label_grid, n_slices, histogram_of_slice)

    computed: dict = {}

    if selected & {'blk_within', 'blk_across'}:
        blocks = list(fm_blocks.all_blocks.items())
        block_lgi = np.zeros(shape, dtype=np.int32)
        block_to_pag = np.zeros(len(blocks) + 1, dtype=np.int32)
        orientations = []
        for index, (bid, vox) in enumerate(blocks, start=1):
            if vox is not None and len(vox):
                block_lgi[vox[:, 0], vox[:, 1], vox[:, 2]] = index
                orientations.append((index, fm_blocks.block_orientations.get(bid, (0., 0., 0.))))
                block_to_pag[index] = int(pag_id_map[vox[0, 0], vox[0, 1], vox[0, 2]])
        block_matrices = _rotation_matrices(len(blocks), orientations)

        def block_histogram(slice_2d, within):
            angles = []
            for ia, ib in _boundary_pairs(slice_2d):
                same = (block_to_pag[ia] == block_to_pag[ib]) & (block_to_pag[ia] > 0)
                mask = same if within else ~same
                if mask.any():
                    angles.extend(_pair_angles(ia[mask], ib[mask], block_matrices).tolist())
            return _density_histogram(angles, bins)

        if 'blk_within' in selected:
            computed['blk_within'] = run(block_lgi, lambda s: block_histogram(s, True))
        if 'blk_across' in selected:
            computed['blk_across'] = run(block_lgi, lambda s: block_histogram(s, False))

    if 'pag_adj' in selected:
        assigner = assigner or OrientationAssigner3D()
        # Dense PAG indices 1..n in order of id, so ids need not be contiguous.
        pag_ids = np.unique(pag_id_map[pag_id_map > 0])
        dense_of = {int(pid): index for index, pid in enumerate(pag_ids, start=1)}
        pag_lgi = np.zeros(shape, dtype=np.int32)
        positive = pag_id_map > 0
        pag_lgi[positive] = np.searchsorted(pag_ids, pag_id_map[positive]) + 1
        euler = {}
        for pid, ori in fm_blocks.pag_orientations.items():
            index = dense_of.get(int(pid))
            if index is not None:
                euler[index] = ori

        # A full table of pair disorientations would cost one expensive symmetry
        # search per pair of PAGs, almost all of them never looked up: only PAGs
        # that meet on a sampled slice matter. Each such pair is computed once.
        cache: dict = {}

        def pair_disorientations(los, his):
            out = np.empty(los.shape, dtype=np.float32)
            unique_pairs, inverse = np.unique(np.stack([los, his], axis=1), axis=0, return_inverse=True)
            values = np.empty(len(unique_pairs), dtype=np.float32)
            for u, (lo, hi) in enumerate(unique_pairs.tolist()):
                key = (lo, hi)
                if key not in cache:
                    ea_i, ea_j = euler.get(lo), euler.get(hi)
                    cache[key] = (assigner.disorientation_deg(ea_i, ea_j)
                                  if (ea_i is not None and ea_j is not None) else np.nan)
                values[u] = cache[key]
            out[:] = values[inverse.ravel()]
            return out

        def pag_histogram(slice_2d):
            angles = []
            for ia, ib in _boundary_pairs(slice_2d):
                values = pair_disorientations(np.minimum(ia, ib), np.maximum(ia, ib))
                angles.extend(values[~np.isnan(values)].tolist())     # NaN: no orientation; 0 deg is kept
            return _density_histogram(angles, bins)

        computed['pag_adj'] = run(pag_lgi, pag_histogram)

    if fm_subblocks is not None and selected & {'sb_within', 'sb_across'}:
        subblocks = list(fm_subblocks.all_subblocks.items())
        block_index = {bid: i for i, bid in enumerate(fm_blocks.all_blocks.keys(), start=1)}
        sub_lgi = np.zeros(shape, dtype=np.int32)
        sub_to_block = np.zeros(len(subblocks) + 1, dtype=np.int32)
        orientations = []
        for index, (sb_id, vox) in enumerate(subblocks, start=1):
            if vox is not None and len(vox):
                sub_lgi[vox[:, 0], vox[:, 1], vox[:, 2]] = index
                orientations.append((index, fm_subblocks.subblock_orientations.get(sb_id, (0., 0., 0.))))
                sub_to_block[index] = block_index.get(parent_block_id(sb_id), 0)
        sub_matrices = _rotation_matrices(len(subblocks), orientations)

        def subblock_histogram(slice_2d, within):
            angles = []
            for ia, ib in _boundary_pairs(slice_2d):
                same = (sub_to_block[ia] == sub_to_block[ib]) & (sub_to_block[ia] > 0)
                mask = same if within else ~same
                if mask.any():
                    angles.extend(_pair_angles(ia[mask], ib[mask], sub_matrices).tolist())
            return _density_histogram(angles, bins)

        if 'sb_within' in selected:
            computed['sb_within'] = run(sub_lgi, lambda s: subblock_histogram(s, True))
        if 'sb_across' in selected:
            computed['sb_across'] = run(sub_lgi, lambda s: subblock_histogram(s, False))

    return computed
