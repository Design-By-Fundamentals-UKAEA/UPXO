"""
Geodesic within-grain splitting for 3D labeled grain structures.

Subdivides each grain in a 3D labeled voxel image (LGI) into one or more
connected sub-regions via multi-source BFS (geodesic nearest-seed
assignment), seeded from farthest-point-sampled voxels within the grain
itself. Standalone and framework-agnostic (pure numpy + stdlib) -- takes and
returns plain label arrays and dicts, with no dependency on any specific
grain-structure class, so it is reusable across UPXO's various 3D
pxtal/grain-structure modules (originally built for, and still used by, the
fm_steel_3d PAG/packet pipeline's "Technique B", but the algorithm itself is
domain-agnostic).

Guarantees, by construction:
  - Every output sub-region is a single connected component (a voxel can
    only be reached by travelling through voxels that belong to the same
    original grain).
  - Sub-regions exactly partition their parent grain's voxels (none lost,
    none duplicated).
  - No dependency on inter-grain adjacency/connectivity -- splitting one
    grain never competes with any other grain for resources, unlike
    clustering-based approaches to forming multi-voxel-region groups.

This is one splitting strategy among several possible ones -- geodesic
nearest-seed assignment from spread-out (farthest-point-sampled) seeds
gives *reasonably* balanced sub-region sizes (an irregular grain shape or
unlucky seed placement can still skew sizes), not an exact equal-volume
partition. A different algorithm (e.g. iteratively-rebalanced/weighted seed
placement, or a power-diagram-style approach) would be needed for a
guaranteed equal-size split; that is a natural direction for future
development in this module without changing its calling contract.
size_balance_metrics (below) exists precisely to quantify that gap -- how
far a given split landed from an equal one -- so it can be tracked as
splitting strategies evolve.
"""
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ['split_grains_geodesic', 'size_balance_metrics']


def _connectivity_offsets(connectivity: int):
    all_offsets = [(dx, dy, dz)
                   for dx in (-1, 0, 1)
                   for dy in (-1, 0, 1)
                   for dz in (-1, 0, 1)
                   if (dx, dy, dz) != (0, 0, 0)]
    if connectivity == 6:
        return [o for o in all_offsets if sum(abs(v) for v in o) == 1]
    elif connectivity == 18:
        return [o for o in all_offsets if sum(abs(v) for v in o) <= 2]
    elif connectivity == 26:
        return all_offsets
    raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")


def _farthest_point_seeds(coords: np.ndarray, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point sampling of ``n_seeds`` row-indices into
    ``coords`` (a single grain's local voxel coordinates), for reasonably
    spread-out sub-region seeds.

    Euclidean distance is used only to pick seed *locations* -- the actual
    region growth is geodesic (graph-distance, see split_grains_geodesic),
    so this heuristic only needs to spread seeds out reasonably, not to be
    an exact geodesic farthest-point solution.
    """
    n = coords.shape[0]
    if n_seeds >= n:
        return rng.choice(n, size=n, replace=False)
    chosen = [int(rng.integers(0, n))]
    dist2 = np.sum((coords - coords[chosen[0]]) ** 2, axis=1).astype(np.float64)
    for _ in range(n_seeds - 1):
        nxt = int(np.argmax(dist2))
        chosen.append(nxt)
        d2 = np.sum((coords - coords[nxt]) ** 2, axis=1)
        np.minimum(dist2, d2, out=dist2)
    return np.array(chosen, dtype=np.int64)


def split_grains_geodesic(
    lgi: np.ndarray,
    size_distribution: Dict[str, Sequence[float]],
    min_voxels_to_split: int,
    connectivity: int = 6,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, List[int]]]:
    """
    Subdivide every grain in ``lgi`` into 1 or more sub-regions.

    Grains with fewer than ``min_voxels_to_split`` voxels are left whole
    (single sub-region == the whole grain; this is the correct, intended
    outcome for a genuinely small grain, not a degenerate/error case).
    Grains at or above the threshold get a sub-region count sampled from
    ``size_distribution``, then their voxels are partitioned into that many
    sub-regions via multi-source BFS (geodesic nearest-seed assignment)
    seeded from farthest-point-sampled voxels within the grain -- guaranteed
    connected sub-regions by construction, since a voxel can only be reached
    by travelling through voxels that belong to the same original grain.

    Parameters
    ----------
    lgi : np.ndarray
        3D labeled grain image. 0 = background, 1..N = grain ids. Every
        nonzero label is assumed to already be a single connected component
        at the given ``connectivity`` (true for any cc3d.connected_components
        or Voronoi-argmin output) -- this is what guarantees every voxel in
        a grain ends up assigned to some sub-region.
    size_distribution : dict
        {'sizes': [...], 'probs': [...]}. Sampled once per eligible grain to
        pick that grain's sub-region count.
    min_voxels_to_split : int
        Grains with fewer voxels than this are not split.
    connectivity : int, optional
        6, 18, or 26. Default 6.
    random_seed : int, optional
        Seeds a local np.random.Generator: same input always reproduces the
        same split.

    Returns
    -------
    sub_lgi : np.ndarray
        Same shape as lgi, same nonzero mask. Fresh unique sub-region ids
        1..M (ids are NOT related to the original grain ids).
    sub_to_grain : dict[int, int]
        sub_region_id -> original grain_id it was split from.
    clusters_dict : dict[int, list[int]]
        original grain_id -> sorted list of sub_region_ids split from it.
    """
    if lgi.ndim != 3:
        raise ValueError("lgi must be a 3D array")
    lgi = np.asarray(lgi, dtype=np.int32)
    rng = np.random.default_rng(random_seed)

    sizes = np.asarray(size_distribution['sizes'])
    probs = np.asarray(size_distribution['probs'], dtype=np.float64)
    probs = probs / probs.sum()

    offsets = _connectivity_offsets(connectivity)

    # Group voxel flat-indices by grain id in one O(N log N) pass.
    shape = lgi.shape
    flat = lgi.ravel()
    order = np.argsort(flat, kind='stable')
    sorted_labels = flat[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_labels)]))
    unique_labels = sorted_labels[starts]

    sub_lgi = np.zeros_like(lgi)
    sub_to_grain: Dict[int, int] = {}
    clusters_dict: Dict[int, List[int]] = {}
    next_sub_id = 1

    for lbl, s, e in zip(unique_labels.tolist(), starts.tolist(), ends.tolist()):
        if lbl == 0:
            continue
        flat_idx = order[s:e]
        n_vox = flat_idx.size
        coords_global = np.column_stack(np.unravel_index(flat_idx, shape))

        if n_vox < min_voxels_to_split:
            new_id = next_sub_id
            next_sub_id += 1
            sub_lgi.ravel()[flat_idx] = new_id
            sub_to_grain[new_id] = int(lbl)
            clusters_dict[int(lbl)] = [new_id]
            continue

        n_subs = int(rng.choice(sizes, p=probs))
        n_subs = max(1, min(n_subs, n_vox))

        # Crop to this grain's local bounding box -- BFS only ever touches
        # this grain's own (typically modest) voxel count, not the full
        # domain.
        mins = coords_global.min(axis=0)
        local_coords = coords_global - mins
        local_shape = tuple((coords_global.max(axis=0) - mins + 1).tolist())

        mask = np.zeros(local_shape, dtype=bool)
        mask[tuple(local_coords.T)] = True

        seed_rows = _farthest_point_seeds(local_coords, n_subs, rng)

        label_local = np.zeros(local_shape, dtype=np.int64)  # 0 = unassigned
        q = deque()
        new_ids_for_grain = []
        for row in seed_rows.tolist():
            new_id = next_sub_id
            next_sub_id += 1
            new_ids_for_grain.append(new_id)
            sx, sy, sz = local_coords[row].tolist()
            label_local[sx, sy, sz] = new_id
            q.append((sx, sy, sz))
            sub_to_grain[new_id] = int(lbl)

        # Multi-source BFS: all seeds start in the queue simultaneously
        # (distance 0) and every voxel is marked the instant it's enqueued,
        # so this is an exact geodesic (graph-distance) nearest-seed
        # assignment -- no seed-processing-order bias, and every voxel
        # reachable within the grain (i.e. all of them, since the grain is
        # one connected component) ends up labeled.
        Lx, Ly, Lz = local_shape
        while q:
            x, y, z = q.popleft()
            cur_id = label_local[x, y, z]
            for dx, dy, dz in offsets:
                nx_, ny_, nz_ = x + dx, y + dy, z + dz
                if 0 <= nx_ < Lx and 0 <= ny_ < Ly and 0 <= nz_ < Lz:
                    if mask[nx_, ny_, nz_] and label_local[nx_, ny_, nz_] == 0:
                        label_local[nx_, ny_, nz_] = cur_id
                        q.append((nx_, ny_, nz_))

        sub_lgi.ravel()[flat_idx] = label_local[tuple(local_coords.T)]
        clusters_dict[int(lbl)] = sorted(new_ids_for_grain)

    return sub_lgi, sub_to_grain, clusters_dict


def size_balance_metrics(sizes: Sequence[float]) -> Dict[str, float]:
    """How far a set of sub-region (or, more generally, any grouped-member)
    sizes is from a perfectly equal split of their shared total.

    Three standard, complementary indicators, each 0 (or 1 for
    min_max_ratio) at a perfectly even split and moving away from that as
    the split becomes more unequal:

      cv            -- coefficient of variation (population std / mean) of
                        the sizes. Since the sizes are assumed to partition
                        a fixed total, their mean already equals the
                        "ideal equal share" (total / n) -- so this is
                        exactly the normalised deviation from that ideal
                        share, not merely from whatever the sizes happen to
                        average to. 0 = perfectly equal.
      min_max_ratio -- smallest / largest size. 1.0 = perfectly equal (every
                        size identical); lower means the worst-case pair is
                        further apart. Only looks at the two extremes, so
                        it can miss unevenness among the middle sizes.
      gini          -- Gini coefficient of the size distribution (0 =
                        perfectly equal, larger = more concentrated in a
                        few large sizes). Accounts for the whole
                        distribution rather than just the extremes --
                        complements min_max_ratio.

    Parameters
    ----------
    sizes : sequence of float
        Sizes (e.g. voxel counts) of the sub-regions/members within one
        parent (grain, PAG, or any other group). A single size (no split
        occurred, or only one member) returns every indicator at its
        "perfectly equal" value, since there is nothing to compare.

    Returns
    -------
    dict with keys 'cv', 'min_max_ratio', 'gini'.
    """
    x = np.asarray(sizes, dtype=np.float64)
    if x.size <= 1:
        return {'cv': 0.0, 'min_max_ratio': 1.0, 'gini': 0.0}

    mean = x.mean()
    cv = float(x.std() / mean) if mean > 0 else 0.0
    min_max_ratio = float(x.min() / x.max()) if x.max() > 0 else 1.0

    xs = np.sort(x)
    n = xs.size
    total = xs.sum()
    gini = float((2.0 * np.sum(np.arange(1, n + 1) * xs) / (n * total)) - (n + 1) / n) if total > 0 else 0.0

    return {'cv': cv, 'min_max_ratio': min_max_ratio, 'gini': gini}
