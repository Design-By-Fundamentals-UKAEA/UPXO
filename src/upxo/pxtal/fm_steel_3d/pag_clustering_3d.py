"""
Technique A, "lever 2": simultaneous multi-source PAG clustering.

FMSteel3DBase.generate_pag_clusters forms PAGs *sequentially* -- one seed is
picked, its cluster is grown to its target size, marked claimed, and only
then is the next seed picked from whatever remains. Late-processed seeds
face an increasingly depleted neighbour pool purely because of *when* they
happen to be processed, independent of their own degree -- this is why
widening neighbour-graph connectivity (lever 1) only modestly helped: it
raises a grain's neighbour count, but the fraction of that count already
claimed by the time the grain is reached is a global property of the
sequential process, not a local one.

This module replaces the sequential seed-by-seed growth with *simultaneous*
multi-source BFS, the same mechanism already validated for Technique B's
within-grain packet splitting (upxo.pxtalops.grain_splitting_3d.split_
grains_geodesic),
now applied across the whole grain-adjacency graph instead of within one
grain's voxels:

  1. A maximal independent set (MIS) of grains is chosen as seeds -- every
     other grain is guaranteed to be adjacent to at least one seed (that is
     what "maximal" means), so no seed has to compete for territory that
     isn't already within one hop of *some* seed. Built directly off the
     cc3d.region_graph-derived neighbour dict (see greedy_maximal_
     independent_set) -- NOT networkx.maximal_independent_set, which is only
     used elsewhere in UPXO for the (unrelated) isolated-grain selection and
     is explicitly not reused here.
  2. All seeds grow their clusters *at the same time* via a single shared
     BFS queue (mark-on-enqueue, exactly as in split_grains_geodesic), each
     capped at its own randomly-sampled target size (<= max_packets_per_pag).
     No seed gets a head start over another.
  3. Whatever remains unclaimed after one wave (most seeds have far fewer
     packets requested than they have immediate neighbours, so a wave
     typically leaves plenty unclaimed) becomes the input to another MIS +
     simultaneous-growth wave, repeated until every grain is clustered.

max_packets_per_pag exists as an explicit hard cap distinct from whatever
target sizes generate_pag_clusters's caller supplies, because there is a
real crystallographic upper bound worth enforcing regardless of the
supplied size distribution: FCC austenite has exactly 4 distinct {111}
close-packed planes, and a "packet" is conventionally defined as the set of
laths/blocks sharing one such habit plane (see with_blocks_3d.py's
grain_to_plane_idx, which already encodes one plane index per packet) -- so
4 is the crystallographically-motivated default, not an arbitrary number.
Exposed as a parameter (not hard-coded) since looser packet definitions do
appear in the literature.

--- "lever 3": repair_singleton_pags ------------------------------------

Testing showed lever 2 does NOT meaningfully reduce the singleton-PAG rate
versus the original sequential algorithm (measured ~19-23% for MC, ~9-10%
for Voronoi either way, at 100^3 across three coarsening stages). The
reason: PAG target sizes (3-4) are small relative to typical grain degree
(~11-14), so most seeds -- MIS-based or not -- are hungry and contesting
overlapping neighbour pools; simultaneity reshuffles who wins each contested
grain but doesn't remove the contention. The scarcity is combinatorial
(many small greedy claims against a shared, finite resource), not a
scheduling artifact, so no single-pass greedy scheme escapes it.

repair_singleton_pags is a genuinely different mechanism: instead of trying
to get every claim right in one greedy pass, it looks at the *finished*
clustering and gives every leftover (singleton) grain a real second chance:
first by pairing/grouping stranded singletons that happen to neighbour each
other into fresh multi-grain PAGs (recovering PAGs from what was pure waste
before), then by absorbing whatever's still alone into a spare-capacity
neighbouring PAG. It is orthogonal to *how* clusters_dict was produced --
works equally on generate_pag_clusters's or cluster_grains_simultaneous's
output.
"""
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ['greedy_maximal_independent_set', 'cluster_grains_simultaneous',
           'repair_singleton_pags', 'select_isolated_units',
           'select_isolated_units_with_tolerance', 'boundary_grain_ids']


def boundary_grain_ids(lgi: np.ndarray) -> set:
    """Grain ids that own at least one voxel on the domain's outer faces.

    Standalone equivalent of the 'boundary' branch inside
    FMSteel3DBase._select_isolated_grains, extracted so it can be reused for
    PAG-level (not just grain-level) retained-austenite selection -- see
    select_isolated_units.
    """
    bm = np.zeros(lgi.shape, dtype=bool)
    bm[0, :, :] = bm[-1, :, :] = True
    bm[:, 0, :] = bm[:, -1, :] = True
    bm[:, :, 0] = bm[:, :, -1] = True
    return {int(g) for g in np.unique(lgi[bm]) if g > 0}


def _build_ordering(
    unit_ids: List,
    unit_sizes_map: Dict,
    strategy: str,
    isolated_size_tol: float,
    boundary_unit_ids: Optional[set],
    rng: np.random.Generator,
) -> List:
    """Candidate visiting order for the greedy independent-set build --
    factored out of select_isolated_units so select_isolated_units_with_
    tolerance can request a *fresh* ordering per retry attempt using the
    exact same strategy semantics."""
    if strategy == 'smallest':
        return sorted(unit_ids, key=lambda u: unit_sizes_map[u])
    elif strategy == 'largest':
        return sorted(unit_ids, key=lambda u: unit_sizes_map[u], reverse=True)
    elif strategy == 'boundary':
        boundary_unit_ids = boundary_unit_ids or set()
        b_list = [u for u in unit_ids if u in boundary_unit_ids]
        i_list = [u for u in unit_ids if u not in boundary_unit_ids]
        rng.shuffle(b_list)
        rng.shuffle(i_list)
        return b_list + i_list
    elif strategy == 'near_mean':
        mean_sz = float(np.mean(list(unit_sizes_map.values())))
        ordered_all = sorted(unit_ids, key=lambda u: abs(unit_sizes_map[u] - mean_sz))
        tol_abs = isolated_size_tol * mean_sz
        primary = [u for u in ordered_all if abs(unit_sizes_map[u] - mean_sz) <= tol_abs]
        secondary = [u for u in ordered_all if u not in set(primary)]
        return primary + secondary
    else:  # 'random' and unknown fallback
        ordered = list(unit_ids)
        rng.shuffle(ordered)
        return ordered


def _greedy_prefix_sequence(
    ordered: List,
    unit_sizes_map: Dict,
    unit_adjacency: Dict[int, List[int]],
) -> List[Tuple]:
    """Run the greedy independent-set build over `ordered` to *completion*
    (no early stop at any target), returning the full sequence of
    (unit_id, cumulative_size_after_adding) in add order. The cumulative
    size column is monotonically increasing, so any target lookup reduces
    to scanning this sequence."""
    seq: List[Tuple] = []
    excluded: set = set()
    cum = 0.0
    for u in ordered:
        if u in excluded:
            continue
        cum += unit_sizes_map[u]
        seq.append((u, cum))
        for n in unit_adjacency.get(u, []):
            excluded.add(n)
    return seq


def select_isolated_units(
    unit_ids: Iterable,
    unit_sizes_map: Dict,
    unit_adjacency: Dict[int, List[int]],
    target_iso_size: float,
    strategy: str = 'near_mean',
    isolated_size_tol: float = 0.25,
    boundary_unit_ids: Optional[set] = None,
    rng: Optional[np.random.Generator] = None,
) -> set:
    """Greedy independent-set selection of "isolated" units reaching a
    target cumulative size -- used for retained-austenite selection.

    Generalises FMSteel3DBase._select_isolated_grains (which only ever
    operates on individual grains, pre-clustering) to work at either level:
    pass grain ids + the grain-adjacency graph for the original grain-level
    behaviour, or pass PAG ids + a PAG-level adjacency graph (see
    pag_technique_selector_3d._pag_level_neigh_and_sizes) to reclassify
    *whole, already-formed PAGs* as retained austenite instead. The latter
    is what the new lever-based clustering paths use, since retained-
    austenite selection now runs after PAG formation rather than as a
    pre-filter -- it works uniformly regardless of which lever produced the
    PAGs, whereas pre-filtering individual grains before clustering doesn't
    generalise cleanly to the MIS-based/simultaneous clustering methods.

    A single greedy pass like this one can only ever reach as far as its
    own random ordering allows before every remaining candidate is excluded
    by an already-selected neighbour -- see select_isolated_units_with_
    tolerance for a version that iterates (fresh orderings, and precise
    trimming) until the achieved size lands within an explicit tolerance of
    the target, rather than accepting whatever this one first-pass attempt
    produces.

    Parameters
    ----------
    unit_ids : iterable
        Grain ids or PAG ids.
    unit_sizes_map : dict
        unit_id -> voxel count.
    unit_adjacency : dict[int, list[int]]
        unit_id -> list of neighbouring unit_ids (no self-loops).
    target_iso_size : float
        Cumulative voxel count to reach before stopping. <= 0 returns an
        empty set immediately (no retained austenite requested).
    strategy : str, optional
        'smallest', 'largest', 'boundary', 'near_mean', or 'random'
        (default 'near_mean') -- same semantics as
        FMSteel3DBase._select_isolated_grains.
    isolated_size_tol : float, optional
        Used only by 'near_mean'. Default 0.25.
    boundary_unit_ids : set, optional
        Required (non-None) for strategy='boundary'; the set of unit ids
        touching the domain's outer boundary. See boundary_grain_ids for
        the grain-level case.
    rng : np.random.Generator, optional
        Defaults to a fresh, unseeded generator if omitted.

    Returns
    -------
    set
        Selected unit ids: an independent set (no two touch) whose combined
        size is >= target_iso_size (or all available units, if that target
        can't be reached).
    """
    if rng is None:
        rng = np.random.default_rng()
    if target_iso_size <= 0:
        return set()

    unit_ids = list(unit_ids)
    ordered = _build_ordering(unit_ids, unit_sizes_map, strategy, isolated_size_tol,
                              boundary_unit_ids, rng)

    isolated: set = set()
    iso_size = 0.0
    excluded: set = set()
    for u in ordered:
        if u in excluded:
            continue
        isolated.add(u)
        iso_size += unit_sizes_map[u]
        for n in unit_adjacency.get(u, []):
            excluded.add(n)
        if iso_size >= target_iso_size:
            break
    return isolated


def select_isolated_units_with_tolerance(
    unit_ids: Iterable,
    unit_sizes_map: Dict,
    unit_adjacency: Dict[int, List[int]],
    target_iso_size: float,
    total_size: float,
    tolerance: float = 0.05,
    max_attempts: int = 10,
    trim_step: int = 1,
    strategy: str = 'near_mean',
    isolated_size_tol: float = 0.25,
    boundary_unit_ids: Optional[set] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Like select_isolated_units, but iterates to land the achieved size
    within `tolerance` of target_iso_size, instead of accepting whatever a
    single greedy pass happens to produce.

    The two directions need different fixes, because a single greedy
    independent-set pass can only ever reach as far as its own random
    ordering allows -- once every remaining candidate is excluded by an
    already-selected neighbour, the pass is over, and asking for a *larger*
    target does not change that outcome for that same ordering:

    * Overshoot (the ordering's first at-or-above-target prefix exceeds the
      tolerance band): fixed exactly, no retry needed. The full greedy
      sequence for this ordering is monotonically increasing, so dropping
      the most-recently-added units (in `trim_step`-sized steps, since
      removing members from an independent set can never break
      independence) finds a prefix inside the band.
    * Undershoot (even this ordering's fullest achievable independent set
      falls short of the tolerance band): only fixable by trying a
      different random ordering/strategy shuffle -- retried up to
      `max_attempts` times. If no attempt ever converges, the closest
      candidate found across all attempts is returned with
      `converged=False`, rather than silently returning an out-of-tolerance
      result unlabeled.

    Parameters
    ----------
    (same as select_isolated_units, plus:)
    total_size : float
        Total voxel count of the full unit population -- converts the
        fractional `tolerance` into an absolute voxel-count band around
        target_iso_size.
    tolerance : float, optional
        Acceptable deviation of achieved/total_size from target_iso_size/
        total_size, as a fraction (e.g. 0.05 = +/-5 percentage points).
        Default 0.05.
    max_attempts : int, optional
        Maximum fresh random orderings tried when even the fullest
        achievable independent set for an ordering undershoots the
        tolerance band. Default 10.
    trim_step : int, optional
        Number of units removed per step when trimming an overshoot back
        into the tolerance band. Default 1 (most precise; larger values
        check fewer candidate prefixes, at the risk of stepping over a
        narrow band in one jump).

    Returns
    -------
    dict
        'selected' : set -- the chosen independent set.
        'target_fraction' : float -- target_iso_size / total_size.
        'achieved_fraction' : float -- actual selected volume / total_size.
        'tolerance' : float -- the tolerance value used.
        'converged' : bool -- whether achieved landed within tolerance.
        'n_attempts' : int -- how many fresh orderings were actually tried.
    """
    target_fraction = (target_iso_size / total_size) if total_size else 0.0
    if rng is None:
        rng = np.random.default_rng()
    if target_iso_size <= 0 or total_size <= 0:
        return {
            'selected': set(), 'target_fraction': target_fraction,
            'achieved_fraction': 0.0, 'tolerance': float(tolerance),
            'converged': True, 'n_attempts': 0,
        }

    unit_ids = list(unit_ids)
    tol_abs = tolerance * total_size
    lo = target_iso_size - tol_abs
    hi = target_iso_size + tol_abs

    best_selected: set = set()
    best_size = 0.0
    best_gap = abs(0.0 - target_iso_size)  # empty selection: always a valid fallback
    n_attempts_used = 0

    for attempt in range(1, int(max_attempts) + 1):
        n_attempts_used = attempt
        ordered = _build_ordering(unit_ids, unit_sizes_map, strategy, isolated_size_tol,
                                  boundary_unit_ids, rng)
        seq = _greedy_prefix_sequence(ordered, unit_sizes_map, unit_adjacency)
        if not seq:
            continue

        first_over = next((i for i, (_, c) in enumerate(seq) if c >= target_iso_size), None)

        if first_over is None:
            # Even the fullest independent set this ordering can build
            # falls short -- remember it if it's the closest seen so far,
            # then try a different ordering.
            cum = seq[-1][1]
            gap = abs(cum - target_iso_size)
            if gap < best_gap:
                best_gap, best_size = gap, cum
                best_selected = {u for u, _ in seq}
            continue

        idx = first_over
        while idx >= 0:
            cum = seq[idx][1]
            if lo <= cum <= hi:
                return {
                    'selected': {u for u, _ in seq[:idx + 1]},
                    'target_fraction': target_fraction,
                    'achieved_fraction': cum / total_size,
                    'tolerance': float(tolerance), 'converged': True,
                    'n_attempts': attempt,
                }
            if cum < lo:
                break
            idx -= trim_step

        # This ordering's neighbourhood of the target jumped straight over
        # the band in one step (one grain too large to land inside it) --
        # remember the two closest bracketing prefixes (including the
        # empty selection when first_over is the very first unit) and try
        # a fresh ordering.
        checks = [(first_over, seq[first_over][1])]
        if first_over > 0:
            checks.append((first_over - 1, seq[first_over - 1][1]))
        else:
            checks.append((-1, 0.0))
        for check_idx, cum in checks:
            gap = abs(cum - target_iso_size)
            if gap < best_gap:
                best_gap, best_size = gap, cum
                best_selected = {u for u, _ in seq[:check_idx + 1]} if check_idx >= 0 else set()

    return {
        'selected': best_selected, 'target_fraction': target_fraction,
        'achieved_fraction': (best_size / total_size) if total_size else 0.0,
        'tolerance': float(tolerance), 'converged': False,
        'n_attempts': n_attempts_used,
    }


def greedy_maximal_independent_set(
    neigh_gid: Dict[int, List[int]],
    grain_ids: Iterable[int],
    rng: np.random.Generator,
) -> List[int]:
    """Greedy maximal independent set over the grain-adjacency graph.

    Standard textbook construction (deliberately not networkx's
    maximal_independent_set, per the requirement that this stay a direct,
    exact greedy build on the cc3d.region_graph-derived neighbour dict):
    visit grains in random order; whichever haven't been excluded yet get
    added to the set and immediately exclude themselves + their neighbours
    from ever being chosen. After a full pass, every grain is either in the
    set or adjacent to a member -- that's what makes it *maximal* (not
    necessarily the largest possible independent set, just one that can't
    have any further member added without breaking independence).

    Parameters
    ----------
    neigh_gid : dict[int, list[int]]
        grain_id -> list of neighbouring grain_ids.
    grain_ids : iterable[int]
        The candidate pool (e.g. still-unclustered grains on a later wave --
        does not have to be every grain in neigh_gid).
    rng : np.random.Generator

    Returns
    -------
    list[int]
        The independent set: no two returned grains are neighbours of each
        other, and every grain in `grain_ids` is either in this list or
        adjacent to something in it.
    """
    pool = list(grain_ids)
    rng.shuffle(pool)
    pool_set = set(pool)

    mis: List[int] = []
    excluded: set = set()
    for g in pool:
        if g in excluded:
            continue
        mis.append(g)
        excluded.add(g)
        for n in neigh_gid.get(g, []):
            if n in pool_set:
                excluded.add(n)
    return mis


def cluster_grains_simultaneous(
    neigh_gid: Dict[int, List[int]],
    grain_ids: Iterable[int],
    size_distribution: Dict[str, Sequence[float]],
    max_packets_per_pag: int = 4,
    random_seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Partition grain_ids into PAGs via simultaneous multi-source BFS,
    waved with a fresh maximal-independent-set seeding each time until every
    grain is clustered.

    Parameters
    ----------
    neigh_gid : dict[int, list[int]]
        grain_id -> list of neighbouring grain_ids (from
        FMSteel3DBase.compute_neighbor_network -- any connectivity).
    grain_ids : iterable[int]
        Every grain to be clustered (already excludes anything pre-selected
        as retained austenite, if that filtering happens upstream).
    size_distribution : dict
        {'sizes': [...], 'probs': [...]} -- same schema as
        FMSteel3DBase.generate_pag_clusters's pag_size_distribution. Any
        value above max_packets_per_pag is clipped down to it.
    max_packets_per_pag : int, optional
        Hard cap on packets per PAG, independent of size_distribution.
        Default 4 (see module docstring: the crystallographic upper bound
        from FCC austenite's 4 {111} habit planes).
    random_seed : int, optional

    Returns
    -------
    dict[int, list[int]]
        {pag_id: [grain_ids]} -- every input grain_id appears in exactly one
        PAG's list, and no PAG's list exceeds max_packets_per_pag entries.
    """
    rng = np.random.default_rng(random_seed)

    sizes = np.minimum(np.asarray(size_distribution['sizes']), max_packets_per_pag)
    probs = np.asarray(size_distribution['probs'], dtype=np.float64)
    probs = probs / probs.sum()

    available = set(grain_ids)
    clusters_dict: Dict[int, List[int]] = {}
    pag_id = 1

    while available:
        seeds = greedy_maximal_independent_set(neigh_gid, available, rng)

        owner: Dict[int, int] = {}
        capacity: Dict[int, int] = {}
        q = deque()
        for seed in seeds:
            pid = pag_id
            pag_id += 1
            target_sz = max(1, int(rng.choice(sizes, p=probs)))
            capacity[pid] = target_sz
            owner[seed] = pid
            clusters_dict[pid] = [seed]
            q.append(seed)

        while q:
            g = q.popleft()
            pid = owner[g]
            if len(clusters_dict[pid]) >= capacity[pid]:
                continue
            for n in neigh_gid.get(g, []):
                if len(clusters_dict[pid]) >= capacity[pid]:
                    break
                if n in available and n not in owner:
                    owner[n] = pid
                    clusters_dict[pid].append(n)
                    q.append(n)

        available -= owner.keys()

    return clusters_dict


def repair_singleton_pags(
    clusters_dict: Dict[int, List[int]],
    neigh_gid: Dict[int, List[int]],
    max_packets_per_pag: int = 4,
    random_seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Post-hoc rescue pass for singleton (1-grain) PAGs.

    Two-step repair, in priority order:

    1. Cluster the singleton grains *among themselves* (their induced
       subgraph under neigh_gid) the same way any PAG is clustered --
       this recovers genuine, properly-sized multi-grain PAGs out of pure
       waste wherever stranded singletons happen to neighbour each other,
       at no cost to any already-formed PAG.
    2. Whatever singleton remains un-paired (no singleton neighbour at all)
       gets absorbed into whichever *existing*, non-singleton neighbouring
       PAG has spare capacity (< max_packets_per_pag), preferring the PAG
       it shares the most adjacency with.

    A grain only remains a singleton after this if neither option was
    available (e.g. a genuinely isolated grain with zero neighbours, or
    every neighbouring PAG -- singleton or not -- is already full) -- this
    should be rare, and is a legitimate outcome, not a failure.

    Parameters
    ----------
    clusters_dict : dict[int, list[int]]
        {pag_id: [grain_ids]} as produced by generate_pag_clusters,
        cluster_grains_simultaneous, or any other PAG-clustering method.
    neigh_gid : dict[int, list[int]]
        grain_id -> list of neighbouring grain_ids.
    max_packets_per_pag : int, optional
        Hard cap respected both for freshly-paired singleton groups and for
        absorption into an existing PAG. Default 4.
    random_seed : int, optional

    Returns
    -------
    dict[int, list[int]]
        A new clusters_dict covering exactly the same grains as the input
        (none lost, none duplicated). PAG ids are renumbered 1..K -- they
        are not guaranteed to match the input's ids, since some PAGs are
        merged away and new ones are created from paired leftovers.
    """
    rng = np.random.default_rng(random_seed)

    singleton_grains = {gids[0] for gids in clusters_dict.values() if len(gids) == 1}

    # --- Step 1: cluster the leftover (singleton) subgraph on its own,
    # requesting the largest allowed size -- greedy BFS naturally caps out
    # at whatever's actually connected, so this just means "grab as many
    # connected leftover neighbours as you can, up to the cap". ---
    leftover_neigh = {g: [n for n in neigh_gid.get(g, []) if n in singleton_grains]
                      for g in singleton_grains}
    rescue_dist = {"sizes": [max_packets_per_pag], "probs": [1.0]}
    rescued = cluster_grains_simultaneous(
        leftover_neigh, singleton_grains, rescue_dist,
        max_packets_per_pag=max_packets_per_pag, random_seed=random_seed,
    ) if singleton_grains else {}

    new_clusters: Dict[int, List[int]] = {}
    next_pag_id = 1
    still_stranded: List[int] = []
    for gids in rescued.values():
        if len(gids) >= 2:
            new_clusters[next_pag_id] = list(gids)
            next_pag_id += 1
        else:
            still_stranded.extend(gids)

    # --- Step 2: absorb any still-stranded grain into a spare-capacity
    # *original* (non-singleton) neighbouring PAG -- singleton peers already
    # had their chance in step 1. ---
    kept_original: Dict[int, List[int]] = {
        pid: list(gids) for pid, gids in clusters_dict.items() if len(gids) >= 2
    }
    grain_to_kept: Dict[int, int] = {g: pid for pid, gids in kept_original.items() for g in gids}

    still_stranded = list(still_stranded)
    rng.shuffle(still_stranded)
    truly_stranded: List[int] = []
    for g in still_stranded:
        candidates = [grain_to_kept[n] for n in neigh_gid.get(g, [])
                      if n in grain_to_kept and len(kept_original[grain_to_kept[n]]) < max_packets_per_pag]
        if candidates:
            host = max(set(candidates), key=candidates.count)  # most shared adjacency
            kept_original[host].append(g)
            grain_to_kept[g] = host
        else:
            truly_stranded.append(g)

    for gids in kept_original.values():
        new_clusters[next_pag_id] = gids
        next_pag_id += 1
    for g in truly_stranded:
        new_clusters[next_pag_id] = [g]
        next_pag_id += 1

    return new_clusters
