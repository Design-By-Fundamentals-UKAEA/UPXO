"""
Unified Technique A/B + lever orchestration for PAG/packet formation.

Single entry point (generate_pags) wiring together everything validated in
this investigation:

  - Technique A's original sequential clustering
    (FMSteel3DBase.generate_pag_clusters), reused as-is for levers 1 and 3.
  - Lever 1: the same sequential clustering, fed a widened-connectivity
    neighbour graph (see base_3d.compute_neighbor_network).
  - Lever 2: pag_clustering_3d.cluster_grains_simultaneous (MIS-seeded
    simultaneous multi-source clustering).
  - Lever 3: pag_clustering_3d.repair_singleton_pags, a post-hoc rescue pass
    stackable on top of either lever 1's or lever 2's output (or the plain
    sequential baseline).
  - Technique B: upxo.pxtalops.grain_splitting_3d.split_grains_geodesic
    (within-grain packet splitting; each PAG is exactly one original base
    grain). That module is general-purpose (any 3D labeled grain structure,
    not FM-steel-specific), so it lives outside fm_steel_3d, in pxtalops
    alongside UPXO's other reusable grain-structure operations.

Retained austenite (pag_grain_fraction < 1.0) is resequenced to run AFTER
PAG formation, reclassifying whole formed PAGs (or, for technique B, whole
original grains -- equivalent, since PAG == grain there) as retained
austenite, rather than pre-filtering individual grains before clustering.
This is necessary because pre-filtering doesn't generalise cleanly across
all five lever combinations (in particular the MIS-based clustering used by
levers 2/2_3 has no natural place to "set aside" grains beforehand the way
the original sequential algorithm's isolated_grain_strategy did) -- see
pag_clustering_3d.select_isolated_units's docstring for the mechanism.

Retained-austenite PAGs are NOT removed from clusters_dict -- they stay in
it as ordinary (if untransformed) PAGs, tagged via retained_austenite_pag_ids.
This matters for orientation: assign_pag_orientations() assigns FCC
orientation to every PAG in clusters_dict uniformly, so a retained-austenite
PAG's grains keep exactly the orientation their (never-realised) parent PAG
was given, the same way a transformed PAG's grains do -- rather than each
grain in it drawing its own separate, uncorrelated orientation later via
assign_isolated_grain_orientations(). generate_blocks() is what actually
withholds block generation for retained_austenite_pag_ids; it never affects
clusters_dict or pag_orientations. isolated_grains is kept (as the flattened
union of retained-austenite PAGs' grains) purely for backward compatibility
with code that already reads it.

FMSteel3DBase.generate_pag_clusters itself is left completely unmodified --
existing sessions/scripts that call it directly keep their original
pre-filter retained-austenite behaviour and are unaffected by this module.
This is the new, additive entry point used by the GUI's Technique/Lever
selection page.
"""
import copy
from typing import Dict, Optional, Sequence

import numpy as np

from .base_3d import FMSteel3DBase
from upxo.pxtalops.grain_splitting_3d import split_grains_geodesic
from .pag_clustering_3d import (
    cluster_grains_simultaneous,
    repair_singleton_pags,
    select_isolated_units_with_tolerance,
    boundary_grain_ids,
)

__all__ = ['generate_pags', 'VALID_TECHNIQUES', 'VALID_LEVERS']

VALID_TECHNIQUES = ('A', 'B')
VALID_LEVERS = ('lever1', 'lever2', 'lever3', 'lever1_3', 'lever2_3')

# Default clustering-connectivity per lever, matching the empirically-tested
# defaults (see pag_technique_selector tests / conversation history): lever1
# and lever1_3 default to the widened graph (18); lever2/lever2_3 are fixed
# at the base connectivity (MIS clustering was only ever validated at 6);
# lever3 alone defaults to the plain baseline connectivity for clustering,
# leaving repair to do the work.
_LEVER_DEFAULTS = {
    'lever1':   {'clustering_connectivity': 18, 'mis': False, 'repair': False},
    'lever2':   {'clustering_connectivity': None, 'mis': True,  'repair': False},
    'lever3':   {'clustering_connectivity': None, 'mis': False, 'repair': True},
    'lever1_3': {'clustering_connectivity': 18, 'mis': False, 'repair': True},
    'lever2_3': {'clustering_connectivity': None, 'mis': True,  'repair': True},
}


def _pag_level_neigh_and_sizes(clusters_dict, grain_sizes_map, neigh_gid):
    """Derive a PAG-level adjacency dict + size map from a grain-level
    clusters_dict -- used both for the final neigh_clid returned to
    FMSteel3DWithPAGs and for post-hoc (post-formation) retained-austenite
    selection at the PAG level.
    """
    grain_to_pag = {g: pid for pid, gids in clusters_dict.items() for g in gids}
    pag_sizes = {pid: sum(grain_sizes_map[g] for g in gids) for pid, gids in clusters_dict.items()}
    pag_neigh: Dict[int, set] = {pid: set() for pid in clusters_dict}
    for pid, gids in clusters_dict.items():
        for g in gids:
            for n in neigh_gid.get(g, []):
                opid = grain_to_pag.get(n)
                if opid is not None and opid != pid:
                    pag_neigh[pid].add(opid)
    return {pid: sorted(s) for pid, s in pag_neigh.items()}, pag_sizes


def generate_pags(
    fm_base: FMSteel3DBase,
    technique: str,
    pag_size_distribution: Dict[str, Sequence[float]],
    max_packets_per_pag: int = 4,
    lever: str = 'lever3',
    clustering_connectivity: Optional[int] = None,
    repair_connectivity: int = 18,
    split_threshold: int = 32,
    pag_grain_fraction: float = 1.0,
    isolated_grain_strategy: str = 'near_mean',
    isolated_size_tol: float = 0.25,
    retained_austenite_tolerance: float = 0.05,
    retained_austenite_max_attempts: int = 10,
    retained_austenite_trim_step: int = 1,
    use_non_neigh_pag: bool = True,
    random_seed: Optional[int] = None,
):
    """Generate PAGs (technique A) or PAGs+packets (technique B) using the
    selected technique and, for technique A, lever.

    Parameters
    ----------
    fm_base : FMSteel3DBase
        Already-cleaned base grain structure -- agnostic to whether it came
        from Voronoi tessellation, Monte-Carlo grain growth, or any future
        base-grain method.
    technique : str
        'A' (grain clustering) or 'B' (within-grain splitting).
    pag_size_distribution : dict
        {'sizes': [...], 'probs': [...]}. Any size above max_packets_per_pag
        is clipped down to it.
    max_packets_per_pag : int, optional
        Hard cap on packets per PAG, independent of pag_size_distribution.
        Default 4 (the crystallographic upper bound from FCC austenite's 4
        {111} habit planes -- see grain_splitting_3d/pag_clustering_3d
        module docstrings).
    lever : str, optional
        Technique A only: one of 'lever1', 'lever2', 'lever3' (default),
        'lever1_3', 'lever2_3'. Ignored (and should be disabled in any UI)
        when technique='B'.
    clustering_connectivity : int, optional
        Technique A only. If None, defaults per lever: 18 for lever1/
        lever1_3, fm_base.connectivity (the base labelling connectivity,
        typically 6) for lever2/lever2_3/lever3.
    repair_connectivity : int, optional
        Technique A only, for levers with repair enabled (lever3, lever1_3,
        lever2_3). Default 18.
    split_threshold : int, optional
        Technique B only: minimum voxels for a grain to be split into
        multiple packets; grains below this stay a single packet. Default 32.
    pag_grain_fraction : float, optional
        Volume fraction that ends up in the PAG/packet hierarchy; the
        remainder becomes retained austenite (an independent set of whole
        PAGs -- or, for technique B, whole original grains -- selected
        AFTER PAG formation). Default 1.0 (no retained austenite).
    isolated_grain_strategy : str, optional
        'smallest', 'largest', 'boundary', 'near_mean' (default), or
        'random' -- passed to the retained-austenite selection.
    isolated_size_tol : float, optional
        Used only by 'near_mean'. Default 0.25.
    retained_austenite_tolerance : float, optional
        Acceptable deviation of the achieved retained-austenite volume
        fraction from `1 - pag_grain_fraction`, as a fraction (e.g. 0.05 =
        +/-5 percentage points). See
        pag_clustering_3d.select_isolated_units_with_tolerance for why
        overshoot and undershoot need different fixes. Default 0.05.
    retained_austenite_max_attempts : int, optional
        Maximum fresh random re-selection attempts when even the fullest
        achievable independent set undershoots the tolerance band (a hard
        ceiling set by the PAG-adjacency graph's structure, not something
        more attempts can always overcome -- see the same docstring).
        Default 10.
    retained_austenite_trim_step : int, optional
        Number of units removed per step when trimming an overshoot back
        into the tolerance band. Default 1 (most precise).
    use_non_neigh_pag : bool, optional
        Technique A, non-MIS levers only (lever1, lever3, lever1_3): passed
        through to the underlying sequential FMSteel3DBase.generate_pag_
        clusters call. Measured to have negligible effect on the singleton
        rate either way; exposed for completeness rather than hard-coded.
        Default True.
    random_seed : int, optional

    Returns
    -------
    FMSteel3DWithPAGs
    """
    from .with_pags_3d import FMSteel3DWithPAGs

    if technique not in VALID_TECHNIQUES:
        raise ValueError(f"technique must be one of {VALID_TECHNIQUES}, got {technique!r}")
    if technique == 'A' and lever not in VALID_LEVERS:
        raise ValueError(f"lever must be one of {VALID_LEVERS}, got {lever!r}")

    # Derive independent sub-seeds for every stage that builds its own
    # np.random.default_rng(...) (this function's own `rng`, plus the ones
    # constructed internally by cluster_grains_simultaneous,
    # repair_singleton_pags, and split_grains_geodesic) instead of handing
    # all of them the SAME `random_seed` integer. Seeding several independent
    # PCG64 Generator instances from one identical seed makes their draw
    # sequences start from the identical state, which can correlate
    # supposedly-independent randomness across stages (e.g. which grains an
    # MIS wave happens to favour vs. which grains the repair pass happens to
    # shuffle first). Each derived sub-seed is itself a deterministic
    # function of `random_seed`, so overall reproducibility (same
    # random_seed -> same output) is unaffected.
    if random_seed is not None:
        _sub_seeds = np.random.SeedSequence(random_seed).generate_state(4).tolist()
    else:
        _sub_seeds = [None, None, None, None]
    rng = np.random.default_rng(_sub_seeds[0])
    if random_seed is not None:
        np.random.seed(random_seed)

    sizes_clipped = list(np.minimum(np.asarray(pag_size_distribution['sizes']), max_packets_per_pag))
    pag_size_distribution = {'sizes': sizes_clipped, 'probs': list(pag_size_distribution['probs'])}

    grain_ids = [g for g in fm_base.grain_locs.keys() if g != 0]
    grain_sizes_map = {g: len(fm_base.grain_locs[g]) for g in grain_ids}
    total_voxels = float(sum(grain_sizes_map.values()))
    target_iso_voxels = (1.0 - pag_grain_fraction) * total_voxels

    if technique == 'B':
        isolated = set()
        ra_selection_info = {
            'target_fraction': 1.0 - pag_grain_fraction, 'achieved_fraction': 0.0,
            'tolerance': retained_austenite_tolerance, 'converged': True, 'n_attempts': 0,
        }
        if target_iso_voxels > 0:
            boundary_ids = boundary_grain_ids(fm_base.lgi) if isolated_grain_strategy == 'boundary' else None
            ra_selection_info = select_isolated_units_with_tolerance(
                grain_ids, grain_sizes_map, fm_base.neigh_gid, target_iso_voxels, total_voxels,
                tolerance=retained_austenite_tolerance,
                max_attempts=retained_austenite_max_attempts,
                trim_step=retained_austenite_trim_step,
                strategy=isolated_grain_strategy, isolated_size_tol=isolated_size_tol,
                boundary_unit_ids=boundary_ids, rng=rng,
            )
            isolated = ra_selection_info.pop('selected')

        lgi_for_split = fm_base.lgi.copy()
        if isolated:
            lgi_for_split[np.isin(lgi_for_split, list(isolated))] = 0

        packet_lgi, packet_to_grain, clusters_dict = split_grains_geodesic(
            lgi_for_split, pag_size_distribution, min_voxels_to_split=split_threshold,
            connectivity=fm_base.connectivity, random_seed=_sub_seeds[1],
        )
        next_id = max(packet_to_grain.keys(), default=0) + 1
        for g in isolated:
            mask = fm_base.lgi == g
            packet_lgi[mask] = next_id
            packet_to_grain[next_id] = g
            clusters_dict[g] = [next_id]
            next_id += 1

        new_base = FMSteel3DBase.from_lfi(
            packet_lgi, physical_dimensions=fm_base.physical_dimensions,
            voxel_size=fm_base.voxel_size, units=fm_base.units,
            connectivity=fm_base.connectivity, min_grain_nvoxels=-1,
            random_seed=random_seed, verbosity=fm_base._verbosity, log_sink=fm_base._log_sink,
        )
        packet_sizes_map = {p: len(v) for p, v in new_base.grain_locs.items()}
        neigh_clid, _ = _pag_level_neigh_and_sizes(clusters_dict, packet_sizes_map, new_base.neigh_gid)
        # Technique B already keeps a retained-austenite grain as a proper
        # (1-packet) PAG entry in clusters_dict -- pag_id == the original
        # grain id here, so retained_austenite_pag_ids is just `isolated`
        # itself, no PAG-id translation needed.
        return FMSteel3DWithPAGs(
            parent=new_base, clusters_dict=clusters_dict, neigh_clid=neigh_clid,
            pag_orientations={}, isolated_grains=isolated,
            retained_austenite_pag_ids=set(isolated),
            target_pag_grain_fraction=pag_grain_fraction,
            retained_austenite_selection_info=ra_selection_info,
            random_seed=random_seed,
            verbosity=fm_base._verbosity, log_sink=fm_base._log_sink,
        )

    # --- Technique A ---
    cfg = _LEVER_DEFAULTS[lever]
    conn = clustering_connectivity if clustering_connectivity is not None else \
        (cfg['clustering_connectivity'] if cfg['clustering_connectivity'] is not None else fm_base.connectivity)

    if cfg['mis']:
        neigh_for_clustering = fm_base.neigh_gid if conn == fm_base.connectivity \
            else fm_base.compute_neighbor_network(connectivity=conn)
        clusters_dict = cluster_grains_simultaneous(
            neigh_for_clustering, grain_ids, pag_size_distribution,
            max_packets_per_pag=max_packets_per_pag, random_seed=_sub_seeds[2],
        )
    else:
        # generate_pag_clusters has no parameter to override the neighbour
        # graph it clusters on, and fm_base is a caller-owned, potentially
        # shared object -- so rather than mutating fm_base.neigh_gid in
        # place (even temporarily, restored via try/finally) to smuggle in
        # the widened-connectivity graph, operate on a shallow copy instead.
        # A shallow copy shares the same lgi/grain_locs arrays (untouched by
        # clustering) but gets its own independent neigh_gid slot, so nothing
        # else holding a reference to fm_base can ever observe the widened
        # graph, even transiently from another thread.
        if conn == fm_base.connectivity:
            fm_base_for_clustering = fm_base
        else:
            fm_base_for_clustering = copy.copy(fm_base)
            fm_base_for_clustering.neigh_gid = fm_base.compute_neighbor_network(connectivity=conn)
        fm_pag_tmp = fm_base_for_clustering.generate_pag_clusters(
            pag_size_distribution=pag_size_distribution, pag_grain_fraction=1.0,
            use_non_neigh_pag=use_non_neigh_pag, isolated_grain_strategy='auto', random_seed=random_seed,
        )
        clusters_dict = fm_pag_tmp.clusters_dict

    if cfg['repair']:
        neigh_for_repair = fm_base.neigh_gid if repair_connectivity == fm_base.connectivity \
            else fm_base.compute_neighbor_network(connectivity=repair_connectivity)
        clusters_dict = repair_singleton_pags(
            clusters_dict, neigh_for_repair, max_packets_per_pag=max_packets_per_pag,
            random_seed=_sub_seeds[3],
        )

    isolated_pag_ids = set()
    ra_selection_info = {
        'target_fraction': 1.0 - pag_grain_fraction, 'achieved_fraction': 0.0,
        'tolerance': retained_austenite_tolerance, 'converged': True, 'n_attempts': 0,
    }
    if target_iso_voxels > 0:
        pag_neigh, pag_sizes = _pag_level_neigh_and_sizes(clusters_dict, grain_sizes_map, fm_base.neigh_gid)
        boundary_pag_ids = None
        if isolated_grain_strategy == 'boundary':
            b_grains = boundary_grain_ids(fm_base.lgi)
            grain_to_pag = {g: pid for pid, gids in clusters_dict.items() for g in gids}
            boundary_pag_ids = {grain_to_pag[g] for g in b_grains if g in grain_to_pag}
        ra_selection_info = select_isolated_units_with_tolerance(
            list(clusters_dict.keys()), pag_sizes, pag_neigh, target_iso_voxels, total_voxels,
            tolerance=retained_austenite_tolerance,
            max_attempts=retained_austenite_max_attempts,
            trim_step=retained_austenite_trim_step,
            strategy=isolated_grain_strategy, isolated_size_tol=isolated_size_tol,
            boundary_unit_ids=boundary_pag_ids, rng=rng,
        )
        isolated_pag_ids = ra_selection_info.pop('selected')

    # Retained-austenite PAGs stay in clusters_dict -- no renumbering, no
    # exclusion. assign_pag_orientations() will assign every PAG here
    # (transformed or not) a proper orientation uniformly; generate_blocks()
    # is what later skips retained_austenite_pag_ids. isolated_grains is
    # kept as a flattened, backward-compatible view onto the same PAGs.
    isolated_grains: set = set()
    for pid in isolated_pag_ids:
        isolated_grains.update(clusters_dict[pid])

    neigh_clid, _ = _pag_level_neigh_and_sizes(clusters_dict, grain_sizes_map, fm_base.neigh_gid)

    return FMSteel3DWithPAGs(
        parent=fm_base, clusters_dict=clusters_dict, neigh_clid=neigh_clid,
        pag_orientations={}, isolated_grains=isolated_grains,
        retained_austenite_pag_ids=isolated_pag_ids,
        target_pag_grain_fraction=pag_grain_fraction,
        retained_austenite_selection_info=ra_selection_info,
        random_seed=random_seed,
        verbosity=fm_base._verbosity, log_sink=fm_base._log_sink,
    )
