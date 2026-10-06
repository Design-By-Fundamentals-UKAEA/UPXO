"""Shared grain-ID relabel/randomize and clean-structure construction helpers.

Used by the transformation steps (``steps.steps_transformations``): every
transform or round-trip ends with the same "clean, contiguous, randomized ID
set, optionally with tiny grains dissolved into their largest neighbour"
guarantee, implemented once here.
"""
import numpy as np


def relabel_and_randomize(lgi):
    """Relabel grain IDs to a contiguous 1..N range, then shuffle them.

    Nearest-neighbour or majority-vote resampling can drop small grains
    entirely, leaving gaps in the surviving grain-ID sequence (for example
    {1, 2, 4, 7, ...} if grains 3, 5 and 6 did not survive). A freshly
    simulated Monte Carlo structure is always relabelled (cc3d) and shuffled
    (``gid_ops.shuffleLFIIDs``); a raw resampled or round-tripped structure is
    neither. This applies the same two-step convention."""
    unique_ids = np.unique(lgi)
    unique_ids = unique_ids[unique_ids != 0]
    id_map = {0: 0}
    for new_id, old_id in enumerate(unique_ids, start=1):
        id_map[int(old_id)] = new_id
    relabeled = np.vectorize(id_map.get)(lgi).astype(lgi.dtype)

    from upxo.gsdataops.gid_ops import shuffleLFIIDs
    return shuffleLFIIDs(None, relabeled)


def build_clean_structure(raw_lgi, voxel_size, physical_dimensions, units,
                          connectivity, seed, cleanup_threshold):
    """Common final-construction step for a transformed structure.

    Relabels and randomizes ``raw_lgi``'s grain IDs to a contiguous, shuffled
    range, then, if ``cleanup_threshold`` > 0, dissolves any grain below that
    many voxels into its largest neighbour (``FMSteel3DBase.from_lfi``'s own
    ``min_grain_nvoxels`` cleanup) and relabels AGAIN, since the cleanup can
    itself reopen gaps in the ID sequence. Returns the final FMSteel3DBase."""
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    relabeled = relabel_and_randomize(raw_lgi)
    threshold = max(0, int(cleanup_threshold)) if cleanup_threshold else 0
    new_fm = FMSteel3DBase.from_lfi(
        relabeled, physical_dimensions=physical_dimensions, voxel_size=voxel_size,
        units=units, connectivity=connectivity,
        min_grain_nvoxels=(threshold if threshold > 0 else -1),
        random_seed=seed, verbosity=0)
    if threshold > 0:
        final_lgi = relabel_and_randomize(new_fm.lgi)
        new_fm = FMSteel3DBase.from_lfi(
            final_lgi, physical_dimensions=physical_dimensions, voxel_size=voxel_size,
            units=units, connectivity=connectivity, min_grain_nvoxels=-1,
            random_seed=seed, verbosity=0)
    return new_fm
