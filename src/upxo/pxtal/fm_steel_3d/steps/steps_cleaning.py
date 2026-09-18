"""Cleaning & Topology Repair -- Part C of the FM Steel 3D walkthrough.

Dissolves undersized grains into their largest neighbour and (optionally)
repairs single-voxel "spikes", matching GsCleanPage's own default
operation order (merge small grains, then remove spikes).
"""


def clean(fm_base, threshold=8, max_passes=100, do_remove_spikes=True):
    """Cleans `fm_base`: dissolves grains below `threshold` voxels into
    their largest neighbour (up to `max_passes` iterations), then optionally
    removes single-voxel spikes. Neither operation modifies `fm_base` --
    each returns a new FMSteel3DBase instance.

    Returns
    -------
    (new_fm_base, stats) : tuple[FMSteel3DBase, dict]
        `stats` is `new_fm_base.get_grain_statistics()` (keys: 'n_grains',
        'min_voxels', 'max_voxels', 'mean_voxels', 'median_voxels',
        'total_voxels', 'domain_voxels').
    """
    cur = fm_base.clean_small_grains(threshold, max_passes=max_passes)
    if do_remove_spikes:
        cur, _n_spikes = cur.remove_spikes()
    return cur, cur.get_grain_statistics()
