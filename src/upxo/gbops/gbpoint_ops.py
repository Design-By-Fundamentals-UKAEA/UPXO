"""Grain-boundary point coordinate and ID helpers."""

import numpy as np
from skimage.segmentation import find_boundaries


def initialise_gbp(gids):
    """Return empty local/global grain-boundary point dictionaries."""
    return {gid: None for gid in gids}, {gid: None for gid in gids}


def local_gbp_for_gid(lgi_subset, gid):
    """Return local subpixel grain-boundary points for one grain."""
    masked = np.array(lgi_subset, copy=True)
    masked[masked == gid] = -1
    masked[masked != -1] = 0
    masked = np.abs(np.multiply(masked, lgi_subset))
    gbp = np.array(find_boundaries(masked, connectivity=1,
                                   mode='subpixel',
                                   background=0), dtype=int)
    return np.argwhere(gbp > 0)/2


def globalise_gbp(local_gbp_by_gid, bounds_ex, gids):
    """Translate local grain-boundary points to global coordinates."""
    local_translation = np.array([0.5, 0.5, 0.5])
    minextreme = {gid: [bounds_ex['zmins'][gid-1],
                        bounds_ex['ymins'][gid-1],
                        bounds_ex['xmins'][gid-1]]
                  for gid in gids}
    return {gid: local_gbp_by_gid[gid] + local_translation + minextreme[gid]
            for gid in gids}


def create_neighbor_pair_ids(neigh_gid):
    """Create unique neighbour grain-pair ID mappings."""
    gid_pair_ids = {}
    seen = set()
    pair_id = 1
    for gid, neighbors in neigh_gid.items():
        for neighbor in neighbors:
            pair = tuple(int(_) for _ in sorted((gid, neighbor)))
            if pair not in seen:
                gid_pair_ids[pair_id] = pair
                seen.add(pair)
                pair_id += 1
    unique_lr = np.unique(np.array(list(gid_pair_ids.values())), axis=0)
    unique_rl = np.flip(unique_lr, axis=1)
    reverse = {v: k for k, v in gid_pair_ids.items()}
    return gid_pair_ids, unique_lr, unique_rl, reverse


def gid_pair_orientation(gid_pair, unique_lr, unique_rl):
    """Return whether a pair is in left-right or right-left orientation."""
    pair = np.asarray(gid_pair)
    in_lr = any((unique_lr[:, 0] == pair[0]) & (unique_lr[:, 1] == pair[1]))
    if in_lr:
        return 'lr'
    in_rl = any((unique_rl[:, 0] == pair[0]) & (unique_rl[:, 1] == pair[1]))
    if in_rl:
        return 'rl'
    raise ValueError('Invalid gid_pair or corrupt database.')


def build_gbp_stack(global_gbp_by_gid, gids):
    """Stack and uniquefy all global grain-boundary points."""
    if len(gids) == 0:
        return np.empty((0, 3))
    stack = np.vstack([global_gbp_by_gid[gid] for gid in gids])
    return np.unique(stack, axis=0)


def build_gbp_ids(gbpstack):
    """Return grain-boundary point IDs for a stack."""
    return [i for i in range(gbpstack.shape[0])]


def build_gbp_id_mappings(gbpstack, gbpids, global_gbp_by_gid, gids):
    """Build coordinate-to-ID and grain-to-point-ID maps."""
    gbp_id_maps = {}
    for point, pointid in zip(gbpstack, gbpids):
        gbp_id_maps[(int(point[0]), int(point[1]), int(point[2]))] = pointid

    gbp_ids = {gid: None for gid in gids}
    for gid in gids:
        gbp_ids[gid] = set(gbp_id_maps[(int(point[0]), int(point[1]),
                                        int(point[2]))]
                           for point in global_gbp_by_gid[gid])
    return gbp_id_maps, gbp_ids


def grain_boundary_surface_point_ids(gid_pair_ids, gbp_ids):
    """Return interface point-ID sets for each neighbour grain-pair ID."""
    return {pair_id: gbp_ids[pair[0]].intersection(gbp_ids[pair[1]])
            for pair_id, pair in gid_pair_ids.items()}


def setup_gid_pair_gbp_ids(gid_pair_ids):
    """Return an empty grain-pair to gbp-ID mapping."""
    return {k: None for k in gid_pair_ids.keys()}


def find_gid_pair_gbp_ids(gidl, gidr, gid_pair_ids_rev, unique_lr,
                          unique_rl, gbsurf_pids_vox):
    """Return gbp IDs at the interface of two grain IDs."""
    gid_pair = (gidl, gidr)
    orientation = gid_pair_orientation(gid_pair, unique_lr, unique_rl)
    if orientation == 'rl':
        gid_pair = (gidr, gidl)
    interface_id = gid_pair_ids_rev[gid_pair]
    return list(gbsurf_pids_vox[interface_id])


def set_gid_pair_gbp_ids(gid_pair_ids, finder):
    """Populate all grain-pair gbp-ID lists using a finder callback."""
    return {gid_pair_id: finder(*gid_pair)
            for gid_pair_id, gid_pair in gid_pair_ids.items()}


def build_gid_to_gid_pair_ids(gids, gid_pair_ids):
    """Build map from grain ID to neighbour grain-pair IDs."""
    gid_gpid = {gid: [] for gid in gids}
    for intid, gpid in gid_pair_ids.items():
        gid_gpid[gpid[0]].append(intid)
        gid_gpid[gpid[1]].append(intid)
    return {gid: set(gid_gpid[gid]) for gid in gids}
