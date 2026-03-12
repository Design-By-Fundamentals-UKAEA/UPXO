"""
Grain/Feature ID operations.

This module provides utilities for Labelled Feature Image (LFI) processing,
including neighbour detection, neighbour subset extraction, feature-ID
classification (boundary/internal/corner/edge), and helper operations for
2D/3D grain-label arrays.

Imports
-------
import upxo.gsdataops.gid_ops as gidOps

Metadata
--------
* Module: upxo.gsdataops.gid_ops
* Package: upxo
* License: GPL-3.0-only
* Author: Dr. Sunil Anandatheertha
* Email: vaasu.anandatheertha@ukaea.uk
* Status: Active development
* Last updated: 2026-03-12

Applications
------------
* O(1) neighbourhood extraction in 2D and 3D LFIs
* Probabilistic neighbour down-selection
* Boundary/internal/corner/edge feature-ID detection
* Island and small-feature identification

Definitions
-----------
* LFI: Labelled Feature Image
"""

import numpy as np
import pandas as pd
from upxo._sup import dataTypeHandlers as dth
from numba import njit, types
from numba.typed import Dict
import random
import cc3d
from collections import defaultdict

DXY_8 = np.array([(-1, -1), (-1, 0), (-1, 1), 
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1),], dtype=np.int32)

DXYZ_26 = np.array([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
                    (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
                    (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
                    ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
                    ( 0,  0, -1),              ( 0,  0, 1),
                    ( 0,  1, -1), ( 0,  1, 0), ( 0,  1, 1),
                    ( 1, -1, -1), ( 1, -1, 0), ( 1, -1, 1),
                    ( 1,  0, -1), ( 1,  0, 0), ( 1,  0, 1),
                    ( 1,  1, -1), ( 1,  1, 0), ( 1,  1, 1)], dtype=np.int32)

def get_all_masks(section2d, as_coordinates=False):
    """Build per-ID index/coordinate masks from a 2D labelled section.

    Parameters
    ----------
    section2d : ndarray
        2D Labelled Feature Image.
    as_coordinates : bool, optional
        If True, return coordinates ``(row, col)`` for each ID. If False,
        return flattened index locations.

    Returns
    -------
    dict
        Mapping of ``feature_id -> indices`` or ``feature_id -> Nx2
        coordinates`` depending on ``as_coordinates``.
    """
    # Flatten the array and get indices
    flat_section = section2d.ravel()
    # Get the unique IDs and the indices where they occur
    # sorted_indices will group all identical IDs together
    sorted_indices = np.argsort(flat_section)
    sorted_ids = flat_section[sorted_indices]
    
    # Find where the IDs change
    diffs = np.diff(sorted_ids)
    boundaries = np.where(diffs != 0)[0] + 1
    
    # Split the sorted indices into groups based on Grain ID
    split_indices = np.split(sorted_indices, boundaries)
    unique_ids = sorted_ids[np.concatenate(([0], boundaries))]
    
    # lookup for pixel indices
    id_to_indices = dict(zip(unique_ids, split_indices))
    if as_coordinates:
        id_to_coords = {}
        for grain_id, indices in id_to_indices.items():
            # unravel_index returns (row_array, col_array)
            rows, cols = np.unravel_index(indices, section2d.shape)
            # Combine them into a single [N, 2] array of [i, j] pairs
            id_to_coords[grain_id] = np.column_stack((rows, cols))
        return id_to_coords
    
    return id_to_indices

def find_O1_neigh_2d(lgi, p=1.0, include_central_grain=False, throw_numba_dict=False,
                     validate_input=True, verbosity_nfids=1000):
    """
    Find first-order neighbours in 2D label grid.

    Parameters
    ----------.
    lgi : ndarray
        2D label grid of grain IDs.
    p : float, optional
        Probability of including each neighbour. Default is 1.0 (all neighbours).
    include_central_grain : bool, optional
        If True, include the central grain ID in its own neighbour list.
    throw_numba_dict : bool, optional
        If True, return the raw numba Dict object. Default is False.
    validate_input : bool, optional
        If True, validate input parameters. Default is True.
    verbosity_nfids : int, optional
        Number of fids for verbosity control (not used here).

    Returns
    -------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are lists of neighbour grain IDs.
    """
    lgi32 = lgi.astype(np.int32)
    if throw_numba_dict:
        return _find_neigh_gid_numba_2d_(lgi32)
    else:
        neigh_fids = {int(k): list(map(int, v)) for k, v in _find_neigh_gid_numba_2d_(lgi32).items() if int(k) != 0}
        neigh_fids = {k: v.append(k) or v for k, v in neigh_fids.items()} if include_central_grain else neigh_fids
    if p == 1.0:
        return neigh_fids
    neigh_fids = select_neighs_with_probability(neigh_fids, p=p, 
                                                include_central_grain=include_central_grain, 
                                                validate_input=validate_input)
    return neigh_fids

def find_O1_neigh_3d(lgi, p=1.0, include_central_grain=False, throw_numba_dict=False,
                     validate_input=True):
    """
    Find first-order neighbours in 3D label grid.

    Parameters
    ----------.
    lgi : ndarray
        3D label grid of grain IDs.
    p : float, optional
        Probability of including each neighbour. Default is 1.0 (all neighbours).
    include_central_grain : bool, optional
        If True, include the central grain ID in its own neighbour list.
    throw_numba_dict : bool, optional
        If True, return the raw numba Dict object. Default is False.
    validate_input : bool, optional
        If True, validate input parameters. Default is True.

    Returns
    -------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are lists of neighbour grain IDs.

    Import
    ------
    import upxo.gsdataops.gid_ops as gidOps
    Use as: gidOps.find_O1_neigh_3d
    """
    lgi32 = lgi.astype(np.int32)
    if throw_numba_dict:
        return _find_neigh_gid_numba_3d_(lgi32)
    else:
        neigh_fids = {int(k): list(map(int, v)) for k, v in _find_neigh_gid_numba_3d_(lgi32).items() if int(k) != 0}
        neigh_fids = {k: v.append(k) or v for k, v in neigh_fids.items()} if include_central_grain else neigh_fids
    neigh_fids = select_neighs_with_probability(neigh_fids, p=p, 
                                                include_central_grain=include_central_grain, 
                                                validate_input=validate_input)
    return neigh_fids

def find_O1_neigh_2d_fids(lgi, fids, p=1.0, include_central_grain=False, throw_numba_dict=False,
                          validate_input=True, verbosity_nfids=1000):
    """
    Find first-order neighbours in 2D label grid for specific feature IDs.
    
    Parameters
    ----------.
    lgi : ndarray
        2D label grid of grain IDs.
    fids : list or ndarray
        List or array of grain IDs for which neighbours are required.
    p : float, optional
        Probability of including each neighbour. Default is 1.0 (all neighbours).
    include_central_grain : bool, optional
        If True, include the central grain ID in its own neighbour list.
    throw_numba_dict : bool, optional
        If True, return the raw numba Dict object. Default is False.
    validate_input : bool, optional
        If True, validate input parameters. Default is True.
    verbosity_nfids : int, optional
        Number of fids for verbosity control (not used here).

    Returns
    -------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are lists of neighbour grain IDs.
    """
    lgi32 = lgi.astype(np.int32)
    fids_arr = np.asarray(fids, dtype=np.int32).ravel()
    if throw_numba_dict:
        return _find_neigh_selected_2d_(lgi32, fids_arr)
    else:
        neigh_fids = {int(k): list(map(int, v)) for k, v in _find_neigh_selected_2d_(lgi32, fids_arr).items() if int(k) != 0}
        neigh_fids = {k: v.append(k) or v for k, v in neigh_fids.items()} if include_central_grain else neigh_fids
    neigh_fids = select_neighs_with_probability(neigh_fids, p=p, 
                                                include_central_grain=include_central_grain,
                                                validate_input=validate_input)
    return neigh_fids

def find_O2_neigh_3d_fids(lgi, fids, p=1.0, include_central_grain=False, throw_numba_dict=False,
                          validate_input=True, verbosity_nfids=2500):
    """Find first-order neighbours in 3D for selected feature IDs.

    Parameters
    ----------
    lgi : ndarray
        3D Labelled Feature Image.
    fids : list or ndarray
        Feature IDs for which neighbours are required.
    p : float, optional
        Probability of retaining each neighbour. Default is 1.0.
    include_central_grain : bool, optional
        If True, include each central feature ID in its own neighbour list.
    throw_numba_dict : bool, optional
        If True, return the raw numba ``Dict`` output.
    validate_input : bool, optional
        If True, validate probability settings in post-processing.
    verbosity_nfids : int, optional
        Reserved verbosity control argument.

    Returns
    -------
    dict or numba.typed.Dict
        Neighbour mapping for requested feature IDs.
    """
    lgi32 = lgi.astype(np.int32)
    fids_arr = np.asarray(fids, dtype=np.int32).ravel()
    if throw_numba_dict:
        return _find_neigh_selected_3d_(lgi32, fids_arr)
    else:
        neigh_fids = {int(k): list(map(int, v)) for k, v in _find_neigh_selected_3d_(lgi32, fids_arr).items() if int(k) != 0}
        neigh_fids = {k: v.append(k) or v for k, v in neigh_fids.items()} if include_central_grain else neigh_fids
    neigh_fids = select_neighs_with_probability(neigh_fids, p=p,
                                                include_central_grain=include_central_grain,
                                                validate_input=validate_input)
    return neigh_fids

def extract_neigh_gid_subset(neigh_fids={}, subset_fids=[], type_correction=True,
                             validate_input=True, verbosity_nfids=1000):
    '''
    Extract a subset of neigh_fid dict for usert specified fids.

    Parameters:
    -----------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are lists of neighbor grain IDs.
    subset_fids : list
        List of grain IDs to extract from the neighbor lists.
    type_correction : bool, optional
        If True, convert all grain IDs to integers. Default is True.
    validate_input : bool, optional
        If True, validate the input types. Default is True.

    Returns:
    --------
    extracted_neigh_fids : dict
        Dictionary with the same keys as neigh_fids, but values are lists of neighbor grain IDs that are in subset_fids.
    '''
    if validate_input:
        if not isinstance(neigh_fids, dict):
            raise ValueError("neigh_fids must be a dictionary.")
        if type(subset_fids) not in dth.dt.ITERABLES:
            raise ValueError("subset_fids must be an iterable.")
    if type_correction:
        # NGSS: Neigh Gid Sub-Set
        NGSS = {int(cgid): [int(i) for i in neigh_fids[cgid]] 
                for cgid in subset_fids if cgid in neigh_fids.keys()}
    else:
        NGSS = {cgid: neigh_fids[cgid] 
                for cgid in subset_fids if cgid in neigh_fids.keys()}

    return NGSS

def select_neighs_with_probability(neigh_fids, p=1.0, include_central_grain=False, 
                                   validate_input=True, verbosity_nfids=1000):
    """
    Helper function to select neighbours with a given probability once neighbours have been found.

    Import
    ------
    import upxo.gsdataops.gid_ops as gidOps
    Use as: gidOps.select_neighs_with_probability
    """
    if validate_input:
        if not (0.0 < p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {p}")
    for gid, original_neighbors in neigh_fids.items():
        # Split central vs non-central neighbours
        noncentral = [n for n in original_neighbors if n != gid]
        # If there are no true neighbours, nothing to enforce
        if not noncentral:
            # list is either [gid] (when include_central_grain=True)
            # or [] (when include_central_grain=False)
            continue
        # Sample non-central neighbours with probability p
        sampled_noncentral = [n for n in noncentral if random.random() <= p]
        # Guarantee: if all were dropped but at least one existed,
        # pick one neighbour at random from original noncentral set
        if not sampled_noncentral:
            sampled_noncentral = [random.choice(noncentral)]
        if include_central_grain:
            # central always present
            neigh_fids[gid] = [gid] + sampled_noncentral
        else:
            neigh_fids[gid] = sampled_noncentral
    return neigh_fids

@njit
def _find_neigh_gid_numba_2d_(lgi):
    """
    Compute neighbouring grain IDs in 2D (8-neighbourhood).

    Parameters
    ----------
    lgi : int32[:, :]
        2D label grid of grain IDs. Assumes labels in [0, max_gid].

    Returns
    -------
    neigh_gid : Dict[int32, int32[:]]
        For each gid in [0, max_gid], an array of unique neighbour IDs.
    """
    shape_x, shape_y = lgi.shape
    max_gid = np.int32(lgi.max())
    neigh_gid = Dict.empty(key_type=types.int32, value_type=types.int32[:])
    max_neighbors = 8  # 2D Moore neighbourhood upper bound
    neighbor_counts = np.zeros(max_gid+1, dtype=np.int32)
    # Preallocate padded arrays
    for gid in range(max_gid + 1):
        neigh_gid[gid] = np.full(max_neighbors, -1, dtype=np.int32)
    # Scan image
    for x in range(shape_x):
        for y in range(shape_y):
            grain_id = lgi[x, y]
            for k in range(DXY_8.shape[0]):
                dx = DXY_8[k, 0]
                dy = DXY_8[k, 1]
                nx = x + dx
                ny = y + dy
                if 0 <= nx < shape_x and 0 <= ny < shape_y:
                    neighbor_id = lgi[nx, ny]
                    if neighbor_id != grain_id:
                        count = neighbor_counts[grain_id]
                        # avoid duplicates
                        found = False
                        for i in range(count):
                            if neigh_gid[grain_id][i] == neighbor_id:
                                found = True
                                break
                        if not found:
                            neigh_gid[grain_id][count] = neighbor_id
                            neighbor_counts[grain_id] += 1
    # Trim padding and return final dict
    final_neigh_gid = Dict.empty(key_type=types.int32, value_type=types.int32[:])
    for gid in range(max_gid + 1):
        n = neighbor_counts[gid]
        trimmed = np.empty(n, dtype=np.int32)
        for i in range(n):
            trimmed[i] = neigh_gid[gid][i]
        final_neigh_gid[gid] = trimmed

    return final_neigh_gid

@njit
def _find_neigh_gid_numba_3d_(lgi):
    """
    Compute neighbouring grain IDs in 3D (26-neighbourhood).

    Parameters
    ----------
    lgi : int32[:, :, :]
        3D label grid of grain IDs. Assumes labels in [0, max_gid].

    Returns
    -------
    neigh_gid : Dict[int32, int32[:]]
        For each gid in [0, max_gid], an array of unique neighbour IDs
        (no padding, no -1s).
    """
    shape_x, shape_y, shape_z = lgi.shape
    max_gid = np.int32(lgi.max())
    # Get unique grain IDs to avoid allocating for non-existent grains
    unique_gids = np.unique(lgi)
    num_grains = len(unique_gids)
    # Use temporary storage dict for building neighbor lists
    neigh_gid_temp = Dict.empty(key_type=types.int32, value_type=types.int32[:])
    max_neighbors = 26  # 3D Moore neighbourhood upper bound
    # Create mapping for efficient lookup and counting
    neighbor_counts = Dict.empty(key_type=types.int32, value_type=types.int32)
    # Initialize only for grains that actually exist
    for gid in unique_gids:
        neigh_gid_temp[gid] = np.full(max_neighbors, -1, dtype=np.int32)
        neighbor_counts[gid] = np.int32(0)
    # Scan volume
    for x in range(shape_x):
        for y in range(shape_y):
            for z in range(shape_z):
                grain_id = lgi[x, y, z]
                for k in range(DXYZ_26.shape[0]):
                    dx = DXYZ_26[k, 0]
                    dy = DXYZ_26[k, 1]
                    dz = DXYZ_26[k, 2]
                    nx = x+dx
                    ny = y+dy
                    nz = z+dz
                    if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                        neighbor_id = lgi[nx, ny, nz]
                        if neighbor_id != grain_id:
                            count = neighbor_counts[grain_id]
                            # check for duplicate
                            found = False
                            for i in range(count):
                                if neigh_gid_temp[grain_id][i] == neighbor_id:
                                    found = True
                                    break
                            if not found:
                                neigh_gid_temp[grain_id][count] = neighbor_id
                                neighbor_counts[grain_id] = count + np.int32(1)
    # Trim padding and return final dict
    final_neigh_gid = Dict.empty(key_type=types.int32, value_type=types.int32[:])
    for gid in unique_gids:
        n = neighbor_counts[gid]
        trimmed = np.empty(n, dtype=np.int32)
        for i in range(n):
            trimmed[i] = neigh_gid_temp[gid][i]
        final_neigh_gid[gid] = trimmed

    return final_neigh_gid


@njit
def _find_neigh_selected_2d_(lgi, selected_fids):
    """
    Compute neighbouring grain IDs in 2D (8-neighbourhood)
    only for a given set of central grain IDs.

    Parameters
    ----------
    lgi : int32[:, :]
        2D label grid of grain IDs. Assumes labels in [0, max_gid].
    selected_fids : int32[:]
        1D array of grain IDs for which neighbours are required.

    Returns
    -------
    neigh_gid : Dict[int32, int32[:]]
        For each gid in selected_fids (and present in lgi),
        an array of unique neighbour IDs.
    """
    H, W = lgi.shape
    ng = selected_fids.size
    # Map gid -> row index (0..ng-1)
    gid2idx = Dict.empty(key_type=types.int32, value_type=types.int32,)
    for i in range(ng):
        gid = selected_fids[i]
        gid2idx[gid] = i
    max_neighbors = 8
    neigh_table = np.full((ng, max_neighbors), -1, dtype=np.int32)
    counts = np.zeros(ng, dtype=np.int32)
    # Scan image
    for x in range(H):
        for y in range(W):
            grain_id = lgi[x, y]
            # Only process central grains we care about
            if grain_id in gid2idx:
                idx = gid2idx[grain_id]
                for k in range(DXY_8.shape[0]):
                    dx = DXY_8[k, 0]
                    dy = DXY_8[k, 1]
                    nx = x+dx
                    ny = y+dy
                    if 0 <= nx < H and 0 <= ny < W:
                        neighbor_id = lgi[nx, ny]
                        if neighbor_id != grain_id:
                            # check for duplicate
                            count = counts[idx]
                            found = False
                            for j in range(count):
                                if neigh_table[idx, j] == neighbor_id:
                                    found = True
                                    break
                            if not found:
                                # There are at most 8 first-order neighbours
                                neigh_table[idx, count] = neighbor_id
                                counts[idx] = count + 1
    # Build Dict[gid -> trimmed neighbour array]
    neigh_gid = Dict.empty(key_type=types.int32, value_type=types.int32[:],)
    for i in range(ng):
        gid = selected_fids[i]
        c = counts[i]
        arr = np.empty(c, dtype=np.int32)
        for j in range(c):
            arr[j] = neigh_table[i, j]
        neigh_gid[gid] = arr

    return neigh_gid

@njit(cache=True)
def _find_neigh_selected_3d_(lgi, selected_fids):
    """
    Compute neighbouring grain IDs in 3D (26-neighbourhood)
    only for a given set of central grain IDs.

    Parameters
    ----------
    lgi : int32[:, :, :]
        3D label grid of grain IDs.
    selected_fids : int32[:]
        1D array of grain IDs for which neighbours are required.

    Returns
    -------
    neigh_gid : Dict[int32, int32[:]]
        For each gid in selected_fids, an array of unique neighbour IDs.
    """
    nx, ny, nz = lgi.shape
    ng = selected_fids.size
    # gid -> row index (0..ng-1)
    gid2idx = Dict.empty(key_type=types.int32, value_type=types.int32,)
    for i in range(ng):
        gid = selected_fids[i]
        gid2idx[gid] = i
    max_neighbors = 26
    neigh_table = np.full((ng, max_neighbors), -1, dtype=np.int32)
    counts = np.zeros(ng, dtype=np.int32)
    # Scan volume
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                grain_id = lgi[x, y, z]
                # Only process central grains we care about
                if grain_id in gid2idx:
                    idx = gid2idx[grain_id]
                    for k in range(DXYZ_26.shape[0]):
                        dx = DXYZ_26[k, 0]
                        dy = DXYZ_26[k, 1]
                        dz = DXYZ_26[k, 2]
                        xx = x+dx
                        yy = y+dy
                        zz = z+dz
                        if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                            neighbor_id = lgi[xx, yy, zz]
                            if neighbor_id != grain_id:
                                c = counts[idx]
                                # avoid duplicates
                                found = False
                                for j in range(c):
                                    if neigh_table[idx, j] == neighbor_id:
                                        found = True
                                        break
                                if not found:
                                    # max 26 first-order neighbours
                                    neigh_table[idx, c] = neighbor_id
                                    counts[idx] = c + 1
    # Build Dict[gid -> trimmed neighbour array]
    neigh_gid = Dict.empty(key_type=types.int32, value_type=types.int32[:],)
    for i in range(ng):
        gid = selected_fids[i]
        c = counts[i]
        arr = np.empty(c, dtype=np.int32)
        for j in range(c):
            arr[j] = neigh_table[i, j]
        neigh_gid[gid] = arr

    return neigh_gid

def shuffleLFIIDs(self, lfi):
    """
    Shuffle LFI IDs to randomize grain labels while preserving structure.
    Import
    ------
    from upxo.gsdataops.gid_ops import shuffleLFIIDs
    """
    unique_ids = np.unique(lfi)
    shuffled_ids = unique_ids.copy()
    np.random.shuffle(shuffled_ids)
    id_map = {old_id: new_id for old_id, new_id in zip(unique_ids, shuffled_ids)}
    lfi = np.vectorize(id_map.get)(lfi)
    return lfi

def find_neighs2d(lfi, conn):
    """Find neighboring grain IDs in 2D using connected components and region graph.
    Parameters
    ----------
    lfi : ndarray
        2D Labelled Feature Image.
    conn : int
        Connectivity for defining neighbors (e.g., 4 or 8).

    Returns
    -------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are arrays of neighboring grain IDs.

    Import
    ------
    from upxo.gsdataops.gid_ops import find_neighs3d
    """
    edges = cc3d.region_graph(lfi, connectivity=conn)
    neigh_fids = defaultdict(set)
    for edge in edges:
        gid1, gid2 = edge
        neigh_fids[gid1].add(gid2)
        neigh_fids[gid2].add(gid1)
    neigh_fids = {int(gid): np.array(sorted(neighbors), dtype=np.int32) for gid, neighbors in neigh_fids.items()}
    return neigh_fids

def find_neighs3d(lfi, conn):
    """Find neighboring grain IDs in 3D using connected components and region graph.
    Parameters
    ----------
    lfi : ndarray
        3D Labelled Feature Image.
    conn : int
        Connectivity for defining neighbors (e.g., 6, 18, or 26).

    Returns
    -------
    neigh_fids : dict
        Dictionary where keys are grain IDs and values are arrays of neighboring grain IDs.

    Import
    ------
    from upxo.gsdataops.gid_ops import find_neighs3d
    """
    edges = cc3d.region_graph(lfi, connectivity=conn)
    neigh_fids = defaultdict(set)
    for edge in edges:
        gid1, gid2 = edge
        neigh_fids[gid1].add(gid2)
        neigh_fids[gid2].add(gid1)
    neigh_fids = {int(gid): np.array(sorted(neighbors), dtype=np.int32) for gid, neighbors in neigh_fids.items()}
    return neigh_fids

def get_nth_order_neighbors(target_fid, neigh_fids, n_order=1):
    """Recursively finds Nth order neighbors using the neigh_fids dictionary.

    Import
    ------
    from upxo.gsdataops.gid_ops import get_nth_order_neighbors
    """
    cluster = {target_fid}
    for _ in range(n_order):
        next_gen = set()
        for fid in cluster:
            if fid in neigh_fids: next_gen.update(neigh_fids[fid])
        cluster.update(next_gen)
    return cluster

def detect_islands(neigh_fids):
    """Detect grain IDs that have exactly one listed neighbour.

    Parameters
    ----------
    neigh_fids : dict
        Mapping ``gid -> iterable of neighbouring gids``.

    Returns
    -------
    list
        Grain IDs whose neighbour list length is exactly one.

    Notes
    -----
    This utility is commonly used with neighbour maps that include the
    central grain ID itself. In that case, an "island" corresponds to a grain
    that has no true external neighbours.
    """
    islands = []
    for gid, neighs in neigh_fids.items():
        if len(neighs) == 1:
            islands.append(gid)
    return islands

def find_small_fids(lfi, threshold):
    """Find grain IDs whose voxel/pixel count is below or equal to threshold.

    Parameters
    ----------
    lfi : ndarray
        N-dimensional Labelled Feature Image.
    threshold : int
        Maximum allowed size (count) for a feature to be classified as small.

    Returns
    -------
    numpy.ndarray
        1D array of small feature IDs.
    """
    small_fids = np.where(np.bincount(lfi.ravel())[1:] <= threshold)[0]+1 
    return small_fids

def find_boundary_fids2d(lfi):
    """Return unique feature IDs touching the outer boundary of a 2D grid.

    Parameters
    ----------
    lfi : ndarray
        2D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs appearing on any of the four outer edges.
    """
    boundLFI = np.unique(np.hstack((np.unique(lfi[0, :]), np.unique(lfi[-1, :]),
                     np.unique(lfi[:, 0]), np.unique(lfi[:, -1]))))
    return boundLFI

def find_boundary_fids3d(lfi):
    """Return unique feature IDs touching the outer boundary of a 3D volume.

    Parameters
    ----------
    lfi : ndarray
        3D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs appearing on any of the six outer faces.
    """
    boundLFI = np.unique(np.hstack((np.unique(lfi[0, :, :]), np.unique(lfi[-1, :, :]),
                     np.unique(lfi[:, 0, :]), np.unique(lfi[:, -1, :]), np.unique(lfi[:, :, 0]),
                     np.unique(lfi[:, :, -1]))))
    return boundLFI

def find_internal_fids2d(lfi):
    """Return 2D feature IDs that do not touch the domain boundary.

    Parameters
    ----------
    lfi : ndarray
        2D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs present only in the interior.
    """
    return np.setdiff1d(np.unique(lfi), find_boundary_fids2d(lfi))

def find_internal_fids3d(lfi):
    """Return 3D feature IDs that do not touch the domain boundary.

    Parameters
    ----------
    lfi : ndarray
        3D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs present only in the interior.
    """
    return np.setdiff1d(np.unique(lfi), find_boundary_fids3d(lfi))

def find_corner_fids2d(lfi):
    """Return unique feature IDs located at 2D domain corners.

    Parameters
    ----------
    lfi : ndarray
        2D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs from the four corner pixels.
    """
    cornerLFI = np.unique(np.hstack((lfi[0, 0], lfi[0, -1], lfi[-1, 0], lfi[-1, -1])))
    return cornerLFI

def find_corner_fids3d(lfi):
    """Return unique feature IDs located at 3D domain corners.

    Parameters
    ----------
    lfi : ndarray
        3D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs from the eight corner voxels.
    """
    cornerLFI = np.unique(np.hstack((lfi[0, 0, 0], lfi[0, 0, -1], lfi[0, -1, 0], lfi[0, -1, -1],
                     lfi[-1, 0, 0], lfi[-1, 0, -1], lfi[-1, -1, 0], lfi[-1, -1, -1])))
    return cornerLFI

def find_edge_fids2d(lfi):
    """Return unique feature IDs present on 2D boundary edges.

    Parameters
    ----------
    lfi : ndarray
        2D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs from all four boundary edges.
    """
    return np.unique(np.hstack((lfi[0, :], lfi[-1, :], lfi[:, 0], lfi[:, -1])))

def find_edge_fids3d(lfi):
    """Return unique feature IDs present on 3D boundary edges.

    Parameters
    ----------
    lfi : ndarray
        3D Labelled Feature Image.

    Returns
    -------
    numpy.ndarray
        Sorted unique IDs sampled from the 12 domain edges.
    """
    return np.unique(np.hstack((lfi[0, 0, :], lfi[0, -1, :], lfi[-1, 0, :], lfi[-1, -1, :],
               lfi[0, :, 0], lfi[0, :, -1], lfi[-1, :, 0], lfi[-1, :, -1],
               lfi[:, 0, 0], lfi[:, 0, -1], lfi[:, -1, 0], lfi[:, -1, -1])))