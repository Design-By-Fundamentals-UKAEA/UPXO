import numpy as np
from numba import njit, prange
from copy import deepcopy
"""
Module to handle neighbor operations in grain structures.

Import
------
import upxo.connops.neighbour_ops as neighOps
"""

@njit(parallel=True)
def NMB_get_neighbor_mask(neigh_mask, gid):
    """
    Scans the array. If a pixel equals gid, it checks 4-neighbors.
    If a neighbor is NOT gid, it is marked in the output mask.
    """
    nrows, ncols = neigh_mask.shape
    # prange allows parallel execution of the outer loop
    for row in prange(nrows):
        for col in range(ncols):
            # If the current pixel belongs to gid
            if neigh_mask[row, col] == gid:
                if row - 1 >= 0: # Check Up
                    if neigh_mask[row-1, col] != gid:
                        neigh_mask[row-1, col] = -1
                if row + 1 < nrows: # Check Down
                    if neigh_mask[row+1, col] != gid:
                        neigh_mask[row+1, col] = -1
                if col - 1 >= 0: # Check Left
                    if neigh_mask[row, col-1] != gid:
                        neigh_mask[row, col-1] = -1
                if col + 1 < ncols: # Check Right
                    if neigh_mask[row, col+1] != gid:
                        neigh_mask[row, col+1] = -1
            if neigh_mask[row, col] != -1:
                neigh_mask[row, col] = 0
    return neigh_mask

def find_neigh_fid(gdict, lfi, fid, nfeatures, include_central_grain=False, 
                   update_grain_object=True, user_defined_bbox_ex_bounds=False, 
                   bbox_ex_bounds_fid=None, use_numba=False, _char_fx_version_=2,
                   get_gbsegs=True, save_gbsegs=False, dtype_gbseg=np.int32,
                   throw=False, throw_gbsegs=False):
    """
    Find the fids of neighbours for a given fid and optionally the grain boundary segments.

    This function identifies neighboring grains for a specified grain ID. It can also
    store the neighbor IDs in the grain object and optionally retrieve grain boundary segments.

    Parameters
    ----------
    gdict : dict
        Dictionary containing grain objects indexed by their IDs.
    fid : int
        The grain ID for which to find neighbors.
    lfi : np.ndarray
        2D array representing the labeled feature image (grain structure).
    nfeatures : int
        Total number of features (grains) in the grain structure.
    include_central_grain : bool, optional
        Whether to include the central grain in the neighbor list. Default is False.
    update_grain_object : bool, optional
        Whether to store the neighbor IDs in the grain object. Default is True.
    user_defined_bbox_ex_bounds : bool, optional
        Whether to use user-defined bounding box extents. Default is False.
    bbox_ex_bounds_fid : tuple, optional
        User-defined bounding box extents for the grain. Required if
        user_defined_bbox_ex_bounds is True. Default is None.
    use_numba : bool, optional
        Whether to use Numba-optimized neighbor mask calculation. Default is False.
    _char_fx_version_ : int, optional
        Characterization function version (1 or 2). Default is 2.
    get_gbsegs : bool, optional
        Whether to get grain boundary segments. Default is True.
    save_gbsegs : bool, optional
        Whether to save grain boundary segments in the grain object. Default is False.
    dtype_gbseg : data-type, optional
        Data type for grain boundary segments. Default is np.int32.
    throw : bool, optional
        Whether to return the neighbor IDs. Default is False.
    throw_gbsegs : bool, optional
        Whether to return the grain boundary segments. Default is False.

    Returns
    -------
    neighbour_ids : np.ndarray
        Array of neighboring grain IDs. Returned if throw is True.
    gbsegs_pre : np.ndarray
        Array marking grain boundary segments. Returned if get_gbseg and throw_gbsegs are True.
        If both throw and throw_gbsegs are True, returns a tuple (neighbour_ids, gbsegs_pre).
        Users are responsible for handling the returned values accordingly.

    Import
    ------
    import upxo.connops.neighbour_ops as neighbour_ops
    Use as: neighbour_ops.find_neigh_gid(...)

    Notes
    -----
    - The function can operate in two modes: returning neighbor IDs directly or updating
      the grain object with neighbor information.
    - Grain boundary segments can be stored in a sparse format within the grain object
        for memory efficiency.

    Examples
    --------
    ../src/demos/neighOps/neighOps-1.ipynb
    """
    # Get the bounds of the grain gid
    if user_defined_bbox_ex_bounds:
        bounds = bbox_ex_bounds_fid
    else:
        if _char_fx_version_ == 1:
            bounds = gdict[fid]['grain'].bbox_ex_bounds
        elif _char_fx_version_ == 2:
            bounds = gdict[fid].bbox_ex_bounds
    # ---------------------------------------------
    # get the sub-array corresponding to the bounding box
    probable_grains_locs = lfi[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1]
    neigh_mask = deepcopy(probable_grains_locs)
    # ---------------------------------------------
    # get the neighbour grain mask
    if nfeatures > 2000 or use_numba:
        neigh_mask = NMB_get_neighbor_mask(neigh_mask, fid)
    else:
        # self.get_neighbor_mask(temp, fid)
        for row in range(neigh_mask.shape[0]):
            for col in range(neigh_mask.shape[1]):
                if neigh_mask[row, col] == fid:
                    if row-1 >= 0:
                        if neigh_mask[row-1, col] != fid:
                            neigh_mask[row-1, col] = -1
                    if row+1 < neigh_mask.shape[0]:
                        if neigh_mask[row+1, col] != fid:
                            neigh_mask[row+1, col] = -1
                    if col-1 >= 0:
                        if neigh_mask[row, col-1] != fid:
                            neigh_mask[row, col-1] = -1
                    if col+1 < neigh_mask.shape[1]:
                        if neigh_mask[row, col+1] != fid:
                            neigh_mask[row, col+1] = -1
                """ if values in probable_grains_locs not
                    equal to -1, then replace them with 0 """
                if neigh_mask[row, col] != -1:
                    neigh_mask[row, col] = 0
    # ---------------------------------------------
    """ Find out the gids of the neighbouring grains """
    neigh_pixel_locs = np.argwhere(neigh_mask == -1)
    neigh_pixel_grain_ids = probable_grains_locs[neigh_pixel_locs[:, 0], neigh_pixel_locs[:, 1]]
    # ---------------------------------------------
    if include_central_grain:
        neighbour_ids = np.unique(np.append(neigh_pixel_grain_ids, fid))
    else:
        neighbour_ids = np.unique(neigh_pixel_grain_ids[neigh_pixel_grain_ids != fid])
    # ---------------------------------------------
    if update_grain_object:
        """ Store the neighbnour_ids inside the grain object """
        if _char_fx_version_ == 1:
            gdict[fid]['grain'].neigh = np.asarray(neighbour_ids, dtype=np.int32)
        elif _char_fx_version_ == 2:
            gdict[fid].neigh = np.asarray(neighbour_ids, dtype=np.int32)
    # ---------------------------------------------
    if get_gbsegs:
        '''Mark the locations of grain boundaries which woulsd be individual
        segments. Each segnment marks the grain boundary interface of the
        'gid' grain with its neighbouring grains. '''
        gbsegs = np.zeros_like(neigh_mask)
        for ni in neighbour_ids:
            gbsegs[np.logical_and(neigh_mask == -1, probable_grains_locs == ni)] = ni
        # --------------
        if save_gbsegs:
            '''Store the grain boundary segment locations inside the grain
            object's data structure.
            NOTE: gbsegs will allways be stored in sparse format inside the grain
            object. So, if you want to retrieve the full gbsegs array, you will have to
            reconstruct it using the sparse data, or use the `lfi_gbsegs` property definition 
            of the grain object.'''
            gbseg_data = {}
            gbsegs_raveled = gbsegs.ravel()
            initShape = gbsegs.shape  # Initial shape
            NZI = np.where(gbsegs_raveled)[0]  # None-Zero indices
            NZV = gbsegs_raveled[NZI]  # None-Zero values
            gbseg_data['shape'] = initShape
            gbseg_data['NZI'] = NZI
            gbseg_data['NZV'] = NZV
            gbseg_data['dtype'] = dtype_gbseg
            if _char_fx_version_ == 1:
                gdict[fid]['grain'].gbsegs = gbseg_data
            elif _char_fx_version_ == 2:
                gdict[fid].gbsegs = gbseg_data
    # ---------------------------------------------
    # Dual behaviour to adpat to too many existing implementations
    if throw:
        neigh_gid__gbsegs = neighbour_ids
    else:
        neigh_gid__gbsegs = None
    if get_gbsegs and throw_gbsegs:
        neigh_gid__gbsegs = (neighbour_ids, gbsegs)
    
    return neigh_gid__gbsegs

def find_neigh(lfi, fids, gdict, nfeatures, include_central_grain=False, 
               char_fx_version=2, print_msg=True,
               user_defined_bbox_ex_bounds=False, bbox_ex_bounds=None, 
               update_grain_object=True, use_numba=False):
    """
    Find the gids of neighbours for every gid.

    Parameters
    ----------
    include_central_grain : bool, optional
        Whether to include the central grain in the neighbor list. Default is False.

    Examples
    --------
    from upxo.ggrowth.mcgs import mcgs
    pxtal = mcgs(study='independent', input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    tslice = 10
    pxtal.gs[tslice].char_morph_2d(char_gb=True)

    pxtal.gs[tslice].find_neigh(include_central_grain=True)
    pxtal.gs[tslice].neigh_gid[10]

    pxtal.gs[tslice].find_neigh(include_central_grain=False)
    pxtal.gs[tslice].neigh_gid[10]
    """
    neigh_gid = {}
    # ---------------------------------------------------------------------
    if print_msg:
        print('\nExtracting neigh list for all grains\n')
    for fid in fids:
        bbox_ex_bounds_fid = bbox_ex_bounds[fid] if user_defined_bbox_ex_bounds else None
        neigh_gid[fid] = [int(ngid) for ngid in find_neigh_fid(gdict, lfi, fid, nfeatures,
                            include_central_grain=include_central_grain, 
                            update_grain_object=update_grain_object,
                            _char_fx_version_=char_fx_version,
                            get_gbsegs=True, save_gbsegs=False, dtype_gbseg=np.int32,
                            throw=True, throw_gbsegs=False,
                            user_defined_bbox_ex_bounds=user_defined_bbox_ex_bounds,
                            bbox_ex_bounds_fid=bbox_ex_bounds_fid,
                            use_numba=use_numba)]
        
    return neigh_gid

def get_upto_nth_order_neighbors(lfi, neigh_fid, grain_id, neigh_order,
                                 fast_estimate=False,
                                 recalculate=False, include_parent=True,
                                 output_type='list'):
    """
    Calculates the nth order neighbours for a given gid.

    Parameters
    ----------
    grain_id : int
        The ID of the grain for which to find neighbors.
    neigh_order : int
        The order of neighbors to calculate (1st order, 2nd order, etc.).
    fast_estimate : bool, optional
        Whether to use a fast estimation method. Default is False.
    recalculate : bool, optional
        Whether to recalculate neighbors even if they are already computed. Default is False.
    include_parent : bool, optional
        Whether to include the parent grain ID in the neighbors list. Default is True.
    output_type : str, optional
        The type of output: 'list', 'nparray', or 'set'. Default is 'list'.

    Returns
    -------
    list, np.ndarray, or set
        Neighbors of the specified order.

    Internal use examples
    ---------------------
    from upxo.ggrowth.mcgs import mcgs
    # pxtal = mcgs(study='independent', input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
    pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    tslice = 18
    pxtal.gs[tslice].char_morph_2d(char_gb=True)
    neigh_order = 3
    gid = 6
    neighbours = pxtal.gs[tslice].get_upto_nth_order_neighbors(gid, neigh_order,
                                                    fast_estimate=False,
                                                    recalculate=True,
                                                    include_parent=True,
                                                    output_type='list')
    pxtal.gs[tslice].neigh_gid[gid]
    pxtal.gs[tslice].plot_grains_gids(pxtal.gs[tslice].gid, gclr='color', title='')
    pxtal.gs[tslice].plot_grains_gids([gid], gclr='color', title='parent gid: '+str(gid))
    pxtal.gs[tslice].plot_grains_gids(neighbours, gclr='color', cmap_name='nipy_spectral')

    External use examples
    ---------------------
    from upxo.connops.neighbour_ops import get_upto_nth_order_neighbors
    """
    # ---------------------------------------------------------------------
    if neigh_order == 0:
        return grain_id
    # ---------------------------------------------------------------------
    if recalculate or neigh_fid is None or len(neigh_fid) == 0:
        if fast_estimate:
            self.find_neigh_gid_fast_all_grains(include_parent=include_parent)
        else:
            self.find_neigh(include_central_grain=include_parent)
    # ---------------------------------------------------------------------
    # print(self.neigh_gid[grain_id])
    # ---------------------------------------------------------------------
    # Start with 1st-order neighbors
    neighbors = set(neigh_fid.get(grain_id, []))
    for _ in range(neigh_order - 1):
        new_neighbors = set()
        for neighbor in neighbors:
            new_neighbors.update(neigh_fid.get(neighbor, []))
        neighbors.update(new_neighbors)
    # ---------------------------------------------------------------------
    if not include_parent:
        neighbors.discard(grain_id)
    if output_type == 'list':
        return list(neighbors)
    if output_type in ('np', 'nparray', 'np.array', 'numpy'):
        return np.array(list(neighbors))
    elif output_type == 'set':
        return neighbors
    
def get_nth_order_neighbors(self, grain_id, neigh_order, fast_estimate=False,
                            recalculate=False, include_parent=True):
    """
    Calculates the 1st till nth order neighbours for a given gid.

    Parameters
    ----------
    grain_id : int
        The ID of the grain for which to find neighbors.
    neigh_order : int
        The order of neighbors to calculate (1st order, 2nd order, etc.).
    fast_estimate : bool, optional
        Whether to use a fast estimation method. Default is False.
    recalculate : bool, optional
        Whether to recalculate neighbors even if they are already computed. Default is False.
    include_parent : bool, optional
        Whether to include the parent grain ID in the neighbors list. Default is True.

    Returns
    -------
    list
        A list containing the nth order neighbors.
    
    Example
    -------
    from upxo.ggrowth.mcgs import mcgs
    pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    gid = 10
    tslice = 16
    neigh_order = 6
    neighbours = pxtal.gs[tslice].get_nth_order_neighbors(gid, neigh_order,
                                            fast_estimate=False,
                                            recalculate=True,
                                            include_parent=True)
    pxtal.gs[tslice].plot_grains_gids(neighbours, gclr='color')
    """
    neigh_upto_n_minus_1 = self.get_upto_nth_order_neighbors(grain_id, neigh_order-1,
                                                                fast_estimate=fast_estimate,
                                                                include_parent=include_parent,
                                                                output_type='set')
    if type(neigh_upto_n_minus_1) in dth.dt.NUMBERS:
        neigh_upto_n_minus_1 = set([neigh_upto_n_minus_1])

    neigh_upto_n = self.get_upto_nth_order_neighbors(grain_id, neigh_order,
                                                        fast_estimate=fast_estimate,
                                                        include_parent=include_parent,
                                                        output_type='set')
    if type(neigh_upto_n) in dth.dt.NUMBERS:
        neigh_upto_n = set([neigh_upto_n])
    return list(neigh_upto_n.difference(neigh_upto_n_minus_1))

def get_upto_nth_order_neighbors_all_grains(self, neigh_order,
                                            recalculate=False, fast_estimate=False,
                                            include_parent=True, output_type='list'):
    """
    Calculates 1st to nth order neighbors of all gids.

    Parameters
    ----------
    neigh_order : int
        The order of neighbors to calculate (1st order, 2nd order, etc.).
    recalculate : bool, optional
        Whether to recalculate neighbors even if they are already computed. Default is False.
    fast_estimate : bool, optional
        Whether to use a fast estimation method. Default is False.
    include_parent : bool, optional
        Whether to include the parent grain ID in the neighbors list. Default is True.
    output_type : str, optional
        The type of output: 'list', 'nparray', or 'set'. Default is 'list'.

    Returns
    -------
    dict
        Dictionary with grain IDs as keys and their neighbors of specified order as values.

    Example
    -------
    from upxo.ggrowth.mcgs import mcgs
    pxtal = mcgs(study='independent',
                    input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    neigh_order = 1
    pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,
                                                            output_type='list')
    """
    neighs_upto_nth_order = {gid: self.get_upto_nth_order_neighbors(gid, neigh_order,
                                                                    fast_estimate=fast_estimate,
                                                                    recalculate=recalculate,
                                                                    include_parent=include_parent,
                                                                    output_type='list')
                                for gid in self.gid}
    return neighs_upto_nth_order

def get_nth_order_neighbors_all_grains(self, neigh_order, fast_estimate=False,
                                        recalculate=False, include_parent=True):
    """
    Calculates the nth order neighbours of all gids.

    Parameters
    ----------
    neigh_order : int
        The order of neighbors to calculate (1st order, 2nd order, etc.).
    fast_estimate : bool, optional
        Whether to use a fast estimation method. Default is False.
    recalculate : bool, optional
        Whether to recalculate neighbors even if they are already computed. Default is False.
    include_parent : bool, optional
        Whether to include the parent grain ID in the neighbors list. Default is True.

    Returns
    -------
    dict
        Dictionary with grain IDs as keys and their neighbors of specified order as values.

    Example
    -------
    from upxo.ggrowth.mcgs import mcgs
    pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    tslice = 99
    pxtal.gs[tslice].char_morph_2d(char_gb=True)

    no_clr = ['k', 'b', 'r', 'g']
    no_mrk = ['o', 's', 'x', '+']
    no_msz = [8, 8, 8, 8]
    gb_clr = ['k', 'b', 'r', 'g']

    cg = 1
    NO = [2]
    ANO = [None, None, None]
    plt.figure(figsize=(5, 5), dpi=100)
    for no in NO:
        A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(no,
                                                                fast_estimate=False,
                                                                recalculate=False,
                                                                include_parent=True,)
        for gid in A[cg]:
            plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                        gb_clr[no]+'.', markersize=2)
            gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
            plt.plot(*gidcentroid, no_clr[no]+no_mrk[no],
                        markersize=no_msz[no])
    plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                'c.', markersize=4)
    cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
    plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                markersize=no_msz[0])
    plt.gca().set_aspect('equal')



    cg = 10  # central_grain
    neigh_order = 1
    A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,
                                                            output_type='list')
    A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,)
    pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                        cmap_name='CMRmap_r', )

    neigh_order = 2
    A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,
                                                            output_type='list')
    A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,)
    pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                        cmap_name='CMRmap_r', )

    neigh_order = 3
    A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,
                                                            output_type='list')
    A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True,)
    pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                        cmap_name='CMRmap_r', )



    # -----------------------------------------
    no_clr = ['k', 'b', 'k', 'k']
    no_mrk = ['o', 's', 'x', '+']
    no_msz = [8, 8, 8, 8]

    plt.figure(figsize=(5, 5), dpi=100)
    for gid in A[cg]:
        plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                    'k.', markersize=3)
        gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
        plt.plot(*gidcentroid, no_clr[neigh_order]+no_mrk[neigh_order],
                    markersize=no_msz[neigh_order])

    plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                'r.', markersize=4)
    cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
    plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                markersize=no_msz[0])
    plt.gca().set_aspect('equal')
    # =================================================================
    all_neighs = pxtal.gs[tslice].neigh_gid
    neigh_order = 1
    cg = 17
    plt.figure(figsize=(5, 5), dpi=100)
    for gid in all_neighs[cg]:
        plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                    'k.', markersize=3)
        gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
        plt.plot(*gidcentroid, no_clr[neigh_order]+no_mrk[neigh_order],
                    markersize=no_msz[neigh_order])

    plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                'r.', markersize=4)
    cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
    plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                markersize=no_msz[0])
    plt.gca().set_aspect('equal')
    """
    neighs_nth_order = {gid: self.get_nth_order_neighbors(gid, neigh_order,
                                                            fast_estimate=fast_estimate,
                                                            recalculate=recalculate,
                                                            include_parent=include_parent)
                        for gid in self.gid}
    return neighs_nth_order

def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                    recalculate=False,
                                                    include_parent=False,
                                                    print_msg=False,
                                                    _int_approx_=0.05):
    """
    Calculates 1st to nth order neighbors of all gids.
    Allows float values for neigh_order for probabilistic selection.

    Parameters
    ----------
    neigh_order : int or float
        The order of neighbors to calculate (1st order, 2nd order, etc.).
        If float, probabilistic selection is done between floor and ceil values.
    recalculate : bool, optional
        Whether to recalculate neighbors even if they are already computed. Default is False.
    include_parent : bool, optional
        Whether to include the parent grain ID in the neighbors list. Default is False.
    print_msg : bool, optional
        Whether to print messages during execution. Default is False.
    _int_approx_ : float, optional
        Threshold to consider a float as an integer. Default is 0.05.

    Returns
    -------
    dict
        Dictionary with grain IDs as keys and their neighbors of specified order as values.

    Example
    -------
    from upxo.ggrowth.mcgs import mcgs
    pxt = mcgs()
    pxt.simulate()
    pxt.detect_grains()
    tslice = 10
    def_neigh = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

    neigh0 = def_neigh(1, recalculate=False, include_parent=True)
    neigh1 = def_neigh(1.06, recalculate=False, include_parent=True)
    neigh2 = def_neigh(1.5, recalculate=False, include_parent=True)
    neigh0[22]
    neigh1[2][22]
    neigh2[2][22]
    """
    # @dev:
        # no: neighbour order in these definitions.
    no = neigh_order
    on_neigh_all_grains_upto = self.get_upto_nth_order_neighbors_all_grains
    on_neigh_all_grains_at = self.get_nth_order_neighbors_all_grains
    if isinstance(no, (int, np.int32)):
        if print_msg:
            print('neigh_order is of type int. Adopting the usual method.')
        neigh_on = on_neigh_all_grains_upto(no, recalculate=recalculate,
                                        include_parent=include_parent)
        return neigh_on
    elif isinstance(no, (float, np.float64)):
        if abs(no-round(no)) < _int_approx_:
            if print_msg:
                print('neigh_order is close to being int. Adopting usual method.')
            neigh_on = on_neigh_all_grains_upto(math.floor(no),
                                                recalculate=recalculate,
                                                include_parent=include_parent)
            return neigh_on
        else:
            if print_msg:
                # Nothing to print
                pass
            no_low, no_high = math.floor(no), math.ceil(no)
            neigh_upto_low = on_neigh_all_grains_upto(no_low,
                                                        recalculate=recalculate,
                                                        include_parent=include_parent)
            neigh_at_high = on_neigh_all_grains_at(no_low+1,
                                                    recalculate=recalculate,
                                                    include_parent=False)
            delno = np.round(abs(neigh_order-math.floor(neigh_order)), 4)
            neighbours = {}
            for gid in self.gid:
                nselect = math.ceil(delno * len(neigh_at_high[gid]))
                if len(neigh_at_high[gid]) > 1:
                    neighbours[gid] = neigh_upto_low[gid] + random.sample(neigh_at_high[gid],
                                                                            nselect)
            return neighbours
    else:
        raise ValueError('Invalid neigh_order')