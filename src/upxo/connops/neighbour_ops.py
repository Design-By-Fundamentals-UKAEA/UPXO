"""
Module: neighbour_ops
---------------------
Neighbour (connectivity) operations for 2D grain structures.

Identifies neighbouring grain IDs and grain-boundary pixel segments for
individual grains or across an entire grain structure.  Provides first- through
nth-order neighbourhood queries with optional Numba acceleration for large
datasets.

Usage
-----
::

    import upxo.connops.neighbour_ops as neighOps
"""
import numpy as np
from numba import njit, prange
from copy import deepcopy


@njit(parallel=True)
def NMB_get_neighbor_mask(neigh_mask, gid):
    """Mark the boundary pixels of a single grain in a 2D label sub-array.

    JIT-compiled with Numba (parallel outer loop via ``prange``).  For every
    pixel that belongs to ``gid``, the four 4-connected neighbours that do
    *not* belong to ``gid`` are marked ``-1``.  All remaining pixels are set
    to ``0``.

    Parameters
    ----------
    neigh_mask : numpy.ndarray of int, shape (R, C)
        Sub-array of the full labelled image covering the extended bounding box
        of the grain.  Modified in-place and returned.
    gid : int
        Label value of the grain whose boundary-adjacent pixels are to be
        identified.

    Returns
    -------
    neigh_mask : numpy.ndarray of int, shape (R, C)
        Same array, now containing ``-1`` at pixels that are direct
        4-connected neighbours of ``gid`` but belong to a different grain, and
        ``0`` everywhere else.

    Notes
    -----
    The outer row loop is parallelised with ``prange``; the inner column loop
    is sequential.  Bounds checks are explicit — no zero-padding of the input
    array is required.
    """
    nrows, ncols = neigh_mask.shape
    for row in prange(nrows):
        for col in range(ncols):
            if neigh_mask[row, col] == gid:
                if row - 1 >= 0:
                    if neigh_mask[row-1, col] != gid:
                        neigh_mask[row-1, col] = -1
                if row + 1 < nrows:
                    if neigh_mask[row+1, col] != gid:
                        neigh_mask[row+1, col] = -1
                if col - 1 >= 0:
                    if neigh_mask[row, col-1] != gid:
                        neigh_mask[row, col-1] = -1
                if col + 1 < ncols:
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
    """Find the neighbour IDs of a single grain and optionally its boundary segments.

    Extracts the extended bounding-box sub-array for ``fid``, constructs a
    neighbour mask (via Numba or pure Python depending on ``nfeatures``), and
    returns the unique IDs of all directly touching grains.  Grain-boundary
    segment maps can be generated and stored in the grain object's sparse
    format.

    Parameters
    ----------
    gdict : dict
        Grain-object dictionary keyed by grain ID.  Each value is either a
        grain object directly (``_char_fx_version_=2``) or a dict with a
        ``'grain'`` key (``_char_fx_version_=1``).
    lfi : numpy.ndarray of int, shape (R, C)
        Full labelled feature image for the grain structure.
    fid : int
        Grain ID for which neighbours are to be found.
    nfeatures : int
        Total number of grains in the structure.  Governs the Numba/pure-Python
        branch threshold (Numba is used when ``nfeatures > 2000`` or
        ``use_numba`` is True).
    include_central_grain : bool, default False
        Include ``fid`` itself in the returned neighbour array.
    update_grain_object : bool, default True
        Write the neighbour ID array into the grain object as ``grain.neigh``.
    user_defined_bbox_ex_bounds : bool, default False
        Use ``bbox_ex_bounds_fid`` instead of the bounding-box stored in the
        grain object.
    bbox_ex_bounds_fid : tuple of int or None, default None
        ``(rmin, rmax, cmin, cmax)`` bounding-box extents to use when
        ``user_defined_bbox_ex_bounds`` is True.
    use_numba : bool, default False
        Force use of :func:`NMB_get_neighbor_mask` regardless of ``nfeatures``.
    _char_fx_version_ : {1, 2}, default 2
        Selects how grain objects are accessed from ``gdict``.  Version 2
        accesses grain attributes directly; version 1 assumes a nested dict
        with a ``'grain'`` key.
    get_gbsegs : bool, default True
        Compute the grain-boundary segment map (a 2D array where each
        non-zero pixel contains the neighbour grain's ID at that boundary
        location).
    save_gbsegs : bool, default False
        Serialise the boundary segment map into the grain object in sparse
        format (``shape``, ``NZI``, ``NZV``, ``dtype`` keys).
    dtype_gbseg : numpy dtype, default numpy.int32
        Data type recorded in the sparse ``dtype`` field when saving segments.
    throw : bool, default False
        Return ``neighbour_ids`` (or a tuple) instead of ``None``.
    throw_gbsegs : bool, default False
        When both ``get_gbsegs`` and ``throw_gbsegs`` are True, return
        ``(neighbour_ids, gbsegs)`` instead of ``neighbour_ids`` alone.

    Returns
    -------
    None
        When ``throw`` is False.
    neighbour_ids : numpy.ndarray of int
        Unique IDs of all grains directly touching ``fid``.  Returned when
        ``throw`` is True and ``throw_gbsegs`` is False.
    (neighbour_ids, gbsegs) : tuple
        Tuple of the neighbour ID array and the 2D boundary segment map.
        Returned when both ``throw`` and ``throw_gbsegs`` are True.

    Notes
    -----
    Grain-boundary segments are always stored in the grain object in sparse
    format (non-zero indices and values).  To reconstruct the dense array use
    the ``lfi_gbsegs`` property of the grain object, or rebuild manually from
    the ``shape``, ``NZI``, and ``NZV`` fields.

    Examples
    --------
    Typical usage through the grain-structure object (which calls this
    function internally):

    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 10
        pxtal.gs[tslice].char_morph_2d(char_gb=True)
        pxtal.gs[tslice].find_neigh(include_central_grain=False)
        print(pxtal.gs[tslice].neigh_gid[10])

    For direct use at module level:

    .. code-block:: python

        import upxo.connops.neighbour_ops as neighOps
        neigh_ids = neighOps.find_neigh_fid(
            gdict, lfi, fid=5, nfeatures=200,
            throw=True, throw_gbsegs=False)
    """
    if user_defined_bbox_ex_bounds:
        bounds = bbox_ex_bounds_fid
    else:
        if _char_fx_version_ == 1:
            bounds = gdict[fid]['grain'].bbox_ex_bounds
        elif _char_fx_version_ == 2:
            bounds = gdict[fid].bbox_ex_bounds
    probable_grains_locs = lfi[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1]
    neigh_mask = deepcopy(probable_grains_locs)
    if nfeatures > 2000 or use_numba:
        neigh_mask = NMB_get_neighbor_mask(neigh_mask, fid)
    else:
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
                if neigh_mask[row, col] != -1:
                    neigh_mask[row, col] = 0
    neigh_pixel_locs = np.argwhere(neigh_mask == -1)
    neigh_pixel_grain_ids = probable_grains_locs[neigh_pixel_locs[:, 0], neigh_pixel_locs[:, 1]]
    if include_central_grain:
        neighbour_ids = np.unique(np.append(neigh_pixel_grain_ids, fid))
    else:
        neighbour_ids = np.unique(neigh_pixel_grain_ids[neigh_pixel_grain_ids != fid])
    if update_grain_object:
        if _char_fx_version_ == 1:
            gdict[fid]['grain'].neigh = np.asarray(neighbour_ids, dtype=np.int32)
        elif _char_fx_version_ == 2:
            gdict[fid].neigh = np.asarray(neighbour_ids, dtype=np.int32)
    if get_gbsegs:
        gbsegs = np.zeros_like(neigh_mask)
        for ni in neighbour_ids:
            gbsegs[np.logical_and(neigh_mask == -1, probable_grains_locs == ni)] = ni
        if save_gbsegs:
            gbseg_data = {}
            gbsegs_raveled = gbsegs.ravel()
            initShape = gbsegs.shape
            NZI = np.where(gbsegs_raveled)[0]
            NZV = gbsegs_raveled[NZI]
            gbseg_data['shape'] = initShape
            gbseg_data['NZI'] = NZI
            gbseg_data['NZV'] = NZV
            gbseg_data['dtype'] = dtype_gbseg
            if _char_fx_version_ == 1:
                gdict[fid]['grain'].gbsegs = gbseg_data
            elif _char_fx_version_ == 2:
                gdict[fid].gbsegs = gbseg_data
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
    """Find the neighbour IDs for every grain in the structure.

    Iterates over all grain IDs in ``fids``, calls :func:`find_neigh_fid` for
    each, and collects the results into a single dictionary.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Full labelled feature image for the grain structure.
    fids : iterable of int
        Grain IDs to process.  Typically ``range(1, nfeatures + 1)``.
    gdict : dict
        Grain-object dictionary keyed by grain ID (see :func:`find_neigh_fid`).
    nfeatures : int
        Total number of grains; governs the Numba/pure-Python branch in
        :func:`find_neigh_fid`.
    include_central_grain : bool, default False
        Include each grain's own ID in its neighbour list.
    char_fx_version : {1, 2}, default 2
        Grain-object access version forwarded to :func:`find_neigh_fid`.
    print_msg : bool, default True
        Print a progress message before the loop begins.
    user_defined_bbox_ex_bounds : bool, default False
        Use bounding-box extents from ``bbox_ex_bounds`` instead of those
        stored in the grain objects.
    bbox_ex_bounds : dict or None, default None
        Mapping of grain ID → ``(rmin, rmax, cmin, cmax)`` when
        ``user_defined_bbox_ex_bounds`` is True.
    update_grain_object : bool, default True
        Write each grain's neighbour array into its grain object.
    use_numba : bool, default False
        Force Numba acceleration regardless of ``nfeatures``.

    Returns
    -------
    neigh_gid : dict[int, list of int]
        Mapping of grain ID → list of neighbour grain IDs.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 10
        pxtal.gs[tslice].char_morph_2d(char_gb=True)

        # With central grain included
        pxtal.gs[tslice].find_neigh(include_central_grain=True)
        print(pxtal.gs[tslice].neigh_gid[10])

        # Without central grain
        pxtal.gs[tslice].find_neigh(include_central_grain=False)
        print(pxtal.gs[tslice].neigh_gid[10])
    """
    neigh_gid = {}
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
    """Return all grains reachable within ``neigh_order`` hops from ``grain_id``.

    Starting from the 1st-order neighbours of ``grain_id``, the function
    iteratively expands the set by one hop per order until ``neigh_order``
    hops are exhausted.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Full labelled feature image.  Used only when ``recalculate`` is True
        to trigger neighbour recomputation via ``self.find_neigh``.
    neigh_fid : dict[int, list of int]
        Pre-computed neighbour map returned by :func:`find_neigh`.  Pass an
        empty dict or ``None`` to force recomputation via ``recalculate=True``.
    grain_id : int
        ID of the central grain.
    neigh_order : int
        Maximum hop count.  ``neigh_order=1`` returns direct neighbours;
        ``neigh_order=2`` returns neighbours-of-neighbours, and so on.
        Passing ``0`` returns ``grain_id`` itself.
    fast_estimate : bool, default False
        Use the fast neighbour estimation method when recomputing (only
        relevant if ``recalculate`` is True).
    recalculate : bool, default False
        Recompute the neighbour map even if ``neigh_fid`` is already populated.
    include_parent : bool, default True
        Include ``grain_id`` in the returned set.
    output_type : {'list', 'nparray', 'set'}, default 'list'
        Return type.  ``'nparray'`` accepts several aliases:
        ``'np'``, ``'np.array'``, ``'numpy'``.

    Returns
    -------
    list, numpy.ndarray, or set
        All grain IDs within ``neigh_order`` hops of ``grain_id``, in the
        format requested by ``output_type``.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 18
        pxtal.gs[tslice].char_morph_2d(char_gb=True)

        neighbours = pxtal.gs[tslice].get_upto_nth_order_neighbors(
            grain_id=6, neigh_order=3,
            fast_estimate=False, recalculate=True,
            include_parent=True, output_type='list')

        pxtal.gs[tslice].plot_grains_gids(
            neighbours, gclr='color', cmap_name='nipy_spectral')

    See Also
    --------
    get_nth_order_neighbors : Returns only the grains *at* exactly order *n*
        (i.e. reachable in exactly ``n`` hops but not fewer).
    """
    if neigh_order == 0:
        return grain_id
    if recalculate or neigh_fid is None or len(neigh_fid) == 0:
        if fast_estimate:
            self.find_neigh_gid_fast_all_grains(include_parent=include_parent)
        else:
            self.find_neigh(include_central_grain=include_parent)
    neighbors = set(neigh_fid.get(grain_id, []))
    for _ in range(neigh_order - 1):
        new_neighbors = set()
        for neighbor in neighbors:
            new_neighbors.update(neigh_fid.get(neighbor, []))
        neighbors.update(new_neighbors)
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
    """Return the grains reachable in *exactly* ``neigh_order`` hops.

    Computes the set difference between the up-to-*n* and up-to-*(n-1)*
    neighbourhood sets to isolate grains that first become reachable at
    exactly the requested order.

    Parameters
    ----------
    grain_id : int
        ID of the central grain.
    neigh_order : int
        Exact hop count.  Grains reachable in fewer hops are excluded.
    fast_estimate : bool, default False
        Use fast neighbour estimation when recomputing neighbours.
    recalculate : bool, default False
        Force recomputation of the neighbour map.
    include_parent : bool, default True
        Include ``grain_id`` in the candidate set before differencing.

    Returns
    -------
    list of int
        Grain IDs reachable in exactly ``neigh_order`` hops from
        ``grain_id``.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 16

        neighbours = pxtal.gs[tslice].get_nth_order_neighbors(
            grain_id=10, neigh_order=6,
            fast_estimate=False, recalculate=True, include_parent=True)

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
    """Return up-to-nth-order neighbours for every grain in the structure.

    Calls :func:`get_upto_nth_order_neighbors` for each grain ID in
    ``self.gid`` and collects the results into a dictionary.

    Parameters
    ----------
    neigh_order : int
        Maximum hop count passed to :func:`get_upto_nth_order_neighbors`.
    recalculate : bool, default False
        Force recomputation of the neighbour map for each grain.
    fast_estimate : bool, default False
        Use fast neighbour estimation when recomputing.
    include_parent : bool, default True
        Include each grain's own ID in its neighbour list.
    output_type : {'list', 'nparray', 'set'}, default 'list'
        Return type for each grain's neighbour collection.

    Returns
    -------
    dict[int, list or numpy.ndarray or set]
        Mapping of grain ID → neighbours up to order ``neigh_order``.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()

        neighs = pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(
            neigh_order=1, recalculate=False,
            include_parent=True, output_type='list')
        print(neighs[5])  # neighbours of grain 5 up to order 1
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
    """Return the exactly-nth-order neighbours for every grain in the structure.

    Calls :func:`get_nth_order_neighbors` for each grain ID in ``self.gid``
    and collects results into a dictionary.  Only grains first reachable at
    exactly ``neigh_order`` hops are included; grains reachable in fewer hops
    are excluded.

    Parameters
    ----------
    neigh_order : int
        Exact hop count.
    fast_estimate : bool, default False
        Use fast neighbour estimation when recomputing.
    recalculate : bool, default False
        Force recomputation of the neighbour map.
    include_parent : bool, default True
        Include each grain's own ID in the candidate set before differencing.

    Returns
    -------
    dict[int, list of int]
        Mapping of grain ID → list of grains reachable in exactly
        ``neigh_order`` hops.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        import matplotlib.pyplot as plt
        import numpy as np

        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 99
        pxtal.gs[tslice].char_morph_2d(char_gb=True)

        cg = 10           # central grain ID
        neigh_order = 2

        A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(
            neigh_order, fast_estimate=False,
            recalculate=False, include_parent=True)

        pxtal.gs[tslice].plot_grains_gids(
            A[cg], gclr='color', title='2nd-order neighbours of grain 10',
            cmap_name='CMRmap_r')
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
    """Return up-to-nth-order neighbours for all grains with probabilistic fractional orders.

    Extends :func:`get_upto_nth_order_neighbors_all_grains` to accept
    non-integer ``neigh_order`` values.  A fractional order such as ``1.5``
    returns all grains up to order 1 plus a random subset of order-2 grains,
    where the fraction included equals the decimal part of ``neigh_order``.

    Parameters
    ----------
    neigh_order : int or float
        Neighbourhood order.

        * **Integer** (or float within ``_int_approx_`` of an integer):
          delegates directly to :func:`get_upto_nth_order_neighbors_all_grains`.
        * **Non-integer float** (e.g. ``1.5``): returns all up-to-floor
          neighbours plus ``ceil(decimal_part × |order-n neighbours|)``
          randomly chosen grains from the next order shell.
    recalculate : bool, default False
        Force recomputation of the base neighbour map.
    include_parent : bool, default False
        Include each grain's own ID in its neighbour set.
    print_msg : bool, default False
        Print a diagnostic message indicating which branch was taken.
    _int_approx_ : float, default 0.05
        Maximum distance from the nearest integer for ``neigh_order`` to be
        treated as an integer.

    Returns
    -------
    dict[int, list of int]
        When ``neigh_order`` is (effectively) an integer, a mapping of grain
        ID → neighbours up to that integer order.

        When ``neigh_order`` is a true fraction, a mapping of grain ID →
        blended neighbour list (floor-order neighbours + probabilistic
        sample of the next order shell).

    Raises
    ------
    ValueError
        If ``neigh_order`` is neither an int nor a float.

    Examples
    --------
    .. code-block:: python

        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 10
        fn = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        neigh_int  = fn(1,    recalculate=False, include_parent=True)
        neigh_near = fn(1.06, recalculate=False, include_parent=True)
        neigh_half = fn(1.5,  recalculate=False, include_parent=True)

        print(neigh_int[22])     # integer-order result for grain 22
        print(neigh_near[22])    # near-integer result (treated as order 1)
        print(neigh_half[22])    # probabilistic blend for grain 22
    """
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
