"""
Module: mcgb2dops
-----------------
Grain-boundary operations for 2D labelled pixel (Monte-Carlo) structures.

Provides routines for detecting, segmenting, and characterising cell
(grain) boundary pixels in a 2D labelled feature image (LFI), along with
utilities for locating and sorting junction points (triple/quadruple points)
and circularly ordering boundary coordinates.

Functions
---------
PL_cell_boundaries
    Pipeline: detect → segment → characterise cell boundary pixels.
PL_cellb_junction_points
    Pipeline: detect → sort → stat-summarise junction points.
detect_cell_boundaries
    Detect grain-boundary pixels using ``skimage.segmentation.find_boundaries``.
pad_lfi
    Pad a labelled feature image with a constant value.
find_common_interface_boundaries
    Detect subpixel common interfaces using the ``'subpixel'`` boundary mode.
segment_cell_boundaries
    Assign each boundary pixel to the grain pair it separates.
see_bsegs_gids
    Plot the boundary-segment pixel map for a single grain.
characterise_boundary_segments
    Count segments and measure segment lengths per grain.
detect_junction_points
    Find pixels shared by two or more boundary segments (junction points).
find_basic_JPO_stats
    Compute min/max/median junction-point order (JPO) tessellation-wide.
sort_junction_points
    Group junction points by order (T-junction, Q-junction, …).
mask_featIDImg_at_coords
    Paint segment IDs onto a copy of the boundary image.
pad_with
    Custom pad function compatible with ``numpy.pad``.
segment_existing_boundaries
    Walk boundary pixels and assign them to grain-pair interface lists.
circular_sort
    Sort boundary coordinates in circular (CW / CCW) order. *(stub)*
circular_sort_method1
    Clockwise/anti-clockwise sort of junction-point coordinates per grain.
method1
    Four-corner sort (TL → TR → BR → BL) adapted from a public reference.
method2
    Angle-based CCW sort centred on the leftmost point.
method3_2d
    Centroid-angle sort with configurable direction and origin.
calculate_junction_order
    Count unique grains in the 3×3 neighbourhood of a junction pixel.

Usage
-----
::

    from upxo.gbops import mcgb2dops as mcgbOps2d

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from collections import defaultdict
import upxo.gsdataops.grid_ops as gridOps


def PL_cell_boundaries(lfi=None, nfeatures=None, neigh_fid=None,
                       connectivity=1, mode='thick', background=0,
                       local_seg_id_nDecPlaces=4, segIDMask_dtype=np.int32):
    """Run the full cell-boundary detection–segmentation–characterisation pipeline.

    Calls :func:`detect_cell_boundaries`, :func:`segment_cell_boundaries`, and
    :func:`characterise_boundary_segments` in sequence and returns their combined
    outputs.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Labelled feature image (grain ID at every pixel).
    nfeatures : int
        Total number of grains in ``lfi``.  Used to choose the pad value
        ``nfeatures + 1`` so the padded border is a unique sentinel grain.
    neigh_fid : dict[int, set[int]]
        Mapping ``{grain_id: {neighbour_ids}}``.
    connectivity : int, optional
        Pixel connectivity for boundary detection (1 = 4-connected,
        2 = 8-connected).  Default is 1.
    mode : str, optional
        Boundary mode passed to ``skimage.segmentation.find_boundaries``.
        Default is ``'thick'``.
    background : int, optional
        Background label value.  Default is 0.
    local_seg_id_nDecPlaces : int, optional
        Decimal places for local segment sub-IDs.  Default is 4.
    segIDMask_dtype : numpy dtype, optional
        dtype of segment ID arrays.  Default is ``np.int32``.

    Returns
    -------
    lfi_boundaries : numpy.ndarray of int, shape (R, C)
        Pixel image where non-zero values are grain IDs of boundary pixels.
    segInfo : dict
        Segmentation result with keys:

        * ``'bsegCoords'`` — ``dict[int, dict[int, ndarray]]``: pixel coords
          per grain pair.
        * ``'local_seg_ids'`` — sub-IDs within each grain.
        * ``'global_seg_ids'`` — globally unique segment IDs per grain.
        * ``'segidList_lcl'`` — flat list of all local IDs.
        * ``'segidList_gbl'`` — flat list of all global IDs.
        * ``'segid_gbl_pfid_map'`` — maps global segment ID to parent grain ID.
    bseg_props : dict
        Boundary-segment properties with keys ``'nsegments'`` and
        ``'segment_lengths'``.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        lfi_boundaries, segInfo, bseg_props = gbops2d.PL_cell_boundaries(
            lfi=pxt.gs[3].lgi,
            nfeatures=pxt.gs[3].n,
            neigh_fid=pxt.gs[3].neigh_gid,
            connectivity=1,
            mode='thick',
            background=0,
            local_seg_id_nDecPlaces=4,
            segIDMask_dtype=np.int32,
        )
    """
    lfi_boundaries = detect_cell_boundaries(lfi=lfi, nfeatures=nfeatures,
                           connectivity=connectivity,
                           mode=mode, background=background)
    segInfo = segment_cell_boundaries(lfi=lfi, lfi_boundaries=lfi_boundaries,
                            neigh_fid=neigh_fid, connectivity=connectivity,
                            local_seg_id_nDecPlaces=local_seg_id_nDecPlaces)
    bseg_props = characterise_boundary_segments(segInfo['bsegCoords'], neigh_fid)
    return lfi_boundaries, segInfo, bseg_props


def PL_cellb_junction_points(bsegCoords, segidList_gbl):
    """Run the full junction-point detection–sort–stat pipeline.

    Calls :func:`detect_junction_points`, :func:`sort_junction_points`, and
    :func:`find_basic_JPO_stats` in sequence.

    Parameters
    ----------
    bsegCoords : dict[int, dict[int, numpy.ndarray]]
        Boundary-segment pixel coordinates produced by
        :func:`segment_cell_boundaries` (``segInfo['bsegCoords']``).
    segidList_gbl : list of int
        Flat list of global segment IDs (``segInfo['segidList_gbl']``).

    Returns
    -------
    junctionPoints : dict[int, numpy.ndarray]
        Grain ID → array of shape (N_junctions, 3) with columns
        ``[row, col, junction_order]``.
    JPSorted : dict[str, dict[int, numpy.ndarray]]
        Junction points grouped by order label (``'jp1'``, ``'jp2'``,
        ``'tjp'``, ``'qjp'``, …), each sub-dict keyed by grain ID.
    JPOStats : dict
        Summary statistics; see :func:`find_basic_JPO_stats` for the full key
        list.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        junctionPoints, JPSorted, JPOStats = gbops2d.PL_cellb_junction_points(
            segInfo['bsegCoords'],
            segInfo['segidList_gbl'],
        )
    """
    junctionPoints = detect_junction_points(bsegCoords)
    JPSorted = sort_junction_points(junctionPoints)
    JPOStats = find_basic_JPO_stats(junctionPoints, segidList_gbl)
    return junctionPoints, JPSorted, JPOStats


def detect_cell_boundaries(lfi=None, nfeatures=None, connectivity=1,
                           mode='thick', background=0):
    """Detect grain-boundary pixels in a labelled feature image.

    Pads ``lfi`` with the sentinel value ``nfeatures + 1`` to ensure the RVE
    outer shell is also detected as a boundary, then calls
    ``skimage.segmentation.find_boundaries`` and multiplies the boolean result
    by the padded image so every boundary pixel retains its original grain ID.
    The padding is stripped before returning.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Labelled feature image.
    nfeatures : int
        Number of grains; the pad sentinel is ``nfeatures + 1``.
    connectivity : int, optional
        Connectivity for ``find_boundaries`` (1 = 4-connected, 2 = 8-connected).
        Default is 1.
    mode : str, optional
        Boundary detection mode (``'thick'``, ``'inner'``, ``'outer'``,
        ``'subpixel'``).  Default is ``'thick'``.
    background : int, optional
        Background label.  Default is 0.

    Returns
    -------
    lfi_boundaries : numpy.ndarray of int, shape (R, C)
        Image where non-zero values are the grain IDs of boundary pixels and
        zero values are interior pixels.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        lfi_boundaries = gbops2d.detect_cell_boundaries(
            lfi=pxt.gs[3].lgi,
            nfeatures=pxt.gs[3].n,
            connectivity=1,
            mode='thick',
            background=0,
        )
    """
    lfi_padded = pad_lfi(lfi=lfi, pad_value=nfeatures+1, _pad_width=1)
    lfi_boundaries = find_boundaries(lfi_padded, connectivity=connectivity,
                            mode=mode, background=background) * lfi_padded
    lfi_boundaries = lfi_boundaries[1:-1, 1:-1]
    return lfi_boundaries


def pad_lfi(lfi=None, pad_value=-100, _pad_width=1):
    """Pad a labelled feature image with a constant sentinel value.

    A thin wrapper around ``numpy.pad`` using a custom pad function
    (:func:`pad_with`) that fills the border ring with ``pad_value``.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Labelled feature image to pad.
    pad_value : int or float, optional
        Constant value written into the padded border.  Choose a value that
        does not appear as a grain ID in ``lfi`` (e.g. ``nfeatures + 1`` or
        ``-100``).  Default is -100.
    _pad_width : int, optional
        Number of pixels to add on each side.  Default is 1; there is normally
        no reason to change this.

    Returns
    -------
    lfi_padded : numpy.ndarray of int, shape (R + 2*_pad_width, C + 2*_pad_width)
        Padded copy of ``lfi``.

    Notes
    -----
    ``_pad_width`` is intentionally fixed at 1 for all internal callers.
    Changing it to values greater than 1 is supported but untested.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        lfi_padded = gbops2d.pad_lfi(lfi=pxt.gs[3].lgi, pad_value=0)
    """
    lfi_padded = np.pad(lfi, _pad_width, pad_with, padder=pad_value)
    return lfi_padded


def find_common_interface_boundaries(lfi=None, nfeatures=None, connectivity=1):
    """Detect subpixel common interfaces between grains.

    Uses the ``'subpixel'`` boundary mode of
    ``skimage.segmentation.find_boundaries`` to locate the precise common
    interface lines between adjacent grains.  Returns a boolean or int mask at
    twice the input resolution (as per the skimage subpixel convention).

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Labelled feature image.
    nfeatures : int
        Number of grains; used to compute the pad sentinel ``nfeatures + 1``.
    connectivity : int, optional
        Connectivity for boundary detection.  Default is 1.

    Returns
    -------
    common_interface : numpy.ndarray, shape (2R + 1, 2C + 1)
        Subpixel boundary image (boolean or uint8) produced by
        ``find_boundaries`` in ``'subpixel'`` mode.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        common_interface = gbops2d.find_common_interface_boundaries(
            lfi=pxt.gs[3].lgi,
            nfeatures=pxt.gs[3].n,
            connectivity=1,
        )
    """
    lfi_padded = pad_lfi(lfi=lfi, pad_value=nfeatures+1)
    common_interface = find_boundaries(lfi_padded, connectivity=connectivity,
                                       mode='subpixel', background=0)
    return common_interface


def segment_cell_boundaries(lfi=None, lfi_boundaries=None,
                            neigh_fid=None, connectivity=1,
                            local_seg_id_nDecPlaces=4):
    """Assign each boundary pixel to the grain pair it separates.

    Calls :func:`segment_existing_boundaries` to walk every boundary pixel and
    record which two grains it lies between, then restructures the result to
    match the ``neigh_fid`` neighbour hierarchy.  Assigns both local sub-IDs
    (decimal fractions of the parent grain ID) and globally unique integer
    segment IDs.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (R, C)
        Labelled feature image.
    lfi_boundaries : numpy.ndarray of int, shape (R, C)
        Boundary pixel image produced by :func:`detect_cell_boundaries`.
    neigh_fid : dict[int, set[int]]
        Mapping ``{grain_id: {neighbour_ids}}``.
    connectivity : int, optional
        Connectivity multiplier (passed as ``4 * connectivity`` to
        :func:`segment_existing_boundaries`).  Default is 1.
    local_seg_id_nDecPlaces : int, optional
        Decimal places for local sub-IDs.  Default is 4.

    Returns
    -------
    segInfo : dict
        Dictionary with the following keys:

        * ``'bsegCoords'`` : dict[int, dict[int, numpy.ndarray]]
          Pixel coordinates of boundary segments, indexed by
          ``{grain_id: {neighbour_id: coords_array}}``.
        * ``'local_seg_ids'`` : dict[int, numpy.ndarray]
          Local decimal sub-IDs for each segment within its parent grain.
        * ``'global_seg_ids'`` : dict[int, list[int]]
          Globally unique integer segment IDs grouped by parent grain.
        * ``'segidList_lcl'`` : list
          Flat list of all local segment IDs.
        * ``'segidList_gbl'`` : list
          Flat list of all global segment IDs.
        * ``'segid_gbl_pfid_map'`` : dict[int, int]
          Maps every global segment ID back to its parent grain ID.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d

        # First detect the cell boundaries
        lfi_boundaries = gbops2d.detect_cell_boundaries(
            lfi=pxt.gs[3].lgi,
            nfeatures=pxt.gs[3].n,
            connectivity=1,
            mode='thick',
            background=0,
        )
        # Then segment those boundaries
        segInfo = gbops2d.segment_cell_boundaries(
            lfi=pxt.gs[3].lgi,
            lfi_boundaries=lfi_boundaries,
            neigh_fid=pxt.gs[3].neigh_gid,
            connectivity=1,
            local_seg_id_nDecPlaces=4,
        )
        bsegCoords = segInfo['bsegCoords']
        global_seg_ids = segInfo['global_seg_ids']
    """
    bsegCoords = segment_existing_boundaries(lfi, lfi_boundaries,
                                           connectivity=4*connectivity)
    bsegCoords = {cg: {ng: bsegCoords[(cg, ng)] for ng in ngs}
                for cg, ngs in neigh_fid.items()}
    bseg_props = characterise_boundary_segments(bsegCoords, neigh_fid)
    nsegments = bseg_props['nsegments']
    nneighs = {cg: len(ngs) for cg, ngs in neigh_fid.items()}
    if not np.any(nsegments != nneighs):
        print("Segmentation successfull",
              "Number of bsegCoords equals number of nearest neighs for all features.")
    else:
        print("Segmentation issues detected!",
              "Number of bsegCoords NOT equal to number of nearest neighs for some features.")
    local_seg_ids = {cg: np.round(np.linspace(cg, cg+1, nsegments[cg]+1)[:-1],
                                  local_seg_id_nDecPlaces)
                                  for cg, ngs in neigh_fid.items()}
    global_seg_ids = {}
    segidList_lcl = []
    segidList_gbl = []
    segid_gbl_pfid_map = {}
    seg_count_id = 1
    for parent_fid, child_fids in local_seg_ids.items():
        global_seg_ids[parent_fid] = []
        segNum = 1
        for child_fid in child_fids:
            global_seg_ids[parent_fid].append(seg_count_id)
            segidList_lcl.append(child_fid)
            segidList_gbl.append(segNum)
            segid_gbl_pfid_map[segNum] = parent_fid
            segNum += 1
            seg_count_id += 1
    segInfo = {'bsegCoords': bsegCoords,
               'local_seg_ids': local_seg_ids,
               'global_seg_ids': global_seg_ids,
               'segidList_lcl': segidList_lcl,
               'segidList_gbl': segidList_gbl,
               'segid_gbl_pfid_map': segid_gbl_pfid_map
               }
    return segInfo


def see_bsegs_gids(localSegIDMasked_lfi, gid):
    """Plot the boundary-segment pixel map for a single grain.

    Extracts and displays the slice of the locally-masked boundary image
    that corresponds to grain ``gid``, masking out all pixels outside the
    ``[gid, gid+1)`` sub-ID range.  Pixels below ``gid`` are set to NaN so
    they render as transparent in the colormap.

    Parameters
    ----------
    localSegIDMasked_lfi : numpy.ndarray of float, shape (R, C)
        Boundary image where each non-zero pixel carries the local decimal
        sub-ID of its boundary segment.
    gid : int
        Grain ID whose boundary segments should be visualised.

    Returns
    -------
    None
        Displays a ``matplotlib`` figure; no value is returned.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        # Select the largest grain
        gid = int(np.argmax(pxt.gs[3].prop.npixels.to_numpy())) + 1
        gbops2d.see_bsegs_gids(localSegIDMasked_lfi, gid)
    """
    scaled_mask = (localSegIDMasked_lfi
                   * np.asarray(localSegIDMasked_lfi >= gid, dtype=float)
                   * np.asarray(localSegIDMasked_lfi < gid + 1, dtype=float))
    scaled_mask[scaled_mask < gid] = np.nan
    plt.figure(figsize=(10, 10), dpi=50)
    plt.imshow(scaled_mask, cmap='nipy_spectral')
    plt.colorbar()
    plt.show()


def characterise_boundary_segments(bsegCoords, neigh_fid):
    """Compute per-grain segment counts and segment pixel lengths.

    Parameters
    ----------
    bsegCoords : dict[int, dict[int, numpy.ndarray]]
        Boundary-segment pixel coordinates indexed by
        ``{grain_id: {neighbour_id: coords_array}}``.
    neigh_fid : dict[int, set[int]]
        Mapping ``{grain_id: {neighbour_ids}}``.

    Returns
    -------
    bseg_props : dict
        Dictionary with two keys:

        * ``'nsegments'`` : dict[int, int]
          Number of boundary segments per grain.
        * ``'segment_lengths'`` : dict[int, list[int]]
          List of pixel counts for each segment, per grain.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        bseg_props = gbops2d.characterise_boundary_segments(
            segInfo['bsegCoords'], pxt.gs[3].neigh_gid
        )
        print(bseg_props['nsegments'])
    """
    nsegments = {cg: len(bsegCoords[cg]) for cg in neigh_fid.keys()}
    segment_lengths = {cg: [len(seg) for ng, seg in bsegCoords[cg].items()]
                       for cg in bsegCoords.keys()}
    bseg_props = {'nsegments': nsegments,
                  'segment_lengths': segment_lengths}
    return bseg_props


def detect_junction_points(segments):
    """Find pixels shared by two or more boundary segments of the same grain.

    For each grain, stacks all boundary-segment coordinate arrays and looks
    for pixel positions that appear in more than one segment.  These are
    junction points where three or more grains meet.  The returned order
    value is ``count + 1`` (e.g. a pixel shared by 2 segments has order 3,
    corresponding to a triple junction).

    Parameters
    ----------
    segments : dict[int, dict[int, numpy.ndarray]]
        Boundary-segment coordinates indexed by
        ``{grain_id: {neighbour_id: coords_array}}``.

    Returns
    -------
    all_junctions : dict[int, numpy.ndarray]
        Grain ID → array of shape (N_junctions, 3).  Columns are
        ``[row, col, junction_order]``.  Only grains that have at least one
        junction point appear as keys.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        junctionPoints = gbops2d.detect_junction_points(segInfo['bsegCoords'])
        # Inspect junctions for grain 5
        print(junctionPoints.get(5))
    """
    all_junctions = {}
    for cg, neighbors in segments.items():
        all_pts = [pts for pts in neighbors.values() if pts.size > 0]
        if not all_pts:
            continue
        stacked_pts = np.vstack(all_pts)
        unique_pts, counts = np.unique(stacked_pts, axis=0, return_counts=True)
        junction_mask = counts >= 2
        if np.any(junction_mask):
            res = np.column_stack((unique_pts[junction_mask], counts[junction_mask] + 1))
            all_junctions[cg] = res
    return all_junctions


def find_basic_JPO_stats(junctionPoints, segidList_gbl):
    """Compute min, max, and median junction-point order (JPO) statistics.

    Calculates per-grain and tessellation-wide summary statistics for the
    junction-point order (JPO), which is the number of grains meeting at a
    junction pixel.

    Parameters
    ----------
    junctionPoints : dict[int, numpy.ndarray]
        Output of :func:`detect_junction_points`.  Each array has shape
        (N_junctions, 3) with the third column holding the junction order.
    segidList_gbl : list of int
        Flat list of global segment IDs (``segInfo['segidList_gbl']``).
        Used to size the tessellation-wide median accumulator.

    Returns
    -------
    JPOStats : dict
        Dictionary with the following keys:

        * ``'min_tess'`` : int — global minimum JPO.
        * ``'max_tess'`` : int — global maximum JPO.
        * ``'median_tess'`` : float — global median JPO across all segments.
        * ``'min_feat'`` : dict[int, int] — per-grain minimum JPO.
        * ``'max_feat'`` : dict[int, int] — per-grain maximum JPO.
        * ``'median_feat'`` : dict[int, float] — per-grain median JPO.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        JPOStats = gbops2d.find_basic_JPO_stats(
            junctionPoints, segInfo['segidList_gbl']
        )
        print("Global max JPO:", JPOStats['max_tess'])
    """
    minJPO_feat = {gid: jp[:, 2].min() for gid, jp in junctionPoints.items()}
    maxJPO_feat = {gid: jp[:, 2].max() for gid, jp in junctionPoints.items()}
    medianJPO_feat = {gid: np.median(jp[:, 2]) for gid, jp in junctionPoints.items()}
    minJPO_tess = min(list(minJPO_feat.values()))
    maxJPO_tess = max(list(maxJPO_feat.values()))
    medianJPO_tess = np.zeros(len(segidList_gbl))
    for jp in junctionPoints.values():
        seg_count = 0
        for _jpo_ in jp[:, 2]:
            medianJPO_tess[seg_count] = _jpo_
    medianJPO_tess = np.median(medianJPO_tess)
    JPOStats = {'min_tess': minJPO_tess, 'max_tess': maxJPO_tess,
                'median_tess': medianJPO_tess, 'min_feat': minJPO_feat,
                'max_feat': maxJPO_feat, 'median_feat': medianJPO_feat}
    return JPOStats


def sort_junction_points(junctionPoints):
    """Group junction points by their junction-point order (JPO).

    Organises the flat ``junctionPoints`` dict into a nested dict keyed by
    human-readable JPO labels (``'jp1'`` … ``'jp6'``, ``'tjp'`` for order 3,
    ``'qjp'`` for order 4), with each entry further keyed by grain ID.

    Parameters
    ----------
    junctionPoints : dict[int, numpy.ndarray]
        Output of :func:`detect_junction_points`.

    Returns
    -------
    JPSorted : dict[str, dict[int, numpy.ndarray]]
        Nested dictionary ``{jpo_label: {grain_id: coords_array}}``.
        ``coords_array`` has shape (N, 2) (row, col only; order column dropped).
        Keys present are only those JPO values actually found in the data.

    Notes
    -----
    The label mapping is: 1 → ``'jp1'``, 2 → ``'jp2'``, 3 → ``'tjp'``
    (triple junction), 4 → ``'qjp'`` (quadruple junction), 5 → ``'jp5'``,
    6 → ``'jp6'``.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        JPSorted = gbops2d.sort_junction_points(junctionPoints)
        # Retrieve all triple-junction pixels for grain 7
        tjp_grain7 = JPSorted['tjp'].get(7)
    """
    jpo_names = {1: 'jp1', 2: 'jp2', 3: 'tjp', 4: 'qjp', 5: 'jp5', 6: 'jp6'}
    all_jpo_values = np.concatenate([jp[:, 2] for jp in junctionPoints.values()])
    min_jpo = int(all_jpo_values.min())
    max_jpo = int(all_jpo_values.max())
    JPSorted = {}
    for jpo in range(min_jpo, max_jpo + 1):
        jpo_dict = {}
        for gid, jp in junctionPoints.items():
            filtered = jp[jp[:, 2] == jpo, 0:2]
            if filtered.shape[0] > 0:
                jpo_dict[gid] = filtered
        JPSorted[jpo_names[jpo]] = jpo_dict
    return JPSorted


def mask_featIDImg_at_coords(featIDImg, coordData,
                             featName='fbseg', maskDType=np.int32):
    """Paint segment IDs onto a copy of the boundary image at given coordinates.

    Creates a deep copy of ``featIDImg`` and overwrites each pixel listed in
    ``coordData`` with its corresponding segment sub-ID.

    Parameters
    ----------
    featIDImg : numpy.ndarray of int, shape (R, C)
        Source boundary image (e.g. ``lfi_boundaries``) to be copied and
        annotated.
    coordData : dict[int, dict[int, numpy.ndarray]]
        Boundary-segment pixel coordinates indexed by
        ``{grain_id: {neighbour_id: coords_array}}``.
    featName : str, optional
        Feature type selector.  Currently only ``'fbseg'``, ``'gbseg'``, and
        ``'feature_boudnary_segments'`` are handled.  Default is ``'fbseg'``.
    maskDType : numpy dtype, optional
        dtype of the output array.  Default is ``np.int32``.

    Returns
    -------
    featIDImg_new : numpy.ndarray of int, shape (R, C)
        Copy of ``featIDImg`` with segment IDs written at boundary-pixel
        positions.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        masked = gbops2d.mask_featIDImg_at_coords(
            lfi_boundaries,
            segInfo['bsegCoords'],
            featName='fbseg',
            maskDType=np.int32,
        )
    """
    if featName in ('fbseg', 'gbseg', 'feature_boudnary_segments'):
        featIDImg_new = np.asarray(deepcopy(featIDImg), dtype=maskDType)
        for cg, ngcoords in coordData.items():
            for i, (ng, ngsegcoords) in enumerate(ngcoords.items()):
                segid = coordData[cg][i]
                for sc in ngsegcoords:
                    featIDImg_new[sc[0]][sc[1]] = segid
    return featIDImg_new


def pad_with(vector, pad_width, iaxis, kwargs):
    """Custom padding function compatible with ``numpy.pad``.

    Fills the leading and trailing pad regions of ``vector`` with the value
    supplied via ``kwargs['padder']``.  This function is passed directly to
    ``numpy.pad`` as the ``mode`` argument (callable form).

    Parameters
    ----------
    vector : numpy.ndarray, 1-D
        The padded edge of one axis, as provided by ``numpy.pad``.
    pad_width : tuple of int
        ``(before, after)`` number of padded values on each end.
    iaxis : int
        The axis currently being padded (not used; required by the interface).
    kwargs : dict
        Must contain key ``'padder'`` whose value is the constant fill value.
        Defaults to 10 if the key is absent.

    Returns
    -------
    vector : numpy.ndarray, 1-D
        The modified vector with pad regions filled.

    Notes
    -----
    Implementation taken verbatim from the NumPy documentation example:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def segment_existing_boundaries(labels, gb_mask, connectivity=8):
    """Walk boundary pixels and assign them to grain-pair interface lists.

    Iterates over every non-zero pixel in ``gb_mask`` and, for each adjacent
    pixel (within the requested connectivity), records the pixel as belonging
    to the interface between the two grain IDs.  Duplicate pixel positions are
    removed with ``numpy.unique``.

    Parameters
    ----------
    labels : numpy.ndarray of int, shape (R, C)
        Labelled feature image (grain ID at every pixel).
    gb_mask : numpy.ndarray of int or bool, shape (R, C)
        Boundary pixel mask (non-zero where boundary pixels exist).
    connectivity : int, optional
        Neighbourhood size: 4 (face neighbours only) or 8 (face + diagonal).
        Default is 8.

    Returns
    -------
    interfaces : dict[tuple[int, int], numpy.ndarray]
        Maps each ``(grain_A, grain_B)`` pair to a unique array of
        ``(row, col)`` pixel positions on their shared boundary.

    Notes
    -----
    Called internally by :func:`segment_cell_boundaries` with
    ``connectivity = 4 * user_connectivity``.

    Usage
    -----
    ::

        import upxo.gbops.mcgb2dops as gbops2d

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        interfaces = gbops2d.segment_existing_boundaries(
            labels=lfi,
            gb_mask=lfi_boundaries,
            connectivity=4,
        )
        coords_1_2 = interfaces.get((1, 2))
    """
    rows, cols = np.where(gb_mask)
    interfaces = defaultdict(list)
    if connectivity == 4:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]
    labels_int = labels.astype(int)
    max_r, max_c = labels.shape
    for r, c in zip(rows, cols):
        g1 = labels_int[r, c]
        if g1 == 0:
            continue
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < max_r and 0 <= nc < max_c:
                g2 = labels_int[nr, nc]
                if g2 != 0 and g2 != g1:
                    interfaces[(int(g1), int(g2))].append((r, c))
    return {pair: np.unique(np.array(pts), axis=0) for pair, pts in interfaces.items()}


def circular_sort(coords=None,
                  order='ccw',
                  origin_spec='tl',
                  origin_point=(0, 0)):
    """Sort boundary-point coordinates in circular (CW or CCW) order.

    Wrapper that dispatches to the chosen sorting method.  Currently only
    ``method1`` (four-corner sort) is called; methods 2 and 3 are stubs
    pending implementation.

    Parameters
    ----------
    coords : numpy.ndarray, shape (N, 2), optional
        Array of (x, y) boundary-point coordinates.  Default is None.
    order : str, optional
        Circular sorting direction.  Accepted values: ``'cw'``,
        ``'clockwise'``, ``'ccw'``, ``'acw'``, ``'counterclockwise'``,
        ``'anticlockwise'``.  Default is ``'ccw'``.
    origin_spec : str, optional
        Specifies which point becomes the first in the sorted output.
        Options:

        * ``'tl'`` — top-left (default)
        * ``'tr'`` — top-right
        * ``'br'`` — bottom-right
        * ``'bl'`` — bottom-left
        * ``'l'`` — leftmost
        * ``'t'`` — topmost
        * ``'r'`` — rightmost
        * ``'b'`` — bottommost
        * ``'closest'`` — point nearest to ``origin_point``
        * ``'ignore'`` — use ``origin_point`` directly as the start

    origin_point : tuple or list of float, optional
        Reference point used when ``origin_spec`` is ``'closest'`` or
        ``'ignore'``.  Default is ``(0, 0)``.

    Returns
    -------
    None
        *This function is currently incomplete.*  It calls
        :func:`circular_sort_method1` but does not propagate or return its
        result.  Methods 2 and 3 are not yet implemented.

    See Also
    --------
    circular_sort_method1 : Full implementation of method 1.
    method2 : Angle-based CCW sort.
    method3_2d : Centroid-angle sort.
    """
    circular_sort_method1(coords)

    if method == 2:
        pass

    if method == 3:
        pass


def circular_sort_method1(jp_grainwise,
                           sort_order,
                           method='method1',
                           debug_mode=True):
    """Sort grain junction-point coordinates clockwise or anti-clockwise.

    Iterates over each grain's junction-point coordinate set and sorts them
    using the selected geometric method.  Builds a structured output dict
    that records the sort method, the effective sort order, and the sorted
    coordinates with their original-index permutation for each grain.

    Parameters
    ----------
    jp_grainwise : dict[int, numpy.ndarray]
        Grain ID → array of shape (2, N) where row 0 is x-coordinates and
        row 1 is y-coordinates of the junction points.
    sort_order : str
        Sorting direction; used only when ``method == 'method2'``.  For
        ``'method1'`` the sort is always clockwise (``'cw'``).
    method : str, optional
        Sorting algorithm to use.  Options:

        * ``'method1'`` — four-corner ordering (TL → TR → BR → BL) via
          :func:`method1`.  Works best for convex, roughly rectangular sets.
        * ``'method2'`` — angle-based CCW sort via :func:`method2`.

        Default is ``'method1'``.
    debug_mode : bool, optional
        When ``True``, prints intermediate coordinate arrays to stdout.
        Default is ``True``.

    Returns
    -------
    jp_sorted : dict
        Dictionary with keys:

        * ``'method'`` : str — the method used.
        * ``'sortorder'`` : str — effective sort order (``'cw'`` for method1,
          ``sort_order`` for method2, or the default method name otherwise).
        * ``'sorted'`` : dict[int, dict] — per-grain result where each entry
          has sub-keys ``'coords'`` (sorted coordinate array) and ``'indices'``
          (index permutation into the original array).

    Notes
    -----
    When a grain has only a single junction point, no sorting is performed
    and the original coordinate is stored directly.  When exactly two points
    exist under method1, they are returned as-is with indices ``[0, 1]``.

    Examples
    --------
    Prerequisite setup:

    .. code-block:: python

        from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs
        PXGS = mcgs()
        PXGS.simulate()
        PXGS.detect_grains()

    Sort the junction points:

    .. code-block:: python

        from upxo.gbops.mcgb2dops import circular_sort_method1
        jp_sorted = circular_sort_method1(PXGS.gs[8].jp_grainwise, sort_order='cw')
    """
    _default_sorting_method = 'method1'
    _jp_sorted = {gid: {'coords': None,
                        'indices': None} for gid in jp_grainwise.keys()}
    jp_sorted = {
        'method': method,
        'sortorder': sort_order if method == 'method2'
        else 'cw' if method == 'method1' else _default_sorting_method,
        'sorted': _jp_sorted}
    for gid, coords in jp_grainwise.items():
        if coords[0].size == 1:
            jp_sorted['values'][gid]['coords'] = coords
            jp_sorted['values'][gid]['indices'] = 0
        elif coords[0].size > 1:
            if debug_mode:
                print('\n', 10*'-', '\n', 'x:', coords[0], 'y:', coords[1])
            if method == 'method1':
                if coords[0].size == 2:
                    sorted_coords = coords
                    sorted_indices = np.array([0, 1])
                elif coords[0].size > 2:
                    sorted_coords, sorted_indices = method1(coords)
            elif method == 'method2':
                sorted_coords, sorted_indices = method2(coords)
            if debug_mode:
                print('Sorted coords: X.',
                      sorted_coords[0],
                      'Y:', sorted_coords[1])
            jp_sorted['values'][gid]['coords'] = sorted_coords
            jp_sorted['values'][gid]['indices'] = sorted_indices
    return jp_sorted


def method1(coords):
    """Sort four points in TL → TR → BR → BL order.

    Separates the input points into the two leftmost and two rightmost by
    x-coordinate, then orders each pair by y-coordinate to identify
    top-left, bottom-left, top-right, and bottom-right corners.

    Parameters
    ----------
    coords : numpy.ndarray, shape (N, 2)
        Point coordinates as rows of (x, y).  Intended for exactly four
        points; behaviour for N ≠ 4 is undefined.

    Returns
    -------
    sorted_coords : numpy.ndarray, shape (4, 2)
        Points ordered TL → TR → BR → BL.
    sorted_indices : None
        Index permutation is not computed by this method; always returns
        ``None``.

    Notes
    -----
    Adapted from:
    https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
    """
    pts = coords.T
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    print(rightMost)
    (tr, br) = rightMost
    sorted_indices = None
    sorted_coords = np.array([tl, tr, br, bl])
    return sorted_coords, sorted_indices


def method2(coords):
    """Sort points in CCW order around the leftmost point.

    Centres the point cloud on its leftmost (then topmost) point and sorts
    by the angle each point subtends relative to that origin.

    Parameters
    ----------
    coords : numpy.ndarray, shape (N, 2)
        Point coordinates as rows of (x, y).

    Returns
    -------
    sorted_coords : numpy.ndarray, shape (N, 2)
        Points in ascending angle order (counter-clockwise from the leftmost
        point).
    sorted_indices : numpy.ndarray of int, shape (N,)
        Index permutation mapping the sorted positions back to the original
        ``coords`` rows.
    """
    leftmost_point = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]
    centered_coords = coords - leftmost_point
    angles = np.arctan2(centered_coords[:, 1], centered_coords[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords, sorted_indices


def method3_2d(coords=None,
               order='ccw',
               origin_spec='tl',
               origin_point=(0, 0),
               method=3):
    """Sort boundary points in circular order using centroid-angle decomposition.

    Computes the centroid of ``coords``, then sorts points by the angle they
    subtend with respect to that centroid.  Supports both clockwise and
    counter-clockwise ordering.

    Parameters
    ----------
    coords : numpy.ndarray, shape (N, 2), optional
        (x, y) boundary-point coordinates.  Default is None.
    order : str, optional
        Circular sorting direction.  Accepted values: ``'cw'``,
        ``'clockwise'``, ``'ccw'``, ``'acw'``, ``'counterclockwise'``,
        ``'anticlockwise'``.  Default is ``'ccw'``.
    origin_spec : str, optional
        Specifies the starting point of the sorted output.  Options: ``'tl'``,
        ``'tr'``, ``'br'``, ``'bl'``, ``'l'``, ``'t'``, ``'r'``, ``'b'``,
        ``'closest'``, ``'ignore'``.  Default is ``'tl'``.
    origin_point : tuple or list of float, optional
        Reference point used when ``origin_spec`` is ``'closest'`` or
        ``'ignore'``.  Default is ``(0, 0)``.
    method : int, optional
        Sorting method selector (1, 2, or 3).  This parameter is accepted for
        API compatibility but is unused by the centroid-angle logic here.
        Default is 3.

    Returns
    -------
    clockwise_sorted_points : numpy.ndarray, shape (N, 2)
        Points sorted in clockwise order.
    clockwise_sorted_indices : list of int
        Index permutation for the clockwise-sorted output.
    counterclockwise_sorted_points : numpy.ndarray, shape (N, 2)
        Points sorted in counter-clockwise order.
    counterclockwise_sorted_indices : list of int
        Index permutation for the CCW-sorted output.

    Notes
    -----
    ``numpy.arctan2`` returns angles in ``[-π, π]``.  Sorting in ascending
    order yields a counter-clockwise sequence; inverting the angle before
    sorting yields clockwise.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import upxo.gbops.mcgb2dops as gbops2d
        pts = np.array([[1, 2], [3, 4], [2, 0], [0, 1]], dtype=float)
        cw, cw_idx, ccw, ccw_idx = gbops2d.method3_2d(coords=pts, order='cw')
    """
    centroid = np.mean(coords, axis=0)

    def angle_and_distance(point, centroid, clockwise=False):
        """Angle and distance."""
        angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        if clockwise:
            angle = -angle
        distance = np.linalg.norm(point - centroid)
        return angle, distance

    def sort_points(points, clockwise=False):
        """Sort points."""
        sorted_indices = sorted(range(len(points)),
                                key=lambda i: angle_and_distance(points[i], centroid, clockwise))
        sorted_points = points[sorted_indices]
        return sorted_points, sorted_indices

    clockwise_sorted_points, clockwise_sorted_indices = sort_points(coords, clockwise=True)
    counterclockwise_sorted_points, counterclockwise_sorted_indices = sort_points(coords, clockwise=False)
    return (clockwise_sorted_points, clockwise_sorted_indices,
            counterclockwise_sorted_points, counterclockwise_sorted_indices)


def calculate_junction_order(matrix, junction_point):
    """Count the number of distinct grains in the 3×3 neighbourhood of a junction pixel.

    Extracts a 3×3 window centred on ``junction_point`` and counts the unique
    non-zero grain IDs within it.  This count is the junction-point order (JPO):
    3 for a triple junction, 4 for a quadruple junction, etc.

    Parameters
    ----------
    matrix : numpy.ndarray of int, shape (R, C)
        Labelled feature image (grain ID at every pixel).
    junction_point : tuple of int
        ``(row, col)`` pixel position of the junction point to evaluate.

    Returns
    -------
    order : int
        Number of distinct non-zero grain IDs in the 3×3 neighbourhood,
        equal to the junction-point order.

    Notes
    -----
    The neighbourhood is clipped to the image boundary at the edges, so
    junction points on the border of the image may return a lower order than
    expected.

    Examples
    --------
    .. code-block:: python

        import upxo.gbops.mcgb2dops as gbops2d
        order = gbops2d.calculate_junction_order(lfi, junction_point=(45, 32))
        print(f"Junction order: {order}")
    """
    y, x = junction_point
    neighborhood = matrix[max(0, y-1):y+2, max(0, x-1):x+2]
    unique_grains = np.unique(neighborhood)
    order = len(unique_grains[unique_grains > 0])
    return order


def build_grain_pairs(neigh_gid):
    """Build unique grain pairs from a neighbour dictionary.

    Parameters
    ----------
    neigh_gid : dict[int, list[int]]
        Mapping of grain ID → list of neighbour grain IDs.

    Returns
    -------
    grain_pairs : list[tuple[int, int]]
        Unique pairs ``(g1, g2)`` with ``g2 > g1``.
    """
    grain_pairs = []
    for grain_id, neighbors in neigh_gid.items():
        for neighbor_id in neighbors:
            if neighbor_id > grain_id:
                grain_pairs.append((grain_id, neighbor_id))
    return grain_pairs


def identify_grain_boundary_pixels(lgi, grain_pairs):
    """Return pixel coordinates of each grain-pair interface.

    Scans horizontal and vertical pixel adjacencies to find pixels where
    two specified grain IDs are immediate neighbours.

    Parameters
    ----------
    lgi : numpy.ndarray of int, shape (R, C)
        Labelled grain image.
    grain_pairs : list of tuple
        Each entry is ``(grain_id1, grain_id2)``; order within the tuple
        does not matter.

    Returns
    -------
    gb_dict : dict[tuple[int, int], numpy.ndarray]
        Sorted pair ``(g1, g2)`` → array of shape (N, 2) with
        ``[row, col]`` coordinates of boundary pixels.
    """
    target_pairs = set(tuple(sorted((int(p[0]), int(p[1])))) for p in grain_pairs)
    temp_store = defaultdict(list)
    # Horizontal pass — detects vertical grain boundaries
    left, right = lgi[:, :-1], lgi[:, 1:]
    mask_h = left != right
    rows_h, cols_h = np.nonzero(mask_h)
    for r, c, g1, g2 in zip(rows_h, cols_h, left[mask_h], right[mask_h]):
        pair = tuple(sorted((int(g1), int(g2))))
        if pair in target_pairs:
            temp_store[pair].append([float(r), float(c)])
    # Vertical pass — detects horizontal grain boundaries
    top, bottom = lgi[:-1, :], lgi[1:, :]
    mask_v = top != bottom
    rows_v, cols_v = np.nonzero(mask_v)
    for r, c, g1, g2 in zip(rows_v, cols_v, top[mask_v], bottom[mask_v]):
        pair = tuple(sorted((int(g1), int(g2))))
        if pair in target_pairs:
            temp_store[pair].append([float(r), float(c)])
    return {pair: np.array(coords, dtype=np.float64)
            for pair, coords in temp_store.items()}


def find_gb_junction_point_map(lgi):
    """Return a binary map of grain-boundary junction pixels.

    A pixel is a junction if three or more distinct non-zero grain IDs
    appear in its 3×3 neighbourhood.  Only boundary pixels (detected by
    ``skimage.segmentation.find_boundaries``) are tested, which avoids
    evaluating interior pixels and reduces cost for large images.

    Parameters
    ----------
    lgi : numpy.ndarray of int, shape (R, C)
        Labelled grain image.

    Returns
    -------
    jmap : numpy.ndarray of int, shape (R, C)
        1 at junction pixels, 0 elsewhere.

    Notes
    -----
    Delegates per-pixel neighbourhood counting to
    :func:`calculate_junction_order`, the single-pixel companion of this
    function.
    """
    from skimage.segmentation import find_boundaries
    gb_mask = find_boundaries(lgi, connectivity=1, mode='thick')
    boundary_coords = np.argwhere(gb_mask)
    jmap = np.zeros(lgi.shape, dtype=int)
    for yx in boundary_coords:
        if calculate_junction_order(lgi, tuple(yx)) >= 3:
            jmap[yx[0], yx[1]] = 1
    return jmap


def compute_grain_boundary_locs(grain_mask):
    """Return pixel indices of a single grain's boundary via binary erosion.

    Boundary pixels are those inside the grain but absent from the eroded
    grain: ``boundary = mask AND NOT erode(mask)``.

    Parameters
    ----------
    grain_mask : numpy.ndarray of bool or uint8, shape (R, C)
        Binary mask where non-zero marks the grain's pixels.

    Returns
    -------
    gbloc : numpy.ndarray of int, shape (N, 2)
        Row/column indices of boundary pixels.
    """
    from scipy.ndimage import binary_erosion
    mask_bool = grain_mask.astype(bool)
    boundary = mask_bool & ~binary_erosion(mask_bool)
    return np.argwhere(boundary)
