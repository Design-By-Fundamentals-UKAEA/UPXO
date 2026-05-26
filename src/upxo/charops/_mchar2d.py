import numpy as np
# import pandas as pd

def detect_features(mcStateArray, connectivity=18, delta=0):
    """
    Detect features in 2D/3D image data using connectivity-based analysis.
    This function identifies and labels connected features in 3D image data based on
    specified connectivity criteria and a delta threshold parameter.
    Parameters
    ----------
    mcStateArray : numpy.ndarray
        3D array containing the image data to analyze for feature detection.
    connectivity : int, optional
        Connectivity criterion for feature detection. Default is 18.
        Common values are 6 (face), 18 (face+edge), or 26 (face+edge+vertex).
    delta : int or float, optional
        Threshold parameter for feature detection. Default is 0.
        Controls the sensitivity of feature detection.
    Returns
    -------
    lfi : numpy.ndarray
        Labeled feature image where each connected feature is assigned a unique
        integer label. Background is typically labeled as 0.
    N : int
        Total number of features detected in the image data.
    connectivity : int
        Connectivity provided by the user. Note: Use this to port value into
        the class attrtibute you are working with to maintain uniformity of 
        connectivity in subsequent operations. In all subsequent operations
        make sure to pass the saved connectivity parameter value rather than 
        having to create a new one. Think of tyhis output to force tyou to 
        save this iff you are working through a class. Else, just ignore this
        output.

    Notes
    -----
    This function wraps the underlying _mchar3d.detect_features implementation
    for 3D morphological character analysis.

    Usage
    -----
    import upxo.charops._mchar2d as mchar2d
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10, 10)
    >>> labeled_features, num_features = detect_features(data, connectivity=18)
    >>> print(f"Detected {num_features} features")
    """
    import upxo.gsdataops.grid_ops as gridOps
    lfi, N = gridOps.detect_grains_cc3d(mcStateArray, 
                                        connectivity=connectivity, 
                                        delta=delta)
    return lfi, N, connectivity

def characterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
                                    make_skprops=True, extract_coords=True,
                                    throw_bounding_box=True
                                    ):
    """
    Charecterise features in an image.
    Parameters
    ----------
    labelled_image : numpy.ndarray
        Labeled image where each feature has a unique integer label.
    Xgrid : numpy.ndarray
        X-coordinates grid corresponding to the image.
    Ygrid : numpy.ndarray
        Y-coordinates grid corresponding to the image.
    make_skprops : bool, optional
        Whether to create scikit-image regionprops for features. Default is True.
    extract_coords : bool, optional
        Whether to extract feature coordinates. Default is True.
    throw_bounding_box : bool, optional
        Whether to throw bounding box for features. Default is True.
    
    Returns
    -------
    skprops : dict
        Dictionary with feature IDs as keys and their scikit-image regionprops as values.
    bbox_limits_ex : dict
        Dictionary with feature IDs as keys and their extended bounding box limits as values.
    bboxes_ex : dict
        Dictionary with feature IDs as keys and their extended bounding box images as values.
    coords_dict : dict
        Dictionary with feature IDs as keys and their coordinates as values.

    Example
    -------
    skprops, bbox_limits, bboxes_ex, coords_dict = ...characterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
                                    make_skprops=True, extract_coords=True,
                                    throw_bounding_box=True
                                    )
    """
    fids = [int(fid) for fid in np.unique(labelled_image)]
    skprops = {fid: None for fid in fids}
    bbox_limits_ex = {fid: None for fid in fids}
    bboxes_ex = {fid: None for fid in fids}
    coords_dict = {fid: None for fid in fids}
    if make_skprops:
        from skimage.measure import regionprops
    for fid in fids:
        mask = (labelled_image == fid).astype(np.uint8)
        loc_ = np.where(mask)
        rmin, rmax = loc_[0].min(), loc_[0].max()+1
        cmin, cmax = loc_[1].min(), loc_[1].max()+1
        Rlab, Clab = mask.shape
        rmin_ex, rmax_ex = rmin-int(rmin != 0), rmax+int(rmin != Rlab)
        cmin_ex, cmax_ex = cmin-int(cmin != 0), cmax+int(cmax != Clab)
        bbox_ex = mask[rmin_ex:rmax_ex, cmin_ex:cmax_ex].copy()
        if throw_bounding_box:
            bbox_limits_ex[fid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
            bboxes_ex[fid] = bbox_ex
        if extract_coords:
            coords_dict[fid] = np.array([[Xgrid[ij[0], ij[1]], Ygrid[ij[0], ij[1]]]
                            for ij in np.argwhere(mask)])
        if make_skprops:
            skprops[fid] = regionprops(bbox_ex, cache=False)[0]
    return skprops, bbox_limits_ex, bboxes_ex, coords_dict

def characterise_features_in_image_v2(labelled_image, Xgrid=None, Ygrid=None,
                                    make_skprops=True, extract_coords=True, 
                                    throw_bounding_box=True):
    """Characterise every labelled feature in a 2D image (single-pass, faster).

    Calls ``skimage.measure.regionprops`` once on the full labelled image rather
    than constructing a binary mask per feature, making it considerably faster
    than :func:`characterise_features_in_image_2d` for images with many grains.
    Returns both tight and one-pixel-padded bounding boxes.

    Parameters
    ----------
    labelled_image : numpy.ndarray of int, shape (R, C)
        Integer-labelled image. 0 is background; each positive integer
        identifies one feature (grain).
    Xgrid : numpy.ndarray of float, shape (R, C), or None
        Physical X-coordinate at every pixel. When ``None``, column index
        is used as the X coordinate.
    Ygrid : numpy.ndarray of float, shape (R, C), or None
        Physical Y-coordinate at every pixel. When ``None``, row index
        is used as the Y coordinate.
    make_skprops : bool, default True
        Retain the ``skimage.measure.RegionProperties`` object for each
        feature in ``skprops``.
    extract_coords : bool, default True
        Record the physical (X, Y) coordinates of every pixel belonging
        to each feature in ``coords_dict``.
    throw_bounding_box : bool, default True
        Populate tight (``bbox_limits``, ``bboxes``) and extended
        (``bbox_limits_ex``, ``bboxes_ex``) bounding-box outputs.

    Returns
    -------
    skprops : dict[int, skimage.measure.RegionProperties or None]
        Feature ID → region-properties object from the full-image
        ``regionprops`` call. ``None`` when ``make_skprops`` is False.
    bbox_limits : dict[int, list[int] or None]
        Feature ID → ``[rmin, rmax, cmin, cmax]`` tight bounding-box
        slice indices, clamped to image boundaries.
        ``None`` when ``throw_bounding_box`` is False.
    bbox_limits_ex : dict[int, list[int] or None]
        Feature ID → ``[rmin_ex, rmax_ex, cmin_ex, cmax_ex]``
        one-pixel-padded bounding-box slice indices.
        ``None`` when ``throw_bounding_box`` is False.
    bboxes : dict[int, numpy.ndarray or None]
        Feature ID → binary int32 crop of the tight bounding box.
        ``None`` when ``throw_bounding_box`` is False.
    bboxes_ex : dict[int, numpy.ndarray or None]
        Feature ID → binary int32 crop of the padded bounding box.
        ``None`` when ``throw_bounding_box`` is False.
    coords_dict : dict[int, numpy.ndarray or None]
        Feature ID → array of shape (N_pixels, 2) with physical
        ``[X, Y]`` coordinates of every pixel in the feature.
        ``None`` when ``extract_coords`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> from upxo.charops._mchar2d import characterise_features_in_image_v2
    >>> lfi = np.array([[1, 1, 0], [0, 2, 2], [0, 2, 0]], dtype=int)
    >>> skp, bl, bl_ex, bx, bx_ex, coords = characterise_features_in_image_v2(lfi)
    """
    if Xgrid is None or Ygrid is None:
        h, w = labelled_image.shape
        indices = np.indices((h, w))
        Xgrid, Ygrid = indices[1], indices[0]
    # -------------------------------------
    from skimage.measure import regionprops
    props = regionprops(labelled_image)

    skprops = {}
    bbox_limits = {}
    bbox_limits_ex = {}
    bboxes = {}
    bboxes_ex = {}
    coords_dict = {}
    Rlab, Clab = labelled_image.shape
    for prop in props:
        fid = prop.label
        # prop.bbox gives (min_row, min_col, max_row, max_col)
        rmin, cmin, rmax, cmax = prop.bbox

        rmin = max(0, rmin)
        rmax = min(Rlab, rmax)
        cmin = max(0, cmin)
        cmax = min(Clab, cmax)
        
        rmin_ex = max(0, rmin - 1)
        rmax_ex = min(Rlab, rmax + 1)
        cmin_ex = max(0, cmin - 1)
        cmax_ex = min(Clab, cmax + 1)
        if throw_bounding_box:
            bbox_limits[fid] = [rmin, rmax, cmin, cmax]
            bbox_limits_ex[fid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
            # slice for the extended bounding box
            bboxes[fid] = (labelled_image[rmin:rmax, cmin:cmax] == fid).astype(np.int32)
            bboxes_ex[fid] = (labelled_image[rmin_ex:rmax_ex, cmin_ex:cmax_ex] == fid).astype(np.int32)
        if extract_coords:
            # prop.coords gives indices (row, col) of all pixels in the grain
            coords = prop.coords 
            rows = coords[:, 0]
            cols = coords[:, 1]
            coords_dict[fid] = np.column_stack((Xgrid[rows, cols], Ygrid[rows, cols]))
        if make_skprops:
            skprops[fid] = prop

    return skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict


def classify_grain_positions_2d(lgi, gid):
    """Classify each grain as corner / edge / internal based on lgi pixel positions.

    Parameters
    ----------
    lgi : numpy.ndarray of int, shape (R, C)
        Labelled grain image.
    gid : array-like of int
        All grain IDs present in lgi.

    Returns
    -------
    positions : dict[str, numpy.ndarray]
        Keys: 'top_left', 'top_right', 'bottom_left', 'bottom_right',
              'pure_top', 'pure_bottom', 'pure_left', 'pure_right',
              'top', 'bottom', 'left', 'right', 'boundary', 'corner', 'internal'.
        Values: arrays of grain IDs belonging to each category.
    """
    top_left     = np.array([lgi[-1,  0]], dtype=int)
    top_right    = np.array([lgi[-1, -1]], dtype=int)
    bottom_left  = np.array([lgi[0,   0]], dtype=int)
    bottom_right = np.array([lgi[0,  -1]], dtype=int)
    left   = np.unique(lgi[:,  0])
    right  = np.unique(lgi[:, -1])
    top    = np.unique(lgi[-1, :])
    bottom = np.unique(lgi[0,  :])
    pure_top    = np.setdiff1d(top,    np.union1d(top_left,    top_right))
    pure_bottom = np.setdiff1d(bottom, np.union1d(bottom_left, bottom_right))
    pure_left   = np.setdiff1d(left,   np.union1d(top_left,    bottom_left))
    pure_right  = np.setdiff1d(right,  np.union1d(top_right,   bottom_right))
    internal = np.setdiff1d(gid, np.union1d(np.union1d(top, bottom),
                                             np.union1d(left, right)))
    return {
        'left': left, 'right': right, 'top': top, 'bottom': bottom,
        'top_left': top_left, 'top_right': top_right,
        'bottom_left': bottom_left, 'bottom_right': bottom_right,
        'pure_top': pure_top, 'pure_bottom': pure_bottom,
        'pure_left': pure_left, 'pure_right': pure_right,
        'internal': internal,
        'boundary': np.union1d(np.union1d(top, bottom), np.union1d(left, right)),
        'corner': np.union1d(np.union1d(top_left, top_right),
                             np.union1d(bottom_left, bottom_right)),
    }


def extract_prop_npixels(locs_list):
    """Return pixel counts for each grain from a list of location arrays."""
    return [len(loc) for loc in locs_list]


def extract_prop_gb_pixels(gblocs_list):
    """Return grain-boundary pixel counts from a list of boundary location arrays."""
    return [len(gbloc) for gbloc in gblocs_list]


def extract_prop_area(skprops):
    return [skprops[fid].area for fid in skprops]


def extract_prop_eq_diameter(skprops):
    return [skprops[fid].equivalent_diameter_area for fid in skprops]


def extract_prop_perimeter(skprops):
    return [skprops[fid].perimeter for fid in skprops]


def extract_prop_perimeter_crofton(skprops):
    return [skprops[fid].perimeter_crofton for fid in skprops]


def extract_prop_solidity(skprops):
    return [skprops[fid].solidity for fid in skprops]


def extract_prop_major_axis_length(skprops):
    return [skprops[fid].major_axis_length for fid in skprops]


def extract_prop_minor_axis_length(skprops):
    return [skprops[fid].minor_axis_length for fid in skprops]


def extract_prop_morph_ori(skprops):
    return [skprops[fid].orientation * 180.0 / np.pi for fid in skprops]


def extract_prop_feret_diameter(skprops):
    return [skprops[fid].feret_diameter_max for fid in skprops]


def extract_prop_euler_number(skprops):
    return [skprops[fid].euler_number for fid in skprops]


def extract_prop_eccentricity(skprops):
    return [skprops[fid].eccentricity for fid in skprops]


def extract_prop_aspect_ratio(skprops, EPS=1e-10):
    result = []
    for fid in skprops:
        maj  = skprops[fid].major_axis_length
        min_ = skprops[fid].minor_axis_length
        result.append(np.inf if min_ <= EPS else maj / min_)
    return result


def extract_prop_compactness(area_list, perimeter_list, EPS=1e-10):
    result = []
    for area, perim in zip(area_list, perimeter_list):
        circle_area = perim ** 2 / (4.0 * np.pi)
        result.append(area / circle_area if circle_area >= EPS else 1)
    return result


def build_grain_props(skprops, prop_flags, locs_list=None, gblocs_list=None, EPS=1e-10):
    """Extract all flagged grain properties from skprops and pixel-location lists.

    Parameters
    ----------
    skprops : dict[int, RegionProperties]
        Ordered mapping of grain ID → skimage RegionProperties.
    prop_flags : dict[str, bool]
        Which properties to compute (same keys as ``self.prop_flag``).
    locs_list : list[numpy.ndarray] or None
        Per-grain pixel-location arrays (needed when ``prop_flags['npixels']`` is True).
    gblocs_list : list[numpy.ndarray] or None
        Per-grain boundary location arrays (needed for npixels_gb / gb_length_px).
    EPS : float
        Numerical guard for zero-denominator properties.

    Returns
    -------
    props : dict[str, list]
        Extracted property lists keyed by property name.
    """
    props = {}
    if prop_flags.get('npixels') and locs_list is not None:
        props['npixels'] = extract_prop_npixels(locs_list)
    if prop_flags.get('npixels_gb') and gblocs_list is not None:
        props['npixels_gb'] = extract_prop_gb_pixels(gblocs_list)
    if prop_flags.get('gb_length_px') and gblocs_list is not None:
        props['gb_length_px'] = extract_prop_gb_pixels(gblocs_list)
    if prop_flags.get('area'):
        props['area'] = extract_prop_area(skprops)
    if prop_flags.get('eq_diameter'):
        props['eq_diameter'] = extract_prop_eq_diameter(skprops)
    if prop_flags.get('perimeter'):
        props['perimeter'] = extract_prop_perimeter(skprops)
    if prop_flags.get('perimeter_crofton'):
        props['perimeter_crofton'] = extract_prop_perimeter_crofton(skprops)
    if prop_flags.get('compactness'):
        a = props.get('area') or extract_prop_area(skprops)
        p = props.get('perimeter') or extract_prop_perimeter(skprops)
        props['compactness'] = extract_prop_compactness(a, p, EPS)
    if prop_flags.get('aspect_ratio'):
        props['aspect_ratio'] = extract_prop_aspect_ratio(skprops, EPS)
    if prop_flags.get('solidity'):
        props['solidity'] = extract_prop_solidity(skprops)
    if prop_flags.get('major_axis_length'):
        props['major_axis_length'] = extract_prop_major_axis_length(skprops)
    if prop_flags.get('minor_axis_length'):
        props['minor_axis_length'] = extract_prop_minor_axis_length(skprops)
    if prop_flags.get('morph_ori'):
        props['morph_ori'] = extract_prop_morph_ori(skprops)
    if prop_flags.get('feret_diameter'):
        props['feret_diameter'] = extract_prop_feret_diameter(skprops)
    if prop_flags.get('euler_number'):
        props['euler_number'] = extract_prop_euler_number(skprops)
    if prop_flags.get('eccentricity'):
        props['eccentricity'] = extract_prop_eccentricity(skprops)
    return props