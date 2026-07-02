"""
mchar — Morphological characterisation of labelled feature images
=================================================================

Public interface for detecting and characterising connected features
(grains) in 2D and 3D labelled images.  Delegates to the private
``_mchar2d`` and ``_mchar3d`` sub-modules.

Functions
---------
detect_features(mcStateArray, connectivity=18, delta=0)
    Detect and label connected features in a 3D state array using
    cc3d connectivity analysis.

characterise_features_in_image_2d(labelled_image, Xgrid, Ygrid, ...)
    Characterise every labelled feature in a 2D image by building a
    binary mask per feature (supports padded bounding boxes, physical
    coordinates, and scikit-image region properties).

characterise_features_in_image_v2(labelled_image, Xgrid=None, Ygrid=None, ...)
    Faster alternative to ``characterise_features_in_image_2d``.
    Uses a single ``skimage.measure.regionprops`` pass over the full
    image and returns both tight and padded bounding boxes.

Usage
-----
>>> import upxo.charops.mchar as mchar
>>> lfi, n_grains, conn = mchar.detect_features(state_array, connectivity=26)
>>> skp, bl, bl_ex, bx, bx_ex, coords = mchar.characterise_features_in_image_v2(lfi)
"""
from upxo.charops import _mchar2d, _mchar3d

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
    import upxo.charops.mchar as mchar
    Use as: mchar.detect_features

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10, 10)
    >>> labeled_features, num_features = detect_features(data, connectivity=18)
    >>> print(f"Detected {num_features} features")
    """
    lfi, N = _mchar3d.detect_features(mcStateArray, connectivity=connectivity, delta=delta)
    return lfi, N, connectivity

def characterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
                make_skprops=True, extract_coords=True, throw_bounding_box=True):
    """Characterise every labelled feature in a 2D image.

    For each unique label in ``labelled_image`` the function builds a binary
    mask, extracts an extended bounding-box crop (one pixel of padding on each
    open side), optionally computes scikit-image region properties on that
    crop, and optionally records the physical coordinates of every pixel.

    Parameters
    ----------
    labelled_image : numpy.ndarray of int, shape (R, C)
        Integer-labelled image where 0 is background and each positive integer
        identifies one feature (grain).
    Xgrid : numpy.ndarray of float, shape (R, C)
        Physical X-coordinate at every pixel position.
    Ygrid : numpy.ndarray of float, shape (R, C)
        Physical Y-coordinate at every pixel position.
    make_skprops : bool, default True
        Compute ``skimage.measure.regionprops`` on the extended bounding-box
        crop for each feature and store in ``skprops``.
    extract_coords : bool, default True
        Record the physical (X, Y) coordinates of every pixel belonging to
        each feature in ``coords_dict``.
    throw_bounding_box : bool, default True
        Populate ``bbox_limits_ex`` and ``bboxes_ex`` with the extended
        bounding-box extents and binary crops.

    Returns
    -------
    skprops : dict[int, skimage.measure.RegionProperties or None]
        Feature ID → scikit-image region-properties object computed on the
        extended bounding-box crop.  Values are ``None`` when
        ``make_skprops`` is False.
    bbox_limits_ex : dict[int, list[int] or None]
        Feature ID → ``[rmin_ex, rmax_ex, cmin_ex, cmax_ex]`` row/column
        slice indices of the one-pixel-padded bounding box.  Values are
        ``None`` when ``throw_bounding_box`` is False.
    bboxes_ex : dict[int, numpy.ndarray or None]
        Feature ID → binary uint8 crop of the extended bounding box.
        Values are ``None`` when ``throw_bounding_box`` is False.
    coords_dict : dict[int, numpy.ndarray or None]
        Feature ID → array of shape (N_pixels, 2) containing the physical
        ``[X, Y]`` coordinates of every pixel in the feature.  Values are
        ``None`` when ``extract_coords`` is False.

    Notes
    -----
    This wrapper delegates to ``upxo.charops._mchar2d.characterise_features_in_image_2d``.
    For a faster alternative that also returns tight (non-padded) bounding boxes,
    use :func:`characterise_features_in_image_v2`.

    Example
    -------
    >>> import upxo.charops.mchar as mchar
    >>> skprops, bbox_limits_ex, bboxes_ex, coords = mchar.characterise_features_in_image_2d(
    ...     labelled_image, Xgrid, Ygrid)
    """
    fx = _mchar2d.characterise_features_in_image_2d
    fxop = fx(labelled_image, Xgrid, Ygrid, make_skprops=make_skprops, 
              extract_coords=extract_coords, throw_bounding_box=throw_bounding_box)
    skprops, bbox_limits_ex, bboxes_ex, coords_dict = fxop
    return skprops, bbox_limits_ex, bboxes_ex, coords_dict
    
def characterise_features_in_image_v2(labelled_image, Xgrid=None, Ygrid=None,
                make_skprops=True, extract_coords=True,
                throw_bounding_box=True):
    """Characterise every labelled feature in a 2D image (faster, richer output).

    An improved alternative to :func:`characterise_features_in_image_2d` that
    calls ``skimage.measure.regionprops`` once on the full labelled image
    (avoiding per-feature mask construction) and returns both tight and
    one-pixel-padded bounding boxes.  ``Xgrid`` and ``Ygrid`` are optional;
    if omitted, pixel-index grids are generated automatically.

    Parameters
    ----------
    labelled_image : numpy.ndarray of int, shape (R, C)
        Integer-labelled image where 0 is background and each positive integer
        identifies one feature (grain).
    Xgrid : numpy.ndarray of float, shape (R, C) or None
        Physical X-coordinate at every pixel position.  When ``None``, the
        column index is used as the X coordinate.
    Ygrid : numpy.ndarray of float, shape (R, C) or None
        Physical Y-coordinate at every pixel position.  When ``None``, the
        row index is used as the Y coordinate.
    make_skprops : bool, default True
        Retain the ``skimage.measure.RegionProperties`` object for each feature
        in ``skprops``.
    extract_coords : bool, default True
        Record the physical (X, Y) coordinates of every pixel belonging to
        each feature in ``coords_dict``.
    throw_bounding_box : bool, default True
        Populate both tight and extended bounding-box outputs.

    Returns
    -------
    skprops : dict[int, skimage.measure.RegionProperties or None]
        Feature ID → region-properties object from the full-image
        ``regionprops`` call.  Values are ``None`` when ``make_skprops``
        is False.
    bbox_limits : dict[int, list[int] or None]
        Feature ID → ``[rmin, rmax, cmin, cmax]`` tight bounding-box slice
        indices (clamped to image boundaries).  Values are ``None`` when
        ``throw_bounding_box`` is False.
    bbox_limits_ex : dict[int, list[int] or None]
        Feature ID → ``[rmin_ex, rmax_ex, cmin_ex, cmax_ex]`` one-pixel-padded
        bounding-box slice indices.  Values are ``None`` when
        ``throw_bounding_box`` is False.
    bboxes : dict[int, numpy.ndarray or None]
        Feature ID → binary int32 crop of the tight bounding box.  Values are
        ``None`` when ``throw_bounding_box`` is False.
    bboxes_ex : dict[int, numpy.ndarray or None]
        Feature ID → binary int32 crop of the extended (padded) bounding box.
        Values are ``None`` when ``throw_bounding_box`` is False.
    coords_dict : dict[int, numpy.ndarray or None]
        Feature ID → array of shape (N_pixels, 2) containing the physical
        ``[X, Y]`` coordinates of every pixel in the feature.  Values are
        ``None`` when ``extract_coords`` is False.

    Notes
    -----
    This wrapper delegates to ``upxo.charops._mchar2d.characterise_features_in_image_v2``.
    Unlike v1, this version uses ``regionprops`` on the whole image in a single
    pass, which is considerably faster for images with many features.

    Example
    -------
    >>> import upxo.charops.mchar as mchar
    >>> skprops, bbox_lim, bbox_lim_ex, bboxes, bboxes_ex, coords = \\
    ...     mchar.characterise_features_in_image_v2(labelled_image)
    """
    fx = _mchar2d.characterise_features_in_image_v2
    fxop = fx(labelled_image, Xgrid, Ygrid, make_skprops=make_skprops,
              extract_coords=extract_coords, throw_bounding_box=throw_bounding_box)
    skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict = fxop
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
        Position category → array of grain IDs. Categories: 'top_left', 'top_right',
        'bottom_left', 'bottom_right', 'pure_top', 'pure_bottom', 'pure_left',
        'pure_right', 'top', 'bottom', 'left', 'right', 'boundary', 'corner', 'internal'.
    """
    return _mchar2d.classify_grain_positions_2d(lgi, gid)


def build_grain_props(skprops, prop_flags, locs_list=None, gblocs_list=None, EPS=1e-10):
    """Extract all flagged grain properties from skimage RegionProperties.

    Parameters
    ----------
    skprops : dict[int, skimage.measure.RegionProperties]
        Mapping of grain ID → skimage RegionProperties object.
    prop_flags : dict[str, bool]
        Which properties to compute (same keys as ``self.prop_flag``).
    locs_list : list[numpy.ndarray] or None
        Per-grain pixel-location arrays (needed when prop_flags['npixels'] is True).
    gblocs_list : list[numpy.ndarray] or None
        Per-grain boundary location arrays (needed for npixels_gb / gb_length_px).
    EPS : float
        Numerical guard for zero-denominator properties.

    Returns
    -------
    props : dict[str, list]
        Extracted property lists keyed by property name.
    """
    return _mchar2d.build_grain_props(skprops, prop_flags,
                                      locs_list=locs_list, gblocs_list=gblocs_list, EPS=EPS)


def extract_prop_area(skprops):
    return _mchar2d.extract_prop_area(skprops)

def extract_prop_eq_diameter(skprops):
    return _mchar2d.extract_prop_eq_diameter(skprops)

def extract_prop_perimeter(skprops):
    return _mchar2d.extract_prop_perimeter(skprops)

def extract_prop_perimeter_crofton(skprops):
    return _mchar2d.extract_prop_perimeter_crofton(skprops)

def extract_prop_solidity(skprops):
    return _mchar2d.extract_prop_solidity(skprops)

def extract_prop_major_axis_length(skprops):
    return _mchar2d.extract_prop_major_axis_length(skprops)

def extract_prop_minor_axis_length(skprops):
    return _mchar2d.extract_prop_minor_axis_length(skprops)

def extract_prop_morph_ori(skprops):
    return _mchar2d.extract_prop_morph_ori(skprops)

def extract_prop_feret_diameter(skprops):
    return _mchar2d.extract_prop_feret_diameter(skprops)

def extract_prop_euler_number(skprops):
    return _mchar2d.extract_prop_euler_number(skprops)

def extract_prop_eccentricity(skprops):
    return _mchar2d.extract_prop_eccentricity(skprops)

def extract_prop_aspect_ratio(skprops, EPS=1e-10):
    return _mchar2d.extract_prop_aspect_ratio(skprops, EPS=EPS)

def extract_prop_compactness(area_list, perimeter_list, EPS=1e-10):
    return _mchar2d.extract_prop_compactness(area_list, perimeter_list, EPS=EPS)

def extract_prop_npixels(locs_list):
    return _mchar2d.extract_prop_npixels(locs_list)

def extract_prop_gb_pixels(gblocs_list):
    return _mchar2d.extract_prop_gb_pixels(gblocs_list)

def get_grain_size_distribution_from_slice(lgi_2d):
    """
    Extract grain area (pixel count) distribution from a 2D labelled
    grain image.

    Typically called on a 2D slice extracted from a 3D grain structure
    (via :func:`~upxo.gsdataops.grid_ops.section_from_3d`) to compare
    grain size distributions against an EBSD reference for
    representativeness assessment.

    Parameters
    ----------
    lgi_2d : ndarray (ny, nx), int
        2D labelled grain image; positive values are grain IDs,
        values <= 0 are ignored.

    Returns
    -------
    grain_areas : ndarray
        1-D array of grain pixel counts (one entry per unique positive
        grain ID found in *lgi_2d*).
    """
    import numpy as np
    grain_ids = np.unique(lgi_2d)
    grain_ids = grain_ids[grain_ids > 0]
    return np.array([int(np.sum(lgi_2d == gid)) for gid in grain_ids])
