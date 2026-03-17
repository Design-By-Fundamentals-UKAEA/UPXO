"""
Module: mchar
----------------
This module provides functions for morphological character analysis in 2D and 3D image data. It includes feature detection and characterization based on connectivity criteria.
Functions---------
- detect_features(mcStateArray, connectivity=18, delta=0): Detects and labels connected features in 3D image data based on specified connectivity and delta threshold.
- charecterise_features_in_image_2d(labelled_image, Xgrid, Ygrid, make_skprops=True, extract_coords=True, throw_bounding_box=True): Characterizes features in a 2D labeled image, extracting properties and coordinates.
- charecterise_features_in_image_v2(labelled_image, Xgrid, Ygrid, make_skprops=True, extract_coords=True, throw_bounding_box=True): An alternative version of the feature characterization function for 2D images.
Import
------
import upxo.charops._mchar2d as _mchar2d
import upxo.charops._mchar3d as _mchar3d
import upxo.charops.mchar as mchar
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

    Import
    ------
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

def charecterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
                make_skprops=True, extract_coords=True, throw_bounding_box=True):
    """
    Import
    ------
    import upxo.charops.mchar as mchar
    Use as: mchar.charecterise_features_in_image_2d
    """
    fx = _mchar2d.charecterise_features_in_image_2d
    fxop = fx(labelled_image, Xgrid, Ygrid, make_skprops=make_skprops, 
              extract_coords=extract_coords, throw_bounding_box=throw_bounding_box)
    skprops, bbox_limits_ex, bboxes_ex, coords_dict = fxop
    return skprops, bbox_limits_ex, bboxes_ex, coords_dict
    
def charecterise_features_in_image_v2(labelled_image, Xgrid, Ygrid,
                make_skprops=True, extract_coords=True,
                throw_bounding_box=True):
    """
    Import
    ------
    import upxo.charops.mchar as mchar
    Use as: mchar.charecterise_features_in_image_v2
    """
    fx = _mchar2d.charecterise_features_in_image_v2
    fxop = fx(labelled_image, Xgrid, Ygrid, make_skprops=make_skprops,
              extract_coords=extract_coords, throw_bounding_box=throw_bounding_box)
    skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict = fxop
    return skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict