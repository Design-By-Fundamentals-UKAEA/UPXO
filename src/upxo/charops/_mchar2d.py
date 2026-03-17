import numpy as np
# import pandas as pd
import cv2

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

def charecterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
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
    skprops, bbox_limits, bboxes_ex, coords_dict = ...charecterise_features_in_image_2d(labelled_image, Xgrid, Ygrid,
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
        _, L = cv2.connectedComponents(np.array(labelled_image == fid,
                                                dtype=np.uint8))
        loc_ = np.where(L == 1)
        rmin, rmax = loc_[0].min(), loc_[0].max()+1
        cmin, cmax = loc_[1].min(), loc_[1].max()+1
        Rlab, Clab = L.shape
        rmin_ex, rmax_ex = rmin-int(rmin != 0), rmax+int(rmin != Rlab)
        cmin_ex, cmax_ex = cmin-int(cmin != 0), cmax+int(cmax != Clab)
        bbox_ex = np.array(L[rmin_ex:rmax_ex, cmin_ex:cmax_ex], dtype=np.uint8)
        if throw_bounding_box:
            bbox_limits_ex[fid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
            bboxes_ex[fid] = bbox_ex
        if extract_coords:
            coords_dict[fid] = np.array([[Xgrid[ij[0], ij[1]], Ygrid[ij[0], ij[1]]]
                            for ij in np.argwhere(L == 1)])
        if make_skprops:
            skprops[fid] = regionprops(bbox_ex, cache=False)[0]
    return skprops, bbox_limits_ex, bboxes_ex, coords_dict

def charecterise_features_in_image_v2(labelled_image, Xgrid=None, Ygrid=None, 
                                    make_skprops=True, extract_coords=True, 
                                    throw_bounding_box=True):
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