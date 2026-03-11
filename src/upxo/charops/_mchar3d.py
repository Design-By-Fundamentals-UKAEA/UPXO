import numpy as np
import pandas as pd

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
    import upxo.charops._mchar3d as mchar3d

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 10, 10)
    >>> labeled_features, num_features = detect_features(data, connectivity=18)
    >>> print(f"Detected {num_features} features")
    """
    import upxo.gsdataops.grid_ops as gridOps
    lfi, N = gridOps.detect_grains_cc3d(mcStateArray, connectivity=connectivity,
                                        delta=delta)
    return lfi, N, connectivity

def get_voxel_counts_bbox_centroids(lfi_array):
    import cc3d
    return cc3d.statistics(lfi_array)

def get_feature_contact_surface_areas(lfi, connectivity=18, 
                                      ignore_junction_edges=False,
                                      voxx=1.0, voxy=1.0, voxz=1.0
                                      ):
    import cc3d
    # Compute the contact surface area between all labels.
    # Only face contacts are counted as edges and corners
    # have zero area. To get a simple count of all contacting
    # voxels, set `surface_area=False`.
    # { (1,2): 16 } aka { (label_1, label_2): contact surface area }
    sarea = cc3d.contacts(lfi, connectivity=connectivity,
                          surface_area=not ignore_junction_edges, 
                          anisotropy=(voxx,voxy,voxz))