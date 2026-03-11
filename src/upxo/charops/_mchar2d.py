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