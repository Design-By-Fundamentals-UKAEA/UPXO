import numpy as np
import cc3d
import upxo.gsdataops.gid_ops as gidOps
# import upxo.gsdataops.grid_ops as gridOps
import upxo.propOps.mpropOps as mpropOps

def preManifold_clean_report(lfi, message='Report: \n'):
    """
    Report on the current state of the LFI, including:
    - Total number of voxels
    - Smallest feature size and count
    - Number of small features (<=5 voxels)
    - Largest feature size
    - Number of islands features
    - Boundary to internal feature volume ratio

    Parameters:
    lfi (numpy.ndarray): The labeled feature image to analyze.
    message (str): A custom message to include in the report.

    Import
    ------
    import upxo.uiOps.outputDisplay as opDisp
    """
    print(40*'=', '\n', message)

    print(f"Total number of voxels: {lfi.size}")

    featSizes = mpropOps.get_feature_volumes(lfi)

    minSize = featSizes.min()
    print(f"Smallest feature size: {minSize}, count: {np.count_nonzero(featSizes==minSize)}")

    smallfids = gidOps.find_small_fids(lfi, 5)
    print(f"Number of small features (<=5 voxels): {len(smallfids)}")

    maxSize = featSizes.max()
    print(f"Largest feature size: {maxSize}")

    islands = gidOps.detect_islands(gidOps.find_neighs3d(lfi, 6))
    print(f"Number of islands features: {len(islands)}")

    volRatio = mpropOps.find_ratio_bfeat_intfeat_volumes(lfi)
    print(f"Boundary to internal feature volume ratio: {volRatio:.4f}")

    print(40*'=')