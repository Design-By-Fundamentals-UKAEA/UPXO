import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import upxo.gsdataops.gid_ops as gidOps

"""
Property Operations
-------------------
This module contains functions for calculating morphological properties of
features in a labeled feature ID (LFI) array.

Import
------
import upxo.propOps.mpropOps as mpropOps
"""

def get_feature_volumes(lfi):
    return np.bincount(lfi.ravel())[1:]

def extract_feature_volumes(lfi):
    feature_volumes = get_feature_volumes(lfi)
    fids = np.arange(1, len(feature_volumes)+1)
    return fids, feature_volumes

def find_ratio_bfeat_intfeat_volumes(lfi):
    """
    Calculate the ratio of boundary feature volume to internal feature volume.

    Parameters
    ----------
    lfi : ndarray
        A 3D array of labeled feature IDs.

    Returns
    -------
    float
        The ratio of boundary feature volume to internal feature volume.
    
    Import
    ------
    import upxo.propOps.mpropOps as mpropOps
    Use as: mpropOps.find_ratio_bfeat_intfeat_volumes
    """
    internal_features = gidOps.find_internal_fids3d(lfi)
    boundary_features = gidOps.find_boundary_fids3d(lfi)
    fids, feature_volumes = extract_feature_volumes(lfi)
    internal_volume = feature_volumes[internal_features-1].sum()
    boundary_volume = feature_volumes[boundary_features-1].sum()
    volRatio = boundary_volume / internal_volume
    return volRatio

def fit_ellipsoid_jekel(points):
    """Source: Charles Jekel (2020) https://jekel.me/2020/Least-Squares-Ellipsoid-Fit/

    Import
    ------
    import upxo.propOps.mpropOps as mpropOps
    Use as: mpropOps.fit_ellipsoid_jekel
    """
    X = np.hstack([points**2, 2.0 * points[:, 0:1] * points[:, 1:2], 2.0 * points[:, 0:1] * points[:, 2:3],
                   2.0 * points[:, 1:2] * points[:, 2:3], 2.0 * points, np.ones((len(points), 1))])
    _, _, V = np.linalg.svd(X)
    v = V[-1, :]
    A = np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])
    b, c = v[6:9], v[9]
    center = -np.linalg.solve(A, b)
    evals, evecs = np.linalg.eigh(A)
    K = 1.0 / (np.dot(b, np.linalg.solve(A, b)) - c)
    radii = np.sqrt(1.0 / (np.abs(evals) * K)) # abs to handle numerical jitter
    return center, radii, evecs

def get_neighborhood_signature(target_fid, neigh_fids, dna, n_order=1):
    """Retrieves metadata-based signature for a specific grain cluster.

    Import
    ------
    import upxo.propOps.mpropOps as mpropOps
    Use as: mpropOps.get_neighborhood_signature
    """
    cluster = gidOps.get_nth_order_neighbors(target_fid, neigh_fids, n_order)
    # Filter valid IDs and compute mean AR of the local neighborhood
    valid_fids = [f for f in cluster if f in dna and dna[f]['valid']]
    if not valid_fids: return 1.0, 0.0 # Fallback
    mean_ar = np.mean([dna[f]['AR'] for f in valid_fids])
    total_vol = np.sum([dna[f]['Vol'] for f in valid_fids])
    return mean_ar, total_vol

def analyze_grain_shapes(lfi, bboxes):
    """Tier 0: DNA pre-calc with ID-1 volume indexing for IDs starting at 1.

    Import
    ------
    import upxo.propOps.mpropOps as mpropOps
    Use as: mpropOps.analyze_grain_shapes
    """
    dna = {}
    # Slicing [1:] aligns array index with (Grain ID - 1)
    npixels = np.bincount(lfi.ravel().astype(np.int64))[1:]
    for fid, bbox in bboxes.items():
        if fid == 0: continue
        mask = (lfi[bbox] == fid)
        skin = mask ^ ndimage.binary_erosion(mask)
        pts = np.argwhere(skin) + np.array([s.start for s in bbox])
        # Direct lookup using fid-1 offset
        vol = npixels[fid-1] if (fid-1) < len(npixels) else np.sum(mask)
        dna[fid] = {'Vol': vol, 'AR': 1.0, 'R': np.eye(3), 'valid': False}
        if len(pts) >= 12:
            try:
                ctr, rad, rot = fit_ellipsoid_jekel(pts)
                r_s = np.sort(rad)
                dna[fid].update({'AR': r_s[-1]/r_s[0], 'R': rot, 'Centroid': ctr, 'valid': True})
            except: pass
    return dna

