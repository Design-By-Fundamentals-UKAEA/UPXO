import numpy as np
from numba import njit, prange
import pyvista as pv
import cc3d

@njit(parallel=True)
def compute_gb_boundary_mask_interiorVoxels(lfi):
    """
    Compute a boolean mask of boundary voxels in the labeled feature image (LFI).
    A voxel is considered a boundary voxel if it has at least one 6-connected 
    neighbor with a different feature ID.

    Parameters
    ----------
    lfi : numpy.ndarray
        A 3D array of labeled feature IDs.

    Returns
    -------
    numpy.ndarray
        A boolean array of the same shape as `lfi`, where True indicates a boundary 
        voxel and False indicates an internal voxel.

    Import
    ------
    import upxo.gbops.grainBoundOps3d as gbOps
    """
    nx, ny, nz = lfi.shape
    boundary = np.zeros((nx, ny, nz), dtype=np.bool_)
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                fi = lfi[i, j, k]
                if (lfi[i+1, j, k] != fi or
                    lfi[i-1, j, k] != fi or
                    lfi[i, j+1, k] != fi or
                    lfi[i, j-1, k] != fi or
                    lfi[i, j, k+1] != fi or
                    lfi[i, j, k-1] != fi):  # 6-neighbours
                    boundary[i, j, k] = True
    return boundary

def compute_gb_boundary_mask(lfi):
    """
    Compute boundary mask including RVE exterior boundaries.

    Parameters
    ----------
    lfi : ndarray[int]
        3D grain ID array.

    Returns
    -------
    boundary_mask : ndarray[bool]
        Boolean mask of boundary voxels including outer RVE shell.
    """
    # Step 1 — interior boundary detection
    boundary = compute_gb_boundary_mask_interiorVoxels(lfi)
    nx, ny, nz = lfi.shape
    # Step 2 — enforce RVE outer shell boundaries
    boundary[0, :, :] = True
    boundary[nx-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, ny-1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, nz-1] = True
    return boundary

def identify_articulation_voxels(lfi):
    """
    Identifies voxels that violate manifold connectivity (6-connectivity) 
    within their 26-connectivity neighborhood.
    """
    nx, ny, nz = lfi.shape
    # Mask to store identified problem sites
    articulation_mask = np.zeros_like(lfi, dtype=bool)

    # 1. Focus only on boundary voxels to save compute time
    # A boundary voxel has at least one neighbor with a different ID.
    shifted_x = np.pad(lfi, ((1,0),(0,0),(0,0)), mode='edge')[:-1,:,:]
    boundary_mask = (lfi != shifted_x) # Simplified boundary detection
    
    # Iterate through the volume (excluding outer padding for safety)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                
                if not boundary_mask[i, j, k]:
                    continue
                
                # 2. Extract the 3x3x3 local stencil
                current_id = lfi[i, j, k]
                neighborhood = lfi[i-1:i+2, j-1:j+2, k-1:k+2]
                
                # 3. Create a binary mask of neighbors sharing the same ID
                # We exclude the center voxel itself to check if its neighbors 
                # stay connected without it.
                local_mask = (neighborhood == current_id).astype(np.uint8)
                local_mask[1, 1, 1] = 0 
                
                if np.sum(local_mask) <= 1:
                    # Isolated or single-neighbor voxels are manifold
                    continue
                
                # 4. Check connectivity of the local neighborhood
                # We use 6-connectivity (connectivity=6) to enforce manifoldedness.
                labels, n_components = cc3d.connected_components(
                    local_mask, 
                    connectivity=6, 
                    return_N=True
                )
                
                # 5. If neighbors are split into >1 component, this is a pinch point
                if n_components > 1:
                    articulation_mask[i, j, k] = True
                    
    return articulation_mask