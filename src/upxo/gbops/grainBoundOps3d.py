"""
Module: grainBoundOps3d
-----------------------
Grain-boundary operations for 3D labelled voxel structures.

Provides fast, Numba-accelerated routines for detecting grain-boundary voxels
in 3-D labelled feature images (LFI) and for identifying topologically
problematic voxels (articulation points) that violate manifold connectivity.

Functions
---------
compute_gb_boundary_mask_interiorVoxels(lfi)
    Detect interior boundary voxels using 6-connectivity (Numba parallel).
compute_gb_boundary_mask(lfi)
    Extend interior boundary mask to include the full RVE outer shell.
identify_articulation_voxels(lfi)
    Flag voxels whose removal would disconnect their local grain neighbourhood.

Usage
-----
::

    import upxo.gbops.grainBoundOps3d as gbOps

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
from numba import njit, prange
import pyvista as pv
import cc3d


@njit(parallel=True)
def compute_gb_boundary_mask_interiorVoxels(lfi):
    """Detect interior grain-boundary voxels using 6-connectivity.

    A voxel is classified as a boundary voxel when at least one of its six
    face-adjacent neighbours (±x, ±y, ±z) carries a different grain ID.
    Voxels on the outermost shell of the array are excluded; use
    :func:`compute_gb_boundary_mask` to include those.

    The loop over the x-axis is parallelised with ``prange``; inner loops
    remain serial so that Numba can apply vectorisation.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (nx, ny, nz)
        3-D array of integer grain (feature) IDs.

    Returns
    -------
    boundary : numpy.ndarray of bool, shape (nx, ny, nz)
        Boolean mask where ``True`` marks a grain-boundary voxel.
        The outermost shell (index 0 and n-1 in each axis) is always
        ``False`` — use :func:`compute_gb_boundary_mask` to set it.

    Notes
    -----
    Decorated with ``@njit(parallel=True)``: the function is JIT-compiled by
    Numba on the first call and cached for subsequent calls.  It is called
    internally by :func:`compute_gb_boundary_mask`; direct use is only needed
    when the RVE outer faces should be excluded from the boundary mask.

    Usage
    -----
    ::

        import upxo.gbops.grainBoundOps3d as gbOps

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import upxo.gbops.grainBoundOps3d as gbOps

        lfi = np.zeros((10, 10, 10), dtype=np.int32)
        lfi[5:, :, :] = 1          # two-grain RVE split at x=5
        mask = gbOps.compute_gb_boundary_mask_interiorVoxels(lfi)
        # voxels at x=4 and x=5 that face the split are True
        print(mask[4, 5, 5], mask[5, 5, 5])  # True True
    """
    nx, ny, nz = lfi.shape
    boundary = np.zeros((nx, ny, nz), dtype=np.bool_)
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                fi = lfi[i, j, k]
                if (lfi[i + 1, j, k] != fi or
                        lfi[i - 1, j, k] != fi or
                        lfi[i, j + 1, k] != fi or
                        lfi[i, j - 1, k] != fi or
                        lfi[i, j, k + 1] != fi or
                        lfi[i, j, k - 1] != fi):
                    boundary[i, j, k] = True
    return boundary


def compute_gb_boundary_mask(lfi):
    """Compute a grain-boundary mask that includes the RVE outer shell.

    Calls :func:`compute_gb_boundary_mask_interiorVoxels` to detect all
    interior boundary voxels (6-connectivity criterion), then sets every voxel
    on the six outer faces of the bounding box to ``True``.  This represents
    the physical convention that grains touching the RVE surface are bounded
    by that surface.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (nx, ny, nz)
        3-D array of integer grain (feature) IDs.

    Returns
    -------
    boundary : numpy.ndarray of bool, shape (nx, ny, nz)
        Boolean mask where ``True`` marks either an interior grain-boundary
        voxel or a voxel on the outer RVE shell.

    Notes
    -----
    The outer shell assignment is O(n²) per face and does not require Numba.
    The dominant cost is the parallel interior scan in
    :func:`compute_gb_boundary_mask_interiorVoxels`.

    Usage
    -----
    ::

        import upxo.gbops.grainBoundOps3d as gbOps

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import upxo.gbops.grainBoundOps3d as gbOps

        lfi = np.zeros((10, 10, 10), dtype=np.int32)
        lfi[5:, :, :] = 1
        mask = gbOps.compute_gb_boundary_mask(lfi)
        # All six outer faces are True
        assert mask[0, :, :].all()
        assert mask[-1, :, :].all()
        # Interior boundary plane is also True
        assert mask[4, 5, 5] and mask[5, 5, 5]
    """
    boundary = compute_gb_boundary_mask_interiorVoxels(lfi)
    nx, ny, nz = lfi.shape
    boundary[0, :, :] = True
    boundary[nx - 1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, ny - 1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, nz - 1] = True
    return boundary


def identify_articulation_voxels(lfi):
    """Identify voxels whose removal disconnects their local grain neighbourhood.

    For every boundary voxel, the function extracts the 3×3×3 neighbourhood
    and checks whether the same-grain voxels within that stencil remain
    6-connected after the centre voxel is excluded.  If they split into more
    than one connected component the centre voxel is an *articulation point*:
    removing it (e.g. during surface reconstruction or mesh smoothing) would
    create a topological hole in the grain.

    Parameters
    ----------
    lfi : numpy.ndarray of int, shape (nx, ny, nz)
        3-D array of integer grain (feature) IDs.

    Returns
    -------
    articulation_mask : numpy.ndarray of bool, shape (nx, ny, nz)
        Boolean mask where ``True`` marks a voxel whose removal would
        disconnect the local 26-neighbourhood same-grain region under
        6-connectivity.

    Notes
    -----
    Complexity is O(N_boundary) × O(1) per voxel (the 3×3×3 stencil is
    constant size), so runtime scales with the total grain-boundary area
    rather than the full volume.

    A simplified x-shift is used for the initial boundary pre-filter: only
    voxels where ``lfi[i,j,k] != lfi[i-1,j,k]`` are considered.  This may
    miss some boundary voxels detected by the full 6-neighbour check in
    :func:`compute_gb_boundary_mask_interiorVoxels`, but reduces unnecessary
    6-connectivity tests on clearly interior voxels.

    The connected-component labelling is performed by ``cc3d.connected_components``
    with ``connectivity=6``.

    Usage
    -----
    ::

        import upxo.gbops.grainBoundOps3d as gbOps

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import upxo.gbops.grainBoundOps3d as gbOps

        lfi = np.zeros((20, 20, 20), dtype=np.int32)
        lfi[10:, :, :] = 1
        art = gbOps.identify_articulation_voxels(lfi)
        # For a clean planar boundary no articulation voxels should be present
        print("Articulation voxels found:", art.sum())

    See Also
    --------
    compute_gb_boundary_mask : Identify all grain-boundary voxels.
    """
    nx, ny, nz = lfi.shape
    articulation_mask = np.zeros_like(lfi, dtype=bool)

    shifted_x = np.pad(lfi, ((1, 0), (0, 0), (0, 0)), mode='edge')[:-1, :, :]
    boundary_mask = (lfi != shifted_x)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if not boundary_mask[i, j, k]:
                    continue

                current_id = lfi[i, j, k]
                neighborhood = lfi[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]
                local_mask = (neighborhood == current_id).astype(np.uint8)
                local_mask[1, 1, 1] = 0

                if np.sum(local_mask) <= 1:
                    continue

                labels, n_components = cc3d.connected_components(
                    local_mask,
                    connectivity=6,
                    return_N=True,
                )

                if n_components > 1:
                    articulation_mask[i, j, k] = True

    return articulation_mask
