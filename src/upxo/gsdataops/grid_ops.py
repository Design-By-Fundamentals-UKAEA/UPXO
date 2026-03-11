import numpy as np
from numba import njit, prange
from copy import deepcopy
from collections import deque
import upxo._sup.decorators as decorators

# ###########################################################################
# ###########################################################################
#                         SECTIONING OPERATIPONS
#         Search ID:      SID_3dSectioningOps
# ###########################################################################
# ###########################################################################

def section_from_3d(db, axis=0, location=0):
        """
        Extract a 2D section from a 3D grid along a specified axis.

        Parameters
        ----------
        db : ndarray
            3D input array from which to extract the section.
        axis : int, optional
            Axis along which to take the section (0, 1, or 2). Default is 0.
        location : int, optional
            Location index along the specified axis to extract the section. Default is 0.
        """
        if axis == 0:
            section = db[location, :, :]
        elif axis == 1:
            section = db[:, location, :]
        elif axis == 2:
            section = db[:, :, location]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        
        return section

def build_pvgrid(data=None, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0)):
    import pyvista as pv
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape)+1
    grid.origin = origin
    grid.spacing = spacing
    grid.cell_data["values"] = data.flatten(order="F")
    return grid

def _interpolate_grid_2d(data, new_h, new_w, method='nearest'):
    """
    Core interpolation function for 2D grid resampling.

    Parameters
    ----------
    data : ndarray
        2D input array to be resampled.
    new_h : int
        New height (number of rows).
    new_w : int
        New width (number of columns).
    method : str, optional
        Interpolation method. Options are 'nearest', 'linear', etc.
        Default is 'nearest'.

    Returns
    -------
    ndarray
        Resampled 2D array with shape (new_h, new_w), maintaining original dtype.
    """
    from scipy.interpolate import RegularGridInterpolator
    h, w = data.shape
    x_old, y_old = np.linspace(0, 1, h), np.linspace(0, 1, w)
    interp = RegularGridInterpolator((x_old, y_old), data, method=method, bounds_error=False, fill_value=None)
    x_new, y_new = np.linspace(0, 1, new_h), np.linspace(0, 1, new_w)
    X, Y = np.meshgrid(x_new, y_new, indexing='ij')
    pts = np.array([X.ravel(), Y.ravel()]).T
    return interp(pts).reshape(new_h, new_w).astype(data.dtype)

def mask_featIDImg_at_coords(featIDImg, bsegCoords, local_seg_ids, 
                             featName='fbseg', maskDType=np.int32):
    """
    Example
    -------
    mask_featIDImg_at_coords(lgi_boundaries, bsegCoords, local_seg_ids,
                        featName='fbseg', maskDType=np.int32)
    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.mask_featIDImg_at_coords
    """
    segID_masked_lfi = np.asarray(deepcopy(featIDImg), dtype=np.float32)
    for cg, ngcoords in bsegCoords.items():
        for i, (ng, ngsegcoords) in enumerate(ngcoords.items()):
            segid = local_seg_ids[cg][i]
            for sc in ngsegcoords:
                segID_masked_lfi[sc[0]][sc[1]] = segid
    return segID_masked_lfi

# ###########################################################################
# ###########################################################################
#-----------------------  BOUNDING BOX OPERATIONS  --------------------------
#         Search ID:      SID_BBoxOps
# ###########################################################################
# ###########################################################################

def find_feature_extended_bbox_pix(fid=None, lfi=None, make_binary=False):
    """
    Find the extended bounded box of a given fid in a 2D local feature ID array.

    Parameters
    ----------
    fid : int
        Feature ID for which to find the extended bounding box. Feature 
        ID is generally the grain ID. Default is None.
    lfi : np.ndarray
        2D array of local feature IDs (grain IDs). Default is None.
    make_binary : bool, optional
        If True, the returned bounding box array is a binary mask. Default is False.

    Returns
    -------
    feat_lfi_ExtBBox : np.ndarray
        Extended bounding box of the specified feature ID.

    Example
    -------
    from upxo.ggrowth.mcgs import mcgs
    pxtal = mcgs(study='independent',
                    input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
    pxtal.simulate()
    pxtal.detect_grains()
    pxtal.gs[16].find_extended_bounding_box(10)

    Notes
    -----
    Applicable only for 2D grain structures of pixellated type.

    The extended bounding box adds a one-pixel margin around the feature's
    actual bounding box, taking care to handle edge cases where the feature
    touches the edges of the array. You could also use this function independently
    to find the extended bounding box of any feature in a 2D local feature ID array.
    For example, youcould use this to find the extended bounding box of a given 
    grain boundary segment ID in a grain boundary segment local feature ID array. You
    could use either local or global feature IDs for this purpose, but make sure to 
    use the right lfi array.

    Comments
    --------
    The line `pxtal.gs[16].find_extended_bounding_box(fid)` is an orchestrator 
    function call in the temporal slice object. The present function is called 
    internally by that function.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.find_feature_extended_bbox_pix
    """
    xmax, ymax = lfi.shape
    loc = np.where(lfi == fid)
    if len(loc[0]) == 0:
        return None
    xi, xj = loc[0].min(), loc[0].max()
    yi, yj = loc[1].min(), loc[1].max()
    feat_lfi_ExtBBox = lfi[max(0, xi-1):min(xmax, xj+2), max(0, yi-1):min(ymax, yj+2)]
    if make_binary:
        feat_lfi_ExtBBox = (feat_lfi_ExtBBox == fid).astype(np.int32)
    return feat_lfi_ExtBBox

def find_extended_bbox_pix_fids(fids=None, lfi=None, make_binary=False):
    """
    Find the extended bounded box of a given fid in a 2D local feature ID array.

    Parameters
    ----------
    fid : int
        Feature ID for which to find the extended bounding box. Feature 
        ID is generally the grain ID. Default is None.
    lfi : np.ndarray
        2D array of local feature IDs (grain IDs). Default is None.
    make_binary : bool, optional
        If True, the returned bounding box arrays are binary masks. Default is False.

    Returns
    -------
    bboxes : dict
        Dictionary with feature IDs as keys and their extended bounding boxes as values.

    Notes
    -----
    Refer to find_feature_extended_bbox_pix for detailed documentation.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.find_extended_bbox_pix_fids
    """
    fids = [fids] if type(fids) is int else fids
    xmax, ymax = lfi.shape
    bboxes = {}
    for fid in fids:
        loc = np.where(lfi == fid)
        if len(loc[0]) == 0:
            continue
        xi, xj = loc[0].min(), loc[0].max()
        yi, yj = loc[1].min(), loc[1].max()
        bbox = lfi[max(0, xi-1):min(xmax, xj+2), max(0, yi-1):min(ymax, yj+2)]
        if make_binary:
            bbox = (bbox == fid).astype(np.int32)
        bboxes[fid] = bbox
    return bboxes

@decorators.port_doc(module_path='upxo.gsdataops.grid_ops', func_name='find_extended_bbox_pix_fids')
def find_extended_bounding_box_all_grains(fids=None, lfi=None, make_binary=False):
    grain_lgi_ex_all = find_extended_bbox_pix_fids(fids=fids, lfi=lfi, make_binary=make_binary)
    return grain_lgi_ex_all

# ###########################################################################
# ###########################################################################
# ###########################################################################
# ###########################################################################

def rescale_grid_2d(data, scale_factor, method='nearest'):
    """
    Rescale a 2D grid by a uniform scale factor using interpolation.

    Parameters
    ----------
    data : ndarray
        2D input array to be rescaled.
    scale_factor : float
        Uniform scaling factor to apply to both dimensions. Values > 1 enlarge
        the grid, values < 1 shrink it.
    method : str, optional
        Interpolation method to use. Options are 'nearest', 'linear', etc.
        Default is 'nearest'.

    Returns
    -------
    ndarray
        Rescaled 2D array with shape determined by scale_factor, maintaining
        the original data type.
    
    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    """
    h, w = data.shape
    new_h, new_w = int(round(h * scale_factor)), int(round(w * scale_factor))
    return _interpolate_grid_2d(data, new_h, new_w, method)

def stretch_grid_2d(data, stretch_x=1.0, stretch_y=1.0, px_size_x=1.0, px_size_y=1.0, method='nearest'):
    """
    Stretch a 2D grid by different factors in each dimension while accounting
    for pixel size and physical dimensions.

    Parameters
    ----------
    data : ndarray
        2D input array to be stretched.
    stretch_x : float, optional
        Stretch factor in the x (width) dimension. Default is 1.0 (no stretch).
    stretch_y : float, optional
        Stretch factor in the y (height) dimension. Default is 1.0 (no stretch).
    px_size_x : float, optional
        Physical pixel size in the x dimension. Default is 1.0.
    px_size_y : float, optional
        Physical pixel size in the y dimension. Default is 1.0.
    method : str, optional
        Interpolation method to use. Options are 'nearest', 'linear', etc.
        Default is 'nearest'.

    Returns
    -------
    stretched_data : ndarray
        Stretched 2D array maintaining the original data type.
    px_size_x : float
        Physical pixel size in the x dimension (unchanged).
    px_size_y : float
        Physical pixel size in the y dimension (unchanged).
    new_shape : tuple
        New shape of the stretched data as (new_h, new_w).

    Notes
    -----
    The stretch is applied to the physical dimensions of the grid, with final
    grid size determined by dividing new physical dimensions by pixel sizes.
    """
    h, w = data.shape
    new_physical_x, new_physical_y = w*px_size_x*stretch_x, h*px_size_y*stretch_y
    new_w, new_h = int(round(new_physical_x / px_size_x)), int(round(new_physical_y / px_size_y))
    stretched_data = _interpolate_grid_2d(data, new_h, new_w, method)
    return stretched_data, px_size_x, px_size_y, (new_h, new_w)

def resample_grid_2d(data, uigrid, sf=0.5, method='nearest'):
    """
    Resample a 2D array at every nth row and column.
    
    Parameters
    ----------
    data : np.ndarray
        2D state matrix to resample
    uigrid : object
        Grid configuration object with attributes: xmin, xmax, xinc, ymin, ymax, yinc
    sf : float
        Sampling factor (0 < sf <= 1). sf=0.5 means every 2nd pixel, sf=0.33 means every 3rd pixel
    method : str, optional
        Interpolation method ('nearest', 'linear', etc.). Default is 'nearest'.
    
    Returns
    -------
    resampled_data : np.ndarray
        Resampled 2D state matrix
    x_new : np.ndarray
        New x grid coordinates
    y_new : np.ndarray
        New y grid coordinates
    xinc_new : float
        New x increment
    yinc_new : float
        New y increment
    """
    from scipy.interpolate import RegularGridInterpolator
    xgr = np.arange(uigrid.xmin, uigrid.xmax+uigrid.xinc, uigrid.xinc)
    ygr = np.arange(uigrid.ymin, uigrid.ymax+uigrid.yinc, uigrid.yinc)
    xinc_new, yinc_new = uigrid.xinc/sf, uigrid.yinc/sf
    x_new = np.arange(uigrid.xmin, uigrid.xmax + xinc_new, xinc_new)
    y_new = np.arange(uigrid.ymin, uigrid.ymax + yinc_new, yinc_new)
    # Ensure we don't exceed original bounds
    x_new, y_new = x_new[x_new <= uigrid.xmax], y_new[y_new <= uigrid.ymax]
    interp = RegularGridInterpolator((ygr, xgr), data, method=method, bounds_error=False, fill_value=None)
    Y_new, X_new = np.meshgrid(y_new, x_new, indexing='ij')
    pts = np.array([Y_new.ravel(), X_new.ravel()]).T
    resampled_data = interp(pts).reshape(len(y_new), len(x_new)).astype(data.dtype)
    return resampled_data, x_new, y_new, xinc_new, yinc_new

import numpy as np

def _interpolate_grid_3d(data, new_d, new_h, new_w, method='nearest'):
    """
    Core interpolation function for 3D grid resampling.

    Parameters
    ----------
    data : ndarray
        3D input array to be resampled.
    new_d : int
        New depth (number of slices).
    new_h : int
        New height (number of rows).
    new_w : int
        New width (number of columns).
    method : str, optional
        Interpolation method. Options: 'nearest', 'linear', etc.
        Default is 'nearest'.

    Returns
    -------
    ndarray
        Resampled 3D array with shape (new_d, new_h, new_w),
        maintaining original dtype.
    """
    from scipy.interpolate import RegularGridInterpolator
    d, h, w = data.shape
    x_old = np.linspace(0, 1, d)
    y_old = np.linspace(0, 1, h)
    z_old = np.linspace(0, 1, w)
    interp = RegularGridInterpolator((x_old, y_old, z_old),
        data, method=method, bounds_error=False, fill_value=None)
    x_new = np.linspace(0, 1, new_d)
    y_new = np.linspace(0, 1, new_h)
    z_new = np.linspace(0, 1, new_w)
    X, Y, Z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    return interp(pts).reshape(new_d, new_h, new_w).astype(data.dtype)

def rescale_grid_3d(data, scale_factor, method='nearest'):
    """
    Rescale a 3D grid by a uniform scale factor using interpolation.

    Parameters
    ----------
    data : ndarray
        3D input array to be rescaled.
    scale_factor : float
        Uniform scaling factor applied to all three dimensions.
        Values > 1 enlarge, values < 1 shrink.
    method : str, optional
        Interpolation method. Options: 'nearest', 'linear', etc.
        Default is 'nearest'.

    Returns
    -------
    ndarray
        Rescaled 3D array with shape determined by scale_factor,
        maintaining original dtype.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps

    Call
    ----
    gridOps.rescale_grid_3d(data, scale_factor, method)
    """
    d, h, w = data.shape
    new_d = int(round(d*scale_factor))
    new_h = int(round(h*scale_factor))
    new_w = int(round(w*scale_factor))
    return _interpolate_grid_3d(data, new_d, new_h, new_w, method)

def detect_grains_cc3d(image_data, connectivity=18, delta=0, return_num_grains=True,
                       verbose=False):
    """
    THe following documentation taken verbatim from 
    https://seunglab.org/connected-components-3d/ to help UPXO users.

    Only a littlbit of doicumentation has been provided below. For a comprehensive
    documentation, the user is referred to the above link.

    # Only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    # By default, cc3d works on multivalued labelings

    If you need the borders to wrap around specify periodic_boundary=True,
    currently only supported for 4 and 8 (2d) and 6 (3d) connectivities.

    If you're working with continuously valued images like microscopy
    images you can use cc3d to perform a very rough segmentation.
    If delta = 0, standard high speed processing. If delta > 0, then
    neighbor voxel values <= delta are considered the same component.
    The algorithm can be 2-10x slower though. Zero is considered
    background and will not join to any other voxel.

    If you're working with an image that's larger than memory you can
    use mmapped files. The input and output files can be used independently.
    In this case an array labels.bin that is 5000x5000x2000 voxels and uint32_t
    in Fortran order is computed and the results are written to out.bin in Fortran
    order. You can find the properties of the file (shape, dtype, order) by inspecting
    labels_out.
    labels_in = np.memmap("labels.bin", order="F", dtype=np.uint32, shape=(5000, 5000, 2000))
    labels_out = cc3d.connected_components(labels_in, out_file="out.bin")

    Here's another strategy that you can use for huge files that won't even
    take up any disk space. Provide any iterator to this function that produces
    thick z sections of the input array that are in sequential order.
    The output is a highly compressed CrackleArray that is still random access.
    See: https://github.com/seung-lab/crackle
    You need to pip install connected-components-3d[stack] to get the extra modules.

    def sections(labels_in):
    '''
    A generator that produces thick Z slices
    of an image
    '''
    for z in range(0, labels_in.shape[2], 100):
        yield labels_in[:,:,z:z+100]

    # You can access compressed_labels_out using array notation
    compressed_labels_out = cc3d.connected_components_stack(sections(labels))
    # convert to numpy array, probably a big mistake since
    # you probably expected it was going to blow up RAM
    cc_labels = compressed_labels_out.numpy()
    # if you don't like hanging onto this exotic format, you
    # can write it as a numpy array to disk in a memory efficient way.
    compressed_labels_out.save("example.npy.gz")
    # or hang onto it
    compressed_labels_out.save("example.ckl")
    """
    import cc3d
    if verbose:
        print("Finding features like grains.")
    labels_out, N = cc3d.connected_components(image_data, connectivity=connectivity,
                                              delta=delta, return_N=True)
    return labels_out, N, connectivity

def detect_grains_3d(state_array, connectivity=3, return_num_grains=False):
    
    """
    Detect grains in a 3D Monte Carlo state array using connected component labeling.
    
    Grains are identified as connected regions of voxels with the same state ID.
    Returns a grain ID (lfi - Local Feature ID) array with the same shape as input.
    
    Parameters
    ----------
    state_array : ndarray
        3D integer array where each value represents a state ID (0 to S-1).
        Shape: (nx, ny, nz)
    connectivity : int, optional
        Connectivity type for 3D neighbor detection:
        - 1: Face connectivity (6 neighbors, most restrictive)
        - 2: Face + edge connectivity (18 neighbors)
        - 3: Face + edge + corner connectivity (26 neighbors, default)
    return_num_grains : bool, optional
        If True, returns tuple (lfi_array, num_grains). Default is False.
    
    Returns
    -------
    lfi_array : ndarray
        3D integer array of grain IDs with same shape as state_array.
        Grain IDs start from 1. Background (if any) is labeled 0.
    num_grains : int (only if return_num_grains=True)
        Total number of grains detected.
    
    Notes
    -----
    - Uses scipy.ndimage.label for connected component analysis
    - Connectivity=1 (6-neighbor) is strictest, useful for well-separated grains
    - Connectivity=3 (26-neighbor) is most common, matches typical grain definitions
    - Processing time scales with array size: ~0.1s for 100³, ~1s for 300³ voxels
    - Memory usage: ~2x input array size for intermediate calculations
    
    Examples
    --------
    >>> import numpy as np
    >>> from upxo.gsdataops.grid_ops import detect_grains_3d
    >>> 
    >>> # Simple 3D state array with two grains
    >>> s = np.zeros((10, 10, 10), dtype=np.int16)
    >>> s[:5, :, :] = 1  # State 1 in left half
    >>> s[5:, :, :] = 2  # State 2 in right half
    >>> 
    >>> # Detect grains
    >>> lfi = detect_grains_3d(s)
    >>> print(f"Grain IDs: {np.unique(lfi)}")  # [1, 2]
    >>> 
    >>> # With grain count
    >>> lfi, num_grains = detect_grains_3d(s, return_num_grains=True)
    >>> print(f"Detected {num_grains} grains")
    >>> 
    >>> # Use strict connectivity
    >>> lfi_strict = detect_grains_3d(s, connectivity=1)
    
    See Also
    --------
    scipy.ndimage.label : Underlying labeling function
    upxo.ggrowth.mcgs.detect_grains : 2D/3D grain detection in MCGS objects
    """
    from scipy import ndimage
    
    # Validate input
    if state_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {state_array.ndim}D array")
    
    if connectivity not in [1, 2, 3]:
        raise ValueError(f"Connectivity must be 1, 2, or 3, got {connectivity}")
    
    # Generate connectivity structure for scipy
    # connectivity=1 -> 3x3x3 structure with only face neighbors
    # connectivity=2 -> 3x3x3 structure with face + edge neighbors
    # connectivity=3 -> 3x3x3 structure with all 26 neighbors
    structure = ndimage.generate_binary_structure(3, connectivity)
    
    # Get unique state IDs
    unique_states = np.unique(state_array)
    
    # Initialize grain ID array
    lfi_array = np.zeros_like(state_array, dtype=np.int32)
    
    # Counter for grain IDs across all states
    current_grain_id = 1
    
    # Label connected components for each state separately
    for state_id in unique_states:
        # Create binary mask for current state
        state_mask = (state_array == state_id)
        
        # Label connected components within this state
        labeled_state, num_features = ndimage.label(state_mask, structure=structure)
        
        # Assign unique grain IDs to this state's grains
        # Avoid relabeling zeros (background)
        mask_nonzero = labeled_state > 0
        lfi_array[mask_nonzero] = labeled_state[mask_nonzero] + current_grain_id - 1
        
        # Update grain ID counter
        current_grain_id += num_features
    
    # Total number of grains detected
    num_grains = current_grain_id - 1
    
    if return_num_grains:
        return lfi_array, num_grains
    else:
        return lfi_array
    
def detect_grains_3d_optimized(state_array, connectivity=3, return_num_grains=False):
    from scipy import ndimage
    
    # Validate input
    if state_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {state_array.ndim}D array")
    
    if connectivity not in [1, 2, 3]:
        raise ValueError(f"Connectivity must be 1, 2, or 3, got {connectivity}")
    
    # Generate connectivity structure for scipy
    # connectivity=1 -> 3x3x3 structure with only face neighbors
    # connectivity=2 -> 3x3x3 structure with face + edge neighbors
    # connectivity=3 -> 3x3x3 structure with all 26 neighbors
    
    structure = ndimage.generate_binary_structure(3, connectivity)
    lfi_array = np.zeros_like(state_array, dtype=np.int32)
    
    # 1. Flatten and get indices that would sort the array
    flat_array = state_array.ravel()
    sorted_idx = np.argsort(flat_array)
    sorted_states = flat_array[sorted_idx]

    # 2. Find where the state IDs change
    diffs = np.diff(sorted_states)
    split_indices = np.where(diffs != 0)[0] + 1
    
    # Split the indices into groups, one for each state ID
    groups = np.split(sorted_idx, split_indices)
    unique_states = sorted_states[np.concatenate(([0], split_indices))]

    current_grain_id = 1

    # 3. Process each state using its pre-found indices
    for state_id, indices in zip(unique_states, groups):
        # Instead of state_array == state_id, create mask using known indices
        # This mask is sparse and only touches necessary memory
        state_mask = np.zeros(state_array.size, dtype=bool)
        state_mask[indices] = True
        state_mask = state_mask.reshape(state_array.shape)
        
        # Label connected components
        labeled_state, num_features = ndimage.label(state_mask, structure=structure)
        
        if num_features > 0:
            mask_nonzero = labeled_state > 0
            lfi_array[mask_nonzero] = labeled_state[mask_nonzero] + (current_grain_id - 1)
            current_grain_id += num_features

    num_grains = current_grain_id - 1
    return (lfi_array, num_grains) if return_num_grains else lfi_array

def detect_grains_using_s(scalar_data, scalar_ids):
    from scipy import ndimage
    from scipy.ndimage import generate_binary_structure as GBS
    results = [ndimage.label(scalar_data == s, structure=GBS(3, 3)) for s in scalar_ids]
    label_volumes = [res[0] for res in results]
    counts = [res[1] for res in results]
    offsets = np.cumsum([0] + counts[:-1])
    final_labels = []
    for i in range(len(label_volumes)):
        vol = label_volumes[i]
        offset = offsets[i]
        mask = (vol > 0)
        vol[mask] += offset
        final_labels.append(vol)
    grain_id_map = np.sum(final_labels, axis=0)
    return grain_id_map


def detect_grains_fast_v3(scalar_data, scalar_ids):
    from scipy import ndimage
    from scipy.ndimage import generate_binary_structure as GBS
    grain_id_map = np.zeros(scalar_data.shape, dtype=np.int32)
    structure = GBS(3, 3)
    current_offset = 0

    # 1. Sort indices by state ID (The "Grouping" step)
    # This is the secret to avoiding scanning the whole volume repeatedly
    flat_data = scalar_data.ravel()
    sort_idx = np.argsort(flat_data)
    sorted_values = flat_data[sort_idx]
    
    # 2. Find transitions between IDs
    diffs = np.where(np.diff(sorted_values) != 0)[0] + 1
    # Only include IDs that are in your 'scalar_ids' list
    split_indices = np.split(sort_idx, diffs)
    found_ids = sorted_values[np.concatenate(([0], diffs))]

    # 3. Process each group
    for s_id, pixels in zip(found_ids, split_indices):
        if s_id not in scalar_ids: continue
        
        # Create a local coordinate mask only for the pixels we KNOW belong to this ID
        # We find the local bounding box of these specific pixels
        z, y, x = np.unravel_index(pixels, scalar_data.shape)
        zmin, zmax = z.min(), z.max() + 1
        ymin, ymax = y.min(), y.max() + 1
        xmin, xmax = x.min(), x.max() + 1
        
        # Slice the region and create a binary mask
        sub_vol = scalar_data[zmin:zmax, ymin:ymax, xmin:xmax]
        mask = (sub_vol == s_id)
        
        # Label the sub-volume (handles multiple grains of the same state perfectly)
        labeled_sub, num_features = ndimage.label(mask, structure=structure)
        
        if num_features > 0:
            # Shift IDs to global unique IDs and place back
            dest = grain_id_map[zmin:zmax, ymin:ymax, xmin:xmax]
            nonzero = labeled_sub > 0
            dest[nonzero] = labeled_sub[nonzero] + current_offset
            current_offset += num_features

    return grain_id_map

def detect_grains_greedy_3d(state_array):
    nx, ny, nz = state_array.shape
    lfi_array = np.zeros_like(state_array, dtype=np.int32)
    visited = np.zeros_like(state_array, dtype=bool)
    
    current_grain_id = 1
    
    # Define neighbor offsets for 26-connectivity (connectivity=3)
    # This is the "greedy" reach of the algorithm
    offsets = np.array(np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1])).T.reshape(-1, 3)
    offsets = offsets[~np.all(offsets == 0, axis=1)] # Remove [0,0,0]

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # If we haven't visited this voxel, it's the start of a new grain
                if not visited[x, y, z]:
                    state_id = state_array[x, y, z]
                    
                    # Start a Greedy BFS (Flood Fill)
                    queue = deque([(x, y, z)])
                    visited[x, y, z] = True
                    lfi_array[x, y, z] = current_grain_id
                    
                    while queue:
                        cx, cy, cz = queue.popleft()
                        
                        for dx, dy, dz in offsets:
                            nx_idx, ny_idx, nz_idx = cx+dx, cy+dy, cz+dz
                            
                            # Bounds check
                            if 0 <= nx_idx < nx and 0 <= ny_idx < ny and 0 <= nz_idx < nz:
                                # Check if same state and not yet visited
                                if not visited[nx_idx, ny_idx, nz_idx] and \
                                   state_array[nx_idx, ny_idx, nz_idx] == state_id:
                                    
                                    visited[nx_idx, ny_idx, nz_idx] = True
                                    lfi_array[nx_idx, ny_idx, nz_idx] = current_grain_id
                                    queue.append((nx_idx, ny_idx, nz_idx))
                    
                    current_grain_id += 1
                    
    return lfi_array
# ###########################################################################
# ###########################################################################
#                   Intra-grain location finding operations
#         Search ID:      SID_IntraGrainLocOps
# ###########################################################################
# ###########################################################################


# ###########################################################################
# ###########################################################################
#                   General LFI operations
#         Search ID:      SID_GeneralLFIOps

def shuffle_feature_IDs(lfi):
    """
    Shuffle the feature IDs in a local feature ID array (lfi) while keeping the
    same number of features and their spatial distribution intact.

    Parameters
    ----------
    lfi : np.ndarray
        2D array of local feature IDs (grain IDs).

    Returns
    -------
    np.ndarray
        Local feature ID array with shuffled feature IDs.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.shuffle_feature_IDs
    """
    unique_ids = np.unique(lfi)
    if 0 in unique_ids:
        unique_ids = unique_ids[unique_ids != 0]
    shuffled_ids = unique_ids.copy()
    np.random.shuffle(shuffled_ids)
    id_map = dict(zip(unique_ids, shuffled_ids))
    lfi = np.vectorize(lambda x: id_map.get(x, x))(lfi)
    return lfi

def merge_features_to_neighs(lfi, fids, neigh_fids):
    """
    Update the labelled feature ID array (lfi) by merging specified fids
    into their corresponding neighboring feature IDs (neigh_fids).

    Parameters
    ----------
    lfi : np.ndarray
        2D array of local feature IDs (grain IDs).
    fids : list or array-like
        List of feature IDs to be merged.
    neigh_fids : dict
        Dictionary mapping each feature ID in fids to its neighboring feature ID.

    Returns
    -------
    np.ndarray
        Updated lfi array with specified fids merged into their neighbors.
    
    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.merge_features_to_neighs
    """
    for island in fids:
        lfi[lfi==island] = neigh_fids[island]
    return lfi

# #################################################################################
# #################################################################################
'''
VOXEL MORPHOLOGY SMOOTHING OPERATIONS

Search ID:      SID_VoxelMorphSmoothOps

List of definitions
-------------------
get_ball_footprint
smooth_voxMorph_npass
smooth_voxMorph_1pass
majority_filter_2d: numba implementation
majority_filter_3d_npass
majority_filter_3d_1pass
'''
def get_ball_footprint(fpSize, removeEndVox=True):
    from skimage.morphology import ball
    fp = ball(fpSize)
    if removeEndVox:
        coords = np.argwhere(fp)
        # cx, cy, cz = np.array(fp.shape) // 2
        for axis in range(3):
            idx = np.argmax(coords[:, axis])
            fp[tuple(coords[idx])] = False
            idx = np.argmin(coords[:, axis])
            fp[tuple(coords[idx])] = False
    return fp

def smooth_voxMorph_npass(lfi, niterations=2, DILfpSizes=[4, 4], 
                ERSfpSizes=[4, 4], footprints=['ball', 'ball'],
                removeEndVox=[True, True]):
    for i in np.arange(1, niterations+1):
        lfi = smooth_voxMorph_1pass(lfi, DILfpSize=DILfpSizes[i-1],
                ERSfpSize=ERSfpSizes[i-1], footprint=footprints[i-1],
                removeEndVox=removeEndVox[i-1])
    return lfi

def smooth_voxMorph_1pass(lfi, DILfpSize=4, ERSfpSize=4,
                footprint='ball', removeEndVox=True):
    from skimage.morphology import dilation, erosion
    if footprint in ('ball', 'sphere'):
        DILfp = get_ball_footprint(DILfpSize)
        ERSfp = get_ball_footprint(ERSfpSize)
    else:
        raise NotImplementedError(f"Footprint {footprint} not implemented yet.")
    return erosion(dilation(lfi, footprint=DILfp), footprint=ERSfp)

@njit(parallel=True)
def majority_filter_2d(lfi):
    """
    Apply a 2D majority filter to the local feature ID array (lfi) to smooth
    the grain structure by replacing each pixel's feature ID with the most common
    feature ID among its 8 neighbors (including itself).

    Parameters
    ----------
    lfi : np.ndarray
        2D array of labelled feature IDs (grain IDs).

    Returns
    -------
    np.ndarray
        Smoothed 2D array of feature IDs after applying the majority filter.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.majority_filter_2d
    """
    h, w = lfi.shape
    lfi_new = lfi.copy()
    for i in prange(1, h-1):
        for j in range(1, w-1):
            # Extract 3x3 neighborhood
            neigh = lfi[i-1:i+2, j-1:j+2].ravel()
            max_label = -1
            max_count = 0
            tie = False
            # Count frequencies
            for a in range(neigh.size):
                label = neigh[a]
                count = 0
                for b in range(neigh.size):
                    if neigh[b] == label:
                        count += 1
                if count > max_count:
                    max_count = count
                    max_label = label
                    tie = False
                elif count == max_count and label != max_label:
                    tie = True
            if not tie:
                lfi_new[i, j] = max_label
    return lfi_new

def majority_filter_3d_npass(lfi=None, n=2, sizes=[3, 3]):
    """
    Apply multiple passes of a 3D majority filter to the local feature ID array (lfi)
    to iteratively smooth the grain structure.

    Parameters
    ----------
    lfi : np.ndarray
        3D array of labelled feature IDs (grain IDs) to be smoothed. Default is None, 
        which will raise an error if not provided.
    n : int
        Number of passes to apply. Default is 2.
    sizes : list of int
        List of window sizes for each pass. Must have length n. Default is [3, 3].

    Returns
    -------
    np.ndarray
        Smoothed 3D array of feature IDs after applying n passes of the majority filter.
    """
    for i in np.arange(1, n+1):
        lfi_new = _majority_filter_3d_(lfi, size=sizes[i-1])
        diff = np.sum(lfi_new != lfi)
        print(f"Smoothing pass {i}. Number of voxels modified: {diff}")
        lfi = lfi_new
    return lfi

def majority_filter_3d_1pass(lfi=None, size=3):
    """
    Apply a 3D majority filter to the local feature ID array (lfi) to smooth
    the grain structure by replacing each voxel's feature ID with the most common
    feature ID among its neighbors in a 3D window.

    Parameters
    ----------
    lfi : np.ndarray
        3D array of labelled feature IDs (grain IDs). Default is None, which will raise 
        an error if not provided.
    size : int, optional
        Size of the neighborhood window (must be odd). Default is 3 for a 3x3x3 window.

    Returns
    -------
    np.ndarray
        Smoothed 3D array of feature IDs after applying the majority filter.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.majority_filter_3d
    """
    return _majority_filter_3d_(lfi, size=size)

@njit(parallel=True)
def _majority_filter_3d_(lfi, size=3):
    """
    Apply a more general 3D majority filter to the local feature ID array (lfi)
    using a window of the specified size. 

    Parameters
    ----------
    lfi : np.ndarray
        3D array of labelled feature IDs.
    size : int
        Size of the neighborhood window (must be odd). Default is 3 for a 3x3x3 window.

    Returns
    -------
    np.ndarray
        Smoothed 3D array.
    """
    r = size // 2  # radius
    d, h, w = lfi.shape
    lfi_new = lfi.copy()
    for i in prange(r, d - r):
        for j in range(r, h - r):
            for k in range(r, w - r):
                max_label = -1
                max_count = 0
                tie = False
                # Count frequencies inside window
                for ii in range(i - r, i + r + 1):
                    for jj in range(j - r, j + r + 1):
                        for kk in range(k - r, k + r + 1):
                            label = lfi[ii, jj, kk]
                            count = 0
                            # Count occurrences of this label
                            for iii in range(i - r, i + r + 1):
                                for jjj in range(j - r, j + r + 1):
                                    for kkk in range(k - r, k + r + 1):
                                        if lfi[iii, jjj, kkk] == label:
                                            count += 1
                            if count > max_count:
                                max_count = count
                                max_label = label
                                tie = False
                            elif count == max_count and label != max_label:
                                tie = True
                if not tie:
                    lfi_new[i, j, k] = max_label
    return lfi_new


def compute_grain_bounding_boxes(lfi, padding=1):
    """
    Identifies the min/max spatial extent for every grain ID in the volume.
    Parameters:
    -----------
    lfi : ndarray
        The 3D grain ID array.
    padding : int
        Extra voxels to include around the box to ensure boundary
        gradients are captured.
    Returns:
    --------
    bboxes : dict
        Mapping of GrainID -> tuple(slice_x, slice_y, slice_z)
    """
    bboxes = {}
    unique_ids = np.unique(lfi)
    # We skip 0 if it represents the background/exterior
    for gid in unique_ids:
        if gid == 0:
            continue
        # Find coordinates where the grain exists
        coords = np.argwhere(lfi == gid)
        if coords.size == 0:
            continue
        # Get min and max for each axis
        min_coords = coords.min(axis=0) - padding
        max_coords = coords.max(axis=0) + padding + 1
        # Clip to ensure we stay within the array dimensions
        min_coords = np.maximum(min_coords, 0)
        max_coords = np.minimum(max_coords, lfi.shape)
        # Create slice objects for easy indexing
        bboxes[gid] = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    return bboxes

def compute_local_grain_mask(lfi, bounding_box, grain_id):
    """
    Extracts a local boolean mask for a specific grain within its bounding box.
    Parameters:
    -----------
    lfi : ndarray
        The 3D grain ID array (can be the original or lfiex).
    bounding_box : tuple of slices
        The (slice_x, slice_y, slice_z) identified for this grain.
    grain_id : int
        The ID of the grain to mask.
    Returns:
    --------
    grain_mask : ndarray[bool]
        A 3D boolean array of the same shape as the bounding box crop,
        True where the voxel belongs to the grain_id.
    """
    # 1. Crop the global array to the bounding box
    local_crop = lfi[bounding_box]
    # 2. Create the boolean mask for the target grain
    grain_mask = (local_crop == grain_id)
    return grain_mask
# #################################################################################
# #################################################################################
# STANDARD LFI GENERATORS
def generate_test_2D_LFI_1(plotlfi=True, figsize=(6, 6), dpi=100):
    """
    Generate a 2D Local Feature ID (LFI) array with multiple grains and features
    for testing purposes. The array includes distinct quadrants, a central feature,
    an island, a thin neck, and a triple point cluster to provide a variety of
    grain structures for testing grain detection and analysis algorithms.

    Returns
    -------
    np.ndarray
        A 2D array of shape (200, 200) representing the LFI with various features.
    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.generate_test_2D_LFI_1
    """
    M, N = 200, 200
    lfi = np.zeros((M, N), dtype=int)

    # --- Original Quadrants and Central Features ---
    lfi[:100, :100] = 1
    lfi[100:, :100] = 2
    lfi[:100, 100:] = 3
    lfi[100:, 100:] = 4
    lfi[75:125, 75:125] = 5  # Central square
    lfi[20:30, 20:40] = 6    # Island
    lfi[150:152, 10:190] = 7 # The Neck (Thin horizontal strip)
    lfi[95:105, 95:105] = 8  # Triple Point Cluster

    # --- New Features using Coordinate Grids ---
    Y, X = np.ogrid[:M, :N]

    # 3. Sinusoidal Interface (challenging Grain 2 / Grain 4 boundary)
    # Redefine the boundary between Y > 100 and Y < 100 with a wave
    wave = 100 + 8 * np.sin(X / 15.0)
    mask_wave = (Y < wave) & (X > 100)
    lfi[mask_wave] = 4
    lfi[~mask_wave & (X > 100) & (Y < 125)] = 3

    # 4. The "Swiss Cheese" Grain (Grain 12 with holes 13, 14)
    center_cheese = (50, 150)
    dist_cheese = np.sqrt((X - center_cheese[1])**2 + (Y - center_cheese[0])**2)
    lfi[dist_cheese < 25] = 12
    # Add holes
    lfi[np.sqrt((X-140)**2 + (Y-45)**2) < 5] = 13
    lfi[np.sqrt((X-160)**2 + (Y-55)**2) < 5] = 14

    # 1. Circle inside a Circle (Concentric)
    # Located in the bottom-right quadrant (Grain 4 area)
    center_concentric = (170, 170) # (y, x)
    dist_concentric = np.sqrt((X - center_concentric[1])**2 + (Y - center_concentric[0])**2)
    lfi[dist_concentric < 20] = 9  # Outer circle
    lfi[dist_concentric < 8] = 10  # Inner circle

    # 2. Coalescing Circles (Two overlapping features)
    # Located in the top-left quadrant (Grain 1 area)
    c1, c2 = (40, 70), (40, 85) # (y, x) Centers shifted horizontally
    dist1 = np.sqrt((X - c1[1])**2 + (Y - c1[0])**2)
    dist2 = np.sqrt((X - c2[1])**2 + (Y - c2[0])**2)
    # Both circles assigned to the same Grain ID to simulate coalescence
    lfi[(dist1 < 12) | (dist2 < 12)] = 11 

    # 5. Triple Triangles meeting at a single pixel (Grains 15, 16, 17)
    # Define exact center (must be floats for precise distance calculation)
    cy, cx = 115.0, 35.0
    radius = 22.0
    
    # 1. Create relative coordinate grids
    # Y, X are already defined as np.ogrid[:M, :N] in your function
    dy = Y - cy
    dx = X - cx
    
    # 2. Calculate Distance and Angle
    # We use degrees [0, 360] to make the 120-degree logic foolproof
    dist = np.sqrt(dx**2 + dy**2)
    # arctan2(y, x) returns [-pi, pi], so we shift to [0, 360]
    angle = (np.arctan2(dy, dx) * 180 / np.pi) % 360
    
    # 3. Create the feature mask
    mask = dist <= radius
    
    # 4. Assign Sectors (Triangle 120-degree divisions)
    # Triangle 1 (Bottom sector in your image)
    lfi[mask & (angle >= 340) & (angle < 360)] = 15
    # Triangle 2 (Top-Right sector)
    lfi[mask & (angle >= 30) & (angle < 50)] = 16
    # Triangle 3 (Top-Left sector)
    lfi[mask & (angle >= 120) & (angle < 240)] = 17

    if plotlfi:
        # --- Visualization Check ---
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(lfi, origin='lower', cmap='tab20')
        plt.title("Updated LFI with Concentric and Coalescing Features")
        plt.colorbar(label='Grain ID')
        plt.show()

    return lfi
# #################################################################################
# #################################################################################
def generate_constrained_hybrid_seeds(lfi, target_spacing=0.5, bulk_spacing=10.0,
                             jitter_factor=1.0, margin=0.0, padding=1.0, plot_seeds=False,
                             figsize=(8, 8), dpi=50, markersize=2):
    """
    Seeding with Rigid Guard Rails to eliminate RVE boundary irregularities.
    Ensures straight edges at the domain limits [0, 200].

    Parameters
    ----------
    lfi : np.ndarray
        2D array of local feature IDs (grain IDs).
    target_spacing : float
        Desired spacing between seeds along the boundary. Default is 0.5 units.
    bulk_spacing : float
        Desired spacing between seeds in the bulk (internal) region. Default is 10.0
        units.
    jitter_factor : float
        Factor to control the amount of jitter applied to bulk seeds. Default is 1.0
        (0 means no jitter, 1 means up to ±bulk_spacing/2 jitter).
    margin : float
        Margin to define an internal buffer to prevent bulk seeds from pushing against RVE edges. Default is 0.0.
    padding : float
        Padding to define an additional buffer around the domain edges for guard rails. Default is 1.0.
    plot_seeds : bool
        Whether to plot the seeds on top of the LFI for visualization. Default is False.
    figsize : tuple
        Size of the figure for plotting seeds if plot_seeds is True. Default is (8, 8).
    dpi : int
        Dots per inch for the seed plot if plot_seeds is True. Default is 50.
    markersize : int
        Size of the markers for the seeds if plot_seeds is True. Default is 2.

    Returns
    -------
    np.ndarray
        Array of seed coordinates with shape (num_seeds, 2), where each row is [x, y].

    Notes
    -----
    - The function first detects the grain boundaries in the LFI and samples seeds along
      these boundaries at a specified target spacing.
    - It then identifies internal regions that are sufficiently far from the boundaries
      and samples seeds in these bulk areas with a specified bulk spacing, applying jitter
        to avoid regular patterns.
    - Finally, it adds rigid guard rails of seeds along the domain edges to ensure that
      the Voronoi tessellation produces straight edges at the boundaries, eliminating
        irregularities caused by boundary effects.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.constrained_hybrid_seeds
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    
    M, N = lfi.shape
    
    # 1. High-Resolution Interface Detection
    gy, gx = np.gradient(lfi)
    boundary_mask = (gx != 0) | (gy != 0)
    
    # 2. Equidistant Boundary Sampling
    coords_boundary = np.argwhere(boundary_mask).astype(float)
    stride = max(1, int(target_spacing / 1.0))
    seeds_boundary = coords_boundary[::stride]

    # 3. Jittered Bulk Sampling (Internal Only)
    dist_to_boundary = distance_transform_edt(~boundary_mask)
    # Define an internal buffer to prevent bulk seeds from pushing against RVE edges
    Y, X = np.indices((M, N))
    internal_mask = (X > margin) & (X < N-margin) & (Y > margin) & (Y < M-margin)
    
    bulk_mask = (dist_to_boundary > target_spacing * 3) & internal_mask
    coords_bulk = np.argwhere(bulk_mask).astype(float)
    
    seeds_bulk = []
    if len(coords_bulk) > 0:
        bulk_stride = max(1, int(bulk_spacing))
        seeds_bulk = coords_bulk[::bulk_stride]
        # Apply jitter to break inclined/aliased patterns
        jitter = (np.random.rand(*seeds_bulk.shape) - 0.5) * (bulk_stride * jitter_factor)
        seeds_bulk += jitter

    # 4. RIGID GUARD RAILS (The Boundary Fix)
    # Place seeds exactly 1 unit outside the boundary to create flat bisectors
    guard_spacing = target_spacing 
    pad = padding
    
    rail_coords = []
    # Horizontal rails (Top and Bottom)
    steps_x = np.arange(-pad, N + pad, guard_spacing)
    for x in steps_x:
        rail_coords.append([-pad, x])      # Bottom rail
        rail_coords.append([M + pad, x])   # Top rail
        
    # Vertical rails (Left and Right)
    steps_y = np.arange(-pad, M + pad, guard_spacing)
    for y in steps_y:
        rail_coords.append([y, -pad])      # Left rail
        rail_coords.append([y, N + pad])   # Right rail
    
    seeds_guard = np.array(rail_coords)
    all_seeds = np.vstack([seeds_boundary, seeds_bulk, seeds_guard])[:, [1, 0]]

    if plot_seeds:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(seeds_boundary[:, 1], seeds_boundary[:, 0], 'ko', label='Boundary Seeds', markersize=markersize)
        ax.plot(seeds_bulk[:, 1], seeds_bulk[:, 0], 'b.', label='Bulk Seeds', markersize=markersize)
        ax.plot(seeds_guard[:, 1], seeds_guard[:, 0], 'ro', label='Guard Rails', markersize=markersize+2)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Formatting
        ax.set_title("RVE optimized constrained hybrid Voronoi tessellation seeds", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        # Rigid RVE Cropping
        ax.set_xlim(0, N)
        ax.set_ylim(0, M)
        plt.show()
    # 5. Standardize to [x, y] Cartesian coordinates
    return all_seeds

def generate_poisson_disk_seeds(xbound, ybound, radius=5, k=6, see_seeds=False, **plotKwargs):
    """
    Generate Poisson Disk Sampling seeds within a specified rectangular domain defined by xbound and ybound.

    Parameters
    ----------
    xbound : tuple
        A tuple (x_min, x_max) defining the horizontal bounds of the sampling area.
    ybound : tuple
        A tuple (y_min, y_max) defining the vertical bounds of the sampling area.
    radius : float
        The minimum distance between any two seeds. Default is 5 units.
    k : int
        The number of attempts to place a new seed around an existing seed before giving up. Default is 6.

    Returns
    -------
    np.ndarray
        An array of seed coordinates with shape (num_seeds, 2), where each row is [x, y].

    Notes
    -----
    - Poisson Disk Sampling is a method for generating points that are evenly distributed while maintaining 
        a minimum distance between them.
    - The algorithm typically starts with an initial random seed and iteratively attempts to place new seeds
        around existing seeds, ensuring that they are at least 'radius' distance apart. The 'k' parameter 
        controls how many attempts are made to place a new seed around an existing seed before it is discarded.
    - This method is particularly useful for applications like seeding in Voronoi tessellations, where a more 
        uniform distribution of seeds is desired compared to purely random sampling.

    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.generate_poisson_disk_seeds
    """
    # pds: Poisson Disk Sampling
    xstart, ystart = xbound[0], ybound[0]
    # xend, yend = xbound[1], ybound[1]
    from upxo.statops.sampling import bridson_uniform_density as bud
    # bud(width=1, height=1, radius=0.2, k=20)
    points = bud(width=xbound[1]-xbound[0], height=ybound[1]-ybound[0], radius=radius, k=k)
    locx = [_[0]+xstart for _ in points]
    locy = [_[1]+ystart for _ in points]
    seeds = np.array(list(zip(locx, locy)))

    if see_seeds:    
        from upxo.viz import gsviz        
        gsviz.see_2dPoints(seeds, figsize=plotKwargs.get('figsize', (6, 6)), 
                    dpi=plotKwargs.get('dpi', 100), title=plotKwargs.get('title', 'Poisson Disk Seeds'),
                    xlabel=plotKwargs.get('xlabel', 'X-axis'), ylabel=plotKwargs.get('ylabel', 'Y-axis'),
                    point_color=plotKwargs.get('point_color', 'black'), 
                    point_size=plotKwargs.get('point_size', 20), 
                    point_marker=plotKwargs.get('point_marker', '.'), 
                    point_alpha=plotKwargs.get('point_alpha', 0.8))
    return seeds

def generate_darted_seeds(xbound, ybound, radius=5, k=6, see_seeds=False, **plotKwargs):
    """
    Generate Darted Sampling seeds within a specified rectangular domain defined by xbound and ybound.

    Parameters
    ----------
    xbound : tuple
        A tuple (x_min, x_max) defining the horizontal bounds of the sampling area.
    ybound : tuple
        A tuple (y_min, y_max) defining the vertical bounds of the sampling area.
    radius : float
        The minimum distance between any two seeds. Default is 5 units.
    k : int
        The number of attempts to place a new seed around an existing seed before giving up. Default
        is 6.

    Returns
    -------
    np.ndarray
        An array of seed coordinates with shape (num_seeds, 2), where each row is [x, y].

    Notes
    -----
    - Darted Sampling is a method for generating points that are evenly distributed while maintaining 
        a minimum distance between them, similar to Poisson Disk Sampling. However, Darted Sampling 
        typically involves a more aggressive rejection of candidate points, which can lead to a more 
        uniform distribution in certain cases.
    - The algorithm starts with an initial random seed and iteratively attempts to place new seeds around
        existing seeds. If a candidate seed is too close to an existing seed (within the specified radius), 
        it is rejected (or "darted"). The 'k' parameter controls how many attempts are made to place a new 
        seed around an existing seed before it is discarded.
    - This method can be particularly effective for applications like seeding in Voronoi tessellations, 
        where a more uniform distribution of seeds is desired compared to purely random sampling, and 
        can sometimes produce better results than Poisson Disk Sampling in terms of uniformity.
    - The 'darting' aspect of the algorithm can help to break up clusters of points that might occur in 
        Poisson Disk Sampling, leading to a more even distribution of seeds across the domain.
    
    Import
    ------
    import upxo.gsdataops.grid_ops as gridOps
    Use as: gridOps.generate_darted_seeds
    """
    from upxo.statops.sampling import dart
    xstart, ystart = xbound[0], ybound[0]
    points = dart(width=xbound[1]-xbound[0], height=ybound[1]-ybound[0], radius=radius, k=k)
    locx = [_[0]+xstart for _ in points]
    locy = [_[1]+ystart for _ in points]
    seeds = np.array(list(zip(locx, locy)))
    if see_seeds:
        from upxo.viz import gsviz
        gsviz.see_2dPoints(seeds, figsize=plotKwargs.get('figsize', (6, 6)), 
                    dpi=plotKwargs.get('dpi', 100), title=plotKwargs.get('title', 'Darted Seeds'),
                    xlabel=plotKwargs.get('xlabel', 'X-axis'), ylabel=plotKwargs.get('ylabel', 'Y-axis'),
                    point_color=plotKwargs.get('point_color', 'blue'), 
                    point_size=plotKwargs.get('point_size', 20), 
                    point_marker=plotKwargs.get('point_marker', '.'), 
                    point_alpha=plotKwargs.get('point_alpha', 0.8),
                    plot_legend=plotKwargs.get('plot_legend', True)
                    )

    return seeds
# #################################################################################
# #################################################################################
# This module is expected to grow quite a lot. In the interest of locating
# the right definitions whilst development, I have provided the below
# search ids. Just copy the search IDs and search for it. The definition 
# must be near by to the location identified.

#    DEVELOPER :---- FUNCTIONALITY LOCATION SEARCH

# Sectioning operations. Search ID: SID_3dSectioningOps
# Bounding box operations. Search ID: SID_BBoxOps
# Intra-grain location finding operations. Search ID: SID_IntraGrainLocOps
# General LFI operations. Search ID: SID_GeneralLFIOps
# Voxel morphology smoothing operations. Search ID: SID_VoxelMorphSmoothOps