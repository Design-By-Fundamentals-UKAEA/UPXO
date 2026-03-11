import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_erosion

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html

def introduce_boundary_zones(gs, fids=[], niterations=5, method='use_min_core_size',
                             min_core_size=50, 
                             threshold_bz_thickness=2,
                             structure_nx=3, structure_ny=3, reset_others=False,
                             renumber_fid_map=True, morph_char=False, topo_char=False,
                             remap_scalars=False, perform_topology_checks=False,
                             see_plot=False, figure_size=(5, 5), figure_dpi=100):
    """
    Introduce boundary zones and cores within grains using morphological erosion.
    
    This function partitions each grain into a boundary zone (outer region) and a core 
    (inner region) by applying binary erosion. The method parameter controls how the 
    boundary zone thickness is determined.

    Parameters
    ----------
    gs : mcgs2_grain_structure
        Grain structure object (temporal slice) from UPXO. **Note:** This object is 
        modified in-place; attributes `bbox_bz` and `bbox_core` are added to each 
        grain. Pass a deep copy if you need to preserve the original.
    fids : list of int, optional
        List of grain IDs (gid) to process. If empty (default), processes all grains 
        in `gs.gid`. Use to selectively apply boundary zone detection to specific grains.
    niterations : int, default=5
        Maximum number of morphological erosion iterations to attempt. Higher values 
        allow thicker boundary zones but may eliminate small grains entirely.
    method : {'use_min_core_size', 'simple', 'thresholded'}, default='use_min_core_size'
        Algorithm for defining boundary zones:
        
        - `'use_min_core_size'`: Erodes until core size drops below `min_core_size`, 
          then backs off to maintain minimum core dimension (recommended for ensuring 
          non-trivial cores).
        - `'simple'`: Uses maximum successful erosion without size constraints (may 
          produce very small cores).
        - `'thresholded'`: Stops erosion when boundary zone reaches specific thickness 
          defined by `threshold_bz_thickness`.
    min_core_size : int, default=50
        Minimum number of pixels required in the grain core (used only with 
        `method='use_min_core_size'`). Grains smaller than this after maximum erosion 
        will have their entire bbox assigned as boundary zone with `bbox_core=None`.
    threshold_bz_thickness : int, default=2
        Number of erosion iterations to define boundary zone thickness (used only with 
        `method='thresholded'`). Boundary zone is the result of eroding exactly this 
        many iterations.
    structure_nx : int, default=3
        Width of the morphological structuring element (connectivity kernel). Value of 3 
        creates 8-connectivity for erosion (includes diagonals). Use 2 for 4-connectivity.
    structure_ny : int, default=3
        Height of the morphological structuring element. Typically set equal to 
        `structure_nx` for isotropic erosion.
    see_plot : bool, default=False
        If True, generates a matplotlib visualization showing boundary zones overlaid 
        on the grain structure with distinct colormap (useful for debugging).
    figure_size : tuple of float, default=(5, 5)
        Width and height of the plot figure in inches (used only if `see_plot=True`).
    figure_dpi : int, default=100
        Resolution of the plot figure in dots per inch (used only if `see_plot=True`).

    Returns
    -------
    lgi_new : ndarray of int
        2D label image with boundary zones visualized as separate grain IDs. Shape matches 
        `gs.lgi`. Original grain cores retain their IDs; boundary zones are assigned new 
        sequential IDs starting from `max(gs.gid) + 1`.

    Raises
    ------
    ValueError
        If `method` is not one of the recognized options.

    Side Effects
    ------------
    Modifies the input `gs` object in-place by adding two attributes to each processed grain:

    - `gs.g[gid]['grain'].bbox_bz` : ndarray of bool
        Boolean mask of the boundary zone within the grain's bounding box. Shape matches 
        `grain.bbox`. Pixels marked True belong to the boundary zone.
    - `gs.g[gid]['grain'].bbox_core` : ndarray of bool or None
        Boolean mask of the core region. If the grain is too small or erosion eliminates 
        the core entirely, this is set to None.

    Notes
    -----
    - Uses `scipy.ndimage.binary_erosion` for morphological operations.
    - For grains with `bbox_core=None`, the entire grain bbox is considered boundary zone.
    - To avoid modifying the original grain structure, pass a deep copy:
      `lgi_new = introduce_boundary_zones(deepcopy(gs), ...)`
    - The returned `lgi_new` is for visualization only; actual boundary zone data is stored 
      in `gs.g[gid]['grain'].bbox_bz`.

    Examples
    --------
    >>> from upxo.pxtalops.grain_boundary_zones import introduce_boundary_zones
    >>> from copy import deepcopy
    >>> # Apply to all grains with minimum core size constraint
    >>> lgi_bz = introduce_boundary_zones(gs, method='use_min_core_size', min_core_size=100)
    >>> 
    >>> # Process specific grains with simple erosion
    >>> lgi_bz = introduce_boundary_zones(gs, fids=[1, 5, 10], method='simple', niterations=3)
    >>> 
    >>> # Preserve original structure
    >>> gs_copy = deepcopy(gs)
    >>> lgi_bz = introduce_boundary_zones(gs_copy, see_plot=True)

    See Also
    --------
    introduce_bz__use_min_core_size : Implementation of 'use_min_core_size' method.
    introduce_bz__simple : Implementation of 'simple' method.
    introduce_bz__thresholded : Implementation of 'thresholded' method.

    Import
    ------
    from upxo.pxtalops.grain_boundary_zones import introduce_boundary_zones

    Authors
    -------
    Dr. Sunil Anandatheertha
    """
    if len(fids) != 0:
        fids = fids
    else:
        fids = gs.gid
    # ------------------------------------------
    if method == 'use_min_core_size':
        lgi_new = introduce_bz__use_min_core_size(gs, fids=fids, niterations=niterations+1,
                        min_core_size=min_core_size,
                        structure_nx=structure_nx, structure_ny=structure_ny,
                        reset_others= reset_others,
                        renumber_fid_map=renumber_fid_map,
                        morph_char=morph_char, topo_char=topo_char,
                        remap_scalars=remap_scalars,
                        perform_topology_checks=perform_topology_checks,
                        see_plot=see_plot,
                        figure_size=figure_size, figure_dpi=figure_dpi)
    elif method == 'simple':
        lgi_new = introduce_bz__simple(gs, fids=fids, niterations=niterations,
                        structure_nx=structure_nx, structure_ny=structure_ny,
                        reset_others= reset_others,
                        renumber_fid_map=renumber_fid_map,
                        morph_char=morph_char, topo_char=topo_char,
                        remap_scalars=remap_scalars,
                        perform_topology_checks=perform_topology_checks,
                        see_plot=see_plot,
                        figure_size=figure_size, figure_dpi=figure_dpi)
    elif method == 'thresholded':
        lgi_new = introduce_bz__thresholded(gs, fids=fids, niterations=niterations,
                        threshold_bz_thickness=threshold_bz_thickness,
                        structure_nx=structure_nx, structure_ny=structure_ny,
                        reset_others= reset_others,
                        renumber_fid_map=renumber_fid_map,
                        morph_char=morph_char, topo_char=topo_char,
                        remap_scalars=remap_scalars,
                        perform_topology_checks=perform_topology_checks,
                        see_plot=see_plot,
                        figure_size=figure_size, figure_dpi=figure_dpi)
    else:
        raise ValueError(f"Method '{method}' is not recognized. Please choose from 'use_min_core_size', 'simple', or 'thresholded'.")

    return lgi_new

def introduce_bz__use_min_core_size(gs, fids=[], niterations=5, min_core_size=50, 
                            structure_nx=3, structure_ny=3, reset_others=False,
                            renumber_fid_map=True, morph_char=False, topo_char=False,
                            remap_scalars=False, perform_topology_checks=False,
                            see_plot=False, figure_size=(5, 5), figure_dpi=100):
    """
    Introduce boundary zones in the grains of a grain structure using minimum core size criterion.

    Parameters
    ----------
    gs : GrainStructure
        The grain structure object containing grains to process.
    fids : list, optional
        List of grain IDs to process. If empty, all grains will be processed.
    niterations : int, optional
        Number of erosion iterations to perform. Default is 5.
    min_core_size : int, optional
        Minimum core size to maintain during erosion. Default is 50.
    structure_nx : int, optional
        Size of the structuring element in x-direction. Default is 3.
    structure_ny : int, optional
        Size of the structuring element in y-direction. Default is 3.
    see_plot : bool, optional
        Whether to plot the resulting grain structure with boundary zones. Default is False.
    figure_size : tuple, optional
        Size of the plot figure. Default is (5, 5).
    figure_dpi : int, optional
        DPI of the plot figure. Default is 100.

    Returns
    -------
    lgi_new : ndarray
        The updated grain structure image with boundary zones introduced.

    Raises
    ------
    ValueError
        If the fids list is empty.
        
    Import
    -------
    from pxtalops.grain_boundary_zones import introduce_bz__use_min_core_size
    """
    if reset_others:
        # Ignore other fids not provided in fids list
        for fid in set(gs.gid)-set(fids):
            grain = gs.g[fid]['grain']
            grain.bbox_bz = None
            grain.bbox_core = None

    structure = np.ones((structure_nx, structure_ny), dtype=bool)
    if len(fids) == 0:
        raise ValueError("fids list is empty. Please provide a list of grain IDs to process.")
    for gid in fids:
        grain = gs.g[gid]['grain']
        g_bbox = deepcopy(grain.bbox)
        # Sinxce we do not know how many iterations will be possible, we store all intermediate results in a list.
        bz_all, _ = _erode_(niterations, g_bbox, structure)
        # Storing the last successful erosion as the boundary zone.
        if len(bz_all) == 0:
            grain.bbox_bz = g_bbox
            grain.bbox_core = None
        else:
            for bz_count, bz in enumerate(bz_all):
                core = g_bbox-bz
                n_pix_core = np.argwhere(core).shape[0]
                if n_pix_core < min_core_size:
                    break
            if bz_count == 0:
                gs.g[gid]['grain'].bbox_bz = g_bbox
                gs.g[gid]['grain'].bbox_core = None
            else:
                bz = bz_all[bz_count-1]
                gs.g[gid]['grain'].bbox_bz = bz
                gs.g[gid]['grain'].bbox_core = g_bbox-bz
    lgi_new = _update_gs_with_bz(gs)
    if see_plot:
        _plot_img_(lgi_new, figure_size=figure_size, figure_dpi=figure_dpi)
    return lgi_new

def introduce_bz__simple(gs, fids=[], niterations=5,
                structure_nx=3, structure_ny=3, reset_others=False,
                renumber_fid_map=True, morph_char=False, topo_char=False,
                remap_scalars=False, perform_topology_checks=False,
                see_plot=False, figure_size=(5, 5), figure_dpi=100):
    """
    Introduce boundary zones in the grains of a grain structure using simple erosion.

    Parameters
    ----------
    gs : GrainStructure
        The grain structure object containing grains to process.
    fids : list, optional
        List of grain IDs to process. If empty, all grains will be processed.
    niterations : int, optional
        Number of erosion iterations to perform. Default is 5.
    structure_nx : int, optional
        Size of the structuring element in x-direction. Default is 3.
    structure_ny : int, optional
        Size of the structuring element in y-direction. Default is 3.
    see_plot : bool, optional
        Whether to plot the resulting grain structure with boundary zones. Default is False.
    figure_size : tuple, optional
        Size of the plot figure. Default is (5, 5).
    figure_dpi : int, optional
        DPI of the plot figure. Default is 100.

    Returns
    -------
    lgi_new : ndarray
        The updated grain structure image with boundary zones introduced.

    Raises
    ------
    ValueError
        If the fids list is empty.
        
    Import
    -------
    from pxtalops.grain_boundary_zones import introduce_bz__simple
    """
    if reset_others:
        # Ignore other fids not provided in fids list
        for fid in set(gs.gid)-set(fids):
            grain = gs.g[fid]['grain']
            grain.bbox_bz = None
            grain.bbox_core = None

    structure = np.ones((structure_nx, structure_ny), dtype=bool)
    if len(fids) == 0:
        raise ValueError("fids list is empty. Please provide a list of grain IDs to process.")
    for gid in fids:
        grain = gs.g[gid]['grain']
        g_bbox = deepcopy(grain.bbox)
        # Sinxce we do not know how many iterations will be possible, we store all intermediate results in a list.
        bz_all, _ = _erode_(niterations, g_bbox, structure)
        # Storing the last successful erosion as the boundary zone.
        if len(bz_all) == 0:
            grain.bbox_bz = g_bbox
            grain.bbox_core = None
        else:
            bz = bz_all[-1]
            grain.bbox_bz = bz
            grain.bbox_core = g_bbox-bz
    lgi_new = _update_gs_with_bz(gs)
    if see_plot:
        _plot_img_(lgi_new, figure_size=figure_size, figure_dpi=figure_dpi)
    return lgi_new

def introduce_bz__thresholded(gs, fids=[], niterations=5, threshold_bz_thickness=2,
                    structure_nx=3, structure_ny=3, reset_others=False,
                    renumber_fid_map=True, morph_char=False, topo_char=False,
                    remap_scalars=False, perform_topology_checks=False,
                    see_plot=False, figure_size=(5, 5), figure_dpi=100):
    """
    Introduce boundary zones in the grains of a grain structure using thresholded erosion.

    Parameters
    ----------
    gs : GrainStructure
        The grain structure object containing grains to process.
    fids : list, optional
        List of grain IDs to process. If empty, all grains will be processed.
    niterations : int, optional
        Number of erosion iterations to perform. Default is 5.
    threshold_bz_thickness : int, optional
        Threshold for boundary zone thickness. Default is 2.
    structure_nx : int, optional
        Size of the structuring element in x-direction. Default is 3.
    structure_ny : int, optional
        Size of the structuring element in y-direction. Default is 3.
    see_plot : bool, optional
        Whether to plot the resulting grain structure with boundary zones. Default is False.
    figure_size : tuple, optional
        Size of the plot figure. Default is (5, 5).
    figure_dpi : int, optional
        DPI of the plot figure. Default is 100.

    Returns
    -------
    lgi_new : ndarray
        The updated grain structure image with boundary zones introduced.

    Raises
    ------
    ValueError
        If the fids list is empty.
        
    Import
    -------
    from pxtalops.grain_boundary_zones import introduce_bz__thresholded
    """
    if reset_others:
        # Ignore other fids not provided in fids list
        for fid in set(gs.gid)-set(fids):
            grain = gs.g[fid]['grain']
            grain.bbox_bz = None
            grain.bbox_core = None

    structure = np.ones((structure_nx, structure_ny), dtype=bool)
    if len(fids) == 0:
        raise ValueError("fids list is empty. Please provide a list of grain IDs to process.")
    for gid in fids:
        grain = gs.g[gid]['grain']
        g_bbox = deepcopy(grain.bbox)
        # Sinxce we do not know how many iterations will be possible, we store all intermediate results in a list.
        bz_all, i = _erode_(niterations, g_bbox, structure)
        # Storing the last successful erosion as the boundary zone.
        if len(bz_all) == 0 or i <= threshold_bz_thickness:
            grain.bbox_bz = g_bbox
            grain.bbox_core = None
        else:
            bz = bz_all[-1]
            grain.bbox_bz = bz
            grain.bbox_core = g_bbox - bz
    lgi_new = _update_gs_with_bz(gs)
    if see_plot:
        _plot_img_(lgi_new, figure_size=figure_size, figure_dpi=figure_dpi)
    return lgi_new

def _erode_(niterations, g_bbox, structure):
    """
    Erode the grain bounding box for a specified number of iterations.

    Parameters
    ----------
    niterations : int
        Number of erosion iterations to perform.
    g_bbox : ndarray
        The grain bounding box to erode.
    structure : ndarray
        The structuring element used for erosion.

    Returns
    -------
    bz_all : list
        List of binary arrays representing the boundary zones after each iteration.
    i : int
        The number of iterations completed.
    """
    bz_all = []
    for i in np.arange(1, niterations+1):
        bz = binary_erosion(g_bbox, iterations=i, structure=structure)
        if not np.any(bz):
            break
        bz_all.append(bz)
    return bz_all, i

def _update_gs_with_bz(gs):
    """
    Update the grain structure image with boundary zones.

    Parameters
    ----------
    gs : GrainStructure
        The grain structure object containing grains with boundary zones.

    Returns
    -------
    lgi_new : ndarray
        The updated grain structure image with boundary zones introduced.
    """
    lgi_new = gs.lgi.copy()
    gid_max = max(gs.gid) + 1
    
    for gid in gs.gid:
        grain = gs.g[gid]['grain']
        bounds = grain.bbox_bounds
        if not hasattr(grain, 'bbox_bz') or grain.bbox_core is None:
            continue
        for r, c in np.argwhere(grain.bbox_bz):
            lgi_new[r+bounds[0], c+bounds[2]] = gid_max
        gid_max += 1
    return lgi_new

def _plot_img_(image, figure_size=(5, 5), figure_dpi=100):
    """
    Plot the given image.

    Parameters
    ----------
    image : ndarray
        The image to plot.

    figure_size : tuple, optional
        Size of the plot figure. Default is (5, 5).

    figure_dpi : int, optional
        DPI of the plot figure. Default is 100.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figure_size, dpi=figure_dpi)
    plt.imshow(image, cmap='viridis')

def determine_buffer_geometric_GS(cell, min_cell_area=5, buffer_quality_factor=0.25, 
                min_area_retention=0.30, max_allowed_holes=2,
                max_iterations=20, verbose=True):
    """
    Determine an appropriate buffer distance for creating a boundary zone within a grain cell
    using geometric properties of the cell. The function iteratively tests buffer distances to
    ensure that the resulting buffered cell does not create too many holes (disconnected interiors).

    Parameters
    ----------
    cell : shapely.geometry.Polygon
        The polygon representing the grain cell to buffer.
    buffer_quality_factor : float, default=0.25
        Fraction of the inscribed radius to use as the initial buffer distance. Higher values
        may create thicker boundary zones but risk creating holes. Adjust based on desired balance between boundary zone thickness and core integrity.
    min_area_retention : float, default=0.30
        Minimum fraction of the original cell area that must be retained in the buffered cell. This is used to calculate a maximum buffer distance based on area loss.
    max_allowed_holes : int, default=2
        Maximum number of holes (disconnected interiors) allowed in the buffered cell. Set to 0 for no holes, or a higher number if some fragmentation is acceptable.
    max_iterations : int, default=20
        Maximum number of iterations to test different buffer distances. The buffer distance is reduced iteratively if
        the resulting buffered cell has too many holes or is empty.
    verbose : bool, default=True
        If True, prints detailed information about the buffering process, including geometric properties, buffer distances tested, and hole counts.

    Returns
    -------
    bufferDist : float or None
        The final buffer distance that meets the criteria, or None if no valid buffer distance was found.
    bufferCell : list of shapely.geometry.Polygon or None
        List of polygons representing the buffered cell structure:
        - If no holes created: list with single polygon (the buffered cell)
        - If holes created: list with outer polygon (index 0) followed by each hole as a separate polygon
        - None if no valid buffer distance was found.
    boundaryZone : shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
        The boundary zone region (original cell minus buffered cell). May be a MultiPolygon if holes were created.
        None if no valid buffer distance was found.

    Import
    ------
    from upxo.pxtalops.grain_boundary_zones import determine_buffer_geometric_GS as find_buffer_geom
    """
    # Get the cell geometry
    cell_area = cell.area
    if cell_area < min_cell_area:
        if verbose:
            print(f"Cell area {cell_area:.2f} is below minimum threshold {min_cell_area}. Skipping buffering.")
        return 0.0, None, cell
    cell_perimeter = cell.length

    # Calculate characteristic dimension: equivalent radius (radius of circle with same area)
    equivalent_radius = np.sqrt(cell_area / np.pi)

    # Calculate inscribed circle radius approximation (more conservative measure)
    compactness = 4 * np.pi * cell_area / (cell_perimeter ** 2)  # 1.0 for circle, < 1 for irregular
    inscribed_radius_estimate = equivalent_radius * np.sqrt(compactness)

    # Initial buffer distance as fraction of inscribed radius
    bufferDist_geometric = buffer_quality_factor * inscribed_radius_estimate

    # Validate using area constraint
    target_inner_area = min_area_retention * cell_area
    max_area_loss = cell_area - target_inner_area
    max_buffer_from_area = max_area_loss / cell_perimeter if cell_perimeter > 0 else 0

    # Starting buffer distance
    bufferDist = min(bufferDist_geometric, max_buffer_from_area)
    bufferDist = max(0.01, bufferDist)

    # Iteratively test buffer distance to control hole creation
    # max_allowed_holes = 1 if allow_holes else 0
    valid_buffer_found = False
    iteration = 0
    reduction_factor = 0.9  # Reduce buffer by 10% each iteration

    original_bufferDist = bufferDist
    tested_bufferCell = None

    while iteration < max_iterations and bufferDist > 0.01:
        # Test buffer operation
        tested_bufferCell = cell.buffer(-bufferDist)
        
        # Check if result is valid (not empty)
        if tested_bufferCell.is_empty:
            # Buffer too large, reduce
            bufferDist *= reduction_factor
            iteration += 1
            continue
        
        # Count interior holes (interiors in shapely Polygon)
        if tested_bufferCell.geom_type == 'Polygon':
            num_holes = len(tested_bufferCell.interiors)
        elif tested_bufferCell.geom_type == 'MultiPolygon':
            # MultiPolygon indicates fragmentation - treat as invalid
            num_holes = float('inf')
        else:
            num_holes = 0
        
        # Check if hole count is acceptable
        if num_holes <= max_allowed_holes:
            valid_buffer_found = True
            break
        else:
            # Too many holes, reduce buffer distance
            bufferDist *= reduction_factor
            iteration += 1

    # Report results
    if verbose:
        print(f"Cell area: {cell_area:.2f}")
        print(f"Equivalent radius: {equivalent_radius:.3f}")
        print(f"Compactness: {compactness:.3f}")
        print(f"Inscribed radius (est): {inscribed_radius_estimate:.3f}")
        print(f"Initial buffer distance: {original_bufferDist:.3f}")
        print(f"Iterations: {iteration}")

    if valid_buffer_found:
        if verbose:
            print(f"✓ Valid buffer found: {bufferDist:.3f}")
            print(f"  Buffer / Inscribed radius ratio: {bufferDist / inscribed_radius_estimate:.2%}")
            if tested_bufferCell.geom_type == 'Polygon':
                print(f"  Number of holes: {len(tested_bufferCell.interiors)}")
        
        # Convert bufferCell to list of polygons
        # If no holes: return list with single polygon (the buffered cell)
        # If holes exist: return list with outer polygon + each hole as separate polygons
        from shapely.geometry import Polygon as ShapelyPolygon
        
        if tested_bufferCell.geom_type == 'Polygon' and len(tested_bufferCell.interiors) > 0:
            # Separate outer boundary and holes
            bufferCell = []
            # Add outer polygon (no holes)
            outer_poly = ShapelyPolygon(tested_bufferCell.exterior)
            bufferCell.append(outer_poly)
            # Add each hole as a separate polygon
            for interior in tested_bufferCell.interiors:
                hole_poly = ShapelyPolygon(interior)
                bufferCell.append(hole_poly)
            if verbose:
                print(f"  Separated into {len(bufferCell)} polygons: 1 outer + {len(bufferCell)-1} hole(s)")
        else:
            # No holes, return single polygon in list
            bufferCell = [tested_bufferCell]
        
        boundaryZone = cell - tested_bufferCell
    else:
        if verbose:
            print(f"✗ No valid buffer found (holes exceed limit)")
            print(f"  Cell buffering skipped - set allow_holes=True or adjust buffer_quality_factor")
        bufferDist = None
        bufferCell = None
        boundaryZone = None

    return bufferDist, bufferCell, boundaryZone

def process_grain_boundary_zones(smoothed_grains, ignoreFids=[], min_cell_area=80, 
                                 buffer_quality_factor=0.55, 
                                 min_area_retention=0.30,
                                 max_allowed_holes=2, 
                                 max_iterations=20, 
                                 verbose=False):
    """
    Orchestrator function to process all grains and create boundary zones.
    
    Parameters
    ----------
    smoothed_grains : dict
        Dictionary of grain IDs to Shapely Polygon geometries
    min_cell_area : int, default=80
        Minimum cell area threshold for processing
    buffer_quality_factor : float, default=0.55
        Buffer quality factor for geometric calculation
    min_area_retention : float, default=0.30
        Minimum area fraction to retain in buffered cell
    max_allowed_holes : int, default=2
        Maximum number of holes allowed in buffered cell
    max_iterations : int, default=20
        Maximum iterations for buffer refinement
    verbose : bool, default=False
        Print detailed processing information
    
    Returns
    -------
    BZ_1_cells : dict
        Dictionary mapping grain IDs to boundary zone polygons
    BZ_1_thickness : dict
        Dictionary mapping grain IDs to buffer distances
    CZ_1_cells : dict
        Dictionary mapping grain IDs to core zone polygon lists
    CZ1 : list
        Flat list of all core zone polygons
    combined_multipolygon : shapely.geometry.MultiPolygon
        MultiPolygon containing all valid boundary zones and core zones for visualization

    Import
    ------
    from upxo.pxtalops.grain_boundary_zones import process_grain_boundary_zones
    """
    from shapely.geometry import MultiPolygon

    # Initialize storage dictionaries
    BZ_1_cells = {pcid: None for pcid in smoothed_grains.keys()}
    BZ_1_thickness = {pcid: None for pcid in smoothed_grains.keys()}
    CZ_1_cells = {pcid: None for pcid in smoothed_grains.keys()}
    
    # Process each grain
    for pcid, cell in smoothed_grains.items():
        if pcid in ignoreFids:
            if verbose:
                print(f"Skipping grain ID {pcid} (in ignoreFids)")
                BZ_1_cells[pcid] = cell
                BZ_1_thickness[pcid] = 0.0
                CZ_1_cells[pcid] = None
            continue
        buffDist, buffPol, boundaryZone = determine_buffer_geometric_GS(
            cell, 
            min_cell_area=min_cell_area, 
            buffer_quality_factor=buffer_quality_factor,  
            min_area_retention=min_area_retention,
            max_allowed_holes=max_allowed_holes, 
            max_iterations=max_iterations, 
            verbose=verbose
        )
        BZ_1_cells[pcid] = boundaryZone
        BZ_1_thickness[pcid] = buffDist
        CZ_1_cells[pcid] = buffPol
    
    # Build flat list of all core zones for visualization
    CZ1 = []
    for cz in CZ_1_cells.values():
        if cz is not None:
            for _cz_ in cz:
                CZ1.append(_cz_)
    
    # Create combined MultiPolygon for visualization
    BZ1_valid = [bz for bz in BZ_1_cells.values() if bz is not None]
    CZ1_valid = [cz for cz in CZ1 if cz is not None]
    all_polygons = BZ1_valid + CZ1_valid
    
    combined_multipolygon = MultiPolygon(all_polygons) if len(all_polygons) > 0 else None
    
    return BZ_1_cells, BZ_1_thickness, CZ_1_cells, CZ1, combined_multipolygon
