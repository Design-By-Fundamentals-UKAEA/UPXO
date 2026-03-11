import numpy as np

def create_grid_points_in_polygon(polygon, grid_spacing=1.0, padding=0.0,
                                   return_classification=True):
    """
    Create a rectangular grid based on polygon bounds and identify points inside/on boundary.
    
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The boundary zone polygon to process
    grid_spacing : float, default=1.0
        Spacing between grid points in both x and y directions
    padding : float, default=0.0
        Additional padding to extend grid beyond polygon bounds
    return_classification : bool, default=True
        If True, returns separate arrays for interior and boundary points.
        If False, returns all points inside or on boundary.
    
    Returns
    -------
    grid_points : ndarray, shape (N, 2)
        All grid points within the bounding box
    interior_points : ndarray, shape (M, 2)
        Points strictly inside the polygon (only if return_classification=True)
    boundary_points : ndarray, shape (K, 2)
        Points on the polygon boundary (only if return_classification=True)
    points_mask : ndarray, shape (N,) dtype=bool
        Boolean mask indicating which grid points are inside or on boundary
        (only if return_classification=False)
    
    Examples
    --------
    >>> # Get all points inside/on boundary
    >>> grid_pts, mask = create_grid_points_in_polygon(bz, grid_spacing=0.5, 
    ...                                                 return_classification=False)
    >>> valid_pts = grid_pts[mask]
    
    >>> # Get separated interior and boundary points
    >>> grid_pts, interior_pts, boundary_pts = create_grid_points_in_polygon(bz, 
    ...                                                                       grid_spacing=0.5)

    Import
    ------
    from upxo.xtalops.xtalops import create_grid_points_in_polygon
    """
    from shapely.geometry import Point
    
    # Get bounds (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = polygon.bounds
    
    # Apply padding
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    
    # Create grid
    x_coords = np.arange(minx, maxx + grid_spacing, grid_spacing)
    y_coords = np.arange(miny, maxy + grid_spacing, grid_spacing)
    
    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flatten to get all grid points
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Check which points are inside or on the boundary
    interior_mask = np.zeros(len(grid_points), dtype=bool)
    boundary_mask = np.zeros(len(grid_points), dtype=bool)
    
    for i, (x, y) in enumerate(grid_points):
        pt = Point(x, y)
        
        # Check if point is in polygon (interior or boundary)
        if polygon.contains(pt):
            interior_mask[i] = True
        elif polygon.boundary.distance(pt) < 1e-9:  # On boundary (numerical tolerance)
            boundary_mask[i] = True
    
    if return_classification:
        interior_points = grid_points[interior_mask]
        boundary_points = grid_points[boundary_mask]
        return grid_points, interior_points, boundary_points
    else:
        # Combined mask for all points inside or on boundary
        combined_mask = interior_mask | boundary_mask
        return grid_points, np.reshape(combined_mask, xx.shape), xx, yy


def skeletonize_polygon(polygon, method='medial_axis', grid_spacing=0.5, padding=0.1):
    """
    Import
    ------
    from upxo.xtalops.xtalops import skeletonize_polygon as skpol
    """

    # from upxo.xtalops.xtalops import create_grid_points_in_polygon
    grid_points, combined_mask, xx, yy = create_grid_points_in_polygon(polygon, 
                                        grid_spacing=grid_spacing, padding=padding,
                                        return_classification=False)

    if method == 'medial_axis':
        from skimage.morphology import medial_axis
        skeleton = medial_axis(combined_mask)
    elif method == 'skeletonize':
        from skimage.morphology import skeletonize
        skeleton = skeletonize(combined_mask)
    elif method == 'skeletonize_lee':
        from skimage.morphology import skeletonize
        skeleton = skeletonize(combined_mask, method='lee')
    else:   
        raise ValueError("Invalid method. Choose 'medial_axis', 'skeletonize', or 'skeletonize_lee'.")
    
    sklCoords = np.vstack((xx[skeleton], yy[skeleton])).T

    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neighbor_count = convolve(skeleton.astype(int), kernel, mode="constant", cval=0)
    # Branch points: skeleton pixels with 3+ neighbors
    branch_mask = (skeleton & (neighbor_count >= 3))
    branch_coords = np.argwhere(branch_mask)


    from scipy.ndimage import label
    # Remove branch points to split skeleton into segments
    skeleton_wo_branches = skeleton.copy()
    for r, c in branch_coords:
        skeleton_wo_branches[r, c] = False
    # Label connected components (8-connectivity)
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    labeled, num_segments = label(skeleton_wo_branches, structure=structure)
    # Extract segment coordinates
    segments = []
    for seg_id in range(1, num_segments + 1):
        seg_coords = np.argwhere(labeled == seg_id)
        if seg_coords.size:
            segments.append(seg_coords)

    return skeleton, branch_coords, segments, labeled, sklCoords, xx, yy