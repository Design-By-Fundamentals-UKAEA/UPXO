import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from collections import defaultdict
import upxo.gsdataops.grid_ops as gridOps

def PL_cell_boundaries(lfi=None, nfeatures=None, neigh_fid=None,
                       connectivity=1, mode='thick', background=0,
                       local_seg_id_nDecPlaces=4, segIDMask_dtype=np.int32):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    gbops2d.PL_cell_boundaries(lfi=pxt.gs[3].lgi, 
                           nfeatures=pxt.gs[3].n,
                           neigh_fid=pxt.gs[3].neigh_gid,
                           connectivity=1,
                           mode='thick',
                           background=0,
                           local_seg_id_nDecPlaces=4,
                           segIDMask_dtype=np.int32
                           )
    """
    lfi_boundaries = detect_cell_boundaries(lfi=lfi, nfeatures=nfeatures,
                           connectivity=connectivity,
                           mode=mode, background=background)
    # --------------------------------------------------------------
    segInfo = segment_cell_boundaries(lfi=lfi,  lfi_boundaries=lfi_boundaries,
                            neigh_fid=neigh_fid, connectivity=connectivity,
                            local_seg_id_nDecPlaces=local_seg_id_nDecPlaces)
    # --------------------------------------------------------------
    bseg_props = characterise_boundary_segments(segInfo['bsegCoords'], neigh_fid)
    # --------------------------------------------------------------
    return lfi_boundaries, segInfo, bseg_props

def PL_cellb_junction_points(bsegCoords, segidList_gbl):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    junctionPoints, JPSorted, JPOStats = gbops2d.PL_cellb_junction_points(segInfo['bsegCoords'], 
                                                  segInfo['segidList_gbl'])
    """
    junctionPoints =  detect_junction_points(bsegCoords)
    JPSorted = sort_junction_points(junctionPoints)
    JPOStats = find_basic_JPO_stats(junctionPoints, segidList_gbl)
    return junctionPoints, JPSorted, JPOStats

def detect_cell_boundaries(lfi=None, nfeatures=None, connectivity=1,
                           mode='thick', background=0,):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    lfi_boundaries = gbops2d.detect_cell_boundaries(lfi=pxt.gs[3].lgi, 
                                nfeatures=pxt.gs[3].n, connectivity=1,
                                mode='thick', background=0)
    """
    lfi_padded = pad_lfi(lfi=lfi, pad_value=nfeatures+1, _pad_width=1)
    '''lfi_padded = np.pad(lfi, 1, pad_with, padder=nfeatures+1)'''
    lfi_boundaries = find_boundaries(lfi_padded, connectivity=connectivity,
                            mode=mode, background=background)*lfi_padded
    lfi_boundaries = lfi_boundaries[1:-1, 1:-1]
    return lfi_boundaries

def pad_lfi(lfi=None, pad_value=-100, _pad_width=1):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    lfi_padded = gbops2d.pad_lfi(lfi=pxt.gs[3].lgi, pad_value=0)

    Note
    ----
    _pad_width is set to 1 by default. There is no need to change this.
    """
    lfi_padded = np.pad(lfi, _pad_width, pad_with, padder=pad_value)
    return lfi_padded

def find_common_interface_boundaries(lfi=None, nfeatures=None, connectivity=1):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    lfi_padded = gbops2d.find_common_interface_boundaries(lfi=pxt.gs[3].lgi,
                                nfeatures=None, connectivity=1)
    """
    lfi_padded = pad_lfi(lfi=lfi, pad_value=nfeatures+1)
    common_interface = find_boundaries(lfi_padded, connectivity=connectivity, 
                                       mode='subpixel', background=0)
    return common_interface

def segment_cell_boundaries(lfi=None, lfi_boundaries=None,
                            neigh_fid=None, connectivity=1,
                            local_seg_id_nDecPlaces=4):
    """
    Example
    -------
    import upxo.gbops.mcgb2dops as gbops2d
    # First we need to detect the cell boundaries
    lfi_boundaries = gbops2d.detect_cell_boundaries(lfi=pxt.gs[3].lgi, 
                                nfeatures=pxt.gs[3].n, connectivity=1,
                                mode='thick', background=0)
    # Now we can segment these detected boundaries
    funcOut = gbops2d.segment_cell_boundaries(lfi=pxt.gs[3].lgi, 
                                    lfi_boundaries=lfi_boundaries,
                                    neigh_fid=pxt.gs[3].neigh_gid,
                                    connectivity=1,
                                    local_seg_id_nDecPlaces=4)
    lfi_boundaries, bsegCoords, local_seg_ids, global_seg_ids, segidList_lcl, segidList_gbl = funcOut
    """
    # --------------------------------------------------------------
    # Segment the grain lfi_boundaries
    bsegCoords = segment_existing_boundaries(lfi, lfi_boundaries,
                                           connectivity=4*connectivity)
    bsegCoords = {cg: {ng: bsegCoords[(cg, ng)] for ng in ngs} 
                for cg, ngs in neigh_fid.items()}
    # --------------------------------------------------------------
    # Basic characterize the cell feature boundaries
    bseg_props = characterise_boundary_segments(bsegCoords, neigh_fid)
    nsegments = bseg_props['nsegments']
    # --------------------------------------------------------------
    # Perform segmentation success check
    nneighs = {cg: len(ngs) for cg, ngs in neigh_fid.items()}
    # --------------------------------------------------------------
    # Report segmentation success
    if not np.any(nsegments != nneighs):
        print("Segmentation successfull", 
              "Number of bsegCoords equals number of nearest neighs for all features.")
    else:
        print("Segmentation issues detected!",
              "Number of bsegCoords NOT equal to number of nearest neighs for some features.")
    # --------------------------------------------------------------
    # Assign a new sub-ID to every segment based on the parent feature ID
    local_seg_ids = {cg: np.round(np.linspace(cg, cg+1, nsegments[cg]+1)[:-1],
                                  local_seg_id_nDecPlaces) 
                                  for cg, ngs in neigh_fid.items()}
    # --------------------------------------------------------------
    # Find the gloabl segment ID
    global_seg_ids = {}
    segidList_lcl = []  # List of all local IDs
    segidList_gbl = []  # List of all global IDs
    segid_gbl_pfid_map = {}
    seg_count_id = 1
    for parent_fid, child_fids in local_seg_ids.items():
        global_seg_ids[parent_fid] = []
        segNum = 1
        for child_fid in child_fids:
            global_seg_ids[parent_fid].append(seg_count_id)
            segidList_lcl.append(child_fid)
            segidList_gbl.append(segNum)
            segid_gbl_pfid_map[segNum] = parent_fid
            segNum += 1
            seg_count_id += 1
    # --------------------------------------------------------------
    # Collect all tre results and return
    segInfo = {'bsegCoords': bsegCoords,
               'local_seg_ids': local_seg_ids, 
               'global_seg_ids': global_seg_ids,
               'segidList_lcl': segidList_lcl,
               'segidList_gbl': segidList_gbl,
               'segid_gbl_pfid_map': segid_gbl_pfid_map
               }
    return segInfo

def see_bsegs_gids(localSegIDMasked_lfi, gid):
    """
    gid = np.argmax(pxt.gs[3].prop.npixels.to_numpy())+1
    """
    scaled_mask = localSegIDMasked_lfi*np.asarray(localSegIDMasked_lfi >= gid, dtype=float)*np.asarray(localSegIDMasked_lfi < gid+1, dtype=float)
    scaled_mask[scaled_mask < gid]= np.nan
    plt.figure(figsize=(10,10), dpi=50)
    plt.imshow(scaled_mask, cmap='nipy_spectral')
    plt.colorbar()
    plt.show()

def characterise_boundary_segments(bsegCoords, neigh_fid):
    """Characterise the detected cell boundary segments

    Example
    -------
    bseg_props = characterise_boundary_segments(bsegCoords, neigh_fid)
    """
    # Calculate the number of segments in each fearture (grain)
    nsegments = {cg: len(bsegCoords[cg]) for cg in neigh_fid.keys()}
    # Calculate segment lengths
    segment_lengths = {cg: [len(seg) for ng, seg in bsegCoords[cg].items()] 
                       for cg in bsegCoords.keys()}
    bseg_props = {'nsegments': nsegments, 
                  'segment_lengths': segment_lengths}
    return bseg_props

def detect_junction_points(segments):
    """
    Example
    -------
    junctionPoints = detect_junction_points(segInfo['bsegCoords'])
    """
    all_junctions = {}
    for cg, neighbors in segments.items():
        all_pts = [pts for pts in neighbors.values() if pts.size > 0]
        if not all_pts:
            continue
        stacked_pts = np.vstack(all_pts)
        unique_pts, counts = np.unique(stacked_pts, axis=0, return_counts=True)
        junction_mask = counts >= 2
        if np.any(junction_mask):
            res = np.column_stack((unique_pts[junction_mask], counts[junction_mask] + 1))
            all_junctions[cg] = res
    return all_junctions

def find_basic_JPO_stats(junctionPoints, segidList_gbl):
    """
    Example
    -------
    JPOStats = find_basic_JPO_stats(junctionPoints, segInfo['segidList_gbl'])
    """
    # Determine the min, max, median Junction Point Order, feature wise
    minJPO_feat = {gid: jp[:, 2].min() for gid, jp in junctionPoints.items()}
    maxJPO_feat = {gid: jp[:, 2].max() for gid, jp in junctionPoints.items()}
    medianJPO_feat = {gid: np.median(jp[:,2]) for gid, jp in junctionPoints.items()}
    # Determine the min, max, median Junction Point Order, feature wise
    minJPO_tess = min(list(minJPO_feat.values()))
    maxJPO_tess = max(list(maxJPO_feat.values()))
    medianJPO_tess = np.zeros(len(segidList_gbl))
    for jp in junctionPoints.values():
        seg_count = 0
        for _jpo_ in jp[:, 2]:
            medianJPO_tess[seg_count] = _jpo_
    medianJPO_tess = np.median(medianJPO_tess)
    JPOStats = {'min_tess': minJPO_tess, 'max_tess': maxJPO_tess, 'median_tess': medianJPO_tess,
                'min_feat': minJPO_feat, 'max_feat': maxJPO_feat, 'median_feat': medianJPO_feat,}
    return JPOStats

def sort_junction_points(junctionPoints):
    """
    Example
    -------
    JPSorted = sort_junction_points(junctionPoints)
    """
    jpo_names = {1: 'jp1', 2: 'jp2', 3: 'tjp', 4: 'qjp', 5: 'jp5', 6: 'jp6'}
    
    # Find min/max junction point order efficiently
    all_jpo_values = np.concatenate([jp[:, 2] for jp in junctionPoints.values()])
    min_jpo = int(all_jpo_values.min())
    max_jpo = int(all_jpo_values.max())
    # Build sorted dictionary for each junction point order
    JPSorted = {}
    for jpo in range(min_jpo, max_jpo + 1):
        jpo_dict = {}
        for gid, jp in junctionPoints.items():
            filtered = jp[jp[:, 2] == jpo, 0:2]
            if filtered.shape[0] > 0:
                # Only include if grain has junction points of this order
                jpo_dict[gid] = filtered
        JPSorted[jpo_names[jpo]] = jpo_dict
    return JPSorted

def mask_featIDImg_at_coords(featIDImg, coordData,
                             featName='fbseg', maskDType=np.int32):
    """
    Example
    -------
    mask_featIDImg_at_coords(lfi_boundaries, coordinates, featName='fbseg',
                             maskDType=np.int32)
    """
    if featName in ('fbseg', 'gbseg', 'feature_boudnary_segments'):
        featIDImg_new = np.asarray(deepcopy(featIDImg), dtype=maskDType)
        for cg, ngcoords in coordData.items():
            for i, (ng, ngsegcoords) in enumerate(ngcoords.items()):
                segid = coordData[cg][i]
                for sc in ngsegcoords:
                    featIDImg_new[sc[0]][sc[1]] = segid
    return featIDImg_new

def pad_with(vector, pad_width, iaxis, kwargs):
    # Taken verbattim from the bwloe reference
    # REF: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def segment_existing_boundaries(labels, gb_mask, connectivity=8):
    """
    Import
    ------
    import upxo.gbops.mcgb2dops as gbops2d
    Use as: gbops2d.segment_existing_boundaries
    """
    rows, cols = np.where(gb_mask)
    interfaces = defaultdict(list)
    if connectivity == 4:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    labels_int = labels.astype(int)
    max_r, max_c = labels.shape
    for r, c in zip(rows, cols):
        g1 = labels_int[r, c]
        if g1 == 0: continue
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < max_r and 0 <= nc < max_c:
                g2 = labels_int[nr, nc]
                if g2 != 0 and g2 != g1:
                    interfaces[(int(g1), int(g2))].append((r, c))
    return {pair: np.unique(np.array(pts), axis=0) for pair, pts in interfaces.items()}


def circular_sort(coords=None,
                  order='ccw',
                  origin_spec='tl',
                  origin_point=(0, 0)
                  ):
    """
    This sorts the grain boundary points in circular order. The grain boundary
    points are specified by coords. Circular sorting order is specified by
    order. Point to start sorting from is specified by the origin.

    Parameters
    ----------
    coords : np.array, optional
        np.vstack of (x, y) corrdinates of points. The default is None.
    order : str, optional
        Specifies circular sorting order. Options include (cw, clockwise) and
        (ccw, acw, counterclockwise, anticlockwise). The default is 'ccw'.
    origin_spec : str, optional
        Specifies the starting point location of the sorted point array.
        Options include 'tl', 'tr', 'br', 'bl', 'l', 't', 'r', 'b', 'closest'
        and 'ignore'. The default is 'tl'. They stand for 'tl': top-left,
        'tr': top-right, 'br': bottom-right, 'bl': bottom-left, 'l': leftmost,
        't': topmost, 'r': right-most, 'b': bottom-most, 'closest': the point
        in the given coords closest to origin_point. If 'ignore': the
        origin_point will be used.
    origin_point : tuple/list, optional
        Helps determine the first point of the sorted list. If origin_spec is
        set to 'ignore', origin_point will be used as the first point. It must
        be a member of coords input point. If the origin_spec is set to
        'closest', distances between origin_point and all coords points will
        be first calculated and the coord point closest to the origin_point
        will be used as the starting point. If more than one coord point has
        the  same distance to the origin_point, then the top-left point of
        these sub-set will be used as the starting point.
    method: int, optional
        Specifies the method of sorting. Three methods are offered, 1, 2 & 3.
    """
    # STEP-1: UNPACK jp_grainwise
    circular_sort_method1(coords)

    if method==2:
        pass

    if method==3:
        pass

def circular_sort_method1(jp_grainwise,
                            sort_order,
                            method='method1',
                            debug_mode=True):
    """
    DOC-STRING HERE

    Author
    ------
    Dr. Sunil Anandatheertha

    Example pre-requisites
    ----------------------
    from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs
    PXGS = mcgs()
    PXGS.simulate()
    PXGS.detect_grains()
    from upxo.gbops.mcgb2dops import sort_gb_junction_points

    Examples
    --------
    sort_gb_junction_points(PXGS.gs[8], sort_order)
    """
    """ VALIDATE coords """
    """ VALIDATE sort_order """
    _default_sorting_method = 'method1'
    # BUild the return variable structure
    _jp_sorted = {gid: {'coords': None,
                        'indices': None} for gid in jp_grainwise.keys()}
    jp_sorted = {
        'method': method,
        'sortorder': sort_order if method == 'method2'
        else 'cw' if method == 'method1' else _default_sorting_method,
        'sorted': _jp_sorted}
    # Sort the coordinates clockwise or anti-clockwise
    for gid, coords in jp_grainwise.items():
        if coords[0].size == 1:  # No sorting for a single point !!
            jp_sorted['values'][gid]['coords'] = coords
            jp_sorted['values'][gid]['indices'] = 0
        elif coords[0].size > 1:
            if debug_mode:
                print('\n', 10*'-', '\n', 'x:', coords[0], 'y:', coords[1])
            if method == 'method1':
                if coords[0].size == 2:
                    sorted_coords = coords
                    sorted_indices = np.array([0, 1])
                elif coords[0].size > 2:
                    sorted_coords, sorted_indices = method1(coords)
            elif method == 'method2':
                sorted_coords, sorted_indices = method2(coords)
            if debug_mode:
                print('Sorted coords: X.',
                      sorted_coords[0],
                      'Y:', sorted_coords[1])
            jp_sorted['values'][gid]['coords'] = sorted_coords
            jp_sorted['values'][gid]['indices'] = sorted_indices
    return jp_sorted


def method1(coords):
    """
    This method is borrowed heavily from the below source
    Source: https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
    """
    pts = coords.T
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order
    # method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    print(rightMost)
    (tr, br) = rightMost

    sorted_indices = None
    sorted_coords = np.array([tl, tr, br, bl])
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return sorted_coords, sorted_indices


def method2(coords):
    # Step 1: Find the leftmost (and then topmost) point
    leftmost_point = coords[np.lexsort((coords[:, 1],
                                        coords[:, 0]))][0]
    # Step 2: Center the points around the leftmost point
    centered_coords = coords - leftmost_point
    # Step 3: Calculate angles
    angles = np.arctan2(centered_coords[:, 1],
                        centered_coords[:, 0])
    # Step 4: Sort by angles
    # Note: np.arctan2 returns angles in [-pi, pi], negative angles
    # are in the 3rd and 4th quadrants
    # Sorting in ascending order effectively gives us a
    # counterclockwise direction, so we might need to adjust based
    # on clockwise requirement.
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords, sorted_indices


def method3_2d(coords=None,
               order='ccw',
               origin_spec='tl',
               origin_point=(0, 0),
               method=3
               ):
    """
    This sorts the grain boundary points in circular order. The grain boundary
    points are specified by coords. Circular sorting order is specified by
    order. Point to start sorting from is specified by the origin.

    Parameters
    ----------
    coords : np.array, optional
        np.vstack of (x, y) corrdinates of points. The default is None.
    order : str, optional
        Specifies circular sorting order. Options include (cw, clockwise) and
        (ccw, acw, counterclockwise, anticlockwise). The default is 'ccw'.
    origin_spec : str, optional
        Specifies the starting point location of the sorted point array.
        Options include 'tl', 'tr', 'br', 'bl', 'l', 't', 'r', 'b', 'closest'
        and 'ignore'. The default is 'tl'. They stand for 'tl': top-left,
        'tr': top-right, 'br': bottom-right, 'bl': bottom-left, 'l': leftmost,
        't': topmost, 'r': right-most, 'b': bottom-most, 'closest': the point
        in the given coords closest to origin_point. If 'ignore': the
        origin_point will be used.
    origin_point : tuple/list, optional
        Helps determine the first point of the sorted list. If origin_spec is
        set to 'ignore', origin_point will be used as the first point. It must
        be a member of coords input point. If the origin_spec is set to
        'closest', distances between origin_point and all coords points will
        be first calculated and the coord point closest to the origin_point
        will be used as the starting point. If more than one coord point has
        the  same distance to the origin_point, then the top-left point of
        these sub-set will be used as the starting point.
    method: int, optional
        Specifies the method of sorting. Three methods are offered, 1, 2 & 3.
    """
    # Calculate the centroid of the points
    centroid = np.mean(coords, axis=0)

    # Function to calculate angle and distance from centroid
    def angle_and_distance(point, centroid, clockwise=False):
        angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        if clockwise:
            # To sort clockwise, we can invert the angle
            angle = -angle
        distance = np.linalg.norm(point - centroid)
        return angle, distance

    # Function to sort points with an option for direction
    def sort_points(points, clockwise=False):
        # Sorting indices based on angles (and distances for collinear points)
        sorted_indices = sorted(range(len(points)), key=lambda i: angle_and_distance(points[i], centroid, clockwise))
        # Sorted points using the sorted indices
        sorted_points = points[sorted_indices]
        return sorted_points, sorted_indices

    # Example usage
    clockwise_sorted_points, clockwise_sorted_indices = sort_points(coords, clockwise=True)
    counterclockwise_sorted_points, counterclockwise_sorted_indices = sort_points(coords, clockwise=False)

def calculate_junction_order(matrix, junction_point):
    """
    DOC-STRING HERE

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    y, x = junction_point
    # Extract the 3x3 neighborhood around the junction point
    neighborhood = matrix[max(0, y-1):y+2, max(0, x-1):x+2]
    # Count unique grains in the neighborhood
    unique_grains = np.unique(neighborhood)
    # Exclude background identifier if necessary (assuming background/grain identifiers are > 0)
    order = len(unique_grains[unique_grains > 0])  # Modify condition based on your background identifier
    return order
