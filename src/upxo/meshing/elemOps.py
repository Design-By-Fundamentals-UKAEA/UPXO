import numpy as np

def rebuild_elConnectivity(availableElTypes=None, availableFeatures=None,
                           filtered_mesh_cells=None):
    elConn = {}
    for eltype in availableElTypes:
        elTypeID = np.where(np.array(availableFeatures, dtype=str) == eltype)[0][0]
        elConn[eltype] = filtered_mesh_cells[elTypeID].data
    return elConn

def get_elCentroids_2d(nodes, elConn, availableElTypes):
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d

    elCentroids_coords, elCentroids_shPoints, elCentroids_Tree = {}, {}, {}
    centroid_data = {'coordinates': {},
                'shapelyPoints': {},
                'STRtree': {},
                'UPXOMultiPoint': {}}
    for eltype in availableElTypes:
        elCentroids_coords[eltype] = nodes[elConn[eltype]].mean(axis=1)[:, :2]
        elCentroids_shPoints[eltype] = [Point(x, y) for x, y, in elCentroids_coords[eltype]]
        elCentroids_Tree[eltype] = STRtree(elCentroids_shPoints[eltype])

        centroid_data['coordinates'][eltype] = elCentroids_coords[eltype]
        centroid_data['shapelyPoints'][eltype] = elCentroids_shPoints[eltype]
        centroid_data['STRtree'][eltype] = elCentroids_Tree[eltype]
        centroid_data['UPXOMultiPoint'][eltype] = mulpoint2d(elCentroids_coords[eltype])
    return centroid_data

def compute_elementQuality_AR_2d(nodes, elConn):
    """
    Import
    ------
    from upxo.meshing.elemOps import compute_elementQuality_AR as compute_elq_AR
    """
    coords = nodes[:, :2]
    aspect_ratios = {}
    for eltype, conn in elConn.items():
        if conn is None or len(conn) == 0:
            aspect_ratios[eltype] = np.array([], dtype=float)
            continue
        edge_lengths = []
        for elem in conn:
            pts = coords[elem]
            edges = np.roll(pts, -1, axis=0) - pts
            edge_lengths.append(np.linalg.norm(edges, axis=1))
        edge_lengths = np.array(edge_lengths)
        max_len = edge_lengths.max(axis=1)
        min_len = edge_lengths.min(axis=1)
        aspect_ratios[eltype] = np.divide(max_len, min_len, out=np.full_like(max_len, np.nan, dtype=float),
                                          where=min_len > 0)
    return aspect_ratios

def find_elIDs_by_quality(elQual=None, quality_parameter='ar', 
                          elTypes=['triangle', 'quad'], vmin=1, vmax=2):
    """
    from upxo.meshing.elemOps import find_elIDs_by_quality_threshold as findelIDsQual
    """
    if quality_parameter not in elQual:
        raise ValueError(f"Element quality parameter {quality_parameter} not avaialble.")
    elqual = elQual[quality_parameter]
    # ---------------------------------
    elIDs = {}
    for eltype in elTypes:
        if eltype not in elqual:
            print(f"Element type {eltype} is not available")
            continue
        elIDs[eltype] = np.where(np.logical_and(elqual[eltype] >= vmin, elqual[eltype] <= vmax))[0]
    return elIDs

def build_global_element_numbering(elConn, elID_ranges=None, start_id=1):
    """
    Build global element numbering for all element types and a map to local ids.

    Returns:
        global_ids_by_type: dict[eltype] -> np.ndarray of global element IDs
        global_to_local: dict[global_id] -> (eltype, local_id)
        local_to_global: dict[eltype] -> np.ndarray mapping local_id -> global_id

    Import
    ------
    from upxo.meshing.elemOps import build_global_element_numbering
    """
    global_ids_by_type = {}
    global_to_local = {}
    local_to_global = {}

    if elID_ranges is None:
        offset = start_id
        for eltype, conn in elConn.items():
            nloc = 0 if conn is None else len(conn)
            gids = np.arange(offset, offset + nloc, dtype=int)
            global_ids_by_type[eltype] = gids
            local_to_global[eltype] = gids
            for lid, gid in enumerate(gids):
                global_to_local[int(gid)] = (eltype, int(lid))
            offset += nloc
    else:
        for eltype, (gstart, gend) in elID_ranges.items():
            conn = elConn.get(eltype)
            nloc = 0 if conn is None else len(conn)
            gids = np.arange(gstart, gstart + nloc, dtype=int)
            global_ids_by_type[eltype] = gids
            local_to_global[eltype] = gids
            for lid, gid in enumerate(gids):
                global_to_local[int(gid)] = (eltype, int(lid))

    return global_ids_by_type, global_to_local, local_to_global

def find_el_neigh(element_ids, elConn, n_order=1, eltype=None, include_self=False):
    """
    Return nth-order neighbor element indices for the given element_ids.
    Neighbors are defined by shared nodes in the chosen element connectivity.
    """
    if isinstance(elConn, dict):
        if eltype is None:
            if len(elConn) == 1:
                eltype = next(iter(elConn.keys()))
            else:
                raise ValueError("Specify eltype when elConn is a dict with multiple types.")
        conn = elConn[eltype]
    else:
        conn = elConn

    element_ids = np.asarray(element_ids, dtype=int)
    if element_ids.size == 0:
        return np.array([], dtype=int)

    # Build node -> elements map
    node_to_elems = {}
    for elem_idx, elem_nodes in enumerate(conn):
        for n in elem_nodes:
            node_to_elems.setdefault(int(n), set()).add(elem_idx)

    # Build adjacency list
    elem_neighbors = [set() for _ in range(len(conn))]
    for elems in node_to_elems.values():
        for e in elems:
            elem_neighbors[e].update(elems)
    for e in range(len(conn)):
        elem_neighbors[e].discard(e)

    # BFS expansion to nth order
    visited = set(element_ids.tolist())
    frontier = set(element_ids.tolist())

    for _ in range(max(0, int(n_order))):
        next_frontier = set()
        for e in frontier:
            next_frontier.update(elem_neighbors[e])
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    if not include_self:
        visited -= set(element_ids.tolist())

    return np.array(sorted(visited), dtype=int)

def find_nthOrderNeigh(n_order, el_subset, availableElTypes, elConn, include_self=False):

    neighbor_ids_by_type = {et: [] for et in availableElTypes}
    elem_ids_by_type = {et: [] for et in availableElTypes}

    for gname, etypes in el_subset.items():
        for et in availableElTypes:
            elem_ids = etypes.get(et, {}).get("elem_ids", np.array([], dtype=int))
            if elem_ids.size == 0:
                continue
            elem_ids_by_type[et].extend(elem_ids.tolist())
            neigh_ids = find_el_neigh(elem_ids, elConn, n_order=n_order, eltype=et, include_self=include_self)
            neighbor_ids_by_type[et].extend(neigh_ids.tolist())

    neighbor_ids_by_type = {et: np.unique(np.array(ids, dtype=int)) for et, ids in neighbor_ids_by_type.items()}
    elem_ids_by_type = {et: np.unique(np.array(ids, dtype=int)) for et, ids in elem_ids_by_type.items()}
    combined_ids_by_type = {et: np.unique(np.concatenate((neighbor_ids_by_type[et], elem_ids_by_type[et])))
                    for et in availableElTypes}
    return neighbor_ids_by_type, elem_ids_by_type, combined_ids_by_type

def extract_elements_within_distance_to_gb(grain_name=None, gb_lines=None,
                                           elConn=None, nodes=None,
                                           distance=1.0, eltypes=None,
                                           gblines_by_grain=None,
                                           chunk_size=2000, return_distances=False):
    """
    Extract element IDs whose centroids lie within `distance` of a grain boundary.

    Provide either:
      - grain_name with gblines_by_grain, or
      - gb_lines (array of line node pairs)

    Returns:
      dict[eltype] -> np.ndarray of element IDs (and optionally distances)
    """
    if elConn is None or nodes is None:
        raise ValueError("elConn and nodes are required.")
    if gb_lines is None:
        if grain_name is None or gblines_by_grain is None:
            raise ValueError("Provide gb_lines or grain_name with gblines_by_grain.")
        gb_lines = gblines_by_grain.get(grain_name)
    if gb_lines is None or len(gb_lines) == 0:
        return {} if not return_distances else ({}, {})

    if eltypes is None:
        eltypes = list(elConn.keys())

    a = nodes[gb_lines[:, 0], :2]
    b = nodes[gb_lines[:, 1], :2]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0, denom, np.finfo(float).eps)

    def min_dist_to_segments(points):
        min_dist = np.full(points.shape[0], np.inf, dtype=float)
        for start in range(0, points.shape[0], chunk_size):
            p = points[start:start + chunk_size]
            ap = p[:, None, :] - a[None, :, :]
            t = np.einsum("kij,ij->ki", ap, ab) / denom[None, :]
            t = np.clip(t, 0.0, 1.0)
            proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            d = np.linalg.norm(p[:, None, :] - proj, axis=2)
            min_dist[start:start + chunk_size] = d.min(axis=1)
        return min_dist

    selected_by_type = {}
    distances_by_type = {}

    for et in eltypes:
        conn = elConn.get(et)
        if conn is None or len(conn) == 0:
            selected_by_type[et] = np.array([], dtype=int)
            distances_by_type[et] = np.array([], dtype=float)
            continue
        centroids = nodes[conn].mean(axis=1)[:, :2]
        min_dist = min_dist_to_segments(centroids)
        elem_ids = np.where(min_dist <= distance)[0].astype(int)
        selected_by_type[et] = elem_ids
        distances_by_type[et] = min_dist

    if return_distances:
        return selected_by_type, distances_by_type
    return selected_by_type


def resolve_eltypes(grain_name, grainElements, grainCoordinates, eltypes=None):
    if eltypes is not None:
        return list(eltypes)

    available = []
    for etype, el_ids in grainElements[grain_name].items():
        coords = grainCoordinates[grain_name].get(etype)
        if el_ids is None or coords is None:
            continue
        if hasattr(el_ids, 'size') and el_ids.size > 0 and coords.size > 0:
            available.append(etype)
    return available


def _compute_nearest_distances_to_gb(grain_name, eltypes, gbnodeCoords, grainCoordinates):
    from scipy.spatial import cKDTree

    gbcoords = gbnodeCoords[grain_name][:, :2]
    gb_tree = cKDTree(gbcoords)

    nearest_dist_by_type = {}
    for etype in eltypes:
        centroids = grainCoordinates[grain_name].get(etype)
        if centroids is None or centroids.size == 0:
            nearest_dist_by_type[etype] = np.array([], dtype=float)
            continue

        workers = 2 if centroids.shape[0] >= 1000 else 1
        nearest_dist, _ = gb_tree.query(centroids, k=1, p=2, workers=workers)
        nearest_dist_by_type[etype] = nearest_dist

    return nearest_dist_by_type


def build_element_ids_by_band(grain_name, bands, eltypes, grainElements, nearest_dist_by_type):
    element_ids_by_band = {}
    for band in bands:
        band_min = min(band)
        band_max = max(band)
        element_ids_by_band[(band_min, band_max)] = {}

        for etype in eltypes:
            el_ids_all = grainElements[grain_name].get(etype)
            nearest_dist = nearest_dist_by_type.get(etype)
            if el_ids_all is None or nearest_dist is None:
                element_ids_by_band[(band_min, band_max)][etype] = np.array([], dtype=int)
                continue
            if el_ids_all.size == 0 or nearest_dist.size == 0:
                element_ids_by_band[(band_min, band_max)][etype] = np.array([], dtype=int)
                continue

            band_mask = np.logical_and(nearest_dist >= band_min, nearest_dist <= band_max)
            element_ids_by_band[(band_min, band_max)][etype] = el_ids_all[band_mask]

    return element_ids_by_band


def select_elements_in_bands(grain_name, bands, gbnodeCoords, grainCoordinates,
                             grainElements, eltypes=None):
    selected_eltypes = resolve_eltypes(
        grain_name, grainElements, grainCoordinates, eltypes=eltypes
    )
    nearest_dist_by_type = _compute_nearest_distances_to_gb(
        grain_name, selected_eltypes, gbnodeCoords, grainCoordinates
    )
    element_ids_by_band = build_element_ids_by_band(
        grain_name, bands, selected_eltypes, grainElements, nearest_dist_by_type
    )
    return element_ids_by_band, nearest_dist_by_type, selected_eltypes