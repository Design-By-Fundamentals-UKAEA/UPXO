"""
High-fidelity 2D Voronoi computational geometry engine for UPXO.

This module provides mathematical and algorithmic foundations for:
- Standard bounded 2D Voronoi tessellation with exact clipping
- Periodic Boundary Conditions (PBC) in 2D (X, Y, or both)
- Laguerre tessellation / Power diagrams (weighted Voronoi via 3D lower convex hull)
- Centroidal Voronoi Tessellation (CVT) via Lloyd's algorithm
- Grain boundary interface perturbation and curvature modeling

Dependencies
------------
numpy, scipy, shapely
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Dict, List, Optional, Union

import numpy as np
from scipy.spatial import ConvexHull, Voronoi, cKDTree
from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
from shapely.ops import unary_union


def coerce_bounds_2d(
    bounds: Optional[Union[Sequence, Dict]] = None,
    seeds: Optional[np.ndarray] = None,
    padding_fraction: float = 0.05,
) -> List[List[float]]:
    """
    Coerce bounds specification into standard ``[[xmin, xmax], [ymin, ymax]]``.

    Parameters
    ----------
    bounds : sequence or dict, optional
        Bounds as ``[[xmin, xmax], [ymin, ymax]]``, ``(xmin, xmax, ymin, ymax)``,
        or ``{'xbound': [xmin, xmax], 'ybound': [ymin, ymax]}``.
    seeds : np.ndarray, optional
        `(N, 2)` coordinate array to infer bounds from if `bounds` is None.
    padding_fraction : float, optional
        Padding factor applied when inferring bounds from seeds. Default 0.05.

    Returns
    -------
    list of list of float
        ``[[xmin, xmax], [ymin, ymax]]``
    """
    if bounds is not None:
        if isinstance(bounds, dict):
            xb = list(bounds.get('xbound', bounds.get('x', [0.0, 100.0])))
            yb = list(bounds.get('ybound', bounds.get('y', [0.0, 100.0])))
            return [[float(xb[0]), float(xb[1])], [float(yb[0]), float(yb[1])]]
        b_arr = np.asarray(bounds, dtype=float)
        if b_arr.shape == (2, 2):
            return b_arr.tolist()
        elif b_arr.ndim == 1 and len(b_arr) == 4:
            return [[float(b_arr[0]), float(b_arr[1])], [float(b_arr[2]), float(b_arr[3])]]
        else:
            raise ValueError(f"Cannot coerce bounds of shape {b_arr.shape} to 2D bounds.")

    if seeds is not None:
        seeds_arr = np.asarray(seeds, dtype=float)
        if seeds_arr.ndim != 2 or seeds_arr.shape[1] != 2:
            raise ValueError("seeds must have shape (N, 2)")
        mins = seeds_arr.min(axis=0)
        maxs = seeds_arr.max(axis=0)
        span = maxs - mins
        pad = np.where(span > 0.0, padding_fraction * span, 1.0)
        return [[float(mins[0] - pad[0]), float(maxs[0] + pad[0])],
                [float(mins[1] - pad[1]), float(maxs[1] + pad[1])]]

    return [[0.0, 100.0], [0.0, 100.0]]


def _generate_ghost_seeds_2d(
    seeds: np.ndarray,
    bounds: List[List[float]],
    periodic: Tuple[bool, bool] = (False, False),
    weights: Optional[np.ndarray] = None,
    pad_ratio: float = 0.25,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Generate periodic replica and reflective ghost seeds to ensure finite cells.

    Returns
    -------
    all_seeds : np.ndarray, shape (M, 2)
    all_weights : np.ndarray or None, shape (M,)
    n_orig : int
        Number of original seeds. Original seeds occupy indices 0 .. n_orig - 1.
    """
    seeds_arr = np.ascontiguousarray(seeds, dtype=float)
    n_orig = len(seeds_arr)
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    Lx = xmax - xmin
    Ly = ymax - ymin

    p_x, p_y = periodic
    offsets_x = [-1, 0, 1] if p_x else [0]
    offsets_y = [-1, 0, 1] if p_y else [0]

    replicated_seeds = []
    replicated_weights = [] if weights is not None else None

    # 1. Periodic replication
    for dx in offsets_x:
        for dy in offsets_y:
            shift = np.array([dx * Lx, dy * Ly], dtype=float)
            replicated_seeds.append(seeds_arr + shift)
            if weights is not None:
                replicated_weights.append(np.asarray(weights, dtype=float))

    all_seeds = np.vstack(replicated_seeds)
    all_w = np.concatenate(replicated_weights) if weights is not None else None

    # 2. For non-periodic axes, reflect boundary points outward to trap infinite rays
    ghost_reflections = []
    ghost_w = [] if weights is not None else None

    curr_xmin = xmin - (Lx if p_x else 0.0)
    curr_xmax = xmax + (Lx if p_x else 0.0)
    curr_ymin = ymin - (Ly if p_y else 0.0)
    curr_ymax = ymax + (Ly if p_y else 0.0)

    pad_x = max(Lx * pad_ratio, 1e-3)
    pad_y = max(Ly * pad_ratio, 1e-3)

    if not p_x:
        # Reflect near left boundary
        mask_l = all_seeds[:, 0] < (xmin + pad_x)
        if np.any(mask_l):
            refl = all_seeds[mask_l].copy()
            refl[:, 0] = 2.0 * xmin - refl[:, 0]
            ghost_reflections.append(refl)
            if weights is not None:
                ghost_w.append(all_w[mask_l])

        # Reflect near right boundary
        mask_r = all_seeds[:, 0] > (xmax - pad_x)
        if np.any(mask_r):
            refl = all_seeds[mask_r].copy()
            refl[:, 0] = 2.0 * xmax - refl[:, 0]
            ghost_reflections.append(refl)
            if weights is not None:
                ghost_w.append(all_w[mask_r])

    if not p_y:
        # Reflect near bottom boundary
        mask_b = all_seeds[:, 1] < (ymin + pad_y)
        if np.any(mask_b):
            refl = all_seeds[mask_b].copy()
            refl[:, 1] = 2.0 * ymin - refl[:, 1]
            ghost_reflections.append(refl)
            if weights is not None:
                ghost_w.append(all_w[mask_b])

        # Reflect near top boundary
        mask_t = all_seeds[:, 1] > (ymax - pad_y)
        if np.any(mask_t):
            refl = all_seeds[mask_t].copy()
            refl[:, 1] = 2.0 * ymax - refl[:, 1]
            ghost_reflections.append(refl)
            if weights is not None:
                ghost_w.append(all_w[mask_t])

    # Also add 4 far corner anchors to guarantee boundedness in extreme cases
    diag = math.hypot(Lx, Ly) * 3.0
    corners = np.array([
        [xmin - diag, ymin - diag],
        [xmax + diag, ymin - diag],
        [xmax + diag, ymax + diag],
        [xmin - diag, ymax + diag]
    ], dtype=float)
    ghost_reflections.append(corners)
    if weights is not None:
        mean_w = float(np.mean(weights))
        ghost_w.append(np.full(len(corners), mean_w, dtype=float))

    if ghost_reflections:
        all_seeds = np.vstack([all_seeds] + ghost_reflections)
        if weights is not None:
            all_w = np.concatenate([all_w] + ghost_w)

    return all_seeds, all_w, n_orig


def compute_standard_voronoi_2d(
    seeds: np.ndarray,
    bounds: Optional[Union[Sequence, Dict]] = None,
    periodic: Tuple[bool, bool] = (False, False),
    clip: bool = True,
) -> Dict[int, Polygon]:
    """
    Compute bounded 2D Voronoi polygons mapped exactly to seed indices.

    Parameters
    ----------
    seeds : np.ndarray, shape (N, 2)
        Seed coordinates.
    bounds : sequence or dict, optional
        Domain bounds ``[[xmin, xmax], [ymin, ymax]]``. If None, inferred from seeds.
    periodic : tuple of bool, optional
        ``(periodic_x, periodic_y)``. Default is (False, False).
    clip : bool, optional
        Whether to clip cells to the bounding box. Default is True.

    Returns
    -------
    dict of {int: Polygon or MultiPolygon}
        Mapping from seed index (0 .. N-1) to Shapely polygon.
    """
    seeds_arr = np.ascontiguousarray(seeds, dtype=float)
    b2d = coerce_bounds_2d(bounds, seeds=seeds_arr)
    n_orig = len(seeds_arr)
    if n_orig == 0:
        return {}
    if n_orig == 1:
        xmin, xmax = b2d[0]
        ymin, ymax = b2d[1]
        return {0: box(xmin, ymin, xmax, ymax)}

    # Prepare seeds with ghost padding / periodic tiling
    all_seeds, _, _ = _generate_ghost_seeds_2d(seeds_arr, b2d, periodic=periodic)
    vor = Voronoi(all_seeds)

    xmin, xmax = b2d[0]
    ymin, ymax = b2d[1]
    domain_box = box(xmin, ymin, xmax, ymax)

    cells: Dict[int, Polygon] = {}
    p_x, p_y = periodic
    offsets_x = [-1, 0, 1] if p_x else [0]
    offsets_y = [-1, 0, 1] if p_y else [0]
    n_tiles = len(offsets_x) * len(offsets_y)

    for i in range(n_orig):
        cell_parts = []
        # Check all periodic replicas of seed i that might intersect domain_box
        for tile_idx in range(n_tiles):
            seed_idx = tile_idx * n_orig + i
            region_idx = vor.point_region[seed_idx]
            region = vor.regions[region_idx]

            if not region or -1 in region:
                continue

            verts = vor.vertices[region]
            if len(verts) < 3:
                continue

            poly = Polygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)

            if clip:
                intersected = poly.intersection(domain_box)
                if not intersected.is_empty and intersected.area > 1e-12:
                    cell_parts.append(intersected)
            else:
                cell_parts.append(poly)

        if len(cell_parts) == 1:
            cells[i] = cell_parts[0]
        elif len(cell_parts) > 1:
            merged = unary_union(cell_parts)
            if not merged.is_valid:
                merged = merged.buffer(0)
            cells[i] = merged
        else:
            cells[i] = Polygon()

    return cells


def compute_power_diagram_2d(
    seeds: np.ndarray,
    weights: Optional[Union[Sequence[float], np.ndarray]] = None,
    bounds: Optional[Union[Sequence, Dict]] = None,
    periodic: Tuple[bool, bool] = (False, False),
    clip: bool = True,
) -> Dict[int, Polygon]:
    """
    Compute 2D Laguerre tessellation / Power diagram using the lifted 3D lower convex hull.

    For each seed point :math:`\\mathbf{s}_i = (x_i, y_i)` with weight :math:`w_i`,
    the power cell is the set of points :math:`\\mathbf{x}` satisfying:
    :math:`\\|\\mathbf{x} - \\mathbf{s}_i\\|^2 - w_i \\le \\|\\mathbf{x} - \\mathbf{s}_j\\|^2 - w_j`.

    Parameters
    ----------
    seeds : np.ndarray, shape (N, 2)
        Seed coordinates.
    weights : sequence or np.ndarray, shape (N,), optional
        Seed weights (e.g. squared target radii). If None, defaults to uniform weights.
    bounds : sequence or dict, optional
        Domain bounds ``[[xmin, xmax], [ymin, ymax]]``. If None, inferred from seeds.
    periodic : tuple of bool, optional
        ``(periodic_x, periodic_y)``. Default is (False, False).
    clip : bool, optional
        Whether to clip cells to the bounding box. Default is True.

    Returns
    -------
    dict of {int: Polygon or MultiPolygon}
        Mapping from seed index (0 .. N-1) to Shapely polygon.
    """
    seeds_arr = np.ascontiguousarray(seeds, dtype=float)
    if weights is None:
        weights_arr = np.ones(len(seeds_arr), dtype=float)
    else:
        weights_arr = np.ascontiguousarray(weights, dtype=float)

    b2d = coerce_bounds_2d(bounds, seeds=seeds_arr)
    n_orig = len(seeds_arr)

    if n_orig == 0:
        return {}
    if n_orig == 1:
        xmin, xmax = b2d[0]
        ymin, ymax = b2d[1]
        return {0: box(xmin, ymin, xmax, ymax)}

    # Generate ghost and replica seeds with corresponding weights
    all_seeds, all_weights, _ = _generate_ghost_seeds_2d(
        seeds_arr, b2d, periodic=periodic, weights=weights_arr, pad_ratio=0.35
    )

    # Lift points to 3D: (x, y, x^2 + y^2 - w)
    z_coords = np.sum(all_seeds ** 2, axis=1) - all_weights
    lifted = np.column_stack([all_seeds, z_coords])

    # Compute 3D Convex Hull
    try:
        hull = ConvexHull(lifted)
    except Exception:
        # Fallback to standard Voronoi if collinear or singular
        return compute_standard_voronoi_2d(seeds_arr, b2d, periodic=periodic, clip=clip)

    # Filter lower hull facets: normal nz < 0
    # Equation of facet: A*x + B*y + C*z + D = 0
    equations = hull.equations
    lower_facet_indices = [
        idx for idx, eq in enumerate(equations)
        if eq[2] < -1e-7  # C is negative (pointing downwards)
    ]

    if not lower_facet_indices:
        return compute_standard_voronoi_2d(seeds_arr, b2d, periodic=periodic, clip=clip)

    # Compute 2D dual vertices for each lower-hull facet:
    # Plane: A*x + B*y + C*z + D = 0 => z = -A/C * x - B/C * y - D/C
    # In power diagram duality: 2*x_v = -A/C, 2*y_v = -B/C
    facet_dual_vertex = {}
    for f_idx in lower_facet_indices:
        eq = equations[f_idx]
        A, B, C = eq[0], eq[1], eq[2]
        vx = -A / (2.0 * C)
        vy = -B / (2.0 * C)
        facet_dual_vertex[f_idx] = (vx, vy)

    # Map each seed point to its incident lower hull facets
    # hull.simplices contains indices of lifted points for each facet
    point_to_facets: Dict[int, List[int]] = {i: [] for i in range(len(all_seeds))}
    for f_idx in lower_facet_indices:
        simplex = hull.simplices[f_idx]
        for pt_idx in simplex:
            point_to_facets[pt_idx].append(f_idx)

    xmin, xmax = b2d[0]
    ymin, ymax = b2d[1]
    domain_box = box(xmin, ymin, xmax, ymax)

    cells: Dict[int, Polygon] = {}
    p_x, p_y = periodic
    offsets_x = [-1, 0, 1] if p_x else [0]
    offsets_y = [-1, 0, 1] if p_y else [0]
    n_tiles = len(offsets_x) * len(offsets_y)

    for i in range(n_orig):
        cell_parts = []
        for tile_idx in range(n_tiles):
            seed_idx = tile_idx * n_orig + i
            f_list = point_to_facets.get(seed_idx, [])
            if len(f_list) < 3:
                continue

            dual_pts = [facet_dual_vertex[f] for f in f_list if f in facet_dual_vertex]
            if len(dual_pts) < 3:
                continue

            # Compute convex hull of dual vertices in 2D
            pts_arr = np.array(dual_pts, dtype=float)
            try:
                ch2d = ConvexHull(pts_arr)
                poly = Polygon(pts_arr[ch2d.vertices])
            except Exception:
                continue

            if not poly.is_valid:
                poly = poly.buffer(0)

            if clip:
                intersected = poly.intersection(domain_box)
                if not intersected.is_empty and intersected.area > 1e-12:
                    cell_parts.append(intersected)
            else:
                cell_parts.append(poly)

        if len(cell_parts) == 1:
            cells[i] = cell_parts[0]
        elif len(cell_parts) > 1:
            merged = unary_union(cell_parts)
            if not merged.is_valid:
                merged = merged.buffer(0)
            cells[i] = merged
        else:
            cells[i] = Polygon()

    return cells


def cvt_relax_2d(
    seeds: np.ndarray,
    bounds: Optional[Union[Sequence, Dict]] = None,
    iterations: int = 10,
    periodic: Tuple[bool, bool] = (False, False),
    weights: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """
    Perform Centroidal Voronoi Tessellation (CVT) relaxation via Lloyd's algorithm.

    Iteratively updates each seed to the centroid of its Voronoi cell, producing
    regular, energy-minimized, equiaxed polycrystal seed configurations.

    Parameters
    ----------
    seeds : np.ndarray, shape (N, 2)
        Initial seed coordinates.
    bounds : sequence or dict, optional
        Domain bounds ``[[xmin, xmax], [ymin, ymax]]``. If None, inferred from seeds.
    iterations : int, optional
        Number of Lloyd relaxation steps. Default is 10.
    periodic : tuple of bool, optional
        ``(periodic_x, periodic_y)``. Default is (False, False).
    weights : np.ndarray, optional
        Optional seed weights for weighted CVT. Default is None.
    max_iter : int, optional
        Alias for `iterations`.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Relaxed seed coordinates.
    """
    if max_iter is not None:
        iterations = max_iter

    seeds_arr = np.ascontiguousarray(seeds, dtype=float)
    b2d = coerce_bounds_2d(bounds, seeds=seeds_arr)
    current_seeds = np.copy(seeds_arr)

    xmin, xmax = b2d[0]
    ymin, ymax = b2d[1]
    Lx = xmax - xmin
    Ly = ymax - ymin
    p_x, p_y = periodic

    for _ in range(iterations):
        if weights is not None:
            cells = compute_power_diagram_2d(
                current_seeds, weights, b2d, periodic=periodic, clip=True
            )
        else:
            cells = compute_standard_voronoi_2d(
                current_seeds, b2d, periodic=periodic, clip=True
            )

        new_seeds = np.copy(current_seeds)
        for i, poly in cells.items():
            if poly.is_empty or poly.area <= 1e-12:
                continue
            c = poly.centroid
            cx, cy = c.x, c.y

            # Periodic wrap or box clamp
            if p_x:
                cx = xmin + ((cx - xmin) % Lx)
            else:
                cx = max(xmin + 1e-6, min(xmax - 1e-6, cx))

            if p_y:
                cy = ymin + ((cy - ymin) % Ly)
            else:
                cy = max(ymin + 1e-6, min(ymax - 1e-6, cy))

            new_seeds[i] = [cx, cy]

        current_seeds = new_seeds

    return current_seeds


def perturb_interfaces_2d(
    cells: Union[Dict[int, Polygon], Sequence[Polygon], MultiPolygon],
    bounds: Optional[Union[Sequence, Dict]] = None,
    perturb_factor: float = 0.05,
    n_subdivisions: int = 2,
    random_seed: Optional[int] = 42,
    factor: Optional[float] = None,
    seed: Optional[int] = None,
) -> Union[Dict[int, Polygon], List[Polygon], MultiPolygon]:
    """
    Apply natural non-linear curvature perturbations to shared grain boundaries.

    Internal grain boundaries are subdivided and displaced while domain boundary
    edges and triple/multiple junctions remain topologically pinned, ensuring
    watertight manifolds.

    Parameters
    ----------
    cells : dict, list, or MultiPolygon of Polygons
        Input Voronoi cells.
    bounds : sequence or dict, optional
        Domain bounds ``[[xmin, xmax], [ymin, ymax]]``.
    perturb_factor : float, optional
        Magnitude of perpendicular perturbation relative to edge length. Default 0.05.
    n_subdivisions : int, optional
        Number of recursive midpoint subdivisions per edge. Default 2.
    random_seed : int, optional
        RNG seed for reproducible perturbations. Default 42.
    factor : float, optional
        Alias for `perturb_factor`.
    seed : int, optional
        Alias for `random_seed`.

    Returns
    -------
    dict, list, or MultiPolygon
        Cells with curved/perturbed interfaces, preserving input collection type.
    """
    if factor is not None:
        perturb_factor = factor
    if seed is not None:
        random_seed = seed

    is_dict = isinstance(cells, dict)
    is_mpoly = isinstance(cells, MultiPolygon)
    is_list = isinstance(cells, (list, tuple))

    if is_dict:
        cells_dict = cells
    elif is_mpoly:
        cells_dict = {i: p for i, p in enumerate(cells.geoms)}
    elif is_list:
        cells_dict = {i: p for i, p in enumerate(cells)}
    else:
        cells_dict = {0: cells}

    if perturb_factor <= 0.0:
        return cells

    if bounds is not None:
        b2d = coerce_bounds_2d(bounds)
    else:
        all_polys = [p for p in cells_dict.values() if isinstance(p, (Polygon, MultiPolygon)) and not p.is_empty]
        if all_polys:
            u = unary_union(all_polys)
            minx, miny, maxx, maxy = u.bounds
            b2d = [[float(minx), float(maxx)], [float(miny), float(maxy)]]
        else:
            b2d = [[0.0, 100.0], [0.0, 100.0]]

    rng = np.random.default_rng(random_seed)
    xmin, xmax = b2d[0]
    ymin, ymax = b2d[1]
    tol = 1e-5

    # 1. Extract all unique segments
    # Map canonical edge (p1, p2) with p1 < p2 to its perturbed polyline
    edge_map: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List[Tuple[float, float]]] = {}

    def is_on_boundary(pt: Tuple[float, float]) -> bool:
        x, y = pt
        return (abs(x - xmin) < tol or abs(x - xmax) < tol or
                abs(y - ymin) < tol or abs(y - ymax) < tol)

    for gid, geom in cells_dict.items():
        polys = list(geom.geoms) if isinstance(geom, (MultiPolygon,)) else [geom]
        for poly in polys:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            coords = list(poly.exterior.coords)
            for k in range(len(coords) - 1):
                p1 = (round(coords[k][0], 6), round(coords[k][1], 6))
                p2 = (round(coords[k+1][0], 6), round(coords[k+1][1], 6))
                if p1 == p2:
                    continue
                canonical = (p1, p2) if p1 < p2 else (p2, p1)
                if canonical not in edge_map:
                    edge_map[canonical] = []

    # 2. Perturb each internal edge
    for (p1, p2) in edge_map.keys():
        # If both points lie on the outer boundary, leave perfectly flat
        if is_on_boundary(p1) and is_on_boundary(p2):
            edge_map[(p1, p2)] = [p1, p2]
            continue

        # Subdivide and perturb
        pts = [p1, p2]
        edge_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-6:
            edge_map[(p1, p2)] = [p1, p2]
            continue

        # Normal vector
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=float) / edge_len

        # Recursive midpoint displacement
        current_pts = [np.array(p1, dtype=float), np.array(p2, dtype=float)]
        for sub in range(n_subdivisions):
            next_pts = [current_pts[0]]
            scale = perturb_factor * edge_len / (2.0 ** sub)
            for j in range(len(current_pts) - 1):
                mid = (current_pts[j] + current_pts[j+1]) / 2.0
                disp = rng.normal(0.0, scale) * normal
                perturbed_mid = mid + disp
                # Clamp to bounds
                perturbed_mid[0] = max(xmin, min(xmax, perturbed_mid[0]))
                perturbed_mid[1] = max(ymin, min(ymax, perturbed_mid[1]))
                next_pts.append(perturbed_mid)
                next_pts.append(current_pts[j+1])
            current_pts = next_pts

        edge_map[(p1, p2)] = [(float(p[0]), float(p[1])) for p in current_pts]

    # 3. Reconstruct perturbed polygons
    perturbed_cells: Dict[int, Polygon] = {}
    domain_box = box(xmin, ymin, xmax, ymax)

    for gid, geom in cells_dict.items():
        polys = list(geom.geoms) if isinstance(geom, (MultiPolygon,)) else [geom]
        reconstructed_polys = []

        for poly in polys:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            coords = list(poly.exterior.coords)
            new_ring = []

            for k in range(len(coords) - 1):
                p1 = (round(coords[k][0], 6), round(coords[k][1], 6))
                p2 = (round(coords[k+1][0], 6), round(coords[k+1][1], 6))
                if p1 == p2:
                    continue
                canonical = (p1, p2) if p1 < p2 else (p2, p1)
                sub_pts = edge_map.get(canonical, [p1, p2])

                # Maintain orientation
                if canonical == (p1, p2):
                    pts_to_add = sub_pts[:-1]
                else:
                    pts_to_add = list(reversed(sub_pts))[:-1]

                new_ring.extend(pts_to_add)

            if len(new_ring) >= 3:
                new_ring.append(new_ring[0])
                p_new = Polygon(new_ring)
                if not p_new.is_valid:
                    p_new = p_new.buffer(0)
                p_new = p_new.intersection(domain_box)
                if not p_new.is_empty and p_new.area > 1e-12:
                    reconstructed_polys.append(p_new)

        if len(reconstructed_polys) == 1:
            perturbed_cells[gid] = reconstructed_polys[0]
        elif len(reconstructed_polys) > 1:
            perturbed_cells[gid] = unary_union(reconstructed_polys)
        else:
            perturbed_cells[gid] = cells_dict[gid]

    if is_mpoly:
        return MultiPolygon(list(perturbed_cells.values()))
    elif is_list:
        return list(perturbed_cells.values())
    else:
        return perturbed_cells


def generate_voronoi_2d(
    seeds: np.ndarray,
    bounds: Optional[Union[Sequence, Dict]] = None,
    periodic: Tuple[bool, bool] = (False, False),
    weights: Optional[np.ndarray] = None,
    cvt_iterations: int = 0,
    perturb_factor: float = 0.0,
    clip: bool = True,
    random_seed: Optional[int] = 42,
) -> Dict[str, Union[Dict[int, Polygon], MultiPolygon, np.ndarray, List[List[float]], Tuple[bool, bool]]]:
    """
    Main orchestration function for high-fidelity 2D Voronoi geometry generation.

    Parameters
    ----------
    seeds : np.ndarray, shape (N, 2)
        Seed point coordinates.
    bounds : sequence or dict, optional
        Domain bounds ``[[xmin, xmax], [ymin, ymax]]``. If None, inferred from seeds.
    periodic : tuple of bool, optional
        ``(periodic_x, periodic_y)`` flags for Periodic Boundary Conditions.
    weights : np.ndarray, shape (N,), optional
        Seed weights for Laguerre / Power diagram tessellation.
    cvt_iterations : int, optional
        Number of Lloyd relaxation iterations for Centroidal Voronoi.
    perturb_factor : float, optional
        Magnitude of non-linear grain boundary interface perturbation.
    clip : bool, optional
        Whether to clip cells to the bounding box. Default is True.
    random_seed : int, optional
        RNG seed for perturbation reproducibility. Default is 42.

    Returns
    -------
    dict
        'cells' : dict of {int: Polygon}
        'pxtal' : Shapely MultiPolygon containing all cells
        'seeds' : np.ndarray of final seed positions
        'bounds' : standard bounds ``[[xmin, xmax], [ymin, ymax]]``
        'periodic' : (bool, bool)
    """
    seeds_arr = np.ascontiguousarray(seeds, dtype=float)
    standard_bounds = coerce_bounds_2d(bounds, seeds=seeds_arr)

    # 1. CVT Relaxation if requested
    if cvt_iterations > 0:
        seeds_arr = cvt_relax_2d(
            seeds_arr, standard_bounds, iterations=cvt_iterations,
            periodic=periodic, weights=weights
        )

    # 2. Tessellation: Power diagram if weights provided, else standard Voronoi
    if weights is not None:
        cells = compute_power_diagram_2d(
            seeds_arr, weights, standard_bounds, periodic=periodic, clip=clip
        )
    else:
        cells = compute_standard_voronoi_2d(
            seeds_arr, standard_bounds, periodic=periodic, clip=clip
        )

    # 3. Interface perturbation if requested
    if perturb_factor > 0.0:
        cells = perturb_interfaces_2d(
            cells, standard_bounds, perturb_factor=perturb_factor,
            random_seed=random_seed
        )

    # 4. Form MultiPolygon collection
    valid_polys = [p for p in cells.values() if isinstance(p, (Polygon, MultiPolygon)) and not p.is_empty]
    pxtal_mp = MultiPolygon([p for p in valid_polys if isinstance(p, Polygon)] +
                            [sub for p in valid_polys if isinstance(p, MultiPolygon) for sub in p.geoms])

    return {
        'cells': cells,
        'pxtal': pxtal_mp,
        'seeds': seeds_arr,
        'bounds': standard_bounds,
        'periodic': periodic,
    }
