"""
gssmooth2d.py — 2D grain-structure post-pixellation smoothing pipeline.

Public API
----------
smooth_gs_slice   : Full single-slice pipeline; returns a result dict.

Private helpers (module-internal)
----------------------------------
_merge_small_grains        : Merge isolated grains into their largest neighbour.
_compute_polygon_neighbors : Build adjacency dict via Shapely .touches().
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
import cc3d
from shapely.geometry import Point, Polygon, MultiPolygon

from upxo.pxtal.geometrification import GrainManifold2D
from upxo.gsdataops.gid_ops import find_neighs2d
import upxo.gsdataops.grid_ops as gridOps


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fix_diagonal_pinches(lfi: np.ndarray) -> np.ndarray:
    """
    For each grain whose pixels are only diagonally connected (no 4-connected path),
    fill one bridge pixel at each pinch to establish full 4-connectivity.
    Applied before upscaling so bridges become F×F blocks after np.repeat(×F).
    """
    lfi_out = lfi.copy()
    M, N = lfi_out.shape
    diag_steps = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    changed = True
    while changed:
        changed = False
        unique_grains = np.unique(lfi_out)
        unique_grains = unique_grains[unique_grains > 0]
        for g in unique_grains:
            mask = (lfi_out == g).astype(np.uint8)
            cc = cc3d.connected_components(mask, connectivity=4)
            if int(cc.max()) <= 1:
                continue
            rows, cols = np.nonzero(mask)
            for r, c in zip(rows, cols):
                comp = int(cc[r, c])
                bridged = False
                for dr, dc in diag_steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < M and 0 <= nc < N and mask[nr, nc] and int(cc[nr, nc]) != comp:
                        for br, bc in [(r, nc), (nr, c)]:
                            if 0 <= br < M and 0 <= bc < N and lfi_out[br, bc] != g:
                                lfi_out[br, bc] = g
                                changed = True
                                bridged = True
                                break
                    if bridged:
                        break
                if bridged:
                    break
    return lfi_out


def _fill_staircase_concavities(lfi: np.ndarray, thin_px_threshold: float) -> np.ndarray:
    """
    For each thin grain (bounding-box min-dim < thin_px_threshold), fill BOTH orthogonal
    bridge pixels at every diagonal step to convert a 1-px staircase into a 2-px ribbon.
    Contrast with _fix_diagonal_pinches which fills ONE bridge to restore connectivity;
    here BOTH are filled to widen an already-connected staircase-shaped grain.
    """
    lfi_out = lfi.copy()
    M, N = lfi_out.shape
    diag_steps = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for g in np.unique(lfi_out):
        if g <= 0:
            continue
        rows, cols = np.nonzero(lfi_out == g)
        if len(rows) == 0:
            continue
        min_dim = min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
        if min_dim >= thin_px_threshold:
            continue
        changed_g = True
        while changed_g:
            changed_g = False
            mask = lfi_out == g
            g_rows, g_cols = np.nonzero(mask)
            for r, c in zip(g_rows, g_cols):
                for dr, dc in diag_steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < M and 0 <= nc < N and mask[nr, nc]:
                        for br, bc in [(r, nc), (nr, c)]:
                            if 0 <= br < M and 0 <= bc < N and lfi_out[br, bc] != g:
                                lfi_out[br, bc] = g
                                changed_g = True
    return lfi_out


def _merge_small_grains(lfi: np.ndarray, area_threshold: int = 1) -> np.ndarray:
    """
    Merge grains whose pixel count is <= area_threshold into their largest neighbour.
    Recomputes connected components and neighbour graph after each merger.
    """
    def _dominant(gid, neigh_gid, areas):
        if gid in neigh_gid and len(neigh_gid[gid]) > 0:
            nbrs = neigh_gid[gid]
            return nbrs[np.argmax(areas[nbrs - 1])]
        return gid

    lfi_merged = deepcopy(lfi)
    neigh_gid = find_neighs2d(lfi_merged, conn=4)
    areas = np.bincount(lfi_merged.ravel())[1:]

    small_grains = np.where(areas <= area_threshold)[0] + 1
    for gid in small_grains:
        dom = _dominant(gid, neigh_gid, areas)
        lfi_merged[lfi_merged == gid] = dom
        lfi_merged = cc3d.connected_components(lfi_merged, connectivity=4)
        neigh_gid = find_neighs2d(lfi_merged, conn=4)
        areas = np.bincount(lfi_merged.ravel())[1:]

    return lfi_merged


def _merge_enclosed_polygons(cells: dict) -> dict:
    """
    If polygon A lies entirely within polygon B, union B with A and remove A.
    Operates only on Shapely geometry — the LFI and all original statistics are unchanged.
    """
    cells = dict(cells)
    changed = True
    while changed:
        changed = False
        gids = list(cells.keys())
        for gid_inner in gids:
            if gid_inner not in cells:
                continue
            poly_inner = cells[gid_inner]
            for gid_outer in gids:
                if gid_outer == gid_inner or gid_outer not in cells:
                    continue
                if poly_inner.within(cells[gid_outer]):
                    cells[gid_outer] = cells[gid_outer].union(poly_inner)
                    del cells[gid_inner]
                    changed = True
                    break
            if changed:
                break
    return cells


def _compute_polygon_neighbors(cells: dict) -> dict:
    """
    Build {gid: [gids]} adjacency using Shapely .touches().
    Tests each (i, j) pair once; result is symmetric.
    """
    gids = sorted(int(g) for g in cells.keys())
    neighbors: dict = {g: [] for g in gids}
    for i, gid1 in enumerate(gids):
        geom1 = cells[gid1]
        for gid2 in gids[i + 1:]:
            if geom1.touches(cells[gid2]):
                neighbors[gid1].append(gid2)
                neighbors[gid2].append(gid1)
    return neighbors


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def smooth_gs_slice(
        lfi: np.ndarray,
        seeds: np.ndarray | None = None,
        area_threshold: int = 1,
        smooth_iter: int = 10,
        smooth_lambda: float = 0.5,
        smooth_mu: float = -0.53,
        trim_bounds: tuple = (0, 0, 200, 200),
        coord_decimals: int = 6,
        verbose: bool = False,
        method: str = 'taubin',
        ma_window: int = 3,
        corner_angle_deg: float = 30.0,
        upscale_factor: int = 1,
        thin_grain_px: float = 0.0,
        fix_diagonal: bool = True,
        merge_enclosed: bool = True,
        close_staircase: bool = True,
        **seed_kwargs,
) -> dict:
    """
    Full post-pixellation smoothing pipeline for a single MC grain-structure slice.

    Steps
    -----
    1  Merge grains <= area_threshold px into their largest neighbour.
    2  Generate seeds (if not supplied) and tessellate into GrainManifold2D.
    3  Taubin smoothing of polygon boundaries.
    4  Trim to RVE.
    5  Remove empty / invalid polygons.
    6  Build polygon neighbour graph.
    7  Renumber GIDs to contiguous 1 → N.
    8  Geometry validation.
    9  Symmetric grain-pair list.
    10 Shared-boundary interface extraction.
    11 Junction-point (triple-point) extraction.

    Parameters
    ----------
    lfi            : 2-D integer label field (0 = background).
    seeds          : (N, 2) Cartesian seed array.  Generated internally when None.
    area_threshold : Grains with <= this many pixels are merged (Step 1).
    smooth_iter    : Taubin smoothing iterations.
    smooth_lambda  : Taubin forward-pass weight.
    smooth_mu      : Taubin backward-pass weight (must be negative).
    trim_bounds    : (xmin, ymin, xmax, ymax) — RVE clip rectangle.
    coord_decimals : Decimal places for junction-point coordinate rounding.
    verbose        : Print step-by-step progress.
    **seed_kwargs  : Extra kwargs forwarded to generate_constrained_hybrid_seeds
                     when seeds is None (e.g. target_spacing, bulk_spacing).

    Returns
    -------
    dict
        manifold              GrainManifold2D instance (cells already updated).
        cells                 {gid: Shapely Polygon}.
        polygon_neighbors     {gid: [gids]}.
        validity_report       {gid: {'is_valid': bool, 'has_area': bool, 'area': float}}.
        cell_pairs_list       [(gid1, gid2), ...] — sorted symmetric pairs.
        cell_pair_interfaces  {(gid1, gid2): LineString | MultiLineString}.
        junction_points       [Point, ...].
        jp_dict               {jp_id: [order, (gids,)]}.
        old_to_new_gid        {old_gid: new_gid}.
        n_grains              int.
        n_invalid             int.
    """
    # ── Step -2: fix diagonal-only pixel connectivity ────────────────────────
    if fix_diagonal:
        if verbose:
            print('[smooth_gs_slice] Step -2 — fix diagonal pinches')
        lfi = _fix_diagonal_pinches(lfi)

    # ── Step -1: fill staircase concavities in thin grains ───────────────────
    if close_staircase and thin_grain_px > 0:
        if verbose:
            print(f'[smooth_gs_slice] Step -1 — fill staircase concavities '
                  f'(thin_grain_px={thin_grain_px})')
        lfi = _fill_staircase_concavities(lfi, thin_grain_px)

    # ── Step 0: nearest-neighbour upscaling ──────────────────────────────────
    if upscale_factor > 1:
        if verbose:
            print(f'[smooth_gs_slice] Step 0 — upscale LFI ×{upscale_factor}')
        lfi = np.repeat(np.repeat(lfi, upscale_factor, axis=0), upscale_factor, axis=1)
        trim_bounds = tuple(v * upscale_factor for v in trim_bounds)
        area_threshold = area_threshold * upscale_factor ** 2
        thin_grain_px = thin_grain_px * upscale_factor  # width scales linearly

    # ── Step 1: merge sub-threshold grains ───────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 1 — merge small grains')
    lfi_merged = _merge_small_grains(lfi, area_threshold=area_threshold)

    # ── Step 2: generate seeds + tessellate ──────────────────────────────────
    if seeds is None:
        if verbose:
            print('[smooth_gs_slice] Step 2a — generating seeds')
        _defaults = dict(target_spacing=1.0, bulk_spacing=1.0,
                         jitter_factor=0.25, margin=0.5, padding=2.0)
        _defaults.update(seed_kwargs)
        seeds = gridOps.generate_constrained_hybrid_seeds(lfi_merged, **_defaults)

    if verbose:
        print('[smooth_gs_slice] Step 2b — tessellate into GrainManifold2D')
    manifold = GrainManifold2D.by_tessellation(lfi_merged, seeds)

    # ── Step 2c: merge enclosed polygons ─────────────────────────────────────
    if merge_enclosed:
        n_before_enc = len(manifold.cells)
        manifold.cells = _merge_enclosed_polygons(manifold.cells)
        if verbose:
            n_merged = n_before_enc - len(manifold.cells)
            print(f'[smooth_gs_slice] Step 2c — merged {n_merged} enclosed polygon(s); '
                  f'{len(manifold.cells)} remain')

    # ── Step 3: smooth ──────────────────────────────────────────────────────
    if verbose:
        label = (f'Taubin ({smooth_iter} iters, λ={smooth_lambda}, μ={smooth_mu})'
                 if method == 'taubin'
                 else f'moving_average (window={ma_window}, {smooth_iter} passes)')
        print(f'[smooth_gs_slice] Step 3 — {label}')
    manifold.smooth_interfaces(
        iterations=smooth_iter, lmbda=smooth_lambda, mu=smooth_mu,
        method=method, ma_window=ma_window,
        corner_angle_deg=corner_angle_deg,
        thin_grain_px=thin_grain_px,
    )

    # ── Step 4: trim to RVE ──────────────────────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 4 — trim to RVE')
    manifold.trim_to_rve(bounds=trim_bounds)

    # ── Step 5: remove empty / invalid polygons ───────────────────────────────
    n_before = len(manifold.cells)
    cleaned = {gid: geom for gid, geom in manifold.cells.items()
               if geom is not None and geom.is_valid and geom.area > 0}
    manifold.cells = cleaned
    if verbose:
        print(f'[smooth_gs_slice] Step 5 — removed {n_before - len(cleaned)} '
              f'empty/invalid; {len(cleaned)} remain')

    # ── Step 6: build neighbour graph ────────────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 6 — build neighbour graph')
    polygon_neighbors = _compute_polygon_neighbors(manifold.cells)

    # ── Step 7: renumber GIDs to contiguous 1 → N ───────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 7 — renumber GIDs')
    old_to_new_gid: dict = {}
    new_cells: dict = {}
    for new_gid, old_gid in enumerate(sorted(manifold.cells.keys()), start=1):
        old_to_new_gid[old_gid] = new_gid
        new_cells[new_gid] = manifold.cells[old_gid]
    manifold.cells = new_cells

    polygon_neighbors = {
        old_to_new_gid[og]: [old_to_new_gid[n] for n in nbrs if n in old_to_new_gid]
        for og, nbrs in polygon_neighbors.items()
        if og in old_to_new_gid
    }

    # ── Step 8: geometry validation ──────────────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 8 — geometry validation')
    validity_report: dict = {}
    for gid, geom in manifold.cells.items():
        validity_report[gid] = {
            'is_valid': geom.is_valid if geom is not None else False,
            'has_area': geom.area > 0 if geom is not None else False,
            'area':     geom.area if geom is not None else 0.0,
        }
    n_invalid = sum(1 for v in validity_report.values() if not v['is_valid'])

    # ── Step 9: symmetric grain-pair list ────────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 9 — build symmetric pair list')
    cell_pairs: set = set()
    for gid1, nbrs in polygon_neighbors.items():
        for gid2 in nbrs:
            cell_pairs.add(tuple(sorted((gid1, gid2))))
    cell_pairs_list = sorted(cell_pairs)

    # ── Step 10: shared-boundary interface extraction ─────────────────────────
    if verbose:
        print(f'[smooth_gs_slice] Step 10 — extract {len(cell_pairs_list)} interfaces')
    cell_pair_interfaces: dict = {}
    for pair in cell_pairs_list:
        gid1, gid2 = pair
        cell_pair_interfaces[pair] = (
            manifold.cells[gid1].boundary.intersection(manifold.cells[gid2].boundary)
        )

    # ── Step 11: junction-point extraction ───────────────────────────────────
    if verbose:
        print('[smooth_gs_slice] Step 11 — extract junction points')
    vertex_to_gids: dict = defaultdict(set)
    for gid, geom in manifold.cells.items():
        if geom is None or geom.is_empty:
            continue
        polys = ([geom] if isinstance(geom, Polygon)
                 else list(geom.geoms) if isinstance(geom, MultiPolygon)
                 else [])
        for poly in polys:
            for x, y in list(poly.exterior.coords)[:-1]:
                key = (round(float(x), coord_decimals), round(float(y), coord_decimals))
                vertex_to_gids[key].add(int(gid))
            for ring in poly.interiors:
                for x, y in list(ring.coords)[:-1]:
                    key = (round(float(x), coord_decimals), round(float(y), coord_decimals))
                    vertex_to_gids[key].add(int(gid))

    junction_items = sorted(
        [(coord, tuple(sorted(gids)))
         for coord, gids in vertex_to_gids.items() if len(gids) >= 3],
        key=lambda t: (t[0][0], t[0][1]),
    )
    junction_points = [Point(x, y) for (x, y), _ in junction_items]
    jp_dict = {
        jp_id: [len(gids), gids]
        for jp_id, (_, gids) in enumerate(junction_items, start=1)
    }

    if verbose:
        print(f'[smooth_gs_slice] Done — {len(manifold.cells)} grains, '
              f'{n_invalid} invalid, {len(junction_points)} junction points')

    return {
        'manifold':             manifold,
        'cells':                manifold.cells,
        'polygon_neighbors':    polygon_neighbors,
        'validity_report':      validity_report,
        'cell_pairs_list':      cell_pairs_list,
        'cell_pair_interfaces': cell_pair_interfaces,
        'junction_points':      junction_points,
        'jp_dict':              jp_dict,
        'old_to_new_gid':       old_to_new_gid,
        'n_grains':             len(manifold.cells),
        'n_invalid':            n_invalid,
    }
