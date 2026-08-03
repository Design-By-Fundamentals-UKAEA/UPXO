"""
validation.py
=============
Pre-tet surface validation gate for the confMesh3d pipeline.
Adapted from kali.validation.pre_tet_surface_validation.

Runs before gmsh tet meshing to detect:
  - Non-watertight grain surfaces
  - Winding-inconsistent normals
  - Zero-area triangles
  - RVE-bounds violations (surface points outside domain)
  - Volume deficit vs voxel volume (shrinkage check)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import time
import numpy as np
import pyvista as pv
import trimesh as tm

from upxo.meshing.confMesh3d.config import ValidationConfig
from upxo.meshing.confMesh3d.complex_builder import ConformalSurfaceComplex


# ---------------------------------------------------------------------------
@dataclass
class ValidationIssue:
    severity   : str   # 'error' | 'warning'
    check_name : str
    grain_id   : int
    value      : float
    message    : str


@dataclass
class ValidationReport:
    """Structured pre-tet surface validation report."""
    issues  : List[ValidationIssue] = field(default_factory=list)
    metrics : Dict[str, object]     = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'error')

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'warning')

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def add(self, *, severity, check_name, grain_id, value, message):
        self.issues.append(ValidationIssue(
            severity=severity, check_name=check_name, grain_id=int(grain_id),
            value=float(value) if value is not None else 0.0, message=message))

    def print_summary(self):
        print(f'Pre-tet validation: '
              f'{"PASS" if self.passed else "FAIL"}  '
              f'errors={self.error_count}  warnings={self.warning_count}')
        for iss in self.issues:
            sym = 'ERR' if iss.severity == 'error' else 'WARN'
            print(f'  [{sym}] grain={iss.grain_id:>5d}  {iss.check_name:30s}  '
                  f'value={iss.value:.4g}  {iss.message}')


# ---------------------------------------------------------------------------
def validate_surface_complex(
        complex_    : ConformalSurfaceComplex,
        lgi         : np.ndarray,
        config      : Optional[ValidationConfig] = None,
        verbose     : bool = True,
) -> ValidationReport:
    """
    Validate every grain surface in the ConformalSurfaceComplex.

    Parameters
    ----------
    complex_ : ConformalSurfaceComplex
    lgi      : ndarray (nx,ny,nz) — used to compute voxel volumes
    config   : ValidationConfig
    verbose  : bool

    Returns
    -------
    ValidationReport
    """
    cfg    = config or ValidationConfig()
    report = ValidationReport()
    verts  = complex_.vertices
    tris   = complex_.triangles
    vs     = complex_.voxel_size
    wall   = complex_.wall_label
    nx, ny, nz = lgi.shape

    rve_bounds = np.array([
        [0.0, float(nx-1)],
        [0.0, float(ny-1)],
        [0.0, float(nz-1)],
    ])

    counts = np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1)

    n_grains     = len(complex_.grain_to_triangle_ids)
    _report_every = max(1, n_grains // 20)
    _done         = 0
    _t0           = time.perf_counter()
    if verbose:
        print(f'[Validation] Checking {n_grains} grain surfaces...')

    for gid, tri_ids in complex_.grain_to_triangle_ids.items():
        gid = int(gid)
        if gid == wall:
            continue
        _done += 1
        if verbose and (_done % _report_every == 0 or _done == n_grains):
            _ela = time.perf_counter() - _t0
            _eta = (_ela / max(_done, 1)) * (n_grains - _done)
            print(f'  [{_done:>5d}/{n_grains}  {100*_done/n_grains:5.1f}%]  '
                  f'elapsed={_ela:.1f}s  ETA={_eta:.0f}s  '
                  f'issues={len(report.issues)}', flush=True)
        if not tri_ids:
            report.add(severity='warning', check_name='empty_surface',
                       grain_id=gid, value=0,
                       message='No triangles found for this grain.')
            continue

        g_tris_raw = tris[np.array(tri_ids, dtype=np.int64)].copy()

        # Fix winding per-grain BEFORE building trimesh.
        # SurfaceNets3D normals point from lower-label → higher-label grain.
        # For interior triangles where this grain is the HIGHER-label neighbour,
        # the stored normal points INTO the grain (inward) — flip those triangles
        # so all normals are outward for this grain's trimesh.
        for i, ti in enumerate(tri_ids):
            src_type = complex_.triangle_source_types[ti]
            adj      = complex_.triangle_adjacent_grains[ti]
            if src_type == 'interior' and len(adj) == 2:
                la = min(int(adj[0]), int(adj[1]))
                lb = max(int(adj[0]), int(adj[1]))
                if gid == lb:                              # grain is higher-label → inward → flip
                    g_tris_raw[i] = g_tris_raw[i, [0, 2, 1]]

        # Build compact trimesh using only this grain's vertices (not full 394K table)
        try:
            unique_vidx, inv = np.unique(g_tris_raw.ravel(), return_inverse=True)
            local_verts = verts[unique_vidx]
            local_tris  = inv.reshape(-1, 3)
            g_mesh = tm.Trimesh(vertices=local_verts, faces=local_tris, process=False)
        except Exception as e:
            report.add(severity='error', check_name='trimesh_build',
                       grain_id=gid, value=0, message=str(e))
            continue

        # 1. Zero-area triangles
        areas = g_mesh.area_faces
        n_zero = int((areas < cfg.zero_area_tol).sum())
        if n_zero > 0:
            report.add(severity='warning', check_name='zero_area_triangles',
                       grain_id=gid, value=n_zero,
                       message=f'{n_zero} triangles have area < {cfg.zero_area_tol:.2g}')

        # 1b. Open-edge diagnostic (first 3 failing grains only)
        _open_edge_diag = (verbose and not g_mesh.is_watertight
                           and sum(1 for i in report.issues if i.check_name == 'not_watertight') < 3)
        if _open_edge_diag:
            # boundary_edges in trimesh is accessed via g_mesh.faces_unique_edges
            # and edge appearance count — open edges appear exactly once
            edges_all = g_mesh.edges_sorted          # (E, 2) all half-edges sorted
            unique, counts = np.unique(edges_all, axis=0, return_counts=True)
            b_edges = unique[counts == 1]            # open (boundary) edges
            if len(b_edges) > 0:
                b_verts = local_verts[np.unique(b_edges)]
                print(f'    Grain {gid}: {len(b_edges)} open edges  '
                      f'x=[{b_verts[:,0].min():.3f},{b_verts[:,0].max():.3f}]  '
                      f'y=[{b_verts[:,1].min():.3f},{b_verts[:,1].max():.3f}]  '
                      f'z=[{b_verts[:,2].min():.3f},{b_verts[:,2].max():.3f}]', flush=True)

        if not g_mesh.is_winding_consistent:
            g_mesh.fix_normals()   # in-place BFS-based winding repair

        # 2. Watertight (closed manifold) — check AFTER winding repair
        if not g_mesh.is_watertight:
            report.add(severity='error', check_name='not_watertight',
                       grain_id=gid, value=0,
                       message='Surface is not watertight (open boundary exists).')

        # 3. Winding consistency — only flag if fix_normals() still failed
        if not g_mesh.is_winding_consistent:
            report.add(severity='error', check_name='winding_inconsistent',
                       grain_id=gid, value=0,
                       message='Winding inconsistent even after fix_normals() — '
                               'likely a non-manifold topology issue.')

        # 4. RVE bounds check
        g_verts = verts[np.unique(g_tris_raw)]
        for axis, (lo, hi) in enumerate(rve_bounds):
            out_lo = float((g_verts[:, axis] < lo - cfg.rve_bounds_tol).sum())
            out_hi = float((g_verts[:, axis] > hi + cfg.rve_bounds_tol).sum())
            if out_lo > 0 or out_hi > 0:
                report.add(severity='warning', check_name='rve_bounds_violation',
                           grain_id=gid, value=out_lo + out_hi,
                           message=f'Axis {axis}: {out_lo+out_hi:.0f} vertices '
                                   f'outside [{lo:.2f}, {hi:.2f}] by > tol.')

        # 5. Volume conservation — only meaningful after winding is consistent
        if g_mesh.is_watertight and g_mesh.is_winding_consistent:
            V_mesh = float(abs(g_mesh.volume))
            V_vox  = float(counts[gid] if gid < len(counts) else 0) * vs**3
            if V_vox > 0:
                err = abs(V_mesh - V_vox) / V_vox
                if err > cfg.volume_tol:
                    sev = 'error' if err > 0.20 else 'warning'
                    report.add(severity=sev, check_name='volume_deficit',
                               grain_id=gid, value=err,
                               message=f'|V_mesh - V_vox|/V_vox = {err:.3f}  '
                                       f'(V_mesh={V_mesh:.3g}, V_vox={V_vox:.3g} um3)')

        # 6. Small grain warning
        n_vox = int(counts[gid]) if gid < len(counts) else 0
        if n_vox < cfg.warn_min_voxels:
            report.add(severity='warning', check_name='small_grain',
                       grain_id=gid, value=n_vox,
                       message=f'Grain has only {n_vox} voxels — '
                               f'tet elements may have poor aspect ratios.')

    report.metrics['n_grains_checked'] = n_grains
    report.metrics['n_errors']         = report.error_count
    report.metrics['n_warnings']       = report.warning_count

    if verbose:
        report.print_summary()

    return report


# ---------------------------------------------------------------------------
def fix_winding(complex_: ConformalSurfaceComplex,
                verbose: bool = True) -> ConformalSurfaceComplex:
    """
    Use trimesh to ensure outward-pointing normals on all grain surfaces.
    Modifies triangle winding in-place where needed.
    """
    import copy
    tris = complex_.triangles.copy()
    wall = complex_.wall_label

    fixed_count = 0
    for gid, tri_ids in complex_.grain_to_triangle_ids.items():
        if int(gid) == wall or not tri_ids:
            continue
        idx = np.array(tri_ids, dtype=np.int64)
        g_tris = tris[idx]
        try:
            g_mesh = tm.Trimesh(vertices=complex_.vertices, faces=g_tris, process=False)
            if not g_mesh.is_winding_consistent:
                g_mesh.fix_normals()
                tris[idx] = g_mesh.faces
                fixed_count += 1
        except Exception:
            pass

    if verbose and fixed_count:
        print(f'[Validation] Fixed winding for {fixed_count} grain surfaces.')

    new_complex = copy.copy(complex_)
    new_complex.triangles = tris
    return new_complex


# ---------------------------------------------------------------------------
def visualize_validation(
        complex_ : ConformalSurfaceComplex,
        report   : ValidationReport,
        show     : bool = True,
) -> pv.Plotter:
    """
    PyVista visualisation highlighting validation failures.
    Error grains → red, warning grains → orange, passing grains → blue.
    """
    verts = complex_.vertices
    tris  = complex_.triangles

    error_grains   = {i.grain_id for i in report.issues if i.severity == 'error'}
    warning_grains = {i.grain_id for i in report.issues if i.severity == 'warning'}

    # Build per-triangle status label (0=pass, 1=warn, 2=error)
    status = np.zeros(len(tris), dtype=np.uint8)
    for gid, tri_ids in complex_.grain_to_triangle_ids.items():
        gid = int(gid)
        s = 2 if gid in error_grains else (1 if gid in warning_grains else 0)
        for t in tri_ids:
            status[t] = s

    faces_pv = np.hstack([np.full((len(tris),1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(verts, faces_pv)
    poly.cell_data['status'] = status

    pvp = pv.Plotter()
    pvp.set_background('white')
    pvp.add_mesh(poly, scalars='status',
                 cmap=['steelblue', 'darkorange', 'crimson'],
                 clim=[0, 2], show_edges=False,
                 scalar_bar_args={'title': 'Status: 0=pass 1=warn 2=error'})
    pvp.add_title(f'Pre-tet validation  '
                  f'errors={report.error_count}  warnings={report.warning_count}',
                  font_size=10)
    pvp.add_axes()
    if show:
        pvp.show()
    return pvp
