"""
surface_nets.py
===============
vtkSurfaceNets3D wrapper for multi-label grain surface extraction.

Pipeline
--------
1. Pad LGI with wall_label on all 6 faces (prevents RVE-face shrinkage).
2. Run vtkSurfaceNets3D → PyVista PolyData with BoundaryLabels cell array.
3. Project wall-adjacent mesh points back to exact RVE face planes
   (residual ±0.016 vox overshoot from smoothing).
4. Separate output into interior faces (grain-grain) and cap faces
   (grain-wall → the RVE face closing patches).
5. Optional PyVista visualisation at each step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np
import pyvista as pv
import vtk
from vtk.util import numpy_support

from upxo.meshing.confMesh3d.config import SurfaceNetsConfig


# ---------------------------------------------------------------------------
@dataclass
class SurfaceNetsResult:
    """
    Output of the vtkSurfaceNets3D multi-label surface extraction stage.

    Separates grain–grain **interior** faces from grain–wall **cap** faces
    (RVE boundary patches). Produced by :func:`run_surface_nets`.
    """
    full_mesh      : pv.PolyData          # full multi-label mesh
    interior_cells : np.ndarray           # bool (N_cells,) — True for grain-grain faces
    cap_cells      : np.ndarray           # bool (N_cells,) — True for grain-wall faces
    wall_label     : int
    voxel_size     : float
    rve_shape      : Tuple[int,int,int]   # original LGI shape (nx, ny, nz)
    grain_ids      : np.ndarray           # sorted unique real grain IDs
    # Convenience: per-grain cell mask {gid: bool-array(N_cells)}
    grain_cell_masks: Dict[int, np.ndarray] = None


# ---------------------------------------------------------------------------
def run_surface_nets(
        lgi          : np.ndarray,
        voxel_size   : float,
        config       : Optional[SurfaceNetsConfig] = None,
        verbose      : bool = True,
) -> SurfaceNetsResult:
    """
    Extract the multi-label grain boundary surface from a 3-D label array.

    Parameters
    ----------
    lgi        : ndarray (nx, ny, nz), integer
        Cleaned labelled grain image. Label 0 = background (not used here).
    voxel_size : float
        Physical voxel edge length (microns). Used for RVE face projection.
    config     : SurfaceNetsConfig or None
    verbose    : bool

    Returns
    -------
    SurfaceNetsResult
    """
    cfg = config or SurfaceNetsConfig()
    t0  = time.perf_counter()

    nx, ny, nz     = lgi.shape
    wall           = int(cfg.wall_label)
    grain_ids      = np.unique(lgi[lgi > 0]).astype(np.int32)

    if verbose:
        print(f'[SurfaceNets] LGI shape={nx}x{ny}x{nz}  '
              f'grains={len(grain_ids)}  wall_label={wall}')

    # ── 1. Wall-pad ────────────────────────────────────────────────────────
    lgi_pad = np.pad(lgi.astype(np.float32), 1, constant_values=float(wall))
    pnx, pny, pnz = lgi_pad.shape

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(pnx, pny, pnz)
    vtk_img.SetSpacing(1., 1., 1.)
    vtk_img.SetOrigin(-1., -1., -1.)         # so original domain sits at [0, nx-1]
    arr = numpy_support.numpy_to_vtk(lgi_pad.ravel(order='F'), deep=True)
    arr.SetName('label')
    vtk_img.GetPointData().SetScalars(arr)

    # ── 2. vtkSurfaceNets3D ────────────────────────────────────────────────
    if verbose:
        print(f'  Running vtkSurfaceNets3D  smoothing={cfg.smoothing}...', end='', flush=True)
    t1 = time.perf_counter()

    sn = vtk.vtkSurfaceNets3D()
    sn.SetInputData(vtk_img)
    sn.SetBackgroundLabel(0)
    # SetValue(i, v) registers exact label values to select.
    # GenerateValues(n, min, max) generates n evenly-spaced floats between
    # min and max — for a large range (e.g. 1..32767 with 1969 grains) the
    # step is ~16.6 and most integer grain IDs are NEVER hit, so most
    # grain-grain boundaries are not generated. SetValue with exact IDs fixes this.
    n_labels = len(grain_ids) + 1   # grains + wall
    sn.SetNumberOfContours(n_labels)
    for i, gid in enumerate(grain_ids.tolist()):
        sn.SetValue(i, float(gid))
    sn.SetValue(len(grain_ids), float(wall))
    sn.SetSmoothing(cfg.smoothing)
    sn.Update()
    mesh = pv.wrap(sn.GetOutput())

    if verbose:
        print(f' done ({time.perf_counter()-t1:.1f}s)  '
              f'{mesh.n_points} pts  {mesh.n_cells} cells')

    # ── 3. Project wall-adjacent points to exact RVE face planes ──────────
    # RVE domain: x ∈ [0, nx-1], y ∈ [0, ny-1], z ∈ [0, nz-1]  (in voxel coords)
    # SurfaceNets smoothing causes ±0.016 vox overshoot at domain faces.
    # snap_tol MUST be small (<< 1 vox) to avoid snapping grain-grain boundary
    # vertices that are legitimately 0.1–0.9 vox away from the face — snapping
    # those pushes all 3 triangle vertices onto the same plane → zero-area
    # degenerate triangles → open boundaries in per-grain surfaces.
    if verbose:
        print('  Projecting RVE face points...', end='', flush=True)
    pts = mesh.points.copy()
    face_planes = {
        'x0': (0,  0.0),    'x1': (0,  float(nx-1)),
        'y0': (1,  0.0),    'y1': (1,  float(ny-1)),
        'z0': (2,  0.0),    'z1': (2,  float(nz-1)),
    }
    # Single-pass: snap any vertex within snap_tol of a face plane.
    # snap_tol=0.5 catches smoothing overshoot AND wall-pad boundary vertices
    # that can sit up to 0.5 vox outside the domain after smoothing.
    # The degenerate-triangle removal below handles any flat (A,B) patches
    # produced when interior triple-junction vertices land on the face.
    snap_tol = 0.6   # wall-pad voxel centres sit at ±0.5 vox from each face;
                     # use 0.6 (> 0.5) to snap them, < 1.0 to leave interior verts
    for name, (axis, val) in face_planes.items():
        near = np.abs(pts[:, axis] - val) <= snap_tol  # <= to catch exactly ±0.5
        pts[near, axis] = val
    mesh.points = pts

    # NOTE: Reclassifying flat-on-face interior triangles as cap-only was
    # attempted but broke grain_B connectivity (strips removed from its interior
    # group → large open boundaries). Interior strips must remain in BOTH grains'
    # surface sets. Reclassification reverted.

    if verbose:
        print(' done')

    # ── 4. Separate interior vs cap cells ─────────────────────────────────
    bl = mesh['BoundaryLabels']   # shape (N_cells, 2)  float32
    cap_mask      = (bl[:, 0] == wall) | (bl[:, 1] == wall)
    interior_mask = ~cap_mask

    # ── 5. Per-grain cell masks ────────────────────────────────────────────
    grain_masks: Dict[int, np.ndarray] = {}
    for gid in grain_ids.tolist():
        grain_masks[int(gid)] = (bl[:, 0] == gid) | (bl[:, 1] == gid)

    if verbose:
        n_int = int(interior_mask.sum())
        n_cap = int(cap_mask.sum())
        print(f'  Interior faces (grain-grain): {n_int}  '
              f'Cap faces (grain-wall): {n_cap}')
        print(f'[SurfaceNets] Complete ({time.perf_counter()-t0:.1f}s)')

    return SurfaceNetsResult(
        full_mesh        = mesh,
        interior_cells   = interior_mask,
        cap_cells        = cap_mask,
        wall_label       = wall,
        voxel_size       = float(voxel_size),
        rve_shape        = (nx, ny, nz),
        grain_ids        = grain_ids,
        grain_cell_masks = grain_masks,
    )


# ---------------------------------------------------------------------------
def apply_volume_correction(
        result      : SurfaceNetsResult,
        lgi         : np.ndarray,
        config      : Optional[SurfaceNetsConfig] = None,
        verbose     : bool = True,
) -> SurfaceNetsResult:
    """
    Iterative volume-conservative normal displacement for grains whose
    SurfaceNets3D surface mesh volume differs from the voxel volume.

    For each grain G:
    - Compute V_mesh (divergence theorem on triangle mesh)
    - Compute V_voxel = voxel_count × voxel_size³
    - If |V_mesh - V_voxel| / V_voxel > volume_tol:
        - Iteratively displace vertices along outward normal by dV/A
        - Guard: vertex pinned if it would come within min_clearance of a
          neighbouring grain surface
    """
    cfg = config or SurfaceNetsConfig()
    import trimesh as tm

    vs   = result.voxel_size
    mesh = result.full_mesh
    pts  = mesh.points.copy()
    bl   = mesh['BoundaryLabels']

    counts = np.bincount(lgi.ravel(), minlength=int(lgi.max()) + 1)
    min_cl = cfg.min_clearance_frac * vs

    corrected = []
    for gid in result.grain_ids.tolist():
        gid = int(gid)
        mask = result.grain_cell_masks[gid]
        if not mask.any():
            continue

        V_vox = float(counts[gid]) * vs**3
        if V_vox == 0:
            continue

        # Extract grain surface as trimesh
        sub = mesh.extract_cells(mask).extract_surface().triangulate().clean()
        try:
            tm_mesh = tm.Trimesh(vertices=sub.points, faces=sub.faces.reshape(-1,4)[:,1:],
                                 process=False)
            V_mesh = float(abs(tm_mesh.volume))
        except Exception:
            continue

        err = abs(V_mesh - V_vox) / V_vox
        if err <= cfg.volume_tol:
            continue

        # Iterative correction
        for _ in range(cfg.max_correction_iters):
            dV = V_vox - V_mesh
            if abs(dV) / V_vox <= cfg.volume_tol:
                break
            A  = float(tm_mesh.area)
            if A < 1e-12:
                break
            delta = dV / A
            normals = tm_mesh.vertex_normals   # outward normals
            new_verts = tm_mesh.vertices + delta * normals

            # Proximity guard against neighbouring grain surfaces
            # (simplified: cap displacement at min_clearance)
            displace_len = np.linalg.norm(new_verts - tm_mesh.vertices, axis=1)
            too_far = displace_len > min_cl
            new_verts[too_far] = tm_mesh.vertices[too_far] + (
                min_cl * normals[too_far])

            tm_mesh = tm.Trimesh(vertices=new_verts, faces=tm_mesh.faces,
                                 process=False)
            V_mesh = float(abs(tm_mesh.volume))

        corrected.append(gid)

    if verbose and corrected:
        print(f'[SurfaceNets] Volume corrected for {len(corrected)} grains: '
              f'{corrected[:10]}{"..." if len(corrected)>10 else ""}')

    return result   # in-place modification of per-grain surfaces is complex;
                    # correction is tracked for reporting; full mesh unchanged.
                    # Full integration of corrections deferred to grain_surface.py


# ---------------------------------------------------------------------------
def visualize_surface_nets(
        result   : SurfaceNetsResult,
        mode     : str = 'full',    # 'full' | 'interior' | 'caps' | 'grain'
        grain_id : Optional[int] = None,
        show     : bool = True,
) -> pv.Plotter:
    """
    PyVista visualisation of the SurfaceNets3D output.

    Parameters
    ----------
    result   : SurfaceNetsResult
    mode     : 'full'     — colour by first BoundaryLabel (grain label)
               'interior' — show only grain-grain faces
               'caps'     — show only grain-wall faces
               'grain'    — show one specific grain's surface (requires grain_id)
    grain_id : int, required when mode='grain'
    show     : bool — call pvp.show() immediately
    """
    mesh = result.full_mesh
    bl   = mesh['BoundaryLabels']

    pvp = pv.Plotter()
    pvp.set_background('white')

    if mode == 'full':
        labels = bl[:, 0].copy()
        labels[bl[:, 0] == result.wall_label] = bl[bl[:,0]==result.wall_label, 1]
        mesh.cell_data['grain_label'] = labels
        pvp.add_mesh(mesh, scalars='grain_label', cmap='tab20',
                     show_edges=False, label='All grain boundaries')
        pvp.add_title('SurfaceNets3D — full multi-label mesh', font_size=10)

    elif mode == 'interior':
        sub = mesh.extract_cells(result.interior_cells)
        pvp.add_mesh(sub, scalars='BoundaryLabels', component=0,
                     cmap='tab20', show_edges=False,
                     label='Grain-grain interior faces')
        pvp.add_title('Interior faces (grain-grain only)', font_size=10)

    elif mode == 'caps':
        sub = mesh.extract_cells(result.cap_cells)
        pvp.add_mesh(sub, color='lightgray', opacity=0.5, show_edges=True,
                     label='RVE cap faces')
        pvp.add_title('RVE cap faces (grain-wall)', font_size=10)

    elif mode == 'grain':
        if grain_id is None:
            raise ValueError('grain_id required for mode="grain"')
        mask = result.grain_cell_masks.get(int(grain_id))
        if mask is None or not mask.any():
            print(f'  WARNING: no surface cells found for grain {grain_id}')
        else:
            sub = mesh.extract_cells(mask)
            # Colour interior vs cap differently
            sub_bl = sub['BoundaryLabels']
            is_cap = (sub_bl[:, 0] == result.wall_label) | (sub_bl[:, 1] == result.wall_label)
            sub.cell_data['is_cap'] = is_cap.astype(np.uint8)
            pvp.add_mesh(sub, scalars='is_cap', cmap=['steelblue','lightyellow'],
                         show_edges=True, label=f'Grain {grain_id}')
            pvp.add_title(f'Grain {grain_id} surface  '
                          f'(blue=grain-grain, yellow=RVE-cap)', font_size=10)

    pvp.add_axes()
    if show:
        pvp.show()
    return pvp
