"""
export.py
=========
Export the conformal tet mesh to Abaqus INP (partitioned) and optional
additional formats via meshio.

Follows the same file-split convention as abaqus_exporter_3d.py:
  model_master.inp
  01_nodes.inp
  02_elements.inp
  03a_elsets_cells.inp      per-grain cell ELSETs (material assignment)
  03b_elsets_roles.inp      role grouping (if twin_role provided)
  04_nsets_bc.inp           boundary-condition node sets (RVE faces)
  05_materials.inp          one *Material per grain (Bunge-Euler)
  06_sections.inp           *Solid Section linking ELSETs to materials
  07_interactions.inp       stub
  08_steps_output.inp       stub

ELSET naming: es_cell_<role>_<gid>  / MAT_CELL_<ROLE>_<gid>
(identical convention to abaqus_exporter_3d.py; underscores throughout --
Abaqus keyword-file names do not permit periods)
"""
from __future__ import annotations

import os
import math
import time
from typing import Dict, Optional

import numpy as np

from upxo.meshing.confMesh3d.config import ExportConfig
from upxo.meshing.confMesh3d.complex_builder import GmshVolumeResult

try:
    import gmsh
except ImportError:
    gmsh = None

try:
    import meshio
except ImportError:
    meshio = None

# Role → (ELSET prefix, MAT prefix)
_ROLE_PREFIX = {
    'non_host':       ('es_cell_nonpart',  'MAT_CELL_NONPART'),
    'host':           ('es_cell_host',     'MAT_CELL_HOST'),
    'primary_twin':   ('es_cell_ptwin',    'MAT_CELL_PTWIN'),
    'stwin_a':        ('es_cell_stwin_a',  'MAT_CELL_STWIN_A'),
    'stwin_b':        ('es_cell_stwin_b',  'MAT_CELL_STWIN_B'),
}


def _quat_to_bunge(q: np.ndarray):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    r33 = 1 - 2*(x*x + y*y)
    r31 = 2*(x*z - y*w); r32 = 2*(y*z + x*w)
    r13 = 2*(x*z + y*w); r23 = 2*(y*z - x*w); r11 = 1-2*(y*y+z*z); r12=2*(x*y-z*w)
    sin_Phi = math.sqrt(max(0.0, 1.0 - r33*r33))
    if sin_Phi > 1e-6:
        Phi  = math.acos(max(-1.0, min(1.0, r33)))
        phi1 = math.atan2(r31, -r32)
        phi2 = math.atan2(r13,  r23)
    else:
        Phi  = 0.0 if r33 > 0 else math.pi
        phi1 = math.atan2(-r12, r11)
        phi2 = 0.0
    return math.degrees(phi1), math.degrees(Phi), math.degrees(phi2)


def _write_id_list(f, ids: np.ndarray, per_line: int = 16):
    ids = ids.ravel()
    for start in range(0, len(ids), per_line):
        f.write(', '.join(str(v) for v in ids[start:start+per_line]) + '\n')


# ---------------------------------------------------------------------------

def export_conformal_mesh(
        gmsh_result : GmshVolumeResult,
        all_quats   : Dict[int, np.ndarray],
        twin_role   : Optional[Dict[int, str]]   = None,
        twin_parent_of: Optional[Dict[int,int]]  = None,
        voxel_size  : float = 1.0,
        config      : Optional[ExportConfig]     = None,
        verbose     : bool = True,
) -> None:
    """
    Export the conformal tet mesh from the active gmsh model.

    Must be called BEFORE gmsh.finalize().

    Parameters
    ----------
    gmsh_result    : GmshVolumeResult from generate_conformal_tet_mesh()
    all_quats      : {gid: quaternion(4,)} for material orientations
    twin_role      : {gid: role_str} optional — for role-based ELSETs
    twin_parent_of : {child: parent} optional — for 2a/2b distinction
    voxel_size     : float — physical voxel edge (microns)
    config         : ExportConfig
    """
    if gmsh is None:
        raise RuntimeError('gmsh required for export.')
    cfg = config or ExportConfig()
    if not cfg.out_dir:
        raise ValueError('ExportConfig.out_dir must be set.')
    os.makedirs(cfg.out_dir, exist_ok=True)

    t0 = time.perf_counter()
    if verbose:
        print(f'[Export] Writing Abaqus INP to {cfg.out_dir}')

    # ── Extract nodes from gmsh ────────────────────────────────────────────
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
    node_coords_um = node_coords * voxel_size  # convert voxel → microns
    tag_to_idx = {int(t): i + 1 for i, t in enumerate(node_tags)}  # 1-based
    n_nodes = len(node_tags)

    # ── Extract elements per grain ─────────────────────────────────────────
    # grain_id → list of (element_tag, [node_ids 1-based])
    grain_elements: Dict[int, list] = {}
    eid_counter = [1]

    for gid, vtag in gmsh_result.volume_tags_by_grain_id.items():
        gid = int(gid)
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim=3, tag=int(vtag))
        elems = []
        for etyp, et_arr, en_arr in zip(etypes, etags, enodes):
            npe = 4 if int(etyp) == 4 else 10
            flat = np.array(en_arr, dtype=np.int64).reshape(-1, npe)
            for row in flat:
                mapped = [tag_to_idx[n] for n in row.tolist()]
                elems.append((eid_counter[0], mapped))
                eid_counter[0] += 1
        grain_elements[gid] = elems

    n_elems = eid_counter[0] - 1
    elem_type_abq = 'C3D4' if cfg.material_format != 'C3D10' else 'C3D10'

    # Determine role for each grain
    _primary_gids = {g for g, r in (twin_role or {}).items() if r == 'primary_twin'}
    def _role_key(gid):
        if not twin_role: return 'non_host'
        r = twin_role.get(int(gid), 'non_host')
        if r == 'secondary_twin' and twin_parent_of:
            parent = twin_parent_of.get(int(gid))
            return 'stwin_b' if parent in _primary_gids else 'stwin_a'
        return r

    steps = [
        ('01_nodes.inp',           lambda f: _write_nodes(f, node_tags, node_coords_um, tag_to_idx)),
        ('02_elements.inp',        lambda f: _write_elements(f, grain_elements, elem_type_abq)),
        ('03a_elsets_cells.inp',   lambda f: _write_elsets_cells(f, grain_elements, twin_role, _role_key)),
        ('03b_elsets_roles.inp',   lambda f: _write_elsets_roles(f, grain_elements, twin_role, _role_key)),
        ('04_nsets_bc.inp',        lambda f: _write_nsets_bc(f, node_tags, node_coords, tag_to_idx)),
        ('05_materials.inp',       lambda f: _write_materials(f, grain_elements, all_quats, twin_role, _role_key, cfg)),
        ('06_sections.inp',        lambda f: _write_sections(f, grain_elements, twin_role, _role_key)),
        ('07_interactions.inp',    lambda f: f.write('** stub\n')),
        ('08_steps_output.inp',    lambda f: f.write('** stub\n')),
    ]

    for fname, writer in steps:
        fpath = os.path.join(cfg.out_dir, fname)
        if verbose:
            print(f'  Writing {fname}...', end='', flush=True)
        t1 = time.perf_counter()
        with open(fpath, 'w') as f:
            writer(f)
        print(f' done ({time.perf_counter()-t1:.1f}s)')

    _write_master(cfg.out_dir, n_nodes, n_elems, [s[0] for s in steps], elem_type_abq)

    # ── Optional additional formats via meshio ─────────────────────────────
    if cfg.extra_formats and meshio:
        _export_extra(cfg, grain_elements, node_coords_um, tag_to_idx, twin_role, verbose)

    if verbose:
        print(f'[Export] Done ({time.perf_counter()-t0:.1f}s)  '
              f'nodes={n_nodes}  elements={n_elems}')


# ---------------------------------------------------------------------------
# Abaqus writers
# ---------------------------------------------------------------------------

def _write_nodes(f, node_tags, coords_um, tag_to_idx):
    f.write('** Node coordinates (microns)\n*Node\n')
    for i, (tag, xyz) in enumerate(zip(node_tags, coords_um)):
        f.write(f'{tag_to_idx[int(tag)]}, {xyz[0]:.8g}, {xyz[1]:.8g}, {xyz[2]:.8g}\n')


def _write_elements(f, grain_elements, elem_type):
    f.write(f'** Element connectivity\n*Element, type={elem_type}\n')
    for gid, elems in sorted(grain_elements.items()):
        for eid, nodes in elems:
            f.write(f'{eid}, ' + ', '.join(str(n) for n in nodes) + '\n')


def _write_elsets_cells(f, grain_elements, twin_role, role_key_fn):
    f.write('** Per-grain cell ELSETs (material assignment, no overlap)\n')
    for gid, elems in sorted(grain_elements.items()):
        rk  = role_key_fn(gid)
        pfx = _ROLE_PREFIX.get(rk, _ROLE_PREFIX['non_host'])[0]
        eids = np.array([e[0] for e in elems], dtype=np.int64)
        f.write(f'*Elset, elset={pfx}_{gid}\n')
        _write_id_list(f, eids)


def _write_elsets_roles(f, grain_elements, twin_role, role_key_fn):
    if not twin_role:
        f.write('** (no twin_role provided — role ELSETs skipped)\n')
        return
    f.write('** Role grouping ELSETs (deduplication allowed)\n')
    role_buckets: Dict[str, list] = {}
    for gid, elems in grain_elements.items():
        rk = role_key_fn(gid)
        role_buckets.setdefault(rk, []).extend(e[0] for e in elems)
    role_to_es = {
        'non_host':     'es_role_nonpart',
        'host':         'es_role_host',
        'primary_twin': 'es_role_ptwin',
        'stwin_a':      'es_role_stwin_a',
        'stwin_b':      'es_role_stwin_b',
    }
    for rk, eids in sorted(role_buckets.items()):
        name = role_to_es.get(rk, f'es_role_{rk}')
        f.write(f'*Elset, elset={name}\n')
        _write_id_list(f, np.array(eids, dtype=np.int64))


def _write_nsets_bc(f, node_tags, coords, tag_to_idx):
    f.write('** Boundary-condition node sets (RVE faces)\n')
    tol = 0.05
    bounds = {
        'XMIN': coords[:,0] < coords[:,0].min() + tol,
        'XMAX': coords[:,0] > coords[:,0].max() - tol,
        'YMIN': coords[:,1] < coords[:,1].min() + tol,
        'YMAX': coords[:,1] > coords[:,1].max() - tol,
        'ZMIN': coords[:,2] < coords[:,2].min() + tol,
        'ZMAX': coords[:,2] > coords[:,2].max() - tol,
    }
    for name, mask in bounds.items():
        nids = np.array([tag_to_idx[int(t)] for t, m in zip(node_tags, mask) if m],
                        dtype=np.int64)
        f.write(f'*Nset, nset=ns_face_{name}\n')
        _write_id_list(f, nids)


def _write_materials(f, grain_elements, all_quats, twin_role, role_key_fn, cfg):
    f.write('** One *Material per grain (Bunge-Euler in degrees)\n**\n')
    for gid in sorted(grain_elements.keys()):
        rk  = role_key_fn(gid)
        pfx = _ROLE_PREFIX.get(rk, _ROLE_PREFIX['non_host'])[1]
        q   = all_quats.get(int(gid), np.array([1.,0.,0.,0.]))
        p1, P, p2 = _quat_to_bunge(q)
        f.write(f'*Material, name={pfx}_{gid}\n')
        f.write(f'** Bunge-Euler (deg): phi1={p1:.4f}, Phi={P:.4f}, phi2={p2:.4f}\n')
        if cfg.material_format == 'bunge_euler':
            f.write(f'*User Material, constants=3\n{p1:.6f}, {P:.6f}, {p2:.6f}\n')
            f.write(f'*Depvar\n{cfg.n_depvar}\n')
        else:
            f.write('*Elastic\n210000., 0.3\n')
        f.write('**\n')


def _write_sections(f, grain_elements, twin_role, role_key_fn):
    f.write('** *Solid Section linking per-grain ELSETs to materials\n**\n')
    for gid in sorted(grain_elements.keys()):
        rk  = role_key_fn(gid)
        es_pfx  = _ROLE_PREFIX.get(rk, _ROLE_PREFIX['non_host'])[0]
        mat_pfx = _ROLE_PREFIX.get(rk, _ROLE_PREFIX['non_host'])[1]
        f.write(f'*Solid Section, elset={es_pfx}_{gid}, material={mat_pfx}_{gid}\n,\n')


def _write_master(out_dir, n_nodes, n_elems, fnames, elem_type):
    path = os.path.join(out_dir, 'model_master.inp')
    with open(path, 'w') as f:
        f.write('** Made with UPXO confMesh3d -- conformal twinned grain structure\n**\n')
        f.write('*Heading\n')
        f.write(f'** Conformal tet mesh  element type: {elem_type}\n')
        f.write(f'** Nodes: {n_nodes:,}  Elements: {n_elems:,}  (micron units)\n**\n')
        f.write('*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n**\n')
        for fname in fnames:
            f.write(f'*INCLUDE, INPUT={fname}\n')


def _export_extra(cfg, grain_elements, coords_um, tag_to_idx, twin_role, verbose):
    """Export additional mesh formats via meshio."""
    all_nodes  = coords_um                         # (N, 3)
    all_conn   = []
    all_gids   = []
    for gid, elems in sorted(grain_elements.items()):
        for eid, nodes in elems:
            all_conn.append([n-1 for n in nodes])  # 0-based for meshio
            all_gids.append(gid)

    conn_arr = np.array(all_conn, dtype=np.int64)
    cells    = [('tetra', conn_arr)]
    point_data = {}
    cell_data  = {'grain_id': [np.array(all_gids, dtype=np.int32)]}
    mi_mesh = meshio.Mesh(points=all_nodes, cells=cells,
                          cell_data=cell_data)
    for fmt in cfg.extra_formats:
        ext = {'vtk': '.vtk', 'xdmf': '.xdmf', 'vtu': '.vtu'}.get(fmt, f'.{fmt}')
        fpath = os.path.join(cfg.out_dir, f'conformal_mesh{ext}')
        if verbose:
            print(f'  Writing {fmt} format...', end='', flush=True)
        try:
            meshio.write(fpath, mi_mesh)
            print(' done')
        except Exception as e:
            print(f' FAILED: {e}')
