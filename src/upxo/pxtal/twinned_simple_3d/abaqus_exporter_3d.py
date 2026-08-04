"""
abaqus_exporter_3d.py
=====================
Partitioned Abaqus *.inp* writer for the twinned simple 3D pipeline.

File layout (mirrors FM Steel convention)
-----------------------------------------
model_master.inp
  01_nodes.inp
  02_elements.inp
  03a_elsets_cells.inp   per-grain cell ELSETs  (material assignment, no overlap)
  03b_elsets_roles.inp      role grouping ELSETs   (Option 2, deduplication allowed)
  03c_elsets_families.inp   family ELSETs          (Option 3, user flag)
  03d_elsets_variants.inp   Sigma3 variant ELSETs  (Option 5, user flag)
  04_nsets_bc.inp           boundary-condition node sets (faces of the domain)
  05_materials.inp          one *Material per grain
  06_sections.inp           *Solid Section linking 03a ELSETs to 05 materials
  07_interactions.inp       (stub — extend for cohesive zones etc.)
  08_steps_output.inp       (stub — extend for load steps)

ELSET naming (03a — material assignment, no voxel overlap)
----------------------------------------------------------
  es_cell_nonpart_<gid>     non-participating grains
  es_cell_host_<gid>        host grains (remaining voxels after carving)
  es_cell_ptwin_<gid>       primary twin lamellae
  es_cell_stwin_a_<gid>     secondary twins carved from host (outward / 2a)
  es_cell_stwin_b_<gid>     secondary twins carved from primary twin (inward / 2b)

Material naming: MAT_CELL_NONPART_<gid>, MAT_CELL_HOST_<gid>, etc.
(underscores throughout -- Abaqus keyword-file names do not permit periods)

2a vs 2b distinction
--------------------
  twin_parent_of[gid] is a HOST grain  → secondary is 2a (outward)
  twin_parent_of[gid] is a PRIMARY TWIN → secondary is 2b (inward)
"""

from __future__ import annotations
import os
import time
import math
import warnings
import numpy as np
from typing import Optional, Dict, Set

# ---------------------------------------------------------------------------
# Default output directory — resolved relative to this file so the path is
# portable across machines regardless of installation location.
#
# __file__ : .../upxo_library/src/upxo/pxtal/twinned_simple_3d/abaqus_exporter_3d.py
#              ↑  up 4 levels  ↑                                  upxo_library/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
DEFAULT_ABQ_OUT_DIR = os.path.join(_LIB_ROOT, 'data', 'ABQInputFiles', 'ofhcCu')


# ---------------------------------------------------------------------------
# Bunge-Euler conversion helper
# ---------------------------------------------------------------------------
def _quat_to_bunge(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert unit quaternion [w, x, y, z] to Bunge-Euler angles (phi1, Phi, phi2)
    in degrees.  Uses the ZXZ active-rotation convention.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Rotation matrix from quaternion
    r11 = 1 - 2*(y*y + z*z);  r12 = 2*(x*y - z*w);  r13 = 2*(x*z + y*w)
    r21 = 2*(x*y + z*w);       r22 = 1 - 2*(x*x+z*z); r23 = 2*(y*z - x*w)
    r31 = 2*(x*z - y*w);       r32 = 2*(y*z + x*w);   r33 = 1 - 2*(x*x+y*y)
    # Bunge ZXZ
    sin_Phi = math.sqrt(max(0.0, 1.0 - r33*r33))
    if sin_Phi > 1e-6:
        Phi  = math.acos(max(-1.0, min(1.0, r33)))
        phi1 = math.atan2(r31, -r32)
        phi2 = math.atan2(r13,  r23)
    else:
        Phi  = 0.0 if r33 > 0 else math.pi
        phi1 = math.atan2(-r12, r11)
        phi2 = 0.0
    return (math.degrees(phi1), math.degrees(Phi), math.degrees(phi2))


# ---------------------------------------------------------------------------
# Grain-to-element index -- the expensive, settings-independent part of
# constructing an AbaqusExporter3D, split out so it can be built ONCE and
# reused across repeated exports of the SAME cleaned structure with
# different settings (voxel size, element type, elset toggles, ...)
# instead of re-indexing on every export.
# ---------------------------------------------------------------------------
class GrainElementIndex:
    """
    Pre-built grain-to-element index for :class:`AbaqusExporter3D`.

    Depends only on ``(lgi, twin_role, twin_parent_of)`` -- NOT on any
    export setting -- so build it once via
    :meth:`AbaqusExporter3D.build_index` and pass the result to
    ``AbaqusExporter3D(index=..., ...)`` for every subsequent export of
    the same structure, regardless of what settings change between
    exports.
    """

    __slots__ = ('lgi', 'nx', 'ny', 'nz', 'role_map', 'grain_elems')

    def __init__(self, lgi, nx, ny, nz, role_map, grain_elems):
        self.lgi = lgi
        self.nx, self.ny, self.nz = nx, ny, nz
        self.role_map = role_map
        self.grain_elems = grain_elems


# ---------------------------------------------------------------------------
# Main exporter class
# ---------------------------------------------------------------------------
class AbaqusExporter3D:
    """
    Writes a partitioned Abaqus input model for the twinned 3D structure.

    Parameters
    ----------
    lgi : ndarray (nz, ny, nx) or None
        Cleaned labelled grain image from ``StructureCleaner3D``
        (``cleaner.lgi_clean``), in the twinned_simple_3d pipeline's
        native axis order (axis0=Z, axis2=X -- see
        ``TwinnedSimple3DBase.plot_temporal_slice_3d``'s P/R/C convention
        comment). Transposed internally to (nx, ny, nz) before any node/
        element indexing, so callers should pass ``cleaner.lgi_clean``
        exactly as produced -- do not pre-transpose it yourself. May be
        omitted (None) if ``index`` is given instead (see below).
    twin_role : dict {gid: str}
        ``twin_role_clean`` from cleaner: one of
        'non_host', 'host', 'primary_twin', 'secondary_twin'.
    twin_parent_of : dict {child_gid: parent_gid}
        ``twin_parent_of_clean`` from cleaner.
    all_quats : dict {gid: ndarray(4,)}
        Per-grain unit quaternions [w, x, y, z].
    twinmake : TwinGenerator3D or None
        Provides ``twinmake.twin_halfwidths_vox`` and optionally variant
        index when Option 5 (variant ELSETs) is requested.
    voxel_size_um : float
        Physical voxel edge length in microns for node coordinate output.
    element_type : str
        Abaqus element type string, default 'C3D8' (linear hex brick).
    material_format : str
        'bunge_euler' → *User Material with 3 Bunge-Euler constants.
        'orientation' → *Orientation + *Elastic stub (for elastic studies).
    n_depvar : int
        Number of UMAT state variables (used only for 'bunge_euler' format).
    write_role_elsets : bool
        Write 03b_elsets_roles.inp (Option 2 grouping).
    write_family_elsets : bool
        Write 03c_elsets_families.inp (Option 3 grouping).
    write_variant_elsets : bool
        Write 03d_elsets_variants.inp (Option 5 grouping).
    index : GrainElementIndex or None
        A pre-built index from :meth:`build_index`. When given, the
        expensive grain-to-element indexing pass is skipped entirely
        (``lgi`` is then optional and ignored if also given) -- use this
        to re-export the same cleaned structure with different settings
        without re-indexing each time. When None (default), the index is
        built fresh from ``lgi``/``twin_role``/``twin_parent_of``, exactly
        matching the previous (pre-split) behaviour.
    """

    # ELSET / material prefix constants
    _PREFIX = {
        'non_host':      ('es_cell_nonpart',  'MAT_CELL_NONPART'),
        'host':          ('es_cell_host',      'MAT_CELL_HOST'),
        'primary_twin':  ('es_cell_ptwin',     'MAT_CELL_PTWIN'),
        'stwin_a':       ('es_cell_stwin_a',   'MAT_CELL_STWIN_A'),
        'stwin_b':       ('es_cell_stwin_b',   'MAT_CELL_STWIN_B'),
    }

    @staticmethod
    def build_index(
            lgi: np.ndarray,
            twin_role: Dict[int, str],
            twin_parent_of: Dict[int, int],
            verbose: bool = True,
    ) -> 'GrainElementIndex':
        """
        Build the grain-to-element index from a cleaned labelled
        structure -- the expensive part of constructing an
        ``AbaqusExporter3D`` (transposing to native (nx,ny,nz),
        classifying secondary twins into 2a/2b, and indexing every
        grain's element IDs). Depends only on
        ``(lgi, twin_role, twin_parent_of)``, NOT on any export setting,
        so build it once and reuse the same :class:`GrainElementIndex`
        across repeated exports that only change settings.
        """
        # lgi arrives in the pipeline's native (nz, ny, nx) axis order;
        # transpose once here so every node/element/coordinate calculation
        # can correctly assume (nx, ny, nz), matching this class's
        # documented output geometry (XMAX/YMAX/ZMAX node sets, node
        # coordinates, etc.) -- see the lgi parameter docstring above.
        lgi_t = np.transpose(lgi, (2, 1, 0))
        nx, ny, nz = lgi_t.shape

        # Classify secondary twins into 2a / 2b using twin_parent_of
        _primary_gids = {g for g, r in twin_role.items() if r == 'primary_twin'}
        role_map: Dict[int, str] = {}
        for gid, role in twin_role.items():
            if role == 'secondary_twin':
                parent = twin_parent_of.get(gid)
                if parent in _primary_gids:
                    role_map[gid] = 'stwin_b'   # inward: parent is a primary twin
                else:
                    role_map[gid] = 'stwin_a'   # outward: parent is a host grain
            else:
                role_map[gid] = role  # 'non_host', 'host', 'primary_twin'

        # Build element → grain lookup once (voxel flat index = element ID - 1)
        lgi_flat = lgi_t.ravel(order='C')   # C-order: z varies fastest

        # Build grain → element list (1-indexed element IDs)
        if verbose:
            print('AbaqusExporter3D: indexing grain-to-element map...', end='', flush=True)
        _t = time.perf_counter()
        grain_elems: Dict[int, np.ndarray] = {}
        unique_gids = np.unique(lgi_flat)
        unique_gids = unique_gids[unique_gids > 0]
        for gid in unique_gids:
            grain_elems[int(gid)] = (
                np.where(lgi_flat == gid)[0] + 1).astype(np.int32)
        if verbose:
            print(f'  done ({time.perf_counter()-_t:.1f}s)  '
                  f'{len(grain_elems)} grains')

        return GrainElementIndex(lgi_t, nx, ny, nz, role_map, grain_elems)

    def __init__(
            self,
            lgi:                  Optional[np.ndarray] = None,
            twin_role:            Optional[Dict[int, str]] = None,
            twin_parent_of:       Optional[Dict[int, int]] = None,
            all_quats:            Optional[Dict[int, np.ndarray]] = None,
            twinmake=None,
            voxel_size_um:        float = 1.0,
            element_type:         str   = 'C3D8',
            material_format:      str   = 'bunge_euler',
            n_depvar:             int   = 100,
            write_role_elsets:    bool  = True,
            write_family_elsets:  bool  = True,
            write_variant_elsets: bool  = True,
            index:                Optional['GrainElementIndex'] = None,
    ):
        if twin_role is None or twin_parent_of is None or all_quats is None:
            raise ValueError(
                'twin_role, twin_parent_of, and all_quats are all required '
                '(even when index=... is given -- the index only covers '
                'per-grain element membership, these are still used '
                'directly by the write_* methods).')

        if index is None:
            if lgi is None:
                raise ValueError('Either index=... or lgi=... must be provided.')
            index = self.build_index(lgi, twin_role, twin_parent_of)

        self.lgi              = index.lgi
        self.twin_role        = twin_role
        self.twin_parent_of   = twin_parent_of
        self.all_quats        = all_quats
        self.twinmake         = twinmake
        self.vox_um           = float(voxel_size_um)
        self.element_type     = element_type
        self.material_format  = material_format
        self.n_depvar         = int(n_depvar)
        self.write_roles      = write_role_elsets
        self.write_families   = write_family_elsets
        self.write_variants   = write_variant_elsets

        self.nx, self.ny, self.nz = index.nx, index.ny, index.nz
        self._role_map    = index.role_map
        self._lgi_flat     = self.lgi.ravel(order='C')
        self._grain_elems = index.grain_elems

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def write(self, out_dir: str = DEFAULT_ABQ_OUT_DIR) -> None:
        """
        Write all Abaqus input files to *out_dir*.

        Parameters
        ----------
        out_dir : str, optional
            Destination directory.  Defaults to
            ``DEFAULT_ABQ_OUT_DIR`` (``upxo_library/data/ABQInputFiles/ofhcCu/``
            resolved relative to the package root).
            Pass any absolute or relative path to override.
        """
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.perf_counter()

        steps = [
            ('01_nodes.inp',            self._write_nodes),
            ('02_elements.inp',         self._write_elements),
            ('03a_elsets_cells.inp', self._write_elsets_features),
            ('03b_elsets_roles.inp',    self._write_elsets_roles),
            ('03c_elsets_families.inp', self._write_elsets_families),
            ('03d_elsets_variants.inp', self._write_elsets_variants),
            ('04_nsets_bc.inp',         self._write_nsets_bc),
            ('05_materials.inp',        self._write_materials),
            ('06_sections.inp',         self._write_sections),
            ('07_interactions.inp',     self._write_interactions),
            ('08_steps_output.inp',     self._write_steps_output),
        ]

        for fname, writer in steps:
            skip = False
            if fname == '03b_elsets_roles.inp'    and not self.write_roles:    skip = True
            if fname == '03c_elsets_families.inp' and not self.write_families: skip = True
            if fname == '03d_elsets_variants.inp' and not self.write_variants: skip = True
            fpath = os.path.join(out_dir, fname)
            if skip:
                open(fpath, 'w').write(f'** {fname} skipped (disabled by user flag)\n')
                print(f'  [skipped]  {fname}')
                continue
            t1 = time.perf_counter()
            print(f'  Writing  {fname}...', end='', flush=True)
            with open(fpath, 'w') as f:
                writer(f)
            print(f'  done ({time.perf_counter()-t1:.1f}s)')

        self._write_master(out_dir, steps)
        print(f'AbaqusExporter3D.write: complete  total={time.perf_counter()-t0:.1f}s')
        print(f'  Output: {out_dir}')

    # -----------------------------------------------------------------------
    # 01 nodes
    # -----------------------------------------------------------------------
    def _write_nodes(self, f):
        f.write('** Node coordinates  (units: microns)\n')
        f.write('*Node\n')
        nx, ny, nz, vs = self.nx, self.ny, self.nz, self.vox_um
        node_id = 0
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz + 1):
                    node_id += 1
                    f.write(f'{node_id}, {ix*vs:.6g}, {iy*vs:.6g}, {iz*vs:.6g}\n')

    # -----------------------------------------------------------------------
    # 02 elements
    # -----------------------------------------------------------------------
    def _write_elements(self, f):
        """
        Write C3D8 brick elements.  Each voxel (ix, iy, iz) in C-order maps to
        element  ix*(ny*nz) + iy*nz + iz + 1  with 8 corner nodes ordered
        per the Abaqus C3D8 convention.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        nn_y = ny + 1
        nn_z = nz + 1

        def nid(ix, iy, iz):
            return ix * nn_y * nn_z + iy * nn_z + iz + 1

        f.write(f'** Element connectivity  type={self.element_type}\n')
        f.write(f'*Element, type={self.element_type}\n')
        eid = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    eid += 1
                    n1  = nid(ix,   iy,   iz  )
                    n2  = nid(ix+1, iy,   iz  )
                    n3  = nid(ix+1, iy+1, iz  )
                    n4  = nid(ix,   iy+1, iz  )
                    n5  = nid(ix,   iy,   iz+1)
                    n6  = nid(ix+1, iy,   iz+1)
                    n7  = nid(ix+1, iy+1, iz+1)
                    n8  = nid(ix,   iy+1, iz+1)
                    f.write(f'{eid}, {n1},{n2},{n3},{n4},{n5},{n6},{n7},{n8}\n')

    # -----------------------------------------------------------------------
    # 03a  per-grain feature ELSETs (material assignment, no overlap)
    # -----------------------------------------------------------------------
    def _write_elsets_features(self, f):
        f.write('** Per-grain cell ELSETs -- used for material assignment.\n')
        f.write('** Each element appears in exactly ONE ELSET here.\n')
        f.write('** Naming: es_cell_<role>_<gid>\n**\n')
        for gid, eids in self._grain_elems.items():
            role_key  = self._role_map.get(gid, 'non_host')
            es_prefix = self._PREFIX[role_key][0]
            self._write_elset_block(f, f'{es_prefix}_{gid}', eids)

    # -----------------------------------------------------------------------
    # 03b  role grouping ELSETs (Option 2, deduplication fine)
    # -----------------------------------------------------------------------
    def _write_elsets_roles(self, f):
        f.write('** Role grouping ELSETs (Option 2).\n')
        f.write('** Elements may appear in multiple ELSETs here.\n**\n')
        role_buckets = {
            'es_role_nonpart': [],
            'es_role_host':    [],
            'es_role_ptwin':   [],
            'es_role_stwin_a': [],
            'es_role_stwin_b': [],
        }
        _map = {'non_host': 'es_role_nonpart', 'host': 'es_role_host',
                'primary_twin': 'es_role_ptwin',
                'stwin_a': 'es_role_stwin_a', 'stwin_b': 'es_role_stwin_b'}
        for gid, eids in self._grain_elems.items():
            key = _map[self._role_map.get(gid, 'non_host')]
            role_buckets[key].extend(eids.tolist())
        # Super-groupings
        role_buckets['es_role_stwin'] = (
            role_buckets['es_role_stwin_a'] + role_buckets['es_role_stwin_b'])
        role_buckets['es_role_twin'] = (
            role_buckets['es_role_ptwin'] + role_buckets['es_role_stwin'])
        for name, eids in role_buckets.items():
            if eids:
                self._write_elset_block(f, name, np.array(eids, dtype=np.int32))

    # -----------------------------------------------------------------------
    # 03c  parent-twin family ELSETs (Option 3)
    # -----------------------------------------------------------------------
    def _write_elsets_families(self, f):
        f.write('** Host-twin family ELSETs (Option 3).\n')
        f.write('** es_family_<host_gid> = host voxels + all its twin descendants.\n**\n')
        host_gids = {g for g, r in self.twin_role.items() if r == 'host'}
        # Map every twin to its host ancestor
        def _host_ancestor(gid):
            visited, current = set(), gid
            while current in self.twin_parent_of:
                if current in visited:
                    break
                visited.add(current)
                current = self.twin_parent_of[current]
            return current

        family_elems: Dict[int, list] = {h: [] for h in host_gids}
        for gid, eids in self._grain_elems.items():
            ancestor = _host_ancestor(gid)
            if ancestor in family_elems:
                family_elems[ancestor].extend(eids.tolist())
            else:
                # non-host with no family
                pass
        for host_gid, eids in family_elems.items():
            if eids:
                self._write_elset_block(
                    f, f'es_family_{host_gid}', np.array(eids, dtype=np.int32))

    # -----------------------------------------------------------------------
    # 03d  Sigma3 variant ELSETs (Option 5)
    # -----------------------------------------------------------------------
    def _write_elsets_variants(self, f):
        f.write('** Sigma3 variant ELSETs (Option 5).\n')
        f.write('** es_variant_ptwin_<v>  (v = 0..3 for the 4 FCC {111}<112> variants).\n**\n')
        if self.twinmake is None or not hasattr(self.twinmake, 'twin_halfwidths_vox'):
            f.write('** twinmake not provided -- variant ELSETs cannot be written.\n')
            return
        # twinmake does not persist the real {111} variant index chosen per
        # grain during generation (see twin_generator_3d.py var_idx, which is
        # used locally then discarded) -- only a round-robin assignment is
        # available here, so the es_variant_ptwin_<v> grouping below does NOT
        # reflect the actual crystallographic variant of each twin.
        warnings.warn(
            "es_variant_ptwin_<v> ELSETs use a round-robin placeholder, not "
            "the real Sigma3 {111} variant selected during twin generation "
            "(that index is not currently persisted per grain) -- do not "
            "assign variant-specific material behaviour based on this grouping.",
            stacklevel=2,
        )
        variant_elems: Dict[int, list] = {0: [], 1: [], 2: [], 3: []}
        ptwin_gids = list(self.twinmake.primary_twin_quats.keys())
        for k, gid in enumerate(ptwin_gids):
            v = k % 4   # round-robin placeholder -- see warning above
            if gid in self._grain_elems:
                variant_elems[v].extend(self._grain_elems[gid].tolist())
        for v, eids in variant_elems.items():
            if eids:
                self._write_elset_block(
                    f, f'es_variant_ptwin_{v}', np.array(eids, dtype=np.int32))

    # -----------------------------------------------------------------------
    # 04  boundary-condition node sets
    # -----------------------------------------------------------------------
    def _write_nsets_bc(self, f):
        f.write('** Node sets for boundary conditions (domain faces).\n')
        nx, ny, nz = self.nx, self.ny, self.nz
        nn_y, nn_z = ny + 1, nz + 1
        def nid(ix, iy, iz): return ix * nn_y * nn_z + iy * nn_z + iz + 1
        faces = {
            'XMIN': [(0,   iy, iz) for iy in range(ny+1) for iz in range(nz+1)],
            'XMAX': [(nx,  iy, iz) for iy in range(ny+1) for iz in range(nz+1)],
            'YMIN': [(ix, 0,   iz) for ix in range(nx+1) for iz in range(nz+1)],
            'YMAX': [(ix, ny,  iz) for ix in range(nx+1) for iz in range(nz+1)],
            'ZMIN': [(ix, iy,  0)  for ix in range(nx+1) for iy in range(ny+1)],
            'ZMAX': [(ix, iy, nz)  for ix in range(nx+1) for iy in range(ny+1)],
        }
        for name, coords in faces.items():
            nodes = np.array([nid(*c) for c in coords], dtype=np.int32)
            f.write(f'*Nset, nset=ns_face_{name}\n')
            self._write_id_list(f, nodes)

    # -----------------------------------------------------------------------
    # 05  materials
    # -----------------------------------------------------------------------
    def _write_materials(self, f):
        f.write('** One *Material per grain (Bunge-Euler angles in degrees).\n')
        f.write('** Replace with full CPFEM constitutive block as needed.\n**\n')
        for gid in self._grain_elems:
            role_key = self._role_map.get(gid, 'non_host')
            mat_name = f'{self._PREFIX[role_key][1]}_{gid}'
            q    = self.all_quats.get(gid, np.array([1., 0., 0., 0.]))
            phi1, Phi, phi2 = _quat_to_bunge(q)
            f.write(f'*Material, name={mat_name}\n')
            f.write(f'** Bunge-Euler (deg): phi1={phi1:.4f}, Phi={Phi:.4f}, phi2={phi2:.4f}\n')
            if self.material_format == 'bunge_euler':
                f.write(f'*User Material, constants=3\n')
                f.write(f'{phi1:.6f}, {Phi:.6f}, {phi2:.6f}\n')
                f.write(f'*Depvar\n{self.n_depvar}\n')
            else:  # 'orientation' stub
                f.write(f'*Elastic\n210000., 0.3\n')
            f.write('**\n')

    # -----------------------------------------------------------------------
    # 06  sections
    # -----------------------------------------------------------------------
    def _write_sections(self, f):
        f.write('** *Solid Section -- links per-grain ELSETs (03a) to materials (05).\n**\n')
        for gid in self._grain_elems:
            role_key = self._role_map.get(gid, 'non_host')
            es_name  = f'{self._PREFIX[role_key][0]}_{gid}'
            mat_name = f'{self._PREFIX[role_key][1]}_{gid}'
            f.write(f'*Solid Section, elset={es_name}, material={mat_name}\n,\n')

    # -----------------------------------------------------------------------
    # 07  interactions (stub)
    # -----------------------------------------------------------------------
    def _write_interactions(self, f):
        warnings.warn(
            "07_interactions.inp contains no interaction/constraint definitions "
            "(cohesive zone, contact, etc.) -- this .inp is not simulation-ready "
            "as exported and must be edited before submission.",
            stacklevel=3,
        )
        f.write('** Interaction definitions (stub).\n')
        f.write('** Extend for cohesive zone models, contact, etc.\n')

    # -----------------------------------------------------------------------
    # 08  steps / output (stub)
    # -----------------------------------------------------------------------
    def _write_steps_output(self, f):
        warnings.warn(
            "08_steps_output.inp contains no *Step/boundary-condition/*Output "
            "definitions -- this .inp is not simulation-ready as exported and "
            "must be edited before submission.",
            stacklevel=3,
        )
        f.write('** Step and output request definitions (stub).\n')
        f.write('** Define *Step, *Static / *Dynamic, *Output as required.\n')

    # -----------------------------------------------------------------------
    # master file
    # -----------------------------------------------------------------------
    def _write_master(self, out_dir: str, steps) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = nx * ny * nz
        path = os.path.join(out_dir, 'model_master.inp')
        with open(path, 'w') as f:
            f.write('** Made with UPXO -- twinned OFHC Cu 3D microstructure\n**\n')
            f.write('*Heading\n')
            f.write(f'** Twinned OFHC Cu  element type: {self.element_type}\n')
            f.write(f'** Grid: {nx} x {ny} x {nz} voxels  '
                    f'|  Active elements: {n_elem:,}\n')
            f.write(f'** Voxel size: {self.vox_um} microns\n**\n')
            f.write('*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n**\n')
            for fname, _ in steps:
                f.write(f'*INCLUDE, INPUT={fname}\n')

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _write_elset_block(f, name: str, eids: np.ndarray, per_line: int = 16):
        f.write(f'*Elset, elset={name}\n')
        AbaqusExporter3D._write_id_list(f, eids, per_line)

    @staticmethod
    def _write_id_list(f, ids: np.ndarray, per_line: int = 16):
        ids = ids.ravel()
        for start in range(0, len(ids), per_line):
            chunk = ids[start:start + per_line]
            f.write(', '.join(str(v) for v in chunk) + '\n')
