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
  05_materials.inp          one *Material per grain (reference UMAT layout by default)
  06_sections.inp           *Solid Section linking 03a ELSETs to 05 materials
  07_interactions.inp       none (as in the reference); extend for cohesive zones etc.
  08_steps_output.inp       uniaxial static step, BCs and output requests (reference)

The materials and the step follow the reference model
upxo_support/collab/T25C.inp, which runs in Abaqus with the target UMAT.

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
from upxo._sup.data_dir import default_data_dir

# ---------------------------------------------------------------------------
# Default output directory: <checkout>/data in a source checkout, otherwise
# ./data under the working directory (see upxo._sup.data_dir).
# ---------------------------------------------------------------------------
SUPPORTED_ELEMENT_TYPES = ('C3D8', 'C3D4')

# Kuhn decomposition of a voxel into 6 tetrahedra sharing the body diagonal
# (i,j,k)->(i+1,j+1,k+1); rows are (di,dj,dk) offsets from the voxel's min
# corner, ordered for a positive C3D4 Jacobian. Same table as
# fm_steel_3d.mesh_exporter_3d._KUHN_TETS (copied: importing that module
# loads the whole fm_steel_3d package). Every voxel uses the same diagonal,
# so the faces of neighbouring voxels' tets match.
_KUHN_TETS = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)),
)
_ELEMS_PER_VOXEL = {'C3D8': 1, 'C3D4': len(_KUHN_TETS)}

# ---------------------------------------------------------------------------
# Materials and load step copied from the reference model that runs in
# Abaqus with the target UMAT (upxo_support/collab/T25C.inp). Constants 5 and
# 6 are fixed at the reference's values; their meaning is set by that UMAT.
# ---------------------------------------------------------------------------
MATERIAL_FORMATS = ('reference_umat', 'bunge_euler', 'orientation')
REFERENCE_UMAT_TAIL = (15.0, 0.0)
REFERENCE_N_DEPVAR = 1
LOAD_AXES = ('x', 'y', 'z')
_FACE_NAMES = {'x': ('XMIN', 'XMAX'), 'y': ('YMIN', 'YMAX'), 'z': ('ZMIN', 'ZMAX')}
_DOF = {'x': 1, 'y': 2, 'z': 3}

# Step controls of the reference model; every one is an exporter setting.
REFERENCE_STEP = dict(
    step_time=80.0, initial_inc=0.01, min_inc=1e-8, max_inc=1.0,
    max_increments=10000, output_interval=0.5,
    element_outputs=('LE', 'NE', 'S', 'SDV'), node_outputs=('U',))


def _as_output_list(value):
    """'LE, S' or ['LE', 'S'] -> ('LE', 'S')."""
    items = value.split(',') if isinstance(value, str) else list(value)
    return tuple(str(v).strip().upper() for v in items if str(v).strip())


def validate_step_controls(step_time, initial_inc, min_inc, max_inc,
                           max_increments, output_interval, element_outputs,
                           node_outputs):
    """Raise ValueError for step controls Abaqus would reject or that cannot
    produce output. Returns the output lists as tuples."""
    if not step_time > 0:
        raise ValueError('step_time must be > 0.')
    if not 0 < min_inc <= initial_inc <= max_inc:
        raise ValueError('increments must satisfy 0 < min_inc <= initial_inc <= max_inc.')
    if max_inc > step_time:
        raise ValueError('max_inc cannot exceed step_time.')
    if int(max_increments) < 1:
        raise ValueError('max_increments must be >= 1.')
    if not 0 < output_interval <= step_time:
        raise ValueError('output_interval must be > 0 and <= step_time.')
    element_outputs = _as_output_list(element_outputs)
    node_outputs = _as_output_list(node_outputs)
    if not element_outputs and not node_outputs:
        raise ValueError('request at least one element or node output.')
    for v in element_outputs + node_outputs:
        if not v.replace('_', '').isalnum():
            raise ValueError(f'output variable {v!r} is not a valid Abaqus name.')
    return element_outputs, node_outputs


def required_bc_faces(load_axis):
    """Face node sets the uniaxial step needs: the loaded face (max face of
    ``load_axis``) and the min face of every axis."""
    return {_FACE_NAMES[load_axis][1]} | {_FACE_NAMES[a][0] for a in LOAD_AXES}


def write_reference_umat_material(f, name, euler_deg, grain_number,
                                  n_depvar=REFERENCE_N_DEPVAR, comment=None):
    """One ``*Material`` in the reference layout: ``*Depvar`` then
    ``*User Material, constants=6`` = phi1, Phi, phi2 (degrees, wrapped into
    [0, 360)), the grain number, then ``REFERENCE_UMAT_TAIL``."""
    phi1, Phi, phi2 = (float(a) % 360.0 for a in euler_deg)
    f.write(f'*Material, name={name}\n')
    if comment:
        f.write(f'** {comment}\n')
    f.write(f'*Depvar\n{int(n_depvar)},\n')
    f.write('*User Material, constants=6\n')
    tail = ', '.join(f'{v:g}.' if float(v).is_integer() else f'{v:g}'
                     for v in REFERENCE_UMAT_TAIL)
    f.write(f'{phi1:.6f}, {Phi:.6f}, {phi2:.6f}, {int(grain_number)}., {tail}\n')


def write_uniaxial_static_step(f, nset_names, load_axis, displacement,
                               step_time=80.0, initial_inc=0.01, min_inc=1e-8,
                               max_inc=1.0, max_increments=10000,
                               output_interval=0.5,
                               element_outputs=('LE', 'NE', 'S', 'SDV'),
                               node_outputs=('U',)):
    """The reference load step: static, nlgeom, the max face of ``load_axis``
    displaced by ``displacement`` along that axis, the min face of every axis
    held in its own direction, and the output requests. The defaults are the
    reference model's (``REFERENCE_STEP``).

    Field output is written every ``output_interval`` of step time, so the
    output database gets step_time / output_interval + 1 frames, and Abaqus
    shortens increments to land on those times.

    ``nset_names`` maps 'XMIN'..'ZMAX' to the node-set names in the model.
    """
    if load_axis not in LOAD_AXES:
        raise ValueError(f'load_axis must be one of {LOAD_AXES}.')
    element_outputs, node_outputs = validate_step_controls(
        step_time, initial_inc, min_inc, max_inc, max_increments,
        output_interval, element_outputs, node_outputs)
    top = nset_names[_FACE_NAMES[load_axis][1]]
    d = _DOF[load_axis]
    f.write('** STEP: Step-1\n**\n')
    f.write(f'*Step, name=Step-1, nlgeom=YES, inc={int(max_increments)}\n')
    f.write('*Static\n')
    f.write(f'{initial_inc:g}, {step_time:g}, {min_inc:g}, {max_inc:g}\n')
    f.write('**\n** BOUNDARY CONDITIONS\n**\n')
    f.write(f'** Name: BC-1 Type: Displacement/Rotation (loaded face, {load_axis})\n')
    f.write(f'*Boundary\n{top}, {d}, {d}, {displacement:.10g}\n')
    for k, axis in enumerate(LOAD_AXES, start=2):
        face = nset_names[_FACE_NAMES[axis][0]]
        dof = _DOF[axis]
        f.write(f'** Name: BC-{k} Type: Displacement/Rotation (min face, {axis})\n')
        f.write(f'*Boundary\n{face}, {dof}, {dof}\n')
    f.write('**\n** OUTPUT REQUESTS\n**\n')
    f.write('*Restart, write, frequency=0\n')
    f.write('**\n** FIELD OUTPUT: F-Output-1\n**\n')
    f.write(f'*Output, field, time interval={output_interval:g}\n')
    if element_outputs:
        f.write('*Element Output, directions=YES\n')
        f.write(', '.join(element_outputs) + '\n')
    if node_outputs:
        f.write('**\n** FIELD OUTPUT: F-Output-2\n**\n')
        f.write('*Node Output\n' + ', '.join(node_outputs) + ',\n')
    f.write('**\n** HISTORY OUTPUT: H-Output-1\n**\n')
    f.write('*Output, history, variable=PRESELECT\n')
    f.write('*End Step\n')

DEFAULT_ABQ_OUT_DIR = os.path.join(default_data_dir(), 'ABQInputFiles', 'ofhcCu')


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
        Abaqus element type, one of ``SUPPORTED_ELEMENT_TYPES``: 'C3D8'
        (default; one linear hex brick per voxel) or 'C3D4' (six linear
        tetrahedra per voxel). Element sets list the elements of every voxel
        they contain, so they hold six times as many ids for C3D4.
    material_format : str
        'reference_umat' (default) → the layout of the reference model
        T25C: *Depvar, then *User Material with 6 constants: Bunge-Euler
        angles in degrees wrapped into [0, 360), a sequential grain number
        1..N, then ``REFERENCE_UMAT_TAIL``.
        'bunge_euler' → *User Material with 3 Bunge-Euler constants.
        'orientation' → *Elastic stub (for elastic studies).
    n_depvar : int
        Number of UMAT state variables (*Depvar); default 1, as in the
        reference. Not used by 'orientation'.
    length_scale : float
        Multiplies every node coordinate. Coordinates are voxel index x
        voxel_size_um x length_scale; the default 1e-3 writes mm.
    load_axis : str
        'x', 'y' or 'z' (default): the axis of the uniaxial load in 08.
    applied_strain : float
        Nominal strain of the load step (default 0.2): the max face of
        ``load_axis`` is displaced by applied_strain x the domain length
        along that axis, in the scaled units.
    step_time : float
        Total time of the static step (default 80, as in the reference).
    write_step : bool
        Write the step in 08 (default True). The face node sets the step
        needs are written even if disabled in ``nset_config``.
    initial_inc, min_inc, max_inc, max_increments : float, float, float, int
        *Static increment controls and the *Step inc= limit (defaults 0.01,
        1e-8, 1.0, 10000, as in the reference).
    output_interval : float
        Field output every this much step time (default 0.5): the output
        database gets step_time / output_interval + 1 frames.
    element_outputs, node_outputs : str or sequence of str
        Field output variables, e.g. 'LE, NE, S, SDV' and 'U' (the defaults).
        The output database grows with the number of variables and frames.
    write_role_elsets : bool
        Write 03b_elsets_roles.inp (Option 2 grouping).
    write_family_elsets : bool
        Write 03c_elsets_families.inp (Option 3 grouping).
    write_variant_elsets : bool
        Write 03d_elsets_variants.inp (Option 5 grouping).
    role_enabled : dict {role_key: bool} or None
        Per-role toggle for whether that role's bucket ELSET is written in
        03b (role_key is one of 'non_host'/'host'/'primary_twin'/
        'stwin_a'/'stwin_b'). Defaults to all-enabled. Has no effect on
        03a (per-grain elsets are always written for every grain,
        regardless of role, since every element must belong to exactly
        one 03a ELSET for material/section assignment) -- this only
        controls the optional 03b convenience groupings.
    role_prefix : dict {role_key: str} or None
        Per-role ELSET name override for 03b (same keys as
        ``role_enabled``). Falls back to ``_DEFAULT_ROLE_PREFIX`` for any
        role not given.
    family_prefix : str
        ELSET name prefix for 03c (``<family_prefix><host_gid>`` -- include
        your own trailing separator, e.g. ``'es_family_'``).
    variant_prefix : str
        ELSET name prefix for 03d (``<variant_prefix><v>`` -- include your
        own trailing separator, e.g. ``'es_variant_ptwin_'``).
    nset_config : dict {face: {'enabled': bool, 'prefix': str}} or None
        Per-face (``'XMIN'``/``'XMAX'``/``'YMIN'``/``'YMAX'``/``'ZMIN'``/
        ``'ZMAX'``) toggle and name override for 04. Falls back to
        ``_DEFAULT_NSET_CONFIG`` (all enabled) for any face not given.
    material_level : str
        One of ``'feature'`` (default -- one *Material per grain, today's
        only implemented behaviour) or a role key. Non-``'feature'``
        values are accepted but NOT implemented: materials/sections are
        still written per-grain, and a warning is raised at ``write()``
        time rather than silently honouring the request.
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

    # Default 03b role-grouping ELSET names -- used whenever role_prefix
    # (constructor arg) doesn't override a given role.
    _DEFAULT_ROLE_PREFIX = {
        'non_host':      'es_role_nonpart',
        'host':          'es_role_host',
        'primary_twin':  'es_role_ptwin',
        'stwin_a':       'es_role_stwin_a',
        'stwin_b':       'es_role_stwin_b',
    }

    # Default 04 boundary-face node-set names/enablement -- used whenever
    # nset_config (constructor arg) doesn't override a given face.
    _DEFAULT_NSET_CONFIG = {
        'XMIN': {'enabled': True, 'prefix': 'ns_face_XMIN'},
        'XMAX': {'enabled': True, 'prefix': 'ns_face_XMAX'},
        'YMIN': {'enabled': True, 'prefix': 'ns_face_YMIN'},
        'YMAX': {'enabled': True, 'prefix': 'ns_face_YMAX'},
        'ZMIN': {'enabled': True, 'prefix': 'ns_face_ZMIN'},
        'ZMAX': {'enabled': True, 'prefix': 'ns_face_ZMAX'},
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
            material_format:      str   = 'reference_umat',
            n_depvar:             int   = REFERENCE_N_DEPVAR,
            length_scale:         float = 1e-3,
            load_axis:            str   = 'z',
            applied_strain:       float = 0.2,
            step_time:            float = 80.0,
            write_step:           bool  = True,
            initial_inc:          float = 0.01,
            min_inc:              float = 1e-8,
            max_inc:              float = 1.0,
            max_increments:       int   = 10000,
            output_interval:      float = 0.5,
            element_outputs              = ('LE', 'NE', 'S', 'SDV'),
            node_outputs                 = ('U',),
            write_role_elsets:    bool  = True,
            write_family_elsets:  bool  = True,
            write_variant_elsets: bool  = True,
            role_enabled:         Optional[Dict[str, bool]] = None,
            role_prefix:          Optional[Dict[str, str]] = None,
            family_prefix:        str   = 'es_family_',
            variant_prefix:       str   = 'es_variant_ptwin_',
            nset_config:          Optional[Dict[str, Dict[str, object]]] = None,
            material_level:       str   = 'feature',
            index:                Optional['GrainElementIndex'] = None,
    ):
        if twin_role is None or twin_parent_of is None or all_quats is None:
            raise ValueError(
                'twin_role, twin_parent_of, and all_quats are all required '
                '(even when index=... is given -- the index only covers '
                'per-grain element membership, these are still used '
                'directly by the write_* methods).')

        if element_type not in SUPPORTED_ELEMENT_TYPES:
            raise ValueError(
                f'element_type={element_type!r} is not supported; '
                f'choose one of {SUPPORTED_ELEMENT_TYPES}.')
        if material_format not in MATERIAL_FORMATS:
            raise ValueError(
                f'material_format={material_format!r} is not supported; '
                f'choose one of {MATERIAL_FORMATS}.')
        if load_axis not in LOAD_AXES:
            raise ValueError(f'load_axis must be one of {LOAD_AXES}.')
        if not length_scale > 0:
            raise ValueError('length_scale must be > 0.')
        element_outputs, node_outputs = validate_step_controls(
            step_time, initial_inc, min_inc, max_inc, max_increments,
            output_interval, element_outputs, node_outputs)

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
        self.role_enabled     = {**{k: True for k in self._DEFAULT_ROLE_PREFIX}, **(role_enabled or {})}
        self.role_prefix      = {**self._DEFAULT_ROLE_PREFIX, **(role_prefix or {})}
        self.family_prefix    = family_prefix
        self.variant_prefix   = variant_prefix
        self.nset_config      = {**self._DEFAULT_NSET_CONFIG, **(nset_config or {})}
        self.material_level   = material_level
        self.length_scale     = float(length_scale)
        self.load_axis        = load_axis
        self.applied_strain   = float(applied_strain)
        self.step_time        = float(step_time)
        self.initial_inc      = float(initial_inc)
        self.min_inc          = float(min_inc)
        self.max_inc          = float(max_inc)
        self.max_increments   = int(max_increments)
        self.output_interval  = float(output_interval)
        self.element_outputs  = element_outputs
        self.node_outputs     = node_outputs
        self.write_step       = bool(write_step)
        if self.write_step:
            # the step's boundary conditions refer to these face node sets
            for face in required_bc_faces(load_axis):
                cfg = dict(self.nset_config.get(face, self._DEFAULT_NSET_CONFIG[face]))
                if not cfg.get('enabled', True):
                    warnings.warn(
                        f"node set {face} is disabled but the load step needs "
                        f"it; writing it anyway.", stacklevel=2)
                    cfg['enabled'] = True
                self.nset_config[face] = cfg

        # Populated by write() -- lets callers report how many ELSETs/NSETs
        # were actually written without re-deriving it from shared_state.
        self.n_role_elsets_written = 0
        self.n_nsets_written = 0

        self.nx, self.ny, self.nz = index.nx, index.ny, index.nz
        self._role_map    = index.role_map
        self._lgi_flat     = self.lgi.ravel(order='C')
        # index.grain_elems holds voxel ids (one per voxel, 1-based). With
        # several elements per voxel, voxel v owns elements
        # k*(v-1)+1 .. k*v. A new dict: the index may be reused for other
        # exports with other settings.
        k = _ELEMS_PER_VOXEL[element_type]
        if k == 1:
            self._grain_elems = index.grain_elems
        else:
            local = np.arange(1, k + 1, dtype=np.int64)
            self._grain_elems = {
                gid: ((vox.astype(np.int64) - 1)[:, None] * k + local).ravel()
                for gid, vox in index.grain_elems.items()}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @property
    def _units(self) -> str:
        names = {1.0: 'microns', 1e-3: 'mm', 1e-6: 'm'}
        return names.get(self.length_scale,
                         f'microns x {self.length_scale:g}')

    @property
    def n_elements(self) -> int:
        """Number of elements written (voxels x elements per voxel)."""
        return self.nx * self.ny * self.nz * _ELEMS_PER_VOXEL[self.element_type]

    def write(self, out_dir: str = DEFAULT_ABQ_OUT_DIR) -> None:
        """
        Write all Abaqus input files to *out_dir*.

        Parameters
        ----------
        out_dir : str, optional
            Destination directory.  Defaults to
            ``DEFAULT_ABQ_OUT_DIR`` (``data/ABQInputFiles/ofhcCu/`` in a source
            checkout, otherwise under ``./data`` in the working directory).
            Pass any absolute or relative path to override.
        """
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.perf_counter()

        if self.material_level != 'feature':
            warnings.warn(
                f"material_level={self.material_level!r} is not implemented -- "
                "materials and *Solid Section (05/06) are still written "
                "Feature Specific (one *Material per grain), matching "
                "material_level='feature'. Aggregated per-level material "
                "association is not currently supported.",
                stacklevel=2,
            )

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
                with open(fpath, 'w') as f:
                    f.write(f'** {fname} skipped (disabled by user flag)\n')
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
        f.write(f'** Node coordinates = voxel index x {self.vox_um:g} um x '
                f'length_scale {self.length_scale:g}  (units: {self._units})\n')
        f.write('*Node\n')
        nx, ny, nz = self.nx, self.ny, self.nz
        vs = self.vox_um * self.length_scale
        node_id = 0
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz + 1):
                    node_id += 1
                    f.write(f'{node_id}, {ix*vs:.10g}, {iy*vs:.10g}, {iz*vs:.10g}\n')

    # -----------------------------------------------------------------------
    # 02 elements
    # -----------------------------------------------------------------------
    def _write_elements(self, f):
        """
        Write the element connectivity.

        C3D8: voxel (ix, iy, iz) in C-order is element
        ix*(ny*nz) + iy*nz + iz + 1, with 8 corner nodes in the Abaqus C3D8
        order.

        C3D4: each voxel is split into 6 tetrahedra (``_KUHN_TETS``); voxel
        v = ix*(ny*nz) + iy*nz + iz owns elements 6*v + 1 .. 6*v + 6.
        """
        if self.element_type == 'C3D4':
            self._write_elements_c3d4(f)
            return
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

    def _write_elements_c3d4(self, f):
        nx, ny, nz = self.nx, self.ny, self.nz
        nn_y, nn_z = ny + 1, nz + 1
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                 indexing='ij')
        ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()     # C-order = voxel order
        n_vox = ix.size
        conn = np.empty((n_vox, len(_KUHN_TETS), 4), dtype=np.int64)
        for t, tet in enumerate(_KUHN_TETS):
            for c, (di, dj, dk) in enumerate(tet):
                conn[:, t, c] = (ix + di) * nn_y * nn_z + (iy + dj) * nn_z + (iz + dk) + 1
        conn = conn.reshape(-1, 4)
        eids = np.arange(1, conn.shape[0] + 1, dtype=np.int64)
        f.write(f'** Element connectivity  type={self.element_type}  '
                f'(6 tetrahedra per voxel)\n')
        f.write(f'*Element, type={self.element_type}\n')
        np.savetxt(f, np.column_stack([eids, conn]), fmt='%d', delimiter=', ')

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
        role_buckets: Dict[str, list] = {
            key: [] for key in self._DEFAULT_ROLE_PREFIX if self.role_enabled.get(key, True)
        }
        for gid, eids in self._grain_elems.items():
            key = self._role_map.get(gid, 'non_host')
            if key in role_buckets:
                role_buckets[key].extend(eids.tolist())
        # Super-groupings -- built only from whichever constituent roles
        # are enabled, so disabling e.g. 'stwin_b' also shrinks es_role_stwin.
        stwin_eids = role_buckets.get('stwin_a', []) + role_buckets.get('stwin_b', [])
        twin_eids = role_buckets.get('primary_twin', []) + stwin_eids
        for key, eids in role_buckets.items():
            if eids:
                name = self.role_prefix.get(key, self._DEFAULT_ROLE_PREFIX[key])
                self._write_elset_block(f, name, np.array(eids, dtype=np.int32))
                self.n_role_elsets_written += 1
        if stwin_eids:
            self._write_elset_block(f, 'es_role_stwin', np.array(stwin_eids, dtype=np.int32))
            self.n_role_elsets_written += 1
        if twin_eids:
            self._write_elset_block(f, 'es_role_twin', np.array(twin_eids, dtype=np.int32))
            self.n_role_elsets_written += 1

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
                    f, f'{self.family_prefix}{host_gid}', np.array(eids, dtype=np.int32))

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
                    f, f'{self.variant_prefix}{v}', np.array(eids, dtype=np.int32))

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
            cfg = self.nset_config.get(name, self._DEFAULT_NSET_CONFIG[name])
            if not cfg.get('enabled', True):
                continue
            nset_name = cfg.get('prefix') or self._DEFAULT_NSET_CONFIG[name]['prefix']
            nodes = np.array([nid(*c) for c in coords], dtype=np.int32)
            f.write(f'*Nset, nset={nset_name}\n')
            self._write_id_list(f, nodes)
            self.n_nsets_written += 1

    # -----------------------------------------------------------------------
    # 05  materials
    # -----------------------------------------------------------------------
    def _write_materials(self, f):
        if self.material_format == 'reference_umat':
            f.write('** One *Material per grain, in the layout of the reference model.\n')
            f.write('** *User Material constants: phi1, Phi, phi2 (Bunge, degrees, [0, 360)),\n')
            f.write('** UMAT grain number (1..N), '
                    + ', '.join(f'{v:g}' for v in REFERENCE_UMAT_TAIL) + '.\n**\n')
            for number, gid in enumerate(self._grain_elems, start=1):
                role_key = self._role_map.get(gid, 'non_host')
                mat_name = f'{self._PREFIX[role_key][1]}_{gid}'
                q = self.all_quats.get(gid, np.array([1., 0., 0., 0.]))
                write_reference_umat_material(
                    f, mat_name, _quat_to_bunge(q), number, self.n_depvar,
                    comment=f'grain {gid} -> UMAT grain number {number}')
            return
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
        f.write('** Interaction definitions.\n')
        f.write('** None: grains share nodes, and the reference model has no\n')
        f.write('** interactions. Add cohesive zones, contact, etc. here.\n')

    # -----------------------------------------------------------------------
    # 08  steps / output (stub)
    # -----------------------------------------------------------------------
    def _write_steps_output(self, f):
        if not self.write_step:
            warnings.warn(
                "08_steps_output.inp contains no *Step: write_step=False -- "
                "add a step before submitting this model.", stacklevel=3)
            f.write('** No step written (write_step=False).\n')
            return
        nset_names = {face: (self.nset_config[face].get('prefix')
                             or self._DEFAULT_NSET_CONFIG[face]['prefix'])
                      for face in self._DEFAULT_NSET_CONFIG}
        n_vox = {'x': self.nx, 'y': self.ny, 'z': self.nz}[self.load_axis]
        length = n_vox * self.vox_um * self.length_scale
        displacement = self.applied_strain * length
        f.write(f'** Uniaxial load along {self.load_axis}: nominal strain '
                f'{self.applied_strain:g} x length {length:g} {self._units} '
                f'= displacement {displacement:g} {self._units}.\n')
        f.write('** Step and output requests as in the reference model.\n**\n')
        write_uniaxial_static_step(
            f, nset_names, self.load_axis, displacement,
            step_time=self.step_time, initial_inc=self.initial_inc,
            min_inc=self.min_inc, max_inc=self.max_inc,
            max_increments=self.max_increments,
            output_interval=self.output_interval,
            element_outputs=self.element_outputs, node_outputs=self.node_outputs)

    # -----------------------------------------------------------------------
    # master file
    # -----------------------------------------------------------------------
    def _write_master(self, out_dir: str, steps) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        n_elem = self.n_elements
        path = os.path.join(out_dir, 'model_master.inp')
        with open(path, 'w') as f:
            f.write('** Made with UPXO -- twinned OFHC Cu 3D microstructure\n**\n')
            f.write('*Heading\n')
            f.write(f'** Twinned OFHC Cu  element type: {self.element_type}\n')
            f.write(f'** Grid: {nx} x {ny} x {nz} voxels  '
                    f'|  Active elements: {n_elem:,}\n')
            f.write(f'** Voxel size: {self.vox_um} microns  |  length scale '
                    f'{self.length_scale:g}  |  units: {self._units}\n**\n')
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
