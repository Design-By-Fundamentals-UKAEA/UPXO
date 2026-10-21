"""Settings of the conformal meshing pipeline: names, types, limits, defaults and versions.

Every setting is described once, in :data:`SETTINGS`. The defaults, the type and range
checks, the stages a setting influences and the migration of settings saved by an
older version all derive from that table. Names keep the notebook spelling.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

CONFIG_VERSION = 2


class SettingsError(ValueError):
    """A setting name or value that cannot be used."""


@dataclass(frozen=True)
class Setting:
    """One setting: its default, type, limits and the stages whose results it decides."""

    key: str
    default: Any
    kind: str = ""
    """bool | int | float | text | choice | int_choice | sequence | ids | any (derived from the default when empty)."""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple = ()
    auto: bool = False
    """``None`` is a valid value and means "decide automatically"."""
    size: int = 0
    """Length of a ``sequence``."""
    container: str = "tuple"
    """``tuple`` or ``list``: what a ``sequence`` is stored as."""
    integral: bool = False
    """A ``sequence`` keeps whole numbers as ``int``."""
    stages: tuple = ()
    """Stages whose results depend on this setting."""
    unit: str = ""
    """What the number measures, so a physical size can be shown beside it: ``length`` (voxel edge
    lengths), ``factor`` (a multiple of the voxel edge length), ``voxels`` (a count of voxels) or
    ``thickness`` (a number of voxels across)."""
    signed_when_changed: bool = False
    """Left out of the stages' saved-results signature while it holds its default, so that adding the setting
    does not make results saved before it existed stale; changing it from the default still does."""

    def __post_init__(self):
        if not self.kind:
            kind = ("bool" if isinstance(self.default, bool) else
                    "int" if isinstance(self.default, int) else
                    "float" if isinstance(self.default, float) else
                    "text" if isinstance(self.default, str) else "any")
            object.__setattr__(self, "kind", kind)


def S(key: str, default: Any, **fields) -> Setting:
    return Setting(key, default, **fields)


# Settings that only steer the run, never a stage's results, and are not part of the table.
INTERNAL_DEFAULTS: dict[str, Any] = {
    "_RUN_FOLDER": None,                # indexed run folder of a subset run
    "_OPEN_FOLDER": None,              # run folder a state was opened from (overrides the above)
}

SETTINGS: tuple[Setting, ...] = (
    # Run
    S('RUN_NAME', 'conformal_run'),
    S('BACKEND', 'auto', kind='choice', choices=('auto', 'numpy', 'parallel', 'numba')),
    S('N_WORKERS', None, kind='int', auto=True, minimum=1),
    S('CHECKPOINT_RESULTS', True),
    S('CHECKPOINT_KEEP_HISTORY', True),     # off: a newer result replaces the older copy (less disk)
    S('WRITE_STAGE_TET_FILES', False),      # on: every tet stage also writes verified_tets.vtu and .json to the run folder

    # Optional stage switches (True = run the stage when it is reached)
    S('STAGE_SMALL_GRAINS', True),
    S('STAGE_SPIKES', True),
    S('STAGE_PINCHES', True),
    S('STAGE_VOXEL_TOPOLOGY', True),
    S('STAGE_FREEZE_THIN', True),
    S('STAGE_UPSCALE', True),
    S('STAGE_GUARD_RELAX', True),
    S('STAGE_UNPINCH', True),
    S('STAGE_PRE_JOINT_QUALITY', True),
    S('STAGE_CONTACT_REPAIR', True),
    S('STAGE_CROSSING_REPAIR', True),
    S('STAGE_SURFACE_QUALITY_REPAIR', True),
    S('STAGE_SURFACE_ANGLE_REPAIR', True),
    S('STAGE_TET_SMOOTH', True),
    S('STAGE_TET_SWAPS', True),
    S('STAGE_SMOOTH_AFTER_SWAPS', True),
    S('STAGE_DIHEDRAL_GATE', True),
    S('STAGE_INSERT_NODES', True),
    S('STAGE_GRAIN_ID_SHUFFLE', True),
    S('STAGE_COMPARISON', True),
    S('STAGE_REGION_QUALITY', True),
    S('STAGE_SUBSET', False),

    # Input: an FM-steel raw export folder, or one labelled image (.npy)
    S('DATASET_TYPE', 'fm_steel', kind='choice', choices=('fm_steel', 'labels', 'twinned_fcc'), stages=('load_lgi',)),
    S('TWINNED_FOLDER', '', stages=('load_lgi',)),
    S('TWINNED_VOXEL_SIZE', 1.0, minimum=1e-12, maximum=1e12, stages=('load_lgi',)),
    S('TWINNED_UNITS', 'microns', kind='choice', choices=('microns', 'mm', 'm'), stages=('load_lgi',)),
    S('RAW_FOLDER', '', stages=('load_lgi',)),
    S('HIERARCHY_LEVEL', '', kind='choice', choices=('', 'pag', 'pck', 'blk', 'sblk'), stages=('load_lgi',)),
    S('SAMPLE_PATH', '', stages=('load_lgi',)),
    S('SPACING', (1.0, 1.0, 1.0), kind='sequence', size=3, stages=('load_lgi',)),

    # Subset extract: a cuboid of the input dataset, taken before cleaning
    S('SUBSET_MODE', 'index', kind='choice', choices=('index', 'percent', 'physical'), stages=('subset_extract',)),
    S('SUBSET_LO', [0, 0, 0], kind='sequence', size=3, container='list', integral=True, stages=('subset_extract',)),
    S('SUBSET_HI', None, kind='sequence', size=3, container='list', auto=True, stages=('subset_extract',)),
    S('RVE_MIN_PAGS', None, kind='int', auto=True, minimum=1, maximum=10**9),
    S('RVE_MIN_PACKETS', None, kind='int', auto=True, minimum=1, maximum=10**9),
    S('RVE_MIN_BLOCKS', None, kind='int', auto=True, minimum=1, maximum=10**9),
    S('SUBSET_UNITS', 'same as source', stages=('subset_extract',)),
    S('SUBSET_GRID', False),

    # Grain id mapping
    S('GRAIN_ID_SEED', 0, stages=('grain_id_map',)),

    # Small grains
    S('REMOVE_SMALL_GRAINS', True, stages=('small_grains',)),
    S('MIN_GRAIN_VOXELS', 4, minimum=1, maximum=1e9, stages=('small_grains',), unit='voxels'),
    S('PROTECTED_GRAIN_IDS', (), kind='ids', stages=('small_grains',)),
    S('PROTECT_RVE_GRAINS', False, stages=('small_grains',)),
    S('REMOVAL_MAX_PASSES', 20, minimum=1, maximum=1e6, stages=('small_grains',)),
    S('RECIPIENT_SELECTION', 'shared_area', kind='choice', choices=('shared_area', 'largest'), stages=('small_grains',)),

    # Spikes
    S('CLEAN_SPIKES', True, stages=('spikes',)),
    S('SPIKE_MAX_PASSES', 100, minimum=1, maximum=1e6, stages=('spikes',)),

    # Pinches
    S('CLEAN_PINCHES', True, stages=('pinches',)),
    S('PINCH_MAX_PASSES', 100, minimum=0, maximum=1e6, stages=('pinches',)),
    S('PINCH_MAX_CHANGES', 200, minimum=0, maximum=1e9, stages=('pinches',)),
    S('PINCH_MAX_VOLUME_FRACTION', 0.03, minimum=0, maximum=1, stages=('pinches',)),
    S('PINCH_PROTECT_GRAINS_UP_TO', 8, minimum=0, maximum=1e9, stages=('pinches',), unit='voxels'),

    # Voxel topology
    S('CLEAN_VOXEL_TOPOLOGY', True, stages=('voxel_topology',)),
    S('TOPOLOGY_MAX_PASSES', 20, minimum=0, maximum=1e6, stages=('voxel_topology',)),
    S('TOPOLOGY_MAX_CHANGES', 5000, minimum=0, maximum=1e9, stages=('voxel_topology',)),
    S('TOPOLOGY_MAX_VOLUME_FRACTION', 0.2, minimum=0, maximum=1, stages=('voxel_topology',)),
    S('TOPOLOGY_PROTECT_GRAINS_UP_TO', 0, minimum=0, maximum=1e9, stages=('voxel_topology',), unit='voxels'),
    S('TOPOLOGY_PRESERVE_COMPONENTS', False, stages=('voxel_topology',)),
    S('TOPOLOGY_MAX_PAIR_TRIALS', 20000, minimum=0, maximum=1e9, stages=('voxel_topology',)),
    S('REQUIRE_CLEAN_TOPOLOGY', False, stages=('voxel_topology',)),

    # Thin grains
    S('FREEZE_THIN_GRAINS', True, stages=('freeze_thin',)),
    S('AUTO_FREEZE_THIN_GRAINS', True, stages=('freeze_thin',)),
    S('THIN_CRITERION', 'inscribed', kind='choice', choices=('inscribed', 'axial_run'), stages=('freeze_thin',)),
    S('THIN_THICKNESS_VOXELS', 1, minimum=1, maximum=100, stages=('freeze_thin',), unit='thickness'),
    S('EXPLICIT_FROZEN_GRAIN_IDS', (), kind='ids', stages=('freeze_thin',)),

    # Upscale
    S('UPSCALE_FACTOR', 1, minimum=1, maximum=3, stages=('upscale',)),
    S('EFFECTIVE_SPACING', None, kind='sequence', size=3, auto=True),     # set by the upscale stage

    # Interface smoothing
    S('LAPLACE_ITERATIONS', 50, minimum=0, maximum=1e6, stages=('smooth_interfaces',)),
    S('LAPLACE_RELAXATION', 0.75, minimum=1e-06, maximum=1, stages=('smooth_interfaces',)),
    S('JUNCTION_ITERATIONS', 10, minimum=0, maximum=1e6, stages=('smooth_interfaces',)),
    S('JUNCTION_RELAXATION', 0.35, minimum=1e-06, maximum=1, stages=('smooth_interfaces',)),
    S('JUNCTION_MAX_DISPLACEMENT', None, kind='float', minimum=0, maximum=1e6, auto=True, stages=('smooth_interfaces',), unit='length'),
    S('THICKNESS_CAP_FRACTION', 0.5, minimum=0, maximum=1, stages=('smooth_interfaces',)),
    S('SMOOTH_MIN_WEDGE_DEGREES', 30.0, minimum=0, maximum=180, stages=('smooth_interfaces',)),
    S('SMOOTH_MIN_CORNER_DEGREES', 20.0, minimum=0, maximum=180, stages=('smooth_interfaces',)),
    S('SMOOTH_ITERATIONS_BY_PAIR', None, kind='any', auto=True, stages=('smooth_interfaces',)),
    S('SMOOTH_RAMP_RINGS', 2, minimum=0, maximum=100, stages=('smooth_interfaces',)),

    # Geometry guard
    S('GUARD_SMOOTHED_GEOMETRY', True, stages=('geometry_guard',)),
    S('MIN_FACET_OPENING_DEGREES', 15.0, minimum=0, maximum=180, stages=('geometry_guard', 'pre_tet_validation')),
    S('GEOMETRY_BOUNDARY_CLEARANCE', None, kind='float', minimum=0, auto=True, stages=('geometry_guard',), unit='length'),
    S('GEOMETRY_MAX_PASSES', 100, minimum=1, maximum=1e6, stages=('geometry_guard',)),
    S('GEOMETRY_ROLLBACK_RELAXATION', 0.5, minimum=1e-06, maximum=0.999, stages=('geometry_guard',)),
    S('GEOMETRY_MAX_CORRECTION', None, kind='float', minimum=0, auto=True, stages=('geometry_guard',), unit='length'),
    S('RELAX_GUARD_ROLLBACK', True, stages=('guard_relax',)),
    S('GUARD_RELAX_ITERATIONS', 3, minimum=0, maximum=1e6, stages=('guard_relax',)),
    S('GUARD_RELAX_MARGIN_RINGS', 2, minimum=0, maximum=100, stages=('guard_relax',)),

    # Point-contact separation
    S('SEPARATE_PINCHES', True, stages=('unpinch',)),
    S('PINCH_PRESS_RADIUS', None, kind='float', minimum=0, auto=True, stages=('unpinch',), unit='length'),
    S('PINCH_PRESS_MAX_DISPLACEMENT', None, kind='float', minimum=0, auto=True, stages=('unpinch',), unit='length'),
    S('PINCH_PRESS_ITERATIONS', 30, minimum=0, maximum=1e6, stages=('unpinch',)),
    S('PINCH_PRESS_RELAXATION', 0.25, minimum=0, maximum=1, stages=('unpinch',)),
    S('PINCH_MIN_OPENING', None, kind='float', minimum=0, auto=True, stages=('unpinch',), unit='length'),

    # Gmsh interface remesh
    S('GMSH_SIZE', None, kind='float', minimum=1e-12, auto=True, stages=('gmsh_interfaces',), unit='length'),
    S('GMSH_SIZE_FACTOR', 0.75, minimum=1e-06, maximum=1e6, stages=('gmsh_interfaces',), unit='factor'),
    S('GMSH_ALGORITHM', 6, kind='int_choice', choices=(1, 2, 5, 6), stages=('gmsh_interfaces',)),
    S('GMSH_CHECK_INTERSECTIONS', True, stages=('gmsh_interfaces',)),
    S('GMSH_INTERSECTION_RETRIES', 4, minimum=0, maximum=100, stages=('gmsh_interfaces',)),
    S('GMSH_CHART_TRIANGLES', 128, minimum=1, maximum=1e9, auto=True, stages=('gmsh_interfaces',)),
    S('GMSH_VERBOSE', False, stages=('gmsh_interfaces',)),
    S('GMSH_MIN_REMESH_OPENING', 30.0, minimum=0, maximum=180, auto=True, stages=('gmsh_interfaces',)),
    S('GMSH_RETRY_FAILED_ONLY', False, stages=('gmsh_interfaces', 'joint_remesh'), signed_when_changed=True),      # on: a retry pass meshes only the charts that failed or changed; the others keep their mesh

    # RVE faces and joint remesh
    S('CAP_MESH_SIZE', None, kind='float', minimum=1e-12, auto=True, stages=('close_rve',), unit='length'),
    S('RVE_EDGE_SIZE', None, kind='float', minimum=1e-12, auto=True, stages=('close_rve',), unit='length'),
    S('PRE_JOINT_QUALITY_REPAIR', True, stages=('pre_joint_quality',)),
    S('PRE_JOINT_QUALITY_TARGET', 0.05, minimum=0, maximum=1, stages=('pre_joint_quality',)),
    S('PRE_JOINT_QUALITY_MAX_PASSES', 10, minimum=0, maximum=1e6, stages=('pre_joint_quality',)),
    S('FINAL_GMSH_SIZE', None, kind='float', minimum=1e-12, auto=True, stages=('joint_remesh',), unit='length'),
    S('FINAL_GMSH_ALGORITHM', 6, kind='int_choice', choices=(1, 2, 5, 6), stages=('joint_remesh',)),

    # Surface validation and repair
    S('MIN_SURFACE_TRIANGLE_QUALITY', 0.05, minimum=0, maximum=1, stages=('pre_tet_validation', 'surface_gate')),
    S('STOP_ON_VALIDATION_ERRORS', False, stages=('pre_tet_validation',)),
    S('REPAIR_CONTACTS', True, stages=('contact_repair',)),
    S('REPAIR_SEPARATION', None, kind='float', minimum=0, auto=True, stages=('contact_repair',), unit='length'),
    S('REPAIR_SEPARATION_FACTOR', 0.15, minimum=0, maximum=10, stages=('contact_repair',), unit='factor'),
    S('REPAIR_INTERFACE_CROSSINGS', True, stages=('crossing_repair',)),
    S('CROSSING_MAX_DISPLACEMENT', None, kind='float', minimum=0, auto=True, stages=('crossing_repair',), unit='length'),
    S('CROSSING_NEIGHBOURHOOD_RADIUS', None, kind='float', minimum=0, auto=True, stages=('crossing_repair',), unit='length'),
    S('CROSSING_OPENING', 0.1, minimum=0, maximum=10, stages=('crossing_repair',)),
    S('REPAIR_SURFACE_QUALITY', True, stages=('surface_repair',)),
    S('SURFACE_QUALITY_TARGET', None, kind='float', minimum=0, maximum=1, auto=True, stages=('surface_repair',)),
    S('SURFACE_QUALITY_MAX_PASSES', 10, minimum=0, maximum=1e6, stages=('surface_repair',)),
    S('REPAIR_SURFACE_ANGLES', True, stages=('surface_repair',)),
    S('SURFACE_MIN_ANGLE_DEGREES', 30.0, minimum=0, maximum=60, stages=('surface_repair', 'surface_gate')),
    S('SURFACE_ANGLE_MAX_PASSES', 12, minimum=0, maximum=1e6, stages=('surface_repair',)),
    S('SURFACE_ANGLE_MAX_DEVIATION', None, kind='float', minimum=0, auto=True, stages=('surface_repair',), unit='length'),
    S('APPROVE_TET_MESHING_WITH_POOR_SURFACE', False, stages=('surface_gate',)),

    # Tetrahedra
    S('TET_MESH_SIZE', None, kind='float', minimum=1e-12, auto=True, stages=('tet_mesh',), unit='length'),
    S('TET_MESH_SIZE_FACTOR', 1.5, minimum=1e-06, maximum=1e6, stages=('tet_mesh',), unit='factor'),
    S('TET_OPTIMIZE_NETGEN', True, stages=('tet_mesh',)),
    S('MIN_TET_QUALITY', 0.05, minimum=0, maximum=1, stages=('tet_mesh',)),
    S('TET_NETGEN_ALL_VOLUMES', True, stages=('tet_mesh',)),
    S('TET_OPTIMIZE_THRESHOLD', 0.5, minimum=0, maximum=1, auto=True, stages=('tet_mesh',)),
    S('MIN_TET_DIHEDRAL_DEGREES', 15.0, minimum=0, maximum=70, stages=('tet_mesh', 'dihedral_gate', 'insert_nodes')),
    S('SMOOTH_TET_DIHEDRALS', True, stages=('tet_smooth', 'tet_smooth_after')),
    S('TET_TARGET_MIN_DIHEDRAL', 30.0, minimum=0, maximum=70, stages=('tet_smooth', 'tet_swaps')),
    S('TET_MAX_DIHEDRAL', 150.0, minimum=90, maximum=180, stages=('tet_smooth', 'tet_swaps', 'dihedral_gate', 'insert_nodes')),
    S('TET_NEIGHBOUR_FLOOR', 20.0, minimum=0, maximum=70, stages=('tet_smooth',)),
    S('TET_SMOOTHING_PASSES', 10, minimum=0, maximum=1e6, stages=('tet_smooth', 'tet_smooth_after', 'dihedral_gate')),
    S('MOVE_SURFACE_NODES', True, stages=('tet_smooth', 'dihedral_gate')),
    S('SURFACE_TRIANGLE_TARGET', 30.0, minimum=0, maximum=60, stages=('tet_smooth',)),
    S('SURFACE_TRIANGLE_FLOOR', 20.0, minimum=0, maximum=60, stages=('tet_smooth',)),
    S('SURFACE_MOVE_MAX_DEVIATION', None, kind='float', minimum=0, auto=True, stages=('tet_smooth', 'tet_smooth_after', 'dihedral_gate'), unit='length'),
    S('SWAP_TETS', True, stages=('tet_swaps',)),
    S('TET_SWAP_PASSES', 5, minimum=0, maximum=1e6, stages=('tet_swaps', 'dihedral_gate')),
    S('SMOOTH_AFTER_SWAPS', True, stages=('tet_smooth_after',)),
    S('GATE_PASS', True, stages=('dihedral_gate',)),
    S('JUNCTION_POINT_MAX_DEVIATION', None, kind='float', minimum=0, auto=True, stages=('dihedral_gate',), unit='length'),
    S('GATE_TRIANGLE_TARGET', 20.0, minimum=0, maximum=60, stages=('dihedral_gate',)),
    S('GATE_TRIANGLE_FLOOR', 15.0, minimum=0, maximum=60, stages=('dihedral_gate',)),
    S('INSERT_NODES', True, stages=('insert_nodes',)),

    # Quality report
    S('QUALITY_WORST_FRACTION', 0.05, minimum=0.001, maximum=1),

    # Quality by region: which places, how many layers of elements out from them, what to leave out
    S('REGION_GB_SURFACE', True, stages=('region_quality',)),
    S('REGION_JUNCTION_LINES', True, stages=('region_quality',)),
    S('REGION_JUNCTION_POINTS', True, stages=('region_quality',)),
    S('REGION_ORDER', 3, minimum=1, maximum=50, stages=('region_quality',)),
    S('REGION_ORDER_MODE', 'layers 1 to N', kind='choice', choices=('layers 1 to N', 'only layer N'),
      stages=('region_quality',)),
    S('REGION_TOUCH', 'face or edge', kind='choice', choices=('face or edge', 'node'), stages=('region_quality',)),
    S('REGION_INCLUDE_RVE_FACES', False, stages=('region_quality',)),
    S('REGION_EXCLUDE_LINES', False, stages=('region_quality',)),
    S('REGION_EXCLUDE_POINTS', False, stages=('region_quality',)),
    S('REGION_EXCLUDE_ORDER', 1, minimum=1, maximum=50, stages=('region_quality',)),
    S('REGION_WRITE_FILES', True, stages=('region_quality',)),

    # Before/after comparison: what is compared, and how it is drawn
    S('COMPARE_INCLUDE_RVE_FACES', False, stages=('compare_before_after',)),
    S('FIDELITY_SHRINK_LIMIT', 0.10, minimum=0, maximum=1),    # flag features that lost more volume than this
    S('COMPARE_BEFORE', 'Voxel structure', kind='choice', choices=(
        'Voxel structure', 'Surface before smoothing', 'Surface after smoothing',
        'Remeshed surface before tet meshing', 'Tetrahedral mesh')),
    S('COMPARE_AFTER', 'Tetrahedral mesh', kind='choice', choices=(
        'Voxel structure', 'Surface before smoothing', 'Surface after smoothing',
        'Remeshed surface before tet meshing', 'Tetrahedral mesh')),
    S('COMPARE_PLOT_KIND', 'histogram', kind='choice',
      choices=('histogram', 'kde', 'histogram + kde', 'cdf', 'box', 'violin')),
    S('COMPARE_PLOT_UNITS', 'auto', kind='choice', choices=('auto', 'microns', 'mm', 'm')),
    S('COMPARE_BINS', 40, minimum=2, maximum=1000),
    S('COMPARE_BIN_WIDTH', None, kind='float', minimum=0, auto=True),
    S('COMPARE_KDE_BANDWIDTH', None, kind='float', minimum=0, auto=True),
    S('COMPARE_LOG', False),
    S('COMPARE_DENSITY', False),
    S('COMPARE_MATCHED_KIND', 'scatter', kind='choice', choices=('scatter', 'ratio')),
    S('COMPARE_COLOR_BEFORE', '#4878CF'),
    S('COMPARE_COLOR_AFTER', '#D65F5F'),
    S('COMPARE_SHOW_MEAN', True),
    S('COMPARE_SHOW_STD', True),
    S('COMPARE_SHOW_MEDIAN', False),
    S('COMPARE_MISORIENTATION_NORM', 'fraction of the boundary area', kind='choice',
      choices=('fraction of the boundary area', 'area per unit volume', 'area')),
    S('COMPARE_GRAIN_WEIGHT', 'number of grains', kind='choice', choices=('number of grains', 'grain volume')),

    # Pole figures of the comparison
    S('TEXTURE_POLE', '100', kind='choice', choices=('100', '110', '111', 'custom')),
    S('TEXTURE_HKL', '1 2 3'),
    S('TEXTURE_CRYSTAL_SYMMETRY', 'cubic', kind='choice', choices=('cubic', 'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic', 'triclinic'),
      stages=('compare_before_after', 'export_abaqus')),
    S('TEXTURE_SAMPLE_SYMMETRY', True),
    S('TEXTURE_USE_RD', True),
    S('TEXTURE_USE_TD', True),
    S('TEXTURE_USE_ND', True),
    S('TEXTURE_HALF_WIDTH', 7.5, minimum=0.5, maximum=45),
    S('TEXTURE_GRID', 'auto'),
    S('TEXTURE_UNIT_NORMALIZE', True),
    S('TEXTURE_MODE', 'density', kind='choice', choices=('density', 'scatter', 'hybrid')),
    S('TEXTURE_PROJECTION', 'equal_area', kind='choice', choices=('equal_area', 'stereographic')),
    S('TEXTURE_HEMISPHERE', 'upper', kind='choice', choices=('upper', 'lower', 'both_mapped', 'both_separate')),
    S('TEXTURE_CMAP', 'viridis'),
    S('TEXTURE_DIFF_CMAP', 'RdBu_r'),
    S('TEXTURE_LEVELS', 10, minimum=2, maximum=60),
    S('TEXTURE_SCALE', 'linear', kind='choice', choices=('linear', 'log', 'sqrt')),

    # Abaqus export
    S('EXPORT_FM_DECK', True, stages=('export_abaqus',)),
    S('EXPORT_MESH_ONLY', False, stages=('export_abaqus',)),
    S('EXPORT_BOUNDARY_SETS', True, stages=('export_abaqus',)),
    S('EXPORT_MSH', True, stages=('export_abaqus',)),
    S('EXPORT_VTU', True, stages=('export_abaqus',)),
    S('EXPORT_BOUNDARY_LAYER', True, stages=('export_abaqus',)),
    S('EXPORT_BOUNDARY_CLASSES', True, stages=('export_abaqus',)),
    S('BOUNDARY_LOW_ANGLE_DEGREES', 15.0, minimum=0, maximum=90, stages=('export_abaqus',)),
    S('EXPORT_FEATURE_TABLE', True, stages=('export_abaqus',)),
    S('EXPORT_DECK_CHECK', True, stages=('export_abaqus',)),
    S('EXPORT_PROVENANCE', True, stages=('export_abaqus',)),
    S('JOB_SCRIPTS', True, stages=('export_abaqus',)),
    S('ABAQUS_COMMAND', 'abaqus', stages=('export_abaqus',)),
    S('UMAT_FILE', '', stages=('export_abaqus',)),
    S('JOB_NAME', 'fm_job', stages=('export_abaqus',)),
    S('JOB_CPUS', 4, minimum=1, maximum=4096, stages=('export_abaqus',)),
    S('JOB_MEMORY_GB', 0.0, minimum=0, maximum=1e6, stages=('export_abaqus',)),
    S('JOB_TIME_HOURS', 24.0, minimum=0.001, maximum=1e5, stages=('export_abaqus',)),
    S('SLURM_PARTITION', '', stages=('export_abaqus',)),
    S('REQUIRE_DECK_CHECK', False, stages=('export_abaqus',)),
    S('IDENTITY_RULE', 'larger', kind='choice', choices=('larger', 'survivor'), stages=('export_abaqus', 'compare_before_after')),
    S('IDENTITY_SEED', 0, minimum=0, maximum=2147483647, stages=('export_abaqus', 'compare_before_after')),
    S('FM_MATERIAL_FORMAT', 'reference_umat', kind='choice', choices=('reference_umat', 'bunge_euler', 'orientation'), stages=('export_abaqus',)),
    S('FM_N_DEPVAR', 1, minimum=1, maximum=10000, stages=('export_abaqus',)),
    S('FM_LOAD_AXIS', 'z', kind='choice', choices=('x', 'y', 'z'), stages=('export_abaqus',)),
    S('FM_LOAD_MODE', 'displacement', kind='choice', choices=('displacement', 'strain'), stages=('export_abaqus',)),
    S('FM_LOAD_CASE', 'monotonic', kind='choice', choices=('monotonic', 'cyclic'), stages=('export_abaqus',)),
    S('FM_LOAD_SENSE', 'tension', kind='choice', choices=('tension', 'compression'), stages=('export_abaqus',)),
    S('FM_TARGET_STRAIN', 0.02, minimum=1e-09, maximum=10, stages=('export_abaqus',)),
    S('FM_STRAIN_RATE', 0.0004, minimum=1e-12, maximum=1e9, stages=('export_abaqus',)),
    S('FM_HOLD_TIME', 0.0, minimum=0, maximum=1e9, stages=('export_abaqus',)),
    S('FM_CYCLE_DEFINITION', 'min_max', kind='choice', choices=('min_max', 'amplitude_r'), stages=('export_abaqus',)),
    S('FM_STRAIN_MIN', -0.01, minimum=-10, maximum=10, stages=('export_abaqus',)),
    S('FM_STRAIN_MAX', 0.01, minimum=-10, maximum=10, stages=('export_abaqus',)),
    S('FM_STRAIN_AMPLITUDE', 0.01, minimum=1e-09, maximum=10, stages=('export_abaqus',)),
    S('FM_R_RATIO', -1.0, minimum=-1e6, maximum=1e6, stages=('export_abaqus',)),
    S('FM_CYCLES', 5, minimum=1, maximum=100000, stages=('export_abaqus',)),
    S('FM_OUTPUT_ELSET', '', stages=('export_abaqus',)),
    S('FM_WRITE_STEP', True, stages=('export_abaqus',)),
    S('FM_APPLIED_DISPLACEMENT', 0.02, minimum=-1e9, maximum=1e9, stages=('export_abaqus',)),
    S('FM_STEP_TIME', 50.0, minimum=1e-09, maximum=1e9, stages=('export_abaqus',)),
    S('FM_INITIAL_INC', 0.01, minimum=1e-12, maximum=1e9, stages=('export_abaqus',)),
    S('FM_MIN_INC', 1e-08, minimum=1e-12, maximum=1e9, stages=('export_abaqus',)),
    S('FM_MAX_INC', 0.5, minimum=1e-12, maximum=1e9, stages=('export_abaqus',)),
    S('FM_MAX_INCREMENTS', 10000, minimum=1, maximum=1000000000, stages=('export_abaqus',)),
    S('FM_OUTPUT_INTERVAL', 2.5, minimum=1e-09, maximum=1e9, stages=('export_abaqus',)),
    S('FM_ELEMENT_OUTPUTS', 'LE, S, SDV', stages=('export_abaqus',)),
    S('FM_NODE_OUTPUTS', 'U, RF', stages=('export_abaqus',)),
    S('FM_LENGTH_SCALE', 1.0, minimum=1e-12, maximum=1e12, stages=('export_abaqus', 'compare_before_after')),
    S('FM_OUTPUT_UNITS', 'same as source', kind='choice', choices=('same as source', 'microns', 'mm', 'm'), stages=('export_abaqus', 'compare_before_after')),
    S('FM_HEADER_MESSAGE', '', stages=('export_abaqus',)),
    S('FM_PREFIX_PAG', 'es_pag_', stages=('export_abaqus',)),
    S('FM_PREFIX_PCK', 'es_pck_', stages=('export_abaqus',)),
    S('FM_PREFIX_BLK', 'es_blk_', stages=('export_abaqus',)),
    S('FM_PREFIX_SBLK', 'es_subblk_', stages=('export_abaqus',)),
    S('FM_NSET_XMIN', 'ns_xminus', stages=('export_abaqus',)),
    S('FM_NSET_XMAX', 'ns_xplus', stages=('export_abaqus',)),
    S('FM_NSET_YMIN', 'ns_yminus', stages=('export_abaqus',)),
    S('FM_NSET_YMAX', 'ns_yplus', stages=('export_abaqus',)),
    S('FM_NSET_ZMIN', 'ns_zminus', stages=('export_abaqus',)),
    S('FM_NSET_ZMAX', 'ns_zplus', stages=('export_abaqus',)),
    S('ABAQUS_ELSET_PREFIX', 'grn_', stages=('export_abaqus',)),
    S('ABAQUS_UNITS', 'unspecified', stages=('export_abaqus',)),
    S('REQUIRE_READY_TET_MESH', False, stages=('export_abaqus',)),

)

SCHEMA: dict[str, Setting] = {s.key: s for s in SETTINGS}
assert len(SCHEMA) == len(SETTINGS), "a setting is listed twice"

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}
_AUTO_WORDS = {"", "auto", "none", "null"}


def _table(extra: Iterable[Setting] = ()) -> dict[str, Setting]:
    table = dict(SCHEMA)
    for setting in extra:
        table[setting.key] = setting
    return table


def default_config(extra: Iterable[Setting] = ()) -> dict:
    """A fresh, mutable settings dict holding every default (plus those of ``extra``)."""
    config = {s.key: copy.deepcopy(s.default) for s in SETTINGS}
    for setting in extra:
        config[setting.key] = copy.deepcopy(setting.default)
    config.update(copy.deepcopy(INTERNAL_DEFAULTS))
    return config


def settings_of_stage(stage_id: str) -> tuple[str, ...]:
    """Names of the settings that decide the results of ``stage_id``."""
    return tuple(s.key for s in SETTINGS if stage_id in s.stages)


# --- versions ------------------------------------------------------------------------------------
def _v1_to_v2(config: dict) -> dict:
    """Version 1 settings predate the dataset type and the subset stage."""
    if "DATASET_TYPE" not in config and config.get("SAMPLE_PATH"):
        config["DATASET_TYPE"] = "labels"          # such a run loaded a labels image
    return config


MIGRATIONS = {1: _v1_to_v2}


def migrate_config(config: Mapping, from_version: int) -> dict:
    """Bring settings saved at ``from_version`` up to :data:`CONFIG_VERSION`."""
    if from_version > CONFIG_VERSION:
        raise ValueError(
            f"Settings version {from_version} is newer than this program ({CONFIG_VERSION}).")
    config = dict(config)
    for version in range(from_version, CONFIG_VERSION):
        config = MIGRATIONS[version](config)
    return config


# --- values --------------------------------------------------------------------------------------
def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


PATH_SETTINGS = ("RAW_FOLDER", "TWINNED_FOLDER", "SAMPLE_PATH", "UMAT_FILE")
_QUOTES = "\"'“”‘’"
_DOUBLE_QUOTES = "\"“”"


def clean_path_text(text: Any) -> str:
    """A path as typed or pasted: no surrounding blanks, line ends or quotes.

    Windows "Copy as path" puts the path in double quotes; they are not part of the path. A pair of
    single quotes is removed too; a double quote never belongs to a Windows path, so a stray one at either
    end goes as well.
    """
    cleaned = str(text if text is not None else "").strip()
    while True:
        before = cleaned
        if len(cleaned) >= 2 and cleaned[0] in _QUOTES and cleaned[-1] in _QUOTES:
            cleaned = cleaned[1:-1].strip()
        cleaned = cleaned.strip(_DOUBLE_QUOTES).strip()
        if cleaned == before:
            return cleaned


def coerce_value(setting: Setting, value: Any) -> tuple[Any, Optional[str]]:
    """``(value, problem)``: ``value`` brought to the setting's type; ``problem`` is ``None`` when it was fine.

    A value that cannot be used is replaced by the default and described in ``problem``.
    """
    key, default, kind = setting.key, setting.default, setting.kind
    if value is None:
        if default is None or setting.auto:
            return None, None
        return copy.deepcopy(default), f"{key}: empty value reset to {default!r}"
    if kind == "bool":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), None
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE, None
        return default, f"{key}: {value!r} is not a yes/no value; reset to {default!r}"
    if kind in ("int", "float"):
        number = _number(value)
        if number is None:
            return default, f"{key}: {value!r} is not a number; reset to {default!r}"
        if kind == "int":
            if number != int(number):
                return default, f"{key}: {value!r} is not a whole number; reset to {default!r}"
            number = int(number)
        low, high = setting.minimum, setting.maximum
        if (low is not None and number < low) or (high is not None and number > high):
            return default, f"{key}: {value!r} is outside [{low}, {high}]; reset to {default!r}"
        return number, None
    if kind == "choice":
        if str(value) in setting.choices:
            return str(value), None
        return default, f"{key}: {value!r} is not one of {list(setting.choices)}; reset to {default!r}"
    if kind == "int_choice":
        number = _number(value)
        if number is not None and number == int(number) and int(number) in setting.choices:
            return int(number), None
        return default, f"{key}: {value!r} is not one of {list(setting.choices)}; reset to {default!r}"
    if kind == "sequence":
        try:
            items = [] if isinstance(value, (str, bytes)) else [float(v) for v in value]
        except (TypeError, ValueError):
            items = []
        if len(items) != setting.size:
            return copy.deepcopy(default), f"{key}: needs {setting.size} numbers; reset to {default!r}"
        if setting.integral:
            items = [int(v) if float(v).is_integer() else v for v in items]
        return (tuple(items) if setting.container == "tuple" else items), None
    if kind == "ids":
        try:
            if isinstance(value, (str, bytes)):
                raise TypeError
            items = [int(v) for v in value]
        except (TypeError, ValueError):
            return copy.deepcopy(default), f"{key}: needs a list of whole numbers; reset to {default!r}"
        return tuple(items), None
    if kind == "text":
        return (clean_path_text(value) if key in PATH_SETTINGS else str(value)), None
    return value, None


def coerce_config(loaded: Mapping, extra: Iterable[Setting] = ()) -> tuple[dict, list[str]]:
    """Defaults overlaid with ``loaded`` values, each brought to its type.

    Returns ``(config, problems)``. A value that cannot be used falls back to the
    default and is listed; settings this version does not know are kept.
    """
    extra = tuple(extra)
    table = _table(extra)
    config = default_config(extra)
    problems: list[str] = []
    for key, value in loaded.items():
        setting = table.get(key)
        if setting is None:
            config[key] = value
            if not str(key).startswith("_"):
                problems.append(f"{key}: unknown setting kept as it is")
            continue
        config[key], problem = coerce_value(setting, value)
        if problem:
            problems.append(problem)
    return config, problems


def load_versioned(loaded: Mapping, version: int, extra: Iterable[Setting] = ()) -> tuple[dict, list[str]]:
    """Migrate settings saved at ``version`` and coerce them; returns ``(config, problems)``."""
    return coerce_config(migrate_config(loaded, version), extra)


# --- text entry (command line, files written by hand) ----------------------------------------------
def parse_value(setting: Setting, text: str) -> Any:
    """The value a person typed for ``setting``, checked against its type and limits."""
    text = str(text).strip()
    kind = setting.kind
    if text.lower() in _AUTO_WORDS - {""} and (setting.auto or setting.default is None):
        value: Any = None
    elif kind in ("sequence", "ids"):
        parts = [p for p in text.replace(",", " ").replace(";", " ").split() if p]
        try:
            value = [float(p) for p in parts] if kind == "sequence" else [int(p) for p in parts]
        except ValueError:
            raise SettingsError(f"{setting.key}: {text!r} is not a list of numbers.") from None
    elif kind == "any":
        try:
            value = json.loads(text)
        except ValueError:
            raise SettingsError(f"{setting.key}: {text!r} is not valid JSON.") from None
    else:
        value = text
    checked, problem = coerce_value(setting, value)
    if problem:
        raise SettingsError(problem)
    return checked


def parse_assignment(text: str, extra: Iterable[Setting] = ()) -> tuple[str, Any]:
    """``"KEY=VALUE"`` -> ``(KEY, checked value)``."""
    key, separator, value = str(text).partition("=")
    key = key.strip()
    if not separator or not key:
        raise SettingsError(f"{text!r} is not of the form KEY=VALUE.")
    setting = _table(extra).get(key)
    if setting is None:
        raise SettingsError(f"{key} is not a setting; see the list of settings for the names.")
    return key, parse_value(setting, value)


def build_config(overrides: Optional[Mapping] = None, *, extra: Iterable[Setting] = (),
                 **named) -> dict:
    """Defaults changed by ``overrides`` and ``named`` values, each checked.

    Unknown names and unusable values raise :class:`SettingsError`.
    """
    extra = tuple(extra)
    table = _table(extra)
    config = default_config(extra)
    changes = dict(overrides or {})
    changes.update(named)
    for key, value in changes.items():
        setting = table.get(key)
        if setting is None:
            if key in INTERNAL_DEFAULTS:
                config[key] = value
                continue
            raise SettingsError(f"{key} is not a setting.")
        checked, problem = coerce_value(setting, value)
        if problem:
            raise SettingsError(problem)
        config[key] = checked
    return config


# --- comparison with the defaults ----------------------------------------------------------------
def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return ["__tuple__", *[_normalize(v) for v in value]]
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, float):
        return round(value, 12)
    return value


def differences(defaults: Mapping, current: Mapping) -> list[dict]:
    """``[{key, default, current}, ...]`` for the settings that differ from ``defaults``.

    Settings present only in ``current`` are listed with ``default=None``.
    """
    changes: list[dict] = []
    for key in sorted(set(defaults) | set(current), key=str):
        if key in defaults and key in current:
            if _normalize(defaults[key]) == _normalize(current[key]):
                continue
            changes.append({"key": key, "default": defaults[key], "current": current[key]})
        elif key in current:
            changes.append({"key": key, "default": None, "current": current[key]})
    return changes
