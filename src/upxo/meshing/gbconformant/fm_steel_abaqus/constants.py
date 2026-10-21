"""Naming, unit and step-control constants of the FM-steel Abaqus deck format."""

MATERIAL_FORMATS = ('reference_umat', 'bunge_euler', 'orientation')
LOAD_AXES = ('x', 'y', 'z')
PART_NAME = 'FM_STEEL_RVE'
INSTANCE_NAME = 'FM_STEEL_RVE-1'
FACE_NSET_NAMES = {'XMIN': 'ns_xminus', 'XMAX': 'ns_xplus',
                   'YMIN': 'ns_yminus', 'YMAX': 'ns_yplus',
                   'ZMIN': 'ns_zminus', 'ZMAX': 'ns_zplus'}
ELSET_PREFIXES = {'pag': 'es_pag_', 'pck': 'es_pck_',
                  'blk': 'es_blk_', 'sblk': 'es_subblk_'}
GRAIN_BOUNDARY_NSET = 'ns_gb_all'
JUNCTION_NSET = 'ns_junction_all'
FEATURE_BOUNDARY_NSET_PREFIX = 'ns_gb_'
MESH_FILE_STEM = 'fm_mesh'
GB_LAYER_ELSET = 'es_gb_layer'
INTERIOR_ELSET = 'es_interior'
FEATURE_TABLE_NAME = 'features.csv'
BOUNDARY_CLASS_TABLE_NAME = 'boundary_classes.csv'
INTERFACE_FILE_NAME = 'fm_interfaces.vtp'
REFERENCE_UMAT_TAIL = (15.0, 0.0)
REFERENCE_N_DEPVAR = 1
PHASE_RETAINED_AUSTENITE = 1
PHASE_MARTENSITE = 2
PHASE_NAMES = {PHASE_RETAINED_AUSTENITE: 'retained_austenite',
               PHASE_MARTENSITE: 'martensite'}
HEADER = (
    "** Made with UPXO: UKAEA Poly-XTAL Operations.\n"
    "** Visit: https://github.com/Design-By-Fundamentals-UKAEA/UPXO\n"
    "** or: pip install upxo\n"
    "**\n"
)

_UNIT_SCALES = {
    ('microns', 'microns'): 1.0, ('microns', 'mm'): 1e-3, ('microns', 'm'): 1e-6,
    ('mm', 'microns'): 1e3, ('mm', 'mm'): 1.0, ('mm', 'm'): 1e-3,
    ('m', 'microns'): 1e6, ('m', 'mm'): 1e3, ('m', 'm'): 1.0,
}
UNITS = ('microns', 'mm', 'm')


def unit_scale(src, dst):
    """Factor converting a length from ``src`` units to ``dst`` units."""
    try:
        return _UNIT_SCALES[(src, dst)]
    except KeyError:
        raise ValueError(f"Units must be among {UNITS}; got {src!r} -> {dst!r}.") from None


def _as_output_list(value):
    items = value.split(',') if isinstance(value, str) else list(value)
    return tuple(str(v).strip().upper() for v in items if str(v).strip())


def validate_step_controls(step_time, initial_inc, min_inc, max_inc, max_increments,
                           output_interval, element_outputs, node_outputs):
    """Check the static-step controls; return normalised output-variable tuples."""
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
    return element_outputs, node_outputs
