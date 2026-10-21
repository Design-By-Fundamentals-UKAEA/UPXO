"""Partitioned C3D4 Abaqus deck for FM-steel conformal tetrahedral meshes.

The deck follows the layout of the voxel FM-steel export: nodes, elements,
hierarchy element sets, face node sets, one material and section per feature
of the chosen level, an interactions template and a static step. Each
tetrahedron belongs to the feature its grain stands for (see
:mod:`.identity`); the parent-level sets are unions of those tetrahedra.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import numpy as np

from upxo._sup.text_rows import write_rows

from . import constants as C
from .audit import location_audit, reallocation_report
from .boundary_classes import class_nodes, classify
from .boundary_sets import boundary_nodes
from .constraints import periodic_pairs, summary as constraint_summary, template_text
from .deckcheck import check_deck, read_sets
from .features import feature_table, id_columns, tet_volumes, write_feature_table
from .hierarchy import parent_tables
from .interfaces import interface_faces
from .jobs import validate_job, write_job_scripts
from .loading import controls as load_controls, plan_load, validate_load
from .outputs import estimate_for_plan
from .mesh_files import tetrahedron_fields, write_msh, write_vtu

_CHUNK = 250_000


@dataclass
class DeckSettings:
    material_format: str = 'reference_umat'
    n_depvar: int = C.REFERENCE_N_DEPVAR
    load_axis: str = 'z'
    applied_displacement: float = 0.2
    step_time: float = 80.0
    write_step: bool = True
    initial_inc: float = 0.01
    min_inc: float = 1e-8
    max_inc: float = 1.0
    max_increments: int = 10000
    output_interval: float = 0.5
    load_mode: str = 'displacement'
    load_case: str = 'monotonic'
    load_sense: str = 'tension'
    target_strain: float = 0.02
    strain_rate: float = 4e-4
    hold_time: float = 0.0
    cycle_definition: str = 'min_max'
    strain_min: float = -0.01
    strain_max: float = 0.01
    strain_amplitude: float = 0.01
    r_ratio: float = -1.0
    cycles: int = 5
    output_elset: str = ''
    job_scripts: bool = False
    abaqus_command: str = 'abaqus'
    umat_file: str = ''
    job_name: str = 'fm_job'
    job_cpus: int = 4
    job_memory_gb: float = 0.0
    job_time_hours: float = 24.0
    slurm_partition: str = ''
    element_outputs: tuple = ('LE', 'NE', 'S', 'SDV')
    node_outputs: tuple = ('U',)
    length_scale: float = 1.0
    output_units: str | None = None
    header_message: str = ''
    elset_prefixes: dict = field(default_factory=lambda: dict(C.ELSET_PREFIXES))
    nset_names: dict = field(default_factory=lambda: dict(C.FACE_NSET_NAMES))
    face_tolerance: float = 1e-8
    boundary_sets: bool = False
    layer_sets: bool = False
    boundary_classes: bool = False
    low_angle_degrees: float = 15.0
    crystal_symmetry: str = 'cubic'
    feature_table: bool = False
    deck_check: bool = False
    write_msh: bool = False
    write_vtu: bool = False

    def validate(self):
        """Raise ``ValueError`` on an invalid setting; return output tuples."""
        if self.material_format not in C.MATERIAL_FORMATS:
            raise ValueError(f'material_format must be one of {C.MATERIAL_FORMATS}.')
        if self.load_axis not in C.LOAD_AXES:
            raise ValueError(f'load_axis must be one of {C.LOAD_AXES}.')
        if int(self.n_depvar) < 1:
            raise ValueError('n_depvar must be >= 1.')
        if not float(self.length_scale) > 0:
            raise ValueError('length_scale must be > 0.')
        if not 0 < float(self.face_tolerance) < 1:
            raise ValueError('face_tolerance must be in (0, 1).')
        if self.output_units is not None and self.output_units not in C.UNITS:
            raise ValueError(f'output_units must be one of {C.UNITS}.')
        if set(self.nset_names) != set(C.FACE_NSET_NAMES):
            raise ValueError('nset_names must name all six faces.')
        validate_load(self)
        validate_job(self)
        c = load_controls(self)
        return C.validate_step_controls(
            c['time'], c['initial_inc'], c['min_inc'], c['max_inc'], int(self.max_increments),
            c['output_interval'], self.element_outputs, self.node_outputs)


def _validate_mesh(points, tets, grains):
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    if tets.ndim != 2 or tets.shape[1] != 4 or not len(tets) or tets.dtype.kind not in 'iu':
        raise ValueError('Expected nonempty integer Mx4 tetrahedra')
    if tets.min() < 0 or tets.max() >= len(points):
        raise ValueError('Connectivity references missing nodes')
    if grains.shape != (len(tets),) or grains.dtype.kind not in 'iu' or np.any(grains < 0):
        raise ValueError('Expected one nonnegative grain ID per tetrahedron')
    total, smallest = 0.0, np.inf
    for start in range(0, len(tets), _CHUNK):
        v = points[tets[start:start + _CHUNK]]
        vol = np.einsum('ij,ij->i', v[:, 1] - v[:, 0],
                        np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])) / 6
        if not np.isfinite(vol).all() or np.any(vol <= 0):
            raise ValueError('Nonpositive or nonfinite tetrahedron volume')
        total += float(vol.sum())
        smallest = min(smallest, float(vol.min()))
    return total, smallest


def _groups(keys):
    """(key, ascending element IDs) per distinct key, keys sorted ascending."""
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(keys)]))
    return {int(sorted_keys[a]): order[a:b] + 1 for a, b in zip(starts, stops)}


def _write_ids(f, ids, per_line=10):
    ids = np.asarray(ids, dtype=np.int64)
    full = (len(ids) // per_line) * per_line
    if full:
        write_rows(f, ids[:full].reshape(-1, per_line), '%d', ', ')
    if full < len(ids):
        f.write(', '.join(map(str, ids[full:].tolist())) + '\n')


def _write_elset(f, name, ids):
    f.write(f'*Elset, elset={name}\n')
    _write_ids(f, ids)


def elset_name(prefixes, level, key):
    return prefixes[level] + '_'.join(str(k) for k in key)


def material_name(level, key):
    tag = {'pag': 'PAG', 'pck': 'PCK', 'blk': 'BLK', 'sblk': 'SUBBLK'}[level]
    return f'MAT_{tag}_' + '_'.join(str(k) for k in key)


def feature_elset_name(hierarchy, prefixes, feature):
    """Element set of a feature: the name its dataset gave it, else prefix and key by level."""
    return hierarchy.names.get(feature.label) or elset_name(prefixes, hierarchy.level, feature.key)


def feature_material_name(hierarchy, feature):
    return hierarchy.material_names.get(feature.label) or material_name(hierarchy.level, feature.key)


def _write_material(f, name, euler, global_id, settings):
    phi1, big_phi, phi2 = euler
    f.write(f'*Material, name={name}\n')
    f.write(f'** Bunge-Euler (deg): phi1={phi1:.4f}, Phi={big_phi:.4f}, phi2={phi2:.4f}\n')
    if settings.material_format == 'reference_umat':
        tail = C.REFERENCE_UMAT_TAIL
        f.write('*Depvar\n%d,\n*User Material, constants=6\n' % settings.n_depvar)
        f.write(f'{phi1:.6f}, {big_phi:.6f}, {phi2:.6f}, {global_id}., '
                f'{tail[0]:g}, {tail[1]:g}\n')
    elif settings.material_format == 'bunge_euler':
        f.write('*User Material, constants=3\n')
        f.write(f'{phi1:.6f}, {big_phi:.6f}, {phi2:.6f}\n')
        f.write(f'*Depvar\n{settings.n_depvar}\n')
    else:
        f.write('*Elastic\n210000., 0.3\n')
    f.write('**\n')


def _write_steps(f, settings, element_outputs, node_outputs, plan=None):
    if not settings.write_step:
        f.write('** No step written (write_step=False).\n')
        return
    faces = {'x': ('XMIN', 'XMAX', 1), 'y': ('YMIN', 'YMAX', 2), 'z': ('ZMIN', 'ZMAX', 3)}
    names = settings.nset_names
    _, max_face, dof = faces[settings.load_axis]
    for amplitude, table in plan.amplitudes.items():
        f.write('*Amplitude, name=%s, definition=TABULAR\n' % amplitude)
        pairs = ['%.10g, %.10g' % pair for pair in table]
        for start in range(0, len(pairs), 4):
            f.write(', '.join(pairs[start:start + 4]) + '\n')
    elset = f', elset={settings.output_elset}' if settings.output_elset else ''
    for index, step in enumerate(plan.steps):
        displacement = step.displacement * settings.length_scale
        f.write('** STEP: %s\n*Step, name=%s, nlgeom=YES, inc=%d\n' % (step.name, step.name, settings.max_increments))
        if step.kind == 'hold':
            f.write('** Hold: the loaded face stays at its final displacement.\n')
        # a step that follows an amplitude table has its time written as exactly as the table, so the table ends with the step
        period = '%.10g' % step.time if step.amplitude else '%g' % step.time
        f.write('*Static\n%g, %s, %g, %g\n' % (step.initial_inc, period, step.min_inc, step.max_inc))
        scaled = f', amplitude={step.amplitude}' if step.amplitude else ''
        f.write('*Boundary%s\n%s, %d, %d, %.10g\n' % (scaled, names[max_face], dof, dof, displacement))
        if index == 0:
            for _, (low, _high, axis_dof) in faces.items():
                f.write('*Boundary\n%s, %d, %d\n' % (names[low], axis_dof, axis_dof))
            f.write('*Restart, write, frequency=0\n')
        f.write('*Output, field, time interval=%g\n' % step.output_interval)
        if element_outputs:
            f.write(f'*Element Output{elset}, directions=YES\n' + ', '.join(element_outputs) + '\n')
        if node_outputs:
            f.write('*Node Output\n' + ', '.join(node_outputs) + ',\n')
        f.write('*Output, history, variable=PRESELECT\n*End Step\n')


def write_deck(directory, points, tetrahedra, tet_grain, hierarchy, identity,
               settings=None, *, mesh_voxel_length=1.0, meshed_labels=None, audit=True):
    """Write the deck into a new ``directory``; return a summary dictionary.

    Parameters
    ----------
    points, tetrahedra : mesh coordinates (in units where one raw voxel edge is
        ``mesh_voxel_length``) and 0-based node indices of linear tetrahedra.
    tet_grain : meshed grain label of every tetrahedron.
    hierarchy : :class:`~.hierarchy.Hierarchy` of the chosen level.
    identity : :class:`~.identity.IdentityMap` for the meshed grains.
    meshed_labels : optional cleaned label array (pipeline labels); when given a
        voxel reallocation report is written as well.
    audit : look up the raw voxel under every centroid and stop on an identity
        mismatch (:class:`~.audit.IdentityError`).
    """
    settings = settings or DeckSettings()
    element_outputs, node_outputs = settings.validate()
    if not mesh_voxel_length > 0:
        raise ValueError('mesh_voxel_length must be > 0')
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    g = np.asarray(tet_grain)
    total_volume, min_volume = _validate_mesh(p, t, g)
    lut = identity.lookup()
    if g.max() >= len(lut) or np.any(lut[g] < 0):
        raise ValueError('A tetrahedron carries a grain with no resolved identity')
    assigned = lut[g]
    unknown = sorted(set(np.unique(assigned).tolist()) - set(hierarchy.features))
    if unknown:
        raise ValueError(f'Identities missing from the hierarchy: {unknown[:5]}')

    audit_report = (location_audit(p, t, g, identity, hierarchy, mesh_voxel_length)
                    if audit else None)
    reallocation = (reallocation_report(hierarchy.lgi, meshed_labels, identity, hierarchy)
                    if meshed_labels is not None else None)

    used = np.unique(t)
    dropped_nodes = int(len(p) - len(used))
    compact = np.searchsorted(used, t)
    base = p[used]
    low, high = base.min(axis=0), base.max(axis=0)
    src_units = hierarchy.units
    dst_units = settings.output_units or src_units
    scale = (hierarchy.voxel_size * C.unit_scale(src_units, dst_units)
             * settings.length_scale / mesh_voxel_length)
    tol = settings.face_tolerance * float((high - low).max())
    faces = {}
    for axis, (lo_key, hi_key) in enumerate((('XMIN', 'XMAX'), ('YMIN', 'YMAX'), ('ZMIN', 'ZMAX'))):
        faces[lo_key] = np.flatnonzero(np.abs(base[:, axis] - low[axis]) <= tol) + 1
        faces[hi_key] = np.flatnonzero(np.abs(base[:, axis] - high[axis]) <= tol) + 1
    empty = [k for k, v in faces.items() if not len(v)]
    if empty:
        raise ValueError(f'No nodes found on face(s) {empty}')

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    with (out / '01_nodes.inp').open('w', encoding='ascii') as f:
        f.write('*Node\n')
        for start in range(0, len(base), _CHUNK):
            block = base[start:start + _CHUNK] * scale
            rows = np.column_stack((np.arange(start + 1, start + 1 + len(block)), block))
            write_rows(f, rows, ['%10d', '%14.6e', '%14.6e', '%14.6e'], ', ')
    with (out / '02_elements.inp').open('w', encoding='ascii') as f:
        f.write('*Element, type=C3D4\n')
        for start in range(0, len(compact), _CHUNK):
            block = compact[start:start + _CHUNK] + 1
            rows = np.column_stack((np.arange(start + 1, start + 1 + len(block)), block))
            write_rows(f, rows, '%10d', ', ')

    level = hierarchy.level
    leaf = _groups(assigned)
    pag_lut, pck_lut, pck_keys, blk_lut, blk_keys = parent_tables(hierarchy)
    files = ['01_nodes.inp', '02_elements.inp']

    def parent_sets(lut):
        return _groups(lut[assigned])

    with (out / '03a_elsets_pag.inp').open('w', encoding='ascii') as f:
        f.write('** PAG elsets\n')
        sets = _groups(pag_lut[assigned])
        for pag in hierarchy.pag_order:
            if pag in sets:
                named = hierarchy.names.get(pag) if level == 'pag' else None
                _write_elset(f, named or elset_name(settings.elset_prefixes, 'pag', (pag,)), sets[pag])
    files.append('03a_elsets_pag.inp')
    has_pck = level in ('pck', 'blk', 'sblk')
    has_blk = level in ('blk', 'sblk')
    if has_pck:
        with (out / '03b_elsets_pck.inp').open('w', encoding='ascii') as f:
            f.write('** Packet elsets\n')
            sets = parent_sets(pck_lut)
            rank = {key: i for i, key in enumerate(hierarchy.pck_order)}
            for index, key in sorted(enumerate(pck_keys), key=lambda kv: rank[kv[1]]):
                if index in sets:
                    _write_elset(f, elset_name(settings.elset_prefixes, 'pck', key), sets[index])
        files.append('03b_elsets_pck.inp')
    if has_blk:
        with (out / '03c_elsets_blk.inp').open('w', encoding='ascii') as f:
            f.write('** Block elsets\n')
            sets = parent_sets(blk_lut)
            rank = {key: i for i, key in enumerate(hierarchy.blk_order)}
            for index, key in sorted(enumerate(blk_keys), key=lambda kv: rank[kv[1]]):
                if index in sets:
                    _write_elset(f, elset_name(settings.elset_prefixes, 'blk', key), sets[index])
        files.append('03c_elsets_blk.inp')
    if level == 'sblk':
        with (out / '03d_elsets_subblk.inp').open('w', encoding='ascii') as f:
            f.write('** Sub-block elsets\n')
            for feature in hierarchy.ordered_features():
                if feature.label in leaf:
                    _write_elset(f, feature_elset_name(hierarchy, settings.elset_prefixes, feature),
                                 leaf[feature.label])
        files.append('03d_elsets_subblk.inp')
    if hierarchy.groups:
        with (out / '03e_elsets_groups.inp').open('w', encoding='ascii') as f:
            f.write('** Element sets that join features (by role and by host family)\n')
            for name, members in hierarchy.groups.items():
                ids = [leaf[label] for label in members if label in leaf]
                if ids:
                    _write_elset(f, name, np.sort(np.concatenate(ids)))
        files.append('03e_elsets_groups.inp')

    with (out / '04_nsets_bc.inp').open('w', encoding='ascii') as f:
        for face in ('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'):
            f.write(f'*Nset, nset={settings.nset_names[face]}, instance={C.INSTANCE_NAME}\n')
            _write_ids(f, faces[face])
    files.append('04_nsets_bc.inp')
    pairs = periodic_pairs(base, faces, tol)

    boundary = boundary_nodes(compact, assigned, len(used)) if (settings.boundary_sets or settings.layer_sets) else None
    if settings.boundary_sets:
        with (out / '04b_nsets_boundaries.inp').open('w', encoding='ascii') as f:
            f.write('** Grain-boundary and junction node sets (nodes shared by 2 or more / 3 or more features)\n')
            for name, nodes in ((C.GRAIN_BOUNDARY_NSET, boundary.grain_boundary),
                                (C.JUNCTION_NSET, boundary.junction)):
                if len(nodes):
                    f.write(f'*Nset, nset={name}, instance={C.INSTANCE_NAME}\n')
                    _write_ids(f, nodes + 1)
            for feature in hierarchy.ordered_features():
                nodes = boundary.by_feature.get(feature.label)
                if nodes is not None and len(nodes):
                    name = C.FEATURE_BOUNDARY_NSET_PREFIX + feature_elset_name(hierarchy, settings.elset_prefixes, feature)
                    f.write(f'*Nset, nset={name}, instance={C.INSTANCE_NAME}\n')
                    _write_ids(f, nodes + 1)
        files.append('04b_nsets_boundaries.inp')

    layers = None
    if settings.layer_sets:
        touches = np.zeros(len(used), dtype=bool)
        touches[boundary.grain_boundary] = True
        in_layer = touches[compact].any(axis=1)
        layer_ids, interior_ids = np.flatnonzero(in_layer) + 1, np.flatnonzero(~in_layer) + 1
        with (out / '03f_elsets_layers.inp').open('w', encoding='ascii') as f:
            f.write('** Elements with a node on a grain boundary, and the elements with none\n')
            for name, ids in ((C.GB_LAYER_ELSET, layer_ids), (C.INTERIOR_ELSET, interior_ids)):
                if len(ids):
                    _write_elset(f, name, ids)
        files.append('03f_elsets_layers.inp')
        layers = {'gb_layer_elements': int(len(layer_ids)), 'interior_elements': int(len(interior_ids)),
                  'gb_layer_fraction': float(len(layer_ids) / len(t))}

    present = [feat for feat in hierarchy.ordered_features() if feat.label in leaf]
    names_by_label = {feat.label: feature_elset_name(hierarchy, settings.elset_prefixes, feat) for feat in present}
    materials_by_label = {feat.label: feature_material_name(hierarchy, feat) for feat in present}

    classes, class_sets_written = None, False
    if settings.boundary_classes:
        found = interface_faces(base, compact, assigned)
        classes = classify(hierarchy, found, scale=scale, symmetry=settings.crystal_symmetry,
                           low_angle_degrees=settings.low_angle_degrees)
        by_set = class_nodes(found, classes)
        if by_set:
            with (out / '04c_nsets_boundary_classes.inp').open('w', encoding='ascii') as f:
                f.write('** Nodes on the grain boundaries of each kind (a node on the edge between kinds is in both),\n'
                        '** and on low- and high-angle boundaries\n')
                for name, nodes in by_set.items():
                    f.write(f'*Nset, nset={name}, instance={C.INSTANCE_NAME}\n')
                    _write_ids(f, nodes + 1)
            files.append('04c_nsets_boundary_classes.inp')
            class_sets_written = True
        write_boundary_class_table(out / C.BOUNDARY_CLASS_TABLE_NAME, classes, names_by_label)
        files.append(C.BOUNDARY_CLASS_TABLE_NAME)
        write_interface_surface(out / C.INTERFACE_FILE_NAME, base * scale, found, classes, scale)
        files.append(C.INTERFACE_FILE_NAME)
    with (out / '05_materials.inp').open('w', encoding='ascii') as f:
        if hierarchy.system == 'FM steel':
            f.write(
                "** *User Material stubs with Bunge-Euler angles (phi1, Phi, phi2, degrees).\n"
                "** constants=6: phi1, Phi, phi2 [Bunge-Euler deg],"
                " grain_id [global elset ID], phase_id, mat_data_flag.\n"
                f"** phase_id: {C.PHASE_MARTENSITE}={C.PHASE_NAMES[C.PHASE_MARTENSITE]}, "
                f"{C.PHASE_RETAINED_AUSTENITE}={C.PHASE_NAMES[C.PHASE_RETAINED_AUSTENITE]}.\n"
                "** Replace with your CPFEM constitutive definition as needed.\n**\n")
        else:
            f.write(
                "** One *Material per grain, with Bunge-Euler angles (phi1, Phi, phi2, degrees).\n"
                "** reference_umat constants: phi1, Phi, phi2, grain number (1..N), then the reference tail.\n"
                "** Replace with your CPFEM constitutive definition as needed.\n**\n")
        for feat in present:
            _write_material(f, feature_material_name(hierarchy, feat), feat.euler, feat.global_id, settings)
    files.append('05_materials.inp')
    with (out / '06_sections.inp').open('w', encoding='ascii') as f:
        f.write("** *Solid Section - lowest hierarchy level only.\n**\n")
        for feat in present:
            f.write(f"*Solid Section, elset={feature_elset_name(hierarchy, settings.elset_prefixes, feat)},"
                    f" material={feature_material_name(hierarchy, feat)}\n,\n")
    files.append('06_sections.inp')
    (out / '07_interactions.inp').write_text(
        template_text(pairs, settings.nset_names, C.INSTANCE_NAME, settings.load_axis), encoding='ascii')
    files.append('07_interactions.inp')

    mesh_files = []
    if settings.write_msh or settings.write_vtu:
        fields = tetrahedron_fields(assigned, hierarchy.features, names_by_label,
                                    ids=id_columns(hierarchy, [feat.label for feat in present]))
        scaled = base * scale
        if settings.write_msh:
            mesh_files.append(write_msh(out / f'{C.MESH_FILE_STEM}.msh', scaled, compact, fields).name)
        if settings.write_vtu:
            mesh_files.append(write_vtu(out / f'{C.MESH_FILE_STEM}.vtu', scaled, compact, fields).name)
        files.extend(mesh_files)
    plan, output_figures = None, None
    if settings.write_step:
        axis_length = domain_lengths(hierarchy, settings.output_units)['xyz'.index(settings.load_axis)]
        plan = plan_load(settings, axis_length)
    output_count = None
    if settings.output_elset:
        known = {}
        for path in sorted(out.glob('03*.inp')):
            known.update(read_sets(path, 'Elset'))
        if settings.output_elset not in known:
            raise ValueError(f'output_elset {settings.output_elset!r} is not an element set of the deck '
                             f'(for example: {", ".join(sorted(known)[:4])}).')
        output_count = len(known[settings.output_elset])
    if plan is not None:
        output_figures = estimate_for_plan(settings, plan, len(used), len(t), output_count)
    with (out / '08_steps_output.inp').open('w', encoding='ascii') as f:
        _write_steps(f, settings, element_outputs, node_outputs, plan)
    files.append('08_steps_output.inp')
    scripts = write_job_scripts(out, settings) if settings.job_scripts else []
    files.extend(scripts)

    if settings.feature_table:
        order, inverse = np.unique(assigned, return_inverse=True)
        volumes = np.bincount(inverse, weights=tet_volumes(base, compact)) * scale ** 3
        counts = np.bincount(inverse)
        columns, rows = feature_table(
            hierarchy, names_by_label, materials_by_label,
            {int(l): int(n) for l, n in zip(order, counts)}, {int(l): float(v) for l, v in zip(order, volumes)},
            settings.elset_prefixes)
        write_feature_table(out / C.FEATURE_TABLE_NAME, columns, rows)
        files.append(C.FEATURE_TABLE_NAME)

    extent = (high - low) * scale
    includes = ''.join(f'*INCLUDE, INPUT={name}\n' for name in files
                       if name.startswith(('01', '02', '03')))
    with (out / 'model_master.inp').open('w', encoding='ascii', errors='replace') as f:
        f.write(C.HEADER)
        if settings.header_message:
            f.write(f'** {settings.header_message}\n**\n')
        if hierarchy.origin is not None:
            f.write('** Subset of a parent structure: first voxel '
                    f'{tuple(hierarchy.origin)}, parent shape {tuple(hierarchy.parent_shape)}\n**\n')
        f.write(
            '*Heading\n'
            f'** {hierarchy.system} microstructure - element type: C3D4 (conformal tetrahedra)\n'
            f'** Nodes: {len(used):,}  |  Tetrahedra: {len(t):,}  |  Features: {len(present)}'
            f'  |  Level: {level}\n'
            f'** Scale: {scale:.6g} {dst_units} per mesh unit '
            f'(structure defined in {src_units}'
            f'{", length_scale=%g applied" % settings.length_scale if settings.length_scale != 1 else ""})\n'
            f'** Domain: {extent[0]:.4g} x {extent[1]:.4g} x {extent[2]:.4g} {dst_units}\n**\n'
            f'*Part, name={C.PART_NAME}\n' + includes +
            '*INCLUDE, INPUT=06_sections.inp\n*End Part\n*Assembly, name=Assembly\n'
            f'*Instance, name={C.INSTANCE_NAME}, part={C.PART_NAME}\n*End Instance\n'
            '*INCLUDE, INPUT=04_nsets_bc.inp\n'
            + ('*INCLUDE, INPUT=04b_nsets_boundaries.inp\n' if settings.boundary_sets else '')
            + ('*INCLUDE, INPUT=04c_nsets_boundary_classes.inp\n' if class_sets_written else '') +
            '*End Assembly\n'
            '*INCLUDE, INPUT=05_materials.inp\n*INCLUDE, INPUT=07_interactions.inp\n'
            '*INCLUDE, INPUT=08_steps_output.inp\n')
    files.append('model_master.inp')

    def dump(name, payload):
        (out / name).write_text(json.dumps(payload, indent=2, default=_encode), encoding='utf8')
        files.append(name)

    dump('identity_report.json', identity.report)
    if audit_report is not None:
        dump('location_audit.json', audit_report)
    if reallocation is not None:
        dump('reallocation_report.json', reallocation)
    deck_check = None
    if settings.deck_check:
        result = check_deck(out)
        dump('deck_check.json', result)
        deck_check = {'ok': result['ok'], 'failed': result['failed']}
    info = {
        'level': level, 'system': hierarchy.system, 'element_type': 'C3D4', 'nodes': int(len(used)),
        'unused_nodes_dropped': dropped_nodes, 'tetrahedra': int(len(t)),
        'features_with_elements': len(present),
        'features_in_structure': len(hierarchy.features),
        'features_without_elements': sorted(set(hierarchy.features) - set(leaf)),
        'source_units': src_units, 'output_units': dst_units,
        'voxel_size': hierarchy.voxel_size, 'mesh_voxel_length': mesh_voxel_length,
        'coordinate_scale': scale, 'domain_extent': extent.tolist(),
        'total_volume_mesh_units': total_volume, 'minimum_tet_volume_mesh_units': min_volume,
        'face_node_counts': {k: int(len(v)) for k, v in faces.items()},
        'subset_of_parent': (None if hierarchy.origin is None else {
            'origin_voxels': [int(v) for v in hierarchy.origin],
            'parent_shape': [int(n) for n in hierarchy.parent_shape]}),
        'boundary_sets': boundary.counts() if settings.boundary_sets else None,
        'layer_sets': layers,
        'load_case': None if plan is None else plan.summary,
        'job_scripts': scripts,
        'output_estimate': output_figures,
        'boundary_classes': None if classes is None else classes.summary,
        'feature_table': C.FEATURE_TABLE_NAME if settings.feature_table else None,
        'deck_check': deck_check,
        'constraints': {'active': False, 'periodic_pairs': constraint_summary(pairs)},
        'mesh_files': mesh_files,
        'settings': asdict(settings), 'simulation_ready': False,
        'note': 'Interactions are a template; no periodic, tie or contact constraint is active.'}
    dump('deck_info.json', info)
    return {'directory': out, 'files': files, 'info': info,
            'audit': audit_report, 'reallocation': reallocation}


def write_boundary_class_table(path, classes, names):
    """One row per pair of neighbouring features: their element sets, the kind of boundary, area and misorientation."""
    import csv

    low = classes.low_angle()
    with Path(path).open('w', encoding='ascii', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label_a', 'elset_a', 'label_b', 'elset_b', 'kind', 'triangles', 'area',
                         'misorientation_deg', 'low_angle'])
        for i, (a, b) in enumerate(classes.pairs):
            angle = classes.misorientation[i]
            writer.writerow([int(a), names.get(int(a), ''), int(b), names.get(int(b), ''), classes.kinds[i],
                             int(classes.faces[i]), f'{classes.area[i]:.9g}',
                             '' if angle != angle else f'{angle:.4f}', '' if low is None else int(low[i])])
    return Path(path)


def write_interface_surface(path, points, interfaces, classes, scale=1.0):
    """The interface triangles as a VTK surface; cell data: kind, misorientation, area and the two features."""
    import pyvista as pv

    k = len(interfaces.faces)
    cells = np.column_stack((np.full(k, 3, dtype=np.int64), interfaces.faces)).ravel() if k else np.zeros(0, dtype=np.int64)
    surface = pv.PolyData(np.asarray(points, dtype=float), cells)
    pair = classes.face_pair
    surface.cell_data['boundary_kind'] = classes.face_kind.astype(np.int64)
    surface.cell_data['misorientation_deg'] = classes.misorientation[pair] if k else np.zeros(0)
    surface.cell_data['area'] = interfaces.area * float(scale) ** 2
    surface.cell_data['feature_a'] = interfaces.labels[:, 0].astype(np.int64) if k else np.zeros(0, dtype=np.int64)
    surface.cell_data['feature_b'] = interfaces.labels[:, 1].astype(np.int64) if k else np.zeros(0, dtype=np.int64)
    surface = surface.clean(point_merging=False)
    surface.save(str(path))
    return Path(path)


def _encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def domain_lengths(hierarchy, output_units=None, length_scale=1.0):
    """Edge lengths of the structure along x, y and z in the output units."""
    units = output_units or hierarchy.units
    per_voxel = hierarchy.voxel_size * C.unit_scale(hierarchy.units, units) * length_scale
    return tuple(float(n) * per_voxel for n in hierarchy.lgi.shape)


def applied_strain(settings, hierarchy):
    """Engineering strain the applied displacement gives along the load axis.

    The displacement and the coordinates share the length scale, so it cancels.
    """
    length = domain_lengths(hierarchy, settings.output_units)['xyz'.index(settings.load_axis)]
    if settings.load_mode == 'displacement':
        return float(settings.applied_displacement) / length
    summary = plan_load(settings, length).summary
    if summary['case'] == 'monotonic':
        return float(summary['strain'])
    return max(abs(summary['strain_min']), abs(summary['strain_max']))
