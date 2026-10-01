"""AbaqusExporter3D model data (05-08): materials, units and the load step
follow the reference model upxo_support/collab/T25C.inp, which runs in
Abaqus with the target UMAT. The reference values are written out here."""
import re

import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import (
    AbaqusExporter3D, REFERENCE_UMAT_TAIL, _quat_to_bunge, required_bc_faces,
    write_reference_umat_material)

NX, NY, NZ = 3, 4, 5


def _rot(phi1, Phi, phi2):
    c1, s1 = np.cos(np.radians(phi1)), np.sin(np.radians(phi1))
    c, s = np.cos(np.radians(Phi)), np.sin(np.radians(Phi))
    c2, s2 = np.cos(np.radians(phi2)), np.sin(np.radians(phi2))
    z1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    z2 = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    return z1 @ x @ z2


def _structure():
    lgi = np.full((NZ, NY, NX), 3, dtype=np.int32)
    lgi[:, :2, :] = 1
    lgi[1:4, :2, :] = 2
    lgi[2, 0, 0] = 7                       # sparse id: grain numbers must still be 1..N
    role = {1: 'host', 2: 'primary_twin', 3: 'non_host', 7: 'secondary_twin'}
    parent = {2: 1, 7: 2}
    rng = np.random.default_rng(3)
    quats = {}
    for g in role:
        q = rng.normal(size=4)
        quats[g] = q / np.linalg.norm(q)
    return lgi, role, parent, quats


def _export(tmp_path, **kwargs):
    lgi, role, parent, quats = _structure()
    exp = AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                           all_quats=quats, write_variant_elsets=False, **kwargs)
    exp.write(str(tmp_path))
    return exp, quats


def _read(tmp_path, name):
    return open(tmp_path / name).read()


def _materials(text):
    """[(name, depvar_line, constants_header, values)] in file order."""
    out = []
    blocks = re.split(r'^\*Material, name=', text, flags=re.M)[1:]
    for b in blocks:
        lines = [l for l in b.splitlines() if l and not l.startswith('**')]
        name = lines[0].strip()
        i = lines.index('*Depvar')
        j = next(k for k, l in enumerate(lines) if l.startswith('*User Material'))
        out.append((name, lines[i + 1], lines[j], [float(v) for v in lines[j + 1].split(',')]))
    return out


def _nsets(text):
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r'\*Nset, nset=(\S+)', line)
        if m:
            name = m.group(1)
            out[name] = []
        elif name and line[:1].isdigit():
            out[name] += [int(v) for v in line.split(',')]
    return out


def _nodes(tmp_path):
    rows = [l for l in _read(tmp_path, '01_nodes.inp').splitlines() if l[:1].isdigit()]
    data = np.array([[float(v) for v in l.split(',')] for l in rows])
    return {int(r[0]): r[1:] for r in data}


# ------------------------------------------------------------------ defaults
def test_defaults_match_the_reference(tmp_path):
    exp, _ = _export(tmp_path)
    assert exp.element_type == 'C3D8'
    assert exp.material_format == 'reference_umat'
    assert exp.n_depvar == 1
    assert exp.length_scale == 1e-3
    assert (exp.load_axis, exp.applied_strain, exp.step_time) == ('z', 0.2, 80.0)


# ------------------------------------------------------------------ 01 units
def test_nodes_are_scaled_to_mm(tmp_path):
    _export(tmp_path, voxel_size_um=2.0)
    xyz = np.array(list(_nodes(tmp_path).values()))
    np.testing.assert_allclose(xyz.min(axis=0), 0.0)
    np.testing.assert_allclose(xyz.max(axis=0), np.array([NX, NY, NZ]) * 2.0e-3)
    assert 'units: mm' in _read(tmp_path, '01_nodes.inp')


def test_length_scale_one_keeps_microns(tmp_path):
    _export(tmp_path, length_scale=1.0)
    xyz = np.array(list(_nodes(tmp_path).values()))
    np.testing.assert_allclose(xyz.max(axis=0), [NX, NY, NZ])


# ------------------------------------------------------------------ 05 materials
def test_materials_have_the_reference_layout(tmp_path):
    _export(tmp_path)
    mats = _materials(_read(tmp_path, '05_materials.inp'))
    assert len(mats) == 4
    for k, (_, depvar, header, vals) in enumerate(mats, start=1):
        assert depvar == '1,'
        assert header == '*User Material, constants=6'
        assert len(vals) == 6
        assert all(0.0 <= a < 360.0 for a in vals[:3])
        assert vals[3] == k                                  # sequential 1..N
        assert tuple(vals[4:]) == REFERENCE_UMAT_TAIL == (15.0, 0.0)


def test_material_angles_are_the_grain_orientations(tmp_path):
    _, quats = _export(tmp_path)
    mats = _materials(_read(tmp_path, '05_materials.inp'))
    gids = [int(name.rsplit('_', 1)[1]) for name, *_ in mats]
    assert gids == sorted(quats)
    for (_, _, _, vals), gid in zip(mats, gids):
        np.testing.assert_allclose(_rot(*vals[:3]), _rot(*_quat_to_bunge(quats[gid])),
                                   atol=1e-6)


def test_every_section_names_a_written_material(tmp_path):
    _export(tmp_path)
    mats = {name for name, *_ in _materials(_read(tmp_path, '05_materials.inp'))}
    secs = re.findall(r'material=(\S+)', _read(tmp_path, '06_sections.inp'))
    assert set(secs) == mats and len(secs) == len(mats)


def test_bunge_euler_format_is_still_available(tmp_path):
    _export(tmp_path, material_format='bunge_euler', n_depvar=100)
    text = _read(tmp_path, '05_materials.inp')
    assert text.count('*User Material, constants=3') == 4
    assert '*Depvar\n100\n' in text


def test_write_reference_umat_material_wraps_angles():
    import io
    f = io.StringIO()
    write_reference_umat_material(f, 'M', (-90.0, 30.0, -0.5), 12)
    assert f.getvalue().splitlines()[-1] == '270.000000, 30.000000, 359.500000, 12., 15., 0.'


# ------------------------------------------------------------------ 07 interactions
def test_interactions_file_has_no_keywords(tmp_path):
    _export(tmp_path)
    text = _read(tmp_path, '07_interactions.inp')
    assert not [l for l in text.splitlines() if l.startswith('*') and not l.startswith('**')]


# ------------------------------------------------------------------ 08 step
def _keywords(text):
    return [l.split(',')[0].strip() for l in text.splitlines()
            if l.startswith('*') and not l.startswith('**')]


def test_step_has_the_reference_keywords_and_parameters(tmp_path):
    _export(tmp_path)
    text = _read(tmp_path, '08_steps_output.inp')
    assert _keywords(text) == [
        '*Step', '*Static', '*Boundary', '*Boundary', '*Boundary', '*Boundary',
        '*Restart', '*Output', '*Element Output', '*Node Output', '*Output', '*End Step']
    lines = text.splitlines()
    assert '*Step, name=Step-1, nlgeom=YES, inc=10000' in lines
    assert lines[lines.index('*Static') + 1] == '0.01, 80, 1e-08, 1'
    assert '*Restart, write, frequency=0' in lines
    assert '*Output, field, time interval=0.5' in lines
    assert '*Element Output, directions=YES' in lines
    assert 'LE, NE, S, SDV' in lines
    assert '*Output, history, variable=PRESELECT' in lines


def _boundaries(text):
    lines = text.splitlines()
    return [lines[i + 1] for i, l in enumerate(lines) if l == '*Boundary']


def test_step_loads_the_top_face_to_20_percent_strain_and_fixes_the_min_faces(tmp_path):
    _export(tmp_path)
    bcs = _boundaries(_read(tmp_path, '08_steps_output.inp'))
    top = bcs[0].split(',')
    assert top[0] == 'ns_face_ZMAX' and top[1:3] == [' 3', ' 3']
    assert float(top[3]) == pytest.approx(0.2 * NZ * 1e-3)
    assert bcs[1:] == ['ns_face_XMIN, 1, 1', 'ns_face_YMIN, 2, 2', 'ns_face_ZMIN, 3, 3']


def test_bc_node_sets_exist_and_lie_on_their_faces(tmp_path):
    _export(tmp_path)
    nsets = _nsets(_read(tmp_path, '04_nsets_bc.inp'))
    nodes = _nodes(tmp_path)
    hi = np.array([NX, NY, NZ]) * 1e-3
    expect = {'ns_face_ZMAX': (2, hi[2]), 'ns_face_XMIN': (0, 0.0),
              'ns_face_YMIN': (1, 0.0), 'ns_face_ZMIN': (2, 0.0)}
    for bc in _boundaries(_read(tmp_path, '08_steps_output.inp')):
        name = bc.split(',')[0]
        axis, value = expect[name]
        coords = np.array([nodes[n][axis] for n in nsets[name]])
        np.testing.assert_allclose(coords, value)
        assert len(nsets[name]) == {0: (NY + 1) * (NZ + 1), 1: (NX + 1) * (NZ + 1),
                                    2: (NX + 1) * (NY + 1)}[axis]


def test_load_axis_and_strain_are_configurable(tmp_path):
    _export(tmp_path, load_axis='x', applied_strain=0.05, step_time=10.0)
    text = _read(tmp_path, '08_steps_output.inp')
    top = _boundaries(text)[0].split(',')
    assert top[0] == 'ns_face_XMAX' and top[1:3] == [' 1', ' 1']
    assert float(top[3]) == pytest.approx(0.05 * NX * 1e-3)
    assert '0.01, 10, 1e-08, 1' in text.splitlines()


def test_a_disabled_bc_node_set_is_written_anyway(tmp_path):
    lgi, role, parent, quats = _structure()
    with pytest.warns(UserWarning, match='ZMAX'):
        exp = AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                               all_quats=quats, write_variant_elsets=False,
                               nset_config={'ZMAX': {'enabled': False, 'prefix': 'top'}})
    exp.write(str(tmp_path))
    assert 'top' in _nsets(_read(tmp_path, '04_nsets_bc.inp'))
    assert _boundaries(_read(tmp_path, '08_steps_output.inp'))[0].startswith('top,')


def test_required_bc_faces():
    assert required_bc_faces('z') == {'ZMAX', 'XMIN', 'YMIN', 'ZMIN'}
    assert required_bc_faces('x') == {'XMAX', 'XMIN', 'YMIN', 'ZMIN'}


def test_write_step_false_leaves_no_step(tmp_path):
    lgi, role, parent, quats = _structure()
    exp = AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                           all_quats=quats, write_variant_elsets=False, write_step=False)
    with pytest.warns(UserWarning, match='no \\*Step'):
        exp.write(str(tmp_path))
    assert '*Step' not in _read(tmp_path, '08_steps_output.inp')


@pytest.mark.parametrize('kwargs', [dict(material_format='umat9'), dict(load_axis='w'),
                                    dict(length_scale=0.0)])
def test_bad_settings_are_rejected(kwargs):
    lgi, role, parent, quats = _structure()
    with pytest.raises(ValueError):
        AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                         all_quats=quats, **kwargs)


# ------------------------------------------------------------------ master
def test_master_includes_every_file_in_order(tmp_path):
    _export(tmp_path)
    master = _read(tmp_path, 'model_master.inp')
    includes = re.findall(r'\*INCLUDE, INPUT=(\S+)', master)
    assert includes == ['01_nodes.inp', '02_elements.inp', '03a_elsets_cells.inp',
                        '03b_elsets_roles.inp', '03c_elsets_families.inp',
                        '03d_elsets_variants.inp', '04_nsets_bc.inp', '05_materials.inp',
                        '06_sections.inp', '07_interactions.inp', '08_steps_output.inp']
    assert all((tmp_path / name).exists() for name in includes)
    assert 'units: mm' in master
