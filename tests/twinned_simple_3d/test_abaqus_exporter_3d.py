"""AbaqusExporter3D element types: C3D8 (one brick per voxel) and C3D4
(six tetrahedra per voxel), on a small hand-made twinned structure.
The materials, units and load step are in test_abaqus_exporter_3d_model.py."""
import re
from collections import Counter

import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import (
    SUPPORTED_ELEMENT_TYPES, AbaqusExporter3D)

NX, NY, NZ = 3, 4, 5


def _structure():
    """(nz, ny, nx) label image, the pipeline's axis order: a host grain 1
    with a twin slab 2 through it, a non-participating grain 3 and a
    secondary twin 4 inside the twin."""
    lgi = np.full((NZ, NY, NX), 3, dtype=np.int32)
    lgi[:, :2, :] = 1
    lgi[1:4, :2, :] = 2
    lgi[2, 0, 0] = 4
    twin_role = {1: 'host', 2: 'primary_twin', 3: 'non_host', 4: 'secondary_twin'}
    twin_parent_of = {2: 1, 4: 2}
    quats = {g: np.array([1.0, 0.0, 0.0, 0.0]) for g in twin_role}
    return lgi, twin_role, twin_parent_of, quats


def _export(tmp_path, element_type, **kwargs):
    """Geometry tests use length_scale=1, so a voxel is a unit cube."""
    kwargs.setdefault('length_scale', 1.0)
    lgi, role, parent, quats = _structure()
    exp = AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                           all_quats=quats, element_type=element_type,
                           write_variant_elsets=False, **kwargs)
    exp.write(str(tmp_path))
    return exp


def _nodes(path):
    rows = [l for l in open(path / '01_nodes.inp') if l[:1].isdigit()]
    data = np.array([[float(v) for v in l.split(',')] for l in rows])
    return {int(r[0]): r[1:] for r in data}


def _elements(path):
    text = open(path / '02_elements.inp').read()
    header = re.search(r'\*Element, type=(\S+)', text).group(1)
    rows = [l for l in text.splitlines() if l[:1].isdigit()]
    conn = np.array([[int(v) for v in l.split(',')] for l in rows])
    return header, conn


def _elsets(path, fname):
    out, name = {}, None
    for line in open(path / fname):
        m = re.match(r'\*Elset, elset=(\S+)', line)
        if m:
            name = m.group(1)
            out[name] = []
        elif name and line[:1].isdigit():
            out[name] += [int(v) for v in line.split(',')]
    return out


def _tet_volumes(nodes, conn):
    p = np.array([[nodes[n] for n in row[1:]] for row in conn])
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0


def test_unsupported_element_type_is_rejected():
    lgi, role, parent, quats = _structure()
    for bad in ('C3D20', 'C3D10', 'C3D4R'):
        with pytest.raises(ValueError, match='not supported'):
            AbaqusExporter3D(lgi=lgi, twin_role=role, twin_parent_of=parent,
                             all_quats=quats, element_type=bad)
    assert SUPPORTED_ELEMENT_TYPES == ('C3D8', 'C3D4')


def test_c3d8_is_one_eight_node_brick_per_voxel(tmp_path):
    exp = _export(tmp_path, 'C3D8')
    header, conn = _elements(tmp_path)
    assert header == 'C3D8'
    assert conn.shape == (NX * NY * NZ, 9)
    assert exp.n_elements == NX * NY * NZ


def test_c3d4_is_six_positive_tets_per_voxel_filling_the_domain(tmp_path):
    exp = _export(tmp_path, 'C3D4')
    header, conn = _elements(tmp_path)
    assert header == 'C3D4'
    assert conn.shape == (6 * NX * NY * NZ, 5)
    assert exp.n_elements == 6 * NX * NY * NZ
    np.testing.assert_array_equal(conn[:, 0], np.arange(1, len(conn) + 1))
    vol = _tet_volumes(_nodes(tmp_path), conn)
    assert (vol > 0).all()                                   # positive Jacobian
    np.testing.assert_allclose(vol, 1.0 / 6.0)               # unit voxels
    assert vol.sum() == pytest.approx(NX * NY * NZ)


def test_c3d4_mesh_is_conforming(tmp_path):
    """Every interior triangle is shared by exactly two tets, every boundary
    triangle lies on the domain surface: no hanging faces between voxels."""
    _export(tmp_path, 'C3D4')
    _, conn = _elements(tmp_path)
    nodes = _nodes(tmp_path)
    faces = Counter()
    for row in conn[:, 1:]:
        for skip in range(4):
            faces[tuple(sorted(np.delete(row, skip)))] += 1
    assert set(faces.values()) <= {1, 2}
    lo, hi = np.zeros(3), np.array([NX, NY, NZ], float)
    for face, count in faces.items():
        if count == 1:
            pts = np.array([nodes[n] for n in face])
            on_surface = ((np.isclose(pts, lo) | np.isclose(pts, hi)).all(axis=0)).any()
            assert on_surface, face
    assert sum(1 for c in faces.values() if c == 1) == 4 * (NX * NY + NY * NZ + NX * NZ)


@pytest.mark.parametrize("etype,per_voxel", [('C3D8', 1), ('C3D4', 6)])
def test_cell_elsets_cover_every_element_once_and_match_the_grains(tmp_path, etype, per_voxel):
    _export(tmp_path, etype)
    cells = _elsets(tmp_path, '03a_elsets_cells.inp')
    ids = [e for v in cells.values() for e in v]
    assert len(ids) == len(set(ids)) == per_voxel * NX * NY * NZ
    lgi, *_ = _structure()
    voxel_label = np.transpose(lgi, (2, 1, 0)).ravel(order='C')
    for name, eids in cells.items():
        gid = int(name.rsplit('_', 1)[1])
        voxels = (np.asarray(eids) - 1) // per_voxel
        assert (voxel_label[voxels] == gid).all(), name
        assert len(eids) == per_voxel * int((voxel_label == gid).sum())


def test_c3d4_role_and_family_elsets_are_six_times_the_c3d8_ones(tmp_path):
    hexa, tet = tmp_path / 'hex', tmp_path / 'tet'
    hexa.mkdir(), tet.mkdir()
    _export(hexa, 'C3D8')
    _export(tet, 'C3D4')
    for fname in ('03b_elsets_roles.inp', '03c_elsets_families.inp'):
        a, b = _elsets(hexa, fname), _elsets(tet, fname)
        assert a.keys() == b.keys() and a
        for name in a:
            expected = sorted(6 * (e - 1) + k for e in a[name] for k in range(1, 7))
            assert sorted(b[name]) == expected, (fname, name)


def test_the_index_is_not_changed_by_a_c3d4_export(tmp_path):
    lgi, role, parent, quats = _structure()
    index = AbaqusExporter3D.build_index(lgi, role, parent, verbose=False)
    before = {g: v.copy() for g, v in index.grain_elems.items()}
    AbaqusExporter3D(index=index, twin_role=role, twin_parent_of=parent,
                     all_quats=quats, element_type='C3D4')
    assert all(np.array_equal(before[g], index.grain_elems[g]) for g in before)


def test_master_reports_the_element_count(tmp_path):
    _export(tmp_path, 'C3D4')
    master = open(tmp_path / 'model_master.inp').read()
    assert 'element type: C3D4' in master
    assert f'Active elements: {6 * NX * NY * NZ:,}' in master
