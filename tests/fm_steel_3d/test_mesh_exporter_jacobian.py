"""Regression test: every Kuhn-tet element written by MeshExporter3D must
have a positive Abaqus C3D4/C3D10 Jacobian.

All 6 rows of the ``_KUHN_TETS`` table (used by ``_write_elements_c3d4``)
and all 6 independently hand-written tet blocks inside
``_write_elements_c3d10`` previously had inverted node winding -- every
single emitted tetrahedron had det([n2-n1, n3-n1, n4-n1]) < 0, which Abaqus
rejects as a negative-volume element. This test drives the real writer
methods against a small synthetic active-voxel mask, parses the .inp text
they produce back into node coordinates and element connectivity, and
checks the sign of every element's Jacobian directly -- it would have
failed against the pre-fix code for all 96 tets (48 voxels * 2 grids * ...
here: 8 voxels * 6 tets = 48 tets per element type).
"""
import io

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import MeshExporter3D


def _signed_vol6(n1, n2, n3, n4):
    n1, n2, n3, n4 = (np.array(p) for p in (n1, n2, n3, n4))
    v1, v2, v3 = n2 - n1, n3 - n1, n4 - n1
    return float(np.dot(v1, np.cross(v2, v3)))


def _parse_nodes(text):
    """*Node section -> {node_id: (x, y, z)}."""
    nodes = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('*'):
            continue
        parts = [p.strip() for p in line.split(',')]
        nid = int(parts[0])
        x, y, z = (float(v) for v in parts[1:4])
        nodes[nid] = (x, y, z)
    return nodes


def _parse_c3d4_elements(text):
    """*Element, type=C3D4 -> list of (eid, [n1,n2,n3,n4])."""
    elems = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('*'):
            continue
        parts = [int(p.strip()) for p in line.split(',') if p.strip()]
        elems.append((parts[0], parts[1:5]))
    return elems


def _parse_c3d10_elements(text):
    """*Element, type=C3D10 -> list of (eid, [n1..n10]); each element is 2 lines."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('*')]
    assert len(lines) % 2 == 0, "expected exactly two lines per C3D10 element"
    elems = []
    for i in range(0, len(lines), 2):
        head = [p.strip() for p in lines[i].split(',') if p.strip()]
        tail = [p.strip() for p in lines[i + 1].split(',') if p.strip()]
        nums = [int(v) for v in (head + tail)]
        elems.append((nums[0], nums[1:11]))
    return elems


@pytest.fixture
def exporter():
    return MeshExporter3D(verbosity=0)


def _active_mask(nx=2, ny=2, nz=2):
    return np.ones((nx, ny, nz), dtype=bool)


def test_c3d4_elements_have_positive_jacobian(exporter):
    NX = NY = NZ = 2
    active_mask = _active_mask(NX, NY, NZ)
    node_mask = exporter._build_active_node_mask(active_mask, NX, NY, NZ)

    node_buf = io.StringIO()
    exporter._write_nodes(node_buf, NX, NY, NZ, dx=1.0, node_mask=node_mask)
    nodes = _parse_nodes(node_buf.getvalue())

    elem_buf = io.StringIO()
    exporter._write_elements_c3d4(elem_buf, NX, NY, NZ, active_mask)
    elems = _parse_c3d4_elements(elem_buf.getvalue())

    assert len(elems) == NX * NY * NZ * 6  # 6 Kuhn tets per voxel
    bad = []
    for eid, conn in elems:
        pts = [nodes[n] for n in conn]
        det = _signed_vol6(*pts)
        if det <= 0:
            bad.append((eid, det))
    assert not bad, f"{len(bad)} C3D4 elements with non-positive Jacobian: {bad[:5]}"


def test_c3d10_elements_have_positive_jacobian(exporter):
    NX = NY = NZ = 2
    active_mask = _active_mask(NX, NY, NZ)
    node_mask = exporter._build_active_node_mask(active_mask, NX, NY, NZ)
    xm_m = exporter._build_active_xmid_mask(active_mask, NX, NY, NZ)
    ym_m = exporter._build_active_ymid_mask(active_mask, NX, NY, NZ)
    zm_m = exporter._build_active_zmid_mask(active_mask, NX, NY, NZ)
    xydm_m = exporter._build_active_xydm_mask(active_mask, NX, NY, NZ)
    xzdm_m = exporter._build_active_xzdm_mask(active_mask, NX, NY, NZ)
    yzdm_m = exporter._build_active_yzdm_mask(active_mask, NX, NY, NZ)
    bdm_m = active_mask.copy()

    node_buf = io.StringIO()
    exporter._write_nodes_c3d10(node_buf, NX, NY, NZ, 1.0, node_mask,
                                xm_m, ym_m, zm_m, xydm_m, xzdm_m, yzdm_m, bdm_m)
    nodes = _parse_nodes(node_buf.getvalue())

    Nc = (NX + 1) * (NY + 1) * (NZ + 1)
    elem_buf = io.StringIO()
    exporter._write_elements_c3d10(elem_buf, NX, NY, NZ, active_mask, Nc)
    elems = _parse_c3d10_elements(elem_buf.getvalue())

    assert len(elems) == NX * NY * NZ * 6
    bad = []
    for eid, conn in elems:
        corners = [nodes[n] for n in conn[:4]]  # first 4 = corner nodes
        det = _signed_vol6(*corners)
        if det <= 0:
            bad.append((eid, det))
    assert not bad, f"{len(bad)} C3D10 elements with non-positive corner Jacobian: {bad[:5]}"

    # Every mid-edge node must sit exactly at the midpoint of its corner pair
    # (sanity: catches a mismatched N5..N10 relabelling that the Jacobian
    # check alone wouldn't -- wrong-but-still-positive-volume tets).
    edge_pairs = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    for eid, conn in elems:
        corners = [np.array(nodes[n]) for n in conn[:4]]
        mids = conn[4:10]
        for (a, b), mid_nid in zip(edge_pairs, mids):
            expected = (corners[a] + corners[b]) / 2
            actual = np.array(nodes[mid_nid])
            assert np.allclose(expected, actual), (
                f"elem {eid}: mid-node for edge {a+1}-{b+1} not at midpoint "
                f"(expected {expected}, got {actual})"
            )
