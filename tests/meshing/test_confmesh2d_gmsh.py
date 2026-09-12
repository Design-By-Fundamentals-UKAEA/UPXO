"""Tests for confMesh2dGMSH — skip the whole module if gmsh is absent."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="gmsh Python package is required")

from shapely.geometry import box, MultiPolygon

from upxo.meshing.conformal_mesher2d import confMesh2dGMSH
from upxo.meshing.gsmesh2d import mesh_gs


def _two_squares():
    return {1: box(0, 0, 1, 1), 2: box(1, 0, 2, 1)}


def test_two_squares_triangles():
    m = confMesh2dGMSH()
    m.femesh_gmsh(_two_squares(), mesh_size_gb=0.4, mesh_size_bulk=0.6,
                  recombine_to_quads=False)
    assert 'triangle' in m.elConn
    assert m.validation_report['grains_missing_elements'] == []
    assert m.validation_report['degenerate_elements'] == 0
    m.form_elsets_gmsh()
    assert set(m.elsets) == {'grain.1', 'grain.2'}
    assert len(m.elsets['grain.1']) > 0
    assert len(m.elsets['grain.2']) > 0
    total = len(m.elsets['grain.1']) + len(m.elsets['grain.2'])
    assert total == len(m.elConn['triangle'])


def test_two_squares_quads():
    m = confMesh2dGMSH()
    m.femesh_gmsh(_two_squares(), mesh_size_gb=0.4, mesh_size_bulk=0.6,
                  mesh_algo=8, recombine_to_quads=True)
    n_el = sum(len(c) for c in m.elConn.values())
    assert n_el > 0
    assert m.validation_report['degenerate_elements'] == 0
    m.form_elsets_gmsh()
    m.build_boundary_nsets()
    assert m.nsets['LEFT'].size > 0
    assert m.nsets['RIGHT'].size > 0


def test_island_grain():
    cells = {
        1: box(0, 0, 3, 3).difference(box(1, 1, 2, 2)),
        2: box(1, 1, 2, 2),
    }
    m = confMesh2dGMSH()
    m.femesh_gmsh(cells, mesh_size_gb=0.35, mesh_size_bulk=0.6,
                  recombine_to_quads=False)
    m.form_elsets_gmsh()
    assert len(m.elsets['grain.1']) > 0
    assert len(m.elsets['grain.2']) > 0
    assert m.validation_report['grains_missing_elements'] == []


def test_interior_void_not_a_grain():
    cells = {1: box(0, 0, 3, 3).difference(box(1, 1, 2, 2))}
    m = confMesh2dGMSH()
    m.femesh_gmsh(cells, mesh_size_gb=0.35, mesh_size_bulk=0.6,
                  recombine_to_quads=False)
    m.form_elsets_gmsh()
    assert 'grain.1' in m.elsets
    assert len(m.elsets['grain.1']) > 0


def test_multipolygon_parts_share_elset():
    cells = {
        7: MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
    }
    result = mesh_gs(cells, method='conformal', mesh_size_gb=0.4,
                     mesh_size_bulk=0.6, recombine_to_quads=False, mesh_algo=6)
    m = result['mesher']
    m.form_elsets_gmsh()
    assert 'grain.7' in m.elsets
    assert len(m.elsets['grain.7']) == sum(len(c) for c in m.elConn.values())


def test_from_geometric_pxtal():
    pxtal = MultiPolygon([box(0, 0, 1, 1), box(1, 0, 2, 1)])
    m = confMesh2dGMSH.from_geometric_pxtal(
        pxtal=pxtal, xbound=(0, 2), ybound=(0, 1))
    m.femesh_gmsh(mesh_size_gb=0.4, mesh_size_bulk=0.6, recombine_to_quads=False)
    m.form_elsets_gmsh()
    assert len(m.elsets) == 2
    pts, lines, triangles, quads = m.get_mesh_geometry()
    assert triangles is not None and len(triangles) > 0


def test_abaqus_inp_export(tmp_path):
    m = confMesh2dGMSH()
    m.femesh_gmsh(_two_squares(), mesh_size_gb=0.4, mesh_size_bulk=0.6,
                  recombine_to_quads=False)
    m.form_elsets_gmsh()
    m.build_boundary_nsets()
    out = tmp_path / 'rve.inp'
    written = m.export_abaqus_inp(out, plane='stress')
    text = Path(written).read_text(encoding='utf-8')
    assert '*Node' in text
    assert '*Element, type=CPS3' in text
    assert '*Elset, elset=GRAIN_1' in text
    assert '*Nset, nset=NS_LEFT' in text
    assert '*Solid Section' in text
    fig, ax = m.plot_by_grain()
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_missing_gmsh_message(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'gmsh', None)
    m = confMesh2dGMSH()
    with pytest.raises(ImportError, match='pip install gmsh'):
        m.femesh_gmsh(_two_squares())
