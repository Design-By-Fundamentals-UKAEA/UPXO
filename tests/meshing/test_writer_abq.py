"""
Unit tests for upxo.meshing.writer_ABQ.export_confmesh2d_inp's material
sections: the pre-existing dummy-isotropic path and the new
``material_format='bunge_euler'`` per-grain Bunge-Euler path.

Pure numpy/pathlib -- no gmsh, no meshing pipeline, no external FE tool.

Run from repo root:
    pytest tests/meshing/test_writer_abq.py -v
"""

import numpy as np
import pytest

from upxo.meshing.writer_ABQ import export_confmesh2d_inp


def _two_triangle_mesh():
    """Two disjoint triangles, one grain each."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    elConn = {'triangle': np.array([[0, 1, 2], [1, 3, 4]])}
    elsets_eltype = {'triangle': {'grain.1': np.array([0]),
                                  'grain.2': np.array([1])}}
    return nodes, elConn, elsets_eltype


def test_isotropic_default_unchanged(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'iso.inp'

    export_confmesh2d_inp(path, nodes, elConn, elsets_eltype)

    text = path.read_text(encoding='utf-8')
    assert text.count('*Material,') == 2
    assert '*Elastic\n210000.0, 0.3\n' in text
    assert '*User Material' not in text
    assert text.count('*Solid Section') == 2


def test_isotropic_custom_elastic_constants(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'iso_custom.inp'

    export_confmesh2d_inp(path, nodes, elConn, elsets_eltype,
                          elastic_constants=(70000.0, 0.33))

    text = path.read_text(encoding='utf-8')
    assert '*Elastic\n70000.0, 0.33\n' in text


def test_bunge_euler_writes_one_material_per_grain(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'bunge.inp'
    grain_euler_deg = {1: (10.0, 20.0, 30.0), 2: (40.0, 50.0, 60.0)}

    export_confmesh2d_inp(path, nodes, elConn, elsets_eltype,
                          material_format='bunge_euler',
                          grain_euler_deg=grain_euler_deg)

    text = path.read_text(encoding='utf-8')
    assert text.count('*Material,') == 2
    assert text.count('*User Material, constants=3') == 2
    assert text.count('*Depvar') == 2
    assert '10.000000, 20.000000, 30.000000' in text
    assert '40.000000, 50.000000, 60.000000' in text
    assert 'MAT_GRAIN_1' in text and 'MAT_GRAIN_2' in text
    assert '*Solid Section, elset=GRAIN_1, material=MAT_GRAIN_1' in text
    assert '*Elastic' not in text


def test_bunge_euler_respects_n_depvar(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'bunge_depvar.inp'

    export_confmesh2d_inp(path, nodes, elConn, elsets_eltype,
                          material_format='bunge_euler',
                          grain_euler_deg={1: (0.0, 0.0, 0.0),
                                          2: (0.0, 0.0, 0.0)},
                          n_depvar=5)

    text = path.read_text(encoding='utf-8')
    assert '*Depvar\n5,\n' in text


def test_bunge_euler_requires_grain_euler_deg(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'bunge_missing.inp'

    with pytest.raises(ValueError):
        export_confmesh2d_inp(path, nodes, elConn, elsets_eltype,
                              material_format='bunge_euler')


def test_no_sections_when_write_sections_false(tmp_path):
    nodes, elConn, elsets_eltype = _two_triangle_mesh()
    path = tmp_path / 'no_sections.inp'

    export_confmesh2d_inp(path, nodes, elConn, elsets_eltype,
                          write_sections=False)

    text = path.read_text(encoding='utf-8')
    assert '*Material' not in text
    assert '*Solid Section' not in text
