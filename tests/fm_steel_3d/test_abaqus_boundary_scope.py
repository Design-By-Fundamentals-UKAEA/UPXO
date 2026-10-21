"""Regression checks for Abaqus assembly-level boundary sets."""

from io import StringIO

import numpy as np

from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import MeshExporter3D
from upxo.pxtal.fm_steel_3d.deck_export import ConfigurableMeshExporter3D


def _assert_named_boundary_deck(exporter, tmp_path, expected_zmax):
    node_sets = StringIO()
    exporter._write_nsets_bc(node_sets, 1, 1, 1, np.ones((2, 2, 2), dtype=bool))
    steps = StringIO()
    exporter._write_steps(steps, exporter._boundary_nset_config())
    exporter._write_master(tmp_path, 'C3D8', False, '', 'microns',
                           'microns', 1.0, 1, 1, 1, 1)
    master = (tmp_path / 'model_master.inp').read_text()

    assert '*Part, name=FM_STEEL_RVE\n' in master
    assert '*PREPRINT' not in master
    assert '*Assembly, name=Assembly\n' in master
    assert master.index('*End Part') < master.index('*Assembly')
    assert master.index('*Assembly') < master.index('*INCLUDE, INPUT=04_nsets_bc.inp') < master.index('*End Assembly')
    assert f'*Nset, nset={expected_zmax}, instance=FM_STEEL_RVE-1\n' in node_sets.getvalue()
    assert f'{expected_zmax}, 3, 3, 0.2\n' in steps.getvalue()
    assert '*Boundary\n1, 3, 3' not in steps.getvalue()


def test_library_boundary_sets_are_assembly_level(tmp_path):
    _assert_named_boundary_deck(MeshExporter3D(), tmp_path, 'ns_zplus')


def test_configured_prefixes_are_used_by_sets_and_step(tmp_path):
    exporter = ConfigurableMeshExporter3D({
        'prefix_z_max': 'my_loaded_face',
        'prefix_x_min': 'my_x_support',
        'nset_z_max': False,  # The analysis step still needs this set.
    })
    _assert_named_boundary_deck(exporter, tmp_path, 'my_loaded_face')
