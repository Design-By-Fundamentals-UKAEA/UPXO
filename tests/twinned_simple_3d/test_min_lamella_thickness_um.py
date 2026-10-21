"""Min Lamella Thickness (um): an optional physical-units floor on top
of (never below) TwinGenerator3D's existing voxel-based meshing-route
floor -- see twin_generator_3d.py's _min_hw().
"""
import types

import pytest

from upxo.pxtal.twinned_simple_3d.twin_generator_3d import TwinGenerator3D


def _tg(voxel_size=1.0, meshing_route='conformal', min_thick_c=2, min_thick_nc=1,
        min_lamella_thickness_um=None):
    base = types.SimpleNamespace(voxel_size=voxel_size, host_grain_ids={1, 2})
    return TwinGenerator3D(
        base, meshing_route=meshing_route,
        min_lamella_thickness_conformal=min_thick_c,
        min_lamella_thickness_non_conformal=min_thick_nc,
        min_lamella_thickness_um=min_lamella_thickness_um,
        rng_seed=0)


def test_min_lamella_thickness_um_defaults_to_none():
    tg = _tg()
    assert tg.min_lamella_thickness_um is None


def test_min_hw_falls_back_to_voxel_floor_when_um_floor_is_none():
    tg = _tg(voxel_size=2.0, meshing_route='conformal', min_thick_c=4)
    assert tg._min_hw() == pytest.approx(2.0)   # 4 vox / 2


def test_min_hw_uses_um_floor_when_it_exceeds_the_voxel_floor():
    # voxel floor: 4 vox / 2 = 2.0 hw-vox. um floor: 10 um / 2.0 voxel_size / 2 = 2.5 hw-vox.
    tg = _tg(voxel_size=2.0, meshing_route='conformal', min_thick_c=4,
             min_lamella_thickness_um=10.0)
    assert tg._min_hw() == pytest.approx(2.5)


def test_min_hw_keeps_the_voxel_floor_when_it_exceeds_the_um_floor():
    # voxel floor: 10 vox / 2 = 5.0 hw-vox. um floor: 2 um / 1.0 voxel_size / 2 = 1.0 hw-vox.
    tg = _tg(voxel_size=1.0, meshing_route='conformal', min_thick_c=10,
             min_lamella_thickness_um=2.0)
    assert tg._min_hw() == pytest.approx(5.0)


def test_min_hw_respects_meshing_route_selection_for_the_voxel_floor():
    tg = _tg(voxel_size=1.0, meshing_route='non_conformal',
             min_thick_c=10, min_thick_nc=3, min_lamella_thickness_um=None)
    assert tg._min_hw() == pytest.approx(1.5)   # 3 vox / 2, not the conformal 10
