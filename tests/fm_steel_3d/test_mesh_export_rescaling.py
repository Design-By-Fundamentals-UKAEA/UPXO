"""Regression tests for non-cumulative mesh-export rescaling."""

from types import SimpleNamespace

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.mesh_rescaling import (
    RescaledMeshState, mesh_state_for_scale, scaled_grid_shape)


def _base_state():
    lgi = np.ones((4, 4, 4), dtype=int)
    lgi[2:, :, :] = 2
    grain_locs = {gid: np.argwhere(lgi == gid) for gid in (1, 2)}
    all_blocks = {
        "b1": grain_locs[1].copy(),
        "b2": grain_locs[2].copy(),
    }
    all_subblocks = {
        "sb1": grain_locs[1].copy(),
        "sb2": grain_locs[2].copy(),
    }
    return SimpleNamespace(
        lgi=lgi,
        voxel_size=2.0,
        grain_locs=grain_locs,
        all_blocks=all_blocks,
        all_subblocks=all_subblocks,
        units="microns",
    )


def test_mesh_export_rescaling_is_always_derived_from_base_state():
    base = _base_state()

    doubled = mesh_state_for_scale(base, 2.0)
    halved = mesh_state_for_scale(base, 0.5)

    assert base.lgi.shape == (4, 4, 4)
    assert base.voxel_size == 2.0
    assert doubled.lgi.shape == (8, 8, 8)
    assert doubled.voxel_size == 1.0
    assert halved.lgi.shape == (2, 2, 2)
    assert halved.voxel_size == 4.0

    assert sum(len(v) for v in doubled.grain_locs.values()) == 8 ** 3
    assert sum(len(v) for v in doubled.all_blocks.values()) == 8 ** 3
    assert sum(len(v) for v in doubled.all_subblocks.values()) == 8 ** 3
    assert sum(len(v) for v in halved.grain_locs.values()) == 2 ** 3


def test_scale_one_reuses_the_unchanged_base_state():
    base = _base_state()
    assert mesh_state_for_scale(base, 1.0) is base


def test_scale_one_does_not_look_at_the_state():
    state = object()
    assert mesh_state_for_scale(state, 1) is state


def test_scale_that_collapses_an_axis_is_rejected():
    base = _base_state()
    with pytest.raises(ValueError, match="at least one voxel"):
        mesh_state_for_scale(base, 0.01)


@pytest.mark.parametrize("factor", [0, -1.0, float("nan"), float("inf")])
def test_scale_must_be_finite_and_positive(factor):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        mesh_state_for_scale(_base_state(), factor)
    with pytest.raises(ValueError, match="finite number greater than zero"):
        scaled_grid_shape((4, 4, 4), factor)


def test_scaled_grid_shape_rounds_each_axis():
    assert scaled_grid_shape((4, 6, 10), 1.5) == (6, 9, 15)
    assert scaled_grid_shape((4, 6, 10), 1) == (4, 6, 10)


def test_rescaled_state_reads_other_attributes_from_the_source():
    base = _base_state()
    base.pag_orientations = {1: (1.0, 2.0, 3.0)}
    view = mesh_state_for_scale(base, 2.0)
    assert isinstance(view, RescaledMeshState)
    assert view.pag_orientations is base.pag_orientations
    assert view.units == "microns"
    assert view.scale_factor == 2.0


def test_rescaled_locations_index_the_rescaled_grid():
    base = _base_state()
    view = mesh_state_for_scale(base, 2.0)
    for gid, voxels in view.grain_locs.items():
        assert (view.lgi[tuple(voxels.T)] == gid).all()
    for blocks in (view.all_blocks, view.all_subblocks):
        every = np.concatenate(list(blocks.values()))
        assert len(np.unique(every, axis=0)) == len(every)
