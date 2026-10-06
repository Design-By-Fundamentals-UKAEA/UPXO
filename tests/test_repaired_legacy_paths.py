"""Paths that failed with an import error through 1.3.0 and now work, and the
ones that now say plainly that they are unavailable."""
import numpy as np
import pytest
from scipy import ndimage


def _fm_structure():
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    rng = np.random.default_rng(0)
    seeds = np.zeros((14, 14, 14), int)
    for i, (x, y, z) in enumerate(rng.integers(0, 14, size=(30, 3)), 1):
        seeds[x, y, z] = i
    _, idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    return FMSteel3DBase.from_lfi(seeds[tuple(idx)], physical_dimensions=(14., 14., 14.), voxel_size=1.0, random_seed=1)


def test_transformation_steps_use_the_shipped_helpers():
    from upxo.pxtal.fm_steel_3d.steps.steps_transformations import apply_transform, sweep_clean
    base = _fm_structure()
    assert apply_transform(base, scale_factor=1.0).lgi.shape == base.lgi.shape
    out = sweep_clean(base, factor=2.0, cleanup_threshold=0)
    assert out.lgi.shape == base.lgi.shape
    ids = np.unique(out.lgi)
    ids = ids[ids != 0]
    assert ids.min() == 1 and ids.max() == len(ids)             # contiguous grain IDs after relabelling


def test_relabel_and_randomize_is_contiguous():
    from upxo.pxtal.fm_steel_3d.transform_shared import relabel_and_randomize
    lgi = np.array([[[1, 1, 4], [4, 7, 7], [7, 9, 9]]])
    out = relabel_and_randomize(lgi)
    assert sorted(np.unique(out)) == [1, 2, 3, 4]
    assert out.shape == lgi.shape


def test_parameter_sweep_creates_one_mcgs_per_instance():
    from upxo.parswep.mcgs2d_parameter_sweeping import parameter_sweep
    sweep = parameter_sweep.__new__(parameter_sweep)
    sweep.initialize(2)
    assert sorted(sweep.gsi) == [1, 2]
    assert all(g.study == 'para_sweep' for g in sweep.gsi.values())


def test_random_upxo_points():
    from upxo._sup.dataTypeHandlers import make_upxo_point2d_RANDU
    points = make_upxo_point2d_RANDU(4)
    assert len(points) == 4 and all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in points)


def test_removed_apis_say_so():
    from upxo._sup import gops, dataTypeHandlers
    from upxo.heirGs import pdomain_2
    with pytest.raises(NotImplementedError, match='no longer exists'):
        gops.PROFILE_up2d_INST(1, 'ignore')
    with pytest.raises(NotImplementedError, match='not available'):
        dataTypeHandlers.coords_to_UpxoPointList(coords=[[0, 0]], dim=2, coords_format='locxy', lean='no')
    with pytest.raises(NotImplementedError, match='no longer exists'):
        pdomain_2.make_PDI()


def test_importing_the_repaired_modules_needs_no_legacy_packages():
    import sys
    import upxo.heirGs.pdomain_2, upxo.pxtal.polyxtal, upxo.geoEntities.edge2d, upxo.geoEntities.pops  # noqa: F401
    for name in ('point2d', 'point2d_04', 'vt', 'distr_01', 'mulpoint2d', 'eops', 'upxo_math', 'tabulate'):
        assert name not in sys.modules, name
