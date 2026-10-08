import numpy as np
import pytest

from upxo.ggrowth.mcgs_v2 import (
    mcgs_v2, MCGSConfig, _appended_index_arrays_2d)


def _cfg2d(**kw):
    base = dict(dim=2, xmin=0, xmax=29, xinc=1, ymin=0, ymax=19, yinc=1,
                Q=8, mcalg='200', mcsteps=6, save_interval=3,
                consider_boltzmann=False)
    base.update(kw)
    return MCGSConfig(**base)


def _cfg3d(**kw):
    base = dict(xmin=0, xmax=7, xinc=1, ymin=0, ymax=7, yinc=1,
                zmin=0, zmax=7, zinc=1, Q=4, mcalg='300a', mcsteps=4,
                save_interval=2, consider_boltzmann=False)
    base.update(kw)
    return MCGSConfig(**base)


def test_dim_defaults_to_3():
    assert _cfg3d().dim == 3


def test_2d_config_needs_no_z_fields():
    cfg = _cfg2d()
    cfg.validate()
    assert (cfg.zmin, cfg.zmax, cfg.zinc) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("kw", [
    dict(dim=2, mcalg='300a'),
    dict(dim=3, mcalg='200'),
    dict(dim=4),
])
def test_dim_and_algorithm_must_agree(kw):
    cfg = _cfg3d(**kw) if kw.get('dim') != 2 else _cfg2d(**kw)
    with pytest.raises(ValueError):
        cfg.validate()


def test_3d_still_requires_valid_z_axis():
    with pytest.raises(ValueError):
        _cfg3d(zmax=0, zmin=0).validate()


def test_q_related_needs_one_factor_per_state():
    cfg = _cfg2d(consider_boltzmann=True, boltzmann_mode='q_related',
                 boltzmann_temp_factors=[1.0, 2.0])
    with pytest.raises(ValueError):
        cfg.validate()


def test_appended_index_arrays_wrap_on_nonsquare_grid():
    ny, nx = 3, 5
    aia0, aia1 = _appended_index_arrays_2d(ny, nx)
    assert aia0.shape == aia1.shape == (ny + 2, nx + 2)
    # Interior maps to itself.
    rows, cols = np.indices((ny, nx))
    assert (aia0[1:-1, 1:-1] == rows).all()
    assert (aia1[1:-1, 1:-1] == cols).all()
    # Padding wraps: row above 0 is the last row, column left of 0 is the
    # last column, and so on.
    assert (aia0[0, 1:-1] == ny - 1).all()
    assert (aia0[-1, 1:-1] == 0).all()
    assert (aia1[1:-1, 0] == nx - 1).all()
    assert (aia1[1:-1, -1] == 0).all()


@pytest.mark.parametrize("alg", ['200', '201', '202'])
def test_2d_simulation_state_shape_and_range(alg):
    cfg = _cfg2d(mcalg=alg)
    pxt = mcgs_v2(cfg, verbose=False)
    pxt.simulate(verbose=False)
    assert pxt.m[0] == 0
    assert pxt.m == sorted(pxt.gs.keys())
    for t in pxt.m:
        s = pxt.gs[t].s
        assert s.shape == (20, 30)
        assert s.min() >= 1 and s.max() <= cfg.Q


def test_2d_detect_grains_labels_every_pixel():
    pxt = mcgs_v2(_cfg2d(), verbose=False)
    pxt.simulate(verbose=False)
    pxt.detect_grains()
    for t in pxt.m:
        gs = pxt.gs[t]
        assert gs.lgi.shape == gs.s.shape
        assert gs.lgi.min() >= 1
        assert gs.n == np.unique(gs.lgi).size


def test_2d_detect_grains_rejects_unsaved_slice():
    pxt = mcgs_v2(_cfg2d(), verbose=False)
    pxt.simulate(verbose=False)
    with pytest.raises(ValueError):
        pxt.detect_grains(mcsteps=[1])


def test_detect_grains_before_simulate_raises():
    pxt = mcgs_v2(_cfg2d(), verbose=False)
    with pytest.raises(RuntimeError):
        pxt.detect_grains()


def test_detect_grains_not_available_for_3d():
    pxt = mcgs_v2(_cfg3d(), verbose=False)
    with pytest.raises(NotImplementedError):
        pxt.detect_grains()


def test_3d_default_path_state_shape():
    pxt = mcgs_v2(_cfg3d(), verbose=False)
    pxt.simulate(verbose=False)
    assert pxt.dim == 3
    assert pxt.gs[pxt.m[0]].s.shape == (8, 8, 8)


def test_2d_q_related_boltzmann_runs():
    cfg = _cfg2d(Q=4, consider_boltzmann=True, boltzmann_mode='q_related',
                 boltzmann_temp_factors=[0.5, 1.0, 2.0, 4.0])
    pxt = mcgs_v2(cfg, verbose=False)
    pxt.simulate(verbose=False)
    assert len(pxt.m) >= 1
