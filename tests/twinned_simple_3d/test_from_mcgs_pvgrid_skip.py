"""TwinnedSimple3DBase.from_mcgs()'s fallback char_morphology_of_grains()
call (used when .lgi hasn't been populated yet, e.g. calculate_lfi() was
never run) skips make_pvgrid/set_mprops for the same reason calculate_lfi()
does -- nothing ever reads gstslice.pvgrid or gstslice.mprop['volnv']; the
returned TwinnedSimple3DBase has its own separate, lazily-populated mprop.
find_grain_voxel_locs/find_spatial_bounds_of_grains stay on, unlike
calculate_lfi()'s call -- this path is used where those ARE needed.
"""
import numpy as np


def _simulate():
    from upxo.ggrowth.mcgs_v2 import mcgs_v2, MCGSConfig
    config = MCGSConfig(xmin=0, xmax=9, xinc=1, ymin=0, ymax=9, yinc=1,
                        zmin=0, zmax=9, zinc=1, Q=4, mcalg='300b', mcsteps=4,
                        save_interval=2, consider_boltzmann=False, rng_seed=1)
    pxt = mcgs_v2(config, verbose=False)
    pxt.simulate(verbose=False)
    return pxt


def test_from_mcgs_fallback_does_not_build_pvgrid_or_set_mprops():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    tslice_key = pxt.m[0]
    gstslice = pxt.gs[tslice_key]
    assert not hasattr(gstslice, 'lgi')   # calculate_lfi() deliberately never ran

    base = TwinnedSimple3DBase.from_mcgs(pxt, tslice_key)

    assert gstslice.pvgrid is None
    assert gstslice.mprop.get('volnv') is None
    assert base.lgi is not None and base.lgi.shape == gstslice.s.shape


def test_from_mcgs_fallback_still_computes_voxel_locs_and_spatial_bounds():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    tslice_key = pxt.m[0]
    gstslice = pxt.gs[tslice_key]

    TwinnedSimple3DBase.from_mcgs(pxt, tslice_key)

    assert gstslice.grain_locs   # non-empty -- find_grain_voxel_locs=True unchanged
    assert gstslice.spbound is not None
    assert gstslice.spboundex is not None


def test_from_mcgs_skips_the_fallback_entirely_when_lgi_is_already_populated():
    """If calculate_lfi() already ran, from_mcgs() must not recompute
    anything -- it just reads the existing .lgi."""
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    tslice_key = pxt.m[0]
    TwinnedSimple3DBase.calculate_lfi(pxt, tslice_keys=[tslice_key], verbose=False)
    gstslice = pxt.gs[tslice_key]
    lgi_before = gstslice.lgi.copy()

    base = TwinnedSimple3DBase.from_mcgs(pxt, tslice_key)

    assert np.array_equal(gstslice.lgi, lgi_before)
    assert gstslice.pvgrid is None   # calculate_lfi() already skipped it too
    assert np.array_equal(base.lgi, lgi_before)
