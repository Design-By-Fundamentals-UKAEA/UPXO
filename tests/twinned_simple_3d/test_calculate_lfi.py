"""calculate_lfi() skips find_grain_voxel_locs/find_spatial_bounds_of_grains,
make_pvgrid and set_mprops -- nothing in this pipeline reads what any of
them produce (see base_3d.py's calculate_lfi comment), so they must stay
off, and .lgi/n_grains must stay correct without them.
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


def test_calculate_lfi_does_not_compute_voxel_locs_or_spatial_bounds():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)
    for tslice_key in pxt.m:
        gstslice = pxt.gs[tslice_key]
        assert not getattr(gstslice, 'grain_locs', None)
        assert getattr(gstslice, 'spbound', None) is None
        assert getattr(gstslice, 'spboundex', None) is None


def test_calculate_lfi_does_not_build_pvgrid_or_set_mprops():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)
    for tslice_key in pxt.m:
        gstslice = pxt.gs[tslice_key]
        assert getattr(gstslice, 'pvgrid', None) is None
        assert gstslice.mprop.get('volnv') is None


def test_calculate_lfi_still_populates_lgi_and_reports_correct_n_grains():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    summary = TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)
    assert set(summary) == set(pxt.m)
    for tslice_key in pxt.m:
        gstslice = pxt.gs[tslice_key]
        assert gstslice.lgi is not None and gstslice.lgi.shape == gstslice.s.shape
        true_n = int(np.unique(gstslice.lgi[gstslice.lgi > 0]).size)
        assert summary[tslice_key]['n_grains'] == true_n
        assert true_n > 0


def test_calculate_lfi_matches_requesting_the_skipped_extras_explicitly():
    """Same .lgi/grain count whether or not the (now-skipped) extras are
    computed -- confirms the skip changes no observable result, just cost.

    Compares calculate_lfi()'s result against a second, explicit
    char_morphology_of_grains() call with the extras turned back on, on the
    SAME already-simulated slices (re-labelling .s is a pure function of
    it) -- not against a second independent simulation: mcgs_v2 is not
    reproducible run to run even with the same rng_seed (known project
    constraint), so two separate _simulate() calls cannot be compared."""
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    summary = TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)
    lgi_before = {k: pxt.gs[k].lgi.copy() for k in pxt.m}

    for tslice_key in pxt.m:
        pxt.gs[tslice_key].char_morphology_of_grains(
            label_str_order=1, find_grain_voxel_locs=True,
            find_spatial_bounds_of_grains=True, force_compute=True)

    for tslice_key in pxt.m:
        lgi_after = pxt.gs[tslice_key].lgi
        assert np.array_equal(lgi_before[tslice_key], lgi_after)
        assert summary[tslice_key]['n_grains'] == int(np.unique(lgi_after[lgi_after > 0]).size)
