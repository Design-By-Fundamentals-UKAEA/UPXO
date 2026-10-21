"""rank_temporal_slices_by_n's tslice_keys param -- an explicit list of
pxt.m keys to rank, overriding start/step entirely. Added so a caller can
rank exactly the slices that were cleaned (a sparse, non-prefix set of
slices once temporal slices are ignored), which a positional start/step
range into pxt.m cannot express.
"""
import numpy as np


def _simulate():
    from upxo.ggrowth.mcgs_v2 import mcgs_v2, MCGSConfig
    config = MCGSConfig(xmin=0, xmax=9, xinc=1, ymin=0, ymax=9, yinc=1,
                        zmin=0, zmax=9, zinc=1, Q=4, mcalg='300b', mcsteps=6,
                        save_interval=2, consider_boltzmann=False, rng_seed=1)
    pxt = mcgs_v2(config, verbose=False)
    pxt.simulate(verbose=False)
    return pxt


def test_tslice_keys_ranks_exactly_and_only_those_keys():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    assert len(pxt.m) >= 3
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)

    chosen = [pxt.m[0], pxt.m[2]]   # a sparse, non-contiguous subset
    rows = TwinnedSimple3DBase.rank_temporal_slices_by_n(
        pxt, tslice_keys=chosen, n_comparison_slices=2, verbose=False)
    assert [r['tslice_key'] for r in rows] == chosen


def test_tslice_keys_overrides_start_and_step_when_both_given():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)

    chosen = [pxt.m[-1]]
    rows = TwinnedSimple3DBase.rank_temporal_slices_by_n(
        pxt, start=0, step=1, tslice_keys=chosen,
        n_comparison_slices=2, verbose=False)
    assert [r['tslice_key'] for r in rows] == chosen


def test_omitting_tslice_keys_still_uses_start_and_step_as_before():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)

    rows = TwinnedSimple3DBase.rank_temporal_slices_by_n(
        pxt, start=0, step=1, n_comparison_slices=2, verbose=False)
    assert [r['tslice_key'] for r in rows] == list(pxt.m)


def test_tslice_keys_duplicate_rows_are_allowed_and_preserve_order():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    TwinnedSimple3DBase.calculate_lfi(pxt, verbose=False)

    chosen = [pxt.m[1], pxt.m[0], pxt.m[1]]
    rows = TwinnedSimple3DBase.rank_temporal_slices_by_n(
        pxt, tslice_keys=chosen, n_comparison_slices=2, verbose=False)
    assert [r['tslice_key'] for r in rows] == chosen


def test_tslice_keys_still_requires_calculate_lfi_to_have_run():
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    pxt = _simulate()
    import pytest
    with pytest.raises(RuntimeError, match="calculate_lfi"):
        TwinnedSimple3DBase.rank_temporal_slices_by_n(
            pxt, tslice_keys=[pxt.m[0]], n_comparison_slices=2, verbose=False)
