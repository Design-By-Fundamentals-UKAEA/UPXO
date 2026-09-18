"""Base Grain Structure -- Part E of the Twinned FCC walkthrough.

Method selection is trivial on this path (Monte-Carlo is the base grain
structure method this walkthrough covers -- Voronoi/Image-Import are
different, unworked paths). Runs the 3D MC grain-growth simulation,
computes the labelled feature index (LFI) for every saved temporal slice,
then cleans it.
"""


def run_mc_simulation(xmax=50.0, ymax=50.0, zmax=50.0, xinc=1.0, yinc=1.0, zinc=1.0,
                       q_states=10, mcsteps=100, save_interval=5, mcalg='300b',
                       consider_boltzmann=True, boltzmann_temp_factor=0.01, rng_seed=0,
                       verbose=True):
    """Step 1 -- runs a 3D Potts-model Monte-Carlo grain growth
    simulation. Domain runs from 0 to the given max in each direction.

    Returns
    -------
    mcgsV1_1 : the simulated object -- pxt.m holds the list of saved
    temporal-slice step indices, pxt.gs[t] the grain structure at step t.
    """
    from upxo.ggrowth.mcgsV1_1 import mcgsV1_1, MCGSConfig
    config = MCGSConfig(
        xmin=0.0, xmax=xmax, xinc=xinc, ymin=0.0, ymax=ymax, yinc=yinc,
        zmin=0.0, zmax=zmax, zinc=zinc, Q=q_states, mcalg=mcalg,
        mcsteps=mcsteps, save_interval=save_interval,
        consider_boltzmann=consider_boltzmann, boltzmann_mode='q_unrelated',
        boltzmann_temp_factor=boltzmann_temp_factor if consider_boltzmann else None,
        boltzmann_temp_factors=None, rng_seed=rng_seed or None,
    )
    pxt = mcgsV1_1(config, verbose=verbose)
    pxt.simulate(verbose=verbose)
    return pxt


def calculate_lfi(pxt, verbose=True):
    """Step 2 -- computes the labelled feature index (grain labels)
    for every saved temporal slice at once; cleaning (below) then
    operates on this persisted LFI.

    Returns
    -------
    dict : {tslice_key: {'n_grains': int, ...}}
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    return TwinnedSimple3DBase.calculate_lfi(pxt, verbose=verbose)


def clean_structure(pxt, min_grain_size=4, start_index=1, n_passes=5,
                     do_merge_small=True, do_spike_removal=True, verbose=True):
    """Step 3 -- merges too-small grains and removes single-voxel
    spikes, recursively, for every saved slice from `start_index` on
    (slices before it are left uncleaned/unavailable to ranking).
    `n_passes` is capped at 20 internally as a safety limit.

    Returns
    -------
    (cumulative_summary, n_passes_run)
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    tslice_keys = pxt.m[start_index:]
    return TwinnedSimple3DBase.clean_temporal_slices_recursive(
        pxt, min_grain_size=min_grain_size, tslice_keys=tslice_keys,
        do_merge_small=do_merge_small, do_spike_removal=do_spike_removal,
        n_passes=n_passes, verbose=verbose)
