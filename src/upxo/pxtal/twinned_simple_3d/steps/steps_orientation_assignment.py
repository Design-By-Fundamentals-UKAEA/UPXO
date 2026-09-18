"""Orientation Assignment -- Part J of the Twinned FCC walkthrough.

Assigns a crystal orientation (quaternion) to every grain in the
host-allocated base structure, using EBSD-derived pools (parent-grain
quaternions for hosts, a fallback pool -- random by default -- for
non-hosts), resolving twin-pair conflicts per the selected mode.
optimize_mapping() below runs many mode/seed combinations and scores each
against a real EBSD pole-figure reference, keeping the best-matching
assignment instead of a single, unscored run.
"""


def assign_orientations(base, rg, parent_info, csl_label="S3  (twin)",
                         ori_mode='conflict_free', rng_seed=42, connectivity=6,
                         n_fallback=500, fallback_tc_info=None, fallback_apply_symmetry=None,
                         mdf_n_bins=65, mdf_angle_max=65.0, pair_similarity_deg=10.0,
                         max_retries=50, mrf_max_sweeps=20, mrf_eps_convergence=0.3,
                         mrf_eps_quality=1.5, mrf_init_mode='mdf_analytical',
                         mrf_kde_threshold=None, mrf_bilateral_symmetry=False,
                         mrf_sa_t_start=5.0, mrf_sa_t_end=0.05):
    """Runs one full orientation assignment. `ori_mode='conflict_free'`
    is the simplest/fastest mode -- the mrf_*/pair_similarity_deg/
    max_retries parameters only matter for the other modes
    ('paired_pool', 'mdf_conditioned_pairs', 'mdf_analytical', 'mrf_gibbs',
    'mrf_map'), but are always accepted regardless of which mode is
    active.

    Returns
    -------
    OrientationAssigner3D : `assigner` -- assigner.n_conflicts and its
    assigned quaternions/grain-orientation maps are the results; pass
    this straight into steps_twin_generation's functions.
    """
    from upxo.pxtal.twinned_simple_3d.orientation_3d import run_orientation_assignment
    return run_orientation_assignment(
        base=base, rg=rg, parent_info=parent_info, csl_label=csl_label,
        ori_mode=ori_mode, rng_seed=rng_seed, connectivity=connectivity,
        n_fallback=n_fallback, fallback_tc_info=fallback_tc_info,
        fallback_apply_symmetry=fallback_apply_symmetry,
        mdf_n_bins=mdf_n_bins, mdf_angle_max=mdf_angle_max,
        pair_similarity_deg=pair_similarity_deg, max_retries=max_retries,
        mrf_max_sweeps=mrf_max_sweeps, mrf_eps_convergence=mrf_eps_convergence,
        mrf_eps_quality=mrf_eps_quality, mrf_init_mode=mrf_init_mode,
        mrf_kde_threshold=mrf_kde_threshold, mrf_bilateral_symmetry=mrf_bilateral_symmetry,
        mrf_sa_t_start=mrf_sa_t_start, mrf_sa_t_end=mrf_sa_t_end,
        cancel_event=None, prebuilt_pools=None)


def optimize_mapping(base, rg, parent_info, csl_label="S3  (twin)",
                      selected=(('conflict_free', 5), ('paired_pool', 5)),
                      base_seed=42, pole_family='100', resolution='Medium', unit_normalize=True,
                      connectivity=6, n_fallback=500, fallback_tc_info=None, fallback_apply_symmetry=None,
                      mdf_n_bins=65, mdf_angle_max=65.0, pair_similarity_deg=10.0, max_retries=50,
                      mrf_max_sweeps=20, mrf_eps_convergence=0.3, mrf_eps_quality=1.5,
                      mrf_init_mode='mdf_analytical', mrf_kde_threshold=None, mrf_bilateral_symmetry=False,
                      mrf_sa_t_start=5.0, mrf_sa_t_end=0.05):
    """Sweeps orientation-assignment mode/seed combinations, scoring each
    run's synthetic host-grain pole figure against the real EBSD "pure
    parents" reference for `csl_label` (grains that, in the EBSD data,
    always host this CSL type's twin and are never one themselves), via
    the interquartile range (IQR) of their cell-by-cell MUD-grid
    difference -- smaller IQR means a tighter match. `unit_normalize=True`
    (default) compares pattern shape rather than absolute texture
    strength.

    `selected`: an iterable of (mode, n_iterations) pairs. Available
    modes: 'conflict_free', 'paired_pool', 'mdf_conditioned_pairs',
    'mdf_analytical', 'mrf_gibbs', 'mrf_map' -- the last two are
    substantially slower and excluded from the default selection.

    Every successful run is scored; the 10 lowest-IQR runs (across every
    swept mode) are re-run once more so their full OrientationAssigner3D
    objects are available for inspection -- cheap relative to the full
    sweep, and avoids holding every single run's assigner in memory at
    once.

    Returns
    -------
    (records, top10) : records is every run's score dict ('mode', 'seed',
    'iqr', 'min', 'max', 'q25', 'q75', 'n_host'); top10 is the 10
    lowest-IQR runs, each {**record, 'assigner': OrientationAssigner3D},
    ranked ascending (index 0 = best match) -- top10[0]['assigner'] is
    the recommended assignment to carry into Twin Generation.
    """
    from upxo.pxtal.twinned_simple_3d.orientation_3d import run_optimize_sweep
    from upxo.viz.xphy.pole_figure import PoleFigure

    resolution_fraction = {'Low': 0.25, 'Medium': 0.5, 'High': 1.0}[resolution]
    grid_points = max(int(PoleFigure.auto_grid_points(7.5) * resolution_fraction), 20)
    return run_optimize_sweep(
        base=base, rg=rg, parent_info=parent_info, csl_label=csl_label, selected=list(selected),
        connectivity=connectivity, n_fallback=n_fallback, fallback_tc_info=fallback_tc_info,
        fallback_apply_symmetry=fallback_apply_symmetry,
        mdf_n_bins=mdf_n_bins, mdf_angle_max=mdf_angle_max,
        pair_similarity_deg=pair_similarity_deg, max_retries=max_retries,
        mrf_max_sweeps=mrf_max_sweeps, mrf_eps_convergence=mrf_eps_convergence,
        mrf_eps_quality=mrf_eps_quality, mrf_init_mode=mrf_init_mode,
        mrf_kde_threshold=mrf_kde_threshold, mrf_bilateral_symmetry=mrf_bilateral_symmetry,
        mrf_sa_t_start=mrf_sa_t_start, mrf_sa_t_end=mrf_sa_t_end,
        pole_family=pole_family, grid_points=grid_points, unit_normalize=unit_normalize,
        base_seed=base_seed, cancel_event=None)


def plot_optimize_mapping_distribution(records):
    """Boxplot of IQR by mode across every run in `records` (from
    optimize_mapping()) -- lower IQR means a tighter EBSD match; compares
    modes and run-to-run variability within a mode at a glance.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    by_mode = {}
    for r in records:
        by_mode.setdefault(r['mode'], []).append(r['iqr'])
    modes = list(by_mode.keys())
    data = [by_mode[m] for m in modes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=modes)
    ax.set_ylabel('IQR (lower = better match)')
    ax.set_title(f'Optimize Mapping -- IQR distribution by mode ({len(records)} runs)')
    return fig, ax
