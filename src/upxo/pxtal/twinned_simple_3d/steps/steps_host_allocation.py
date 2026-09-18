"""Host Allocation -- Part H of the Twinned FCC walkthrough.

Allocates which grains in the base structure become twin "hosts" (a
Maximal-Independent-Set-constrained spatial selection targeting a
physical volume fraction), then re-assesses the result via 2D
cross-sections for comparability with the EBSD 2D target.
"""


def allocate_hosts(pxt, tslice_key, transformed_base=None,
                    target_fraction=0.3, min_voxels=4, scale_factor=1.65,
                    ranking_weight=1.0, mis_fraction=0.25, mis_runs=10, seed=0,
                    pool_b_size_measure='vol_vox', pool_b_band_method='std',
                    pool_b_band_shape='above', pool_b_n_std=1.0,
                    pool_b_percentile_lo=25.0, pool_b_percentile_hi=75.0,
                    pool_b_neighbour_frac=50.0):
    """Builds (or reuses a Transformations-modified) base structure,
    characterizes its morphology, then runs the MIS-constrained spatial
    host allocation.

    `transformed_base`: pass the result of
    steps_transformations.apply_transform() to allocate on a
    rescaled/stretched structure instead of the raw MC slice; leave as
    None to use the raw slice directly.

    Returns
    -------
    TwinnedSimple3DBase : `base`, now carrying the allocation result
    (host/non-host grain roles) -- pass this straight into
    reassess_hosting_2d() and later steps_pre_twin_validation.
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase

    base = transformed_base if transformed_base is not None else \
        TwinnedSimple3DBase.from_mcgs(pxt, int(tslice_key))
    base.char_morphology(volnv=True, eqdia=True)
    base.allocate_twin_hosts_spatial(
        target_hosting_fraction=target_fraction,
        min_host_voxels=min_voxels,
        host_fraction_2d_to_3d_scale_factor=scale_factor,
        host_ranking_volume_weight=ranking_weight,
        mis_fraction=mis_fraction,
        pool_b_size_measure=pool_b_size_measure,
        pool_b_band_method=pool_b_band_method,
        pool_b_band_shape=pool_b_band_shape,
        pool_b_n_std=pool_b_n_std,
        pool_b_percentile_lo=pool_b_percentile_lo,
        pool_b_percentile_hi=pool_b_percentile_hi,
        neighbour_frac=pool_b_neighbour_frac,
        mis_runs=mis_runs,
        seed=seed,
    )
    return base


def reassess_hosting_2d(base, axes=('x', 'y', 'z'), n_per_axis=5, connectivity=4):
    """2D Slice-Based Assessment -- re-checks the 3D allocation
    result via 2D cross-sections (comparable to the EBSD 2D target), NOT
    a re-run of the allocation itself.

    Returns
    -------
    dict : {'count_ratio_mean', 'count_ratio_std', 'area_ratio_mean',
    'area_ratio_std', 'n_slices_used', ...}
    """
    n_per_axis_map = {a: n_per_axis for a in axes}
    return base.assess_hosting_representativeness_2d(
        n_comparison_slices=n_per_axis_map, comparison_axes=list(axes),
        connectivity_2d=connectivity)
