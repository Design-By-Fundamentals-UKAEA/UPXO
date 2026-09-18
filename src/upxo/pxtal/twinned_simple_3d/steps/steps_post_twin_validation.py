"""Post-Twin Validation -- Part L of the Twinned FCC walkthrough.

Builds the post-twin voxel quaternion field, validates the CLEANED,
twinned structure's misorientation distribution against the full EBSD
MDF (twins present -- unlike Pre-Twin Validation, which deliberately
used the twins-merged-out reference), per axis, then compares the full
3D MDF curve directly.
"""


def validate_crystallographic_representativeness(
        cleaner, mdf, n_slices=(10, 10, 10), test_axes=('x', 'y', 'z'),
        p_percentage=60.0, wasserstein_threshold=5.0):
    """Builds the post-twin voxel quaternion field (no parameters of its
    own -- always freshly derived from the cleaned structure) then
    validates it against the EBSD full MDF.

    Returns
    -------
    (validator, quat_3d_clean) : `validator.axis_acceptance`/
    `validator.overall_accepted`/`validator.report()` are the results;
    `quat_3d_clean` feeds compare_full_3d_mdf() below.
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.repr_validator_3d import RepresentativenessValidator3D
    from upxo.xtalphy.crystal_orientation import expand_grain_quats_to_voxels

    quat_3d_clean = expand_grain_quats_to_voxels(cleaner.lgi_clean, cleaner.all_quats_clean)

    n_x, n_y, n_z = n_slices
    validator = RepresentativenessValidator3D(
        n_slices_x=n_x, n_slices_y=n_y, n_slices_z=n_z,
        test_along_x='x' in test_axes, test_along_y='y' in test_axes,
        test_along_z='z' in test_axes,
        p_percentage=p_percentage, wasserstein_threshold=wasserstein_threshold,
    )
    validator.validate_crystallographic(cleaner.lgi_clean, quat_3d_clean, mdf['miso_deg'])
    return validator, quat_3d_clean


def compare_full_3d_mdf(cleaner, quat_3d_clean, n_bins=65, angle_max=65.0):
    """Computes the post-twin structure's full 3D misorientation
    distribution (every grain-grain neighbour pair, not
    just the slice sample validate_crystallographic_representativeness
    used), directly comparable to the EBSD full MDF curve computed back
    in EBSD Analysis-2.

    Returns
    -------
    dict : mc_mdf_post, same shape as steps_ebsd_analysis_2's MDF result
    (hist_bin_centers, hist_density, mean_angle, ...).
    """
    import numpy as np
    from upxo.gsdataops.gid_ops import find_neighs3d
    from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

    neigh_raw = find_neighs3d(cleaner.lgi_clean.astype(np.int32), conn=6)
    neigh_list = {int(g): list(ns) for g, ns in neigh_raw.items()}
    return compute_mdf_from_quats(
        cleaner.lgi_clean, quat_3d_clean, neigh_list, n_bins=n_bins,
        angle_range=(0.0, angle_max))


def representative_slice_mdfs(validator, cleaner, quat_3d_clean, axes=None):
    """Per-axis MDF for every slice `validator` already found passing
    (during validate_crystallographic_representativeness above), plus
    their mean -- the representative-slice MDF a passing 2D section
    should reproduce, directly comparable to the full EBSD MDF curve.

    Returns
    -------
    dict {axis_name: {'n_passing', 'slice_mdfs', 'mean_bin_centers',
    'mean_density'}} -- one entry per axis with at least one slice
    result recorded.
    """
    from upxo.pxtal.twinned_simple_3d.repr_validator_3d import compute_representative_slice_mdfs
    return compute_representative_slice_mdfs(validator, cleaner, quat_3d_clean, axes=axes)


def twin_thickness_three_way_comparison(tg, cleaner, twin_thickness, validator):
    """Three-way twin-thickness population comparison: the EBSD target,
    the actual 3D thickness as introduced (from tg.twin_halfwidths_vox),
    and the apparent 2D thickness measured on the validator's own
    representative slices, per axis -- shows how much a 2D-slice
    measurement understates or overstates the true 3D lamella thickness.

    Returns
    -------
    dict {'ebsd_um', 'ebsd_mean_um', 'actual_3d_um', 'actual_3d_mean_um',
    'per_axis_um', 'per_axis_means_um'}.
    """
    from upxo.pxtal.twinned_simple_3d.twin_generator_3d import compute_twin_thickness_comparison
    return compute_twin_thickness_comparison(tg, cleaner, twin_thickness, validator)


def twin_thickness_representativeness_score(comparison):
    """A single [0, 1] "how well does the actual 3D twin thickness
    introduced during generation match the EBSD target" score, from
    twin_thickness_three_way_comparison()'s output. Compares 'ebsd_um'
    against 'actual_3d_um' -- the per-axis apparent 2D populations are
    left out, since they illustrate how much a 2D slice understates the
    true 3D thickness rather than representing a separate target to be
    matched -- via TwinnedSimple3DBase.compare_property_distributions,
    the same generic distribution comparator
    mdf_representativeness_score() applies to the misorientation
    distribution.

    Returns
    -------
    dict or None : see compare_property_distributions -- None if either
    side has no values left after outlier trimming.
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    return TwinnedSimple3DBase.compare_property_distributions(
        comparison['ebsd_um'], comparison['actual_3d_um'])


def mdf_representativeness_score(mdf, mc_mdf_post):
    """A single [0, 1] "how well does the achieved neighbour-grain
    misorientation distribution match EBSD" score -- distinct from
    validate_crystallographic_representativeness's per-axis pass/fail,
    which only says whether individual 2D slices cleared a threshold,
    not how close the OVERALL match is. Compares the full pooled angle
    populations (mdf['miso_deg'] vs mc_mdf_post['miso_deg'], both raw
    per-neighbour-pair arrays, not the binned histograms) via
    TwinnedSimple3DBase.compare_property_distributions -- the same
    generic distribution comparator the temporal-slice ranking (Part G)
    uses internally.

    Returns
    -------
    dict or None : {'wasserstein', 'energy', 'ks_similarity', 'ratio',
    'wasserstein_score', 'energy_score', 'representativeness_score'} --
    None if either side has no values left after outlier trimming.
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    return TwinnedSimple3DBase.compare_property_distributions(
        mdf['miso_deg'], mc_mdf_post['miso_deg'])
