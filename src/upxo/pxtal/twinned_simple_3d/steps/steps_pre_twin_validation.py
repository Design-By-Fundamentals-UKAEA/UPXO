"""Pre-Twin Validation -- Part I of the Twinned FCC walkthrough.

Validates the host-allocated base structure's grain-SIZE distribution
against the EBSD twin-merged reference (twins merged into their parent,
since the base structure has no twins yet -- comparing against
still-twinned EBSD grains would be apples-to-oranges), per axis, via 2D
cross-sections + a Wasserstein-distance pass threshold.
"""


def validate_morphological_representativeness(
        base, rg, parent_info, n_slices=(10, 10, 10), test_axes=('x', 'y', 'z'),
        p_percentage=(60.0, 60.0, 60.0), wasserstein_threshold=0.5):
    """Validates `base.lgi` (the host-allocated structure) against the
    EBSD twin-merged grain-size reference.

    n_slices/p_percentage: (x, y, z) tuples, one value per axis.
    test_axes: which of x/y/z to actually test (all three by default).

    Returns
    -------
    RepresentativenessValidator3D : `validator` -- validator.axis_acceptance,
    validator.overall_accepted, validator.report() are the results;
    pass this straight into steps_orientation_assignment's functions.
    """
    from upxo.pxtal.twinned_simple_3d.repr_validator_3d import RepresentativenessValidator3D
    from upxo.charops.mchar import get_grain_size_distribution_from_slice

    if getattr(rg, 'lfi_ebsd_merged', None) is None:
        rg.build_merged_ebsd_lfi(parent_info, plot=False)
    ebsd_areas = get_grain_size_distribution_from_slice(rg.lfi_ebsd_merged)

    n_x, n_y, n_z = n_slices
    p_x, p_y, p_z = p_percentage
    validator = RepresentativenessValidator3D(
        n_slices_x=n_x, n_slices_y=n_y, n_slices_z=n_z,
        test_along_x='x' in test_axes, test_along_y='y' in test_axes,
        test_along_z='z' in test_axes,
        p_percentage_x=p_x, p_percentage_y=p_y, p_percentage_z=p_z,
        wasserstein_threshold=wasserstein_threshold,
    )
    validator.validate_morphological(base.lgi, ebsd_areas)
    return validator
