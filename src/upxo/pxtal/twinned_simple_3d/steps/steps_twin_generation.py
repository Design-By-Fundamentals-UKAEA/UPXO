"""Twin Generation -- Part K of the Twinned FCC walkthrough.

Introduces primary + secondary twins into the oriented, host-allocated
base structure, then cleans the result (removes spike voxels / splits
non-convex lobes left behind by the voxel carving).
"""


def generate_twins(base, rg, parent_info, assigner, twin_thickness, vf_targets=None,
                    csl_label="S3  (twin)", n_lamellae_per_host=10,
                    nucleation_site='gb_centroid', meshing_route='conformal',
                    min_thick_conformal=2, min_thick_nonconformal=1, min_host_vox=8,
                    tvf_tolerance=0.05, tvf_2d_to_3d_scale=1.15, thick_scale_factor=0.80,
                    max_thick_um=None, max_vf_per_host=0.50,
                    prob_separated=0.30, prob_contacting=0.70,
                    prob_sec_outward=0.60, prob_sec_inward=0.40,
                    schmid_dir=(0.0, 0.0, 1.0), host_schmid_weight=1.0,
                    use_schmid=True, orient_scatter_deg=1.5, rng_seed=123):
    """Introduces primary twins (at host grain-boundary centroids by
    default) then secondary twins (nucleating inward/outward from the
    primaries), targeting the EBSD twin-area-fraction (TVF) for
    `csl_label`.

    `twin_thickness`: the dict from steps_ebsd_analysis_2.compute_twin_thickness
    -- Twin Generation reuses that measurement rather than recomputing it
    (including whatever abrupt-boundary threshold it was computed with).
    `assigner`: the result of steps_orientation_assignment.assign_orientations.
    `vf_targets`: the dict from steps_ebsd_analysis_2.compute_vf_partition
    ('tvf_stage1'/'tvf_secondary_2a'/'tvf_secondary_2b'). Passing this is
    what makes primary-twin introduction stop at its own Stage-1 share of
    the total EBSD twin fraction, instead of the combined overall
    fraction -- without it, primary twins alone can reach the full target
    and secondary introduction never triggers (0 secondary twins). None
    (default) falls back to that combined-fraction behaviour.
    `max_thick_um`: None (default) means no cap; pass a number to enable one.

    Returns
    -------
    TwinGenerator3D : `tg` -- tg.lgi_twinned/tg.all_quats/tg.twin_role/
    tg.twin_parent_of feed steps_twin_generation.clean_structure();
    tg.summary()/tg.diagnose(tvf) give the achieved-vs-target report.
    """
    from upxo.pxtal.twinned_simple_3d.twin_generator_3d import TwinGenerator3D

    tvf = rg.compute_ebsd_tvf(parent_info, csl_label=csl_label)

    tg = TwinGenerator3D(
        base,
        n_lamellae_per_host=n_lamellae_per_host,
        twin_nucleation_site=nucleation_site,
        tvf_tolerance=tvf_tolerance,
        twin_orient_scatter_deg=orient_scatter_deg,
        meshing_route=meshing_route,
        min_lamella_thickness_conformal=min_thick_conformal,
        min_lamella_thickness_non_conformal=min_thick_nonconformal,
        min_host_vox_for_lamella=min_host_vox,
        prob_lamella_separated=prob_separated,
        prob_lamella_contacting=prob_contacting,
        prob_secondary_outward_twinNucleation=prob_sec_outward,
        prob_secondary_inward_twinNucleation=prob_sec_inward,
        twin_thick_scale_factor=thick_scale_factor,
        tvf_2d_to_3d_scale_factor=tvf_2d_to_3d_scale,
        max_lamella_thickness_um=max_thick_um,
        max_lamella_vf_per_host=max_vf_per_host,
        schmid_loading_direction=schmid_dir,
        host_schmid_weight=host_schmid_weight,
        use_schmid_for_variant_selection=use_schmid,
        rng_seed=rng_seed,
    )
    tvf_stage1 = vf_targets['tvf_stage1'] if vf_targets else None
    tvf_2a = vf_targets['tvf_secondary_2a'] if vf_targets else None
    tvf_2b = vf_targets['tvf_secondary_2b'] if vf_targets else None
    tg.introduce_primary_twins(
        host_orientations=assigner.all_grain_orientations,
        twin_thickness=twin_thickness, tvf=tvf, tvf_stage1=tvf_stage1)
    tg.introduce_secondary_twins(
        tvf=tvf, twin_thickness=twin_thickness, tvf_2a=tvf_2a, tvf_2b=tvf_2b)
    return tg, tvf


def clean_structure(tg, min_voxels=0, n_passes=5, upscale_fallback=False,
                     split_jitter_deg=0.0, rng_seed=7,
                     do_spike_removal=True, do_lobe_split=True, verbose=True):
    """Topology Cleaning -- removes spike voxels and splits non-convex
    ("lobed") grains the twin-carving step can leave behind, recursively
    up to `n_passes` (capped at 20 as a safety limit).

    Returns
    -------
    (cleaner, n_passes_run) : `cleaner` (StructureCleaner3D) carries
    lgi_clean/all_quats_clean/twin_role_clean/twin_parent_of_clean --
    pass it into steps_post_twin_validation's functions.
    """
    from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D
    n_passes = min(n_passes, 20)
    return StructureCleaner3D.clean_recursive(
        tg.lgi_twinned, tg.all_quats, tg.twin_role, tg.twin_parent_of,
        n_passes=n_passes, upscale_fallback=upscale_fallback,
        split_jitter_deg=split_jitter_deg, rng_seed=rng_seed,
        min_clean_voxels=min_voxels, do_spike_removal=do_spike_removal,
        do_lobe_split=do_lobe_split, verbose=verbose)
