"""EBSD Analysis-2 -- Part D of the Twinned FCC walkthrough.

In order: MDF & Twin-Peak Selection, CSL Segregation & TVF, Grain-Role
Properties, EBSD VF Partition, Twin Thickness Statistics -- one function per
stage, each calling the underlying repgen2d/core methods directly. This is
the densest stage: its outputs (parent_info, tvf, VF-partition targets,
twin thickness) feed almost everything downstream.
"""

# Twin-relevant CSL type the rest of the pipeline defaults to.
DEFAULT_CSL_LABEL = "S3  (twin)"

# Reference misorientation angle (degrees) per cubic CSL type.
# compute_mdf_ebsd's `csl=` argument wants {label: angle}, NOT {label: bool}
# -- passing booleans silently coerces True -> 1.0deg and produces nonsense
# CSL matches (every peak's "nearest CSL" reports ~1deg off, matching NO
# real CSL type), a real bug caught while building this module.
CUBIC_CSL_ANGLES = {
    'S3  (twin)': 60.00,
    'S5': 36.87,
    'S7': 38.21,
    'S9': 38.94,
    'S11': 50.48,
    'S13b': 27.80,
}


def compute_mdf_and_peaks(rg, n_bins=65, angle_max=65.0, prominence=0.002,
                           distance=3, csl_tol=2.0, bw_method='scott', n_kde=500,
                           csl_include=None):
    """MDF & Twin-Peak Selection -- computes the misorientation-distribution
    function and auto-detects candidate CSL peaks. `csl_include` defaults
    to every cubic CSL type in CUBIC_CSL_ANGLES (Sigma3/5/7/9/11/13b) --
    pass a subset of its keys to narrow it.

    Returns
    -------
    (mdf, peaks)
    """
    if csl_include is None:
        csl_include = dict(CUBIC_CSL_ANGLES)
    else:
        csl_include = {label: CUBIC_CSL_ANGLES[label] for label in csl_include}
    return rg.compute_mdf_ebsd(
        n_bins=n_bins, angle_range=(0.0, angle_max), prominence=prominence,
        distance=distance, csl=csl_include, csl_tol=csl_tol,
        bw_method=bw_method, n_kde=n_kde, plot=False)


def select_all_peaks(peaks):
    """Confirms every detected peak -- the default is to keep every one
    unless you deliberately want to narrow the set.

    Returns
    -------
    dict : {'indices': [...], 'angles': [...]} -- the shape
    segregate_csl_pairs() expects.
    """
    return {'indices': list(peaks['peak_indices']), 'angles': list(peaks['peak_angles'])}


def segregate_csl_pairs(rg, mdf, peaks, selected_peaks):
    """CSL Segregation & TVF -- pairs grains across each selected
    CSL-angle peak into parent/twin candidate pairs."""
    return rg.segregate_csl_pairs(mdf, selected_peaks, peaks['csl'], peaks['csl_tol'])


def identify_parent_grains(rg, csl_grains):
    """Classifies every grain touching a CSL pair as pure_parent /
    pure_twin / intermediate, then builds the twin-merged EBSD grain map
    (twins relabelled into their parent) that the Host Allocation target
    hosting fraction and Pre-Twin Validation grain-size reference both
    need -- there is no other way to get it, so this always runs both
    calls together.

    Returns
    -------
    dict : parent_info, keyed by CSL label.
    """
    parent_info = rg.identify_parent_grains(
        csl_grains, plot_parent_twin_maps=False, plot_combined_parent_twin_map=False)
    rg.build_merged_ebsd_lfi(parent_info, plot=False)
    return parent_info


def compute_twin_area_fractions(rg, parent_info, csl_labels=(DEFAULT_CSL_LABEL,)):
    """Twin area fraction (2D proxy for 3D TVF) per selected CSL label.

    Returns
    -------
    dict : {csl_label: tvf_dict}
    """
    return {label: rg.compute_ebsd_tvf(parent_info, csl_label=label) for label in csl_labels}


def compute_grain_role_properties(rg, parent_info, selected_props=('area',),
                                   selected_groups=('pure_parents', 'pure_twins',
                                                     'intermediates', 'non_role')):
    """Grain-Role Properties -- per-role-group property distributions
    (pure_parents/pure_twins/intermediates/non_role); 22 possible
    properties across 4 tiers are available, only 'area' selected by
    default.

    Returns
    -------
    dict : {prop_name: {group_name: ndarray}}
    """
    from upxo.viz.ebsdviz import compute_grain_role_property_distributions
    return compute_grain_role_property_distributions(
        lfi=rg.lfi_ebsd, parent_info=parent_info, prop=rg.prop_ebsd,
        neigh_gid=rg.neigh_gid_ebsd, selected_props=list(selected_props),
        selected_groups=list(selected_groups), step_size=rg.ebsd_step)


def compute_vf_partition(rg, parent_info, tvf_by_label, csl_label=DEFAULT_CSL_LABEL,
                          scale_2d_to_3d=1.00):
    """EBSD VF Partition -- splits pure twins into Type 2a (outward,
    touches a pure-parent) / Type 2b (inward, fully enclosed by
    intermediates), then derives the 3D VF targets Twin Generation uses
    (Stage-1, Secondary-2a, Secondary-2b).

    scale_2d_to_3d=1.00 is the physically correct default for randomly
    oriented {111} twin lamellae (Cavalieri principle: 2D area fraction
    approx-equals 3D volume fraction).

    Returns
    -------
    (vf_partition, vf_targets) : vf_targets has keys 'tvf_stage1',
    'tvf_secondary_2a', 'tvf_secondary_2b', 'prob_secondary_outward'.
    """
    tvf = tvf_by_label[csl_label]
    vf_partition = rg.compute_ebsd_twin_vf_partition(parent_info, tvf)

    prim_frac = tvf.get('primary_twin_frac', 0.0)
    int_frac = tvf.get('intermediate_frac', 0.0)
    sec_frac = tvf.get('secondary_twin_frac', 0.0)
    prob_out = vf_partition['prob_secondary_outward']
    if prob_out > 0.95 or prob_out < 0.05:
        # Partition unreliable at extreme skew (either tail) -- fall back
        # to an even 50/50 split rather than trusting a near-0/near-1
        # estimate from limited grain counts.
        prob_out = 0.5

    vf_targets = {
        'tvf_stage1': (prim_frac + int_frac) * scale_2d_to_3d,
        'tvf_secondary_2a': sec_frac * prob_out * scale_2d_to_3d,
        'tvf_secondary_2b': sec_frac * (1.0 - prob_out) * scale_2d_to_3d,
        'prob_secondary_outward': prob_out,
    }
    return vf_partition, vf_targets


def compute_twin_thickness(rg, parent_info, abrupt_threshold=0.8,
                            linear_intercept_axes=('x', 'y', 'z'),
                            linear_intercept_n_lines=20):
    """Twin Thickness Statistics -- per-grain major-axis-intercept
    thickness (the primary measurement) plus the classical ASTM
    E112-style linear-intercept method, pooled across axes. Feeds the
    thickness scale factor Twin Generation uses.

    Note: this measurement is sensitive to EBSD step size -- a coarser
    step size measurably inflates apparent thickness. Keep the same
    subsampling stride used earlier in the notebook when interpreting
    this result.

    Returns
    -------
    dict : same shape as compute_mc_twin_thickness()'s result (mean,
    median, thick_um, n_twins, ...).
    """
    n_lines = linear_intercept_n_lines
    if not isinstance(n_lines, dict):
        n_lines = {ax: n_lines for ax in linear_intercept_axes}
    return rg.compute_mc_twin_thickness(
        parent_info, abrupt_threshold=abrupt_threshold,
        linear_intercept_axes=list(linear_intercept_axes),
        linear_intercept_n_lines=n_lines)


def ebsd_pole_figure_stages(rg, parent_info, csl_label=None):
    """Per-grain mean orientations for the five EBSD pole-figure
    populations: 'full' (every grain), 'parents' (twins merged back into
    their host), 'primary_twins', 'secondary_twins' (undifferentiated
    generation 2+), and 'all_twins' (primary+secondary combined). Each
    is a {'gids', 'quats'} pair ready for
    steps_visualization_export.plot_pole_figure()/plot_pole_figure_overlay().

    csl_label=None uses parent_info's first CSL label -- irrelevant to
    'full'/'parents' (not CSL-specific), only affects which pairing the
    twin-only stages are drawn from.

    Returns
    -------
    dict : {'csl_label', 'full', 'parents', 'primary_twins',
    'secondary_twins', 'all_twins'}
    """
    return rg.compute_ebsd_texture_stages(parent_info, csl_label=csl_label)
