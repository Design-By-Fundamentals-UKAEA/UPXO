"""Distribution Viewer -- Part M of the Twinned FCC walkthrough.

An EBSD-vs-SGS "Comparison Properties" overlay (misorientation / twin
thickness / host grain size / twin volume fraction) -- the headline "did
the finished synthetic structure end up matching EBSD" comparison. A
second comparison mode, compare_role_properties() below, breaks
morphology down by grain role (non-hosting matrix / host / twin) instead
of pooling everything together. Two further functions,
compute_texture_component_vf_comparison() and
compute_pole_figure_delta_mud_iqr(), compare crystallographic TEXTURE
(the population of absolute grain orientations) rather than morphology
or neighbour-pair misorientation.
"""

COMPARISON_PROP_LABELS = {
    "misorientation": "Misorientation Angle -- MDF (deg)",
    "twin_thickness": "Twin Lamella Thickness (um)",
    "host_grain_size": "Host Grain Equivalent Diameter (um)",
    "twin_volume_fraction": "Twin Volume / Area Fraction",
}


def compare_ebsd_vs_sgs(cleaner, tg, base, rg, parent_info, mdf, twin_thickness,
                         n_per_axis=5, axes=('x', 'y', 'z'),
                         properties=('misorientation', 'twin_thickness',
                                     'host_grain_size', 'twin_volume_fraction')):
    """Assembles the pooled EBSD-vs-SGS arrays for the requested
    comparison properties -- ready to hand to
    upxo.viz.vizDistr.plot_grouped_distributions for a side-by-side
    overlay, or just inspect directly (e.g. np.mean(arr)).

    Returns
    -------
    dict : {prop_name: {'ebsd': ndarray, 'sgs': ndarray}}
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.feature_props_3d import assemble_ebsd_sgs_comparison_data

    ebsd_ref, sgc_ref = assemble_ebsd_sgs_comparison_data(
        cleaner, tg, base, rg, parent_info, mdf, twin_thickness,
        n_slices_per_axis=n_per_axis, axes=list(axes))

    cmp_source = {
        'misorientation': (ebsd_ref['miso_deg_full'], sgc_ref['miso_deg_posttwin']),
        'twin_thickness': (ebsd_ref['twin_thick_um'], sgc_ref['twin_thick_3d_um']),
        'host_grain_size': (ebsd_ref['host_eqdia_um'], sgc_ref['host_eqdia_um']),
        'twin_volume_fraction': (np.array([ebsd_ref['tvf_2d']]), sgc_ref['tvf_2d_slices']),
    }
    return {
        pname: {'ebsd': np.asarray(cmp_source[pname][0], dtype=float),
                'sgs': np.asarray(cmp_source[pname][1], dtype=float)}
        for pname in properties
    }


def compare_role_properties(rg, parent_info, cleaner, voxel_size=1.0,
                             selected_props=('area', 'aspect_ratio'),
                             n_slices_per_axis=5, axes=('x', 'y', 'z')):
    """Per-property morphology comparison between EBSD and the synthetic
    structure (SGS), broken down by grain role -- not just pooled overall
    like compare_ebsd_vs_sgs() above.

    EBSD's grain-role taxonomy (pure_parents/pure_twins/intermediates/
    non_role, from Grain-Role Properties in EBSD Analysis-2) and the SGS
    structure's taxonomy (nonhost/host/primary/seca/secb, from twin
    generation) don't correspond 1:1 -- 'intermediates' (an EBSD twin
    that itself hosts a further twin) has no direct SGS equivalent, since
    the SGS primary/secondary split is about nucleation generation, not
    about whether a twin itself hosts one. Both sides are therefore
    pooled down to the same three coarse buckets this function CAN
    defend: 'non_hosting' (EBSD non_role / SGS nonhost), 'host' (EBSD
    pure_parents / SGS host), and 'twin' (EBSD pure_twins+intermediates
    pooled / SGS primary+seca+secb pooled -- any generation, either side).

    Returns
    -------
    dict : {prop_name: {'non_hosting': {'ebsd':ndarray,'sgs':ndarray},
    'host': {...}, 'twin': {...}}}
    """
    import numpy as np
    from upxo.viz.ebsdviz import compute_grain_role_property_distributions
    from upxo.pxtal.twinned_simple_3d.feature_props_3d import compute_sgs_role_property_distributions

    props = list(selected_props)
    ebsd_groups = compute_grain_role_property_distributions(
        lfi=rg.lfi_ebsd, parent_info=parent_info, prop=rg.prop_ebsd,
        neigh_gid=rg.neigh_gid_ebsd, selected_props=props,
        selected_groups=['pure_parents', 'pure_twins', 'intermediates', 'non_role'],
        step_size=rg.ebsd_step)

    sgs_levels = compute_sgs_role_property_distributions(
        cleaner.lgi_clean, cleaner.twin_role_clean, cleaner.twin_parent_of_clean,
        selected_props=props, selected_levels=['nonhost', 'host', 'primary', 'seca', 'secb'],
        n_slices_per_axis=n_slices_per_axis, axes=axes, voxel_size=voxel_size)

    result = {}
    for prop in props:
        ebsd_p, sgs_p = ebsd_groups[prop], sgs_levels[prop]
        result[prop] = {
            'non_hosting': {'ebsd': ebsd_p['non_role'], 'sgs': sgs_p['nonhost']},
            'host': {'ebsd': ebsd_p['pure_parents'], 'sgs': sgs_p['host']},
            'twin': {
                'ebsd': np.concatenate([ebsd_p['pure_twins'], ebsd_p['intermediates']]),
                'sgs': np.concatenate([sgs_p['primary'], sgs_p['seca'], sgs_p['secb']]),
            },
        }
    return result


def role_property_match_summary(role_prop_data, tolerance_pct=25.0):
    """Per-property EBSD-vs-SGS match summary for
    compare_role_properties()'s output. For each property, averages
    TwinnedSimple3DBase.compare_property_distributions()'s per-bucket
    'representativeness_score' (a [0, 1] measure) and 'ratio' (trimmed
    SGS mean / trimmed EBSD mean) across the three role buckets, then
    applies the same ratio-tolerance acceptance rule
    mc_qualification.prop_qualifies() uses for temporal-slice
    qualification (Part G): a property "accepts" when its mean ratio
    falls within +/- tolerance_pct% of 1.0. The default (25%) is looser
    than Part G's own property-tolerance defaults (typically 5%), since
    those compare EBSD-detection-derived candidate statistics at a much
    larger sample size, whereas this compares role-bucket populations
    drawn from 2D cross-sections of the finished, cleaned structure --
    inherently smaller samples with more run-to-run variability.

    Returns
    -------
    dict : {prop_name: {'score': float or None, 'ratio': float or None,
    'accepted': bool}} -- 'accepted' is False whenever 'ratio' is None
    (no bucket had values on both sides after outlier trimming).
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase

    summary = {}
    for prop, buckets in role_prop_data.items():
        bucket_scores, bucket_ratios = [], []
        for vals in buckets.values():
            if vals['ebsd'].size == 0 or vals['sgs'].size == 0:
                continue
            cmp = TwinnedSimple3DBase.compare_property_distributions(vals['ebsd'], vals['sgs'])
            if cmp is not None:
                bucket_scores.append(cmp['representativeness_score'])
                bucket_ratios.append(cmp['ratio'])
        mean_score = float(np.mean(bucket_scores)) if bucket_scores else None
        mean_ratio = float(np.mean(bucket_ratios)) if bucket_ratios else None
        accepted = mean_ratio is not None and abs(mean_ratio - 1.0) <= tolerance_pct / 100.0
        summary[prop] = {'score': mean_score, 'ratio': mean_ratio, 'accepted': accepted}
    return summary


def _sgs_quat_population(cleaner):
    """Shared helper: (gids, quats, weights) for every grain in the
    cleaned structure -- weights are voxel counts, for volume-fraction-
    correct texture-component detection."""
    import numpy as np

    gids = np.array(sorted(cleaner.all_quats_clean.keys()))
    quats = np.array([cleaner.all_quats_clean[g] for g in gids])
    weights = np.bincount(cleaner.lgi_clean.ravel().astype(np.int64))[gids].astype(float)
    return gids, quats, weights


def compute_texture_component_vf_comparison(rg, cleaner, n_peaks=4, bandwidth_deg=10.0):
    """Texture representativeness, measure 1: does the same set of
    texture components exist in similar amounts, EBSD vs. the synthetic
    structure? Runs detect_texture_component_peaks() independently on
    each population's grain orientations, then compares the volume
    fractions of components that auto-matched to the SAME standard name
    on both sides (generic "Component N" peaks -- ones that didn't match
    any standard component closely enough -- aren't comparable across
    the two independent detection runs, so they're excluded from the
    matched comparison, though still returned in full).

    No pass/fail threshold is applied here (none is established anywhere
    in the pipeline for texture) -- this reports the achieved numbers for
    you to judge, not a verdict.

    Returns
    -------
    dict : {'ebsd_components', 'sgs_components' (full peak-detection
    results, each), 'matched' (list of {'component', 'ebsd_vf_pct',
    'sgs_vf_pct', 'abs_diff_pct'}), 'total_abs_deviation_pct' (float,
    sum of |diff| across matched components -- lower is more alike)}
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import grain_avg_quats, detect_texture_component_peaks

    ebsd_gids, ebsd_q = grain_avg_quats(rg.lfi_ebsd, rg.quat_ebsd)
    ebsd_weights = np.bincount(rg.lfi_ebsd.ravel().astype(np.int64))[ebsd_gids].astype(float)
    ebsd_result = detect_texture_component_peaks(
        ebsd_q, weights=ebsd_weights, n_peaks=n_peaks, bandwidth_deg=bandwidth_deg)

    _, sgs_q, sgs_weights = _sgs_quat_population(cleaner)
    sgs_result = detect_texture_component_peaks(
        sgs_q, weights=sgs_weights, n_peaks=n_peaks, bandwidth_deg=bandwidth_deg)

    ebsd_by_name = {c['name']: c['vf_pct'] for c in ebsd_result['components']
                     if not c['name'].startswith('Component ')}
    sgs_by_name = {c['name']: c['vf_pct'] for c in sgs_result['components']
                    if not c['name'].startswith('Component ')}

    matched = []
    total_abs_deviation = 0.0
    for name in sorted(set(ebsd_by_name) & set(sgs_by_name)):
        ev, sv = ebsd_by_name[name], sgs_by_name[name]
        diff = abs(ev - sv)
        matched.append({'component': name, 'ebsd_vf_pct': ev, 'sgs_vf_pct': sv, 'abs_diff_pct': diff})
        total_abs_deviation += diff

    return {
        'ebsd_components': ebsd_result['components'],
        'sgs_components': sgs_result['components'],
        'matched': matched,
        'total_abs_deviation_pct': total_abs_deviation,
    }


def compute_pole_figure_delta_mud_iqr(rg, cleaner, pole_family='100', grid_points='auto',
                                       half_width_deg=7.5, unit_normalize=False):
    """Texture representativeness, measure 2: builds a {pole_family}
    pole figure for each population (EBSD vs. the synthetic structure)
    and computes the interquartile range of their MUD (Multiples of
    Uniform Density) grid difference -- a single-number summary of how
    much the two pole figures disagree, independent of any texture-
    component naming/matching (measure 1 above). Lower IQR = more alike.

    No pass/fail threshold is applied here either -- reports the
    achieved number, not a verdict.

    Returns
    -------
    dict : {'Xi', 'Yi', 'zi_diff' (grid arrays, for plotting the
    difference map yourself), 'iqr' (float)}
    """
    from upxo.xtalphy.crystal_orientation import grain_avg_quats
    from upxo.viz.xphy.pole_figure import PoleFigure

    ebsd_gids, ebsd_q = grain_avg_quats(rg.lfi_ebsd, rg.quat_ebsd)
    sgs_gids, sgs_q, _ = _sgs_quat_population(cleaner)

    pf_ebsd = PoleFigure(ebsd_q, convention='quaternion', gids=ebsd_gids)
    pf_sgs = PoleFigure(sgs_q, convention='quaternion', gids=sgs_gids)

    return pf_ebsd.compute_density_difference(
        pf_sgs, pole_family=pole_family, grid_points=grid_points,
        half_width_deg=half_width_deg, unit_normalize=unit_normalize)


def plot_texture_residual(ebsd_stage, sgs_stage, pole_family='100', grid_points='auto',
                           half_width_deg=7.5, unit_normalize=False,
                           title_ebsd='EBSD', title_sgs='SGS'):
    """The three-panel EBSD | SGS | Difference pole-figure plot -- MUD
    density for each population side by side, then their difference,
    with its IQR shown in the difference panel's own title.

    ebsd_stage/sgs_stage: one matching entry each from
    steps_ebsd_analysis_2.ebsd_pole_figure_stages()/
    steps_visualization_export.sgs_pole_figure_stages() (e.g. both
    'full', or EBSD 'parents' paired with SGS 'host').

    Returns
    -------
    (fig, (ax_ebsd, ax_sgs, ax_diff), iqr)
    """
    import matplotlib.pyplot as plt
    from upxo.viz.xphy.pole_figure import PoleFigure

    def _pf(stage):
        quats = stage['quats'].copy()
        quats[:, 1:] *= -1
        return PoleFigure(quats, convention='quaternion', gids=stage['gids'])

    pf_ebsd = _pf(ebsd_stage)
    pf_sgs = _pf(sgs_stage)

    fig, (ax_ebsd, ax_sgs, ax_diff) = plt.subplots(1, 3, figsize=(18, 6))
    pf_ebsd.plot_density(pole_family=pole_family, ax=ax_ebsd, grid_points=grid_points,
                          half_width_deg=half_width_deg,
                          title=f"{title_ebsd} ({len(ebsd_stage['gids'])})")
    pf_sgs.plot_density(pole_family=pole_family, ax=ax_sgs, grid_points=grid_points,
                         half_width_deg=half_width_deg,
                         title=f"{title_sgs} ({len(sgs_stage['gids'])})")
    diff = pf_ebsd.compute_density_difference(
        pf_sgs, pole_family=pole_family, grid_points=grid_points,
        half_width_deg=half_width_deg, unit_normalize=unit_normalize)
    pf_ebsd.plot_density_difference(
        pf_sgs, pole_family=pole_family, ax=ax_diff, grid_points=grid_points,
        half_width_deg=half_width_deg, unit_normalize=unit_normalize,
        title=f"Difference (IQR={diff['iqr']:.4f})")
    fig.tight_layout()
    return fig, (ax_ebsd, ax_sgs, ax_diff), diff['iqr']
