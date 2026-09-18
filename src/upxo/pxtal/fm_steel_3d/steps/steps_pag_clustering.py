"""PAG Clustering -- Part E of the FM Steel 3D walkthrough.

Groups grains into Prior Austenite Grains (PAGs), then assigns each PAG a
crystal orientation. `generate_pags` wraps the same technique-selector
entry point (`pag_technique_selector_3d.generate_pags`) the real GUI's PAG
generation button calls -- not the lower-level
`FMSteel3DBase.generate_pag_clusters`, which is only ever reached
internally, for technique A's non-MIS levers.
"""

DEFAULT_PAG_SIZE_DISTRIBUTION = {'sizes': [3, 4, 5, 6], 'probs': [0.2, 0.35, 0.3, 0.15]}


def generate_pags(fm_base, technique='A', pag_size_distribution=None,
                   max_packets_per_pag=4, lever='lever3', pag_grain_fraction=1.0,
                   use_non_neigh_pag=True, random_seed=None):
    """Groups `fm_base`'s grains into PAGs. `technique`: 'A' (grain
    clustering, uses `lever`) or 'B' (within-grain splitting).

    Returns
    -------
    FMSteel3DWithPAGs
    """
    from upxo.pxtal.fm_steel_3d.pag_technique_selector_3d import generate_pags as _generate_pags
    return _generate_pags(
        fm_base, technique=technique,
        pag_size_distribution=pag_size_distribution or DEFAULT_PAG_SIZE_DISTRIBUTION,
        max_packets_per_pag=max_packets_per_pag, lever=lever,
        pag_grain_fraction=pag_grain_fraction, use_non_neigh_pag=use_non_neigh_pag,
        random_seed=random_seed)


def assign_pag_orientations(fm_pag, pag_ori_mode='random', pag_ori_params=None, random_seed=None):
    """Assigns a crystal orientation to every PAG. Modes: 'random',
    'hagb_constrained', 'textured', 'fixed'.

    Mutates `fm_pag` IN PLACE (stores into `fm_pag.pag_orientations`) and
    returns None -- unlike every other stage in this pipeline, which
    returns a new object. Call this as a bare statement, not an assignment.
    """
    fm_pag.assign_pag_orientations(
        pag_ori_mode=pag_ori_mode, pag_ori_params=pag_ori_params, random_seed=random_seed)


def compare_techniques(fm_base, configs, random_seed=None):
    """Trial-runs `generate_pags` once per config in `configs` (each a dict
    of kwargs accepted by `generate_pags` above, e.g.
    `{'technique': 'A', 'lever': 'lever1'}`), for side-by-side comparison
    before committing to one (mirrors the PAG Technique Advisor page's own
    trial-run loop). Does not mutate `fm_base` or keep any of the trial
    FMSteel3DWithPAGs objects.

    Returns
    -------
    list[dict] : one summary per config, each
    `{'config': config, 'n_pags': int, 'n_isolated_grains': int}`.
    """
    summaries = []
    for config in configs:
        fm_pag = generate_pags(fm_base, random_seed=random_seed, **config)
        summaries.append({
            'config': config,
            'n_pags': len(fm_pag.clusters_dict),
            'n_isolated_grains': len(fm_pag.isolated_grains),
        })
    return summaries
