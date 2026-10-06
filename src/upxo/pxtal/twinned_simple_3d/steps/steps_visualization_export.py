"""Visualization & Export -- Part O of the Twinned FCC walkthrough.

Covers a 3D PyVista render, a raw npy/pickle dump, and an Abaqus mesh +
elsets/nodesets export. Elset/nodeset configuration is pure data (no
compute step of its own) -- it is all consumed by export_abaqus_mesh
below via its role_enabled/role_prefix/nset_config arguments. Finishes
by rendering the accumulated upxo.reporting.ReportSession to report.html.
"""
from pathlib import Path

from .steps_start import DEFAULT_OUTPUT_DIR

# Per-role-level elset defaults (every level enabled, prefix "es_<level>_").
DEFAULT_ROLE_ENABLED = {
    'non_host': True, 'host': True, 'primary_twin': True,
    'stwin_a': True, 'stwin_b': True,
}
DEFAULT_ROLE_PREFIX = {
    'non_host': 'es_nonhost_', 'host': 'es_host_', 'primary_twin': 'es_primary_',
    'stwin_a': 'es_seca_', 'stwin_b': 'es_secb_',
}
# Per-face nodeset defaults -- origins/rve_edges nodesets are not
# implemented and are omitted here.
DEFAULT_NSET_CONFIG = {
    'XMIN': {'enabled': True, 'prefix': 'ns_x-'}, 'XMAX': {'enabled': True, 'prefix': 'ns_x+'},
    'YMIN': {'enabled': True, 'prefix': 'ns_y-'}, 'YMAX': {'enabled': True, 'prefix': 'ns_y+'},
    'ZMIN': {'enabled': True, 'prefix': 'ns_z-'}, 'ZMAX': {'enabled': True, 'prefix': 'ns_z+'},
}


def render_3d_structure(cleaner, role_opacity=None, role_color=None, nonhost_cmap=None):
    """Interactive PyVista 3D render (renders inline if called from a
    Jupyter notebook with PyVista's Jupyter backend enabled; opens a
    separate window otherwise). No return value -- this is a plotting
    side effect.

    role_opacity/role_color default to None, which lets render_3d apply
    its own tuned defaults -- notably primary_twin at partial opacity
    (0.6, not fully solid), so secondary twins nucleated at or inside the
    primary-twin/host interface stay visible instead of being hidden
    behind an opaque primary-twin surface. Pass your own dicts only if
    you deliberately want different opacities/colours.
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.viz_3d import render_3d

    lgi_xyz = np.transpose(cleaner.lgi_clean, (2, 1, 0))
    render_3d(lgi_xyz, cleaner.twin_role_clean, role_opacity=role_opacity,
              role_color=role_color, nonhost_cmap=nonhost_cmap)


def render_ipf_slice(cleaner, axis=2, slice_idx=None, sample_direction=(0., 0., 1.)):
    """Plots an IPF-coloured 2D slice through the cleaned structure --
    plain matplotlib (unlike render_3d_structure above, this has no
    interactive-window step and runs fine in a headless/automated
    execution).

    `axis`: 0=X, 1=Y, 2=Z. `slice_idx`: None (default) uses the domain's
    mid-slice.

    Returns
    -------
    (fig, ax)
    """
    from upxo.pxtal.twinned_simple_3d.viz_3d import plot_ipf_slice
    return plot_ipf_slice(cleaner.lgi_clean, cleaner.all_quats_clean, axis=axis,
                           slice_idx=slice_idx, sample_direction=sample_direction)


def sgs_pole_figure_stages(cleaner):
    """Per-grain orientations for the five SGS (synthetic structure)
    pole-figure populations, matching
    steps_ebsd_analysis_2.ebsd_pole_figure_stages()'s shape exactly for
    direct pairing in steps_distribution_viewer.plot_texture_residual():
    'full' (every grain), 'host' (every non-twin grain -- both allocated
    hosts and untouched non-hosts, the synthetic analogue of EBSD's
    'parents'), 'primary_twins', 'secondary_twins', and 'all_twins'.

    Returns
    -------
    dict : {'full', 'host', 'primary_twins', 'secondary_twins',
    'all_twins'} -- each a {'gids': ndarray, 'quats': ndarray}.
    """
    import numpy as np

    role_filters = {
        'full': None,
        'host': ('host', 'non_host'),
        'primary_twins': ('primary_twin',),
        'secondary_twins': ('secondary_twin',),
        'all_twins': ('primary_twin', 'secondary_twin'),
    }
    stages = {}
    for stage_key, wanted_roles in role_filters.items():
        if wanted_roles is None:
            gids = list(cleaner.all_quats_clean.keys())
        else:
            gids = [g for g in cleaner.all_quats_clean.keys()
                    if cleaner.twin_role_clean.get(g) in wanted_roles]
        quats = np.array([cleaner.all_quats_clean[g] for g in gids], dtype=np.float64)
        stages[stage_key] = {'gids': np.array(gids), 'quats': quats}
    return stages


def sample_frame_quats(stage_data, apply_sample_symmetry=True, use_rd=True, use_td=True, use_nd=True):
    """Pole-figure-ready orientations of one population.

    stage_data: {'gids', 'quats'} with quats in DefDAP's passive convention
    (as from ebsd_pole_figure_stages()/sgs_pole_figure_stages()). The
    vector part is negated (crystal->sample, defdap_passive_to_active), then,
    with apply_sample_symmetry, every orientation is replicated through the
    sample-symmetry group of 180 deg rotations about the selected RD/TD/ND
    axes -- default: all three on (4 operations).

    Returns
    -------
    (quats, gids) : gids tiled to match the replicated quats.
    """
    import numpy as np
    from upxo.viz.xphy.pole_figure import compute_sample_symmetry_group, apply_sample_symmetry as _apply_ss

    quats = np.asarray(stage_data['quats'], dtype=np.float64).copy()
    quats[:, 1:] *= -1
    gids = np.asarray(stage_data['gids'])
    group_ops = compute_sample_symmetry_group(apply_sample_symmetry, use_rd, use_td, use_nd)
    if len(group_ops) > 1 and len(quats):
        quats = _apply_ss(quats, 'quaternion', group_ops)
        gids = np.tile(gids, len(group_ops))
    return quats, gids


def plot_pole_figure(stage_data, pole_family='100', plot_mode='density',
                      half_width_deg=7.5, title=None, ax=None, apply_sample_symmetry=True,
                      use_rd=True, use_td=True, use_nd=True, projection='stereographic',
                      hemisphere='upper', cmap='viridis', grid_points='auto', mud_clip_min=None,
                      mud_clip_max=None, colorbar_decimals=2, x_label='X', y_label='Y', z_label='Z'):
    """Plots one population's pole figure -- a {'gids', 'quats'} dict
    from ebsd_pole_figure_stages()/sgs_pole_figure_stages().

    plot_mode: 'density' (MUD contour), 'scatter', or 'hybrid' (density
    with an IPF-coloured scatter overlay).
    half_width_deg: the density kernel's angular half-width (degrees) --
    ignored for plot_mode='scatter'.
    apply_sample_symmetry/use_rd/use_td/use_nd: sample symmetry (default
    on, RD+TD+ND). Crystal (cubic) symmetry is always applied
    through the pole family's symmetric equivalents.
    projection: 'stereographic' (default) or 'equal_area'.
    hemisphere: 'upper', 'lower' or 'both_mapped' (scatter and hybrid only).

    Returns
    -------
    (fig, ax)
    """
    from upxo.viz.xphy.pole_figure import PoleFigure

    if plot_mode not in ('density', 'scatter', 'hybrid'):
        raise ValueError("plot_mode must be 'density', 'scatter' or 'hybrid'")
    quats, gids = sample_frame_quats(stage_data, apply_sample_symmetry, use_rd, use_td, use_nd)
    pf = PoleFigure(quats, convention='quaternion', gids=gids)
    labels = dict(x_label=x_label, y_label=y_label, z_label=z_label)
    if plot_mode == 'scatter':
        return pf.plot_scatter(pole_family=pole_family, ax=ax, title=title, projection=projection,
                               hemisphere=hemisphere, **labels)
    density = dict(pole_family=pole_family, ax=ax, half_width_deg=half_width_deg, title=title,
                   projection=projection, cmap=cmap, grid_points=grid_points, mud_clip_min=mud_clip_min,
                   mud_clip_max=mud_clip_max, colorbar_decimals=colorbar_decimals, **labels)
    if plot_mode == 'hybrid':
        return pf.plot_density(overlay_scatter=True, scatter_color_by='ipf', scatter_hemisphere=hemisphere,
                               **density)
    return pf.plot_density(**density)


def plot_pole_figure_comparison(ebsd_stage, synth_stage, ebsd_label='EBSD', synth_label='Synthetic',
                                 pole_family='111', plot_mode='scatter', apply_sample_symmetry=True,
                                 use_rd=True, use_td=True, use_nd=True, projection='stereographic',
                                 hemisphere='upper', density_cmap='viridis', diff_cmap='RdBu_r',
                                 grid_points='auto', unit_normalize=False, colorbar_decimals=2,
                                 x_label='X', y_label='Y', z_label='Z'):
    """EBSD-against-synthetic pole-figure comparison (orientation assignment
    and post-twin validation): three panels side by side.

    plot_mode 'scatter' (default): EBSD | Synthetic | overlay.
    plot_mode 'density': EBSD | Synthetic MUD density on one shared colour
    scale, then their difference. grid_points 'auto' resolves to 150 so
    both sides share one grid.
    Both stages are {'gids', 'quats'} in DefDAP's passive convention; the
    same sample symmetry is applied to both, so their counts stay
    comparable. Panel titles give (grain count, plotted orientation count).

    Returns
    -------
    (fig, (ax_ebsd, ax_synth, ax_third))
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from upxo.viz.xphy.pole_figure import PoleFigure

    if plot_mode not in ('scatter', 'density'):
        raise ValueError("plot_mode must be 'scatter' or 'density'")
    ebsd_q, _ = sample_frame_quats(ebsd_stage, apply_sample_symmetry, use_rd, use_td, use_nd)
    synth_q, _ = sample_frame_quats(synth_stage, apply_sample_symmetry, use_rd, use_td, use_nd)
    pf_ebsd = PoleFigure(ebsd_q, convention='quaternion')
    pf_synth = PoleFigure(synth_q, convention='quaternion')
    labels = dict(x_label=x_label, y_label=y_label, z_label=z_label)
    ebsd_title = f"{ebsd_label} ({len(ebsd_stage['gids'])}, {len(ebsd_q)})"
    synth_title = f"{synth_label} ({len(synth_stage['gids'])}, {len(synth_q)})"
    fig, (ax_ebsd, ax_synth, ax_third) = plt.subplots(1, 3, figsize=(18, 6))
    if plot_mode == 'scatter':
        view = dict(pole_family=pole_family, color_by='fixed', projection=projection, hemisphere=hemisphere)
        pf_ebsd.plot_scatter(ax=ax_ebsd, marker='o', s=16, alpha=0.8, color='#4878CF',
                             title=ebsd_title, **view, **labels)
        pf_synth.plot_scatter(ax=ax_synth, marker='x', s=30, linewidths=1.2, alpha=0.8, color='#D65F5F',
                              title=synth_title, **view, **labels)
        pf_ebsd.plot_scatter(ax=ax_third, marker='o', s=16, alpha=0.6, color='#4878CF', label=ebsd_label, **view)
        pf_synth.plot_scatter(ax=ax_third, marker='x', s=30, linewidths=1.2, alpha=0.8, color='#D65F5F',
                              edgecolors=None, label=synth_label, title='Overlay', **view, **labels)
        ax_third.legend(loc='upper right')
    else:
        grid = 150 if grid_points == 'auto' else grid_points
        _, _, zi_ebsd, _ = pf_ebsd._compute_mud_grid(pf_ebsd._get_symmetric_poles(pole_family), grid)
        _, _, zi_synth, _ = pf_synth._compute_mud_grid(pf_synth._get_symmetric_poles(pole_family), grid)
        lo = float(min(np.nanmin(zi_ebsd), np.nanmin(zi_synth)))
        hi = float(max(np.nanmax(zi_ebsd), np.nanmax(zi_synth)))
        shared = dict(pole_family=pole_family, grid_points=grid, mud_clip_min=lo, mud_clip_max=hi,
                      cmap=density_cmap, colorbar_decimals=colorbar_decimals)
        pf_ebsd.plot_density(ax=ax_ebsd, title=ebsd_title, **shared, **labels)
        pf_synth.plot_density(ax=ax_synth, title=synth_title, **shared, **labels)
        pf_ebsd.plot_density_difference(pf_synth, pole_family=pole_family, ax=ax_third, grid_points=grid,
                                        unit_normalize=unit_normalize, cmap=diff_cmap, title='Difference',
                                        colorbar_decimals=colorbar_decimals, **labels)
    fig.suptitle(f"{ebsd_label} vs {synth_label} — {{{pole_family}}}", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, (ax_ebsd, ax_synth, ax_third)


def pretwin_pole_figure_populations(rg, parent_info, base, assigner, csl_label):
    """Populations for the pre-twin comparison (orientation assignment):
    EBSD 'pure_parents' (grains that host a twin of csl_label and
    are never one) and synthetic 'hosts' (assigned twin-host grains) and
    'all' (every assigned grain). Each {'gids', 'quats'} in DefDAP's
    passive convention, ready for plot_pole_figure_comparison().
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.orientation_3d import compute_ebsd_pure_parents_for_csl

    q, g = compute_ebsd_pure_parents_for_csl(rg, parent_info, csl_label)
    out = {}
    if q is not None:
        q = np.asarray(q, dtype=np.float64).copy()
        q[:, 1:] *= -1                       # back to passive; plotting converts once
        out['pure_parents'] = {'gids': np.asarray(g), 'quats': q}
    orientations = assigner.all_grain_orientations
    all_gids = sorted(orientations)
    host_gids = sorted(gid for gid in base.host_grain_ids if gid in orientations)
    for key, gids in (('hosts', host_gids), ('all', all_gids)):
        out[key] = {'gids': np.asarray(gids),
                    'quats': np.array([orientations[gid] for gid in gids], dtype=np.float64).reshape(-1, 4)}
    return out


def plot_pole_figure_overlay(stage_a, label_a, stage_b, label_b, pole_family='100',
                              marker_a='o', marker_b='^', color_a='steelblue',
                              color_b='darkorange', title=None, apply_sample_symmetry=True,
                              use_rd=True, use_td=True, use_nd=True, projection='stereographic',
                              hemisphere='upper'):
    """Overlays two populations' scatter pole figures on one axes,
    distinguished by marker shape (not just colour) -- e.g. parents vs.
    twins on the same plot. Always scatter -- density contours from two
    different-sized populations can't be meaningfully overlaid the same
    way; use plot_pole_figure() per population for density instead.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from upxo.viz.xphy.pole_figure import PoleFigure

    fig, ax = plt.subplots(figsize=(6, 6))
    for stage_data, label, marker, color in (
            (stage_a, label_a, marker_a, color_a), (stage_b, label_b, marker_b, color_b)):
        quats, gids = sample_frame_quats(stage_data, apply_sample_symmetry, use_rd, use_td, use_nd)
        pf = PoleFigure(quats, convention='quaternion', gids=gids)
        pf.plot_scatter(pole_family=pole_family, ax=ax, color_by='fixed', projection=projection,
                         hemisphere=hemisphere, color=color, marker=marker,
                         label=f'{label} (n={len(stage_data["gids"])})')
    ax.legend(loc='upper right')
    default_title = f"{{{pole_family}}} Pole Figure -- {label_a} vs. {label_b}"
    ax.set_title(title or default_title, fontsize=12, fontweight='bold')
    return fig, ax


def next_master_folder(output_dir=DEFAULT_OUTPUT_DIR, base_filename="grain_structure"):
    """Picks the next unused "<base_filename><N>" folder name under
    <output_dir>/TwinnedFCC/Grain Structures -- the collision-avoidance
    convention export_raw() uses by default, and the one
    steps_temporal_slice_sweep.sweep_temporal_slices() uses once per
    sweep (not once per slice) to give every slice in that sweep a
    shared master folder. Creates the "Grain Structures" directory if it
    does not already exist (needed to check what's already there), but
    NOT the returned folder itself -- that happens when something is
    actually written into it.

    Returns
    -------
    str : the chosen folder name (not a full path), e.g. "grain_structure3".
    """
    out_base = Path(output_dir) / "TwinnedFCC" / "Grain Structures"
    out_base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (out_base / f"{base_filename}{idx}").exists():
        idx += 1
    return f"{base_filename}{idx}"


def export_raw(cleaner, output_dir=DEFAULT_OUTPUT_DIR, base_filename="grain_structure", master_folder=None):
    """Dumps the cleaned structure's raw arrays (label field,
    orientations, twin role/parent maps) as .npy/.pkl -- the lightest-
    weight, format-agnostic save, useful for reloading straight back into
    Python later without re-running the pipeline.

    Files are written directly under
    <output_dir>/TwinnedFCC/Grain Structures/<master_folder>/. When
    `master_folder` is None (the default), a fresh, auto-numbered
    "<base_filename><N>" folder is picked via next_master_folder(), so
    repeated exports never overwrite each other. Pass an explicit
    `master_folder` (e.g. "<sweep_folder>/tslice_<key>", as
    sweep_temporal_slices() does) to nest this export inside a folder
    shared across a whole sweep instead.

    Returns
    -------
    dict : {'out_dir', 'master_folder', 'files' (name -> byte size),
    'n_grains', 'timestamp'}
    """
    import numpy as np
    import pickle
    import datetime

    if master_folder is None:
        master_folder = next_master_folder(output_dir, base_filename)
    out_dir = Path(output_dir) / "TwinnedFCC" / "Grain Structures" / master_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lgi_twinned_clean.npy", cleaner.lgi_clean)
    with open(out_dir / "orientations_all.pkl", "wb") as f:
        pickle.dump(cleaner.all_quats_clean, f)
    with open(out_dir / "twin_role.pkl", "wb") as f:
        pickle.dump(cleaner.twin_role_clean, f)
    with open(out_dir / "twin_parent_of.pkl", "wb") as f:
        pickle.dump(cleaner.twin_parent_of_clean, f)

    file_names = ["lgi_twinned_clean.npy", "orientations_all.pkl", "twin_role.pkl", "twin_parent_of.pkl"]
    return {
        "out_dir": str(out_dir),
        "master_folder": master_folder,
        "files": {name: (out_dir / name).stat().st_size for name in file_names},
        "n_grains": len(cleaner.twin_role_clean),
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
    }


def build_abaqus_index(cleaner, verbose=True):
    """Builds the grain-element index the Abaqus export needs (which
    elements belong to which grain/role/family/variant). Cache this
    yourself if exporting the same cleaned structure more than once --
    it's the expensive part.
    """
    from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D
    return AbaqusExporter3D.build_index(
        cleaner.lgi_clean, cleaner.twin_role_clean, cleaner.twin_parent_of_clean, verbose=verbose)


def export_abaqus_mesh(cleaner, tg, index, output_dir=DEFAULT_OUTPUT_DIR,
                        base_filename="twinned_fcc_mesh", voxel_size_um=1.0,
                        element_type="C3D8", material_format="reference_umat",
                        n_depvar=1, length_scale=1e-3, load_axis="z",
                        applied_strain=0.2, step_time=80.0,
                        initial_inc=0.01, min_inc=1e-8, max_inc=1.0,
                        max_increments=10000, output_interval=0.5,
                        element_outputs=("LE", "NE", "S", "SDV"),
                        node_outputs=("U",),
                        write_role_elsets=True, write_family_elsets=True,
                        write_variant_elsets=True, role_enabled=None, role_prefix=None,
                        family_prefix="es_family_", variant_prefix="es_variant_",
                        nset_config=None, material_level="feature"):
    """Writes the full Abaqus .inp mesh (voxel-conformal
    C3D8 elements by default) plus elsets (by role/family/variant) and
    nodesets (domain faces), into a fresh, auto-numbered subfolder so
    repeated exports never overwrite each other.

    `index`: from build_abaqus_index() above. `tg`: the TwinGenerator3D
    from steps_twin_generation.generate_twins (needed for family/variant
    elset provenance).

    Returns
    -------
    dict : {'output_dir', 'n_elements', 'n_elsets', 'n_nsets', 'files'
    (name -> byte size)}
    """
    from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D

    out_base = Path(output_dir) / "TwinnedFCC" / "ABQInputFiles"
    out_base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (out_base / f"{base_filename}{idx}").exists():
        idx += 1
    out_dir = out_base / f"{base_filename}{idx}"

    exporter = AbaqusExporter3D(
        twin_role=cleaner.twin_role_clean, twin_parent_of=cleaner.twin_parent_of_clean,
        all_quats=cleaner.all_quats_clean, twinmake=tg, voxel_size_um=voxel_size_um,
        element_type=element_type, material_format=material_format, n_depvar=n_depvar,
        length_scale=length_scale, load_axis=load_axis,
        applied_strain=applied_strain, step_time=step_time,
        initial_inc=initial_inc, min_inc=min_inc, max_inc=max_inc,
        max_increments=max_increments, output_interval=output_interval,
        element_outputs=element_outputs, node_outputs=node_outputs,
        write_role_elsets=write_role_elsets, write_family_elsets=write_family_elsets,
        write_variant_elsets=write_variant_elsets,
        role_enabled=role_enabled or DEFAULT_ROLE_ENABLED,
        role_prefix=role_prefix or DEFAULT_ROLE_PREFIX,
        family_prefix=family_prefix, variant_prefix=variant_prefix,
        nset_config=nset_config or DEFAULT_NSET_CONFIG,
        material_level=material_level, index=index,
    )
    exporter.write(str(out_dir))

    file_sizes = {f.name: f.stat().st_size for f in sorted(Path(out_dir).glob("*.inp"))}
    return {
        "output_dir": str(out_dir),
        "n_elements": exporter.n_elements,
        "n_elsets": len(exporter._grain_elems) + exporter.n_role_elsets_written,
        "n_nsets": exporter.n_nsets_written,
        "files": file_sizes,
    }


def finish_report(report):
    """Renders the accumulated report (every add_image/add_table call
    made throughout this notebook) to report.html.

    Returns
    -------
    pathlib.Path : path to the written report.html.
    """
    from upxo.reporting.render_html import render_html
    return render_html(report)
